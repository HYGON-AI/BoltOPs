# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Autotune sweep for all activation kernels in moe_activation.py.

Sweeps BLOCK_SIZE_N/D and num_warps for each kernel across M values
from 1 to 65536, then picks the best 2 configs per kernel (small-M, large-M).
"""

import argparse
import csv
import itertools
import sys
import time

import torch
import torch.nn.functional as F
import triton

# ── references ────────────────────────────────────────────────────────
from boltops.fused_moe.triton.moe_activation import (
    activation_and_mul_kernel,
    activation_no_mul_1d_kernel,
    gelu_tanh_and_mul_kernel,
    gelu_tanh_no_mul_1d_kernel,
    swiglu_variant_1d_kernel,
    swiglu_oai_interleaved_1d_kernel,
    get_triton_activation_and_mul_config,
    get_triton_activation_no_mul_config,
    get_triton_gelu_tanh_and_mul_config,
    get_triton_gelu_tanh_no_mul_config,
    get_triton_swiglu_variant_config,
    get_triton_swiglu_oai_interleaved_config,
)

# ── torch references for correctness ──────────────────────────────────
def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.silu(x) * y

def torch_gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.gelu(x, approximate="none") * y

def torch_gelu_tanh_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.gelu(x, approximate="tanh") * y

def torch_relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))

def torch_swiglu_gpt_oss(input: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    d = input.shape[-1] // 2
    gate, up = input.split([d, d], dim=-1)
    gate_p = gate.float().clamp(max=limit)
    up_p = up.float().clamp(min=-limit, max=limit)
    return (gate_p * torch.sigmoid(alpha * gate_p) * (up_p + 1.0)).to(input.dtype)

def torch_swiglu_oai_interleaved(input: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = input[..., ::2].float().clamp(max=limit)
    up = input[..., 1::2].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + 1.0)).to(input.dtype)


def check_close(a, b, rtol=1e-2, atol=1e-2):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ── config search space ──────────────────────────────────────────────
# For 1D elementwise kernels, the key tunables are:
#   BLOCK_SIZE_N (or BLOCK_SIZE_D): tile size along the N/D dimension
#   num_warps: threads per warp group
BLOCK_SIZE_OPTIONS = [64, 128, 256, 512, 1024]
NUM_WARPS_OPTIONS = [1, 2, 4, 8]

# M values to sweep (powers of 2, typical MoE token counts)
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Typical N/D values for MoE models
N_VALUES = [352, 704, 1408, 2048, 4096, 7168, 11008, 14336]

NUM_WARMUP = 3
NUM_ITERS = 50


def benchmark_us(fn, *args, **kwargs):
    """Simple CUDA-event based benchmark, returns median time in us."""
    # Warmup
    for _ in range(NUM_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]  # median


def bytes_moved(inp, out, is_gated):
    """Total bytes read + written."""
    return inp.nbytes + out.nbytes


# ── kernel runners ───────────────────────────────────────────────────
def run_activation_and_mul(out, inp, M, N, act, block_size_n, num_warps):
    config = {"BLOCK_SIZE_N": block_size_n, "num_warps": num_warps}
    grid = (M * triton.cdiv(N, block_size_n),)
    activation_and_mul_kernel[grid](
        out, inp, M, N,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        ACT=act, **config,
    )


def run_activation_no_mul(out, inp, M, N, act, block_size_n, num_warps):
    config = {"BLOCK_SIZE_N": block_size_n, "num_warps": num_warps}
    grid = (M * triton.cdiv(N, block_size_n),)
    activation_no_mul_1d_kernel[grid](
        out, inp, M, N,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        ACT=act, **config,
    )


def run_gelu_tanh_and_mul(out, inp, M, N, block_size_n, num_warps):
    config = {"BLOCK_SIZE_N": block_size_n, "num_warps": num_warps}
    grid = (M * triton.cdiv(N, block_size_n),)
    gelu_tanh_and_mul_kernel[grid](
        out, inp, M, N,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        **config,
    )


def run_gelu_tanh_no_mul(out, inp, M, N, block_size_n, num_warps):
    config = {"BLOCK_SIZE_N": block_size_n, "num_warps": num_warps}
    grid = (M * triton.cdiv(N, block_size_n),)
    gelu_tanh_no_mul_1d_kernel[grid](
        out, inp, M, N,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        **config,
    )


def run_swiglu_variant(out, inp, M, D, mode, alpha, limit, block_size_d, num_warps):
    config = {"BLOCK_SIZE_D": block_size_d, "num_warps": num_warps}
    grid = (M * triton.cdiv(D, block_size_d),)
    swiglu_variant_1d_kernel[grid](
        out, inp, M, D,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        alpha, limit, MODE=mode, **config,
    )


def run_swiglu_oai_interleaved(out, inp, M, D, alpha, limit, block_size_d, num_warps):
    config = {"BLOCK_SIZE_D": block_size_d, "num_warps": num_warps}
    grid = (M * triton.cdiv(D, block_size_d),)
    swiglu_oai_interleaved_1d_kernel[grid](
        out, inp, M, D,
        inp.stride(0), inp.stride(1),
        out.stride(0), out.stride(1),
        alpha, limit, **config,
    )


# ── correctness check helpers ────────────────────────────────────────
def check_and_mul(act_name, act_val, inp, ref_out, M, N, block_size_n, num_warps):
    out = torch.empty_like(ref_out)
    run_activation_and_mul(out, inp, M, N, act_val, block_size_n, num_warps)
    return check_close(ref_out, out)


def check_no_mul(act_name, act_val, inp, ref_out, M, N, block_size_n, num_warps):
    out = torch.empty_like(ref_out)
    run_activation_no_mul(out, inp, M, N, act_val, block_size_n, num_warps)
    return check_close(ref_out, out)


def check_swiglu_variant(inp, ref_out, M, D, mode, alpha, limit, block_size_d, num_warps):
    out = torch.empty_like(ref_out)
    run_swiglu_variant(out, inp, M, D, mode, alpha, limit, block_size_d, num_warps)
    return check_close(ref_out, out)


def check_swiglu_interleaved(inp, ref_out, M, D, alpha, limit, block_size_d, num_warps):
    out = torch.empty_like(ref_out)
    run_swiglu_oai_interleaved(out, inp, M, D, alpha, limit, block_size_d, num_warps)
    return check_close(ref_out, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=704, help="N value (gated: input has 2N cols)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--out_csv", type=str, default="act_kernel_autotune.csv")
    parser.add_argument("--skip_correctness", action="store_true")
    parser.add_argument("--kernel", type=str, default="all",
                        choices=["all", "silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul",
                                 "relu2", "silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul",
                                 "swiglu_gpt_oss", "swiglu_oai_interleaved"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    N = args.n  # for gated: output is [M, N], input is [M, 2N]

    # Collect all results
    results = []  # list of dicts

    kernels_to_run = []
    if args.kernel == "all":
        kernels_to_run = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "relu2",
                          "silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul",
                          "swiglu_gpt_oss", "swiglu_oai_interleaved"]
    else:
        kernels_to_run = [args.kernel]

    alpha = 1.702
    limit = 7.0

    for kernel_name in kernels_to_run:
        print(f"\n{'='*60}")
        print(f"Kernel: {kernel_name}, N={N}, dtype={dtype}")
        print(f"{'='*60}")

        for M in M_VALUES:
            # Prepare input/output tensors
            if kernel_name in ("silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"):
                inp = torch.randn(M, 2 * N, dtype=dtype, device=device)
                if kernel_name == "silu_and_mul":
                    ref_out = torch_silu_and_mul(inp)
                    act_val = 0
                elif kernel_name == "gelu_and_mul":
                    ref_out = torch_gelu_and_mul(inp)
                    act_val = 1
                else:
                    ref_out = torch_gelu_tanh_and_mul(inp)
                    act_val = None
                out = torch.empty(M, N, dtype=dtype, device=device)
                is_gated = True
            elif kernel_name in ("relu2", "silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul"):
                inp = torch.randn(M, N, dtype=dtype, device=device)
                if kernel_name == "relu2":
                    ref_out = torch_relu2(inp)
                    act_val = 2
                elif kernel_name == "silu_no_mul":
                    ref_out = F.silu(inp)
                    act_val = 0
                elif kernel_name == "gelu_no_mul":
                    ref_out = F.gelu(inp, approximate="none")
                    act_val = 1
                else:
                    ref_out = F.gelu(inp, approximate="tanh")
                    act_val = None
                out = torch.empty(M, N, dtype=dtype, device=device)
                is_gated = False
            elif kernel_name == "swiglu_gpt_oss":
                inp = torch.randn(M, 2 * N, dtype=dtype, device=device)
                ref_out = torch_swiglu_gpt_oss(inp, alpha, limit)
                out = torch.empty(M, N, dtype=dtype, device=device)
                is_gated = True
            elif kernel_name == "swiglu_oai_interleaved":
                inp = torch.randn(M, 2 * N, dtype=dtype, device=device)
                ref_out = torch_swiglu_oai_interleaved(inp, alpha, limit)
                out = torch.empty(M, N, dtype=dtype, device=device)
                is_gated = True
            else:
                continue

            # Also get baseline (current config) performance
            if kernel_name in ("silu_and_mul", "gelu_and_mul"):
                base_config = get_triton_activation_and_mul_config(M, N)
                base_us = benchmark_us(
                    run_activation_and_mul, out, inp, M, N, act_val,
                    base_config["BLOCK_SIZE_N"], base_config["num_warps"])
            elif kernel_name == "gelu_tanh_and_mul":
                base_config = get_triton_gelu_tanh_and_mul_config(M, N)
                base_us = benchmark_us(
                    run_gelu_tanh_and_mul, out, inp, M, N,
                    base_config["BLOCK_SIZE_N"], base_config["num_warps"])
            elif kernel_name in ("relu2", "silu_no_mul", "gelu_no_mul"):
                base_config = get_triton_activation_no_mul_config(M, N)
                base_us = benchmark_us(
                    run_activation_no_mul, out, inp, M, N, act_val,
                    base_config["BLOCK_SIZE_N"], base_config["num_warps"])
            elif kernel_name == "gelu_tanh_no_mul":
                base_config = get_triton_gelu_tanh_no_mul_config(M, N)
                base_us = benchmark_us(
                    run_gelu_tanh_no_mul, out, inp, M, N,
                    base_config["BLOCK_SIZE_N"], base_config["num_warps"])
            elif kernel_name == "swiglu_gpt_oss":
                base_config = get_triton_swiglu_variant_config(M, N)
                base_us = benchmark_us(
                    run_swiglu_variant, out, inp, M, N, 0, alpha, limit,
                    base_config["BLOCK_SIZE_D"], base_config["num_warps"])
            elif kernel_name == "swiglu_oai_interleaved":
                base_config = get_triton_swiglu_oai_interleaved_config(M, N)
                base_us = benchmark_us(
                    run_swiglu_oai_interleaved, out, inp, M, N, alpha, limit,
                    base_config["BLOCK_SIZE_D"], base_config["num_warps"])

            total_bytes = bytes_moved(inp, out, is_gated)
            base_tbps = total_bytes / base_us / 1e6

            best_config = None
            best_us = float('inf')
            best_tbps = 0.0
            correct_configs = []

            for block_size in BLOCK_SIZE_OPTIONS:
                for num_warps in NUM_WARPS_OPTIONS:
                    # Skip configs where block_size > N (no point)
                    if block_size > 2 * N:
                        continue

                    # Correctness check
                    ok = True
                    if not args.skip_correctness:
                        try:
                            if kernel_name in ("silu_and_mul", "gelu_and_mul"):
                                ok = check_and_mul(kernel_name, act_val, inp, ref_out,
                                                   M, N, block_size, num_warps)
                            elif kernel_name == "gelu_tanh_and_mul":
                                out_chk = torch.empty_like(out)
                                run_gelu_tanh_and_mul(out_chk, inp, M, N, block_size, num_warps)
                                ok = check_close(ref_out, out_chk)
                            elif kernel_name in ("relu2", "silu_no_mul", "gelu_no_mul"):
                                ok = check_no_mul(kernel_name, act_val, inp, ref_out,
                                                  M, N, block_size, num_warps)
                            elif kernel_name == "gelu_tanh_no_mul":
                                out_chk = torch.empty_like(out)
                                run_gelu_tanh_no_mul(out_chk, inp, M, N, block_size, num_warps)
                                ok = check_close(ref_out, out_chk)
                            elif kernel_name == "swiglu_gpt_oss":
                                ok = check_swiglu_variant(inp, ref_out, M, N, 0,
                                                          alpha, limit, block_size, num_warps)
                            elif kernel_name == "swiglu_oai_interleaved":
                                ok = check_swiglu_interleaved(inp, ref_out, M, N,
                                                              alpha, limit, block_size, num_warps)
                        except Exception as e:
                            ok = False
                            print(f"  ERROR: M={M} bs={block_size} nw={num_warps}: {e}")

                    if not ok:
                        if not args.skip_correctness:
                            print(f"  FAIL: M={M} bs={block_size} nw={num_warps} (incorrect)")
                        continue

                    # Benchmark
                    try:
                        if kernel_name in ("silu_and_mul", "gelu_and_mul"):
                            us = benchmark_us(
                                run_activation_and_mul, out, inp, M, N, act_val,
                                block_size, num_warps)
                        elif kernel_name == "gelu_tanh_and_mul":
                            us = benchmark_us(
                                run_gelu_tanh_and_mul, out, inp, M, N,
                                block_size, num_warps)
                        elif kernel_name in ("relu2", "silu_no_mul", "gelu_no_mul"):
                            us = benchmark_us(
                                run_activation_no_mul, out, inp, M, N, act_val,
                                block_size, num_warps)
                        elif kernel_name == "gelu_tanh_no_mul":
                            us = benchmark_us(
                                run_gelu_tanh_no_mul, out, inp, M, N,
                                block_size, num_warps)
                        elif kernel_name == "swiglu_gpt_oss":
                            us = benchmark_us(
                                run_swiglu_variant, out, inp, M, N, 0, alpha, limit,
                                block_size, num_warps)
                        elif kernel_name == "swiglu_oai_interleaved":
                            us = benchmark_us(
                                run_swiglu_oai_interleaved, out, inp, M, N, alpha, limit,
                                block_size, num_warps)
                    except Exception as e:
                        print(f"  BENCH ERROR: M={M} bs={block_size} nw={num_warps}: {e}")
                        continue

                    tbps = total_bytes / us / 1e6
                    correct_configs.append({
                        "M": M, "block_size": block_size, "num_warps": num_warps,
                        "us": us, "TB/s": tbps,
                    })

                    if us < best_us:
                        best_us = us
                        best_tbps = tbps
                        best_config = {"block_size": block_size, "num_warps": num_warps}

            if best_config is None:
                print(f"  M={M}: NO valid config found!")
                continue

            speedup = base_us / best_us if best_us > 0 else 0
            print(f"  M={M:6d}: base={base_us:8.1f}us ({base_tbps:.1f}TB/s) "
                  f"best={best_us:8.1f}us ({best_tbps:.1f}TB/s) "
                  f"config=BS{best_config['block_size']}_W{best_config['num_warps']} "
                  f"speedup={speedup:.2f}x")

            results.append({
                "kernel": kernel_name,
                "M": M,
                "N": N,
                "dtype": str(dtype),
                "base_us": base_us,
                "base_tbps": base_tbps,
                "best_us": best_us,
                "best_tbps": best_tbps,
                "best_block_size": best_config["block_size"],
                "best_num_warps": best_config["num_warps"],
                "base_block_size": base_config.get("BLOCK_SIZE_N", base_config.get("BLOCK_SIZE_D")),
                "base_num_warps": base_config["num_warps"],
                "speedup": speedup,
                "all_correct_configs": str(correct_configs),
            })

    # ── Write CSV ────────────────────────────────────────────────────
    if results:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "kernel", "M", "N", "dtype",
                "base_us", "base_tbps", "best_us", "best_tbps",
                "best_block_size", "best_num_warps",
                "base_block_size", "base_num_warps",
                "speedup",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({k: v for k, v in r.items()
                                 if k != "all_correct_configs"})

    # ── Summary: pick 2 configs per kernel (small-M, large-M) ───────
    print("\n" + "=" * 80)
    print("SUMMARY: Best 2 configs per kernel (small-M <= 32, large-M > 32)")
    print("=" * 80)

    for kernel_name in kernels_to_run:
        kr = [r for r in results if r["kernel"] == kernel_name]
        if not kr:
            continue

        small = [r for r in kr if r["M"] <= 32]
        large = [r for r in kr if r["M"] > 32]

        # For small-M, pick config that's best on average across small M values
        # Weight by inverse of M (decode scenarios are dominated by small M)
        small_config_scores = {}
        for r in small:
            key = (r["best_block_size"], r["best_num_warps"])
            if key not in small_config_scores:
                small_config_scores[key] = {"total_speedup": 0, "count": 0}
            small_config_scores[key]["total_speedup"] += r["speedup"]
            small_config_scores[key]["count"] += 1

        # For large-M, pick config that's best on average across large M values
        large_config_scores = {}
        for r in large:
            key = (r["best_block_size"], r["best_num_warps"])
            if key not in large_config_scores:
                large_config_scores[key] = {"total_speedup": 0, "count": 0}
            large_config_scores[key]["total_speedup"] += r["speedup"]
            large_config_scores[key]["count"] += 1

        small_best = max(small_config_scores.items(),
                         key=lambda x: x[1]["total_speedup"] / x[1]["count"],
                         default=None)
        large_best = max(large_config_scores.items(),
                         key=lambda x: x[1]["total_speedup"] / x[1]["count"],
                         default=None)

        print(f"\n  {kernel_name}:")
        if small_best:
            bs, nw = small_best[0]
            avg_sp = small_best[1]["total_speedup"] / small_best[1]["count"]
            print(f"    small-M (M<=32): BLOCK_SIZE={bs}, num_warps={nw} (avg speedup={avg_sp:.2f}x)")
        if large_best:
            bs, nw = large_best[0]
            avg_sp = large_best[1]["total_speedup"] / large_best[1]["count"]
            print(f"    large-M (M>32):  BLOCK_SIZE={bs}, num_warps={nw} (avg speedup={avg_sp:.2f}x)")

        # Also show per-M speedup
        print(f"    Per-M speedup:")
        for r in sorted(kr, key=lambda x: x["M"]):
            print(f"      M={r['M']:6d}: {r['speedup']:.2f}x  "
                  f"(base: BS{r['base_block_size']}_W{r['base_num_warps']} -> "
                  f"best: BS{r['best_block_size']}_W{r['best_num_warps']})")

    print(f"\nResults saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
