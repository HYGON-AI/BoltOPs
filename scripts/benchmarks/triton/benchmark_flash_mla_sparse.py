#!/usr/bin/env python3
"""Benchmark FlashMLA sparse forward on representative vLLM shapes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import triton

from boltops.mla import flash_mla_sparse_fwd

BACKEND = "triton"
KERNEL = "flash_mla_sparse"
DTYPE = torch.bfloat16
WARMUP_MS = 100
REPEAT_MS = 500
# (query tokens, KV tokens, query heads, KV heads, top-k, d_qk, d_v)
SHAPES = (
    (1, 16384, 16, 1, 2048, 576, 512),
    (4, 16384, 16, 1, 2048, 576, 512),
    (32, 16384, 16, 1, 2048, 576, 512),
    (4, 32768, 16, 1, 4096, 576, 512),
)


def benchmark_case(
    s_q: int,
    s_kv: int,
    h_q: int,
    h_kv: int,
    topk: int,
    d_qk: int,
    d_v: int,
) -> tuple[float, float]:
    q = torch.randn((s_q, h_q, d_qk), device="cuda", dtype=DTYPE)
    kv = torch.randn((s_kv, h_kv, d_qk), device="cuda", dtype=DTYPE)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    topk_length = torch.full(
        (s_q,), topk, device="cuda", dtype=torch.int32
    )
    softmax_scale = 1.0 / math.sqrt(d_qk)

    def run() -> None:
        flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            softmax_scale=softmax_scale,
            d_v=d_v,
            topk_length=topk_length,
        )

    run()
    latency_ms = float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )
    # QK and PV each perform one multiply-add per selected value.
    operations = 2.0 * s_q * h_q * topk * (d_qk + d_v)
    tflops = operations / (latency_ms * 1e-3) / 1e12
    return latency_ms, tflops


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP device is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{BACKEND}_{KERNEL}.csv"
    fields = (
        "backend",
        "operator",
        "s_q",
        "s_kv",
        "h_q",
        "h_kv",
        "topk",
        "d_qk",
        "d_v",
        "dtype",
        "latency_ms",
        "tflops",
    )
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shape in SHAPES:
            latency_ms, tflops = benchmark_case(*shape)
            s_q, s_kv, h_q, h_kv, topk, d_qk, d_v = shape
            writer.writerow(
                {
                    "backend": BACKEND,
                    "operator": KERNEL,
                    "s_q": s_q,
                    "s_kv": s_kv,
                    "h_q": h_q,
                    "h_kv": h_kv,
                    "topk": topk,
                    "d_qk": d_qk,
                    "d_v": d_v,
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "latency_ms": f"{latency_ms:.6f}",
                    "tflops": f"{tflops:.2f}",
                }
            )
            handle.flush()
            print(
                f"[bench] Sq={s_q} Skv={s_kv} Hq={h_q} topk={topk}: "
                f"{latency_ms:.6f} ms, {tflops:.2f} TFLOPS",
                flush=True,
            )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
