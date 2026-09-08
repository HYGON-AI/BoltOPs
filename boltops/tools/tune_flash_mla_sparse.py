#!/usr/bin/env python3
"""Autotune sparse FlashMLA and emit a version/device/shape config file."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
import triton

from boltops.mla import flash_mla_sparse_fwd
from boltops.mla.triton.flash_mla_sparse import (
    _device_identity,
    _supports_matrix_load,
)

DTYPE = torch.bfloat16
DEFAULT_SHAPES = (
    (1, 16384, 16, 1, 2048, 576, 512),
    (4, 16384, 16, 1, 2048, 576, 512),
    (32, 16384, 16, 1, 2048, 576, 512),
    (4, 32768, 16, 1, 4096, 576, 512),
)


def _structural_candidates(quick: bool, matrix_load: bool) -> list[dict]:
    block_ns = (8, 16, 32) if quick else (8, 16, 32, 64)
    block_hs = (8, 16) if quick else (4, 8, 16)
    block_ds = (64, 128) if quick else (64, 128, 256)
    warps = (4,) if quick else (2, 4, 8)
    candidates = []
    for block_n, block_h, block_d, num_warps in itertools.product(
        block_ns, block_hs, block_ds, warps
    ):
        base = {
            "BLOCK_N": block_n,
            "BLOCK_H": block_h,
            "BLOCK_DMODEL": block_d,
            "num_warps": num_warps,
            "num_stages": 1,
            "USE_MATRIX_LOAD_Q": False,
        }
        candidates.append(base)
        # The HCU matrix-load/MFMA lowering requires both output tile
        # dimensions to be at least one 16x16 instruction tile.
        if matrix_load and block_n >= 16 and block_h >= 16:
            candidates.append({**base, "USE_MATRIX_LOAD_Q": True})
    return candidates


def _compiler_candidates(base: dict, quick: bool) -> list[dict]:
    stages = (1, 2)
    waves = (1, 2) if quick else (1, 2, 3, 4)
    kpacks = (1,) if quick else (1, 2)
    warps = (base["num_warps"],) if quick else (2, 4, 8)
    candidates = []
    for num_warps, num_stages, waves_per_eu, kpack in itertools.product(
        warps, stages, waves, kpacks
    ):
        config = dict(base)
        config.update(
            {
                "num_warps": num_warps,
                "num_stages": num_stages,
                "waves_per_eu": waves_per_eu,
                "matrix_instr_nonkdim": 16,
                "kpack": kpack,
            }
        )
        candidates.append(config)
    return candidates


def _optimization_candidates(base: dict, quick: bool) -> list[dict]:
    """Tune HCU scheduling/pipeline passes after tile and launch tuning."""
    del quick
    choices = itertools.product((False, True), repeat=4)
    candidates = []
    for max_ilp, regpressure, optimize_epilogue, block_pingpong in choices:
        config = dict(base)
        config.update(
            {
                "amdgpu_enable_max_ilp_scheduling_strategy": max_ilp,
                "misched_regpressure": regpressure,
                "optimize_epilogue": optimize_epilogue,
                "use_block_pingpong": block_pingpong,
            }
        )
        candidates.append(config)
    return candidates


def _close(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs.float(), rhs.float(), atol=2e-2, rtol=2e-2)


def tune_shape(
    shape: tuple[int, ...], warmup_ms: int, rep_ms: int, quick: bool
) -> tuple[dict, float, float, int]:
    s_q, s_kv, h_q, h_kv, topk, d_qk, d_v = shape
    torch.manual_seed(1234 + s_q + topk)
    q = torch.randn((s_q, h_q, d_qk), device="cuda", dtype=DTYPE)
    kv = torch.randn((s_kv, h_kv, d_qk), device="cuda", dtype=DTYPE)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    lengths = torch.full((s_q,), topk, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d_qk)

    def run(config: dict) -> tuple[torch.Tensor, ...]:
        return flash_mla_sparse_fwd(
            q, kv, indices, scale, d_v=d_v, topk_length=lengths, config=config
        )

    baseline_config = {
        "BLOCK_N": 16,
        "BLOCK_H": 16,
        "BLOCK_DMODEL": 128,
        "num_warps": 4,
        "num_stages": 1,
        "USE_MATRIX_LOAD_Q": False,
    }
    reference = tuple(t.detach().clone() for t in run(baseline_config))
    baseline_ms = float(
        triton.testing.do_bench(
            lambda: run(baseline_config),
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
    )
    tested = 0

    def sweep(candidates: list[dict]) -> tuple[dict, float]:
        nonlocal tested
        best_config = None
        best_ms = math.inf
        for index, config in enumerate(candidates, 1):
            try:
                result = run(config)
                _close(result, reference)
                ms = float(
                    triton.testing.do_bench(
                        lambda config=config: run(config),
                        warmup=warmup_ms,
                        rep=rep_ms,
                        return_mode="median",
                    )
                )
                tested += 1
                print(f"  [{index:03d}] {ms:.6f} ms {config}", flush=True)
                if ms < best_ms:
                    best_config, best_ms = dict(config), ms
            except Exception as exc:
                print(f"  [{index:03d}] FAIL {config}: {exc}", flush=True)
        if best_config is None:
            raise RuntimeError(f"all candidates failed for shape {shape}")
        return best_config, best_ms

    structural_candidates = _structural_candidates(
        quick, _supports_matrix_load(q.device)
    )
    branch_results = []
    for use_matrix_load_q in (False, True):
        branch = [
            config
            for config in structural_candidates
            if config["USE_MATRIX_LOAD_Q"] is use_matrix_load_q
        ]
        if not branch:
            continue
        print(f"  BRANCH USE_MATRIX_LOAD_Q={use_matrix_load_q}", flush=True)
        structural, _ = sweep(branch)
        compiler, _ = sweep(_compiler_candidates(structural, quick))
        branch_results.append(
            sweep(_optimization_candidates(compiler, quick))
        )
    best_config, best_ms = min(branch_results, key=lambda result: result[1])
    del q, kv, indices, lengths, reference
    torch.cuda.empty_cache()
    return best_config, best_ms, baseline_ms, tested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--quick", action="store_true", help="use a smaller smoke-test search space"
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP device is required")

    arch, device_name = _device_identity(torch.device("cuda"))
    entries = []
    for shape in DEFAULT_SHAPES:
        print(f"TUNING shape={shape}", flush=True)
        best, best_ms, baseline_ms, tested = tune_shape(
            shape, args.warmup_ms, args.rep_ms, args.quick
        )
        keys = ("s_q", "s_kv", "h_q", "h_kv", "topk", "d_qk", "d_v")
        match = dict(zip(keys, shape))
        match["dtype"] = str(DTYPE).replace("torch.", "")
        entries.append(
            {
                "match": match,
                "config": best,
                "baseline_ms": baseline_ms,
                "latency_ms": best_ms,
                "speedup": baseline_ms / best_ms,
                "valid_candidates": tested,
            }
        )
        print(
            f"BEST {best_ms:.6f} ms ({baseline_ms / best_ms:.3f}x): {best}",
            flush=True,
        )

    result = {
        "metadata": {
            "triton_version": triton.__version__,
            "device_name": device_name,
            "arch": arch,
        },
        "configs": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
