#!/usr/bin/env python3
"""Benchmark every configured batched_gemm_bf16 size for NT and TT."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton

from boltops.gemm import batched_gemm_bf16
from boltops.gemm.triton.batched_gemm_bf16 import _get_config

BACKEND = "triton"
KERNEL = "batched_gemm_bf16"
DTYPE = torch.float16
WARMUP_MS = 10
REPEAT_MS = 30
M_VALUES = (1, 8, 16, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384)
SHAPES = (
    (16, M_VALUES, 1024, 4096),
    (16, M_VALUES, 8192, 1024),
    (16, M_VALUES, 1280, 8192),
    (56, (234,), 4928, 1232),
    (56, (234,), 1232, 2464),
)


def benchmark_case(
    B: int, M: int, N: int, K: int, layout: str, grid_only: bool
) -> tuple[float, float, dict]:
    x = torch.empty((B, M, K), device="cuda", dtype=DTYPE)
    if layout == "tn":
        weight = torch.empty((B, N, K), device="cuda", dtype=DTYPE)
    else:
        weight = torch.empty((B, K, N), device="cuda", dtype=DTYPE).transpose(1, 2)
    output = torch.empty((B, M, N), device="cuda", dtype=DTYPE)
    config = _get_config(M, N, K, DTYPE, B=B, layout=layout)
    config["BATCH_LAST_GRID"] = True
    if grid_only:
        config["use_block_pingpong"] = False

    def run() -> None:
        batched_gemm_bf16(
            x, weight, dtype=DTYPE, y=output, config=config
        )

    run()
    latency_ms = float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )
    tflops = 2.0 * B * M * N * K / (latency_ms * 1e-3) / 1e12
    return latency_ms, tflops, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--layouts", nargs="+", choices=("tn", "tt"), default=("tn", "tt")
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="force built-in two-slot block ping-pong off",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{BACKEND}_{KERNEL}.csv"

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "backend", "operator", "layout", "mode", "B", "M", "N", "K",
                "dtype", "latency_ms", "tflops", "block_pingpong",
                "batch_last_grid",
            ),
        )
        writer.writeheader()
        for layout in args.layouts:
            for B, m_values, N, K in SHAPES:
                for M in m_values:
                    latency_ms, tflops, config = benchmark_case(
                        B, M, N, K, layout, args.grid_only
                    )
                    writer.writerow(
                        {
                            "backend": BACKEND, "operator": KERNEL,
                            "layout": layout,
                            "mode": "grid-only" if args.grid_only else "tuned",
                            "B": B, "M": M, "N": N, "K": K,
                            "dtype": str(DTYPE).replace("torch.", ""),
                            "latency_ms": f"{latency_ms:.6f}",
                            "tflops": f"{tflops:.2f}",
                            "block_pingpong": config.get(
                                "use_block_pingpong", False
                            ),
                            "batch_last_grid": config.get(
                                "BATCH_LAST_GRID", False
                            ),
                        }
                    )
                    handle.flush()
                    print(
                        f"[bench] {layout} B={B} M={M} N={N} K={K}: "
                        f"{latency_ms:.6f} ms, {tflops:.2f} TFLOPS",
                        flush=True,
                    )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
