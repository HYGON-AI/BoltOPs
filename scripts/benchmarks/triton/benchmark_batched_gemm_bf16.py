#!/usr/bin/env python3
"""Benchmark all tuned Triton batched_gemm_bf16 sizes and write a CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton

from boltops.gemm import batched_gemm_bf16

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


def benchmark_case(B: int, M: int, N: int, K: int) -> float:
    x = torch.empty((B, M, K), device="cuda", dtype=DTYPE)
    weight = torch.empty((B, N, K), device="cuda", dtype=DTYPE)
    output = torch.empty((B, M, N), device="cuda", dtype=DTYPE)

    def run() -> None:
        batched_gemm_bf16(x, weight, dtype=DTYPE, y=output)

    run()
    latency_ms = float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )
    return 2.0 * B * M * N * K / (latency_ms * 1e-3) / 1e12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{BACKEND}_{KERNEL}.csv"

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("backend", "operator", "B", "M", "N", "K", "dtype", "tflops"),
        )
        writer.writeheader()
        for B, m_values, N, K in SHAPES:
            for M in m_values:
                tflops = benchmark_case(B, M, N, K)
                writer.writerow(
                    {
                        "backend": BACKEND, "operator": KERNEL,
                        "B": B, "M": M, "N": N, "K": K,
                        "dtype": str(DTYPE).replace("torch.", ""),
                        "tflops": f"{tflops:.2f}",
                    }
                )
                handle.flush()
                print(
                    f"[bench] {BACKEND}/{KERNEL} B={B} M={M} N={N} K={K}: "
                    f"{tflops:.2f} TFLOPS",
                    flush=True,
                )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
