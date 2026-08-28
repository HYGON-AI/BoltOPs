#!/usr/bin/env python3
"""Compare the FFN's original fixed config with tuned NT/TT configs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton

from boltops.gemm.triton.batched_gemm_bf16 import (
    _get_config,
    batched_gemm_bf16,
)


SHAPES = (("stage1", 56, 234, 4928, 1232), ("stage2", 56, 234, 1232, 2464))
INITIAL_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "use_block_pingpong": False,
    "BATCH_LAST_GRID": False,
}


def bench_case(b: int, m: int, n: int, k: int, layout: str, config: dict) -> float:
    x = torch.empty((b, m, k), device="cuda", dtype=torch.float16)
    if layout == "tn":
        w = torch.empty((b, n, k), device="cuda", dtype=torch.float16)
    else:
        w = torch.empty((b, k, n), device="cuda", dtype=torch.float16).transpose(
            1, 2
        )
    y = torch.empty((b, m, n), device="cuda", dtype=torch.float16)

    def run() -> None:
        batched_gemm_bf16(
            x, w, dtype=torch.float16, y=y, config=config
        )

    run()
    return float(triton.testing.do_bench(run, warmup=20, rep=100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stage, b, m, n, k in SHAPES:
        cases = [("initial-tn", "tn", dict(INITIAL_CONFIG))]
        for layout in ("tn", "tt"):
            tuned = _get_config(m, n, k, torch.float16, B=b, layout=layout)
            cases.append((f"tuned-{layout}", layout, tuned))
            grid_only = dict(tuned)
            grid_only["BATCH_LAST_GRID"] = True
            grid_only["use_block_pingpong"] = False
            cases.append((f"grid-only-{layout}", layout, grid_only))
        for case, layout, config in cases:
            latency_ms = bench_case(b, m, n, k, layout, config)
            rows.append(
                {
                    "stage": stage,
                    "case": case,
                    "layout": layout,
                    "latency_ms": f"{latency_ms:.6f}",
                    "tflops": f"{2 * b * m * n * k / latency_ms / 1e9:.3f}",
                    "vs_initial": "",
                    "block_pingpong": config.get("use_block_pingpong", False),
                    "batch_last_grid": config.get("BATCH_LAST_GRID", False),
                }
            )
        baseline = float(rows[-5]["latency_ms"])
        for row in rows[-5:]:
            row["vs_initial"] = f"{baseline / float(row['latency_ms']):.4f}"

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
