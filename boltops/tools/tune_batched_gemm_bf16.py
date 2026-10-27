#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Autotune BoltOps Triton batched A16W16 GEMM for selected shapes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import triton

from boltops.gemm.triton.batched_gemm_bf16 import batched_gemm_bf16


AITer_M_VALUES = (1, 8, 16, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384)


def candidates(
    m: int,
    optimize_epilogue: bool = False,
    use_block_pingpong: bool | None = False,
    max_ilp: bool | None = None,
    batch_last_grid: bool = False,
) -> list[dict]:
    # Start with the gfx942 AITER choices, then add BW200-oriented larger tiles.
    # Keeping this list compact makes the full 15-point sweep practical while
    # still covering the important dimensions.
    if m <= 8:
        tiles = [(8, 64, 128, 4), (16, 64, 64, 4), (32, 64, 32, 4)]
    elif m <= 32:
        tiles = [(16, 64, 128, 4), (32, 64, 128, 4), (64, 64, 64, 4)]
    elif m <= 64:
        tiles = [(32, 64, 128, 4), (64, 64, 128, 4), (64, 128, 64, 4)]
    elif m <= 256:
        tiles = [
            (64, 128, 64, 4), (128, 64, 128, 4),
            (128, 128, 128, 8), (128, 128, 64, 4),
            (256, 128, 32, 8), (256, 256, 32, 8),
        ]
    else:
        tiles = [
            (128, 128, 64, 4), (128, 128, 128, 8),
            (128, 256, 32, 8), (256, 128, 32, 8),
            (256, 256, 32, 8), (256, 256, 64, 8),
        ]

    if optimize_epilogue and m >= 128:
        tiles.extend([(256, 128, 16, 8), (256, 256, 16, 8)])
    tiles = list(dict.fromkeys(tiles))

    result = []
    max_ilp_values = (False, True) if max_ilp is None else (max_ilp,)
    pingpong_values = (
        (False, True) if use_block_pingpong is None else (use_block_pingpong,)
    )
    for bm, bn, bk, nw in tiles:
        stages = (2,)
        if optimize_epilogue:
            if m < 128:
                stages = (1, 2, 3)
            elif bk == 16:
                stages = (2, 3)
            elif bm == 256 and bn == 256 and bk == 32:
                stages = (1, 2, 3)
        for stage in stages:
            for waves in (1, 2):
                for enable_max_ilp in max_ilp_values:
                    for enable_pingpong in pingpong_values:
                        result.append(
                            {
                                "BLOCK_SIZE_M": bm,
                                "BLOCK_SIZE_N": bn,
                                "BLOCK_SIZE_K": bk,
                                "GROUP_SIZE_M": 4 if bm >= 64 else 1,
                                "num_warps": nw,
                                "num_stages": stage,
                                "use_block_pingpong": enable_pingpong,
                                "waves_per_eu": waves,
                                "matrix_instr_nonkdim": 16,
                                "optimize_epilogue": optimize_epilogue,
                                "amdgpu_enable_max_ilp_scheduling_strategy": enable_max_ilp,
                                "BATCH_LAST_GRID": batch_last_grid,
                            }
                        )
    return result


def bench(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
        )
    )


def tune_shape(
    b: int,
    m: int,
    n: int,
    k: int,
    warmup_ms: int,
    rep_ms: int,
    optimize_epilogue: bool = False,
    use_block_pingpong: bool | None = False,
    max_ilp: bool | None = None,
    layout: str = "tn",
    batch_last_grid: bool = False,
) -> tuple[dict, float]:
    x = torch.empty((b, m, k), device="cuda", dtype=torch.float16)
    if layout == "tn":
        w = torch.empty((b, n, k), device="cuda", dtype=torch.float16)
    else:
        w = torch.empty((b, k, n), device="cuda", dtype=torch.float16).transpose(
            1, 2
        )
    y = torch.empty((b, m, n), device="cuda", dtype=torch.float16)
    best_cfg = None
    best_ms = math.inf
    for index, cfg in enumerate(
        candidates(
            m,
            optimize_epilogue,
            use_block_pingpong,
            max_ilp,
            batch_last_grid,
        ),
        1,
    ):
        try:
            fn = lambda cfg=cfg: batched_gemm_bf16(
                x, w, dtype=torch.float16, y=y, config=cfg
            )
            ms = bench(fn, warmup_ms, rep_ms)
        except Exception as exc:
            print(f"  [{index:02d}] ERROR {cfg}: {exc}", flush=True)
            continue
        print(f"  [{index:02d}] {ms:.6f} ms {cfg}", flush=True)
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    del x, w, y
    torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError(f"all candidates failed for B={b}, M={m}, N={n}, K={k}")
    return best_cfg, best_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m-values", type=int, nargs="+", default=AITer_M_VALUES)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--optimize-epilogue", action="store_true")
    parser.add_argument(
        "--block-pingpong",
        action="store_true",
        help="deprecated alias for --block-pingpong-mode=on",
    )
    parser.add_argument(
        "--block-pingpong-mode",
        choices=("both", "on", "off"),
        default="off",
        help="autotune built-in two-slot block ping-pong (default: off)",
    )
    parser.add_argument(
        "--batch-last-grid",
        action="store_true",
        help="launch tile ids in grid.x and batch ids in grid.z",
    )
    parser.add_argument("--layout", choices=("tn", "tt"), default="tn")
    parser.add_argument(
        "--max-ilp",
        choices=("both", "on", "off"),
        default="both",
        help="autotune HCU max-ILP scheduling (default: compare on and off)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    max_ilp = {"both": None, "on": True, "off": False}[args.max_ilp]
    pingpong_mode = "on" if args.block_pingpong else args.block_pingpong_mode
    use_block_pingpong = {"both": None, "on": True, "off": False}[pingpong_mode]

    timing_path = args.output.with_suffix(".timings.json")
    output = {}
    timing = {}
    if args.resume and args.output.exists():
        output = json.loads(args.output.read_text())
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
    for m in args.m_values:
        if str(m) in output and str(m) in timing:
            print(f"SKIP B={args.b} M={m} N={args.n} K={args.k}", flush=True)
            continue
        print(f"TUNING B={args.b} M={m} N={args.n} K={args.k}", flush=True)
        cfg, ms = tune_shape(
            args.b,
            m,
            args.n,
            args.k,
            args.warmup_ms,
            args.rep_ms,
            args.optimize_epilogue,
            use_block_pingpong,
            max_ilp,
            args.layout,
            args.batch_last_grid,
        )
        output[str(m)] = cfg
        timing[str(m)] = ms
        print(f"BEST M={m}: {ms:.6f} ms {cfg}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=4) + "\n")
        timing_path.write_text(json.dumps(timing, indent=4) + "\n")


if __name__ == "__main__":
    main()
