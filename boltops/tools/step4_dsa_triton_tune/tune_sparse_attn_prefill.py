#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Autotune BoltOps per-query sparse prefill for selected production shapes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import triton

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_ROOT = _REPO_ROOT / "test"
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from step4_dsa.bench_inputs import _build_prefill_inputs  # noqa: E402
from step4_dsa.benchmark_cases import BenchmarkCase, tp8_geometry  # noqa: E402

from boltops.step4_dsa.triton.prefill import (  # noqa: E402
    sparse_attention_prefill,
    sparse_attn_prefill_config_path,
)

OUT_RTOL, OUT_ATOL = 1e-2, 2e-2
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3


def candidates(block_h: int, block_d: int) -> list[dict]:
    result = []
    for block_r in (1, 2, 4, 8, 16):
        for num_warps in (2, 4, 8, 16):
            for num_stages in (1, 2, 3):
                for waves in (1, 2):
                    result.append(
                        {
                            "BLOCK_R": block_r,
                            "BLOCK_H": block_h,
                            "BLOCK_D": block_d,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "waves_per_eu": waves,
                        }
                    )
    return result


def bench(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
        )
    )


def _prefill_kwargs(data: dict) -> dict:
    geo = data["geo"]
    query = data["query"]
    return {
        "query": query,
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "packed_regions": data["packed"],
        "region_counts": data["counts"],
        "num_kv_groups": data["groups"],
        "region_size": geo.region_block_size,
        "softmax_scale": float(query.shape[2]) ** -0.5,
        "block_regions": 16,
    }


def _check_winner(data: dict, cfg: dict) -> None:
    kwargs = _prefill_kwargs(data)
    ref_out, ref_lse = sparse_attention_prefill(**kwargs, config={})
    out, lse = sparse_attention_prefill(**kwargs, config=cfg)
    if not torch.allclose(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL):
        raise RuntimeError(
            f"winner diverged from default out (rtol={OUT_RTOL}, atol={OUT_ATOL}): {cfg}"
        )
    if not torch.allclose(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL):
        raise RuntimeError(
            f"winner diverged from default lse (rtol={LSE_RTOL}, atol={LSE_ATOL}): {cfg}"
        )


def tune_shape(
    data: dict,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    query = data["query"]
    heads_per_group = query.shape[1] // data["groups"]
    block_h = max(16, triton.next_power_of_2(heads_per_group))
    block_d = triton.next_power_of_2(query.shape[2])
    kwargs = _prefill_kwargs(data)
    best_cfg = None
    best_ms = math.inf
    for index, cfg in enumerate(candidates(block_h, block_d), 1):
        try:
            fn = lambda cfg=cfg: sparse_attention_prefill(**kwargs, config=cfg)
            ms = bench(fn, warmup_ms, rep_ms)
        except Exception as exc:
            print(f"  [{index:03d}] ERROR {cfg}: {exc}", flush=True)
            continue
        print(f"  [{index:03d}] {ms:.6f} ms {cfg}", flush=True)
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError("all sparse prefill candidates failed")
    _check_winner(data, best_cfg)
    return best_cfg, best_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--cache-regions", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260843)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.heads % args.groups:
        raise SystemExit(
            f"--heads {args.heads} does not divide into --groups {args.groups}"
        )

    device = torch.device("cuda")
    geo = tp8_geometry()
    if args.head_dim != geo.head_dim:
        geo = replace(geo, head_dim=args.head_dim)
    case = BenchmarkCase(
        "tp8.prefill.tune",
        "tp8-production",
        "prefill",
        {
            "queries": args.queries,
            "cache_regions": args.cache_regions,
            "topk": args.topk,
            "heads": args.heads,
            "groups": args.groups,
        },
        "sparse attn prefill tune",
    )
    data = _build_prefill_inputs(case, device, args.seed, geo=geo)
    packed_width = int(data["packed"].shape[1])
    heads_per_group = args.heads // args.groups
    head_dim = int(data["query"].shape[2])
    output = args.output or Path(
        sparse_attn_prefill_config_path(
            packed_width,
            heads_per_group,
            head_dim,
            geo.region_block_size,
            data["query"].dtype,
        )
    )
    timing_path = output.with_suffix(".timings.json")
    if args.resume and output.exists():
        print(f"SKIP {output}", flush=True)
        return

    print(
        "TUNING "
        f"Q={args.queries} R={args.cache_regions} K={args.topk} "
        f"packed={packed_width} hpg={heads_per_group} D={head_dim}",
        flush=True,
    )
    cfg, ms = tune_shape(data, args.warmup_ms, args.rep_ms)
    print(f"BEST: {ms:.6f} ms {cfg}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=4) + "\n")
    timing_path.write_text(json.dumps({"ms": ms, "config": cfg}, indent=4) + "\n")


if __name__ == "__main__":
    main()
