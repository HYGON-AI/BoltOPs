#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Autotune BoltOps query-tiled sparse prefill for selected production shapes.

Candidates include HIP ``waves_per_eu``; ``_run_tiled_prefill`` forwards it at
launch with ``num_warps`` / ``num_stages``.
"""

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
    _build_prefill_tiles,
    _CONFIG_DIR,
    _prefill_tiled_config_filename,
    _run_tiled_prefill,
    sparse_attention_prefill,
    sparse_attn_prefill_tiled_union_bucket,
)

OUT_RTOL, OUT_ATOL = 1e-2, 2e-2
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3


def candidates(block_q: int, block_d: int) -> list[dict]:
    result = []
    for block_r in (1, 2, 4, 8, 16):
        for num_warps in (2, 4, 8, 16):
            for num_stages in (1, 2, 3):
                for waves in (1, 2):
                    result.append(
                        {
                            "BLOCK_Q": block_q,
                            "BLOCK_D": block_d,
                            "BLOCK_R": block_r,
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


def _prepared_q_positions(data: dict) -> torch.Tensor:
    query = data["query"]
    total_q = int(query.shape[0])
    q_positions = data["q_positions"].to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    if q_positions.numel() == total_q:
        return q_positions
    return q_positions.reshape(data["groups"], total_q)[0].contiguous()


def _launch_kwargs(data: dict, tiles: tuple[torch.Tensor, ...], *, block_q: int) -> dict:
    geo = data["geo"]
    query = data["query"]
    q_starts, q_counts, tile_phys, tile_logical, lengths, mask = tiles
    return {
        "query": query,
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "q_starts": q_starts,
        "q_counts": q_counts,
        "tile_phys": tile_phys,
        "tile_logical": tile_logical,
        "lengths": lengths,
        "mask": mask,
        "q_positions": _prepared_q_positions(data),
        "packed_width": int(data["packed"].shape[1]),
        "num_kv_groups": data["groups"],
        "region_size": geo.region_block_size,
        "softmax_scale": float(query.shape[2]) ** -0.5,
        "block_q": block_q,
        "block_regions": 4,
    }


def _check_winner(data: dict, cfg: dict, *, block_q: int) -> None:
    kwargs = {
        "query": data["query"],
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "packed_regions": data["packed"],
        "region_counts": data["counts"],
        "num_kv_groups": data["groups"],
        "region_size": data["geo"].region_block_size,
        "softmax_scale": float(data["query"].shape[2]) ** -0.5,
        "block_regions": 4,
        "logical_regions": data["logical_ids"],
        "q_positions": data["q_positions"],
        "request_indices": data["request_indices"],
        "block_q": block_q,
    }
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
    *,
    block_q: int,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    query = data["query"]
    block_d = triton.next_power_of_2(query.shape[2])
    tiles = _build_prefill_tiles(
        data["logical_ids"].contiguous(),
        data["packed"].contiguous(),
        _prepared_q_positions(data),
        data["request_indices"].to(device=query.device, dtype=torch.int32).contiguous(),
        total_q=int(query.shape[0]),
        num_kv_groups=data["groups"],
        region_size=data["geo"].region_block_size,
        block_q=block_q,
    )
    launch = _launch_kwargs(data, tiles, block_q=block_q)
    out = torch.empty_like(query)
    lse = torch.empty(
        (query.shape[0], query.shape[1]),
        device=query.device,
        dtype=torch.float32,
    )
    launch["out"] = out
    launch["lse"] = lse
    best_cfg = None
    best_ms = math.inf
    for index, cfg in enumerate(candidates(block_q, block_d), 1):
        try:
            fn = lambda cfg=cfg: _run_tiled_prefill(**launch, config=cfg)
            ms = bench(fn, warmup_ms, rep_ms)
        except Exception as exc:
            print(f"  [{index:03d}] ERROR {cfg}: {exc}", flush=True)
            continue
        print(f"  [{index:03d}] {ms:.6f} ms {cfg}", flush=True)
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError("all tiled sparse prefill candidates failed")
    _check_winner(data, best_cfg, block_q=block_q)
    return best_cfg, best_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--cache-regions", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=192)
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260844)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.heads % args.groups:
        raise SystemExit(
            f"--heads {args.heads} does not divide into --groups {args.groups}"
        )
    if args.block_q <= 0 or args.block_q & (args.block_q - 1):
        raise SystemExit(f"--block-q must be a positive power of two, got {args.block_q}")

    device = torch.device("cuda")
    geo = tp8_geometry()
    if args.head_dim != geo.head_dim:
        geo = replace(geo, head_dim=args.head_dim)
    case = BenchmarkCase(
        "tp8.tiled.tune",
        "tp8-production",
        "tiled_prefill",
        {
            "queries": args.queries,
            "cache_regions": args.cache_regions,
            "topk": args.topk,
            "heads": args.heads,
            "groups": args.groups,
            "request_lengths": (args.queries,),
        },
        "sparse attn tiled prefill tune",
    )
    data = _build_prefill_inputs(case, device, args.seed, geo=geo)
    packed_width = int(data["packed"].shape[1])
    heads_per_group = args.heads // args.groups
    head_dim = int(data["query"].shape[2])
    q_starts, _, tile_phys, _, _, _ = _build_prefill_tiles(
        data["logical_ids"].contiguous(),
        data["packed"].contiguous(),
        data["q_positions"],
        data["request_indices"],
        total_q=int(data["query"].shape[0]),
        num_kv_groups=data["groups"],
        region_size=geo.region_block_size,
        block_q=args.block_q,
    )
    union = int(tile_phys.shape[1])
    union_bucket = sparse_attn_prefill_tiled_union_bucket(union)
    from boltops.step4_dsa.triton.common import _device_name_for_config
    import os
    device_tag = _device_name_for_config()
    output = args.output or Path(
        os.path.join(_CONFIG_DIR, _prefill_tiled_config_filename(device_tag))
    )
    timing_path = output.with_suffix(".timings.json")
    if args.resume and output.exists():
        print(f"SKIP {output}", flush=True)
        return

    print(
        "TUNING "
        f"Q={args.queries} R={args.cache_regions} K={args.topk} "
        f"packed={packed_width} union={union} u={union_bucket} "
        f"hpg={heads_per_group} D={head_dim} BQ={args.block_q}",
        flush=True,
    )
    cfg, ms = tune_shape(
        data,
        block_q=args.block_q,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    print(f"BEST: {ms:.6f} ms {cfg}", flush=True)
    missing = [key for key in ("num_warps", "num_stages", "waves_per_eu") if key not in cfg]
    if missing:
        raise RuntimeError(f"tuned config omitted launch keys {missing}: {cfg}")
    # Build the new-format entry to upsert into the configs list.
    new_entry = {
        "match": {
            "s_q": int(data["query"].shape[0]),
            "num_tiles": int(q_starts.numel()),
            "h_q": int(data["query"].shape[1]),
            "h_kv": int(data["key_cache"].shape[1]),
            "d": head_dim,
            "topk": packed_width,
            "union_bucket": union_bucket,
            "region_size": int(geo.region_block_size),
            "block_q": args.block_q,
            "dtype": str(data["query"].dtype).replace("torch.", ""),
        },
        "config": cfg,
        "latency_ms": ms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Load existing file (if any) and upsert the entry with the same match.
    existing: dict = {}
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    if not isinstance(existing, dict) or "configs" not in existing:
        existing = {"metadata": {"device_name": device_tag}, "configs": []}
    configs = existing.get("configs", [])
    match_key = new_entry["match"]
    configs = [e for e in configs if e.get("match") != match_key]
    configs.append(new_entry)
    existing["configs"] = configs
    output.write_text(json.dumps(existing, indent=2) + "\n")
    timing_path.write_text(json.dumps({"ms": ms, "config": cfg}, indent=4) + "\n")


if __name__ == "__main__":
    main()
