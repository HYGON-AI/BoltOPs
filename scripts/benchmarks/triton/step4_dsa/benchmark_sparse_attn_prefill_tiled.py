#!/usr/bin/env python3
"""Benchmark query-tiled sparse prefill at the production Q=128 tile.

Launch JSON is keyed by packed width / union bucket / hpg / head dim /
region size / block_q (not by total Q). Scalar prefill lives in the sibling
script.

``latency_ms`` is the tiled Triton kernel only. ``_build_prefill_tiles`` is
measured separately as ``tile_build_ms`` and is not inside ``do_bench``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import triton

# This file is one directory deeper than the other triton benches.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TEST_ROOT = _REPO_ROOT / "test"
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from step4_dsa.bench_inputs import _build_prefill_inputs  # noqa: E402
from step4_dsa.benchmark_cases import BenchmarkCase, tp8_geometry  # noqa: E402

from boltops.step4_dsa.triton.common import _fit_block_regions  # noqa: E402
from boltops.step4_dsa.triton.prefill import (  # noqa: E402
    _build_prefill_tiles,
    _run_tiled_prefill,
    sparse_attn_prefill_tiled_config_path,
    sparse_attn_prefill_tiled_union_bucket,
)

BACKEND = "triton"
KERNEL = "sparse_attn_prefill_tiled"
DTYPE = torch.bfloat16
WARMUP_MS = 10
REPEAT_MS = 30
SEED = 20260844
HEADS = 8
GROUPS = 1
TOPK = 512
CACHE_REGIONS = 4096
BLOCK_REGIONS = 4
BLOCK_Q = 128
Q_VALUES = (128,)


def _prepared_q_positions(data: dict) -> torch.Tensor:
    query = data["query"]
    q_positions = data["q_positions"].to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    if q_positions.numel() == int(query.shape[0]):
        q_positions = q_positions.repeat(data["groups"])
    return q_positions


def _prepared_request_indices(data: dict) -> torch.Tensor:
    query = data["query"]
    return data["request_indices"].to(
        device=query.device, dtype=torch.int32
    ).contiguous()


def _make_tiles(data: dict) -> tuple[torch.Tensor, ...]:
    geo = data["geo"]
    return _build_prefill_tiles(
        data["logical_ids"].contiguous(),
        data["packed"].contiguous(),
        _prepared_q_positions(data),
        _prepared_request_indices(data),
        total_q=int(data["query"].shape[0]),
        num_kv_groups=data["groups"],
        region_size=geo.region_block_size,
        block_q=BLOCK_Q,
    )


def _build_case(queries: int, device: torch.device) -> dict:
    geo = tp8_geometry()
    case = BenchmarkCase(
        "tp8.tiled.bench",
        "tp8-production",
        "tiled_prefill",
        {
            "queries": queries,
            "cache_regions": CACHE_REGIONS,
            "topk": TOPK,
            "heads": HEADS,
            "groups": GROUPS,
            "request_lengths": (queries,),
        },
        "sparse attn tiled prefill Q=128",
    )
    return _build_prefill_inputs(case, device, SEED, geo=geo)


def _tile_union(tiles: tuple[torch.Tensor, ...]) -> tuple[int, int]:
    union = int(tiles[2].shape[1])
    return union, sparse_attn_prefill_tiled_union_bucket(union)


def _tuned_config_path(data: dict, union_bucket: int) -> Path:
    query = data["query"]
    return Path(
        sparse_attn_prefill_tiled_config_path(
            int(data["packed"].shape[1]),
            union_bucket,
            query.shape[1] // data["groups"],
            int(query.shape[2]),
            data["geo"].region_block_size,
            BLOCK_Q,
            query.dtype,
        )
    )


def _default_launch(data: dict, union: int) -> dict:
    query = data["query"]
    block_d = triton.next_power_of_2(int(query.shape[2]))
    block_r = _fit_block_regions(
        min(BLOCK_REGIONS, triton.next_power_of_2(union)),
        region_size=data["geo"].region_block_size,
        block_d=block_d,
        element_size=query.element_size(),
    )
    return {
        "BLOCK_Q": BLOCK_Q,
        "BLOCK_D": block_d,
        "BLOCK_R": block_r,
        "num_warps": 4,
        "num_stages": 2,
    }


def _bench_kernel(
    data: dict,
    tiles: tuple[torch.Tensor, ...],
    q_positions: torch.Tensor,
    config: dict | None,
) -> float:
    geo = data["geo"]
    query = data["query"]
    q_starts, q_counts, tile_phys, tile_logical, lengths, mask = tiles
    out = torch.empty_like(query)
    lse = torch.empty(
        (query.shape[0], query.shape[1]),
        device=query.device,
        dtype=torch.float32,
    )

    def run() -> None:
        _run_tiled_prefill(
            query,
            data["key_cache"],
            data["value_cache"],
            q_starts,
            q_counts,
            tile_phys,
            tile_logical,
            lengths,
            mask,
            q_positions,
            packed_width=int(data["packed"].shape[1]),
            num_kv_groups=data["groups"],
            region_size=geo.region_block_size,
            softmax_scale=float(query.shape[2]) ** -0.5,
            block_q=BLOCK_Q,
            block_regions=BLOCK_REGIONS,
            config=config,
            out=out,
            lse=lse,
        )

    run()
    return float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )


def _bench_tile_build(data: dict) -> float:
    def run() -> None:
        _make_tiles(data)

    run()
    return float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )


def _csv_config_fields(launch: dict) -> dict:
    return {
        "BLOCK_Q": launch.get("BLOCK_Q", ""),
        "BLOCK_R": launch.get("BLOCK_R", ""),
        "BLOCK_D": launch.get("BLOCK_D", ""),
        "num_warps": launch.get("num_warps", ""),
        "num_stages": launch.get("num_stages", ""),
        "waves_per_eu": launch.get("waves_per_eu", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("tuned", "default"),
        default=("tuned", "default"),
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    device = torch.device("cuda")
    geo = tp8_geometry()
    packed_width = TOPK + 1
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{BACKEND}_{KERNEL}.csv"

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "backend",
                "operator",
                "mode",
                "Q",
                "heads",
                "groups",
                "topk",
                "packed_width",
                "cache_regions",
                "head_dim",
                "region_size",
                "block_q",
                "union",
                "union_bucket",
                "dtype",
                "latency_ms",
                "tile_build_ms",
                "BLOCK_Q",
                "BLOCK_R",
                "BLOCK_D",
                "num_warps",
                "num_stages",
                "waves_per_eu",
            ),
        )
        writer.writeheader()
        for queries in Q_VALUES:
            data = _build_case(queries, device)
            tiles = _make_tiles(data)
            q_positions = _prepared_q_positions(data)
            union, union_bucket = _tile_union(tiles)
            config_path = _tuned_config_path(data, union_bucket)
            print(f"tuned config: {config_path}", flush=True)
            if "tuned" in args.modes and not config_path.is_file():
                raise SystemExit(f"tuned config not found: {config_path}")
            tuned_launch = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if "tuned" in args.modes
                else {}
            )
            default_launch = _default_launch(data, union)
            tile_build_ms = _bench_tile_build(data)
            print(
                f"[bench] tile_build Q={queries}: {tile_build_ms:.6f} ms",
                flush=True,
            )
            for mode in args.modes:
                if mode == "tuned":
                    latency_ms = _bench_kernel(data, tiles, q_positions, tuned_launch)
                    launch = tuned_launch
                else:
                    latency_ms = _bench_kernel(data, tiles, q_positions, {})
                    launch = default_launch
                writer.writerow(
                    {
                        "backend": BACKEND,
                        "operator": KERNEL,
                        "mode": mode,
                        "Q": queries,
                        "heads": HEADS,
                        "groups": GROUPS,
                        "topk": TOPK,
                        "packed_width": packed_width,
                        "cache_regions": CACHE_REGIONS,
                        "head_dim": geo.head_dim,
                        "region_size": geo.region_block_size,
                        "block_q": BLOCK_Q,
                        "union": union,
                        "union_bucket": union_bucket,
                        "dtype": "bf16",
                        "latency_ms": f"{latency_ms:.6f}",
                        "tile_build_ms": f"{tile_build_ms:.6f}",
                        **_csv_config_fields(launch),
                    }
                )
                handle.flush()
                print(
                    f"[bench] {mode} Q={queries}: {latency_ms:.6f} ms "
                    f"(tile_build {tile_build_ms:.6f} ms excluded)",
                    flush=True,
                )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
