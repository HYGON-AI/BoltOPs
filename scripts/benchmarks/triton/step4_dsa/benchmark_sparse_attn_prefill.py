#!/usr/bin/env python3
"""Benchmark scalar sparse prefill over a query-count sweep.

Uses one launch JSON keyed by packed width / hpg / head dim / region size
(not by Q). The tiled kernel lives in a sibling script.
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

from boltops.step4_dsa.triton.prefill import (  # noqa: E402
    _per_query_tile_defaults,
    sparse_attention_prefill,
    sparse_attn_prefill_config_path,
)

BACKEND = "triton"
KERNEL = "sparse_attn_prefill"
DTYPE = torch.bfloat16
WARMUP_MS = 10
REPEAT_MS = 30
SEED = 20260843
HEADS = 8
GROUPS = 1
TOPK = 512
CACHE_REGIONS = 4096
BLOCK_REGIONS = 16
Q_VALUES = (128,)


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
        "block_regions": BLOCK_REGIONS,
    }


def _build_case(queries: int, device: torch.device) -> dict:
    geo = tp8_geometry()
    case = BenchmarkCase(
        "tp8.prefill.bench",
        "tp8-production",
        "prefill",
        {
            "queries": queries,
            "cache_regions": CACHE_REGIONS,
            "topk": TOPK,
            "heads": HEADS,
            "groups": GROUPS,
        },
        "sparse attn prefill Q sweep",
    )
    return _build_prefill_inputs(case, device, SEED, geo=geo)


def _default_launch(data: dict) -> dict:
    query = data["query"]
    packed_width = int(data["packed"].shape[1])
    heads_per_group = query.shape[1] // data["groups"]
    block_h, block_d, block_r = _per_query_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=int(query.shape[2]),
        topk=packed_width,
        block_regions=BLOCK_REGIONS,
        region_size=data["geo"].region_block_size,
        element_size=query.element_size(),
    )
    return {"BLOCK_H": block_h, "BLOCK_D": block_d, "BLOCK_R": block_r}


def _bench(data: dict, config: dict | None) -> float:
    kwargs = _prefill_kwargs(data)

    def run() -> None:
        sparse_attention_prefill(**kwargs, config=config)

    run()
    return float(
        triton.testing.do_bench(
            run, warmup=WARMUP_MS, rep=REPEAT_MS, return_mode="median"
        )
    )


def _csv_config_fields(launch: dict) -> dict:
    return {
        "BLOCK_R": launch.get("BLOCK_R", ""),
        "BLOCK_H": launch.get("BLOCK_H", ""),
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
    heads_per_group = HEADS // GROUPS
    config_path = Path(
        sparse_attn_prefill_config_path(
            packed_width,
            heads_per_group,
            geo.head_dim,
            geo.region_block_size,
            DTYPE,
        )
    )
    print(f"tuned config: {config_path}", flush=True)
    if "tuned" in args.modes and not config_path.is_file():
        raise SystemExit(f"tuned config not found: {config_path}")

    tuned_launch: dict = {}
    if "tuned" in args.modes:
        tuned_launch = json.loads(config_path.read_text(encoding="utf-8"))

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
                "dtype",
                "latency_ms",
                "BLOCK_R",
                "BLOCK_H",
                "BLOCK_D",
                "num_warps",
                "num_stages",
                "waves_per_eu",
            ),
        )
        writer.writeheader()
        for queries in Q_VALUES:
            data = _build_case(queries, device)
            default_launch = _default_launch(data)
            for mode in args.modes:
                if mode == "tuned":
                    latency_ms = _bench(data, None)
                    launch = tuned_launch
                else:
                    latency_ms = _bench(data, {})
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
                        "dtype": "bf16",
                        "latency_ms": f"{latency_ms:.6f}",
                        **_csv_config_fields(launch),
                    }
                )
                handle.flush()
                print(
                    f"[bench] {mode} Q={queries}: {latency_ms:.6f} ms",
                    flush=True,
                )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
