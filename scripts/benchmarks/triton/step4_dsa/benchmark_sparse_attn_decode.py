#!/usr/bin/env python3
"""Benchmark sparse decode attention over a hardcoded occupancy list.

Launch JSON is keyed by packed width / splits / batch / context / hpg /
head dim / region size. Add a ``CASES`` entry when that occupancy JSON exists.
Tuned mode passes the loaded launch dict so ``do_bench`` does not include
``config=None`` occupancy lookup (``torch.unique`` on ``kv_seqlens``).
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

from step4_dsa.bench_inputs import _build_decode_inputs  # noqa: E402
from step4_dsa.benchmark_cases import BenchmarkCase, tp8_geometry  # noqa: E402

from boltops.step4_dsa import triton_decode_sparse_meta  # noqa: E402
from boltops.step4_dsa.triton.decode import (  # noqa: E402
    _decode_tile_defaults,
    sparse_attention_decode,
    sparse_attn_decode_config_path,
)

BACKEND = "triton"
KERNEL = "sparse_attn_decode"
DTYPE = torch.bfloat16
WARMUP_MS = 10
REPEAT_MS = 30
SEED = 20260847
HEADS = 8
GROUPS = 1
TOPK = 512
BLOCK_REGIONS = 16
# (batch, context, splits). Append a row when the occupancy JSON lands.
CASES = ((4, 16384, 1),)


def _decode_kwargs(data: dict, *, num_splits: int) -> dict:
    geo = data["geo"]
    query = data["query"]
    return {
        "query": query,
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "packed_regions": data["packed"],
        "region_counts": data["counts"],
        "kv_seqlens": data["kv_seqlens"],
        "num_kv_groups": data["groups"],
        "region_size": geo.region_block_size,
        "softmax_scale": float(query.shape[2]) ** -0.5,
        "num_splits": num_splits,
        "block_regions": BLOCK_REGIONS,
    }


def _build_case(batch: int, context: int, num_splits: int, device: torch.device) -> dict:
    geo = tp8_geometry()
    case = BenchmarkCase(
        "tp8.decode.bench",
        "tp8-production",
        "decode",
        {
            "batch": batch,
            "context": context,
            "topk": TOPK,
            "splits": num_splits,
            "heads": HEADS,
            "groups": GROUPS,
        },
        "sparse attn decode occupancy",
    )
    inputs = _build_decode_inputs(case, device, SEED, geo=geo)
    packed, counts = triton_decode_sparse_meta(
        inputs["logits"],
        inputs["kv_seqlens"],
        inputs["block_table"],
        topk=TOPK,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
    )
    return {
        **inputs,
        "packed": packed,
        "counts": counts,
        "groups": GROUPS,
    }


def _default_launch(data: dict, *, num_splits: int) -> dict:
    query = data["query"]
    packed_width = int(data["packed"].shape[1])
    heads_per_group = query.shape[1] // data["groups"]
    slots_per_split = -(-packed_width // num_splits)
    block_h, block_d, block_r = _decode_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=int(query.shape[2]),
        slots_per_split=slots_per_split,
        block_regions=BLOCK_REGIONS,
        region_size=data["geo"].region_block_size,
        element_size=query.element_size(),
    )
    return {"BLOCK_H": block_h, "BLOCK_D": block_d, "BLOCK_R": block_r}


def _bench(data: dict, config: dict | None, *, num_splits: int) -> float:
    kwargs = _decode_kwargs(data, num_splits=num_splits)

    def run() -> None:
        sparse_attention_decode(**kwargs, config=config)

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


def _config_path(
    *,
    packed_width: int,
    num_splits: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    batch: int,
    context: int,
) -> Path:
    return Path(
        sparse_attn_decode_config_path(
            packed_width,
            num_splits,
            heads_per_group,
            head_dim,
            region_size,
            DTYPE,
            batch=batch,
            context=context,
        )
    )


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
    config_paths = [
        (
            batch,
            context,
            num_splits,
            _config_path(
                packed_width=packed_width,
                num_splits=num_splits,
                heads_per_group=heads_per_group,
                head_dim=geo.head_dim,
                region_size=geo.region_block_size,
                batch=batch,
                context=context,
            ),
        )
        for batch, context, num_splits in CASES
    ]
    for batch, context, num_splits, path in config_paths:
        print(
            f"tuned config B={batch} ctx={context} s={num_splits}: {path}",
            flush=True,
        )

    tuned_launches: dict[tuple[int, int, int], dict] = {}
    if "tuned" in args.modes:
        missing = [str(path) for *_, path in config_paths if not path.is_file()]
        if missing:
            raise SystemExit("tuned config not found:\n" + "\n".join(missing))
        tuned_launches = {
            (batch, context, num_splits): json.loads(path.read_text(encoding="utf-8"))
            for batch, context, num_splits, path in config_paths
        }

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
                "batch",
                "context",
                "splits",
                "heads",
                "groups",
                "topk",
                "packed_width",
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
        for batch, context, num_splits in CASES:
            data = _build_case(batch, context, num_splits, device)
            default_launch = _default_launch(data, num_splits=num_splits)
            for mode in args.modes:
                if mode == "tuned":
                    launch = tuned_launches[(batch, context, num_splits)]
                    latency_ms = _bench(data, launch, num_splits=num_splits)
                else:
                    latency_ms = _bench(data, {}, num_splits=num_splits)
                    launch = default_launch
                writer.writerow(
                    {
                        "backend": BACKEND,
                        "operator": KERNEL,
                        "mode": mode,
                        "batch": batch,
                        "context": context,
                        "splits": num_splits,
                        "heads": HEADS,
                        "groups": GROUPS,
                        "topk": TOPK,
                        "packed_width": packed_width,
                        "head_dim": geo.head_dim,
                        "region_size": geo.region_block_size,
                        "dtype": "bf16",
                        "latency_ms": f"{latency_ms:.6f}",
                        **_csv_config_fields(launch),
                    }
                )
                handle.flush()
                print(
                    f"[bench] {mode} B={batch} ctx={context} s={num_splits}: "
                    f"{latency_ms:.6f} ms",
                    flush=True,
                )
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
