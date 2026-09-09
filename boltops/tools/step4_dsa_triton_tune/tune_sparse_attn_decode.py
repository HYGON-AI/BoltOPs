#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Autotune BoltOps sparse decode attention for selected production shapes.

Default output is
``decode-k{width}-s{splits}-b{batch}-ctx{context}-hpg{hpg}-d{dim}-r{region}_{dtype}_{device}.json``.
``kv_seqlens`` must be a single context; otherwise the script exits.

Candidates include HIP ``waves_per_eu`` and ``BALANCE_LIVE``; ``sparse_attention_decode``
forwards warps / stages / waves at launch. ``_sparse_attn_decode_dsplit_kernel``
(non-pow2 ``head_dim``, e.g. 192) omits ``BLOCK_D``; the padded kernel writes it.
JSON never contains ``BLOCK_D1`` / ``BLOCK_D2`` (those tiles are derived from
``head_dim`` at launch).

``--splits 1`` times the production fast path (auto KV-split + merge) and still
writes the ``s1`` occupancy JSON. Pass ``--splits N`` with N>1 to tune an
explicit split count. ``--all`` retunes every occupancy JSON already in the
config directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")

import torch
import triton

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_ROOT = _REPO_ROOT / "test"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from step4_dsa.bench_inputs import _build_decode_inputs  # noqa: E402
from step4_dsa.benchmark_cases import BenchmarkCase, tp8_geometry  # noqa: E402

from boltops.step4_dsa import triton_decode_sparse_meta  # noqa: E402
from boltops.step4_dsa.triton.decode import (  # noqa: E402
    _CONFIG_DIR,
    _decode_head_dim_split,
    choose_decode_kv_splits,
    sparse_attention_decode,
    sparse_attn_decode_config_path,
)

# Match ``test/step4_dsa/test_decode.py``.
OUT_RTOL, OUT_ATOL = 1e-2, 4e-3
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3

_OCCUPANCY_RE = re.compile(
    r"decode-k(?P<packed>\d+)-s(?P<splits>\d+)-b(?P<batch>\d+)"
    r"-ctx(?P<context>\d+)-hpg(?P<hpg>\d+)-d(?P<head_dim>\d+)"
    r"-r(?P<region>\d+)_(?P<dtype>[^_]+)_(?P<device>.+)\.json$"
)
_JSON_KEY_ORDER = (
    "BALANCE_LIVE",
    "BLOCK_R",
    "BLOCK_H",
    "BLOCK_D",
    "num_warps",
    "num_stages",
    "waves_per_eu",
)
_FORBIDDEN_JSON_KEYS = frozenset({"BLOCK_D1", "BLOCK_D2"})


def candidates(block_h: int, *, block_d: int | None = None) -> list[dict]:
    result = []
    for block_r in (1, 2, 4, 8, 16):
        for num_warps in (2, 4, 8, 16):
            for num_stages in (1, 2, 3):
                for waves in (1, 2):
                    for balance_live in (True, False):
                        cfg: dict = {
                            "BALANCE_LIVE": balance_live,
                            "BLOCK_R": block_r,
                            "BLOCK_H": block_h,
                        }
                        if block_d is not None:
                            cfg["BLOCK_D"] = int(block_d)
                        cfg["num_warps"] = num_warps
                        cfg["num_stages"] = num_stages
                        cfg["waves_per_eu"] = waves
                        result.append(cfg)
    return result


def _ordered_cfg(cfg: dict, *, dsplit: bool) -> dict:
    filtered = {
        key: value
        for key, value in cfg.items()
        if key not in _FORBIDDEN_JSON_KEYS
    }
    if dsplit:
        filtered.pop("BLOCK_D", None)
    ordered = {key: filtered[key] for key in _JSON_KEY_ORDER if key in filtered}
    for key, value in filtered.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def bench(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
        )
    )


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
        "block_regions": 16,
        "return_partials": num_splits > 1,
    }


def _check_winner(data: dict, cfg: dict, *, num_splits: int) -> None:
    kwargs = _decode_kwargs(data, num_splits=num_splits)
    ref_out, ref_lse = sparse_attention_decode(**kwargs, config={})
    out, lse = sparse_attention_decode(**kwargs, config=cfg)
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
    num_splits: int,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    query = data["query"]
    heads_per_group = query.shape[1] // data["groups"]
    block_h = max(16, triton.next_power_of_2(heads_per_group))
    head_dim = int(query.shape[2])
    d_split = _decode_head_dim_split(head_dim)
    block_d = triton.next_power_of_2(head_dim)
    kwargs = _decode_kwargs(data, num_splits=num_splits)
    if d_split is not None:
        print(
            "  kernel=_sparse_attn_decode_dsplit_kernel "
            f"(omit BLOCK_D; chunks {d_split[0]}+{d_split[1]} from head_dim)",
            flush=True,
        )
        grid = candidates(block_h)
    else:
        print(
            f"  kernel=_sparse_attn_decode_kernel BLOCK_D={block_d}",
            flush=True,
        )
        grid = candidates(block_h, block_d=block_d)
    best_cfg = None
    best_ms = math.inf
    for index, cfg in enumerate(grid, 1):
        try:
            fn = lambda cfg=cfg: sparse_attention_decode(**kwargs, config=cfg)
            ms = bench(fn, warmup_ms, rep_ms)
        except Exception as exc:
            print(f"  [{index:03d}/{len(grid)}] ERROR {cfg}: {exc}", flush=True)
            continue
        print(f"  [{index:03d}/{len(grid)}] {ms:.6f} ms {cfg}", flush=True)
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError("all sparse decode candidates failed")
    _check_winner(data, best_cfg, num_splits=num_splits)
    return best_cfg, best_ms


def _discover_occupancy_jobs(config_dir: Path) -> list[dict]:
    jobs = []
    for path in sorted(config_dir.glob("decode-*.json")):
        if path.name.endswith(".timings.json"):
            continue
        match = _OCCUPANCY_RE.fullmatch(path.name)
        if match is None:
            print(f"SKIP unparsed occupancy name {path.name}", flush=True)
            continue
        packed = int(match.group("packed"))
        groups = 1
        heads_per_group = int(match.group("hpg"))
        jobs.append(
            {
                "batch": int(match.group("batch")),
                "context": int(match.group("context")),
                "splits": int(match.group("splits")),
                "topk": packed - 1,
                "heads": heads_per_group * groups,
                "groups": groups,
                "head_dim": int(match.group("head_dim")),
                "path": path,
            }
        )
    jobs.sort(key=lambda job: (job["batch"], job["context"], job["splits"]))
    return jobs


def _run_one(
    *,
    batch: int,
    context: int,
    topk: int,
    splits: int,
    heads: int,
    groups: int,
    head_dim: int,
    seed: int,
    warmup_ms: int,
    rep_ms: int,
    resume: bool,
    output: Path | None,
) -> None:
    if heads % groups:
        raise SystemExit(f"--heads {heads} does not divide into --groups {groups}")
    if splits < 1:
        raise SystemExit(f"--splits must be >= 1, got {splits}")

    device = torch.device("cuda")
    geo = tp8_geometry()
    if head_dim != geo.head_dim:
        geo = replace(geo, head_dim=head_dim)
    case = BenchmarkCase(
        "tp8.decode.tune",
        "tp8-production",
        "decode",
        {
            "batch": batch,
            "context": context,
            "topk": topk,
            "splits": splits,
            "heads": heads,
            "groups": groups,
        },
        "sparse attn decode tune",
    )
    inputs = _build_decode_inputs(case, device, seed, geo=geo)
    packed, counts = triton_decode_sparse_meta(
        inputs["logits"],
        inputs["kv_seqlens"],
        inputs["block_table"],
        topk=topk,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
    )
    data = {
        **inputs,
        "packed": packed,
        "counts": counts,
        "groups": groups,
    }
    packed_width = int(packed.shape[1])
    heads_per_group = heads // groups
    named_head_dim = int(data["query"].shape[2])
    unique_seqlens = torch.unique(data["kv_seqlens"])
    if int(unique_seqlens.numel()) != 1:
        raise SystemExit(
            "kv_seqlens must be a single context to name decode-*-b*-ctx* "
            f"JSON; unique={unique_seqlens.detach().cpu().tolist()}"
        )
    named_context = int(unique_seqlens.item())
    if named_context != context:
        raise SystemExit(
            f"--context {context} != kv_seqlens unique value {named_context}"
        )
    named_batch = int(data["query"].shape[0])
    output = output or Path(
        sparse_attn_decode_config_path(
            packed_width,
            splits,
            heads_per_group,
            named_head_dim,
            geo.region_block_size,
            data["query"].dtype,
            batch=named_batch,
            context=named_context,
        )
    )
    timing_path = output.with_suffix(".timings.json")
    if resume and output.exists():
        print(f"SKIP {output}", flush=True)
        return

    print(
        "TUNING "
        f"B={batch} ctx={context} K={topk} splits={splits} "
        f"packed={packed_width} hpg={heads_per_group} D={named_head_dim} "
        f"-> {output}",
        flush=True,
    )
    if splits == 1:
        launched = choose_decode_kv_splits(batch, groups, packed_width)
        print(
            f"  num_splits=1 fast path launches {launched}-way "
            f"(grid B={batch}, G={groups}, splits={launched}); "
            f"JSON still named s{splits}",
            flush=True,
        )
    cfg, ms = tune_shape(
        data,
        num_splits=splits,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )
    d_split = _decode_head_dim_split(named_head_dim)
    cfg = _ordered_cfg(cfg, dsplit=d_split is not None)
    print(f"BEST: {ms:.6f} ms {cfg}", flush=True)
    missing = [key for key in ("num_warps", "num_stages", "waves_per_eu") if key not in cfg]
    if missing:
        raise RuntimeError(f"tuned config omitted launch keys {missing}: {cfg}")
    if "BALANCE_LIVE" not in cfg:
        raise RuntimeError(f"tuned config omitted BALANCE_LIVE: {cfg}")
    forbidden = [key for key in _FORBIDDEN_JSON_KEYS if key in cfg]
    if forbidden:
        raise RuntimeError(f"tuned config must not contain {forbidden}: {cfg}")
    if d_split is not None:
        if "BLOCK_D" in cfg:
            raise RuntimeError(f"dsplit winner still has BLOCK_D: {cfg}")
    elif "BLOCK_D" not in cfg:
        raise RuntimeError(f"padded winner omitted BLOCK_D: {cfg}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=4) + "\n")
    timing_path.write_text(json.dumps({"ms": ms, "config": cfg}, indent=4) + "\n")
    del data, inputs, packed, counts
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260847)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retune every occupancy JSON in the sparse_attn_decode config dir.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.all and args.output is not None:
        raise SystemExit("--output cannot be combined with --all")

    jobs = (
        _discover_occupancy_jobs(Path(_CONFIG_DIR))
        if args.all
        else [
            {
                "batch": args.batch,
                "context": args.context,
                "splits": args.splits,
                "topk": args.topk,
                "heads": args.heads,
                "groups": args.groups,
                "head_dim": args.head_dim,
                "path": None,
            }
        ]
    )
    if not jobs:
        raise SystemExit(f"no occupancy JSON found under {_CONFIG_DIR}")

    print(f"jobs={len(jobs)}", flush=True)
    for job in jobs:
        _run_one(
            batch=job["batch"],
            context=job["context"],
            topk=job["topk"],
            splits=job["splits"],
            heads=job["heads"],
            groups=job["groups"],
            head_dim=job["head_dim"],
            seed=args.seed,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            resume=args.resume,
            output=args.output if job.get("path") is None else job["path"],
        )


if __name__ == "__main__":
    main()
