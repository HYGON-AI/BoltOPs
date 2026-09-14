#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Autotune BoltOps sparse decode attention for selected production shapes.

Default output is ``sparse_attn_decode_{device}.json``. Multiple runtime shapes
are stored as ``configs`` entries and selected through each entry's ``match``
dict. ``kv_seqlens`` must contain a single context while tuning.

Candidates include HIP ``waves_per_eu`` and ``BALANCE_LIVE``; ``sparse_attention_decode``
forwards warps / stages / waves at launch. ``_sparse_attn_decode_dsplit_kernel``
(non-pow2 ``head_dim``, e.g. 192) omits ``BLOCK_D``; the padded kernel writes it.
JSON never contains ``BLOCK_D1`` / ``BLOCK_D2`` (those tiles are derived from
``head_dim`` at launch).

``--splits 1`` times the production fast path (auto KV-split + merge) and still
writes the ``s1`` occupancy JSON, including ``AUTO_KV_SPLITS`` for the launch
grid that won. Pass ``--auto-kv-splits 64,72,80`` to search those grids
(measured with ``auto_kv_split=False`` and ``return_partials=False`` so merge
is included). Multiple split counts use a two-stage search: a seed tile picks
the split, then the full tile grid runs at that split. Without that flag, the
heuristic from :func:`choose_decode_kv_splits` is used as the sole launch split
and still written into the JSON.

Pass ``--splits N`` with N>1 to tune an explicit split count (partials, no merge;
no ``AUTO_KV_SPLITS`` in the JSON). ``--all`` retunes every occupancy JSON already
in the config directory and may be combined with ``--auto-kv-splits`` (s1 jobs
only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from boltops.step4_dsa.triton.common import _device_name_for_config  # noqa: E402
from boltops.step4_dsa.triton.decode import (  # noqa: E402
    _CONFIG_DIR,
    _clamp_auto_kv_splits,
    _decode_config_filename,
    _decode_head_dim_split,
    choose_decode_kv_splits,
    sparse_attention_decode,
)

# Match ``test/step4_dsa/test_decode.py``.
OUT_RTOL, OUT_ATOL = 1e-2, 4e-3
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3

_JSON_KEY_ORDER = (
    "AUTO_KV_SPLITS",
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
        for num_warps in (1, 2, 4, 8, 16):
            for num_stages in (1, 2, 3, 4):
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


def _parse_auto_kv_splits(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise SystemExit(f"--auto-kv-splits entries must be >= 1, got {value}")
        values.append(value)
    if not values:
        raise SystemExit("--auto-kv-splits must list at least one positive int")
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(values))


def bench(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
        )
    )


def _decode_kwargs(
    data: dict,
    *,
    caller_splits: int,
    launch_splits: int | None = None,
) -> dict:
    """Build kwargs for a timed call.

    ``caller_splits==1``: production fast path. Pin the launch grid with
    ``auto_kv_split=False`` and ``return_partials=False`` so attention + merge
    are both measured. ``caller_splits>1``: explicit partials path (no merge).
    """
    geo = data["geo"]
    query = data["query"]
    kwargs = {
        "query": query,
        "key_cache": data["key_cache"],
        "value_cache": data["value_cache"],
        "packed_regions": data["packed"],
        "region_counts": data["counts"],
        "kv_seqlens": data["kv_seqlens"],
        "num_kv_groups": data["groups"],
        "region_size": geo.region_block_size,
        "softmax_scale": float(query.shape[2]) ** -0.5,
        "block_regions": 16,
    }
    if caller_splits == 1:
        if launch_splits is None:
            raise ValueError("launch_splits required when caller_splits==1")
        kwargs["num_splits"] = int(launch_splits)
        kwargs["auto_kv_split"] = False
        kwargs["return_partials"] = False
    else:
        kwargs["num_splits"] = int(caller_splits)
        kwargs["return_partials"] = True
    return kwargs


def _seed_tile(block_h: int, *, block_d: int | None = None) -> dict:
    """Known-good tile used to pick ``AUTO_KV_SPLITS`` before the full grid."""
    cfg: dict = {
        "BALANCE_LIVE": False,
        "BLOCK_R": 2,
        "BLOCK_H": block_h,
        "num_warps": 1,
        "num_stages": 3,
        "waves_per_eu": 2,
    }
    if block_d is not None:
        cfg["BLOCK_D"] = int(block_d)
    return cfg


def _check_winner(
    data: dict,
    cfg: dict,
    *,
    caller_splits: int,
    launch_splits: int | None,
) -> None:
    kwargs = _decode_kwargs(
        data, caller_splits=caller_splits, launch_splits=launch_splits
    )
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
    caller_splits: int,
    auto_kv_split_candidates: list[int],
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    query = data["query"]
    packed_width = int(data["packed"].shape[1])
    heads_per_group = query.shape[1] // data["groups"]
    block_h = max(16, triton.next_power_of_2(heads_per_group))
    head_dim = int(query.shape[2])
    d_split = _decode_head_dim_split(head_dim)
    block_d = triton.next_power_of_2(head_dim)
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

    if caller_splits == 1:
        split_candidates = [
            _clamp_auto_kv_splits(value, packed_width)
            for value in auto_kv_split_candidates
        ]
        split_candidates = list(dict.fromkeys(split_candidates))
        print(
            f"  s1 fast path: search AUTO_KV_SPLITS={split_candidates} "
            f"(attention + merge)",
            flush=True,
        )
    else:
        split_candidates = [caller_splits]
        print(
            f"  explicit splits={caller_splits} (partials, no merge)",
            flush=True,
        )

    two_stage = caller_splits == 1 and len(split_candidates) > 1
    if two_stage:
        seed = _seed_tile(block_h, block_d=None if d_split is not None else block_d)
        print(f"  stage1 seed tile {seed}", flush=True)
        best_split = None
        best_split_ms = math.inf
        for index, launch_splits in enumerate(split_candidates, 1):
            kwargs = _decode_kwargs(
                data, caller_splits=1, launch_splits=launch_splits
            )
            try:
                fn = lambda kwargs=kwargs, seed=seed: sparse_attention_decode(
                    **kwargs, config=seed
                )
                ms = bench(fn, warmup_ms, rep_ms)
            except Exception as exc:
                print(
                    f"  stage1 [{index:03d}/{len(split_candidates)}] "
                    f"ERROR splits={launch_splits}: {exc}",
                    flush=True,
                )
                continue
            print(
                f"  stage1 [{index:03d}/{len(split_candidates)}] "
                f"{ms:.6f} ms splits={launch_splits}",
                flush=True,
            )
            if ms < best_split_ms:
                best_split, best_split_ms = launch_splits, ms
        if best_split is None:
            raise RuntimeError("all AUTO_KV_SPLITS seed candidates failed")
        print(
            f"  stage1 winner AUTO_KV_SPLITS={best_split} ({best_split_ms:.6f} ms); "
            f"stage2 tile grid",
            flush=True,
        )
        split_candidates = [best_split]

    best_cfg = None
    best_ms = math.inf
    best_launch_splits = None
    total = len(split_candidates) * len(grid)
    index = 0
    for launch_splits in split_candidates:
        kwargs = _decode_kwargs(
            data,
            caller_splits=caller_splits,
            launch_splits=launch_splits if caller_splits == 1 else None,
        )
        for cfg in grid:
            index += 1
            try:
                fn = lambda cfg=cfg, kwargs=kwargs: sparse_attention_decode(
                    **kwargs, config=cfg
                )
                ms = bench(fn, warmup_ms, rep_ms)
            except Exception as exc:
                print(
                    f"  [{index:03d}/{total}] ERROR splits={launch_splits} {cfg}: {exc}",
                    flush=True,
                )
                continue
            print(
                f"  [{index:03d}/{total}] {ms:.6f} ms splits={launch_splits} {cfg}",
                flush=True,
            )
            if ms < best_ms:
                best_cfg, best_ms, best_launch_splits = cfg, ms, launch_splits
    torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError("all sparse decode candidates failed")
    winner = dict(best_cfg)
    if caller_splits == 1:
        winner["AUTO_KV_SPLITS"] = int(best_launch_splits)
    _check_winner(
        data,
        winner,
        caller_splits=caller_splits,
        launch_splits=best_launch_splits if caller_splits == 1 else None,
    )
    return winner, best_ms


def _discover_occupancy_jobs(config_dir: Path) -> list[dict]:
    """Build exact-shape retune jobs from consolidated device config files."""
    jobs = []
    for path in sorted(config_dir.glob("sparse_attn_decode_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIP invalid config {path.name}: {exc}", flush=True)
            continue
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("configs", ()):
            match = entry.get("match", {})
            if not isinstance(match, dict):
                continue
            required = (
                "s_q",
                "s_kv",
                "h_q",
                "h_kv",
                "topk",
                "d",
                "num_splits",
            )
            if any(key not in match for key in required):
                print(f"SKIP non-exact match in {path.name}: {match}", flush=True)
                continue
            jobs.append(
                {
                    "batch": int(match["s_q"]),
                    "context": int(match["s_kv"]),
                    "splits": int(match["num_splits"]),
                    "topk": int(match["topk"]) - 1,
                    "heads": int(match["h_q"]),
                    "groups": int(match["h_kv"]),
                    "head_dim": int(match["d"]),
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
    auto_kv_splits: list[int] | None,
) -> None:
    if heads % groups:
        raise SystemExit(f"--heads {heads} does not divide into --groups {groups}")
    if splits < 1:
        raise SystemExit(f"--splits must be >= 1, got {splits}")
    if auto_kv_splits is not None and splits != 1:
        raise SystemExit("--auto-kv-splits only applies when --splits 1")

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
            "kv_seqlens must contain one context per tuning job; "
            f"unique={unique_seqlens.detach().cpu().tolist()}"
        )
    named_context = int(unique_seqlens.item())
    if named_context != context:
        raise SystemExit(
            f"--context {context} != kv_seqlens unique value {named_context}"
        )
    named_batch = int(data["query"].shape[0])
    device_tag = _device_name_for_config()
    output = output or Path(_CONFIG_DIR) / _decode_config_filename(device_tag)
    match_key = {
        "s_q": named_batch,
        "s_kv": named_context,
        "h_q": int(data["query"].shape[1]),
        "h_kv": int(data["key_cache"].shape[1]),
        "topk": packed_width,
        "d": named_head_dim,
        "num_splits": splits,
        "region_size": int(geo.region_block_size),
        "dtype": str(data["query"].dtype).replace("torch.", ""),
    }
    if resume and output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if any(
            entry.get("match") == match_key
            for entry in existing.get("configs", ())
            if isinstance(entry, dict)
        ):
            print(f"SKIP {match_key} in {output}", flush=True)
            return

    print(
        "TUNING "
        f"B={batch} ctx={context} K={topk} splits={splits} "
        f"packed={packed_width} hpg={heads_per_group} D={named_head_dim} "
        f"-> {output}",
        flush=True,
    )
    if splits == 1:
        if auto_kv_splits is None:
            heuristic = choose_decode_kv_splits(batch, groups, packed_width)
            auto_kv_split_candidates = [heuristic]
            print(
                f"  num_splits=1 fast path: heuristic AUTO_KV_SPLITS={heuristic} "
                f"(grid B={batch}, G={groups}); JSON still named s{splits}",
                flush=True,
            )
        else:
            auto_kv_split_candidates = auto_kv_splits
            print(
                f"  num_splits=1 fast path: searching AUTO_KV_SPLITS="
                f"{auto_kv_split_candidates}; JSON still named s{splits}",
                flush=True,
            )
    else:
        auto_kv_split_candidates = [splits]

    cfg, ms = tune_shape(
        data,
        caller_splits=splits,
        auto_kv_split_candidates=auto_kv_split_candidates,
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
    if splits == 1 and "AUTO_KV_SPLITS" not in cfg:
        raise RuntimeError(f"s1 tuned config omitted AUTO_KV_SPLITS: {cfg}")
    if splits != 1 and "AUTO_KV_SPLITS" in cfg:
        raise RuntimeError(f"explicit-split config must not contain AUTO_KV_SPLITS: {cfg}")
    forbidden = [key for key in _FORBIDDEN_JSON_KEYS if key in cfg]
    if forbidden:
        raise RuntimeError(f"tuned config must not contain {forbidden}: {cfg}")
    if d_split is not None:
        if "BLOCK_D" in cfg:
            raise RuntimeError(f"dsplit winner still has BLOCK_D: {cfg}")
    elif "BLOCK_D" not in cfg:
        raise RuntimeError(f"padded winner omitted BLOCK_D: {cfg}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if output.exists():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    if not isinstance(payload, dict) or not isinstance(payload.get("configs"), list):
        payload = {
            "metadata": {
                "triton_version": triton.__version__,
                "device_name": device_tag,
            },
            "configs": [],
        }
    configs = [
        entry
        for entry in payload["configs"]
        if isinstance(entry, dict) and entry.get("match") != match_key
    ]
    configs.append({"match": match_key, "config": cfg, "latency_ms": ms})
    payload["configs"] = configs
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
        "--auto-kv-splits",
        type=str,
        default=None,
        help=(
            "Comma-separated launch split counts to search on --splits 1 "
            "(attention+merge). Default: heuristic choose_decode_kv_splits only."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retune every occupancy JSON in the sparse_attn_decode config dir. Can combine with --auto-kv-splits for s1 jobs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.all and args.output is not None:
        raise SystemExit("--output cannot be combined with --all")
    auto_kv_splits = _parse_auto_kv_splits(args.auto_kv_splits)

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
            auto_kv_splits=auto_kv_splits if job["splits"] == 1 else None,
        )


if __name__ == "__main__":
    main()
