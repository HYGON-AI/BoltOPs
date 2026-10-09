#!/usr/bin/env python3
"""Launch-parameter sweep for half-precision ``mqa_logits_half`` (bf16).

Writes:
  * ``mqa_logits_half_tune_<arch>.json`` — full sweep dump
  * ``mqa_logits_half,arch=<arch>.json`` — shippable by_seq_kv buckets

Example::

    cd BoltOPs
    HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_mqa_logits_half.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from boltops.mqa_logits.triton.mqa_logits_half import mqa_logits_half
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities


def _code_half_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)

# Full bf16 vs lightop size grid (for per-shape ship / dispatch).
_Q_LIST = (16, 32, 64, 127)
_KV_LIST = (66, 4096, 8192, 16384, 32768, 65536, 131072)
_H_LIST = (8, 64)

# Production prefill shapes ("4608 group") — not in the dense 54-grid.
GROUP_4608_SHAPES: list[tuple[int, int, int, str]] = [
    (4608, 16384, 4, "prefill-m4608-n16k-h4"),
    (4608, 32768, 4, "prefill-m4608-n32k-h4"),
    (4608, 65536, 4, "prefill-m4608-n64k-h4"),
    (2048, 32768, 4, "prefill-m2048-n32k-h4"),
    (1024, 16384, 4, "prefill-m1024-n16k-h4"),
    (256, 16384, 4, "prefill-m256-n16k-h4"),
    (4608, 16384, 8, "prefill-m4608-n16k-h8"),
    (4608, 32768, 8, "prefill-m4608-n32k-h8"),
    (1024, 32768, 8, "prefill-m1024-n32k-h8"),
    (256, 65536, 8, "prefill-m256-n64k-h8"),
    (4608, 16384, 16, "prefill-m4608-n16k-h16"),
    (1024, 32768, 16, "prefill-m1024-n32k-h16"),
]

# Flagship production shape that drove the gap analysis.
TARGET_4608_SHAPE: tuple[int, int, int, str] = (
    4608,
    16384,
    4,
    "target-m4608-n16k-h4",
)


def _build_all_size_shapes() -> list[tuple[int, int, int, str]]:
    out: list[tuple[int, int, int, str]] = []
    for s_q in _Q_LIST:
        for s_k in _KV_LIST:
            if s_q > s_k:
                continue
            for h in _H_LIST:
                out.append((s_q, s_k, h, f"m{s_q}-n{s_k}-h{h}"))
    return out


# Representative subset (legacy / faster smoke).
TUNE_SHAPES: list[tuple[int, int, int, str]] = [
    (16, 66, 8, "tiny-h8"),
    (16, 66, 64, "tiny-h64"),
    (64, 4096, 8, "mid-h8"),
    (64, 4096, 64, "mid-h64"),
    (64, 8192, 64, "edge-bn64-h64"),
    (16, 16384, 64, "xl-m16-h64"),
    (64, 16384, 8, "xl-h8"),
    (64, 16384, 64, "xl-h64"),
    (127, 32768, 64, "xxl-h64"),
    (64, 65536, 64, "huge-h64"),
]

ALL_SIZE_SHAPES: list[tuple[int, int, int, str]] = _build_all_size_shapes()

# Mid BN64 losers vs lightop (N in {4096,8192}).
MID_TUNE_SHAPES: list[tuple[int, int, int, str]] = [
    (16, 4096, 8, "mid-m16-n4k-h8"),
    (16, 4096, 64, "mid-m16-n4k-h64"),
    (16, 8192, 8, "mid-m16-n8k-h8"),
    (16, 8192, 64, "mid-m16-n8k-h64"),
    (32, 4096, 64, "mid-m32-n4k-h64"),
    (32, 8192, 64, "mid-m32-n8k-h64"),
    (64, 4096, 8, "mid-m64-n4k-h8"),
    (64, 4096, 64, "mid-m64-n4k-h64"),
    (64, 8192, 8, "mid-m64-n8k-h8"),
    (64, 8192, 64, "mid-m64-n8k-h64"),
    (127, 4096, 64, "mid-m127-n4k-h64"),
    (127, 8192, 64, "mid-m127-n8k-h64"),
]


def make_bf16(
    s_q: int, s_k: int, num_heads: int, head_dim: int = 128, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int32, device="cuda") + (s_k - s_q)
    return q, kv, w, ks, ke


def per_iter_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return sum(times) / len(times)


def _w_group_candidates(
    head_chunk: int, *, include_one: bool, expanded: bool = False
) -> list[int]:
    # Default pruned set: {1,4,8}. Expanded also tries {2,16}.
    out: list[int] = []
    if expanded:
        cands = (1, 2, 4, 8, 16) if include_one else (2, 4, 8, 16)
    else:
        cands = (1, 8, 4) if include_one else (8, 4)
    for g in cands:
        if g <= head_chunk and head_chunk % g == 0:
            out.append(g)
    return out or ([1] if include_one else [4 if head_chunk % 4 == 0 else 1])


def _split_k_candidates(base_sk: int, num_heads: int, large_n: bool) -> list[int]:
    del large_n  # large N also uses {1,2,4}
    # Prefer {1,2,4}; keep base_sk if it still divides (tiny may add 16).
    cands = {1, 2, 4, int(base_sk)}
    return sorted(sk for sk in cands if sk >= 1 and num_heads % sk == 0)


def sweep_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    warmup: int,
    repeat: int,
    quick: bool,
    include_w1: bool = False,
    expanded: bool = False,
    pruned: bool = False,
) -> dict[str, Any]:
    q, kv, w, ks, ke = make_bf16(s_q, s_k, num_heads)
    # Default geometry family; ``block_n`` is swept unless ``--quick``.
    large_n = s_k > 8192
    base_sk = 1 if large_n else _code_half_split_k(s_q, s_k, num_heads)

    # ``--pruned``: shortlist for large-M / low-H (e.g. 4608x16384 H=4).
    # Drop knobs that historically never win here: bm=16, bn=256, warps=2,
    # waves=2, sk>1, w_group∉{1,2,4}. Keeps ~96 configs vs ~1000 expanded.
    if pruned:
        block_ms = [32, 64]
        block_ns = [64, 128]
        warps = [4, 8]
        stages = [1, 2]
        waves = [0, 4]
        split_ks = [1]
        wg_cands = [1, 2, 4]
    elif quick or s_q < 32:
        block_ms = [16]
        block_ns = [128 if large_n else 64]
        warps = [4, 8]
        stages = [1, 2]
        waves = [0, 2]
        split_ks = _split_k_candidates(base_sk, num_heads, large_n)
        if not large_n:
            split_ks = [base_sk]
        wg_cands = None
    elif expanded and s_q >= 32:
        # ``block_m=64`` pays off on large-M / low-head (dispatch heuristics
        # already pin it for H<=8 & M>=256); include it in the sweep so the
        # ship JSON records the measured winner rather than a heuristic.
        block_ms = [16, 32, 64]
        block_ns = [64, 128, 256] if large_n else [32, 64, 128]
        warps = [2, 4, 8]
        stages = [1, 2]
        waves = [0, 2, 4]
        split_ks = _split_k_candidates(base_sk, num_heads, large_n)
        wg_cands = None
    else:
        block_ms = [16, 32]
        block_ns = [64, 128, 256] if large_n else [32, 64, 128]
        warps = [4, 8] if large_n else [2, 4, 8]
        stages = [1, 2]
        waves = [0, 4]
        split_ks = _split_k_candidates(base_sk, num_heads, large_n)
        wg_cands = None

    results: list[dict[str, Any]] = []
    best_ms = float("inf")
    best_cfg: dict[str, Any] = {}

    # Materialize the search space once for progress.
    space: list[dict[str, Any]] = []
    for bm, bn, nw, ns, wv, sk in itertools.product(
        block_ms, block_ns, warps, stages, waves, split_ks
    ):
        if num_heads % sk != 0:
            continue
        head_chunk = num_heads // sk
        if wg_cands is not None:
            wgs = [g for g in wg_cands if g <= head_chunk and head_chunk % g == 0]
            if not wgs:
                wgs = [1]
        else:
            wgs = _w_group_candidates(
                head_chunk, include_one=include_w1, expanded=expanded
            )
        for wg in wgs:
            space.append(
                {
                    "block_m": bm,
                    "block_n": bn,
                    "head_split_k": sk,
                    "num_warps": nw,
                    "num_stages": ns,
                    "waves_per_eu": wv,
                    "w_group": wg,
                }
            )
    print(
        f"  search space: {len(space)} "
        f"(bm={block_ms} bn={block_ns} warps={warps} stages={stages} "
        f"waves={waves} sk={split_ks} w1={include_w1} "
        f"expanded={expanded} pruned={pruned})",
        flush=True,
    )

    for i, cfg in enumerate(space, 1):
        def run(cfg=cfg):
            mqa_logits_half(
                q,
                kv,
                w,
                ks,
                ke,
                assume_ks_zero=True,
                **cfg,
            )

        try:
            ms = per_iter_ms(run, warmup, repeat)
        except Exception as exc:  # noqa: BLE001
            results.append({**cfg, "ms": None, "error": str(exc)[:200]})
            if i % 40 == 0 or i == len(space):
                print(f"  … {i}/{len(space)} best={best_cfg} {best_ms:.4f}ms", flush=True)
            continue

        row = {**cfg, "ms": ms}
        results.append(row)
        if ms < best_ms:
            best_ms = ms
            best_cfg = dict(cfg)
        if i % 40 == 0 or i == len(space):
            print(
                f"  … {i}/{len(space)} best={best_cfg} {best_ms:.4f}ms",
                flush=True,
            )

    results.sort(key=lambda r: r.get("ms") if r.get("ms") is not None else 1e9)
    return {
        "s_q": s_q,
        "s_k": s_k,
        "num_heads": num_heads,
        "best_ms": None if best_ms == float("inf") else best_ms,
        "best_config": best_cfg,
        "top5": results[:5],
        "n_ok": sum(1 for r in results if r.get("ms") is not None),
        "n_fail": sum(1 for r in results if r.get("ms") is None),
        "n_cfgs": len(space),
        "all": results,
    }


# Coarse Q/KV/H buckets — same grid as FP8 Gluon ``by_region`` ship.
_REGION_SEQ_LEN_LEQ = (32, 64, 128)
_REGION_SEQ_KV_LEQ = (128, 4096, 8192, 16384, 32768, 65536, 131072)
_REGION_NUM_HEADS_LEQ = (16, 64)


def _region_bounds(seq_len: int, seq_kv: int, num_heads: int) -> tuple[int, int, int]:
    """Map a concrete shape onto the coarsest covering ``*_leq`` cell."""
    m_leq = min(b for b in _REGION_SEQ_LEN_LEQ if seq_len <= b)
    n_leq = min(b for b in _REGION_SEQ_KV_LEQ if seq_kv <= b)
    h_leq = min(b for b in _REGION_NUM_HEADS_LEQ if num_heads <= b)
    return m_leq, n_leq, h_leq


def _build_ship_json(results: list[dict[str, Any]], arch: str, caps) -> dict[str, Any]:
    """Emit ``by_region`` (FP8-style) + exact ``by_shape`` + ``by_seq_kv`` fallback."""
    by_shape: list[dict[str, Any]] = []
    by_n: dict[int, dict[str, Any]] = {}
    best_by_region: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in results:
        if not row.get("best_config"):
            continue
        s_q = int(row["s_q"])
        s_k = int(row["s_k"])
        h = int(row["num_heads"])
        cfg = dict(row["best_config"])
        by_shape.append(
            {
                "seq_len": s_q,
                "seq_kv": s_k,
                "num_heads": h,
                "label": row.get("label") or f"m{s_q}-n{s_k}-h{h}",
                "best_ms": row.get("best_ms"),
                "main": cfg,
            }
        )
        # Coarse N bucket: prefer H=64, then larger M.
        prev = by_n.get(s_k)
        if prev is None:
            by_n[s_k] = row
        else:
            ph = int(prev["num_heads"])
            pm = int(prev["s_q"])
            if (h == 64 and ph != 64) or (h == ph and s_q > pm):
                by_n[s_k] = row

        try:
            key = _region_bounds(s_q, s_k, h)
        except ValueError:
            continue
        ms = row.get("best_ms")
        ms_f = float(ms) if ms is not None else float("inf")
        shape_key = (s_q, s_k, h)
        prev_r = best_by_region.get(key)
        # Prefer larger concrete shapes in the cell (align mid/large M).
        if prev_r is None or shape_key > tuple(prev_r["source_shape"]):
            best_by_region[key] = {
                "best_ms": ms_f,
                "source_shape": list(shape_key),
                "source_label": row.get("label") or f"m{s_q}-n{s_k}-h{h}",
                "main": cfg,
            }

    buckets: list[dict[str, Any]] = []
    for n in sorted(by_n):
        row = by_n[n]
        buckets.append(
            {
                "seq_kv_leq": n,
                "label": f"n{n}-m{row['s_q']}-h{row['num_heads']}",
                "main": dict(row["best_config"]),
            }
        )

    by_region: list[dict[str, Any]] = []
    for m_leq, n_leq, h_leq in sorted(
        best_by_region.keys(), key=lambda k: (k[2], k[1], k[0])
    ):
        cell = best_by_region[(m_leq, n_leq, h_leq)]
        by_region.append(
            {
                "seq_len_leq": m_leq,
                "seq_kv_leq": n_leq,
                "num_heads_leq": h_leq,
                "label": f"m{m_leq}-n{n_leq}-h{h_leq}",
                "source_label": cell["source_label"],
                "best_ms": None if cell["best_ms"] == float("inf") else cell["best_ms"],
                "main": dict(cell["main"]),
            }
        )

    default_main = (
        dict(by_region[-1]["main"])
        if by_region
        else (
            dict(buckets[-1]["main"])
            if buckets
            else {
                "block_m": 16,
                "block_n": 128,
                "head_split_k": 1,
                "num_warps": 8,
                "num_stages": 2,
                "waves_per_eu": 0,
                "w_group": 8,
            }
        )
    )
    return {
        "meta": {
            "arch": arch,
            "triton": caps.package_version,
            "source": f"mqa_logits_half_tune_{arch}.json",
            "note": (
                "Prefer by_region (seq_len_leq / seq_kv_leq / num_heads_leq); "
                "legacy by_shape / by_seq_kv still readable. "
                "Launch knobs include block_m/n, split_k, warps, stages, waves, w_group."
            ),
        },
        "default": {"main": default_main},
        "by_region": by_region,
        "by_shape": by_shape,
        "by_seq_kv": buckets,
    }


def _merge_mid_into_ship(
    existing: dict[str, Any] | None,
    mid_ship: dict[str, Any],
) -> dict[str, Any]:
    """Upsert mid-sweep buckets by ``seq_kv_leq``; keep untouched buckets."""
    if not existing:
        return mid_ship
    mid_by_n = {
        int(b["seq_kv_leq"]): b for b in mid_ship.get("by_seq_kv") or []
    }
    kept: list[dict[str, Any]] = []
    for b in existing.get("by_seq_kv") or []:
        n = int(b["seq_kv_leq"])
        if n in mid_by_n:
            continue
        kept.append(b)
    merged_buckets = sorted(
        kept + list(mid_by_n.values()),
        key=lambda b: int(b["seq_kv_leq"]),
    )
    out = dict(existing)
    out["by_seq_kv"] = merged_buckets
    meta = dict(out.get("meta") or {})
    meta["mid_source"] = mid_ship.get("meta", {}).get("source")
    note = meta.get("note") or ""
    if "full mid param sweep" not in note:
        note = (note + " Mid buckets refreshed with full mid param sweep.").strip()
    meta["note"] = note
    out["meta"] = meta
    return out


def _merge_shapes_into_ship(
    existing: dict[str, Any] | None,
    new_ship: dict[str, Any],
) -> dict[str, Any]:
    """Upsert ``by_shape`` entries by (M,N,H); leave other sections intact."""
    if not existing:
        return new_ship
    key = lambda e: (int(e["seq_len"]), int(e["seq_kv"]), int(e["num_heads"]))
    by_shape = {key(e): e for e in existing.get("by_shape") or []}
    for e in new_ship.get("by_shape") or []:
        by_shape[key(e)] = e
    out = dict(existing)
    out["by_shape"] = sorted(
        by_shape.values(),
        key=lambda e: (int(e["seq_len"]), int(e["seq_kv"]), int(e["num_heads"])),
    )
    meta = dict(out.get("meta") or {})
    meta["expanded_source"] = new_ship.get("meta", {}).get("source")
    note = meta.get("note") or ""
    if "expanded search" not in note:
        note = (note + " by_shape refreshed with expanded search.").strip()
    meta["note"] = note
    out["meta"] = meta
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeat", type=int, default=25)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller search space for a fast first look",
    )
    p.add_argument(
        "--mid",
        action="store_true",
        help="Only sweep BN64 mid shapes (N in {4096,8192})",
    )
    p.add_argument(
        "--all-sizes",
        action="store_true",
        help="Sweep the full M×N×H size grid (54 points) for per-shape dispatch",
    )
    p.add_argument(
        "--group-4608",
        action="store_true",
        help="Sweep the 12 production prefill shapes (4608 group)",
    )
    p.add_argument(
        "--target-4608",
        action="store_true",
        help="Sweep only the flagship shape 4608×16384 H=4",
    )
    p.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Single shape M,N,H (e.g. 4608,16384,4); overrides other shape flags",
    )
    p.add_argument(
        "--expanded",
        action="store_true",
        help=(
            "Widen the search: block_m+=64, waves+=2, warps+=2 on large-N, "
            "w_group+={2,16}"
        ),
    )
    p.add_argument(
        "--pruned",
        action="store_true",
        help=(
            "Shortlist for large-M/low-H: bm∈{32,64} bn∈{64,128} warps∈{4,8} "
            "waves∈{0,4} sk=1 wg∈{1,2,4} (~96 configs)"
        ),
    )
    p.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard shapes as I/N (0-based), e.g. 0/8 for multi-GPU",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override tune JSON output path (used by sharded runs)",
    )
    p.add_argument(
        "--w1",
        action="store_true",
        help="Include w_group=1 (lightop-style per-head) in the sweep",
    )
    p.add_argument(
        "--full-params",
        action="store_true",
        help="Alias: force non-quick full param grid (split_k/warps/waves/…)",
    )
    p.add_argument("--ship", action="store_true", help="Also write ship JSON")
    p.add_argument(
        "--write-ship",
        action="store_true",
        help="Alias of --ship (FP8-style CLI)",
    )
    args = p.parse_args()

    if args.write_ship:
        args.ship = True
    if args.full_params:
        args.quick = False
    # Always include w_group=1 when doing a serious full-param pass.
    if args.full_params or args.expanded or args.pruned:
        args.w1 = True
    # Target-4608 defaults to pruned shortlist unless user asked for expanded.
    if args.target_4608 and not args.expanded and not args.quick:
        args.pruned = True

    torch.cuda.set_device(args.gpu)
    caps = get_triton_capabilities()
    arch = arch_info.get_arch()

    shapes: list[tuple[int, int, int, str]] = []
    if args.shape:
        m, n, h = (int(x) for x in args.shape.split(","))
        shapes = [(m, n, h, f"m{m}-n{n}-h{h}")]
    elif args.target_4608:
        shapes = [TARGET_4608_SHAPE]
    elif args.mid:
        shapes = list(MID_TUNE_SHAPES)
    else:
        if args.all_sizes:
            shapes.extend(ALL_SIZE_SHAPES)
        if args.group_4608:
            shapes.extend(GROUP_4608_SHAPES)
        if not shapes:
            shapes = list(TUNE_SHAPES)

    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_i < shard_n, f"bad --shard {args.shard}"
        shapes = shapes[shard_i::shard_n]

    include_w1 = bool(
        args.w1
        or args.mid
        or args.all_sizes
        or args.group_4608
        or args.target_4608
        or args.expanded
        or args.pruned
    )

    payload: dict[str, Any] = {
        "meta": {
            "device": torch.cuda.get_device_name(args.gpu),
            "arch": arch,
            "triton": caps.package_version,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "quick": args.quick,
            "mid": args.mid,
            "all_sizes": args.all_sizes,
            "group_4608": args.group_4608,
            "target_4608": args.target_4608,
            "shape": args.shape,
            "expanded": args.expanded,
            "pruned": args.pruned,
            "shard": args.shard,
            "include_w1": include_w1,
            "n_shapes": len(shapes),
            "timing": "cuda_event_per_iter",
            "dtype": "bfloat16",
        },
        "results": [],
    }

    for s_q, s_k, h, label in shapes:
        print(f"Sweep {label} {s_q}×{s_k} H={h} ...", flush=True)
        row = sweep_shape(
            s_q,
            s_k,
            h,
            warmup=args.warmup,
            repeat=args.repeat,
            quick=args.quick,
            include_w1=include_w1,
            expanded=args.expanded,
            pruned=args.pruned,
        )
        row["label"] = label
        payload["results"].append(row)
        print(
            f"  best {row['best_ms']:.4f} ms  cfg={row['best_config']}  "
            f"ok={row['n_ok']} fail={row['n_fail']}",
            flush=True,
        )

    cap_dir = f"triton_{caps.version.replace('.', '_')}"
    out_dir = Path(__file__).resolve().parent / "configs" / cap_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        tune_path = Path(args.out)
        tune_path.parent.mkdir(parents=True, exist_ok=True)
        tune_name = tune_path.name
    elif args.mid:
        tune_name = f"mqa_logits_half_mid_tune_{arch}.json"
        tune_path = out_dir / tune_name
    elif args.target_4608 or (
        args.shape and args.shape.replace(" ", "") == "4608,16384,4"
    ):
        tag = "pruned" if args.pruned else ("expanded" if args.expanded else "std")
        tune_name = f"mqa_logits_half_target4608_{tag}_tune_{arch}.json"
        tune_path = out_dir / tune_name
    elif args.group_4608 and not args.all_sizes:
        tag = "expanded" if args.expanded else ("pruned" if args.pruned else "std")
        shard_sfx = f"_shard{shard_i}of{shard_n}" if shard_n else ""
        tune_name = f"mqa_logits_half_group4608_{tag}_tune_{arch}{shard_sfx}.json"
        tune_path = out_dir / tune_name
    elif args.all_sizes:
        tag = "expanded" if args.expanded else "std"
        shard_sfx = f"_shard{shard_i}of{shard_n}" if shard_n else ""
        tune_name = f"mqa_logits_half_all_sizes_{tag}_tune_{arch}{shard_sfx}.json"
        tune_path = out_dir / tune_name
    else:
        tune_name = f"mqa_logits_half_tune_{arch}.json"
        tune_path = out_dir / tune_name
    # Drop full ``all`` from disk if huge — keep top5 + best.
    slim = {
        "meta": payload["meta"],
        "results": [
            {k: v for k, v in r.items() if k != "all"} for r in payload["results"]
        ],
    }
    tune_path.write_text(json.dumps(slim, indent=2) + "\n")
    print(f"Wrote {tune_path}")

    if args.ship and not args.shard:
        ship_path = out_dir / f"mqa_logits_half,arch={arch}.json"
        mid_ship = _build_ship_json(payload["results"], arch, caps)
        mid_ship["meta"]["source"] = tune_name
        merge_shapes = (
            args.group_4608
            or args.target_4608
            or args.shape
            or args.expanded
            or args.pruned
            or args.all_sizes
        )
        if ship_path.is_file() and (args.mid or merge_shapes):
            existing = json.loads(ship_path.read_text())
            if args.mid and not merge_shapes:
                ship = _merge_mid_into_ship(existing, mid_ship)
            else:
                ship = _merge_shapes_into_ship(existing, mid_ship)
        else:
            ship = mid_ship
        ship_path.write_text(json.dumps(ship, indent=2) + "\n")
        print(f"Wrote {ship_path}")


if __name__ == "__main__":
    main()
