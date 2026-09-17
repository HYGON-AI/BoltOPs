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


def _w_group_candidates(head_chunk: int, *, include_one: bool) -> list[int]:
    # Pruned set: {1,4,8} is enough in practice (drop 2/16).
    out: list[int] = []
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
) -> dict[str, Any]:
    q, kv, w, ks, ke = make_bf16(s_q, s_k, num_heads)
    # Default geometry family; ``block_n`` is swept unless ``--quick``.
    large_n = s_k > 8192
    base_sk = 1 if large_n else _code_half_split_k(s_q, s_k, num_heads)
    block_ms = [16] if quick or s_q < 32 else [16, 32]
    if quick:
        block_ns = [128 if large_n else 64]
    elif large_n:
        block_ns = [64, 128, 256]
    else:
        block_ns = [32, 64, 128]
    warps = [4, 8] if quick else ([4, 8] if large_n else [2, 4, 8])
    stages = [1, 2]
    # Pruned waves: {0,4} (drop 2).
    waves = [0, 4] if not quick else [0, 2]
    split_ks = _split_k_candidates(base_sk, num_heads, large_n)
    if quick and not large_n:
        split_ks = [base_sk]

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
        for wg in _w_group_candidates(head_chunk, include_one=include_w1):
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
        f"waves={waves} sk={split_ks} w1={include_w1})",
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
    if args.full_params:
        args.w1 = True

    torch.cuda.set_device(args.gpu)
    caps = get_triton_capabilities()
    arch = arch_info.get_arch()
    if args.mid:
        shapes = MID_TUNE_SHAPES
    elif args.all_sizes:
        shapes = ALL_SIZE_SHAPES
    else:
        shapes = TUNE_SHAPES
    include_w1 = bool(args.w1 or args.mid or args.all_sizes)

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
            "include_w1": include_w1,
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
    if args.mid:
        tune_name = f"mqa_logits_half_mid_tune_{arch}.json"
    elif args.all_sizes:
        tune_name = f"mqa_logits_half_all_sizes_tune_{arch}.json"
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

    if args.ship:
        ship_path = out_dir / f"mqa_logits_half,arch={arch}.json"
        mid_ship = _build_ship_json(payload["results"], arch, caps)
        mid_ship["meta"]["source"] = tune_name
        if args.mid and ship_path.is_file():
            existing = json.loads(ship_path.read_text())
            ship = _merge_mid_into_ship(existing, mid_ship)
        else:
            ship = mid_ship
        ship_path.write_text(json.dumps(ship, indent=2) + "\n")
        print(f"Wrote {ship_path}")


if __name__ == "__main__":
    main()
