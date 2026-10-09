#!/usr/bin/env python3
"""Full-size launch sweep for Gluon ``mqa_logits_half_gluon_bn_align``.

Same lightop size grid as ``tune_mqa_logits_half.py`` / FP8 bn_align. Ablation
flags freeze Q/W buffering for the sweep (default = plain BN-style registers).

Example (plain on GPU 2)::

    cd BoltOPs
    HIP_VISIBLE_DEVICES=2 PYTHONPATH=. \\
      python3 -u boltops/mqa_logits/triton/tune_mqa_logits_half_gluon_bn_align.py \\
      --all-sizes --ship --warmup 5 --repeat 15 --gpu 0
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import gluon_language_available
from boltops.mqa_logits.triton.mqa_logits_half_gluon_bn_align import (
    mqa_logits_half_gluon_bn_align,
)
from boltops.mqa_logits.triton.tune_mqa_logits_half import (
    ALL_SIZE_SHAPES,
    TUNE_SHAPES,
    make_bf16,
)
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities


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


def _code_half_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


def _split_k_candidates(base_sk: int, num_heads: int) -> list[int]:
    cands = {1, 2, 4, int(base_sk)}
    return sorted(sk for sk in cands if sk >= 1 and num_heads % sk == 0)


def _w_group_candidates(head_chunk: int) -> list[int]:
    """Even w_group only (Gluon kernel asserts W_GROUP % 2 == 0)."""
    out: list[int] = []
    for g in (16, 8, 4, 2):
        if g <= head_chunk and head_chunk % g == 0:
            out.append(g)
    return out or ([2] if head_chunk % 2 == 0 else [])


def sweep_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    warmup: int,
    repeat: int,
    q_double_buf: bool,
    w_lds_pp: bool,
) -> dict[str, Any]:
    q, kv, w, ks, ke = make_bf16(s_q, s_k, num_heads)
    large_n = s_k > 8192
    base_sk = 1 if large_n else _code_half_split_k(s_q, s_k, num_heads)
    block_ms = [16] if s_q < 32 else [16, 32]
    block_ns = [64, 128]
    stages = [1, 2]
    waves = [0, 4]
    kpacks = [1, 2]
    split_ks = _split_k_candidates(base_sk, num_heads)
    kv_tiles = [1, 2]

    space: list[dict[str, Any]] = []
    for bm, bn, ns, wv, kp, sk, ktn in itertools.product(
        block_ms, block_ns, stages, waves, kpacks, split_ks, kv_tiles
    ):
        if bn % (16 * ktn) != 0:
            continue
        n_warps = bn // (16 * ktn)
        if n_warps < 1:
            continue
        head_chunk = num_heads // sk
        for wg in _w_group_candidates(head_chunk):
            space.append(
                {
                    "block_m": bm,
                    "block_n": bn,
                    "head_split_k": sk,
                    "w_group": wg,
                    "kv_tiles_n": ktn,
                    "num_stages": ns,
                    "waves_per_eu": wv,
                    "kpack": kp,
                    "num_warps": n_warps,
                }
            )

    print(
        f"  search space: {len(space)} "
        f"(bm={block_ms} bn={block_ns} sk={split_ks} "
        f"qpp={int(q_double_buf)} wpp={int(w_lds_pp)})",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    best_ms = float("inf")
    best_cfg: dict[str, Any] = {}

    for i, cfg in enumerate(space, 1):

        def run(cfg=cfg):
            mqa_logits_half_gluon_bn_align(
                q,
                kv,
                w,
                ks,
                ke,
                assume_ks_zero=False,
                block_m=cfg["block_m"],
                block_n=cfg["block_n"],
                head_split_k=cfg["head_split_k"],
                w_group=cfg["w_group"],
                kv_tiles_n=cfg["kv_tiles_n"],
                num_stages=cfg["num_stages"],
                waves_per_eu=cfg["waves_per_eu"],
                kpack=cfg["kpack"],
                q_double_buf=q_double_buf,
                w_lds_pp=w_lds_pp,
            )

        try:
            ms = per_iter_ms(run, warmup, repeat)
        except Exception as exc:  # noqa: BLE001
            results.append({**cfg, "ms": None, "error": str(exc)[:200]})
            if i % 40 == 0 or i == len(space):
                print(
                    f"  … {i}/{len(space)} best={best_cfg} "
                    f"{best_ms if best_ms < float('inf') else float('nan'):.4f}ms",
                    flush=True,
                )
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
    }


def _variant_tag(q_double_buf: bool, w_lds_pp: bool) -> str:
    parts = ["bn_align"]
    if q_double_buf:
        parts.append("qpp")
    if w_lds_pp:
        parts.append("wpp")
    if not q_double_buf and not w_lds_pp:
        parts.append("plain")
    return "_".join(parts)


def _build_ship_json(
    results: list[dict[str, Any]],
    arch: str,
    caps,
    *,
    source: str,
    variant: str,
) -> dict[str, Any]:
    by_shape: list[dict[str, Any]] = []
    by_n: dict[int, dict[str, Any]] = {}
    keys = (
        "block_m",
        "block_n",
        "head_split_k",
        "w_group",
        "kv_tiles_n",
        "num_stages",
        "waves_per_eu",
        "kpack",
    )
    for row in results:
        if not row.get("best_config"):
            continue
        s_q = int(row["s_q"])
        s_k = int(row["s_k"])
        h = int(row["num_heads"])
        cfg = dict(row["best_config"])
        main = {k: cfg[k] for k in keys if k in cfg}
        by_shape.append(
            {
                "seq_len": s_q,
                "seq_kv": s_k,
                "num_heads": h,
                "label": row.get("label") or f"m{s_q}-n{s_k}-h{h}",
                "best_ms": row.get("best_ms"),
                "main": main,
            }
        )
        prev = by_n.get(s_k)
        if prev is None:
            by_n[s_k] = row
        else:
            ph = int(prev["num_heads"])
            pm = int(prev["s_q"])
            if (h == 64 and ph != 64) or (h == ph and s_q > pm):
                by_n[s_k] = row

    buckets = []
    for n in sorted(by_n):
        row = by_n[n]
        cfg = dict(row["best_config"])
        buckets.append(
            {
                "seq_kv_leq": n,
                "label": f"n{n}-m{row['s_q']}-h{row['num_heads']}",
                "main": {k: cfg[k] for k in keys if k in cfg},
            }
        )

    default_main = (
        dict(buckets[-1]["main"])
        if buckets
        else {"w_group": 8, "kv_tiles_n": 2, "head_split_k": 1}
    )
    return {
        "meta": {
            "arch": arch,
            "triton": caps.package_version,
            "source": source,
            "variant": variant,
            "note": (
                "half Gluon bn_align launch sweep. Prefer exact by_shape (M,N,H); "
                "fall back to by_seq_kv. Does not overwrite half BN ship JSON."
            ),
        },
        "default": {"main": default_main},
        "by_shape": by_shape,
        "by_seq_kv": buckets,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=15)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--all-sizes",
        action="store_true",
        help="Sweep full 54-point size grid",
    )
    p.add_argument("--ship", action="store_true")
    p.add_argument(
        "--qpp",
        action="store_true",
        help="Enable q_double_buf (Q LDS ping-pong)",
    )
    p.add_argument(
        "--wpp",
        action="store_true",
        help="Enable w_lds_pp (W LDS ping-pong)",
    )
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    if not gluon_language_available():
        raise RuntimeError("Gluon is not available in this Triton build")

    caps = get_triton_capabilities()
    arch = arch_info.get_arch()
    shapes = ALL_SIZE_SHAPES if args.all_sizes else TUNE_SHAPES
    variant = _variant_tag(args.qpp, args.wpp)

    payload: dict[str, Any] = {
        "meta": {
            "device": torch.cuda.get_device_name(args.gpu),
            "arch": arch,
            "triton": caps.package_version,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "all_sizes": args.all_sizes,
            "n_shapes": len(shapes),
            "kernel": "mqa_logits_half_gluon_bn_align",
            "q_double_buf": bool(args.qpp),
            "w_lds_pp": bool(args.wpp),
            "variant": variant,
            "timing": "cuda_event_per_iter",
            "dtype": "bfloat16",
        },
        "results": [],
    }

    print(
        f"half bn_align tune variant={variant} device={payload['meta']['device']} "
        f"n_shapes={len(shapes)}",
        flush=True,
    )

    for s_q, s_k, h, label in shapes:
        print(f"Sweep {label} {s_q}×{s_k} H={h} ...", flush=True)
        row = sweep_shape(
            s_q,
            s_k,
            h,
            warmup=args.warmup,
            repeat=args.repeat,
            q_double_buf=bool(args.qpp),
            w_lds_pp=bool(args.wpp),
        )
        row["label"] = label
        payload["results"].append(row)
        best = row["best_ms"]
        print(
            f"  best {best if best is not None else float('nan'):.4f} ms  "
            f"cfg={row['best_config']}  ok={row['n_ok']} fail={row['n_fail']}",
            flush=True,
        )

    cap_dir = f"triton_{caps.version.replace('.', '_')}"
    out_dir = Path(__file__).resolve().parent / "configs" / cap_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tune_name = (
        f"mqa_logits_half_{variant}_all_sizes_tune_{arch}.json"
        if args.all_sizes
        else f"mqa_logits_half_{variant}_tune_{arch}.json"
    )
    tune_path = out_dir / tune_name
    tune_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {tune_path}", flush=True)

    if args.ship:
        ship = _build_ship_json(
            payload["results"],
            arch,
            caps,
            source=tune_name,
            variant=variant,
        )
        ship_path = out_dir / f"mqa_logits_half_{variant},arch={arch}.json"
        ship_path.write_text(json.dumps(ship, indent=2) + "\n")
        print(f"Wrote {ship_path}", flush=True)


if __name__ == "__main__":
    main()
