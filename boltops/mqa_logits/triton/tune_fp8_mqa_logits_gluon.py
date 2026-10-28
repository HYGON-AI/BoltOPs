#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Full-size launch sweep for FP8 Gluon ``fp8_mqa_logits_gluon`` (Triton 3.6+).

Sweeps the lightop size grid and writes:
  * ``fp8_mqa_logits_gluon_all_sizes_tune_<arch>.json`` (per-shape exact results)
  * ``fp8_mqa_logits_gluon,arch=<arch>.json`` with coarse ``by_region`` (Q/KV/H leq)

Example::

    cd BoltOPs
    HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \\
      python3 -u boltops/mqa_logits/triton/tune_fp8_mqa_logits_gluon.py \\
      --all-sizes --ship --warmup 5 --repeat 15
"""


from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
    fp8_mqa_logits_gluon,
    gluon_prefill_available,
)
from boltops.mqa_logits.triton.tune_fp8_mqa_logits import make_inputs
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities

_Q_LIST = (16, 32, 64, 127)
_KV_LIST = (66, 4096, 8192, 16384, 32768, 65536, 131072)
_H_LIST = (8, 64)

# Coarse ship buckets (not a copy of the lightop exact grid).
_REGION_SEQ_LEN_LEQ = (32, 64, 128)
_REGION_SEQ_KV_LEQ = (128, 4096, 8192, 16384, 32768, 65536, 131072)
_REGION_NUM_HEADS_LEQ = (16, 64)

_GLUON_MAIN_KEYS = (
    "block_m",
    "block_n",
    "head_split_k",
    "w_group",
    "kv_tiles_n",
    "num_stages",
    "waves_per_eu",
    "kpack",
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


ALL_SIZE_SHAPES = _build_all_size_shapes()


def _region_bounds(seq_len: int, seq_kv: int, num_heads: int) -> tuple[int, int, int]:
    """Map a concrete shape onto the coarsest covering ``*_leq`` cell."""
    m_leq = min(b for b in _REGION_SEQ_LEN_LEQ if seq_len <= b)
    n_leq = min(b for b in _REGION_SEQ_KV_LEQ if seq_kv <= b)
    h_leq = min(b for b in _REGION_NUM_HEADS_LEQ if num_heads <= b)
    return m_leq, n_leq, h_leq


def _main_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in _GLUON_MAIN_KEYS if k in cfg}


# Smaller smoke set.
TUNE_SHAPES: list[tuple[int, int, int, str]] = [
    (16, 66, 64, "tiny-h64"),
    (64, 4096, 64, "mid-h64"),
    (64, 8192, 64, "edge-h64"),
    (64, 16384, 64, "xl-h64"),
    (127, 32768, 64, "xxl-h64"),
]


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


def _code_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
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
    # Even groups only (Gluon assert); prefer {4,8,16}.
    out: list[int] = []
    for g in (16, 8, 4, 2):
        if g <= head_chunk and head_chunk % g == 0:
            out.append(g)
    return out or [2]


def sweep_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    q, kv, scales, w, ks, ke = make_inputs(s_q, s_k, num_heads, 128, False)
    base_sk = _code_split_k(s_q, s_k, num_heads)
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
        # warps derived: bn / (16 * kv_tiles_n) must be >= 1 integer.
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
        f"wg/ktn/stages/waves/kpack pruned)",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    best_ms = float("inf")
    best_cfg: dict[str, Any] = {}

    for i, cfg in enumerate(space, 1):
        def run(cfg=cfg):
            fp8_mqa_logits_gluon(
                q,
                kv,
                scales,
                w,
                ks,
                ke,
                force_gluon_kernel=True,
                assume_ks_zero=False,
                block_m=cfg["block_m"],
                block_n=cfg["block_n"],
                head_split_k=cfg["head_split_k"],
                w_group=cfg["w_group"],
                kv_tiles_n=cfg["kv_tiles_n"],
                num_stages=cfg["num_stages"],
                waves_per_eu=cfg["waves_per_eu"],
                kpack=cfg["kpack"],
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


def _build_ship_json(results: list[dict[str, Any]], arch: str, caps) -> dict[str, Any]:
    """Aggregate exact tune rows into coarse ``by_region`` Q/KV/H upper bounds.

    Each region keeps the config from the **largest** in-cell tune point
    ``(s_q, s_k, num_heads)`` so mid/large M keep their own knobs when
    ``seq_len_leq`` has a 64 tier. Exact per-shape rows stay in the tune
    artifact only.
    """
    best_by_region: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in results:
        cfg = row.get("best_config")
        if not cfg:
            continue
        s_q = int(row["s_q"])
        s_k = int(row["s_k"])
        h = int(row["num_heads"])
        try:
            key = _region_bounds(s_q, s_k, h)
        except ValueError:
            continue
        ms = row.get("best_ms")
        ms_f = float(ms) if ms is not None else float("inf")
        shape_key = (s_q, s_k, h)
        prev = best_by_region.get(key)
        # Prefer larger concrete shapes in the cell (align mid/large M).
        if prev is None or shape_key > tuple(prev["source_shape"]):
            best_by_region[key] = {
                "best_ms": ms_f,
                "source_shape": list(shape_key),
                "source_label": row.get("label") or f"m{s_q}-n{s_k}-h{h}",
                "main": _main_from_cfg(dict(cfg)),
            }

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
                "main": cell["main"],
            }
        )

    default_main: dict[str, Any]
    if by_region:
        # Prefer the largest-N / H=64 cell as default when present.
        default_main = dict(by_region[-1]["main"])
        for entry in reversed(by_region):
            if int(entry["num_heads_leq"]) >= 64 and int(entry["seq_kv_leq"]) >= 131072:
                default_main = dict(entry["main"])
                break
    else:
        default_main = {
            "block_m": 32,
            "block_n": 128,
            "head_split_k": 1,
            "w_group": 16,
            "kv_tiles_n": 2,
            "num_stages": 2,
            "waves_per_eu": 0,
            "kpack": 2,
        }

    return {
        "meta": {
            "arch": arch,
            "triton": caps.package_version,
            "source": f"fp8_mqa_logits_gluon_all_sizes_tune_{arch}.json",
            "note": (
                "Prefer by_region (seq_len_leq / seq_kv_leq / num_heads_leq); "
                "legacy by_shape / by_seq_kv still readable. Coarse Q/KV/H "
                "buckets (M in {32,64,128}) aggregated from all-sizes Gluon "
                "tune (not exact grid)."
            ),
        },
        "default": {"main": default_main},
        "by_region": by_region,
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
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    if not gluon_prefill_available():
        raise RuntimeError("Gluon is not available in this Triton build")

    caps = get_triton_capabilities()
    arch = arch_info.get_arch()
    shapes = ALL_SIZE_SHAPES if args.all_sizes else TUNE_SHAPES

    payload: dict[str, Any] = {
        "meta": {
            "device": torch.cuda.get_device_name(args.gpu),
            "arch": arch,
            "triton": caps.package_version,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "all_sizes": args.all_sizes,
            "n_shapes": len(shapes),
            "force_gluon_kernel": True,
            "timing": "cuda_event_per_iter",
            "dtype": "fp8_e4m3",
        },
        "results": [],
    }

    for s_q, s_k, h, label in shapes:
        print(f"Sweep {label} {s_q}×{s_k} H={h} ...", flush=True)
        row = sweep_shape(
            s_q, s_k, h, warmup=args.warmup, repeat=args.repeat
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
    tune_name = (
        f"fp8_mqa_logits_gluon_all_sizes_tune_{arch}.json"
        if args.all_sizes
        else f"fp8_mqa_logits_gluon_tune_{arch}.json"
    )
    tune_path = out_dir / tune_name
    tune_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {tune_path}")

    if args.ship:
        ship = _build_ship_json(payload["results"], arch, caps)
        ship["meta"]["source"] = tune_name
        ship_path = out_dir / f"fp8_mqa_logits_gluon,arch={arch}.json"
        ship_path.write_text(json.dumps(ship, indent=2) + "\n")
        print(f"Wrote {ship_path}")


if __name__ == "__main__":
    main()
