#!/usr/bin/env python3
"""Tune half ``mqa_logits_half_row`` on short-KV shapes and compare to BN / Gluon / lightop.

Short-KV grid: N in {66} × M in {16,32,64,127} × H in {8,64} (skip M>N).

Example::

    cd BoltOPs
    HIP_VISIBLE_DEVICES=4 PYTHONPATH=. python3 -u \\
      boltops/mqa_logits/triton/tune_mqa_logits_half_row_short_kv.py
    # reuse prior row configs; only re-bench compare (incl. Gluon):
    PYTHONPATH=. python3 -u boltops/mqa_logits/triton/tune_mqa_logits_half_row_short_kv.py \\
      --compare-only
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import triton

from boltops.mqa_logits.triton.mqa_logits_half import mqa_logits_half, get_half_bn_dispatch
from boltops.mqa_logits.triton.mqa_logits_half_row import (
    _half_row_split_k,
    mqa_logits_half_row,
)
from boltops.mqa_logits.triton.tune_mqa_logits_half import make_bf16
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities

try:
    import lightop
except ImportError:  # pragma: no cover
    lightop = None  # type: ignore

try:
    from boltops.mqa_logits.triton.mqa_logits_half_gluon_bn_align import (
        mqa_logits_half_gluon_bn_align,
    )
except ImportError:  # pragma: no cover
    mqa_logits_half_gluon_bn_align = None  # type: ignore


def _capability_out_dir() -> Path:
    caps = get_triton_capabilities()
    return (
        Path(__file__).resolve().parent
        / "configs"
        / f"triton_{caps.version.replace('.', '_')}"
    )


OUT_DIR = _capability_out_dir()
SHIP_BN = OUT_DIR / "mqa_logits_half,arch=gfx938.json"
SHIP_ROW = OUT_DIR / "mqa_logits_half_row_short_kv,arch=gfx938.json"

SHORT_KV_SHAPES: list[tuple[int, int, int, str]] = []
for s_q in (16, 32, 64, 127):
    for s_k in (66,):
        if s_q > s_k:
            continue
        for h in (8, 64):
            SHORT_KV_SHAPES.append((s_q, s_k, h, f"m{s_q}-n{s_k}-h{h}"))


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


def do_bench_ms(fn) -> float:
    return float(triton.testing.do_bench(fn, warmup=30, rep=80))


def load_bn_by_shape() -> dict[tuple[int, int, int], dict[str, int]]:
    if not SHIP_BN.exists():
        return {}
    data = json.loads(SHIP_BN.read_text())
    out: dict[tuple[int, int, int], dict[str, int]] = {}
    for e in data.get("by_shape") or []:
        main = e.get("main") or {}
        key = (int(e["seq_len"]), int(e["seq_kv"]), int(e["num_heads"]))
        out[key] = {k: int(main[k]) for k in main}
    return out


def bn_cfg_for(
    s_q: int, s_k: int, h: int, by_shape: dict[tuple[int, int, int], dict[str, int]]
) -> dict[str, int]:
    if (s_q, s_k, h) in by_shape:
        return dict(by_shape[(s_q, s_k, h)])
    d = get_half_bn_dispatch(s_q, s_k, h)
    return {
        k: int(d[k])
        for k in (
            "block_m",
            "block_n",
            "head_split_k",
            "num_warps",
            "num_stages",
            "waves_per_eu",
            "w_group",
        )
    }


def gluon_kw(cfg: dict[str, int]) -> dict[str, Any]:
    """Map BN by_shape knobs to plain half Gluon bn_align (same as all-size bench)."""
    bn = int(cfg["block_n"])
    nw = int(cfg["num_warps"])
    if nw > 0 and bn % (16 * nw) == 0 and (bn // (16 * nw)) in (1, 2):
        ktn = bn // (16 * nw)
    else:
        ktn = 1 if bn <= 64 else 2
    return dict(
        block_m=int(cfg["block_m"]),
        block_n=bn,
        head_split_k=int(cfg["head_split_k"]),
        w_group=int(cfg["w_group"]),
        num_stages=int(cfg["num_stages"]),
        waves_per_eu=int(cfg["waves_per_eu"]),
        kv_tiles_n=ktn,
        q_double_buf=False,
        w_lds_pp=False,
    )


def load_row_by_shape() -> dict[tuple[int, int, int], dict[str, Any]]:
    if not SHIP_ROW.exists():
        return {}
    data = json.loads(SHIP_ROW.read_text())
    out: dict[tuple[int, int, int], dict[str, Any]] = {}
    for e in data.get("by_shape") or []:
        main = e.get("main") or {}
        key = (int(e["seq_len"]), int(e["seq_kv"]), int(e["num_heads"]))
        out[key] = {k: int(main[k]) for k in main}
    return out


def compare_shape(
    s_q: int,
    s_k: int,
    h: int,
    label: str,
    rcfg: dict[str, Any],
    bcfg: dict[str, int],
) -> dict[str, Any]:
    q, kv, w, ks, ke = make_bf16(s_q, s_k, h)
    out = torch.empty(s_q, s_k, device="cuda", dtype=torch.float32)
    gkw = gluon_kw(bcfg) if mqa_logits_half_gluon_bn_align is not None else {}

    def fn_row():
        mqa_logits_half_row(q, kv, w, ks, ke, D_out=out, **rcfg)

    def fn_bn():
        mqa_logits_half(q, kv, w, ks, ke, D_out=out, **bcfg)

    t_row = do_bench_ms(fn_row) if rcfg else float("nan")
    t_bn = do_bench_ms(fn_bn)

    t_g = float("nan")
    err_g = None
    if mqa_logits_half_gluon_bn_align is not None:

        def fn_gluon():
            mqa_logits_half_gluon_bn_align(q, kv, w, ks, ke, D_out=out, **gkw)

        try:
            t_g = do_bench_ms(fn_gluon)
        except Exception as exc:  # noqa: BLE001
            err_g = f"{type(exc).__name__}: {exc}"[:160]

    t_lo = float("nan")
    if lightop is not None:

        def fn_lo():
            lightop.mqa_logits(q, kv, w, ks, ke, None, True, out)

        try:
            t_lo = do_bench_ms(fn_lo)
        except Exception:  # noqa: BLE001
            t_lo = float("nan")

    def ratio(a: float, b: float) -> float:
        if not (a == a and b == b) or b <= 0:
            return float("nan")
        return a / b

    # Auto dispatch only chooses row vs BN (Gluon stays opt-in).
    path = "row" if (t_row == t_row and t_bn == t_bn and t_row < t_bn) else "bn"
    cands = [(t_row, "row"), (t_bn, "BN"), (t_g, "Gluon"), (t_lo, "LO")]
    cands = [(t, n) for t, n in cands if t == t]
    best = min(cands, key=lambda x: x[0])[1] if cands else "?"
    cmp = {
        "label": label,
        "s_q": s_q,
        "s_k": s_k,
        "num_heads": h,
        "path": path,
        "row_ms": t_row,
        "bn_ms": t_bn,
        "gluon_ms": t_g,
        "lightop_ms": t_lo,
        "row_over_lo": ratio(t_row, t_lo),
        "row_over_bn": ratio(t_row, t_bn),
        "gluon_over_lo": ratio(t_g, t_lo),
        "gluon_over_bn": ratio(t_g, t_bn),
        "bn_over_lo": ratio(t_bn, t_lo),
        "best": best,
        "row_cfg": rcfg,
        "bn_cfg": bcfg,
        "gluon_cfg": gkw,
    }
    if err_g:
        cmp["gluon_error"] = err_g
    print(
        f"  compare path={path} row={t_row:.4f} bn={t_bn:.4f} "
        f"gluon={t_g:.4f} lo={t_lo:.4f}  "
        f"row/bn={cmp['row_over_bn']:.3f} best={best}",
        flush=True,
    )
    return cmp


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    fin = torch.isfinite(a) & torch.isfinite(b)
    if not fin.any():
        return float("nan")
    return float((a[fin] - b[fin]).abs().max())


def sweep_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    q, kv, w, ks, ke = make_bf16(s_q, s_k, num_heads)
    base_sk = _half_row_split_k(s_q, s_k, num_heads)
    split_ks = sorted(
        {1, 2, 4, 8, 16, base_sk}
        & {sk for sk in (1, 2, 4, 8, 16, base_sk) if sk >= 1 and num_heads % sk == 0}
    )
    block_kvs = [16, 32, 64, 128]
    warps = [2, 4, 8]
    stages = [1, 2]
    waves = [0, 4]

    space: list[dict[str, Any]] = []
    for bkv, nw, ns, wv, sk in itertools.product(
        block_kvs, warps, stages, waves, split_ks
    ):
        space.append(
            {
                "block_kv": bkv,
                "num_warps": nw,
                "num_stages": ns,
                "waves_per_eu": wv,
                "head_split_k": sk,
            }
        )

    print(f"  search space: {len(space)}", flush=True)
    results: list[dict[str, Any]] = []
    best_ms = float("inf")
    best_cfg: dict[str, Any] = {}
    out = torch.empty(s_q, s_k, device="cuda", dtype=torch.float32)
    ref = torch.empty_like(out)
    mqa_logits_half(q, kv, w, ks, ke, D_out=ref)  # correctness ref (runtime BN)

    for i, cfg in enumerate(space, 1):

        def run(cfg=cfg):
            mqa_logits_half_row(
                q,
                kv,
                w,
                ks,
                ke,
                D_out=out,
                block_kv=cfg["block_kv"],
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
                waves_per_eu=cfg["waves_per_eu"],
                head_split_k=cfg["head_split_k"],
            )

        try:
            run()
            err = max_abs_err(out, ref)
            if err > 1e-2:
                results.append({**cfg, "ms": None, "error": f"max_err={err}"})
                continue
            ms = per_iter_ms(run, warmup, repeat)
        except Exception as exc:  # noqa: BLE001
            results.append({**cfg, "ms": None, "error": str(exc)[:160]})
            if i % 40 == 0 or i == len(space):
                print(
                    f"  … {i}/{len(space)} best={best_cfg} "
                    f"{best_ms if best_ms < float('inf') else float('nan'):.4f}ms",
                    flush=True,
                )
            continue

        results.append({**cfg, "ms": ms, "max_err": err})
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=15)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip row sweep; reuse shipped short-KV row configs and re-bench compare",
    )
    p.add_argument(
        "--write-ship",
        action="store_true",
        help="Alias kept for PERFORMANCE.md / FP8-style CLI (ship is always written)",
    )
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    out_dir = _capability_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    global OUT_DIR, SHIP_BN, SHIP_ROW
    OUT_DIR = out_dir
    arch = arch_info.get_arch()
    SHIP_BN = OUT_DIR / f"mqa_logits_half,arch={arch}.json"
    SHIP_ROW = OUT_DIR / f"mqa_logits_half_row_short_kv,arch={arch}.json"
    bn_by_shape = load_bn_by_shape()
    row_by_shape = load_row_by_shape()
    caps = get_triton_capabilities()
    tune_path = OUT_DIR / f"mqa_logits_half_row_short_kv_tune_{arch}.json"
    ship_path = SHIP_ROW
    _ = args.write_ship  # ship always written below

    payload: dict[str, Any]
    if args.compare_only:
        if tune_path.exists():
            payload = json.loads(tune_path.read_text())
        else:
            payload = {"meta": {}, "results": [], "compare": []}
        payload.setdefault("meta", {})
        payload["meta"].update(
            {
                "device": torch.cuda.get_device_name(args.gpu),
                "arch": arch,
                "triton": triton.__version__,
                "n_shapes": len(SHORT_KV_SHAPES),
                "kernel": "mqa_logits_half_row",
                "dtype": "bfloat16",
                "timing_compare": "triton.testing.do_bench",
                "compare_backends": ["row", "bn", "gluon_bn_align", "lightop"],
                "compare_only": True,
            }
        )
        payload["compare"] = []
        print(
            f"half row short-KV compare-only device={payload['meta']['device']} "
            f"n_shapes={len(SHORT_KV_SHAPES)}",
            flush=True,
        )
        for s_q, s_k, h, label in SHORT_KV_SHAPES:
            rcfg = dict(row_by_shape.get((s_q, s_k, h)) or {})
            if not rcfg:
                print(f"Skip {label}: no row cfg in {SHIP_ROW.name}", flush=True)
                continue
            bcfg = bn_cfg_for(s_q, s_k, h, bn_by_shape)
            print(f"Compare {label} {s_q}×{s_k} H={h} ...", flush=True)
            payload["compare"].append(
                compare_shape(s_q, s_k, h, label, rcfg, bcfg)
            )
    else:
        payload = {
            "meta": {
                "device": torch.cuda.get_device_name(args.gpu),
                "arch": arch,
                "triton": triton.__version__,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "n_shapes": len(SHORT_KV_SHAPES),
                "kernel": "mqa_logits_half_row",
                "dtype": "bfloat16",
                "timing_tune": "cuda_event_per_iter",
                "timing_compare": "triton.testing.do_bench",
                "compare_backends": ["row", "bn", "gluon_bn_align", "lightop"],
            },
            "results": [],
            "compare": [],
        }
        print(
            f"half row short-KV tune device={payload['meta']['device']} "
            f"n_shapes={len(SHORT_KV_SHAPES)}",
            flush=True,
        )
        for s_q, s_k, h, label in SHORT_KV_SHAPES:
            print(f"Sweep {label} {s_q}×{s_k} H={h} ...", flush=True)
            row = sweep_shape(
                s_q, s_k, h, warmup=args.warmup, repeat=args.repeat
            )
            row["label"] = label
            payload["results"].append(row)
            print(
                f"  best {row['best_ms'] if row['best_ms'] is not None else float('nan'):.4f} ms  "
                f"cfg={row['best_config']}  ok={row['n_ok']} fail={row['n_fail']}",
                flush=True,
            )
            rcfg = dict(row["best_config"] or {})
            bcfg = bn_cfg_for(s_q, s_k, h, bn_by_shape)
            payload["compare"].append(
                compare_shape(s_q, s_k, h, label, rcfg, bcfg)
            )

    tune_path.write_text(json.dumps(payload, indent=2))

    by_shape_out = []
    for row in payload.get("results") or []:
        if not row.get("best_config"):
            continue
        by_shape_out.append(
            {
                "seq_len": row["s_q"],
                "seq_kv": row["s_k"],
                "num_heads": row["num_heads"],
                "label": row.get("label"),
                "best_ms": row.get("best_ms"),
                "main": dict(row["best_config"]),
            }
        )
    if not by_shape_out and row_by_shape:
        for (s_q, s_k, h), main in row_by_shape.items():
            by_shape_out.append(
                {
                    "seq_len": s_q,
                    "seq_kv": s_k,
                    "num_heads": h,
                    "label": f"m{s_q}-n{s_k}-h{h}",
                    "main": dict(main),
                }
            )
    dispatch = []
    for c in payload["compare"]:
        path = str(c.get("path") or ("row" if float(c.get("row_over_bn", 2.0)) < 1.0 else "bn"))
        ent = {
            "s_q": int(c["s_q"]),
            "s_k": int(c["s_k"]),
            "num_heads": int(c["num_heads"]),
            "label": c.get("label"),
            "path": path,
            "row_ms": c.get("row_ms"),
            "bn_ms": c.get("bn_ms"),
            "row_over_bn": c.get("row_over_bn"),
        }
        if path == "row":
            ent["row_cfg"] = dict(c.get("row_cfg") or {})
        dispatch.append(ent)

    ship = {
        "meta": {
            "arch": arch,
            "triton": caps.package_version,
            "source": tune_path.name,
            "kernel": "mqa_logits_half_row",
            "note": (
                "short-KV row|BN GPU compare drives auto dispatch; "
                "use compare/dispatch path=row|bn. BN knobs stay in mqa_logits_half."
            ),
        },
        "by_shape": by_shape_out,
        "compare": payload["compare"],
        "dispatch": dispatch,
    }
    ship_path.write_text(json.dumps(ship, indent=2) + "\n")

    n = len(payload["compare"])
    win_vs_lo = sum(
        1
        for c in payload["compare"]
        if c.get("lightop_ms") == c.get("lightop_ms")
        and c["row_ms"] < c["lightop_ms"] * 0.98
    )
    win_vs_bn = sum(1 for c in payload["compare"] if c.get("path") == "row")
    win_vs_g = sum(
        1
        for c in payload["compare"]
        if c.get("gluon_ms") == c.get("gluon_ms") and c["row_ms"] < c["gluon_ms"] * 0.98
    )
    g_best = sum(1 for c in payload["compare"] if c.get("best") == "Gluon")
    print(
        f"\nsummary: dispatch_row {win_vs_bn}/{n}  row>lo {win_vs_lo}/{n}  "
        f"row>gluon {win_vs_g}/{n}  gluon_best {g_best}/{n}",
        flush=True,
    )
    print(f"Wrote {tune_path}", flush=True)
    print(f"Wrote {ship_path}", flush=True)


if __name__ == "__main__":
    main()
