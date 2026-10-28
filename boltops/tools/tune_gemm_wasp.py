#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Sweep / record harness for internal ``gemm_fp16`` WASP/WDRA.

One script + ``boltops/gemm/triton/configs/gemm_fp16/`` (created on promote):

  python -m boltops.tools.tune_gemm_wasp --list-shapes
  python -m boltops.tools.tune_gemm_wasp --check
  python -m boltops.tools.tune_gemm_wasp --sweep --shape 512cube --mode wdra48
  python -m boltops.tools.tune_gemm_wasp --mnk 512x512x512 --ab-layout TN --check
  python -m boltops.tools.tune_gemm_wasp --mnk 4096x4608x4096 --mode wdra48 \\
      --blocks 256x256x32 --num-stages 4 --skip-nowasp --ab-layout TN
  python -m boltops.tools.tune_gemm_wasp --promote gemm_wasp_tune.csv --notes "..."

AB layout (``--ab-layout``):
  TT — A K-major + B N-major (default contiguous ``A[M,K]``, ``B[K,N]``)
  TN — A K-major + B K-major (``B`` as ``[N,K].T`` view, ``stride_bk=1``)

``--num-stages``: WASP/WDRA LDS buffer depth ``2`` (default) or ``4``
(``wdra48`` / ``wasp44`` only; ``wdra88`` TwoPTwoC cannot use 4).
Comma / ``both`` sweeps ``2,4``.

``--mmac-layout-force``: HIP ``mmac_layout_force`` ``-1`` (default / auto) or
``0..4`` (mfma / legacy / interleave / transpose / interleave+transpose).
Comma / ``all`` sweeps.

Consumer ping-pong is managed by Triton. ``--consumer-pingpong true`` is
accepted for existing command lines; disabling or sweeping it is unsupported.

``--matrix-store``: use ``tl.matrix_store`` for C (requires ``--mls`` and
``triton-mls-store``; prefer ``--mmac-layout-force 3``).

CSV defaults to ``./gemm_wasp_tune.csv`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import triton

# ---------------------------------------------------------------------------
# Paths / anchors
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = (
    _REPO_ROOT / "boltops" / "gemm" / "triton" / "configs" / "gemm_fp16"
)
_TRITON_ROOT = Path(triton.__file__).resolve().parents[2]
# Measure and clean only this invocation's output, never another checkout.
_M5OUT_ROOTS = (Path.cwd() / "m5out",)

# (name, M, N, K)
SHAPES = (
    ("512cube", 512, 512, 512),
    ("1024cube", 1024, 1024, 1024),
    ("128_128_4096", 128, 128, 4096),
    ("128_128_512", 128, 128, 512),
)

_LAUNCH_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "WARP_SPECIALIZE",
    "wasp_partition_warps",
    "wasp_wdra",
    "wasp_partition_regs",
    "empty_arrive_after_mmac",
    "enable_v_mmac_cluster",
    "mmac_layout_force",
    "num_warps",
    "num_stages",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _git_sha(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _tops(m: int, n: int, k: int, cycles: int, freq_ghz: float) -> float:
    return (2.0 * m * n * k) / (cycles / (freq_ghz * 1e9)) / 1e12


def _iter_stats() -> list[Tuple[Path, int, Optional[float], int]]:
    """Return ``(path, cycles, freq_ghz, mtime_ns)`` for every stats.txt."""
    out: list[Tuple[Path, int, Optional[float], int]] = []
    for root in _M5OUT_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("stats.txt"):
            text = p.read_text(errors="replace")
            m = re.search(r"^totalGcSimCycles\s+(\d+)", text, re.M)
            f = re.search(r"^gcFrequency\s+([0-9.eE+-]+)", text, re.M)
            if not m:
                continue
            try:
                mtime_ns = p.stat().st_mtime_ns
            except OSError:
                continue
            out.append(
                (p, int(m.group(1)), float(f.group(1)) if f else None, mtime_ns)
            )
    return out


def _stats_snapshot() -> dict[str, Tuple[int, int]]:
    """Map stats path → ``(mtime_ns, cycles)``."""
    return {str(p): (mtime_ns, cyc) for p, cyc, _, mtime_ns in _iter_stats()}


def _packet_index(path: Path) -> int:
    """Best-effort ``m5out/<xcd>/<packet>/stats.txt`` packet id; else -1."""
    try:
        return int(path.parent.name)
    except ValueError:
        return -1


def _best_stats_since(
    before: dict[str, Tuple[int, int]],
) -> Tuple[Optional[Path], Optional[int], Optional[float]]:
    """Attribute cycles to the kernel that just ran.

    PMD keeps one process across the sweep, so ``m5out`` must not be wiped
    mid-session. New dispatches usually appear as ``m5out/0/N/stats.txt``, but
    ``totalGcSimCycles`` in that file is often **cumulative from process
    start** (observed: ``0/1`` cycles == ``0/0`` + this kernel). Using the raw
    value for the second kernel under-reports TOPS (harmonic with the first).

    Policy:
    1. Prefer a newly created packet path; else an updated path / newest mtime.
    2. Let ``base`` be the largest prior cycle count in the same XCD output directory (0 if empty).
    3. If picked ``cycles > base``, attribute ``cycles - base`` (cumulative).
       Else attribute ``cycles`` as-is (true per-packet stats).
    """
    newest = None  # (pkt, mtime_ns, cycles, freq, path)
    best_new = None
    best_updated = None

    for p, cyc, freq, mtime_ns in _iter_stats():
        key = str(p)
        pkt = _packet_index(p)
        if newest is None or (pkt, mtime_ns) > (newest[0], newest[1]):
            newest = (pkt, mtime_ns, cyc, freq, p)

        prev = before.get(key)
        if prev is None:
            cand = (pkt, mtime_ns, cyc, freq, p)
            if best_new is None or cand[:2] > best_new[:2]:
                best_new = cand
            continue

        prev_mtime, _prev_cyc = prev
        if mtime_ns <= prev_mtime:
            continue
        cand = (pkt, mtime_ns, cyc, freq, p)
        if best_updated is None or cand[:2] > best_updated[:2]:
            best_updated = cand

    pick = best_new or best_updated or newest
    if pick is None:
        return None, None, None
    _, _, cyc, freq, path = pick
    max_before = max(
        (old_cyc for old_path, (_, old_cyc) in before.items()
         if Path(old_path).parent.parent == path.parent.parent),
        default=0,
    )
    if cyc > max_before:
        attributed = cyc - max_before
    else:
        attributed = cyc
    return path, attributed, freq


def _wipe_m5out() -> None:
    """Only safe before the first kernel in a process (cold PMD start)."""
    for root in _M5OUT_ROOTS:
        if root.exists():
            shutil.rmtree(root)
    (Path.cwd() / "m5out").mkdir(parents=True, exist_ok=True)


def _parse_bool_choice(s: str) -> List[bool]:
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [False, True]
    if s in ("1", "true", "yes", "on"):
        return [True]
    if s in ("0", "false", "no", "off"):
        return [False]
    raise argparse.ArgumentTypeError(f"expected true/false/both, got {s!r}")


def _parse_stages_choice(s: str) -> List[int]:
    """Parse ``--num-stages``: ``2`` | ``4`` | ``2,4`` | ``both``."""
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [2, 4]
    out: List[int] = []
    for tok in s.split(","):
        if not tok.strip():
            continue
        v = int(tok)
        if v not in (2, 4):
            raise argparse.ArgumentTypeError(
                f"num_stages must be 2 or 4; got {v}"
            )
        if v not in out:
            out.append(v)
    if not out:
        raise argparse.ArgumentTypeError(f"empty --num-stages {s!r}")
    return out


def _parse_cluster_choice(s: str) -> List[int]:
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [0, 1, 2]
    if s in ("false", "no", "off"):
        return [0]
    if s in ("true", "yes", "on"):
        return [1]
    out = []
    for tok in s.split(","):
        if not tok.strip():
            continue
        v = int(tok)
        if v not in (0, 1, 2):
            raise argparse.ArgumentTypeError(f"cluster must be 0/1/2; got {v}")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError(f"expected 0|1|2|all, got {s!r}")
    return out


def _parse_mmac_layout_choice(s: str) -> List[int]:
    """Parse ``--mmac-layout-force``: ``-1`` | ``0..4`` | comma list | ``all``."""
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [-1, 0, 1, 2, 3, 4]
    if s in ("auto", "default"):
        return [-1]
    out: List[int] = []
    for tok in s.split(","):
        if not tok.strip():
            continue
        v = int(tok)
        if v < -1 or v > 4:
            raise argparse.ArgumentTypeError(
                f"mmac_layout_force must be -1 or 0..4; got {v}"
            )
        if v not in out:
            out.append(v)
    if not out:
        raise argparse.ArgumentTypeError(
            f"expected -1|0..4|all, got {s!r}"
        )
    return out


def _parse_blocks(spec: str) -> list[tuple[int, int, int]]:
    out = []
    for tok in spec.split(","):
        t = tok.strip().lower().replace("*", "x")
        if not t:
            continue
        a, b, c = t.split("x")
        out.append((int(a), int(b), int(c)))
    if not out:
        raise ValueError(f"empty --blocks: {spec!r}")
    return out


def _resolve_shapes(names: Optional[list[str]], mnk: Optional[str]):
    if mnk:
        raw = mnk.strip().lower().replace("x", ",")
        parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(f"bad --mnk {mnk!r}")
        M, N, K = parts
        return [(f"M{M}_N{N}_K{K}", M, N, K)]
    if names:
        by = {n: (n, M, N, K) for n, M, N, K in SHAPES}
        try:
            return [by[n] for n in names]
        except KeyError as e:
            known = ", ".join(n for n, *_ in SHAPES)
            raise SystemExit(f"unknown shape {e}; known: {known}") from e
    n, M, N, K = SHAPES[0]
    return [(n, M, N, K)]


def _default_blocks(mode: str) -> list[tuple[int, int, int]]:
    if mode.startswith("wdra"):
        return [(64, 64, 64), (128, 128, 64), (64, 64, 128)]
    return [(128, 128, 128), (128, 128, 64), (64, 64, 64)]


def _save_winner(cfg: dict[str, Any]) -> Path:
    arch = str(cfg.get("arch", "gfx946"))
    layout = str(cfg.get("ab_layout") or "TT")
    path = (
        _CONFIG_DIR
        / arch
        / f"M{cfg['M']}_N{cfg['N']}_K{cfg['K']}_{layout}_{cfg['mode']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered: dict[str, Any] = {}
    for k in ("arch", "mode", "M", "N", "K", "ab_layout", "use_mls"):
        if k in cfg:
            ordered[k] = cfg[k]
    for k in _LAUNCH_KEYS:
        if k in cfg:
            ordered[k] = cfg[k]
    for k, v in cfg.items():
        if k not in ordered:
            ordered[k] = v
    path.write_text(json.dumps(ordered, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# Kernel is fp16 in / fp32 acc / fp16 out. Compare against fp32 matmul rounded
# to fp16. torch.assert_close: |Δ| <= atol + rtol*|ref|.
#
# 1e-2 is ~10 fp16 ulps at |C|~8. 5e-4 fails a single store-rounding ulp
# (0.0078125, seen on 2–4 elems of 128-wide tiles). 1e-3 covers 1 ulp
# across typical |C| bins (threshold 0.009 at |C|=8, ~1 ulp at |C|~64)
# while still catching a dropped K-tile.
GOLDEN_ATOL = 1e-3
GOLDEN_RTOL = 1e-3


def fp16_gemm_golden(a: Any, b: Any) -> Any:
    """FP32-accumulate reference, rounded to fp16 to match kernel store."""
    import torch

    return torch.matmul(a.detach().cpu().float(), b.detach().cpu().float()).to(
        torch.float16
    )


def assert_fp16_gemm_close(
    out: Any,
    a: Any,
    b: Any,
    *,
    atol: float = GOLDEN_ATOL,
    rtol: float = GOLDEN_RTOL,
) -> tuple[float, float]:
    """``assert_close`` vs :func:`fp16_gemm_golden`. Returns ``(max_abs, max_rel)``."""
    import torch

    out_cpu = out.detach().cpu()
    ref = fp16_gemm_golden(a, b)
    torch.testing.assert_close(out_cpu, ref, atol=atol, rtol=rtol)
    diff = (out_cpu.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-8)
    return float(diff.max().item()), float(rel.max().item())


def _normalize_ab_layout(layout: str) -> str:
    tag = (layout or "TT").strip().upper()
    if tag not in ("TT", "TN"):
        raise SystemExit(f"unknown --ab-layout {layout!r}; choose TT or TN")
    return tag


def _make_ab(
    M: int,
    N: int,
    K: int,
    ab_layout: str,
    *,
    device: str,
    dtype: Any,
) -> Tuple[Any, Any]:
    """Build A ``[M,K]`` and B ``[K,N]`` for the given AB layout tag.

    TT: contiguous A (K-major), contiguous B (N-major).
    TN: contiguous A (K-major); B is a ``[N,K].T`` view (K-major, ``stride_bk=1``).
    """
    import torch

    a = torch.randn((M, K), device=device, dtype=dtype)
    if ab_layout == "TT":
        b = torch.randn((K, N), device=device, dtype=dtype)
    else:  # TN
        b = torch.randn((N, K), device=device, dtype=dtype).T
    return a, b


def _run_one(
    *,
    gemm_fp16,
    mode_launch_kwargs,
    mode: str,
    M: int,
    N: int,
    K: int,
    bm: int,
    bn: int,
    bk: int,
    empty_after: bool,
    cluster: int,
    use_mls: bool,
    use_matrix_store: bool = False,
    matrix_store_interleave: int = 0,
    check: bool,
    ab_layout: str = "TT",
    num_stages: int = 2,
    mmac_layout_force: int = -1,
    enable_consumer_pingpong: bool = True,
    wdra_num_load_regs: Optional[int] = None,
    wdra_num_mma_regs_main: Optional[int] = None,
    wdra_num_mma_regs_tail: Optional[int] = None,
) -> dict[str, Any]:
    import torch

    if not enable_consumer_pingpong:
        raise ValueError("Triton controls consumer ping-pong; disabling it is unsupported")
    ab_layout = _normalize_ab_layout(ab_layout)
    cfg = mode_launch_kwargs(
        mode,
        empty_arrive_after_mmac=empty_after,
        enable_v_mmac_cluster=cluster,
        num_stages=num_stages,
        mmac_layout_force=mmac_layout_force,
    )
    cfg.update(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk)
    if mode.startswith("wdra") and wdra_num_load_regs is not None:
        cfg["wasp_partition_regs"] = (
            (wdra_num_load_regs,) * (2 if mode == "wdra88" else 1)
            + (wdra_num_mma_regs_main, wdra_num_mma_regs_tail)
        )
    # OPTIMIZE_EPILOGUE is a compiler option, not an environment-controlled
    # knob.  Pass it explicitly so the C-store scratch can reuse the pipelined
    # A/B LDS allocation.
    cfg["optimize_epilogue"] = True
    if use_matrix_store and matrix_store_interleave:
        os.environ["TRITON_MLS_B_INTERLEAVE"] = str(matrix_store_interleave)
    else:
        os.environ.pop("TRITON_MLS_B_INTERLEAVE", None)

    torch.manual_seed(0)
    a_cpu, b_cpu = _make_ab(M, N, K, ab_layout, device="cpu", dtype=torch.float16)
    a, b = a_cpu.cuda(), b_cpu.cuda()

    # Do not wipe m5out here: PMD is still alive and needs m5out/0/1, 0/2, ...
    before = _stats_snapshot()
    t0 = time.time()
    out = gemm_fp16(
        a,
        b,
        wasp=bool(cfg["WARP_SPECIALIZE"]),
        wdra=bool(cfg["wasp_wdra"]),
        use_mls=use_mls,
        use_matrix_store=use_matrix_store,
        wasp_num_load_warps=8 if mode == "wdra88" else 4,
        wasp_num_mma_warps=4 if mode in ("wasp44", "nowasp") else 8,
        autotune=False,
        config=cfg,
    )
    torch.cuda.synchronize()
    wall_s = time.time() - t0

    golden = "SKIP"
    if check:
        max_abs, max_rel = assert_fp16_gemm_close(out, a_cpu, b_cpu)
        golden = "PASS"
        print(
            f"GOLDEN\tmax_abs={max_abs:.6g}\tmax_rel={max_rel:.6g}\t"
            f"atol={GOLDEN_ATOL}\trtol={GOLDEN_RTOL}",
            flush=True,
        )

    stats_path, cycles, freq = _best_stats_since(before)
    p = (
        _tops(M, N, K, cycles, freq)
        if cycles is not None and freq is not None
        else None
    )
    return {
        "mode": mode,
        "M": M,
        "N": N,
        "K": K,
        "ab_layout": ab_layout,
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "num_stages": int(cfg.get("num_stages", num_stages)),
        "empty_arrive_after_mmac": int(empty_after),
        "enable_v_mmac_cluster": int(cluster),
        "enable_consumer_pingpong": int(mode in ("wdra48", "wdra88")),
        "wasp_partition_warps": cfg.get("wasp_partition_warps"),
        "wasp_wdra": cfg["wasp_wdra"],
        "wasp_partition_regs": cfg.get("wasp_partition_regs"),
        "mmac_layout_force": int(cfg.get("mmac_layout_force", mmac_layout_force)),
        "use_mls": int(use_mls),
        "use_matrix_store": int(use_matrix_store),
        "wasp_num_load_warps": (8 if mode == "wdra88" else 4) if cfg["WARP_SPECIALIZE"] else None,
        "wasp_num_mma_warps": (4 if mode == "wasp44" else 8) if cfg["WARP_SPECIALIZE"] else None,
        "wdra_num_load_regs": (cfg.get("wasp_partition_regs") or (None,))[0],
        "wdra_num_mma_regs_main": (cfg.get("wasp_partition_regs") or (None, None))[-2],
        "wdra_num_mma_regs_tail": (cfg.get("wasp_partition_regs") or (None,))[-1],
        "cycles": cycles,
        "gc_frequency_ghz": freq,
        "tops": p,
        "wall_s": round(wall_s, 3),
        "golden": golden,
        "stats": str(stats_path) if stats_path else "",
    }


def _promote(csv_path: Path, notes: str) -> Path:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {csv_path}")

    def row_tops(r: dict[str, Any]) -> float:
        try:
            return float(r["tops"]) if r.get("tops") not in (None, "") else -1.0
        except (TypeError, ValueError):
            return -1.0

    cand = [
        r
        for r in rows
        if r.get("mode") not in (None, "", "nowasp", "nowrap")
    ] or rows
    best = max(cand, key=row_tops)
    baseline = next(
        (r for r in rows if r.get("mode") in ("nowasp", "nowrap")), None
    )

    cfg: dict[str, Any] = {
        "arch": "gfx946",
        "mode": "nowasp" if best["mode"] == "nowrap" else best["mode"],
        "M": int(best["M"]),
        "N": int(best["N"]),
        "K": int(best["K"]),
        # TT: A K-major + B N-major; TN: A K-major + B K-major.
        # See docs/gemm/RESULTS.md / --ab-layout.
        "ab_layout": _normalize_ab_layout(str(best.get("ab_layout") or "TT")),
        "use_mls": bool(int(best.get("use_mls") or 0)),
        "use_matrix_store": bool(int(best.get("use_matrix_store") or 0)),
        "BLOCK_SIZE_M": int(best["BLOCK_SIZE_M"]),
        "BLOCK_SIZE_N": int(best["BLOCK_SIZE_N"]),
        "BLOCK_SIZE_K": int(best["BLOCK_SIZE_K"]),
        "num_stages": int(best["num_stages"]) if best.get("num_stages") not in (None, "") else 2,
        "GROUP_SIZE_M": 1,
        "WARP_SPECIALIZE": best["mode"] not in ("nowasp", "nowrap"),
        "wasp_wdra": str(best["mode"]).startswith("wdra"),
        "empty_arrive_after_mmac": bool(int(best["empty_arrive_after_mmac"])),
        "enable_v_mmac_cluster": int(best["enable_v_mmac_cluster"]),
        "mmac_layout_force": int(
            best["mmac_layout_force"]
            if best.get("mmac_layout_force") not in (None, "")
            else -1
        ),
        "baseline_mode": "nowasp",
        "baseline_tops": (
            float(baseline["tops"])
            if baseline and baseline.get("tops") not in (None, "")
            else None
        ),
        "winner_tops": row_tops(best) if row_tops(best) >= 0 else None,
        "cycles": int(best["cycles"]) if best.get("cycles") not in (None, "") else None,
        "gc_frequency_ghz": (
            float(best["gc_frequency_ghz"])
            if best.get("gc_frequency_ghz") not in (None, "")
            else None
        ),
        "triton_commit": _git_sha(_TRITON_ROOT),
        "bolt_ops_commit": _git_sha(_REPO_ROOT),
        "notes": notes or f"promoted from {csv_path.name}",
    }
    from boltops.gemm.triton.gemm_fp16 import mode_launch_kwargs
    mode_cfg = mode_launch_kwargs(cfg["mode"])
    if not best.get("wasp_partition_warps") and best.get("enable_consumer_pingpong") not in (None, ""):
        cfg["historical_enable_consumer_pingpong"] = bool(int(best["enable_consumer_pingpong"]))
    for key in ("wasp_partition_warps", "wasp_wdra", "wasp_partition_regs"):
        if key in mode_cfg:
            cfg[key] = mode_cfg[key]
    if cfg["wasp_wdra"]:
        load = int(best.get("wdra_num_load_regs") or 160)
        main = int(best.get("wdra_num_mma_regs_main") or 160)
        tail = int(best.get("wdra_num_mma_regs_tail") or 160)
        cfg["wasp_partition_regs"] = (
            (load,) * (2 if cfg["mode"] == "wdra88" else 1) + (main, tail)
        )

    path = _save_winner(cfg)
    print(f"promoted {path}", flush=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list-shapes", action="store_true")
    ap.add_argument("--shape", action="append", default=None)
    ap.add_argument("--mnk", default=None, help="M,N,K or MxNxK")
    ap.add_argument("--mode", default="wdra48", help="target mode (default wdra48)")
    ap.add_argument("--modes", default=None, help="comma list; skips auto nowasp")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--blocks", default=None, help="BMxBNxBK,... ")
    ap.add_argument("--empty-arrive", default="false", help="true|false|both")
    ap.add_argument("--cluster", default="0", help="0|1|2|all")
    ap.add_argument(
        "--consumer-pingpong",
        default="true",
        choices=("true",),
        help="Compatibility option; Triton controls consumer ping-pong (true only)",
    )
    ap.add_argument(
        "--num-stages",
        default="2",
        help="WASP/WDRA LDS depth: 2|4|2,4|both (default 2; wdra88 forbids 4)",
    )
    ap.add_argument(
        "--mmac-layout-force",
        default="-1",
        help="HIP mmac_layout_force: -1|0..4|comma|all (default -1=auto)",
    )
    ap.add_argument("--mls", action="store_true")
    ap.add_argument("--wdra-load-regs", type=int, default=None)
    ap.add_argument("--wdra-mma-main-regs", type=int, default=None)
    ap.add_argument("--wdra-mma-tail-regs", type=int, default=None)
    ap.add_argument(
        "--matrix-store",
        action="store_true",
        help="use tl.matrix_store for C (requires --mls; triton-mls-store)",
    )
    ap.add_argument(
        "--matrix-store-interleave",
        type=int,
        choices=(0, 2),
        default=2,
        help="B MLS interleave factor for matrix-store packing: 0|2 (default 2)",
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--ab-layout",
        default="TT",
        choices=("TT", "TN", "tt", "tn"),
        help="A/B major-order tag: TT (default) or TN (B K-major)",
    )
    ap.add_argument(
        "--no-wipe",
        action="store_true",
        help="Do not wipe m5out at sweep start",
    )
    ap.add_argument("--skip-nowasp", action="store_true")
    ap.add_argument("--out", default="gemm_wasp_tune.csv", help="CSV output path")
    ap.add_argument(
        "--reference-tops",
        type=float,
        default=None,
        help="Fail when the best measured TOPS regresses beyond the allowed percentage",
    )
    ap.add_argument(
        "--max-tops-regression-pct",
        type=float,
        default=2.0,
        help="Maximum TOPS regression from --reference-tops (default 2%%)",
    )
    ap.add_argument("--promote", default=None, help="promote best row from CSV")
    ap.add_argument("--notes", default="", help="notes for --promote")
    args = ap.parse_args(argv)

    if args.list_shapes:
        for name, M, N, K in SHAPES:
            print(f"{name}\t{M}x{N}x{K}")
        return 0

    if args.promote:
        _promote(Path(args.promote), args.notes)
        return 0

    if args.matrix_store and not args.mls:
        raise SystemExit("--matrix-store requires --mls")
    reg_args = (args.wdra_load_regs, args.wdra_mma_main_regs, args.wdra_mma_tail_regs)
    if any(x is not None for x in reg_args):
        if not all(x is not None for x in reg_args):
            raise SystemExit("all three WDRA register budgets must be specified together")
        if any(x <= 0 or x > 256 or x % 4 for x in reg_args):
            raise SystemExit("WDRA register budgets must be multiples of 4 in [4, 256]")
    if args.reference_tops is not None and args.reference_tops <= 0:
        raise SystemExit("--reference-tops must be positive")
    if args.max_tops_regression_pct < 0 or args.max_tops_regression_pct >= 100:
        raise SystemExit("--max-tops-regression-pct must be in [0, 100)")

    os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
    # Wipe only once before PMD starts. Mid-sweep rmtree makes gem5 fatal on
    # './m5out/0/1/' while the model process is still alive.
    if not args.no_wipe:
        _wipe_m5out()

    from boltops.gemm.triton.gemm_fp16 import MODES, gemm_fp16, mode_launch_kwargs

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        modes = [args.mode]
        if not args.skip_nowasp and "nowasp" not in modes:
            modes = ["nowasp", args.mode]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {MODES}")
        if reg_args[0] is not None and m.startswith("wdra"):
            quotas = (reg_args[0],) * (2 if m == "wdra88" else 1) + reg_args[1:]
            if sum(quotas) > 512 or sum(quotas) % (4 * len(quotas)):
                raise SystemExit(
                    f"{m} partition quotas must total <=512 and be divisible by {4 * len(quotas)}"
                )

    shapes = _resolve_shapes(args.shape, args.mnk)
    ab_layout = _normalize_ab_layout(args.ab_layout)

    empty_arg, cluster_arg = args.empty_arrive, args.cluster
    if args.sweep:
        if empty_arg == "false" and "--empty-arrive" not in sys.argv:
            empty_arg = "both"
        if cluster_arg == "0" and "--cluster" not in sys.argv:
            cluster_arg = "all"
    empty_choices = _parse_bool_choice(empty_arg)
    cluster_choices = _parse_cluster_choice(cluster_arg)
    pingpong_choices = _parse_bool_choice(args.consumer_pingpong)
    stage_choices = _parse_stages_choice(args.num_stages)
    layout_choices = _parse_mmac_layout_choice(args.mmac_layout_force)

    rows: list[dict[str, Any]] = []
    for label, M, N, K in shapes:
        for mode in modes:
            blocks = (
                _parse_blocks(args.blocks)
                if args.blocks
                else (
                    _default_blocks(mode)
                    if args.sweep
                    else [_default_blocks(mode)[0]]
                )
            )
            grid = list(
                itertools.product(
                    blocks,
                    empty_choices,
                    cluster_choices,
                    pingpong_choices,
                    stage_choices,
                    layout_choices,
                )
            )
            print(
                f"=== {label} {M}x{N}x{K} ab={ab_layout} mode={mode} "
                f"stages={stage_choices} mmac_layout={layout_choices} "
                f"pingpong={pingpong_choices} combos={len(grid)} ===",
                flush=True,
            )
            for (bm, bn, bk), ea, cl, pp, ns, mlf in grid:
                row = _run_one(
                    gemm_fp16=gemm_fp16,
                    mode_launch_kwargs=mode_launch_kwargs,
                    mode=mode,
                    M=M,
                    N=N,
                    K=K,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    empty_after=ea,
                    cluster=cl,
                    use_mls=args.mls,
                    use_matrix_store=args.matrix_store,
                    matrix_store_interleave=args.matrix_store_interleave,
                    check=args.check,
                    ab_layout=ab_layout,
                    num_stages=ns,
                    mmac_layout_force=mlf,
                    enable_consumer_pingpong=pp,
                    wdra_num_load_regs=args.wdra_load_regs,
                    wdra_num_mma_regs_main=args.wdra_mma_main_regs,
                    wdra_num_mma_regs_tail=args.wdra_mma_tail_regs,
                )
                row["shape"] = label
                rows.append(row)
                print(
                    f"RESULT\t{mode}\t{ab_layout}\t{bm}x{bn}x{bk}\t"
                    f"ns={ns}\tea={int(ea)}\tcl={cl}\tpp={row['enable_consumer_pingpong']}\tmlf={mlf}\t"
                    f"mls={int(args.mls)}\tmstore={int(args.matrix_store)}\t"
                    f"TOPS={row['tops']}\tcycles={row['cycles']}\t"
                    f"wall_s={row['wall_s']}\tgolden={row['golden']}",
                    flush=True,
                )

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"=== wrote {out} ({len(rows)} rows) ===", flush=True)

    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["tops"] is None:
            continue
        cur = best.get(r["mode"])
        if cur is None or r["tops"] > cur["tops"]:
            best[r["mode"]] = r
    print("=== SUMMARY ===", flush=True)
    for mode, r in best.items():
        print(
            f"{mode}\tab={r.get('ab_layout', ab_layout)}\t"
            f"TOPS={r['tops']:.3f}\t"
            f"tile={r['BLOCK_SIZE_M']}x{r['BLOCK_SIZE_N']}x{r['BLOCK_SIZE_K']}\t"
            f"ns={r.get('num_stages', 2)}\t"
            f"ea={r['empty_arrive_after_mmac']}\tcl={r['enable_v_mmac_cluster']}\t"
            f"pp={r.get('enable_consumer_pingpong', 1)}\t"
            f"mlf={r.get('mmac_layout_force', -1)}\t"
            f"mls={r.get('use_mls', 0)}\tmstore={r.get('use_matrix_store', 0)}",
            flush=True,
        )
    if "nowasp" in best and args.mode in best:
        print(
            f"SPEEDUP\t{args.mode}/nowasp = "
            f"{best[args.mode]['tops'] / best['nowasp']['tops']:.3f}x",
            flush=True,
        )
    if args.reference_tops is not None:
        measured = max((r["tops"] for r in rows if r["tops"] is not None),
                       default=None)
        if measured is None:
            raise SystemExit("PERF FAIL: no TOPS measurement was produced")
        minimum = args.reference_tops * (
            1.0 - args.max_tops_regression_pct / 100.0
        )
        status = "PASS" if measured >= minimum else "FAIL"
        print(
            f"PERF {status}\tTOPS={measured:.6f}\treference={args.reference_tops:.6f}\t"
            f"minimum={minimum:.6f}\tmax_regression_pct={args.max_tops_regression_pct:.3f}",
            flush=True,
        )
        if measured < minimum:
            raise SystemExit(
                f"performance regression: {measured:.6f} TOPS is below "
                f"{minimum:.6f} TOPS"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
