# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Official half-precision (bf16/fp16) MQA-logits test grid — 66 shapes.

The canonical shape set is the union of two production-derived groups:

* **54-shape tune grid** — the prefill grid used by
  ``boltops/mqa_logits/triton/tune_mqa_logits_half.py``:
  ``M in {16, 32, 64, 127} x H in {8, 64} x N in {66, 4096, 8192, 16384,
  32768, 65536, 131072}`` with ``M <= N``.
* **12-shape "4608 group"** — production prefill shapes covering
  ``M in {256, 1024, 2048, 4608} x H in {4, 8, 16}``.

Total: ``4x7x2 - 2 + 12 = 66`` shapes, exported as :data:`HALF_SIZE_GRID`.

Concerns covered (half only; FP8 lives in ``test_triton_fp8_mqa_logits.py``):

* **API / backends** — explicit ``row``/``bn``, gluon rejection, optional
  ``gluon_bn_align``, and ``kv_scales`` must not take the fp8 path.
* **correctness** (default pytest run) — vs a pure-torch reference, on a
  compact representative subset, for both dtypes and both ``ks``/``ke``
  window conventions.
* **performance** (``pytest --bench``) — BoltOPs-only ``time_ms`` on all 66
  shapes. Hipc comparison is opt-in (``--bench -k vs_hipc``): two ``time_ms``
  calls and a regression guard (``hipc/ours >= 0.85``). Prefer the sharded
  form below: one process per shard keeps timings isolated.

``ks``/``ke`` conventions
-------------------------
* ``"full"``    — ``ks = 0``, ``ke = N`` for every row (uniform full window).
  This is what the production tuning harness and the shipped
  ``bench_half_*`` tuning JSONs were measured with, so it is the default
  for the benchmark tests.
* ``"sliding"`` — ``ks = 0``, ``ke = i + (N - M)`` (triangular window, the
  convention used by ``tune_mqa_logits_half.py``). Correctness covers this too.

Naming
------
The reference library is imported as ``lightop`` (that is its package name)
but is called **hipc** everywhere else — display strings, identifiers and
prose (same convention as ``test_triton_fp8_mqa_logits.py`` / paged benches).
So ``lightop = pytest.importorskip("lightop")``, called as
``lightop.mqa_logits(...)`` inside ``run_hipc()``, is intentional: the
module binding keeps the real package name, only the display name is hipc.

Usage
-----
::

    # correctness only (fast, default)
    pytest test/mqa_logits/test_triton_mqa_logits_half.py

    # BoltOPs-only latency on all 66 shapes
    pytest --bench test/mqa_logits/test_triton_mqa_logits_half.py

    # optional hipc comparison (not part of a plain --bench / optest per run)
    pytest --bench -k vs_hipc test/mqa_logits/test_triton_mqa_logits_half.py

    # a single shape
    pytest --bench -k "4608x16384-H4"

    # local table + JSON (not an optest bench; no --bench)
    HALF_GRID_REPORT=1 HALF_GRID_JSON=/tmp/mqa_logits_half_66.json \
        pytest -s -k report

Set ``HALF_GRID_SAMPLES`` to override the robust-timing repeat count.

On a multi-GPU node, shard the grid across devices for a parallel sweep::

    for i in 0 1 2 3 4 5 6 7; do
      HIP_VISIBLE_DEVICES=$i HALF_GRID_SHARD="$i/8" python3 -u -m pytest \\
        --bench -q -s -k vs_hipc \\
        test/mqa_logits/test_triton_mqa_logits_half.py \\
        > /tmp/half_shard$i.log 2>&1 &
    done; wait

Note the ``python3 -u``: without unbuffered stdout, pytest discards the
captured ``[bench half grid]`` lines of *passing* cases and only the summary
remains.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Iterator, List, Sequence, Tuple

import pytest
import torch

pytest.importorskip("triton")

import triton  # noqa: E402

from boltops.mqa_logits import triton_mqa_logits  # noqa: E402

# optest CSV name. File stem would be ``triton_mqa_logits_half``.
PERF_OP = "half_mqa_logits"

DEVICE = "cuda"
HEAD_DIM = 128
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

# --------------------------------------------------------------------------- #
# Canonical 66-shape grid
# --------------------------------------------------------------------------- #

_TUNE_GRID_M = (16, 32, 64, 127)
_TUNE_GRID_N = (66, 4096, 8192, 16384, 32768, 65536, 131072)
_TUNE_GRID_H = (8, 64)

# Real prefill shapes from the production tuning harness ("4608 group").
_GROUP_4608 = (
    (4608, 16384, 4),
    (4608, 32768, 4),
    (4608, 65536, 4),
    (2048, 32768, 4),
    (1024, 16384, 4),
    (256, 16384, 4),
    (4608, 16384, 8),
    (4608, 32768, 8),
    (1024, 32768, 8),
    (256, 65536, 8),
    (4608, 16384, 16),
    (1024, 32768, 16),
)


def _tune_grid() -> Iterator[Tuple[int, int, int]]:
    for s_q in _TUNE_GRID_M:
        for s_k in _TUNE_GRID_N:
            if s_q > s_k:
                continue
            for num_heads in _TUNE_GRID_H:
                yield (s_q, s_k, num_heads)


HALF_SIZE_GRID: Tuple[Tuple[int, int, int], ...] = (
    tuple(_tune_grid()) + _GROUP_4608
)
"""The official 66 ``(seq_len, seq_len_kv, num_heads)`` half shapes."""

assert len(HALF_SIZE_GRID) == 66, f"expected 66 shapes, got {len(HALF_SIZE_GRID)}"
assert len(set(HALF_SIZE_GRID)) == 66, "HALF_SIZE_GRID has duplicates"

# Correctness: compact coverage of every family (tiny / small-M / large-M,
# H=4/8/16/64, short and long N), both window conventions.
_CORRECTNESS_SHAPES = (
    pytest.param(16, 66, 8, id="tiny-H8"),
    pytest.param(32, 66, 64, id="tiny-H64"),
    pytest.param(64, 8192, 64, id="small-M-long"),
    pytest.param(127, 16384, 64, id="tall-short"),
    pytest.param(256, 16384, 4, id="h4-small-M"),
    pytest.param(1024, 16384, 4, id="h4-mid"),
    pytest.param(1024, 32768, 16, id="h16-mid"),
    pytest.param(4608, 32768, 8, id="h8-large-M"),
    pytest.param(4608, 16384, 4, id="target-4608-H4"),
)

_SAMPLES_FAST = 7
_SAMPLES_SLOW = 5
# Regression guard: measured worst case across the 66 shapes is a tie
# (hipc/ours ~= 1.00). Anything below this means a real config/perf
# regression (the N=4096 bucket mis-route showed up as ~0.77).
_MIN_RATIO = 0.85


def _samples(s_q: int, s_k: int) -> int:
    override = os.environ.get("HALF_GRID_SAMPLES")
    if override:
        return max(1, int(override))
    return _SAMPLES_SLOW if s_q * s_k > 3e8 else _SAMPLES_FAST


def _shard(shapes: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Apply ``HALF_GRID_SHARD="<index>/<count>"`` for multi-GPU sweeps."""
    spec = os.environ.get("HALF_GRID_SHARD")
    if not spec:
        return list(shapes)
    index_s, _, count_s = spec.partition("/")
    index, count = int(index_s), int(count_s)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"bad HALF_GRID_SHARD={spec!r}")
    return [s for i, s in enumerate(shapes) if i % count == index]


# --------------------------------------------------------------------------- #
# Inputs / reference
# --------------------------------------------------------------------------- #


def _window(
    s_q: int, s_k: int, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    ks = torch.zeros(s_q, dtype=torch.int32, device=DEVICE)
    if mode == "full":
        ke = torch.full((s_q,), s_k, dtype=torch.int32, device=DEVICE)
    elif mode == "sliding":
        ke = torch.arange(s_q, dtype=torch.int32, device=DEVICE) + (s_k - s_q)
    else:  # pragma: no cover - guarded by parametrize
        raise ValueError(f"unknown window mode: {mode!r}")
    return ks, ke


def _make_inputs(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    mode: str = "full",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert s_q <= s_k, "M > N is not a valid prefill shape"
    torch.manual_seed(seed)
    q = torch.randn(s_q, num_heads, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv = torch.randn(s_k, HEAD_DIM, device=DEVICE, dtype=dtype)
    weights = torch.randn(s_q, num_heads, device=DEVICE, dtype=torch.float32)
    ks, ke = _window(s_q, s_k, mode)
    return q, kv, weights, ks, ke


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def ref_half_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Pure-torch reference; chunked over M to keep the (H, M, N) score small."""
    s_q, _, _ = q.shape
    s_k = kv.shape[0]
    out = torch.full((s_q, s_k), float("-inf"), device=q.device, dtype=torch.float32)
    positions = torch.arange(s_k, device=q.device)
    for start in range(0, s_q, chunk_rows):
        end = min(s_q, start + chunk_rows)
        score = torch.einsum("mhd,nd->hmn", q[start:end].float(), kv.float())
        weighted = (score.relu() * weights[start:end].unsqueeze(-1).transpose(0, 1))
        row = weighted.sum(dim=0)
        mask = (positions[None, :] >= ks[start:end, None]) & (
            positions[None, :] < ke[start:end, None]
        )
        out[start:end] = row.masked_fill(~mask, float("-inf"))
    return out


def _assert_matches_ref(
    logits: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    atol_diff: float = 2e-2,
) -> None:
    ref = ref_half_mqa_logits(q, kv, weights, ks, ke)
    ref_neginf = ref == float("-inf")
    out_neginf = logits == float("-inf")
    assert torch.equal(out_neginf, ref_neginf), "mask (-inf) pattern mismatch"
    diff = calc_diff(
        logits.masked_fill(out_neginf, 0), ref.masked_fill(ref_neginf, 0)
    )
    assert diff < atol_diff, f"{diff=}"


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["full", "sliding"])
@pytest.mark.parametrize("s_q,s_k,num_heads", _CORRECTNESS_SHAPES)
@torch.inference_mode()
def test_triton_mqa_logits_half_correctness(
    dtype: torch.dtype,
    mode: str,
    s_q: int,
    s_k: int,
    num_heads: int,
) -> None:
    """66-grid shapes (representative subset) vs pure-torch, both windows."""
    from boltops.mqa_logits.triton.mqa_logits_half import get_half_bn_dispatch

    q, kv, weights, ks, ke = _make_inputs(s_q, s_k, num_heads, dtype=dtype, mode=mode)
    logits = triton_mqa_logits(q, kv, None, weights, ks, ke)
    assert logits.shape == (s_q, s_k)
    _assert_matches_ref(logits, q, kv, weights, ks, ke)

    disp = get_half_bn_dispatch(s_q, s_k, num_heads)
    assert int(disp["block_m"]) >= 1
    assert int(disp["block_n"]) >= 16
    assert int(disp["head_split_k"]) >= 1


# --------------------------------------------------------------------------- #
# API / backends
# --------------------------------------------------------------------------- #


def test_triton_mqa_logits_half_rejects_kv_scales() -> None:
    q, kv, weights, ks, ke = _make_inputs(16, 66, 8, dtype=torch.bfloat16, mode="sliding")
    scales = torch.ones(66, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="float8 path requires"):
        triton_mqa_logits(q, kv, scales, weights, ks, ke)


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["row", "bn"])
@pytest.mark.parametrize(
    "s_q,s_k,num_heads",
    [
        pytest.param(16, 66, 8, id="tiny-H8"),
        pytest.param(32, 128, 64, id="short"),
        pytest.param(64, 4096, 64, id="mid"),
        pytest.param(64, 16384, 64, id="large-N"),
    ],
)
@torch.inference_mode()
def test_triton_mqa_logits_half_backends(
    dtype: torch.dtype,
    backend: str,
    s_q: int,
    s_k: int,
    num_heads: int,
) -> None:
    """Explicit ``row``/``bn`` paths (``auto`` covered by size-grid correctness)."""
    q, kv, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, dtype=dtype, mode="sliding"
    )
    logits = triton_mqa_logits(q, kv, None, weights, ks, ke, backend=backend)
    _assert_matches_ref(logits, q, kv, weights, ks, ke)


@GPU_REQUIRED
@torch.inference_mode()
def test_triton_mqa_logits_half_rejects_gluon() -> None:
    q, kv, weights, ks, ke = _make_inputs(16, 66, 8, dtype=torch.bfloat16, mode="sliding")
    with pytest.raises(ValueError, match="gluon"):
        triton_mqa_logits(q, kv, None, weights, ks, ke, backend="gluon")


@GPU_REQUIRED
@torch.inference_mode()
def test_triton_mqa_logits_half_gluon_bn_align() -> None:
    """Experimental half Gluon path: one mid shape smoke vs torch ref."""
    from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import gluon_prefill_available

    if not gluon_prefill_available():
        pytest.skip("Gluon unavailable")

    s_q, s_k, num_heads = 64, 4096, 64
    q, kv, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, dtype=torch.bfloat16, mode="sliding"
    )
    logits = triton_mqa_logits(
        q, kv, None, weights, ks, ke, backend="gluon_bn_align"
    )
    _assert_matches_ref(logits, q, kv, weights, ks, ke)


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #


def _robust_ms(fn, samples: int, warmup: int, repeat: int) -> Tuple[float, int]:
    """Median of ``samples`` do_bench runs, dropping node-noise stalls.

    Returns ``(median_ms, n_rejected)``. Shared BW1100 nodes periodically stall
    every kernel (ours *and* hipc) by ~19 ms, and a sustained stall can
    inflate *most* samples of a short run.

    Stalls only ever *add* time, so the minimum sample is used as the rejection
    reference (``1.5x`` min) instead of the median: a median reference is
    itself corrupted when the majority of samples are inflated, which silently
    defeats the filter. The returned value is the median of the accepted
    samples.
    """
    fn()
    torch.cuda.synchronize()
    xs = [
        triton.testing.do_bench(fn, warmup=warmup, rep=repeat, return_mode="median")
        for _ in range(samples)
    ]
    reference = min(xs)
    good = [x for x in xs if x <= 1.5 * reference]
    return statistics.median(good), len(xs) - len(good)


# A measurement is only trustworthy while a minority of its samples were
# rejected as spikes -- otherwise the median itself may be a spike.
def _unreliable(rejected: int, samples: int) -> bool:
    return rejected > max(1, samples // 3)


def _measure(fn, samples: int, warmup: int, repeat: int) -> Tuple[float, int]:
    """``_robust_ms`` with one longer retry when the first pass was noisy."""
    ms, rejected = _robust_ms(fn, samples, warmup, repeat)
    if _unreliable(rejected, samples):
        big = samples * 3
        ms2, rejected2 = _robust_ms(fn, big, warmup, repeat)
        if not _unreliable(rejected2, big):
            return ms2, rejected2
    return ms, rejected


def _shape_id(shape: Sequence[object]) -> str:
    s_q, s_k, num_heads = shape  # type: ignore[misc]
    return f"{s_q}x{s_k}-H{num_heads}"


def _case_ids(shapes: Sequence[object]) -> List[str]:
    return [str(getattr(s, "id", None) or _shape_id(tuple(s))) for s in shapes]


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    "s_q,s_k,num_heads", _shard(HALF_SIZE_GRID), ids=_case_ids(_shard(HALF_SIZE_GRID))
)
@torch.inference_mode()
def test_bench_triton_mqa_logits_half(
    s_q: int,
    s_k: int,
    num_heads: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """BoltOPs-only latency on the official 66-shape grid (no hipc required)."""
    q, kv, weights, ks, ke = _make_inputs(s_q, s_k, num_heads, mode="full")
    out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)

    def run():
        triton_mqa_logits(q, kv, None, weights, ks, ke, D_out=out)

    ms = time_ms(run, bench_warmup, bench_repeat)
    print(
        f"[bench half] {s_q}x{s_k} H={num_heads}: {ms * 1e3:.1f} us",
        flush=True,
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    "s_q,s_k,num_heads", _shard(HALF_SIZE_GRID), ids=_case_ids(_shard(HALF_SIZE_GRID))
)
@torch.inference_mode()
def test_bench_triton_mqa_logits_half_vs_hipc(
    s_q: int,
    s_k: int,
    num_heads: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """All 66 official shapes: ours (auto) vs hipc, full-window ks/ke.

    Two ``time_ms`` calls (ours, then hipc). Skipped when hipc is not
    installed. A ratio under the guard is measured once more before failing.
    """
    lightop = pytest.importorskip("lightop")

    q, kv, weights, ks, ke = _make_inputs(s_q, s_k, num_heads, mode="full")
    out_bo = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    out_lo = torch.empty_like(out_bo)

    def run_ours():
        triton_mqa_logits(q, kv, None, weights, ks, ke, D_out=out_bo)

    def run_hipc():
        lightop.mqa_logits(q, kv, weights, ks, ke, None, clean_logit=True, D_out=out_lo)

    def measure():
        ms_o = time_ms(run_ours, bench_warmup, bench_repeat)
        ms_h = time_ms(run_hipc, bench_warmup, bench_repeat)
        ratio_now = ms_h / ms_o if ms_o > 0 else float("inf")
        return ms_o, ms_h, ratio_now

    ms_ours, ms_hipc, ratio = measure()
    if ratio < _MIN_RATIO:
        ms_ours, ms_hipc, ratio = measure()
    gap_pct = (ms_ours - ms_hipc) / ms_hipc * 100.0

    print(
        f"[bench half grid] {s_q}x{s_k} H={num_heads}: "
        f"ours {ms_ours * 1e3:.1f} us, hipc {ms_hipc * 1e3:.1f} us, "
        f"{gap_pct:+.2f}% (hipc/ours {ratio:.3f}x)",
        flush=True,
    )
    assert ratio >= _MIN_RATIO, (
        f"perf regression on {s_q}x{s_k} H={num_heads}: "
        f"ours {ms_ours * 1e3:.1f} us vs hipc {ms_hipc * 1e3:.1f} us "
        f"({gap_pct:+.2f}%, ratio {ratio:.3f}x < {_MIN_RATIO})"
    )


@GPU_REQUIRED
@torch.inference_mode()
def test_triton_mqa_logits_half_report(
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Emit the full 66-shape table + JSON consumed by the perf docs.

    Local table/JSON only (not an optest bench). Skipped unless
    ``HALF_GRID_REPORT=1``.
    """
    if os.environ.get("HALF_GRID_REPORT") != "1":
        pytest.skip("set HALF_GRID_REPORT=1 to emit the 66-shape report")

    lightop = pytest.importorskip("lightop")

    rows: List[dict] = []
    header = (
        f"{'M':>6} {'N':>7} {'H':>3} | {'ours(us)':>9} {'hipc':>9} "
        f"{'gap%':>8} {'ratio':>7} | {'maxdiff':>9}"
    )
    print("\n" + "=" * len(header))
    print("MQA logits half — official 66-shape grid (full-window ks/ke)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s_q, s_k, num_heads in HALF_SIZE_GRID:
        q, kv, weights, ks, ke = _make_inputs(s_q, s_k, num_heads, mode="full")
        out_bo = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
        out_lo = torch.empty_like(out_bo)

        def run_ours():
            return triton_mqa_logits(q, kv, None, weights, ks, ke, D_out=out_bo)

        def run_hipc():
            return lightop.mqa_logits(
                q, kv, weights, ks, ke, None, clean_logit=True, D_out=out_lo
            )

        max_diff = float("nan")
        try:
            got = run_ours().clone()
            ref = run_hipc().clone()
            finite = torch.isfinite(ref) & torch.isfinite(got)
            if bool(finite.any()):
                max_diff = float((got - ref).abs()[finite].max().item())
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"  {s_q}x{s_k} H={num_heads} correctness probe failed: {exc}")

        samples = _samples(s_q, s_k)
        ms_ours, ours_spikes = _measure(run_ours, samples, bench_warmup, bench_repeat)
        ms_hipc, lo_spikes = _measure(
            run_hipc, samples, bench_warmup, bench_repeat
        )
        gap_pct = (ms_ours - ms_hipc) / ms_hipc * 100.0
        ratio = ms_hipc / ms_ours if ms_ours > 0 else float("inf")
        noisy = _unreliable(ours_spikes, samples) or _unreliable(lo_spikes, samples)

        rows.append(
            {
                "M": s_q,
                "N": s_k,
                "H": num_heads,
                "ours_us": ms_ours * 1e3,
                "hipc_us": ms_hipc * 1e3,
                "gap_pct": gap_pct,
                "ratio": ratio,
                "max_diff": max_diff,
                "spikes_ours": ours_spikes,
                "spikes_hipc": lo_spikes,
                "unreliable": noisy,
            }
        )
        print(
            f"{s_q:>6} {s_k:>7} {num_heads:>3} | {ms_ours * 1e3:>9.1f} "
            f"{ms_hipc * 1e3:>9.1f} {gap_pct:>+7.2f}% {ratio:>6.3f}x | "
            f"{max_diff:>9.2e}{'  NOISY' if noisy else ''}",
            flush=True,
        )
        del q, kv, weights, ks, ke, out_bo, out_lo
        torch.cuda.empty_cache()

    gaps = [r["gap_pct"] for r in rows]
    faster = sum(1 for g in gaps if g < -1.0)
    tied = sum(1 for g in gaps if -1.0 <= g <= 1.0)
    slower = sum(1 for g in gaps if g > 1.0)
    summary = {
        "n": len(rows),
        "faster": faster,
        "tied": tied,
        "slower": slower,
        "mean_gap_pct": statistics.mean(gaps),
        "median_gap_pct": statistics.median(gaps),
        "worst_gap_pct": max(gaps),
        "best_gap_pct": min(gaps),
        "window": "full",
        "rows": rows,
    }
    print("-" * len(header))
    print(
        f"n={summary['n']} faster={faster} tied={tied} slower={slower} "
        f"mean={summary['mean_gap_pct']:+.2f}% "
        f"median={summary['median_gap_pct']:+.2f}% "
        f"worst={summary['worst_gap_pct']:+.2f}%"
    )
    print("=" * len(header))

    # Deliberately no hard assertion here: a long single-process sweep on a
    # shared node can produce false "regressions" from ~19 ms stalls. The
    # parametrized ``..._vs_hipc`` cases above are the regression gate --
    # run them sharded (``HALF_GRID_SHARD``) for isolated, spike-filtered timings.
    suspects = [r for r in rows if r["gap_pct"] > 1.0 or r["unreliable"]]
    if suspects:
        print(
            "[report] re-check these shapes in isolation before believing them:",
            flush=True,
        )
        for r in suspects:
            print(
                f"  {r['M']}x{r['N']} H={r['H']}: {r['gap_pct']:+.2f}%"
                f"{' (unreliable timing)' if r['unreliable'] else ''}",
                flush=True,
            )

    out_path = os.environ.get("HALF_GRID_JSON", "/tmp/mqa_logits_half_66.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[report] wrote {out_path}")
