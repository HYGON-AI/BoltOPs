# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and size-grid performance tests for prefill FP8 MQA logits.

Backends under test: ``row``, ``bn``, ``gluon`` (plus public ``auto``).

Size-grid benches reuse the prefill shape list from hipc / aiter lineage tests
(Q×KV×H, D=128). ``pytest --bench`` times BoltOPs only. Hipc comparison is
opt-in and is skipped by optest ``per``:

* ``--bench -k vs_hipc_size`` — synthetic size grid
* ``--bench -k vs_hipc_prod`` — 50 production prefill slices (GLM / V4 CSA / PP_Stage0)
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

pytest.importorskip("triton")

from boltops.mqa_logits import triton_mqa_logits  # noqa: E402
from boltops.utility.arch_info import get_fp8_e4m3_dtype  # noqa: E402

# optest CSV name. File stem would be ``triton_fp8_mqa_logits``.
PERF_OP = "fp8_mqa_logits"

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

e4m3_type = get_fp8_e4m3_dtype()
fp8_max = torch.finfo(e4m3_type).max

# Prefill size grid from hipc / aiter lineage tests (cuda list).
_SIZE_GRID_Q = (16, 32, 64, 127)
_SIZE_GRID_KV = (66, 4096, 8192, 16384, 32768, 65536, 131072)
_SIZE_GRID_H = (8, 64)
_SIZE_GRID_D = 128

# Production prefill slices (50 shapes): GLM chunked / V4 CSA / V4 PP_Stage0.
_PROD_SLICE_D = 128
_PROD_SLICE_SHAPES: list[tuple[int, int, int, str]] = (
    [(6656, 6656 * (i + 1), 32, "glm") for i in range(10)]
    + [(6656, 1664 * (i + 1), 64, "v4_csa") for i in range(20)]
    + [
        (6656, kv, 64, "v4_ppstage0")
        for kv in (
            3167, 6334, 9502, 12669, 15836, 19003, 22170, 25337, 28505,
            31672, 34839, 38006, 41173, 44340, 47508, 50675, 53842, 57009,
            60176, 62369,
        )
    ]
)


def _prod_slice_params():
    for s_q, s_k, num_heads, family in _PROD_SLICE_SHAPES:
        yield pytest.param(
            s_q, s_k, num_heads, family,
            id=f"{family}-{s_q}x{s_k}-H{num_heads}",
        )


# Correctness covers all three backends on a compact, representative set.
_CORRECTNESS_SHAPES = [
    pytest.param(16, 66, 8, 128, id="tiny-H8"),
    pytest.param(32, 66, 64, 128, id="tiny-H64"),
    pytest.param(64, 128, 64, 128, id="short-N128"),
    pytest.param(64, 4096, 64, 128, id="mid"),
    pytest.param(127, 8192, 64, 128, id="mid-tall"),
    pytest.param(64, 16384, 8, 128, id="long-H8"),
]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_amax = x.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    x_scaled = (x * (1.0 / sf)).to(e4m3_type)
    return x_scaled, sf.squeeze(1)


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    q_f = q.float()
    k_f = kv.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q_f, k_f)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def generate_cp_test_data(seq_len: int, seq_len_kv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)
    ke = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def useful_flops(
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> float:
    starts = cu_starts.to("cpu").numpy()
    ends = cu_ends.to("cpu").numpy()
    total = 0.0
    for start, end in zip(starts, ends):
        total += 2.0 * num_heads * head_dim * max(int(end) - int(start), 0)
    return total


def _gluon_available() -> bool:
    try:
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
            gluon_prefill_available,
        )

        return bool(gluon_prefill_available())
    except ImportError:
        return False


def _make_inputs(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    *,
    disable_cp: bool = True,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    q = torch.randn(s_q, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device=DEVICE, dtype=torch.bfloat16)
    kv_fp8, scales = per_token_cast_to_fp8(kv)
    kv_dequant = (kv_fp8.float() * scales[:, None]).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device=DEVICE, dtype=torch.float32)

    if disable_cp or s_k % s_q != 0 or s_q % 2 != 0:
        ks = torch.zeros(s_q, dtype=torch.int32, device=DEVICE)
        ke = torch.arange(s_q, dtype=torch.int32, device=DEVICE) + (s_k - s_q)
    else:
        ks, ke = generate_cp_test_data(s_q, s_k)

    q_fp8 = q.to(e4m3_type)
    kv_fp8, scales = per_token_cast_to_fp8(kv_dequant)
    return q_fp8, kv_fp8, scales, weights, ks, ke


def _assert_close_to_ref(
    logits: torch.Tensor,
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    atol_diff: float = 1e-3,
) -> None:
    ref_logits = ref_fp8_mqa_logits(
        q=q_fp8.float(),
        kv=(kv_fp8.float() * scales[:, None]),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_cmp = ref_logits.masked_fill(ref_neginf_mask, 0)
    out_cmp = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(out_cmp, ref_cmp)
    assert diff < atol_diff, f"{diff=}"


@GPU_REQUIRED
@pytest.mark.parametrize("s_q,s_k,num_heads,head_dim", _CORRECTNESS_SHAPES)
@pytest.mark.parametrize("backend", ["auto", "row", "bn", "gluon"])
@torch.inference_mode()
def test_triton_fp8_mqa_logits_backends(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    backend: str,
) -> None:
    if backend == "gluon" and not _gluon_available():
        pytest.skip("Gluon unavailable")

    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    logits = triton_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, backend=backend
    )
    _assert_close_to_ref(logits, q_fp8, kv_fp8, scales, weights, ks, ke)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "s_q,s_k,num_heads,head_dim,disable_cp",
    [
        # Odd sizes + short KV (dispatch / mask edges).
        pytest.param(17, 76, 16, 64, True, id="odd-short-H16-D64"),
        pytest.param(61, 113, 64, 128, True, id="odd-mid-H64"),
        # CP window path (even s_q, s_k multiple of s_q).
        pytest.param(128, 1024, 64, 128, False, id="cp-window"),
        pytest.param(1, 16, 16, 128, True, id="tiny-M1"),
    ],
)
@torch.inference_mode()
def test_triton_fp8_mqa_logits_auto_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    disable_cp: bool,
) -> None:
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=disable_cp
    )
    logits = triton_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)
    _assert_close_to_ref(logits, q_fp8, kv_fp8, scales, weights, ks, ke)


@GPU_REQUIRED
@torch.inference_mode()
def test_triton_fp8_mqa_logits_dout_inplace() -> None:
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(32, 1024, 64, 128)
    d_out = torch.empty(32, 1024, device=DEVICE, dtype=torch.float32)
    out = triton_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend="auto"
    )
    assert out.data_ptr() == d_out.data_ptr()
    _assert_close_to_ref(out, q_fp8, kv_fp8, scales, weights, ks, ke)


def _size_grid_params():
    for s_q in _SIZE_GRID_Q:
        for s_k in _SIZE_GRID_KV:
            if s_q > s_k:
                continue
            for num_heads in _SIZE_GRID_H:
                yield pytest.param(
                    s_q,
                    s_k,
                    num_heads,
                    _SIZE_GRID_D,
                    id=f"{s_q}x{s_k}-H{num_heads}",
                )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("s_q,s_k,num_heads,head_dim", list(_size_grid_params()))
@torch.inference_mode()
def test_bench_triton_fp8_mqa_logits_size_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """Perf on the hipc / aiter prefill size grid (auto backend)."""
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    d_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)

    def run():
        triton_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend="auto"
        )

    flops = useful_flops(ks, ke, num_heads, head_dim)
    ms = time_ms(run, bench_warmup, bench_repeat, flops=flops)
    print(
        f"[bench size-grid] auto {s_q}x{s_k} H={num_heads} D={head_dim} "
        f"-> {ms:.3f} ms, {flops / (ms * 1e-3) / 1e12:.2f} TFLOPS"
    )


def _require_hipc():
    """Import hipc reference (package still installs as ``lightop``)."""
    try:
        import lightop as hipc  # noqa: WPS433
    except ImportError as exc:
        pytest.skip(f"hipc not available: {exc}")
    return hipc


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("s_q,s_k,num_heads,head_dim", list(_size_grid_params()))
@torch.inference_mode()
def test_bench_triton_fp8_mqa_logits_vs_hipc_size_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """Optional: same hipc size grid, BoltOPs auto vs hipc.mqa_logits."""
    hipc = _require_hipc()
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    d_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    hi_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    flops = useful_flops(ks, ke, num_heads, head_dim)

    def run_boltops():
        triton_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend="auto"
        )

    def run_hipc():
        hipc.mqa_logits(
            q_fp8, kv_fp8, weights, ks, ke, scales, clean_logit=True, D_out=hi_out
        )

    ms_boltops = time_ms(run_boltops, bench_warmup, bench_repeat, flops=flops)
    ms_hipc = time_ms(run_hipc, bench_warmup, bench_repeat, flops=flops)
    ratio = ms_hipc / ms_boltops if ms_boltops > 0 else float("inf")
    print(
        f"[bench vs_hipc size-grid] {s_q}x{s_k} H={num_heads}\n"
        f"  boltops : {ms_boltops:.3f} ms, "
        f"{flops / (ms_boltops * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  hipc    : {ms_hipc:.3f} ms, "
        f"{flops / (ms_hipc * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  hipc/boltops: {ratio:.2f}x (>1 means boltops is faster)"
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("s_q,s_k,num_heads,family", list(_prod_slice_params()))
@torch.inference_mode()
def test_bench_triton_fp8_mqa_logits_vs_hipc_prod_slices(
    s_q: int,
    s_k: int,
    num_heads: int,
    family: str,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """50 production prefill slices: BoltOPs auto vs hipc.

    Run with::

        pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_prod
    """
    hipc = _require_hipc()
    from boltops.mqa_logits.triton.mqa_config_utils import get_fp8_mqa_shape_dispatch

    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, _PROD_SLICE_D, disable_cp=True
    )
    out_bo = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    out_hi = torch.empty_like(out_bo)
    disp = get_fp8_mqa_shape_dispatch(
        s_q, s_k, num_heads, gluon_available=_gluon_available()
    )
    path = disp.get("path", "?")
    sk = int(disp.get("head_split_k", 1))

    def run_boltops():
        triton_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=out_bo, backend="auto"
        )

    def run_hipc():
        hipc.mqa_logits(
            q_fp8, kv_fp8, weights, ks, ke, scales, True, out_hi
        )

    # Accuracy on dedicated buffers (not the timed outs).
    acc_bo = torch.empty_like(out_bo)
    got = triton_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, D_out=acc_bo, backend="auto"
    )
    _assert_close_to_ref(
        got, q_fp8, kv_fp8, scales, weights, ks, ke, atol_diff=2e-2
    )

    flops = useful_flops(ks, ke, num_heads, _PROD_SLICE_D)
    ms_bo = time_ms(run_boltops, bench_warmup, bench_repeat, flops=flops)
    ms_hi = time_ms(run_hipc, bench_warmup, bench_repeat, flops=flops)
    ratio = ms_hi / ms_bo if ms_bo > 0 else float("inf")
    print(
        f"[bench vs_hipc_prod] {family} {s_q}x{s_k} H={num_heads} "
        f"path={path} sk={sk}\n"
        f"  boltops : {ms_bo:.4f} ms, "
        f"{flops / (ms_bo * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  hipc    : {ms_hi:.4f} ms, "
        f"{flops / (ms_hi * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  hipc/boltops: {ratio:.3f}x (>1 means boltops is faster)"
    )


def test_triton_mqa_logits_public_export():
    from boltops.mqa_logits import triton_mqa_logits
    import boltops.mqa_logits as pkg

    assert callable(triton_mqa_logits)
    assert not hasattr(pkg, "mqa_logits")
    assert not hasattr(pkg, "fp8_mqa_logits")

