"""Correctness and size-grid performance tests for prefill FP8 MQA logits.

Backends under test: ``row``, ``bn``, ``gluon`` (plus public ``auto``).

Size-grid benches reuse the prefill shape list from
``lightop/test/mqa_logits/test_mqa_logits.py`` (Q×KV×H, D=128). Implementation
code does not depend on lightop; optional ``--bench -k vs_lightop`` compares
latency when ``lightop`` is importable.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch

pytest.importorskip("triton")

from boltops.mqa_logits import fp8_mqa_logits  # noqa: E402
from boltops.utility.arch_info import get_fp8_e4m3_dtype  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

e4m3_type = get_fp8_e4m3_dtype()
fp8_max = torch.finfo(e4m3_type).max

# Prefill size grid from lightop/test/mqa_logits/test_mqa_logits.py (cuda list).
_SIZE_GRID_Q = (16, 32, 64, 127)
_SIZE_GRID_KV = (66, 4096, 8192, 16384, 32768, 65536, 131072)
_SIZE_GRID_H = (8, 64)
_SIZE_GRID_D = 128

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
def test_fp8_mqa_logits_backends(
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
    logits = fp8_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, backend=backend
    )
    _assert_close_to_ref(logits, q_fp8, kv_fp8, scales, weights, ks, ke)


@GPU_REQUIRED
@pytest.mark.parametrize("s_q", [1, 17, 61, 128])
@pytest.mark.parametrize("s_k", [16, 76, 113, 1024])
@pytest.mark.parametrize("num_heads", [16, 64])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("disable_cp", [True, False])
@torch.inference_mode()
def test_fp8_mqa_logits_auto_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    disable_cp: bool,
) -> None:
    if s_q > s_k:
        pytest.skip("s_q > s_k")

    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=disable_cp
    )
    logits = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)
    _assert_close_to_ref(logits, q_fp8, kv_fp8, scales, weights, ks, ke)


@GPU_REQUIRED
@torch.inference_mode()
def test_fp8_mqa_logits_dout_inplace() -> None:
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(32, 1024, 64, 128)
    d_out = torch.empty(32, 1024, device=DEVICE, dtype=torch.float32)
    out = fp8_mqa_logits(
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
def test_bench_fp8_mqa_logits_size_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    """Perf on the lightop prefill size grid (auto backend)."""
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    d_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)

    def run():
        fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend="auto"
        )

    ms = time_ms(run, bench_warmup, bench_repeat)
    tflops = useful_flops(ks, ke, num_heads, head_dim) / (ms * 1e-3) / 1e12
    print(
        f"[bench size-grid] auto {s_q}x{s_k} H={num_heads} D={head_dim} "
        f"-> {ms:.3f} ms, {tflops:.2f} TFLOPS"
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("s_q,s_k,num_heads,head_dim", list(_size_grid_params()))
@pytest.mark.parametrize("backend", ["row", "bn", "gluon"])
@torch.inference_mode()
def test_bench_fp8_mqa_logits_backends_size_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    backend: str,
    bench_warmup: int,
    bench_repeat: int,
    time_ms,
) -> None:
    if backend == "gluon" and not _gluon_available():
        pytest.skip("Gluon unavailable")
    # Row over ultra-large grids is multi-minute; keep BN/Gluon/auto for those.
    if backend == "row" and s_q * s_k > 64 * 65536:
        pytest.skip("row skipped on ultra-large MN")

    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    d_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)

    def run():
        fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend=backend
        )

    ms = time_ms(run, bench_warmup, bench_repeat)
    tflops = useful_flops(ks, ke, num_heads, head_dim) / (ms * 1e-3) / 1e12
    print(
        f"[bench size-grid] {backend} {s_q}x{s_k} H={num_heads} D={head_dim} "
        f"-> {ms:.3f} ms, {tflops:.2f} TFLOPS"
    )


def _require_lightop():
    try:
        import lightop  # noqa: WPS433
    except ImportError as exc:
        pytest.skip(f"lightop not available: {exc}")
    return lightop


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("s_q,s_k,num_heads,head_dim", list(_size_grid_params()))
@torch.inference_mode()
def test_bench_fp8_mqa_logits_vs_lightop_size_grid(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    bench_warmup: int,
    bench_repeat: int,
    time_ms: Callable[[Callable[[], None], int, int], float],
) -> None:
    """Optional: same lightop size grid, BoltOPs auto vs lightop.mqa_logits."""
    lightop = _require_lightop()
    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        s_q, s_k, num_heads, head_dim, disable_cp=True
    )
    d_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    lo_out = torch.empty(s_q, s_k, device=DEVICE, dtype=torch.float32)
    flops = useful_flops(ks, ke, num_heads, head_dim)

    def run_boltops():
        fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, D_out=d_out, backend="auto"
        )

    def run_lightop():
        lightop.mqa_logits(
            q_fp8, kv_fp8, weights, ks, ke, scales, clean_logit=True, D_out=lo_out
        )

    ms_boltops = time_ms(run_boltops, bench_warmup, bench_repeat)
    ms_lightop = time_ms(run_lightop, bench_warmup, bench_repeat)
    ratio = ms_lightop / ms_boltops if ms_boltops > 0 else float("inf")
    print(
        f"[bench vs_lightop size-grid] {s_q}x{s_k} H={num_heads}\n"
        f"  boltops : {ms_boltops:.3f} ms, "
        f"{flops / (ms_boltops * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  lightop : {ms_lightop:.3f} ms, "
        f"{flops / (ms_lightop * 1e-3) / 1e12:.2f} TFLOPS\n"
        f"  lightop/boltops: {ratio:.2f}x (>1 means boltops is faster)"
    )
def test_fp8_mqa_logits_impl_default_triton():
    from boltops.mqa_logits import fp8_mqa_logits

    q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(16, 128, 8, 128)
    out = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)
    assert out.shape == (16, 128)
    out2 = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, impl="triton")
    assert out2.shape == (16, 128)



