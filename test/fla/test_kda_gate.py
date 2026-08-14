"""Correctness tests for KDA gate and chunk-local cumsum."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from bolt_ops.fla.kda import fused_kda_gate, fused_kda_gate_chunk_cumsum
from bolt_ops.fla.common.triton.l2norm import l2norm_fwd
from ._kda_reference import _normalize, naive_gate_chunk_cumsum, naive_kda_gate

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_l2norm_matches_torch_reference() -> None:
    torch.manual_seed(199)
    x = torch.randn(2, 37, 4, 64, device=DEVICE, dtype=torch.bfloat16)
    actual = l2norm_fwd(x)
    expected = _normalize(x).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
def test_gate_matches_torch_reference(lower_bound: float | None) -> None:
    torch.manual_seed(200)
    T, H, K = 37, 4, 64
    raw_g = torch.randn(T, H, K, device=DEVICE, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32)
    actual = fused_kda_gate(
        raw_g.reshape(T, H * K), A_log, K, g_bias=bias,
        lower_bound=lower_bound,
    ).reshape(T, H, K)
    expected = naive_kda_gate(raw_g, A_log, bias, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_gate_chunk_cumsum_matches_torch(
    lower_bound: float | None, use_varlen: bool
) -> None:
    torch.manual_seed(201)
    B, T, H, K = 1, 70, 2, 32
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32)
    cu = torch.tensor([0, 19, 70], device=DEVICE, dtype=torch.int32) if use_varlen else None
    pointwise = naive_kda_gate(raw_g, A_log, bias, lower_bound=lower_bound)
    expected_g, expected_beta = naive_gate_chunk_cumsum(
        pointwise, raw_beta, cu_seqlens=cu
    )
    actual_g, actual_beta = fused_kda_gate_chunk_cumsum(
        raw_g, raw_beta, A_log, g_bias=bias,
        lower_bound=lower_bound, cu_seqlens=cu,
    )
    torch.testing.assert_close(actual_g, expected_g, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(actual_beta, expected_beta, atol=2e-6, rtol=2e-6)
