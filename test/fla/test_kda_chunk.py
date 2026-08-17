"""Correctness tests for KDA chunk stages and full prefill orchestration."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.fla.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_with_fused_gate,
    recompute_w_u_fwd,
)
from ._kda_reference import naive_kda_gate, naive_recompute_w_u, naive_recurrent_kda

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_recompute_w_u_matches_torch_reference() -> None:
    torch.manual_seed(220)
    B, T, H, K, V, BT = 1, 16, 2, 32, 24, 16
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)
    gk = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.05
    A = torch.tril(torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.bfloat16))
    actual = recompute_w_u_fwd(k, v, beta, A, gk=gk)
    expected = naive_recompute_w_u(k, v, beta, A, gk)
    for got, want in zip((actual[0], actual[1], actual[3]), expected):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=1e-2)


@GPU_REQUIRED
def test_full_fused_gate_prefill_matches_recurrent_reference() -> None:
    torch.manual_seed(221)
    B, T, H, K, V = 1, 64, 2, 32, 24
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    initial = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    gate = naive_kda_gate(raw_g, A_log, bias, lower_bound=-3.0)
    beta = raw_beta.float().sigmoid()
    expected, expected_state = naive_recurrent_kda(
        q, k, v, beta, gate, scale=K**-0.5,
        initial_state=initial, use_qk_l2norm=True,
    )
    actual, actual_state = chunk_kda_with_fused_gate(
        q, k, v, raw_g, raw_beta, A_log, bias,
        scale=K**-0.5, initial_state=initial.clone(),
        output_final_state=True, lower_bound=-3.0,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=6e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=8e-2, rtol=6e-2)
