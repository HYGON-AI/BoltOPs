"""Correctness tests for dense and packed KDA decode."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.fla.kda import (
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)
from ._kda_reference import naive_kda_gate, naive_recurrent_kda

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
@pytest.mark.parametrize("tokens", [1, 3], ids=["decode", "spec"])
def test_decode_protocols_match_torch_reference(
    lower_bound: float | None, tokens: int
) -> None:
    torch.manual_seed(210)
    # Dense/packed recurrent wrappers require equal token strides; use K=V
    # here. Non-square K/V is covered by the prefill chunk tests.
    B, H, K, V = 1, 2, 32, 32
    q = torch.randn(B, tokens, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, tokens, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, tokens, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, tokens, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    state = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    cu = torch.arange(tokens + 1, device=DEVICE, dtype=torch.int32)
    indices = torch.zeros(tokens, device=DEVICE, dtype=torch.int32)
    gate = fused_kda_gate(
        raw_g.reshape(tokens, H * K), A_log, K, g_bias=bias,
        lower_bound=lower_bound,
    ).reshape(B, tokens, H, K)
    beta = raw_beta.float().sigmoid()
    expected_out, expected_state = naive_recurrent_kda(
        q, k, v, beta, gate, scale=K**-0.5,
        initial_state=state, use_qk_l2norm=True,
    )

    dense_state = state.clone()
    dense_out, _ = fused_recurrent_kda_fwd(
        q, k, v, gate, beta, initial_state=dense_state,
        cu_seqlens=cu, ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    fused_state = state.clone()
    fused_out, _ = fused_recurrent_kda(
        q, k, v, raw_g, raw_beta, A_log, bias, lower_bound,
        fused_state, cu, indices,
    )
    mixed_qkv = torch.cat(
        (q[0].reshape(tokens, H * K),
         k[0].reshape(tokens, H * K),
         v[0].reshape(tokens, H * V)),
        dim=-1,
    ).contiguous()
    packed_state = state.clone()
    packed_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv, raw_g, raw_beta, A_log, bias, lower_bound,
        packed_state, indices,
    )
    for actual in (dense_out, fused_out, packed_out):
        torch.testing.assert_close(actual, expected_out, atol=3e-2, rtol=2e-2)
    for name, actual_state in (
        ("dense", dense_state), ("fused", fused_state), ("packed", packed_state)
    ):
        torch.testing.assert_close(
            actual_state, expected_state, atol=8e-2, rtol=4e-2,
            msg=f"{name} state mismatch",
        )
