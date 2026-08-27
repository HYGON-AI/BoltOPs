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


@GPU_REQUIRED
def test_dense_decode_single_sequence_state_chain() -> None:
    """Exercise the multi-token recurrent loop used by software pipelining."""
    torch.manual_seed(211)
    T, H, K, V = 3, 2, 32, 32
    q = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    gate = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.float32) * -0.01
    beta = torch.sigmoid(torch.randn(1, T, H, device=DEVICE, dtype=torch.float32))
    initial = torch.randn(1, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    expected_out, expected_state = naive_recurrent_kda(
        q, k, v, beta, gate, scale=K**-0.5,
        initial_state=initial, use_qk_l2norm=True,
    )

    state = torch.zeros(T + 1, H, V, K, device=DEVICE, dtype=torch.float32)
    state[1] = initial[0]
    out, _ = fused_recurrent_kda_fwd(
        q, k, v, gate, beta, initial_state=state,
        cu_seqlens=torch.tensor([0, T], device=DEVICE, dtype=torch.int32),
        ssm_state_indices=torch.arange(
            1, T + 1, device=DEVICE, dtype=torch.int32,
        ).view(1, T),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out, expected_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        state[T], expected_state[0], atol=8e-2, rtol=4e-2,
    )
