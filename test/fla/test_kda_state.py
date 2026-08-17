"""Correctness tests for KDA chunk state propagation."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.fla.kda.triton.chunk_state import chunk_gated_delta_rule_fwd_h
from ._kda_reference import naive_chunk_state

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_state_matches_torch_chunk_reference() -> None:
    torch.manual_seed(230)
    B, T, H, K, V, BT = 1, 70, 2, 32, 24, 64
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w = torch.randn_like(k) * 0.1
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    cu = None
    gk = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    ranges = [(0, T)]
    for start, end in ranges:
        for chunk_start in range(start, end, BT):
            chunk_end = min(chunk_start + BT, end)
            gk[:, chunk_start:chunk_end] = gk[:, chunk_start:chunk_end].cumsum(1)
    initial = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    actual_h, actual_v, actual_final = chunk_gated_delta_rule_fwd_h(
        k, w, u, gk=gk, initial_state=initial.clone(),
        output_final_state=True, chunk_size=BT, use_exp2=True,
        cu_seqlens=cu,
    )
    expected_h, expected_v, expected_final = naive_chunk_state(
        k, w, u, gk, initial.clone(), chunk_size=BT
    )
    torch.testing.assert_close(actual_h, expected_h, atol=6e-2, rtol=4e-2)
    assert actual_v is not None and expected_v is not None
    torch.testing.assert_close(actual_v, expected_v, atol=6e-2, rtol=4e-2)
    assert actual_final is not None and expected_final is not None
    torch.testing.assert_close(actual_final, expected_final, atol=6e-2, rtol=4e-2)
