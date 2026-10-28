# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for split-KV state merge."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_merge_split_states
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .compare import compare
from .ref import ref_merge_split_states


@C.GPU_REQUIRED
def test_merge_split_states_matches_ref():
    device = torch.device(C.DEVICE)
    geo = STEP4
    splits, batch, heads, dim = 4, 2, 8, geo.head_dim
    generator = torch.Generator(device=device).manual_seed(71)
    partial_out = torch.randn(
        (splits, batch, heads, dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    partial_lse = torch.randn(
        (splits, batch, heads),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    # One empty split so the merge must drop -inf columns.
    partial_lse[3] = float("-inf")
    actual_out, actual_lse = triton_merge_split_states(partial_out, partial_lse)
    ref_out, ref_lse = ref_merge_split_states(partial_out, partial_lse)
    torch.cuda.synchronize()
    out_check = compare(actual_out.float(), ref_out.float(), rtol=1e-3, atol=1e-3)
    lse_check = compare(actual_lse, ref_lse, rtol=1e-3, atol=1e-3)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
def test_merge_all_empty_is_zero_neg_inf():
    device = torch.device(C.DEVICE)
    geo = STEP4
    partial_out = torch.zeros((4, 2, 8, geo.head_dim), device=device, dtype=torch.bfloat16)
    partial_lse = torch.full((4, 2, 8), float("-inf"), device=device, dtype=torch.float32)
    out, lse = triton_merge_split_states(partial_out, partial_lse)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()


def test_merge_split_states_rejects_strided_head_dim():
    partial_out = torch.randn(2, 1, 8, 16)
    partial_lse = torch.randn(2, 1, 8)
    with pytest.raises(ValueError, match="last dimension"):
        triton_merge_split_states(C.strided_last_dim(partial_out), partial_lse)
