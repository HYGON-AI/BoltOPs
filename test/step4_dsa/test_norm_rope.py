# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for indexer norm + partial RoPE (stage 0)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_build_rope_cache, triton_indexer_norm_rope
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .compare import compare_scaled
from .ref import ref_indexer_norm_rope


@C.GPU_REQUIRED
@pytest.mark.parametrize("tokens", [1, 64, 512])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_indexer_norm_rope(tokens: int, zero_centered: bool):
    device = torch.device(C.DEVICE)
    geo = STEP4
    inputs = C.make_norm_inputs(
        geo,
        tokens=tokens,
        max_position=4096,
        device=device,
        seed=20260812 + tokens,
        dtype=torch.bfloat16,
    )
    kwargs = dict(
        head_dim=geo.proxy_dim,
        num_q_heads=geo.indexer_num_heads,
        num_k_heads=geo.indexer_num_k_heads,
        rotary_dim=geo.indexer_rope_dim // 2,
        eps=1e-6,
        q_norm_weight_bias=1.0 if zero_centered else 0.0,
    )
    ordered = (
        inputs["index_q"],
        inputs["index_k"],
        inputs["index_z"],
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["k_norm_bias"],
        inputs["cos"],
        inputs["sin"],
        inputs["positions"],
    )
    actual = triton_indexer_norm_rope(*ordered, **kwargs)
    reference = ref_indexer_norm_rope(*ordered, **kwargs)
    torch.cuda.synchronize()

    assert torch.equal(actual[2], inputs["index_z"])
    for out, ref in zip(actual[:2], reference[:2]):
        check = compare_scaled(out, ref, ulps=2.0)
        assert check.passed, check.format()


@C.GPU_REQUIRED
def test_build_rope_cache_shape():
    device = torch.device(C.DEVICE)
    cos, sin = triton_build_rope_cache(
        rotary_span=32,
        theta=10000.0,
        max_position=128,
        device=device,
        dtype=torch.bfloat16,
    )
    assert cos.shape == (128, 16)
    assert sin.shape == (128, 16)
    assert cos.dtype == torch.bfloat16
    assert torch.isfinite(cos.float()).all()
