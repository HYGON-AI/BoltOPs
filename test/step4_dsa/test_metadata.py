# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for prefill / decode sparse metadata (stage 4)."""

from __future__ import annotations

import torch

from boltops.step4_dsa import triton_decode_sparse_meta, triton_prefill_sparse_meta
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .gen import make_topk_inputs
from .ref import ref_decode_sparse_meta, ref_topk_select


@C.GPU_REQUIRED
def test_prefill_sparse_meta_width_and_logical_ids():
    device = torch.device(C.DEVICE)
    geo = STEP4
    total_q, seq_regions, topk = 32, 64, 16
    inputs = make_topk_inputs(
        geo, total_q=total_q, seq_regions=seq_regions, device=device, seed=21
    )
    packed, counts, logical = triton_prefill_sparse_meta(
        inputs.logits,
        inputs.q_positions,
        inputs.block_table,
        topk=topk,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
        return_logical_ids=True,
    )
    torch.cuda.synchronize()
    assert packed.shape == (inputs.logits.shape[0], topk + 1)
    assert logical.shape == packed.shape
    assert packed.dtype == torch.int32
    assert counts.dtype == torch.int32
    history = torch.div(
        inputs.q_positions, geo.region_block_size, rounding_mode="floor"
    )
    expected_counts = torch.minimum(history, torch.tensor(topk, device=device)) + 1
    assert torch.equal(counts, expected_counts.to(torch.int32))


@C.GPU_REQUIRED
def test_decode_sparse_meta_matches_ref():
    device = torch.device(C.DEVICE)
    geo = STEP4
    batch, groups, regions, topk, context = 2, geo.num_provider_groups, 64, 16, 256
    generator = torch.Generator(device=device).manual_seed(33)
    logits = torch.randn(
        (groups * batch, regions),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    kv_seqlens = torch.full((batch,), context, device=device, dtype=torch.int32)
    pages = -(-regions // geo.regions_per_page)
    block_table = (
        torch.arange(pages, device=device, dtype=torch.int32).view(1, -1).repeat(batch, 1)
    )
    packed, counts = triton_decode_sparse_meta(
        logits,
        kv_seqlens,
        block_table,
        topk=topk,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
    )
    history = torch.div(
        (kv_seqlens - 1).clamp_min(0),
        geo.region_block_size,
        rounding_mode="floor",
    ).repeat(groups)
    logical = ref_topk_select(logits, history, topk=topk)
    ref_packed, ref_counts = ref_decode_sparse_meta(
        logical,
        kv_seqlens,
        block_table,
        topk=topk,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
    )
    torch.cuda.synchronize()
    assert torch.equal(packed, ref_packed)
    assert torch.equal(counts, ref_counts)
