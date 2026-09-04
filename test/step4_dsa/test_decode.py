"""Correctness tests for sparse decode attention (stage 6)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import (
    triton_merge_split_states,
    triton_sparse_attention_decode,
)
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .compare import compare
from .ref import ref_sparse_attention_decode

OUT_RTOL, OUT_ATOL = 1e-2, 4e-3
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3
DECODE_PHYS_SHIFT = 32


def _make_decode_case(
    *,
    num_reqs: int,
    topk: int,
    context_len: int,
    empty_rows: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    device = torch.device(C.DEVICE)
    geo = STEP4
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = geo.num_attention_groups
    heads = geo.num_attention_heads
    head_dim = geo.head_dim
    region_size = geo.region_block_size
    rows = groups * num_reqs
    max_regions = -(-context_len // region_size)
    pages = -(-max_regions * region_size // geo.paged_block_size)
    tokens = pages * geo.paged_block_size

    query = torch.randn(
        (num_reqs, heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    key_cache = torch.randn(
        (tokens, groups, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    value_cache = torch.randn(
        (tokens, groups, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    kv_seqlens = torch.full(
        (num_reqs,), context_len, device=device, dtype=torch.int32
    )
    packed = torch.zeros((rows, topk), device=device, dtype=torch.int64)
    counts = torch.zeros((rows,), device=device, dtype=torch.int32)
    stride = max(1, max_regions // max(topk, 1))
    for row in range(rows):
        if empty_rows and row % 5 == 0:
            continue
        n = min(topk, max_regions)
        logical = torch.arange(n, device=device, dtype=torch.int64) * stride
        logical = torch.clamp(logical, max=max_regions - 1)
        physical = logical
        start_token = logical * region_size
        packed[row, :n] = start_token | (physical << DECODE_PHYS_SHIFT)
        counts[row] = n
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "packed": packed,
        "counts": counts,
        "kv_seqlens": kv_seqlens,
        "groups": groups,
        "region_size": region_size,
        "scale": head_dim**-0.5,
    }


@C.GPU_REQUIRED
@pytest.mark.parametrize("empty_rows", [False, True])
def test_sparse_decode_matches_ref(empty_rows: bool):
    data = _make_decode_case(
        num_reqs=4, topk=8, context_len=256, empty_rows=empty_rows, seed=61
    )
    out, lse = triton_sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        num_splits=1,
        block_regions=4,
    )
    ref_out, ref_lse = ref_sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
def test_sparse_decode_split_overlap_window_matches_ref():
    """packed_width not divisible by splits*BLOCK_R used to double-count slots."""
    data = _make_decode_case(
        num_reqs=4, topk=17, context_len=256, empty_rows=False, seed=63
    )
    from boltops.step4_dsa.triton.decode import sparse_attention_decode

    out, lse = sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        num_splits=8,
        config={"BLOCK_R": 2, "BLOCK_H": 16, "BLOCK_D": 256, "num_warps": 2, "num_stages": 2},
    )
    ref_out, ref_lse = ref_sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
def test_sparse_decode_split_then_merge_matches_single():
    data = _make_decode_case(
        num_reqs=4, topk=16, context_len=256, empty_rows=False, seed=62
    )
    kwargs = dict(
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    single_out, single_lse = triton_sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_splits=1,
        **kwargs,
    )
    partial_out, partial_lse = triton_sparse_attention_decode(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_splits=4,
        **kwargs,
    )
    merged_out, merged_lse = triton_merge_split_states(partial_out, partial_lse)
    torch.cuda.synchronize()
    out_check = compare(
        merged_out.float(), single_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL
    )
    lse_check = compare(merged_lse, single_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()
