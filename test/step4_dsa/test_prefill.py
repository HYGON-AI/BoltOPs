"""Correctness tests for sparse prefill attention (stage 5)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_sparse_attention_prefill
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .compare import compare
from .ref import ref_sparse_attention

OUT_RTOL, OUT_ATOL = 1e-2, 2e-2
LSE_RTOL, LSE_ATOL = 1e-3, 1e-3
PREFILL_REGION_VALID_SHIFT = 24


def _make_prefill_case(
    *,
    total_q: int,
    seq_regions: int,
    topk: int,
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
    rows = groups * total_q
    pages = -(-seq_regions * region_size // geo.paged_block_size)
    tokens = pages * geo.paged_block_size

    query = torch.randn(
        (total_q, heads, head_dim),
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

    packed = torch.full((rows, topk), -1, device=device, dtype=torch.int32)
    counts = torch.zeros((rows,), device=device, dtype=torch.int32)
    for row in range(rows):
        if empty_rows and row % 7 == 0:
            continue
        n = min(topk, seq_regions)
        selected = torch.arange(n, device=device, dtype=torch.int32)
        valid = torch.full((n,), region_size, device=device, dtype=torch.int32)
        packed[row, :n] = selected | (valid << PREFILL_REGION_VALID_SHIFT)
        counts[row] = n
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "packed": packed,
        "counts": counts,
        "groups": groups,
        "region_size": region_size,
        "scale": head_dim**-0.5,
    }


@C.GPU_REQUIRED
@pytest.mark.parametrize("empty_rows", [False, True])
def test_sparse_prefill_matches_ref(empty_rows: bool):
    data = _make_prefill_case(
        total_q=16, seq_regions=32, topk=8, empty_rows=empty_rows, seed=41
    )
    out, lse = triton_sparse_attention_prefill(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()
    empty = data["counts"].view(data["groups"], data["query"].shape[0]).eq(0).all(dim=0)
    if empty.any():
        assert torch.isneginf(lse[empty]).all()
        assert torch.equal(out[empty], torch.zeros_like(out[empty]))


@C.GPU_REQUIRED
def test_tiled_prefill_matches_scalar():
    data = _make_prefill_case(
        total_q=32, seq_regions=16, topk=8, empty_rows=False, seed=42
    )
    total_q = data["query"].shape[0]
    groups = data["groups"]
    rows = groups * total_q
    logical = torch.where(
        data["packed"] >= 0,
        data["packed"] & ((1 << 24) - 1),
        torch.full_like(data["packed"], -1),
    )
    # Positions past the last selected token so tiled causal matches scalar
    # ``valid_tokens`` (no extra clipping).
    q_positions = torch.full(
        (total_q,),
        16 * data["region_size"] - 1,
        device=data["query"].device,
        dtype=torch.int32,
    )
    request_indices = torch.zeros((rows,), device=data["query"].device, dtype=torch.int32)
    tiled_out, tiled_lse = triton_sparse_attention_prefill(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=groups,
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
        logical_regions=logical,
        q_positions=q_positions,
        request_indices=request_indices,
        block_q=16,
    )
    scalar_out, scalar_lse = triton_sparse_attention_prefill(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=groups,
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    torch.cuda.synchronize()
    out_check = compare(tiled_out.float(), scalar_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(tiled_lse, scalar_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
def test_sparse_prefill_split_overlap_window_matches_ref():
    """packed_width not divisible by splits*BLOCK_R must not double-count slots."""
    data = _make_prefill_case(
        total_q=8, seq_regions=32, topk=17, empty_rows=False, seed=43
    )
    from boltops.step4_dsa.triton.prefill import sparse_attention_prefill

    out, lse = sparse_attention_prefill(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        num_splits=8,
        config={"BLOCK_R": 8, "BLOCK_H": 16, "BLOCK_D": 256, "num_warps": 4, "num_stages": 1},
    )
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
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
def test_sparse_prefill_auto_kv_split_matches_ref():
    """Wide packed rows with Q*G underfilling CUs should auto-split and match ref."""
    data = _make_prefill_case(
        total_q=4, seq_regions=64, topk=32, empty_rows=False, seed=44
    )
    out, lse = triton_sparse_attention_prefill(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()
