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
    distinct_group_regions: bool = False,
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
        if distinct_group_regions:
            group = row // total_q
            start = min(group * 3, max(seq_regions - n, 0))
            selected = torch.arange(start, start + n, device=device, dtype=torch.int32)
        else:
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
        "seq_regions": seq_regions,
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
@pytest.mark.parametrize("empty_rows", [False, True])
@pytest.mark.parametrize("distinct_group_regions", [False, True])
def test_tiled_prefill_matches_scalar(empty_rows: bool, distinct_group_regions: bool):
    data = _make_prefill_case(
        total_q=32,
        seq_regions=16,
        topk=8,
        empty_rows=empty_rows,
        seed=42 + int(empty_rows) + 2 * int(distinct_group_regions),
        distinct_group_regions=distinct_group_regions,
    )
    total_q = data["query"].shape[0]
    groups = data["groups"]
    logical = torch.where(
        data["packed"] >= 0,
        data["packed"] & ((1 << 24) - 1),
        torch.full_like(data["packed"], -1),
    )
    # Positions past the last selected token so tiled causal matches scalar
    # ``valid_tokens`` (no extra clipping).
    q_positions = torch.full(
        (total_q,),
        data["seq_regions"] * data["region_size"] - 1,
        device=data["query"].device,
        dtype=torch.int32,
    )
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
def test_tiled_prefill_request_boundary_distinct_groups_matches_scalar():
    data = _make_prefill_case(
        total_q=24,
        seq_regions=16,
        topk=8,
        empty_rows=False,
        seed=47,
        distinct_group_regions=True,
    )
    total_q = data["query"].shape[0]
    groups = data["groups"]
    logical = torch.where(
        data["packed"] >= 0,
        data["packed"] & ((1 << 24) - 1),
        torch.full_like(data["packed"], -1),
    )
    q_positions = torch.full(
        (total_q,),
        data["seq_regions"] * data["region_size"] - 1,
        device=data["query"].device,
        dtype=torch.int32,
    )
    request_indices = torch.tensor(
        [0] * 10 + [1] * 14,
        device=data["query"].device,
        dtype=torch.int32,
    )
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


def test_build_prefill_tiles_uses_query_space_offsets():
    from boltops.step4_dsa.triton.prefill import _build_prefill_tiles

    total_q, groups, block_q = 32, 4, 16
    rows = groups * total_q
    logical = torch.arange(rows, dtype=torch.int32).unsqueeze(1)
    packed = logical.clone()
    q_positions = torch.arange(total_q, dtype=torch.int32)
    request_indices = torch.zeros(total_q, dtype=torch.int32)
    q_starts, q_counts, tile_phys, _tile_logical, lengths, _mask = _build_prefill_tiles(
        logical,
        packed,
        q_positions,
        request_indices,
        total_q=total_q,
        num_kv_groups=groups,
        region_size=8,
        block_q=block_q,
    )
    assert q_starts.tolist() == [0, 16]
    assert q_counts.tolist() == [16, 16]
    assert tile_phys.shape[0] == 2 * groups
    assert lengths.numel() == 2 * groups
    assert not torch.equal(tile_phys[0], tile_phys[1])

    mismatched = torch.zeros(rows, dtype=torch.int32)
    mismatched[total_q:] = 1
    with pytest.raises(ValueError, match="request indices"):
        _build_prefill_tiles(
            logical,
            packed,
            q_positions,
            mismatched,
            total_q=total_q,
            num_kv_groups=groups,
            region_size=8,
            block_q=block_q,
        )


@C.GPU_REQUIRED
def test_sparse_prefill_wide_rows_match_ref():
    """Wide packed rows remain correct when the Q*G grid underfills the device."""
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


def test_prefill_launch_options_follow_backend_capabilities(monkeypatch):
    from boltops.step4_dsa.triton import prefill

    class _Capabilities:
        @staticmethod
        def supports(name: str) -> bool:
            return name in {
                "waves_per_eu",
                "misched_regpressure",
                "empty_arrive_after_mmac",
                "max_memory_cluster_dwords",
            }

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(prefill, "get_triton_capabilities", lambda: _Capabilities())
    options = prefill._build_launch_options(
        {
            "BLOCK_R": 8,
            "num_warps": 2,
            "num_stages": 1,
            "waves_per_eu": 3,
            "misched_regpressure": False,
            "empty_arrive_after_mmac": True,
            "max_memory_cluster_dwords": 16,
            "unknown_option": 7,
        }
    )
    assert options == {
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 3,
        "misched_regpressure": False,
        "empty_arrive_after_mmac": True,
        "max_memory_cluster_dwords": 16,
    }


def test_sparse_prefill_rejects_strided_head_dim():
    query = torch.randn(1, 8, 16)
    key_cache = torch.randn(16, 1, 16)
    value_cache = torch.randn(16, 1, 16)
    packed = torch.full((1, 1), -1, dtype=torch.int32)
    counts = torch.zeros((1,), dtype=torch.int32)
    kwargs = dict(
        packed_regions=packed,
        region_counts=counts,
        num_kv_groups=1,
    )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_prefill(
            C.strided_last_dim(query), key_cache, value_cache, **kwargs
        )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_prefill(
            query, C.strided_last_dim(key_cache), value_cache, **kwargs
        )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_prefill(
            query, key_cache, C.strided_last_dim(value_cache), **kwargs
        )


def test_tiled_prefill_rejects_strided_out():
    from boltops.step4_dsa.triton.prefill import _run_tiled_prefill

    query = torch.randn(1, 8, 16)
    key_cache = torch.randn(16, 1, 16)
    value_cache = torch.randn(16, 1, 16)
    dummy = torch.zeros(1, dtype=torch.int32)
    dummy2d = torch.zeros((1, 1), dtype=torch.int32)
    dummy3d = torch.zeros((1, 1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="last dimension"):
        _run_tiled_prefill(
            query,
            key_cache,
            value_cache,
            dummy,
            dummy,
            dummy2d,
            dummy2d,
            dummy,
            dummy3d,
            dummy,
            packed_width=1,
            num_kv_groups=1,
            region_size=8,
            softmax_scale=1.0,
            block_q=1,
            out=C.strided_last_dim(query),
        )


@C.GPU_REQUIRED
def test_sparse_prefill_accepts_padded_head_dim():
    data = _make_prefill_case(
        total_q=16, seq_regions=32, topk=8, empty_rows=False, seed=45
    )
    query = C.last_dim_padded_view(data["query"], pad=16, poison=3.0)
    key_cache = C.last_dim_padded_view(data["key_cache"], pad=8, poison=-9.0)
    value_cache = C.last_dim_padded_view(data["value_cache"], pad=24, poison=7.0)
    out, lse = triton_sparse_attention_prefill(
        query,
        key_cache,
        value_cache,
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    ref_out, ref_lse = ref_sparse_attention(
        query,
        key_cache,
        value_cache,
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
