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
PREFILL_REGION_ID_MASK = (1 << PREFILL_REGION_VALID_SHIFT) - 1


def _assert_prefill_close(
    actual_out: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_out: torch.Tensor,
    expected_lse: torch.Tensor,
    *,
    what: str,
) -> None:
    out_check = compare(
        actual_out.float(), expected_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL
    )
    lse_check = compare(actual_lse, expected_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, f"{what} out: {out_check.format()}"
    assert lse_check.passed, f"{what} lse: {lse_check.format()}"


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
        data["packed"] & PREFILL_REGION_ID_MASK,
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
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=groups,
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    _assert_prefill_close(tiled_out, tiled_lse, scalar_out, scalar_lse, what="tiled vs scalar")
    _assert_prefill_close(tiled_out, tiled_lse, ref_out, ref_lse, what="tiled vs ref")


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
        data["packed"] & PREFILL_REGION_ID_MASK,
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
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["packed"],
        data["counts"],
        num_kv_groups=groups,
        region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    torch.cuda.synchronize()
    _assert_prefill_close(tiled_out, tiled_lse, scalar_out, scalar_lse, what="tiled vs scalar")
    _assert_prefill_close(tiled_out, tiled_lse, ref_out, ref_lse, what="tiled vs ref")


def _make_tiled_multi_request_case() -> dict[str, torch.Tensor]:
    """Two ragged requests whose boundary would be crossed by a naive 128-query tile.

    Ported from handoff ``run_stage5_attn._make_tiled_multi_request_case``: TP8
    shard (8 heads, 1 KV group), shuffled paged cache, packed ``valid_tokens``
    already encode the causal prefix.
    """
    device = torch.device(C.DEVICE)
    geo = STEP4
    request_lengths = (65, 191)
    total_q = sum(request_lengths)
    num_heads = 8
    num_kv_groups = 1
    head_dim = geo.head_dim
    region_size = geo.region_block_size
    page_size = geo.paged_block_size
    regions_per_page = geo.regions_per_page
    topk = 32
    width = topk + 1
    generator = torch.Generator(device=device).manual_seed(20260831)

    query = torch.randn(
        (total_q, num_heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    total_pages = sum(-(-length // page_size) for length in request_lengths)
    page_permutation = torch.randperm(
        total_pages, generator=generator, device=device
    ).to(torch.int32)
    key_cache = torch.empty(
        (total_pages * page_size, num_kv_groups, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.empty_like(key_cache)

    block_tables: list[torch.Tensor] = []
    page_cursor = 0
    for length in request_lengths:
        pages = -(-length // page_size)
        table = page_permutation[page_cursor : page_cursor + pages]
        block_tables.append(table)
        logical_key = torch.randn(
            (pages * page_size, num_kv_groups, head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        logical_value = torch.randn(
            (pages * page_size, num_kv_groups, head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for logical_page in range(pages):
            physical_page = int(table[logical_page])
            source = slice(logical_page * page_size, (logical_page + 1) * page_size)
            target = slice(physical_page * page_size, (physical_page + 1) * page_size)
            key_cache[target] = logical_key[source]
            value_cache[target] = logical_value[source]
        page_cursor += pages

    packed = torch.full((total_q, width), -1, device=device, dtype=torch.int32)
    logical_ids = torch.full_like(packed, -1)
    counts = torch.empty((total_q,), device=device, dtype=torch.int32)
    q_positions = torch.empty((total_q,), device=device, dtype=torch.int32)
    request_indices = torch.empty((total_q,), device=device, dtype=torch.int32)

    row = 0
    for request, length in enumerate(request_lengths):
        table = block_tables[request]
        for position in range(length):
            current = position // region_size
            first = max(0, current - topk + 1)
            selected = list(range(first, current + 1))
            counts[row] = len(selected)
            q_positions[row] = position
            request_indices[row] = request
            for slot, logical in enumerate(selected):
                physical_page = int(table[logical // regions_per_page])
                physical = physical_page * regions_per_page + logical % regions_per_page
                valid = min(max(position + 1 - logical * region_size, 0), region_size)
                packed[row, slot] = physical | (valid << PREFILL_REGION_VALID_SHIFT)
                logical_ids[row, slot] = logical
            row += 1

    return {
        "query": query,
        "key_cache": key_cache.contiguous(),
        "value_cache": value_cache.contiguous(),
        "packed": packed,
        "logical_ids": logical_ids,
        "counts": counts,
        "q_positions": q_positions,
        "request_indices": request_indices,
        "groups": num_kv_groups,
        "region_size": region_size,
        "scale": head_dim**-0.5,
    }


@C.GPU_REQUIRED
def test_tiled_prefill_multi_request_matches_scalar_and_ref():
    """Handoff stage-5 tiled gate: tiled vs scalar Triton and vs FP32 torch_ref."""
    case = _make_tiled_multi_request_case()
    common = dict(
        query=case["query"],
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        packed_regions=case["packed"],
        region_counts=case["counts"],
        num_kv_groups=case["groups"],
        region_size=case["region_size"],
        softmax_scale=case["scale"],
        block_regions=4,
    )
    scalar_out, scalar_lse = triton_sparse_attention_prefill(**common)
    tiled_out, tiled_lse = triton_sparse_attention_prefill(
        **common,
        logical_regions=case["logical_ids"],
        q_positions=case["q_positions"],
        request_indices=case["request_indices"],
        block_q=128,
    )
    ref_out, ref_lse = ref_sparse_attention(
        case["query"],
        case["key_cache"],
        case["value_cache"],
        case["packed"],
        case["counts"],
        num_kv_groups=case["groups"],
        region_size=case["region_size"],
        softmax_scale=case["scale"],
    )
    torch.cuda.synchronize()
    _assert_prefill_close(tiled_out, tiled_lse, scalar_out, scalar_lse, what="tiled vs scalar")
    _assert_prefill_close(tiled_out, tiled_lse, ref_out, ref_lse, what="tiled vs ref")


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


def test_prefill_head_dim_split_uses_three_64_chunks():
    from boltops.step4_dsa.triton.prefill import _prefill_head_dim_split

    assert _prefill_head_dim_split(192) == (64, 64, 64)
    assert _prefill_head_dim_split(128) is None
    assert _prefill_head_dim_split(256) is None
    assert _prefill_head_dim_split(96) is None


@C.GPU_REQUIRED
def test_sparse_prefill_kv_split_matches_ref():
    """Forced extra-wave KV splits match the reference."""
    data = _make_prefill_case(
        total_q=8, seq_regions=64, topk=32, empty_rows=True, seed=47
    )
    config = {
        "AUTO_KV_SPLITS": 4,
        "BLOCK_R": 4,
        "BLOCK_H": 16,
        "num_warps": 2,
        "num_stages": 1,
    }
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
        config=config,
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
            "AUTO_KV_SPLITS": 4,
            "PIPELINE_KV": True,
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


@C.GPU_REQUIRED
def test_sparse_prefill_bf16_kv_scales_match_ref():
    data = _make_prefill_case(
        total_q=16, seq_regions=32, topk=8, empty_rows=False, seed=47
    )
    k_scale, v_scale = 2.0, 0.5
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
        k_scale=k_scale,
        v_scale=v_scale,
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
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (2.0, 0.5)])
def test_sparse_prefill_fp8_kv_cache_matches_ref(k_scale: float, v_scale: float):
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    data = _make_prefill_case(
        total_q=16, seq_regions=32, topk=8, empty_rows=False, seed=48
    )
    key_fp8 = C.quantize_kv_fp8(data["key_cache"], k_scale)
    value_fp8 = C.quantize_kv_fp8(data["value_cache"], v_scale)
    out, lse = triton_sparse_attention_prefill(
        data["query"],
        key_fp8,
        value_fp8,
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        key_fp8,
        value_fp8,
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
def test_tiled_prefill_fp8_kv_cache_matches_scalar():
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    data = _make_prefill_case(
        total_q=32, seq_regions=16, topk=8, empty_rows=False, seed=49
    )
    k_scale, v_scale = 2.0, 0.5
    key_fp8 = C.quantize_kv_fp8(data["key_cache"], k_scale)
    value_fp8 = C.quantize_kv_fp8(data["value_cache"], v_scale)
    total_q = data["query"].shape[0]
    logical = torch.where(
        data["packed"] >= 0,
        data["packed"] & PREFILL_REGION_ID_MASK,
        torch.full_like(data["packed"], -1),
    )
    q_positions = torch.full(
        (total_q,),
        data["seq_regions"] * data["region_size"] - 1,
        device=data["query"].device,
        dtype=torch.int32,
    )
    kwargs = dict(
        query=data["query"],
        key_cache=key_fp8,
        value_cache=value_fp8,
        packed_regions=data["packed"],
        region_counts=data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    tiled_out, tiled_lse = triton_sparse_attention_prefill(
        **kwargs, logical_regions=logical, q_positions=q_positions, block_q=16
    )
    scalar_out, scalar_lse = triton_sparse_attention_prefill(**kwargs)
    ref_out, ref_lse = ref_sparse_attention(
        data["query"],
        key_fp8,
        value_fp8,
        data["packed"],
        data["counts"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    _assert_prefill_close(tiled_out, tiled_lse, scalar_out, scalar_lse, what="tiled vs scalar")
    _assert_prefill_close(tiled_out, tiled_lse, ref_out, ref_lse, what="tiled vs ref")
