"""Correctness tests and production benches for sparse decode attention (stage 6)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from boltops.step4_dsa import (
    triton_decode_sparse_meta,
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

PERF_OP = "step4_dsa_decode"


def test_decode_config_match_rejects_unavailable_sequence_length():
    from boltops.step4_dsa.triton.decode import _decode_config_match_score

    shape = {"s_q": 4, "topk": 513}
    assert _decode_config_match_score({"s_q": 4}, shape) is not None
    assert _decode_config_match_score({"s_kv": 65536}, shape) is None
    assert _decode_config_match_score({"s_kv_min": 1024}, shape) is None
    assert _decode_config_match_score({"s_kv_max": 65536}, shape) is None


def test_decode_config_lookup_uses_shape_without_sequence_length(monkeypatch):
    from boltops.step4_dsa.triton import decode

    monkeypatch.setattr(decode, "_device_name_for_config", lambda: "test")
    monkeypatch.setattr(
        decode,
        "_load_decode_device_configs",
        lambda device: {
            "configs": [
                {
                    "match": {
                        "s_q": 4,
                        "topk": 513,
                        "dtype": "bfloat16",
                        "s_kv": 65536,
                    },
                    "config": {"BLOCK_R": 2},
                },
                {
                    "match": {"s_q": 4, "topk": 513, "dtype": "bfloat16"},
                    "config": {"BLOCK_R": 4},
                },
            ]
        },
    )
    query = torch.empty((4, 8, 192), dtype=torch.bfloat16)
    key_cache = torch.empty((1, 1, 192), dtype=torch.bfloat16)
    packed_regions = torch.empty((4, 513), dtype=torch.int64)
    assert decode._load_sparse_attn_decode_config(
        query, key_cache, packed_regions, num_splits=1, region_size=8
    )["BLOCK_R"] == 4


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


def test_decode_launch_options_follow_backend_capabilities(monkeypatch):
    from boltops.step4_dsa.triton import decode

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
    monkeypatch.setattr(decode, "get_triton_capabilities", lambda: _Capabilities())
    config = {
        "AUTO_KV_SPLITS": 4,
        "BLOCK_R": 8,
        "BLOCK_H": 16,
        "BLOCK_D": 128,
        "BLOCK_D1": 128,
        "BLOCK_D2": 64,
        "BLOCK_D3": 64,
        "BALANCE_LIVE": True,
        "SPLIT_MAJOR": True,
        "NARROW_INDEX": True,
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 3,
        "misched_regpressure": False,
        "empty_arrive_after_mmac": True,
        "max_memory_cluster_dwords": 16,
        "unknown_option": 7,
    }
    expected_options = {
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 3,
        "misched_regpressure": False,
        "empty_arrive_after_mmac": True,
        "max_memory_cluster_dwords": 16,
    }

    assert decode._filter_launch_config(config, head_dim=128) == {
        "AUTO_KV_SPLITS": 4,
        "BLOCK_R": 8,
        "BLOCK_H": 16,
        "BLOCK_D": 128,
        "BALANCE_LIVE": True,
        "SPLIT_MAJOR": True,
        "NARROW_INDEX": True,
        **expected_options,
    }
    assert decode._decode_head_dim_split(192) == (64, 64, 64)
    assert decode._decode_head_dim_split(128) is None
    assert decode._filter_launch_config(config, head_dim=192) == {
        "AUTO_KV_SPLITS": 4,
        "BLOCK_R": 8,
        "BLOCK_H": 16,
        "BALANCE_LIVE": True,
        "SPLIT_MAJOR": True,
        "NARROW_INDEX": True,
        **expected_options,
    }


@C.GPU_REQUIRED
@pytest.mark.parametrize("empty_rows", [False, True])
@pytest.mark.parametrize("split_major", [False, True])
def test_sparse_decode_matches_ref(empty_rows: bool, split_major: bool):
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
        config={
            "BLOCK_R": 2,
            "BLOCK_H": 16,
            "SPLIT_MAJOR": split_major,
            "num_warps": 1,
            "num_stages": 1,
        },
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
def test_sparse_decode_narrow_index_at_kv_seqlen_65536():
    data = _make_decode_case(
        num_reqs=1, topk=2, context_len=65536, empty_rows=False, seed=63
    )
    starts = torch.tensor(
        [65528, 65520], device=data["packed"].device, dtype=torch.int64
    ).expand(data["packed"].shape[0], -1)
    physical_bits = (data["packed"] >> DECODE_PHYS_SHIFT) << DECODE_PHYS_SHIFT
    data["packed"].copy_(physical_bits | starts)

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
        block_regions=2,
        config={"NARROW_INDEX": True, "BLOCK_R": 2},
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
    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=OUT_RTOL, atol=OUT_ATOL)
    torch.testing.assert_close(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)


@C.GPU_REQUIRED
@pytest.mark.parametrize("num_splits", [1, 16])
@pytest.mark.parametrize("context_len", [512, 8192])
def test_sparse_decode_ragged_live_windows(num_splits, context_len):
    from boltops.step4_dsa.triton.decode import sparse_attention_decode

    data = _make_decode_case(
        num_reqs=4, topk=513, context_len=context_len, empty_rows=False, seed=67
    )
    # Cross tile and split boundaries, including entirely empty split windows.
    counts = torch.tensor(
        [0, 1, 33, 513], device=data["counts"].device, dtype=torch.int32
    ).clamp(max=context_len // 8).repeat(data["groups"])
    data["counts"].copy_(counts)
    kwargs = dict(
        query=data["query"], key_cache=data["key_cache"],
        value_cache=data["value_cache"], packed_regions=data["packed"],
        region_counts=data["counts"], kv_seqlens=data["kv_seqlens"],
        num_kv_groups=data["groups"], region_size=data["region_size"],
        softmax_scale=data["scale"],
    )
    out, lse = sparse_attention_decode(
        **kwargs, num_splits=num_splits,
        config={"BALANCE_LIVE": True, "BLOCK_R": 2},
    )
    if num_splits > 1:
        assert out.shape[0] == num_splits
        original = sparse_attention_decode(**kwargs, num_splits=num_splits,
                                           config={"BLOCK_R": 2})
        torch.testing.assert_close(out, original[0], rtol=0, atol=0)
        torch.testing.assert_close(lse, original[1], rtol=0, atol=0)
        out, lse = triton_merge_split_states(out, lse)
    ref_out, ref_lse = ref_sparse_attention_decode(**kwargs)
    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=OUT_RTOL, atol=OUT_ATOL)
    torch.testing.assert_close(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)


@C.GPU_REQUIRED
def test_sparse_decode_split_overlap_window_matches_ref():
    """packed_width not divisible by splits*BLOCK_R used to double-count slots.

    ``return_partials=False`` merges here so the check is vs the unsplit oracle;
    the vLLM path with ``num_splits>1`` keeps partials instead.
    """
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
        return_partials=False,
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
def test_sparse_decode_matches_vllm_num_splits_shapes():
    """vLLM: num_splits==1 is merged [B,H,D]; num_splits>1 keeps a split axis."""
    data = _make_decode_case(
        num_reqs=4, topk=16, context_len=256, empty_rows=False, seed=65
    )
    from boltops.step4_dsa.triton.decode import sparse_attention_decode

    kwargs = dict(
        query=data["query"],
        key_cache=data["key_cache"],
        value_cache=data["value_cache"],
        packed_regions=data["packed"],
        region_counts=data["counts"],
        kv_seqlens=data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        block_regions=4,
    )
    single_out, single_lse = sparse_attention_decode(num_splits=1, **kwargs)
    partial_out, partial_lse = sparse_attention_decode(num_splits=4, **kwargs)
    assert single_out.shape == data["query"].shape
    assert single_lse.shape == data["query"].shape[:2]
    assert partial_out.shape == (4, *data["query"].shape)
    assert partial_lse.shape == (4, *data["query"].shape[:2])
    merged_out, merged_lse = triton_merge_split_states(partial_out, partial_lse)
    torch.cuda.synchronize()
    out_check = compare(
        merged_out.float(), single_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL
    )
    lse_check = compare(merged_lse, single_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


def test_sparse_decode_rejects_strided_head_dim():
    query = torch.randn(1, 8, 16)
    key_cache = torch.randn(16, 1, 16)
    value_cache = torch.randn(16, 1, 16)
    packed = torch.zeros((1, 1), dtype=torch.int64)
    counts = torch.zeros((1,), dtype=torch.int32)
    seqlens = torch.ones((1,), dtype=torch.int32)
    kwargs = dict(
        packed_regions=packed,
        region_counts=counts,
        kv_seqlens=seqlens,
        num_kv_groups=1,
    )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_decode(
            C.strided_last_dim(query), key_cache, value_cache, **kwargs
        )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_decode(
            query, C.strided_last_dim(key_cache), value_cache, **kwargs
        )
    with pytest.raises(ValueError, match="last dimension"):
        triton_sparse_attention_decode(
            query, key_cache, C.strided_last_dim(value_cache), **kwargs
        )


@C.GPU_REQUIRED
def test_sparse_decode_accepts_padded_head_dim():
    data = _make_decode_case(
        num_reqs=4, topk=8, context_len=256, empty_rows=False, seed=64
    )
    query = C.last_dim_padded_view(data["query"], pad=16, poison=3.0)
    key_cache = C.last_dim_padded_view(data["key_cache"], pad=8, poison=-9.0)
    value_cache = C.last_dim_padded_view(data["value_cache"], pad=24, poison=7.0)
    out, lse = triton_sparse_attention_decode(
        query,
        key_cache,
        value_cache,
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
        query,
        key_cache,
        value_cache,
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
@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("pipeline_kv", [False, True])
def test_sparse_decode_bf16_kv_scales_match_ref(pipeline_kv: bool, head_dim: int):
    data = _make_decode_case(
        num_reqs=4, topk=8, context_len=256, empty_rows=False, seed=70
    )
    data["query"] = data["query"][..., :head_dim]
    data["key_cache"] = data["key_cache"][..., :head_dim]
    data["value_cache"] = data["value_cache"][..., :head_dim]
    data["scale"] = head_dim**-0.5
    k_scale, v_scale = 2.0, 0.5
    config = {"BLOCK_R": 2, "BLOCK_H": 16, "PIPELINE_KV": True} if pipeline_kv else {}
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
        config=config,
        k_scale=k_scale,
        v_scale=v_scale,
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
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    out_check = compare(out.float(), ref_out.float(), rtol=OUT_RTOL, atol=OUT_ATOL)
    lse_check = compare(lse, ref_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    assert out_check.passed, out_check.format()
    assert lse_check.passed, lse_check.format()


@C.GPU_REQUIRED
@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("pipeline_kv", [False, True])
def test_sparse_decode_fp8_kv_cache_matches_ref(
    k_scale: float, v_scale: float, pipeline_kv: bool, head_dim: int
):
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    data = _make_decode_case(
        num_reqs=4, topk=8, context_len=256, empty_rows=False, seed=71
    )
    # Also exercise the shared padded fallback (128) alongside 192-dim paths.
    data["query"] = data["query"][..., :head_dim]
    data["key_cache"] = data["key_cache"][..., :head_dim]
    data["value_cache"] = data["value_cache"][..., :head_dim]
    data["scale"] = head_dim**-0.5
    key_fp8 = C.quantize_kv_fp8(data["key_cache"], k_scale)
    value_fp8 = C.quantize_kv_fp8(data["value_cache"], v_scale)
    config = {"BLOCK_R": 2, "BLOCK_H": 16}
    if pipeline_kv:
        config["PIPELINE_KV"] = True
    out, lse = triton_sparse_attention_decode(
        data["query"],
        key_fp8,
        value_fp8,
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
        num_kv_groups=data["groups"],
        region_size=data["region_size"],
        softmax_scale=data["scale"],
        num_splits=1,
        block_regions=4,
        config=config,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    ref_out, ref_lse = ref_sparse_attention_decode(
        data["query"],
        key_fp8,
        value_fp8,
        data["packed"],
        data["counts"],
        data["kv_seqlens"],
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


# ---------------------------------------------------------------------------
# Performance benchmarks (TP8 per-rank production shapes, shipped configs)
# ---------------------------------------------------------------------------
# One case per entry of ``boltops/step4_dsa/triton/config/sparse_attn_decode``
# (s_q=batch, h_q=8, h_kv=1, d=192, topk=512 (+1 current region),
# num_splits=1, region_size=8).  The config JSON has no sequence length, so
# context=16384 comes from the TP8 production case
# tp8.decode.b4.ctx16384.k512.s1 (dsa-bw1100-tp8-production.jsonl).  Metadata
# is built once by the real decode selector outside the timed region; only
# ``triton_sparse_attention_decode`` is timed.


@dataclass(frozen=True)
class _DecodeBench:
    batch: int
    kv_dtype: str  # "bf16" | "fp8"
    context: int = 16384
    splits: int = 1
    topk: int = 512


_DECODE_BENCH_CASES = [
    _DecodeBench(b, "bf16") for b in (1, 4, 8, 32, 64, 128)
] + [_DecodeBench(b, "fp8") for b in (1, 4, 8)]


def _make_decode_bench_inputs(case: _DecodeBench, device: torch.device) -> dict:
    geo = C.tp8_geometry()
    generator = torch.Generator(device=device).manual_seed(20260831)
    groups, heads, head_dim = (
        geo.num_attention_groups, geo.num_attention_heads, geo.head_dim
    )
    region_size, page_size = geo.region_block_size, geo.paged_block_size
    pages_per_request = -(-case.context // page_size)
    regions = -(-case.context // region_size)
    total_pages = case.batch * pages_per_request

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    query = randn(case.batch, heads, head_dim)
    key_cache = randn(total_pages * page_size, groups, head_dim)
    value_cache = randn(total_pages * page_size, groups, head_dim)
    if case.kv_dtype == "fp8":
        key_cache = C.quantize_kv_fp8(key_cache)
        value_cache = C.quantize_kv_fp8(value_cache)
    block_table = (
        torch.randperm(total_pages, generator=generator, device=device)
        .to(torch.int32)
        .view(case.batch, pages_per_request)
        .contiguous()
    )
    kv_seqlens = torch.full(
        (case.batch,), case.context, device=device, dtype=torch.int32
    )
    logits = randn(groups * case.batch, regions, dtype=torch.float32)
    packed, counts = triton_decode_sparse_meta(
        logits,
        kv_seqlens,
        block_table,
        topk=case.topk,
        region_size=region_size,
        regions_per_page=geo.regions_per_page,
    )
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
@pytest.mark.bench
@pytest.mark.parametrize(
    "case",
    _DECODE_BENCH_CASES,
    ids=lambda c: f"b{c.batch}-ctx{c.context}-{c.kv_dtype}-s{c.splits}",
)
def test_bench_sparse_decode(
    case: _DecodeBench, time_ms, bench_warmup: int, bench_repeat: int
) -> None:
    if case.kv_dtype == "fp8" and getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    data = _make_decode_bench_inputs(case, torch.device(C.DEVICE))

    def run():
        return triton_sparse_attention_decode(
            data["query"],
            data["key_cache"],
            data["value_cache"],
            data["packed"],
            data["counts"],
            data["kv_seqlens"],
            num_kv_groups=data["groups"],
            region_size=data["region_size"],
            softmax_scale=data["scale"],
            num_splits=case.splits,
        )

    out, lse = run()
    torch.cuda.synchronize()
    assert out.shape[-3:] == data["query"].shape
    assert torch.isfinite(lse).all()

    ms = time_ms(run, bench_warmup, bench_repeat)
    print(
        f"[bench] step4_dsa decode b={case.batch} ctx={case.context} "
        f"kv={case.kv_dtype} splits={case.splits}: {ms:.4f} ms"
    )
