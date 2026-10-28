# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for the production Qwen Sparse-GQA interfaces."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
from boltops.tools.sparse_gqa_triton_tune.benchmark_cases import (
    PRODUCTION_CASES,
    make_production_inputs,
)
from boltops.tools.sparse_gqa_triton_tune.reference import (
    reference_compact,
    reference_decode,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_public_signatures_match_production_contract() -> None:
    assert list(
        inspect.signature(
            qwen_sparse_kv_extraction_compact_hcu_fa_triton
        ).parameters
    ) == [
        "k", "v", "req_to_token", "req_indices", "indices", "seq_lens",
        "cu_k", "out_k", "out_v", "batch", "topk", "page_size",
        "k_scale", "v_scale",
    ]
    assert list(inspect.signature(sparse_gqa_packed_decode_triton).parameters) == [
        "q", "k", "v", "indices", "cu_q", "cu_k", "kv_lens", "scale",
        "k_scale", "v_scale",
    ]


def _make_compact_case(
    *,
    batch: int = 3,
    topk: int = 17,
    heads: int = 2,
    dim: int = 64,
    page_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    counts: tuple[int, ...] | None = None,
):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(17)
    if counts is None:
        counts = tuple(topk - 2 * row for row in range(batch))
    if len(counts) != batch or not all(0 < count <= topk for count in counts):
        raise ValueError("counts must contain one value in [1, topk] per request")

    num_requests = batch + 2
    max_context = topk + 7
    pages = math.ceil(num_requests * max_context / page_size)
    k = torch.randn(
        (pages, heads, page_size, dim), device=device, dtype=dtype,
        generator=generator,
    )
    v_nhd = torch.randn(
        (pages, heads, page_size, dim), device=device, dtype=dtype,
        generator=generator,
    )
    v = v_nhd.transpose(2, 3).contiguous()
    slots = torch.randperm(pages * page_size, device=device, generator=generator)
    req_to_token = slots[: num_requests * max_context].reshape(
        num_requests, max_context
    ).to(torch.int32)
    req_indices = torch.tensor(
        [(2 * row + 2) % num_requests for row in range(batch)],
        device=device, dtype=torch.int32,
    )
    seq_lens = torch.tensor(
        [max_context - row for row in range(batch)],
        device=device, dtype=torch.int32,
    )
    counts_tensor = torch.tensor(counts, device=device, dtype=torch.int32)
    cu_k = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), counts_tensor.cumsum(0))
    )
    indices = torch.full((batch, topk), -1, device=device, dtype=torch.int32)
    for row, count in enumerate(counts):
        indices[row, :count] = torch.randperm(
            int(seq_lens[row]), device=device, generator=generator
        )[:count].to(torch.int32)
    out_shape = (sum(counts), heads, dim)
    return {
        "k": k,
        "v": v,
        "req_to_token": req_to_token,
        "req_indices": req_indices,
        "indices": indices,
        "seq_lens": seq_lens,
        "cu_k": cu_k,
        "counts": counts_tensor,
        "out_k": torch.empty(out_shape, device=device, dtype=dtype),
        "out_v": torch.empty(out_shape, device=device, dtype=dtype),
        "batch": batch,
        "topk": topk,
        "page_size": page_size,
    }


def _run_compact(data, *, k_scale=None, v_scale=None):
    result = qwen_sparse_kv_extraction_compact_hcu_fa_triton(
        data["k"], data["v"], data["req_to_token"], data["req_indices"],
        data["indices"], data["seq_lens"], data["cu_k"], data["out_k"],
        data["out_v"], data["batch"], data["topk"], data["page_size"],
        k_scale, v_scale,
    )
    assert result is None


def _fp8_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch does not expose the production FP8 e4m3 dtype")
    return dtype


def _convert_cache_to_fp8(data) -> None:
    dtype = _fp8_dtype()
    try:
        data["k"] = data["k"].to(dtype)
        data["v"] = data["v"].to(dtype)
    except RuntimeError as exc:
        pytest.skip(f"FP8 conversion is unavailable: {exc}")


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hcu_fa_compact_matches_paged_reference(dtype: torch.dtype) -> None:
    data = _make_compact_case(dtype=dtype)
    expected_k, expected_v = reference_compact(data)
    _run_compact(data)
    torch.testing.assert_close(data["out_k"], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(data["out_v"], expected_v, rtol=0, atol=0)


@GPU_REQUIRED
def test_compact_crosses_tile_boundaries_with_ragged_cu_k() -> None:
    counts = (17, 31, 32, 33, 911, 2048, 2050, 2051)
    data = _make_compact_case(
        batch=8, topk=2051, heads=1, dim=256, page_size=64, counts=counts
    )
    expected_k, expected_v = reference_compact(data)
    _run_compact(data)
    torch.testing.assert_close(data["out_k"], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(data["out_v"], expected_v, rtol=0, atol=0)


@GPU_REQUIRED
def test_fp8_compact_descales_to_query_dtype() -> None:
    data = _make_compact_case(dtype=torch.bfloat16)
    _convert_cache_to_fp8(data)
    k_scale, v_scale = 0.5, 0.25
    expected_k, expected_v = reference_compact(
        data, k_scale=k_scale, v_scale=v_scale
    )
    _run_compact(data, k_scale=k_scale, v_scale=v_scale)
    torch.testing.assert_close(data["out_k"], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(data["out_v"], expected_v, rtol=0, atol=0)


def _make_decode_case(
    *,
    q_lens: tuple[int, ...] = (1, 1, 1),
    kv_lens: tuple[int, ...] | None = None,
    topk: int = 33,
    q_heads: int = 3,
    kv_heads: int = 1,
    dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    shuffled_indices: bool = False,
    per_kv_head_indices: bool = False,
):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(29)
    batch = len(q_lens)
    if kv_lens is None:
        kv_lens = tuple(topk - 2 * row for row in range(batch))
    cu_q = torch.tensor(
        (0, *torch.tensor(q_lens).cumsum(0).tolist()),
        device=device, dtype=torch.int32,
    )
    cu_k = torch.tensor(
        (0, *torch.tensor(kv_lens).cumsum(0).tolist()),
        device=device, dtype=torch.int32,
    )
    q = torch.randn(
        (sum(q_lens), q_heads, dim), device=device, dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        (sum(kv_lens), kv_heads, dim), device=device, dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        (sum(kv_lens), kv_heads, dim), device=device, dtype=dtype,
        generator=generator,
    )
    index_shape = (
        (sum(q_lens), kv_heads, topk)
        if per_kv_head_indices
        else (sum(q_lens), topk)
    )
    indices = torch.full(index_shape, -1, device=device, dtype=torch.int32)
    row = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        for relative_q in range(q_len):
            visible = relative_q + kv_len - q_len + 1
            live = min(topk, visible)
            for kv_head in range(kv_heads if per_kv_head_indices else 1):
                selected = (
                    torch.randperm(live, device=device, generator=generator)
                    if shuffled_indices
                    else torch.arange(live, device=device)
                ).to(torch.int32)
                if per_kv_head_indices:
                    indices[row, kv_head, :live] = selected
                else:
                    indices[row, :live] = selected
            row += 1
    return (
        q, k, v, indices, cu_q, cu_k,
        torch.tensor(kv_lens, device=device, dtype=torch.int32),
    )


@GPU_REQUIRED
@pytest.mark.parametrize("batch", [1, 8, 32], ids=["D1", "D8", "D32"])
@pytest.mark.parametrize(
    ("q_heads", "kv_heads"), [(3, 1), (6, 1)], ids=["TP8", "TP4"]
)
def test_production_decode_geometries_match_reference(
    batch: int, q_heads: int, kv_heads: int
) -> None:
    args = _make_decode_case(
        q_lens=(1,) * batch, topk=2051, q_heads=q_heads,
        kv_heads=kv_heads, dim=256,
    )
    scale = 256**-0.5
    expected = reference_decode(*args, scale)
    actual = sparse_gqa_packed_decode_triton(*args, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=4e-3)


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_uses_shuffled_indices(dtype: torch.dtype) -> None:
    args = _make_decode_case(
        q_heads=6, kv_heads=2, dtype=dtype, shuffled_indices=True
    )
    scale = args[0].shape[-1] ** -0.5
    expected = reference_decode(*args, scale)
    actual = sparse_gqa_packed_decode_triton(*args, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)


@GPU_REQUIRED
def test_decode_supports_per_kv_head_indices() -> None:
    args = _make_decode_case(
        q_heads=6, kv_heads=2, shuffled_indices=True,
        per_kv_head_indices=True,
    )
    scale = args[0].shape[-1] ** -0.5
    expected = reference_decode(*args, scale)
    actual = sparse_gqa_packed_decode_triton(*args, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)


@GPU_REQUIRED
def test_packed_decode_accepts_strided_query_and_indices() -> None:
    q, k, v, indices, cu_q, cu_k, kv_lens = _make_decode_case(
        shuffled_indices=True
    )
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    indices_storage = torch.full(
        (indices.shape[0], indices.shape[1] * 2), -1,
        device=indices.device, dtype=indices.dtype,
    )
    indices_storage[:, ::2] = indices
    indices = indices_storage[:, ::2]
    assert q.stride(-1) != 1
    assert indices.stride(-1) != 1

    scale = q.shape[-1] ** -0.5
    expected = reference_decode(q, k, v, indices, cu_q, cu_k, kv_lens, scale)
    actual = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)


@GPU_REQUIRED
def test_fp8_decode_applies_independent_kv_scales() -> None:
    q, k, v, indices, cu_q, cu_k, kv_lens = _make_decode_case(
        shuffled_indices=True
    )
    fp8_dtype = _fp8_dtype()
    try:
        k = k.to(fp8_dtype)
        v = v.to(fp8_dtype)
    except RuntimeError as exc:
        pytest.skip(f"FP8 conversion is unavailable: {exc}")
    scale, k_scale, v_scale = q.shape[-1] ** -0.5, 0.5, 0.25
    expected = reference_decode(
        q, k, v, indices, cu_q, cu_k, kv_lens,
        scale, k_scale, v_scale,
    )
    actual = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, kv_lens,
        scale, k_scale, v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=8e-3)


@GPU_REQUIRED
@pytest.mark.parametrize("q_heads", [3, 6], ids=["TP8", "TP4"])
def test_fp8_compact_then_decode_production_geometry(q_heads: int) -> None:
    """Exercise the real HCU path, including the B=8 tuned configuration."""
    counts = (17, 31, 32, 33, 911, 2048, 2050, 2051)
    data = _make_compact_case(
        batch=8, topk=2051, heads=1, dim=256, page_size=64, counts=counts
    )
    _convert_cache_to_fp8(data)
    k_scale, v_scale = 0.5, 0.25
    expected_k, expected_v = reference_compact(
        data, k_scale=k_scale, v_scale=v_scale
    )
    _run_compact(data, k_scale=k_scale, v_scale=v_scale)
    torch.testing.assert_close(data["out_k"], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(data["out_v"], expected_v, rtol=0, atol=0)

    generator = torch.Generator(device=DEVICE).manual_seed(31 + q_heads)
    q = torch.randn(
        (8, q_heads, 256), device=DEVICE, dtype=torch.bfloat16,
        generator=generator,
    )
    cu_q = torch.arange(9, device=DEVICE, dtype=torch.int32)
    indices = torch.arange(2051, device=DEVICE, dtype=torch.int32).expand(8, -1)
    indices = indices.masked_fill(
        indices >= data["counts"][:, None], -1
    ).contiguous()
    scale = 256**-0.5
    expected = reference_decode(
        q, expected_k, expected_v, indices, cu_q, data["cu_k"],
        data["counts"], scale,
    )
    actual = sparse_gqa_packed_decode_triton(
        q, data["out_k"], data["out_v"], indices, cu_q,
        data["cu_k"], data["counts"], scale,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=4e-3)


@GPU_REQUIRED
def test_decode_validation_rejects_invalid_contracts() -> None:
    q, k, v, indices, cu_q, cu_k, kv_lens = _make_decode_case()
    scale = q.shape[-1] ** -0.5
    with pytest.raises(ValueError, match="divisible"):
        sparse_gqa_packed_decode_triton(
            q[:, :2], k.repeat(1, 3, 1), v.repeat(1, 3, 1),
            indices, cu_q, cu_k, kv_lens, scale,
        )
    with pytest.raises(ValueError, match="BF16/FP16"):
        sparse_gqa_packed_decode_triton(
            q.float(), k, v, indices, cu_q, cu_k, kv_lens, scale
        )
    with pytest.raises(ValueError, match="indices must be int32 or int64"):
        sparse_gqa_packed_decode_triton(
            q, k, v, indices.to(torch.int16), cu_q, cu_k, kv_lens, scale
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        sparse_gqa_packed_decode_triton(
            q, k, v, indices, cu_q[:-1], cu_k, kv_lens, scale
        )


@GPU_REQUIRED
def test_compact_validation_rejects_invalid_contracts() -> None:
    data = _make_compact_case()
    args = (
        data["k"], data["v"], data["req_to_token"], data["req_indices"],
        data["indices"], data["seq_lens"], data["cu_k"], data["out_k"],
        data["out_v"], data["batch"], data["topk"], data["page_size"],
    )
    with pytest.raises(ValueError, match="expected K"):
        qwen_sparse_kv_extraction_compact_hcu_fa_triton(
            data["k"], data["k"], *args[2:]
        )
    with pytest.raises(ValueError, match="page_size"):
        qwen_sparse_kv_extraction_compact_hcu_fa_triton(*args[:-1], 4)
    with pytest.raises(ValueError, match="indices must be int32 or int64"):
        qwen_sparse_kv_extraction_compact_hcu_fa_triton(
            *args[:4], data["indices"].to(torch.int16), *args[5:]
        )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("case", PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_bench_simulated_production_decode(
    case,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Benchmark the production TP4/TP8 D1/D8/D32 HCU decode shapes."""
    data = make_production_inputs(case, access_pattern="production", fp8_cache=True)
    compact_data = data["compact"]
    q, packed_k, packed_v, indices, cu_q, cu_k, kv_lens = data["decode"]

    def compact() -> None:
        qwen_sparse_kv_extraction_compact_hcu_fa_triton(
            compact_data["k"],
            compact_data["v"],
            compact_data["req_to_token"],
            compact_data["req_indices"],
            compact_data["indices"],
            compact_data["seq_lens"],
            compact_data["cu_k"],
            packed_k,
            packed_v,
            case.batch,
            case.topk,
            case.page_size,
            data["k_scale"],
            data["v_scale"],
        )

    def decode():
        return sparse_gqa_packed_decode_triton(
            q,
            packed_k,
            packed_v,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            case.head_dim**-0.5,
        )

    def end_to_end():
        compact()
        return decode()

    expected_k, expected_v = reference_compact(
        compact_data, k_scale=data["k_scale"], v_scale=data["v_scale"]
    )
    compact()
    torch.testing.assert_close(packed_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(packed_v, expected_v, rtol=0, atol=0)
    expected_out = reference_decode(
        q,
        expected_k,
        expected_v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        case.head_dim**-0.5,
    )
    actual_out = decode()
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-2, atol=4e-3)

    compact_ms = time_ms(compact, bench_warmup, bench_repeat)
    decode_ms = time_ms(decode, bench_warmup, bench_repeat)
    end_to_end_ms = time_ms(end_to_end, bench_warmup, bench_repeat)
    print(
        f"Sparse-GQA {case.case_id}: compact={compact_ms:.6f} ms, "
        f"decode={decode_ms:.6f} ms, "
        f"e2e={end_to_end_ms:.6f} ms"
    )


@GPU_REQUIRED
@pytest.mark.parametrize(
    "pattern",
    ["grouped", "unaligned", "random", "duplicates", "holes", "strided", "offset"],
)
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_compact_grouped_v_preserves_general_gather(pattern, fp8, packed):
    """The vectorized V path must preserve arbitrary slots and masked writes."""
    from boltops.sparse_gqa.triton.kv_compact import _launch_compact_hcu_fa

    data = _make_compact_case(
        batch=3, topk=19, heads=2, dim=80, page_size=8, counts=(19, 17, 13)
    )
    if pattern != "random":
        mapping = torch.arange(
            data["req_to_token"].shape[1], device=data["k"].device,
            dtype=torch.int32,
        )
        data["req_to_token"][:] = mapping + (pattern == "unaligned")
        for row, count in enumerate((19, 17, 13)):
            data["indices"][row, :count] = torch.arange(
                count, device=data["k"].device, dtype=torch.int32
            )
    if pattern == "duplicates":
        data["indices"][:, 5:8] = data["indices"][:, 1:4]
    if pattern == "holes":
        data["indices"][:, 0] = -1
        data["indices"][:, 6] = data["seq_lens"]
    if fp8:
        _convert_cache_to_fp8(data)
    if pattern == "offset":
        original = data["v"]
        storage = torch.empty(
            original.numel() + 1, device=original.device, dtype=original.dtype
        )
        data["v"] = storage[1:].view(original.shape)
        data["v"].copy_(original)
    if pattern == "strided":
        # Force the generic V path and exercise strided source/destination rows.
        for name in ("k", "v", "out_k", "out_v"):
            original = data[name]
            storage = torch.empty(
                (*original.shape[:-1], original.shape[-1] * 2),
                device=original.device, dtype=original.dtype,
            )
            data[name] = storage[..., ::2]
            if name in ("k", "v"):
                data[name].copy_(original)
    data["out_k"].fill_(-7)
    data["out_v"].fill_(-7)
    expected_k = data["out_k"].clone()
    expected_v = data["out_v"].clone()
    k_scale, v_scale = (0.5, 0.25) if fp8 else (1.0, 1.0)
    for b in range(data["batch"]):
        start, end = int(data["cu_k"][b]), int(data["cu_k"][b + 1])
        pos = data["indices"][b, :end - start].long()
        valid = (pos >= 0) & (pos < data["seq_lens"][b])
        slots = data["req_to_token"][
            int(data["req_indices"][b]), pos[valid]
        ].long()
        rows = torch.arange(start, end, device=pos.device)[valid]
        pages, offsets = slots // data["page_size"], slots % data["page_size"]
        expected_k[rows] = (
            data["k"][pages, :, offsets, :].float() * k_scale
        ).to(expected_k.dtype)
        expected_v[rows] = (
            data["v"][pages, :, :, offsets].float() * v_scale
        ).to(expected_v.dtype)
    _launch_compact_hcu_fa(
        **{key: value for key, value in data.items() if key != "counts"},
        k_scale=k_scale, v_scale=v_scale,
        config={
            "BLOCK_TOPK": 8, "num_warps": 2, "num_stages": 1,
            "GROUPED_V": True, "TOKEN_MAJOR": True,
            "PACKED_V": packed,
        },
    )
    torch.testing.assert_close(data["out_k"], expected_k, atol=0, rtol=0)
    torch.testing.assert_close(data["out_v"], expected_v, atol=0, rtol=0)


@GPU_REQUIRED
def test_compact_packed_v_preserves_fp8_output_bits():
    from boltops.sparse_gqa.triton.kv_compact import _launch_compact_hcu_fa

    data = _make_compact_case(topk=17, counts=(17, 15, 13))
    _convert_cache_to_fp8(data)
    data["req_to_token"][:] = torch.arange(
        data["req_to_token"].shape[1], device=data["k"].device,
        dtype=torch.int32,
    )
    for row, count in enumerate((17, 15, 13)):
        data["indices"][row, :count] = torch.arange(
            count, device=data["k"].device, dtype=torch.int32
        )
    data["out_k"] = torch.empty_like(data["out_k"], dtype=data["k"].dtype)
    data["out_v"] = torch.empty_like(data["out_v"], dtype=data["v"].dtype)
    expected_k, expected_v = reference_compact(data)
    _launch_compact_hcu_fa(
        **{key: value for key, value in data.items() if key != "counts"},
        k_scale=0.5, v_scale=0.25,
        config={
            "BLOCK_TOPK": 4, "num_warps": 2, "num_stages": 1,
            "GROUPED_V": True, "TOKEN_MAJOR": True, "PACKED_V": True,
        },
    )
    torch.testing.assert_close(
        data["out_k"].view(torch.uint8), expected_k.view(torch.uint8),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        data["out_v"].view(torch.uint8), expected_v.view(torch.uint8),
        atol=0, rtol=0,
    )
