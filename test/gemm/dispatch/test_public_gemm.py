# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Public operation contracts have one entry point each, independent of kernels."""

import pytest
import torch
import triton

from boltops import gemm


def require_hcu():
    if (
        not torch.cuda.is_available()
        or triton.runtime.driver.active.get_current_target().arch != "gfx946"
    ):
        pytest.skip("requires gfx946")


def test_public_surface_has_no_legacy_duplicates():
    expected = {
        "gemm_a16w16",
        "gemm_a32w32",
        "batched_gemm_bf16",
        "gemm_a8w8_blockscale",
        "gemm_a8w8",
        "gemm_a16w4",
        "gemm_w4a4",
        "batched_gemm_a16w4",
    }
    assert set(gemm.__all__) == expected
    assert all(callable(getattr(gemm, name)) for name in expected)
    for old in (
        "gemm_dense",
        "gemm_scaled",
        "gemm_batched",
        "gemm_grouped",
        "gemm_batched_a16wfp4_online_quant",
    ):
        assert not hasattr(gemm, old)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_contiguous", [False, True])
@pytest.mark.parametrize("m_contiguous", [False, True])
def test_batched_native(dtype, n_contiguous, m_contiguous):
    require_hcu()
    torch.manual_seed(631)
    a = (torch.randint(-2, 3, (2, 3, 32)).float() / 4).to(dtype)
    b = (torch.randint(-2, 3, (2, 5, 32)).float() / 4).to(dtype)
    weight = b.transpose(1, 2).contiguous().transpose(1, 2) if n_contiguous else b
    bias = torch.full((2, 1, 5), 0.25, dtype=dtype)
    activation = a.transpose(1, 2).contiguous().transpose(1, 2) if m_contiguous else a
    output = torch.empty((2, 3, 5), dtype=dtype, device="cuda")
    y = gemm.batched_gemm_bf16(
        x=activation.cuda(), w=weight.cuda(), bias=bias.cuda(), dtype=dtype, out=output
    )
    assert y is output
    torch.testing.assert_close(
        y.cpu(),
        (torch.bmm(a.float(), b.float().transpose(1, 2)) + bias).to(dtype),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("block_scale", [False, True])
def test_scaled_a8w8(dtype, block_scale):
    require_hcu()
    torch.manual_seed(634)
    a = torch.randint(-2, 3, (3, 64)).to(dtype)
    b = torch.randint(-2, 3, (17, 64)).to(dtype)
    recipe = ((1, 32), (32, 32)) if block_scale else ((1, 64), (1, 64))
    sa = torch.full((3, 2 if block_scale else 1), 0.5)
    sb = torch.full((1 if block_scale else 17, 2 if block_scale else 1), 0.25)
    bias = None if block_scale else torch.full((17,), 0.5, device="cuda")
    output = torch.empty((3, 17), dtype=torch.float32, device="cuda")
    if block_scale:
        y = gemm.gemm_a8w8_blockscale(
            a.cuda(),
            b.cuda(),
            sa.cuda(),
            sb.cuda(),
            dtype=torch.float32,
            out=output,
            backend="triton",
        )
    else:
        y = gemm.gemm_a8w8(
            a.cuda(),
            b.cuda(),
            sa.cuda(),
            sb.cuda(),
            bias=bias,
            dtype=torch.float32,
            out=output,
            backend="triton",
        )
    assert y is output
    golden = (a.float() @ b.float().T) * 0.125 + (0 if bias is None else 0.5)
    torch.testing.assert_close(y.cpu(), golden, atol=0, rtol=0)


def test_awq_explicit_zero_points():
    require_hcu()
    # Every packed nibble: weight=3, zero=1 => dequantized weight=1.
    a = torch.full((2, 64), 0.25, dtype=torch.float16, device="cuda")
    w = torch.full((64, 8), 0x33333333, dtype=torch.int32, device="cuda")
    z = torch.full((1, 8), 0x11111111, dtype=torch.int32, device="cuda")
    s = torch.full((1, 64), 0.5, dtype=torch.float16, device="cuda")
    output = torch.empty((2, 64), dtype=torch.float16, device="cuda")
    y = gemm.gemm_a16w4(a, w, s, z, out=output)
    assert y is output
    torch.testing.assert_close(
        y.cpu(), torch.full((2, 64), 16.0, dtype=torch.float16), atol=0, rtol=0
    )


def test_awq_zero_point_defaults_to_zero():
    require_hcu()
    a = torch.full((2, 64), 0.25, dtype=torch.float16, device="cuda")
    w = torch.full((64, 8), 0x33333333, dtype=torch.int32, device="cuda")
    s = torch.full((1, 64), 0.5, dtype=torch.float16, device="cuda")
    y = gemm.gemm_a16w4(a, w, s)
    torch.testing.assert_close(
        y.cpu(), torch.full((2, 64), 24.0, dtype=torch.float16), atol=0, rtol=0
    )


def test_scaled_prequantized_mxfp4():
    require_hcu()
    a = torch.full((16, 32), 0x22, dtype=torch.uint8, device="cuda")
    b = torch.full((16, 32), 0x22, dtype=torch.uint8, device="cuda")
    scale = torch.full((16, 2), 127, dtype=torch.uint8, device="cuda")
    output = torch.empty((16, 16), dtype=torch.bfloat16, device="cuda")
    y = gemm.gemm_w4a4(a, b, scale, scale, out=output)
    assert y is output
    torch.testing.assert_close(
        y.cpu(), torch.full((16, 16), 64.0, dtype=torch.bfloat16), atol=0, rtol=0
    )


def test_batched_a16wfp4_online_quant():
    require_hcu()
    a = torch.ones((1, 16, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.full((1, 16, 32), 0x22, dtype=torch.uint8, device="cuda")
    scale = torch.full((1, 16, 2), 127, dtype=torch.uint8, device="cuda")
    output = torch.empty((1, 16, 16), dtype=torch.bfloat16, device="cuda")
    y = gemm.batched_gemm_a16w4(a, b, scale, out=output)
    assert y is output
    torch.testing.assert_close(
        y.cpu(), torch.full((1, 16, 16), 64.0, dtype=torch.bfloat16), atol=0, rtol=0
    )


@pytest.mark.parametrize("operation", [gemm.gemm_a16w16, gemm.batched_gemm_bf16])
def test_strided_bias_is_rejected_before_launch(operation):
    require_hcu()
    shape = (3, 32) if operation is gemm.gemm_a16w16 else (2, 3, 32)
    bshape = (5, 32) if operation is gemm.gemm_a16w16 else (2, 5, 32)
    bias_shape = (10,) if operation is gemm.gemm_a16w16 else (2, 1, 10)
    a = torch.empty(shape, dtype=torch.float16, device="cuda")
    b = torch.empty(bshape, dtype=torch.float16, device="cuda")
    bias = torch.empty(bias_shape, dtype=torch.float16, device="cuda")[..., ::2]
    with pytest.raises(ValueError, match="unit stride along N"):
        operation(a, b, bias=bias)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("transpose_bm", [False, True])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.parametrize("output_scale", [None, 2.0])
def test_batched_a16w4_output_layout(
    dtype, split_k, transpose_bm, preallocate, output_scale
):
    require_hcu()
    batch, m, n, k = 2, 3, 5, 128
    rows = torch.tensor([[1, 2, 4], [2, 4, 8]], dtype=dtype)
    weights = torch.tensor([1, 2, 4, 1, 2], dtype=torch.float32)
    packed = torch.tensor([0x22, 0x44, 0x66, 0x22, 0x44], dtype=torch.uint8)
    a = rows[:, :, None].expand(batch, m, k).contiguous().cuda()
    b = packed[None, :, None].expand(batch, n, k // 2).contiguous().cuda()
    scales = torch.full((batch, n, k // 32), 127, dtype=torch.uint8, device="cuda")
    shape = (m, batch, n) if transpose_bm else (batch, m, n)
    output = (
        torch.full(shape, float("nan"), dtype=dtype, device="cuda")
        if preallocate
        else None
    )
    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        NUM_KSPLIT=split_k,
        num_warps=4,
        num_stages=1,
        cache_modifier=".cg",
    )
    before = config.copy()
    result = gemm.batched_gemm_a16w4(
        a,
        b,
        scales,
        dtype=dtype,
        out=output,
        config=config,
        transpose_bm=transpose_bm,
        out_scale=(
            None if output_scale is None else torch.tensor(output_scale, device="cuda")
        ),
    )
    golden = rows.float()[:, :, None] * weights[None, None, :] * k
    if output_scale is not None:
        golden /= output_scale
    if transpose_bm:
        golden = golden.transpose(0, 1).contiguous()
    assert tuple(result.shape) == shape and result.is_contiguous()
    assert config == before
    if preallocate:
        assert result is output
    torch.testing.assert_close(result.cpu(), golden.to(dtype), atol=0, rtol=0)


def test_batched_a16w4_transposed_out_requires_matching_shape():
    require_hcu()
    a = torch.ones((2, 3, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.full((2, 5, 32), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.full((2, 5, 2), 127, dtype=torch.uint8, device="cuda")
    wrong_output = torch.empty((2, 3, 5), dtype=a.dtype, device=a.device)
    with pytest.raises(ValueError, match="preallocated output"):
        gemm.batched_gemm_a16w4(a, b, scales, out=wrong_output, transpose_bm=True)


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_batched_a16w4_output_scale_dtype(scale_dtype):
    require_hcu()
    a = torch.ones((1, 3, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.full((1, 5, 32), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 5, 2), 127, dtype=torch.uint8, device="cuda")
    divisor = torch.tensor([3.0], dtype=scale_dtype, device="cuda")
    result = gemm.batched_gemm_a16w4(a, b, scales, out_scale=divisor)
    torch.testing.assert_close(
        result.cpu(), torch.full((1, 3, 5), 64 / 3, dtype=a.dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        divisor.cpu(), torch.tensor([3.0], dtype=scale_dtype), atol=0, rtol=0
    )


@pytest.mark.parametrize("case", ["python_scalar", "integer", "multiple", "cpu"])
def test_batched_a16w4_invalid_output_scale(case):
    require_hcu()
    a = torch.ones((1, 3, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.full((1, 5, 32), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 5, 2), 127, dtype=torch.uint8, device="cuda")
    invalid = {
        "python_scalar": 2.0,
        "integer": torch.tensor(2, device="cuda"),
        "multiple": torch.ones(2, device="cuda"),
        "cpu": torch.tensor(2.0),
    }[case]
    error = TypeError if case in ("python_scalar", "integer") else ValueError
    with pytest.raises(error, match="out_scale|same GPU"):
        gemm.batched_gemm_a16w4(a, b, scales, out_scale=invalid)
