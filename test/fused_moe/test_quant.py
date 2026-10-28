# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for fused MoE quantization helpers."""

import pytest
import torch

pytest.importorskip("triton")

from boltops.fused_moe.triton.quant import (  # noqa: E402
    dynamic_per_token_quant_fp8_i8,
    static_per_tensor_quant_fp8_i8,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
@pytest.mark.parametrize("shape", [(2, 128), (257, 128), (1281, 1232)])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.int8])
def test_dynamic_per_token_quant_zero_rows(shape, quant_dtype):
    """Zero rows stay finite on the 1-D, N=128, and four-row paths."""
    torch.manual_seed(20)
    rows, cols = shape
    x = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    zero_rows = torch.tensor(
        sorted({0, rows // 2, rows - 1}), dtype=torch.long, device=DEVICE
    )
    x[zero_rows] = 0

    quantized = torch.empty(shape, dtype=quant_dtype, device=DEVICE)
    scales = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    dynamic_per_token_quant_fp8_i8(quantized, x, scales)

    dtype_max = (
        torch.finfo(quant_dtype).max
        if torch.is_floating_point(quantized)
        else torch.iinfo(quant_dtype).max
    )
    expected_zero_scale = torch.tensor(
        1.0e-10 / dtype_max, dtype=torch.float32, device=DEVICE
    )

    assert torch.isfinite(scales).all()
    assert torch.isfinite(quantized.float()).all()
    torch.testing.assert_close(
        scales[zero_rows], expected_zero_scale.expand_as(scales[zero_rows])
    )
    torch.testing.assert_close(
        quantized[zero_rows].float(), torch.zeros_like(x[zero_rows].float())
    )


@GPU_REQUIRED
@pytest.mark.parametrize("tensor_name", ["x_in", "qx", "scale_out"])
def test_dynamic_per_token_quant_rejects_noncontiguous_tensors(tensor_name):
    values = torch.randn((4, 128), dtype=torch.float16, device=DEVICE)
    quantized = torch.empty(values.shape, dtype=torch.float8_e4m3fn, device=DEVICE)
    scales = torch.empty(values.shape[0], dtype=torch.float32, device=DEVICE)
    if tensor_name == "x_in":
        values = torch.randn((4, 256), dtype=values.dtype, device=DEVICE)[:, ::2]
    elif tensor_name == "qx":
        quantized = torch.empty((4, 256), dtype=quantized.dtype, device=DEVICE)[
            :, ::2
        ]
    else:
        scales = torch.empty(8, dtype=scales.dtype, device=DEVICE)[::2]

    with pytest.raises(AssertionError, match=f"{tensor_name} must be contiguous"):
        dynamic_per_token_quant_fp8_i8(quantized, values, scales)


def _int8_rounding_input(rows, cols):
    values = torch.zeros((rows, cols), device=DEVICE, dtype=torch.float16)
    values[:, :6] = torch.tensor(
        [0.5, -0.5, 1.5, -1.5, 126.5, 127.0],
        device=DEVICE,
        dtype=torch.float16,
    )
    return values


@GPU_REQUIRED
@pytest.mark.parametrize("shape", [(1, 8), (256, 128)])
def test_dynamic_per_token_int8_rounds_to_nearest(shape):
    values = _int8_rounding_input(*shape)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(shape[0], device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, values.round().to(torch.int8))


@GPU_REQUIRED
def test_dynamic_per_token_int8_tiled_path_rounds_to_nearest():
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    rows = max(1025, 16 * num_cus)
    values = _int8_rounding_input(rows, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(rows, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, values.round().to(torch.int8))


@GPU_REQUIRED
def test_dynamic_per_token_fp8_keeps_fractional_values():
    values = torch.tensor(
        [[0.5, 1.5, 2.5, 448.0]], device=DEVICE, dtype=torch.float16
    )
    quantized = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized.float(), values.float())


@GPU_REQUIRED
def test_static_per_tensor_int8_rounds_to_nearest():
    values = _int8_rounding_input(2, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)

    static_per_tensor_quant_fp8_i8(quantized, values, scale)

    torch.testing.assert_close(quantized, values.round().to(torch.int8))
