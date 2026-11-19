# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Batched A16WFP4 GEMM with online activation quantization."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import BatchedA16W4Backend
from .validation import validate_input_tensors


def select_backend(backend: str) -> BatchedA16W4Backend:
    if backend == "gluon":
        raise NotImplementedError("batched_gemm_a16w4: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("batched_gemm_a16w4: no TileLang kernel is implemented")

    from .triton.batched_gemm_a16w4 import batched_gemm_a16w4 as triton_batched_gemm_a16w4

    return triton_batched_gemm_a16w4


def _validate_operands(a, b, b_scale, out_scale) -> None:
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("A16WFP4 expects matching rank-three batches")
    batch, _, k = a.shape
    n = b.shape[1]
    if (
        k <= 0
        or k % 32
        or b.shape[2] != k // 2
        or (b.dtype != torch.uint8)
        or (b_scale.dtype != torch.uint8)
        or (tuple(b_scale.shape) != (batch, n, k // 32))
    ):
        raise ValueError("A16WFP4 requires packed uint8 B and E8M0 K32 scales")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A16WFP4 requires FP16/BF16 A")
    if out_scale is not None:
        if not isinstance(out_scale, torch.Tensor) or not out_scale.is_floating_point():
            raise TypeError("out_scale must be a floating-point tensor")
        if out_scale.numel() != 1:
            raise ValueError("out_scale must contain exactly one value")
    validate_input_tensors(a.device, a, b, b_scale, *(() if out_scale is None else (out_scale,)))


def batched_gemm_a16w4(
    x,
    w,
    w_scales,
    dtype=torch.bfloat16,
    out=None,
    transpose_bm=False,
    out_scale=None,
    backend=None,
):
    """Compute batched A16 x FP4 GEMM with online MXFP4 activation quantization.

    The input x is quantized to E2M1 with E8M0 scales per 32 K elements
    before multiplication by the prequantized weights.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [B,M,K] with positive
            strides. If B > 1 and the batch stride is 1, a matrix dimension
            must also have unit stride. K must be positive and divisible by 32.
        w (torch.Tensor): uint8 packed E2M1 weights [B,N,K/2] on x.device.
            B must match x. The same stride rules as x apply. Earlier and later
            K elements occupy the low and high nibbles.
        w_scales (torch.Tensor): uint8 E8M0 scales [B,N,K/32] on x.device.
            The same stride rules as x apply. One scale per weight row and
            32 K elements.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional output [B,M,N], or [M,B,N] when transpose_bm is True,
            on the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        transpose_bm (bool): Select [M,B,N] output when True. Defaults to
            False for [B,M,N] output.
        out_scale (torch.Tensor | None): Optional single-element floating-point
            tensor on x.device. Divides the result before conversion to dtype.
            None leaves the result unscaled.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [B,M,N], or [M,B,N] when transpose_bm=True result in
        dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Currently requires gfx946 native FP4 support. Config may select split-K
        through NUM_KSPLIT; intermediate partials are FP32. No software FP4 GEMM
        replacement or separate activation-quantization launch is used.

        A known gfx946 PMD/compiler numerical mismatch affects K-contiguous
        FP16/BF16 x with varying K32 weight scales at BM16/BN16/BK64,
        num_warps=4, num_stages=1 and NUM_KSPLIT=1. It reproduces with the
        target-main device kernel; affected numerical tests are strict XFAILs,
        not correctness passes. No software instruction replacement is used.
    """
    a, b, b_scale = (x, w, w_scales)
    out_dtype = x.dtype if dtype is None else dtype
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, b_scale, out_scale)
    batch, m, _ = a.shape
    n = b.shape[1]
    output_shape = (m, batch, n) if transpose_bm else (batch, m, n)
    if out is None:
        out = torch.empty(output_shape, dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, b, out, b_scale=b_scale, out_scale=out_scale, transpose_bm=transpose_bm,
            device=device,
        )
    return out
