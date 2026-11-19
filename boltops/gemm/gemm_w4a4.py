# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Prequantized E2M1 GEMM with E8M0 K32 scales."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import MXFP4GemmBackend
from .validation import validate_input_tensors


def select_backend(backend: str) -> MXFP4GemmBackend:
    if backend == "gluon":
        raise NotImplementedError("gemm_w4a4: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_w4a4: no TileLang kernel is implemented")

    from .triton.gemm_w4a4 import gemm_w4a4 as triton_gemm_w4a4

    return triton_gemm_w4a4


def _validate_operands(a, b, a_scale, b_scale) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("MXFP4 expects packed A[M,K/2] and B[N,K/2]")
    m, stored_k = a.shape
    n, k = (b.shape[0], stored_k * 2)
    if (
        k <= 0
        or k % 32
        or any((t.dtype != torch.uint8 for t in (a, b, a_scale, b_scale)))
    ):
        raise ValueError("MXFP4 requires K%32=0 and uint8 E2M1/E8M0 storage")
    if tuple(a_scale.shape) != (m, k // 32) or tuple(b_scale.shape) != (n, k // 32):
        raise ValueError("MXFP4 scales must have shape [M|N,K/32]")
    validate_input_tensors(a.device, a, b, a_scale, b_scale)


def gemm_w4a4(
    x,
    w,
    x_scales,
    w_scales,
    dtype=torch.bfloat16,
    out=None,
    backend=None,
):
    """Compute native MXFP4 GEMM from prequantized E2M1 A/B operands.

    Both operands already include packed FP4 values and E8M0 K32
    scales. No activation quantization is performed by this interface.

    Args:
        x (torch.Tensor): uint8 packed E2M1 input [M,K/2] on GPU, with positive
            byte strides. Earlier and later K elements occupy the low and
            high nibbles. K must be positive and divisible by 32.
        w (torch.Tensor): uint8 packed E2M1 weights [N,K/2] on x.device, with
            positive byte strides and the same packing as x.
        x_scales (torch.Tensor): uint8 E8M0 scales [M,K/32] on x.device with
            positive strides. One scale per input row and 32 K elements.
        w_scales (torch.Tensor): uint8 E8M0 scales [N,K/32] on x.device with
            positive strides. One scale per weight row and 32 K elements.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None also selects bfloat16.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Currently requires gfx946 native FP4 support. Unsupported instructions
        are reported as compiler/runtime failures; no software FP4 GEMM
        replacement is used.
    """
    a, b, a_scale, b_scale, out_dtype = (x, w, x_scales, w_scales, dtype)
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, a_scale, b_scale)
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, b, out, a_scale=a_scale, b_scale=b_scale,
            device=device,
        )
    return out
