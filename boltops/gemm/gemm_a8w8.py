# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Whole-K tensor/row/channel-scaled native GEMM."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import ScaledGemmBackend
from .validation import validate_input_tensors


def select_backend(backend: str) -> ScaledGemmBackend:
    if backend == "gluon":
        raise NotImplementedError("gemm_a8w8: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_a8w8: no TileLang kernel is implemented")

    from .triton.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8

    return triton_gemm_a8w8


def _validate_operands(a, b, a_scale, b_scale, bias) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or (a.dtype != b.dtype):
        raise ValueError("expected same-dtype A[M,K] and B[N,K]")
    m = a.shape[0]
    n = b.shape[0]
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)
    if a.dtype not in allowed:
        raise TypeError("unsupported input dtype")
    validate_input_tensors(a.device, a, b, a_scale, b_scale)
    if any(
        (
            scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            for scale in (a_scale, b_scale)
        )
    ):
        raise TypeError("numeric scales must be FP16/BF16/FP32")
    if a_scale.numel() not in (1, m) or b_scale.numel() not in (1, n):
        raise ValueError(
            "row scales require one value per tensor or per A row/B output channel"
        )
    if a_scale.dtype != b_scale.dtype:
        raise TypeError("A/B scales must have matching dtype")
    if bias is not None and (
        tuple(bias.shape) != (n,)
        or not bias.is_floating_point()
        or bias.device != a.device
        or (bias.stride(-1) != 1)
    ):
        raise ValueError(
            "bias must be floating point [N] on the input device with unit stride"
        )


def gemm_a8w8(
    x,
    w,
    x_scale,
    w_scale,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    backend=None,
):
    """Compute whole-K tensor/row/channel-scaled FP8 or INT8 GEMM.

    The result is ``(x @ w.T) * x_scale * w_scale + bias``, with scales
    broadcast over the output. Scales are dequantization multipliers and
    are applied after the complete K reduction, not to each K block.

    Args:
        x (torch.Tensor): GPU input [M,K] in float8_e4m3fn, float8_e5m2, or int8,
            with positive strides.
        w (torch.Tensor): GPU weights [N,K] with the same dtype and device as x
            and positive strides.
        x_scale (torch.Tensor): float16, bfloat16, or float32 scales on x.device.
            Supply a scalar or a positive-stride view flattenable to M values,
            one per input row.
        w_scale (torch.Tensor): Scales with the same dtype and device as
            x_scale. Supply a scalar or a positive-stride view flattenable to
            N values, one per output channel.
        bias (torch.Tensor | None): Optional floating-point [N] bias on x.device
            with unit stride. Added after scaling.
        dtype (torch.dtype | None): Allocation dtype (float16, bfloat16, or float32).
            Defaults to bfloat16. None also selects bfloat16.
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
        Scale values are ordinary numbers, not encoded E8M0 bytes. Input
        FP8/INT8 operands are passed to native matrix instructions without
        software upcasting; unsupported target combinations fail compilation.
    """
    a, b, a_scale, b_scale, out_dtype = (x, w, x_scale, w_scale, dtype)
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, a_scale, b_scale, bias)
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, b, out, a_scale=a_scale, b_scale=b_scale, bias=bias,
            device=device,
        )
    return out
