# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Batched GEMM with independent args/configuration/backend selection."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import BiasGemmBackend
from .validation import validate_input_tensors


def select_backend(backend: str) -> BiasGemmBackend:
    if backend == "gluon":
        raise NotImplementedError("batched_gemm_bf16: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("batched_gemm_bf16: no TileLang kernel is implemented")

    from .triton.batched_gemm_a16w16 import batched_gemm_bf16 as triton_batched_gemm_bf16

    return triton_batched_gemm_bf16


def _validate_operands(a, b, bias) -> None:
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("batched expects rank-three A/B with equal batch counts")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("batched_gemm_bf16 requires FP16/BF16 inputs")
    batch, _, k = a.shape
    n = b.shape[1]
    if b.shape[2] != k or a.dtype != b.dtype:
        raise ValueError("batched A/B must have matching K and dtype")
    if bias is not None and (
        tuple(bias.shape) != (batch, 1, n) or bias.dtype != a.dtype
    ):
        raise ValueError("batched bias must be [B,1,N] in the input dtype")
    validate_input_tensors(a.device, a, b, *(() if bias is None else (bias,)))
    if bias is not None and bias.stride(-1) != 1:
        raise ValueError("bias must have unit stride along N")


def batched_gemm_bf16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    backend=None,
):
    """Compute batched FP16/BF16 ``x[b] @ w[b].T + bias[b]``.

    FP16 and BF16 share this interface despite its historical name.
    Inputs have equal batch counts; batch broadcasting is not supported.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [B,M,K] with positive
            strides. If B > 1 and the batch stride is 1, a matrix dimension
            must also have unit stride.
        w (torch.Tensor): GPU weights [B,N,K] with the same dtype and device
            as x, with matching B and K. The same stride rules as x apply.
        bias (torch.Tensor | None): Optional [B,1,N] bias in x.dtype on
            x.device, with unit stride along N. Broadcast across M.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional [B,M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [B,M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Equal-length grouped GEMM with this same storage contract uses this
        entry. Ragged offsets, indexed tokens and group masks are not supported.
    """
    a, b = (x, w)
    out_dtype = x.dtype if dtype is None else dtype
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, bias)
    if out is None:
        out = torch.empty((a.shape[0], a.shape[1], b.shape[1]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, b, out, bias=bias,
            device=device,
        )
    return out
