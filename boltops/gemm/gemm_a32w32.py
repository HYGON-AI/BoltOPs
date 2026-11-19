# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP32 dense GEMM with backend selection."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import BiasGemmBackend
from .validation import validate_input_tensors


def select_backend(backend: str) -> BiasGemmBackend:
    if backend == "gluon":
        raise NotImplementedError("gemm_a32w32: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_a32w32: no TileLang kernel is implemented")

    from .triton.gemm_a32w32 import gemm_a32w32 as triton_gemm_a32w32

    return triton_gemm_a32w32


def _validate_operands(a, b, bias) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    if a.dtype != torch.float32:
        raise TypeError("gemm_a32w32 received an unsupported input dtype")
    n = b.shape[0]
    if bias is not None and (bias.shape != (n,) or bias.dtype != a.dtype):
        raise ValueError("bias must have shape [N] and the input dtype")
    validate_input_tensors(a.device, a, b, *(() if bias is None else (bias,)))
    if bias is not None and bias.stride(-1) != 1:
        raise ValueError("bias must have unit stride along N")


def gemm_a32w32(x, w, bias=None, dtype=None, out=None, backend=None):
    """Compute dense FP32 ``x @ w.T + bias``.

    Args:
        x (torch.Tensor): float32 GPU input matrix [M,K] with positive strides.
        w (torch.Tensor): float32 GPU weight matrix [N,K] on x.device, with
            positive strides.
        bias (torch.Tensor | None): Optional float32 [N] bias on x.device,
            with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Dtype used to allocate the output. None selects float32, the
            supported output dtype.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in torch.float32.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, out_dtype = (x, w, torch.float32 if dtype is None else dtype)
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, bias)
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, b, out, bias=bias,
            device=device,
        )
    return out
