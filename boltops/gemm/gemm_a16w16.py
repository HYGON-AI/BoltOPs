# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP16/BF16 dense GEMM with backend selection."""

import torch

from boltops.utility.arch_info import DeviceInfo, get_device_info
from .backend_interfaces import A16W16Backend


def select_backend(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *,
    bias: torch.Tensor | None, activation: str | None,
    backend: str, device: DeviceInfo,
) -> A16W16Backend:
    """Select an implementation for the requested backend and operands."""
    if backend == "gluon":
        raise NotImplementedError("gemm_a16w16: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_a16w16: no TileLang kernel is implemented")

    from .triton.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16

    return triton_gemm_a16w16


def _validate_operands(a, b, bias):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("gemm_a16w16 received an unsupported input dtype")

    device = a.device
    if not a.is_cuda or b.device != device or (bias is not None and bias.device != device):
        raise ValueError("all tensors must be on the same GPU")
    for tensor in (a, b):
        if tensor.numel() and min(tensor.stride()) <= 0:
            raise ValueError("dense tensors must have positive strides")

    if bias is not None:
        if bias.shape != (b.shape[0],) or bias.dtype != a.dtype:
            raise ValueError("bias must have shape [N] and the input dtype")
        if bias.stride(-1) != 1:
            raise ValueError("bias must have unit stride along N")


def gemm_a16w16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    activation=None,
    backend=None,
):
    """Compute dense FP16/BF16 GEMM, optional bias and fused activation.

    Args:
        x (torch.Tensor): GPU input matrix [M,K] in float16 or bfloat16.
        w (torch.Tensor): GPU weight matrix [N,K] on the same device and in the
            same dtype as x. The product is x @ w.T.
        bias (torch.Tensor | None): Optional [N] bias in x.dtype on the same
            device, with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Dtype used when allocating the output.
            Defaults to bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional preallocated [M,N] output on the
            input device. Returned directly when supplied. It must not overlap
            the inputs.
        activation (str | None): Activation applied after the matrix product and
            bias, before output conversion. Supports "gelu", "gelu_tanh",
            "silu", "silu_exp2", "relu", and "relu6". None applies no activation.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, out_dtype = (x, w, x.dtype if dtype is None else dtype)
    backend = "auto" if backend is None else backend
    _validate_operands(a, b, bias)
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(
            a, b, out, bias=bias, activation=activation,
            backend=backend, device=device,
        )
        implementation(
            a, b, out, bias=bias, activation=activation, device=device,
        )
    return out
