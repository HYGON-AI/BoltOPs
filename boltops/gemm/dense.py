# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Unscaled dense GEMM through existing Triton implementations."""
import torch
from boltops.utility.arch_info import get_device_info

from .descriptors import GemmProblem, OperandSpec
from .registry import launch, select_plan
from .catalog import validate_backend


def gemm_dense(a, b, *, bias=None, out_dtype=None, backend="auto"):
    """Compute A[M,K] @ B[N,K].T; positive noncontiguous strides are allowed."""
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    m, k = a.shape
    n = b.shape[0]
    if bias is not None and (bias.shape != (n,) or bias.dtype != a.dtype):
        raise ValueError("bias must have shape [N] and the input dtype")
    for tensor in (a, b, *(() if bias is None else (bias,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("dense tensors must have positive strides")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if not m or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand(m, n))
            return out
        problem = GemmProblem(
            "dense", m, n, k, OperandSpec(a.dtype, strides=tuple(a.stride())),
            OperandSpec(b.dtype, strides=tuple(b.stride())), out_dtype,
            get_device_info(a.device).arch,
            num_cus=get_device_info(a.device).cu_count, has_bias=bias is not None,
            a_aligned_16=a.data_ptr() % 16 == 0, b_aligned_16=b.data_ptr() % 16 == 0,
        )
        launch(select_plan(problem, backend=backend), problem, a=a, b=b, bias=bias, out=out)
    return out
