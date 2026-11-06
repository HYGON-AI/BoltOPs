# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Batched GEMM with independent problem/configuration/backend selection."""
import torch
from boltops.utility.arch_info import get_device_info
from .catalog import validate_backend
from .descriptors import GemmProblem, OperandSpec
from .registry import launch, select_plan


def gemm_batched(a, b, *, bias=None, out_dtype=None, backend="auto"):
    """A[B,M,K] @ B[B,N,K].T with optional [B,1,N] bias, no broadcasting."""
    validate_backend(backend)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("batched expects rank-three A/B with equal batch counts")
    batch, m, k = a.shape
    n = b.shape[1]
    if b.shape[2] != k or a.dtype != b.dtype:
        raise ValueError("batched A/B must have matching K and dtype")
    if bias is not None and (tuple(bias.shape) != (batch, 1, n) or bias.dtype != a.dtype):
        raise ValueError("batched bias must be [B,1,N] in the input dtype")
    for tensor in (a, b, *(() if bias is None else (bias,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("batched inputs must have positive strides")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((batch, m, n), device=a.device, dtype=out_dtype)
        if not batch or not m or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand_as(out))
            return out
        device = get_device_info(a.device)
        p = GemmProblem("batched", m, n, k,
            OperandSpec(a.dtype, strides=tuple(a.stride())),
            OperandSpec(b.dtype, strides=tuple(b.stride())),
            out_dtype, device.arch, batch=batch, num_cus=device.cu_count)
        launch(select_plan(p, backend=backend), p, a=a, b=b, bias=bias, out=out)
    return out
