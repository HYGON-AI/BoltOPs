# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Distinct quantized computations, separate from ordinary native GEMM."""
import operator
import torch
from boltops.utility.arch_info import get_device_info
from .catalog import validate_backend
from .descriptors import GemmProblem, OperandSpec, ScaleSpec
from .registry import launch, select_plan


def _check_device(a, tensors):
    for t in tensors:
        if not t.is_cuda or t.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if not t.is_contiguous():
            raise ValueError("specialized quantized inputs must use the documented contiguous storage")


def _awq_scaled(a, packed_b, b_scale, b_zero_points, *, group_size, backend="auto"):
    """AWQ: A[M,K], packed B[K,N/8], scales[K/G,N], zeros[K/G,N/8].

    Weights/zero points use the existing AWQ nibble order. Output dtype equals
    the scale dtype. No inferred packing, offline dequantization or repacking.
    """
    validate_backend(backend)
    group_size = operator.index(group_size)
    if a.ndim != 2 or packed_b.ndim != 2 or packed_b.shape[0] != a.shape[1]:
        raise ValueError("AWQ expects A[M,K] and packed B[K,N/8]")
    m, k = a.shape
    n = packed_b.shape[1] * 8
    if (k <= 0 or n <= 0 or group_size <= 0 or k % group_size or
            group_size not in (32, 64, 128, k)):
        raise ValueError("AWQ requires positive K/N and a dividing group size of 32/64/128 or K")
    if (a.dtype not in (torch.float16, torch.bfloat16) or b_scale.dtype != a.dtype or
            packed_b.dtype != torch.int32 or (b_zero_points is not None and b_zero_points.dtype != torch.int32)):
        raise TypeError("AWQ requires matching 16-bit A/scales and int32 packed B/zero points")
    if (tuple(b_scale.shape) != (k // group_size, n) or
            (b_zero_points is not None and tuple(b_zero_points.shape) != (k // group_size, n // 8))):
        raise ValueError("AWQ scale/zero-point shapes must match explicit group_size")
    _check_device(a, (a, packed_b, b_scale))
    if b_zero_points is None:
        b_zero_points = torch.zeros((k // group_size, n // 8), dtype=torch.int32, device=a.device)
    _check_device(a, (b_zero_points,))
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=b_scale.dtype, device=a.device)
        if not m:
            return out
        d = get_device_info(a.device)
        p = GemmProblem("dense", m, n, k, OperandSpec(a.dtype, strides=tuple(a.stride())),
            OperandSpec(packed_b.dtype, "int4", "awq", 1, 8,
                        zero_point_dtype=b_zero_points.dtype, strides=tuple(packed_b.stride())),
            out.dtype, d.arch, b_scale=ScaleSpec(b_scale.dtype, (1, group_size)), num_cus=d.cu_count)
        launch(select_plan(p, backend=backend), p, a=a, b=packed_b, b_scale=b_scale,
               b_zero_points=b_zero_points, out=out)
    return out


def _mxfp4_scaled(a, b, a_scale, b_scale, *, out_dtype=None, backend="auto"):
    """Packed E2M1 A[M,K/2], B[N,K/2], uint8 E8M0 scales [M|N,K/32]."""
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("MXFP4 expects packed A[M,K/2] and B[N,K/2]")
    m, stored_k = a.shape
    n, k = b.shape[0], stored_k * 2
    if k <= 0 or k % 32 or any(t.dtype != torch.uint8 for t in (a, b, a_scale, b_scale)):
        raise ValueError("MXFP4 requires K%32=0 and uint8 E2M1/E8M0 storage")
    if tuple(a_scale.shape) != (m, k // 32) or tuple(b_scale.shape) != (n, k // 32):
        raise ValueError("MXFP4 scales must have shape [M|N,K/32]")
    _check_device(a, (a, b, a_scale, b_scale))
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if not m or not n:
            return out
        d = get_device_info(a.device)
        p = GemmProblem("dense", m, n, k,
            OperandSpec(a.dtype, "fp4", "e2m1_low_high", -1, 2, strides=tuple(a.stride())),
            OperandSpec(b.dtype, "fp4", "e2m1_low_high", -1, 2, strides=tuple(b.stride())),
            out_dtype, d.arch, a_scale=ScaleSpec(a_scale.dtype, (1, 32), "e8m0"),
            b_scale=ScaleSpec(b_scale.dtype, (1, 32), "e8m0"), num_cus=d.cu_count)
        launch(select_plan(p, backend=backend), p, a=a, b=b, a_scale=a_scale, b_scale=b_scale, out=out)
    return out


def gemm_batched_a16wfp4_online_quant(a, b, b_scale, *, out_dtype=None, backend="auto"):
    """A[B,M,K] is quantized online to MXFP4; B[B,N,K/2] is already packed.

    This intentionally retains the legacy quantization algorithm. It is not
    exact A16 x FP4 multiplication and does not bypass missing native support.
    """
    validate_backend(backend)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("A16WFP4 expects matching rank-three batches")
    batch, m, k = a.shape
    n = b.shape[1]
    if (k <= 0 or k % 32 or b.shape[2] != k // 2 or b.dtype != torch.uint8 or
            b_scale.dtype != torch.uint8 or tuple(b_scale.shape) != (batch, n, k // 32)):
        raise ValueError("A16WFP4 requires packed uint8 B and E8M0 K32 scales")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A16WFP4 requires FP16/BF16 A")
    _check_device(a, (a, b, b_scale))
    out_dtype = a.dtype if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)
        if not batch or not m or not n:
            return out
        d = get_device_info(a.device)
        p = GemmProblem("batched", m, n, k, OperandSpec(a.dtype, strides=tuple(a.stride())),
            OperandSpec(b.dtype, "fp4", "e2m1_low_high", -1, 2, strides=tuple(b.stride())),
            out_dtype, d.arch, batch=batch, b_scale=ScaleSpec(b_scale.dtype, (1, 32), "e8m0"),
            num_cus=d.cu_count, activation_quantization="mxfp4")
        launch(select_plan(p, backend=backend), p, a=a, b=b, b_scale=b_scale, out=out)
    return out
