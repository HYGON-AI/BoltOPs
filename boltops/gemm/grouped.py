# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Grouped interface over existing native 16-bit and packed FP4 implementations."""
import torch
from boltops.utility.arch_info import get_device_info

from .descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec
from .registry import launch, select_plan
from .catalog import validate_backend
from .scaled import _parse_recipe


def _cdiv(x, y):
    return (x + y - 1) // y


def gemm_grouped(a, b, *, group_offsets=None, a_scale=None, b_scale=None,
                 scale_recipe=((1, 128), (1, 32)),
                 scale_encodings=("numeric", "numeric"), b_format="native",
                 bias=None, out_dtype=None, backend="auto"):
    """Equal A[E,M,K]/B[E,N,K] or offset-grouped A[T,K].

    Packed E2M1 weights use b_format='fp4_e2m1' and B[E,N,ceil(K/2)].
    Offsets must be contiguous int32/int64 [E+1], nondecreasing from 0 to T;
    callers supply already grouped rows. No routing or expert-level tuning runs.
    """
    validate_backend(backend)
    if b_format not in ("native", "fp4_e2m1"):
        raise ValueError("b_format must be native or fp4_e2m1")
    ragged = group_offsets is not None
    if a.ndim != (2 if ragged else 3) or b.ndim != 3:
        raise ValueError("grouped expects A[E,M,K] or A[T,K] with offsets, and rank-three B")
    e, n, stored_k = b.shape
    k = a.shape[-1]
    if e < 1 or (not ragged and a.shape[0] != e):
        raise ValueError("A/B must have matching positive group count")
    packed = b_format == "fp4_e2m1"
    if stored_k != (_cdiv(k, 2) if packed else k):
        raise ValueError("incompatible B storage and logical K")
    t = a.shape[0] if ragged else e * a.shape[1]
    if ragged and (group_offsets.shape != (e + 1,) or
                   group_offsets.dtype not in (torch.int32, torch.int64) or
                   not group_offsets.is_contiguous()):
        raise ValueError("group_offsets must be contiguous int32/int64 [E+1]")
    if (a_scale is None) != (b_scale is None):
        raise ValueError("provide both A/B scales or neither")
    recipe = _parse_recipe(scale_recipe) if a_scale is not None else None
    if (not isinstance(scale_encodings, (tuple, list)) or len(scale_encodings) != 2 or
            any(x not in ("numeric", "e8m0") for x in scale_encodings)):
        raise ValueError("scale_encodings must contain two numeric/e8m0 values")
    if recipe is not None:
        shapes = ((t, _cdiv(k, recipe[0][1])) if ragged else
                  (e, a.shape[1], _cdiv(k, recipe[0][1])),
                  (e, _cdiv(n, recipe[1][0]), _cdiv(k, recipe[1][1])))
        if recipe[0][0] != 1 or tuple(a_scale.shape) != shapes[0] or tuple(b_scale.shape) != shapes[1]:
            raise ValueError("grouped scale shapes must match the explicit recipe")
        for scale, encoding in zip((a_scale, b_scale), scale_encodings):
            if encoding == "e8m0" and scale.dtype != torch.uint8:
                raise ValueError("e8m0 requires uint8 storage")
    if packed and b.dtype != torch.uint8:
        raise TypeError("fp4_e2m1 requires packed uint8 storage")
    tensors = [a, b] + [x for x in (group_offsets, a_scale, b_scale, bias) if x is not None]
    for tensor in tensors:
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("grouped tensors must have positive strides")
    if bias is not None and (packed or bias.shape != (e, 1, n) or bias.dtype != a.dtype):
        raise ValueError("native grouped bias must be [E,1,N] with input dtype; FP4 bias is unsupported")
    out_dtype = (torch.bfloat16 if packed else a.dtype) if out_dtype is None else out_dtype
    out_shape = (t, n) if ragged else (e, a.shape[1], n)
    with torch.cuda.device(a.device):
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
        if not t or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand(out_shape))
            return out
        a_spec = OperandSpec(a.dtype, strides=tuple(a.stride()))
        b_spec = OperandSpec(b.dtype, "fp4", "e2m1_low_high", -1, 2,
                             strides=tuple(b.stride())) if packed else OperandSpec(b.dtype, strides=tuple(b.stride()))
        problem = GemmProblem(
            "grouped", t, n, k, a_spec, b_spec, out_dtype,
            get_device_info(a.device).arch,
            groups=GroupSpec("offsets" if ragged else "equal", e),
            a_scale=None if recipe is None else ScaleSpec(a_scale.dtype, recipe[0], scale_encodings[0]),
            b_scale=None if recipe is None else ScaleSpec(b_scale.dtype, recipe[1], scale_encodings[1]),
            num_cus=get_device_info(a.device).cu_count, has_bias=bias is not None,
        )
        launch(select_plan(problem, backend=backend), problem, a=a, b=b, a_scale=a_scale, b_scale=b_scale,
               group_offsets=group_offsets, bias=bias, out=out)
    return out
