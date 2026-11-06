# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Native block-scaled dense GEMM with independent A/B scale blocks."""
import operator

import torch
from boltops.utility.arch_info import get_device_info

from .registry import launch, select_plan
from .catalog import validate_backend
from .descriptors import GemmProblem, OperandSpec, ScaleSpec


def _cdiv(x, y):
    return (x + y - 1) // y


def _parse_recipe(scale_recipe):
    try:
        recipe = tuple(tuple(operator.index(x) for x in block) for block in scale_recipe)
        if len(recipe) != 2 or any(len(block) != 2 or any(x <= 0 for x in block) for block in recipe):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("scale_recipe must be two positive integer pairs: ((A_M,A_K),(B_N,B_K))") from None
    return recipe


def gemm_scaled(a, b, a_scale=None, b_scale=None, *,
                      scale_recipe=((1, 128), (128, 128)),
                      scale_encodings=("numeric", "numeric"), out_dtype=None, a_format="native", b_format="native",
                      b_zero_points=None, bias=None, backend="auto"):
    """Compute block-scaled A[M,K] @ B[N,K].T using native dot operations.

    Scale shapes are [ceil(M/A_M),ceil(K/A_K)] and
    [ceil(N/B_N),ceil(K/B_K)]. Blocks may differ; K-group is contiguous.
    Scale encoding is explicit per input: "numeric" (default) or "e8m0".
    E8M0 requires uint8 storage; uint8 remains numeric by default.
    Input/accumulator legality is checked by the selected target compiler;
    unsupported combinations raise compiler errors, without input upcasting.
    AWQ uses a_scale=None, b_format="awq_int4", explicit per-channel B
    recipe and optional packed b_zero_points (defaults to zero). Its B/scale/
    zero-point storage is [K,N/8], [K/G,N], [K/G,N/8]. MXFP4 sets both
    formats to "fp4_e2m1" with recipe ((1,32),(1,32)) and E8M0 encodings.
    Output dtype defaults to BF16 (AWQ: B scale dtype). Scale
    arithmetic uses FP32, promoted to FP64 for FP64 inputs or scales.
    Automatic selection consults a versioned cross-backend performance map,
    then registered general implementations. An explicit backend never falls back.
    """
    validate_backend(backend)
    recipe = _parse_recipe(scale_recipe)
    if (not isinstance(scale_encodings, (tuple, list)) or len(scale_encodings) != 2 or
            any(e not in ("numeric", "e8m0") for e in scale_encodings)):
        raise ValueError('scale_encodings must be a pair of "numeric" or "e8m0"')
    if b_format == "awq_int4":
        if (a_format != "native" or a_scale is not None or recipe[1][0] != 1 or
                tuple(scale_encodings) != ("numeric", "numeric") or bias is not None):
            raise ValueError("AWQ requires native unscaled A, per-channel B scales, numeric encoding and no bias")
        if b_scale is None or (out_dtype is not None and out_dtype != b_scale.dtype):
            raise ValueError("AWQ output dtype must equal B scale dtype")
        from .specialized import _awq_scaled
        return _awq_scaled(a, b, b_scale, b_zero_points, group_size=recipe[1][1], backend=backend)
    if a_format == "fp4_e2m1":
        if (b_format != "fp4_e2m1" or recipe != ((1, 32), (1, 32)) or
                tuple(scale_encodings) != ("e8m0", "e8m0") or b_zero_points is not None or bias is not None):
            raise ValueError("MXFP4 requires E2M1 A/B, E8M0 recipe ((1,32),(1,32)), no zero points/bias")
        if a_scale is None or b_scale is None:
            raise ValueError("prequantized MXFP4 requires both scales")
        from .specialized import _mxfp4_scaled
        return _mxfp4_scaled(a, b, a_scale, b_scale, out_dtype=out_dtype, backend=backend)
    if a_format != "native" or b_format not in ("native", "fp4_e2m1") or b_zero_points is not None:
        raise ValueError("unsupported operand format or zero-point contract")
    if a_scale is None or b_scale is None:
        raise ValueError("this scaled implementation requires both A/B scales")
    packed = b_format == "fp4_e2m1"
    if packed and b.dtype != torch.uint8:
        raise TypeError("fp4_e2m1 requires packed uint8 storage")
    if bias is not None and (tuple(bias.shape) != (b.shape[0],) or not bias.is_floating_point()):
        raise ValueError("scaled bias must be floating point [N]")
    if bias is not None and (not bias.is_cuda or bias.device != a.device):
        raise ValueError("bias must be on the input GPU")
    for tensor, encoding in zip((a_scale, b_scale), scale_encodings):
        if encoding == "e8m0" and tensor.dtype != torch.uint8:
            raise ValueError("e8m0 scale encoding requires uint8 storage")
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != (_cdiv(a.shape[1], 2) if packed else a.shape[1]):
        raise ValueError("expected A[M,K] and B[N,K] with matching K")
    m, k = a.shape
    n = b.shape[0]
    (am, ak), (bn, bk) = recipe
    expected = (("a", a, (m, k)), ("b", b, (n, _cdiv(k, 2) if packed else k)),
                ("a_scale", a_scale, (_cdiv(m, am), _cdiv(k, ak))),
                ("b_scale", b_scale, (_cdiv(n, bn), _cdiv(k, bk))))
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape} for scale_recipe={recipe}, got {tuple(tensor.shape)}")
    for _, tensor, _ in expected:
        if tensor.device != a.device or not tensor.is_cuda:
            raise ValueError("inputs must be on the same GPU")
        if packed:
            if tensor.numel() and any(s <= 0 for s in tensor.stride()):
                raise ValueError("packed GEMM inputs must have positive strides")
        elif not tensor.is_contiguous():
            raise ValueError("native scaled inputs must be contiguous")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if m == 0 or n == 0:
            return out
        if k == 0:
            out.zero_() if bias is None else out.copy_(bias.expand_as(out))
        else:
            problem = GemmProblem(
                kind="dense", m=m, n=n, k=k, a=OperandSpec(a.dtype, strides=tuple(a.stride())), b=(OperandSpec(b.dtype, "fp4", "e2m1_low_high", -1, 2, strides=tuple(b.stride()))
                    if packed else OperandSpec(b.dtype, strides=tuple(b.stride()))),
                has_bias=bias is not None, out_dtype=out_dtype, arch=get_device_info(a.device).arch,
                num_cus=get_device_info(a.device).cu_count,
                a_scale=ScaleSpec(a_scale.dtype, recipe[0], scale_encodings[0]),
                b_scale=ScaleSpec(b_scale.dtype, recipe[1], scale_encodings[1]),
                a_aligned_16=a.data_ptr() % 16 == 0, b_aligned_16=b.data_ptr() % 16 == 0,
            )
            launch(select_plan(problem, backend=backend), problem, a=a, b=b, a_scale=a_scale, b_scale=b_scale, out=out, **({"bias": bias} if bias is not None else {}))
    return out
