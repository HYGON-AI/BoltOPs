# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Whole-K tensor/row/channel-scaled native GEMM."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_a8w8(
    x,
    w,
    x_scale,
    w_scale,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    backend=None,
):
    """Compute whole-K tensor/row/channel-scaled FP8 or INT8 GEMM.

    The result is ``(x @ w.T) * x_scale * w_scale + bias``, with scales
    broadcast over the output. Scales are dequantization multipliers and
    are applied after the complete K reduction, not to each K block.

    Args:
        x (torch.Tensor): GPU activation [M,K], float8_e4m3fn, float8_e5m2 or int8.
            Any positive matrix strides are accepted, including M-contiguous,
            K-contiguous, transposed and padded/sliced views.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Any positive
            matrix strides are accepted independently of x; the logical transpose
            w.T is internal.
        x_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales on x.device.
            One value per tensor or M values, one per row and covering the full K.
            Scalar and flattenable vector views with positive element stride are
            supported; no implicit contiguous copy is made.
        w_scale (torch.Tensor): Numeric GPU scales on x.device, same dtype as x_scale.
            One value per tensor or N values, one per output channel covering
            full K. Scalar and flattenable positive-stride vector views are
            supported.
        bias (torch.Tensor | None): Optional floating-point [N] bias on x.device
            with unit stride, added after scale multiplication.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Both the
            default and None select bfloat16.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
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
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or (a.dtype != b.dtype):
        raise ValueError("expected same-dtype A[M,K] and B[N,K]")
    m, k = a.shape
    n = b.shape[0]
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)
    if a.dtype not in allowed:
        raise TypeError("unsupported input dtype")
    for tensor in (a, b, a_scale, b_scale):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("inputs must have positive strides")
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
    try:
        a_scale = a_scale.view(-1, 1)
        b_scale = b_scale.view(-1, 1)
    except RuntimeError:
        raise ValueError("whole-K scales must be scalar or flattenable vector views") from None
    recipe = (
        (1 if a_scale.numel() == m else max(m, 1), max(k, 1)),
        (1 if b_scale.numel() == n else max(n, 1), max(k, 1)),
    )
    if bias is not None and (
        tuple(bias.shape) != (n,)
        or not bias.is_floating_point()
        or bias.device != a.device
        or (bias.stride(-1) != 1)
    ):
        raise ValueError(
            "bias must be floating point [N] on the input device with unit stride"
        )
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("output must be FP16/BF16/FP32")
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty((m, n), dtype=out_dtype, device=a.device)
        )
        if (
            tuple(out.shape) != (m, n)
            or out.dtype != out_dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not m or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand_as(out))
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                config, _ = get_gemm_config(
                    "GEMM-A8W8",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.gemm_a8w8 import scaled_mm_kernel
            import triton.language as tl

            M, N, K, result = (m, n, k, out)
            b = b.T
            block_size_m, block_size_n, block_size_k = (
                config["BLOCK_SIZE_" + dim] for dim in ("M", "N", "K")
            )
            block_size_sa = 1 if a_scale.numel() == 1 else block_size_m
            block_size_sb = 1 if b_scale.numel() == 1 else block_size_n
            accumulator_dtype = tl.float32 if a.is_floating_point() else tl.int32
            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            )
            scaled_mm_kernel[grid](
                a,
                b,
                a_scale,
                b_scale,
                result,
                bias,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                result.stride(0),
                result.stride(1),
                a_scale.stride(0),
                b_scale.stride(0),
                accumulator_dtype,
                BLOCK_SIZE_M=block_size_m,
                BLOCK_SIZE_N=block_size_n,
                BLOCK_SIZE_K=block_size_k,
                **(
                    {
                        k: v
                        for (k, v) in config.items()
                        if k not in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
                    }
                    if config
                    else {}
                ),
                BLOCK_SIZE_SCALE_A=block_size_sa,
                BLOCK_SIZE_SCALE_B=block_size_sb,
            )
        elif backend == "gluon":
            raise NotImplementedError("gemm_a8w8: no Gluon kernel is implemented")
        elif backend == "tilelang":
            raise NotImplementedError("gemm_a8w8: no TileLang kernel is implemented")
    return out
