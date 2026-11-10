# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Batched A16WFP4 GEMM with online activation quantization."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def batched_gemm_a16w4(
    x,
    w,
    w_scales,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    transpose_bm=False,
    out_scale=None,
    backend=None,
):
    """Compute batched A16 x FP4 GEMM with online MXFP4 activation quantization.

    Each K tile of x is quantized inside the GEMM kernel to packed E2M1
    values and E8M0 K32 scales, then multiplied by prequantized w. This
    changes activation values through quantization; it is not exact
    FP16/BF16 x dequantized-FP4 multiplication.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [B,M,K], with positive strides.
            M-contiguous, K-contiguous and padded/strided matrix views are
            supported. Batch-only contiguous storage (only the batch stride is 1)
            is rejected. Logical K must be positive and divisible by 32.
        w (torch.Tensor): uint8 GPU weights [B,N,K/2] on x.device, with positive
            stored byte strides. Earlier/later K elements occupy the low/high
            nibble of each E2M1 byte. Physical transpose/padded matrix views are
            supported; batch count must match x. Batch-only contiguous storage
            is rejected.
        w_scales (torch.Tensor): uint8 [B,N,K/32] E8M0-encoded GPU scales on x.device;
            one scale per weight row and 32 logical K elements. Positive matrix
            strides are supported; batch-only contiguous storage is rejected.
        dtype (torch.dtype | None): Output float16 or bfloat16. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [B,M,N], or [M,B,N] when
            transpose_bm=True, on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        transpose_bm (bool): False writes contiguous [B,M,N]; True writes contiguous
            [M,B,N]. Both direct stores and split-K reduction use output strides
            to write the requested layout without a separate transpose kernel.
        out_scale (torch.Tensor | None): Optional single-element floating-point GPU
            tensor on x.device, shared by all batches and output elements.
            Divide FP32 accumulators by this scalar before output conversion.
            Split-K applies it to each FP32 partial before reduction. None
            leaves the result unscaled. It does not select output quantization
            or change dtype, and the caller tensor is not modified.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [B,M,N], or [M,B,N] when transpose_bm=True result in
        dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Currently requires gfx946 native FP4 support. Config may select split-K
        through NUM_KSPLIT; intermediate partials are FP32. No software FP4 GEMM
        replacement or separate activation-quantization launch is used.

        A known gfx946 PMD/compiler numerical mismatch affects K-contiguous
        FP16/BF16 x with varying K32 weight scales at BM16/BN16/BK64,
        num_warps=4, num_stages=1 and NUM_KSPLIT=1. It reproduces with the
        target-main device kernel; affected numerical tests are strict XFAILs,
        not correctness passes. No software instruction replacement is used.
    """
    a, b, b_scale = (x, w, w_scales)
    out_dtype = x.dtype if dtype is None else dtype
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("A16WFP4 expects matching rank-three batches")
    batch, m, k = a.shape
    n = b.shape[1]
    if (
        k <= 0
        or k % 32
        or b.shape[2] != k // 2
        or (b.dtype != torch.uint8)
        or (b_scale.dtype != torch.uint8)
        or (tuple(b_scale.shape) != (batch, n, k // 32))
    ):
        raise ValueError("A16WFP4 requires packed uint8 B and E8M0 K32 scales")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A16WFP4 requires FP16/BF16 A")
    if out_scale is not None:
        if not isinstance(out_scale, torch.Tensor) or not out_scale.is_floating_point():
            raise TypeError("out_scale must be a floating-point tensor")
        if out_scale.numel() != 1:
            raise ValueError("out_scale must contain exactly one value")
    for tensor in (a, b, b_scale, *(() if out_scale is None else (out_scale,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("inputs must have positive strides")
    for tensor in (a, b, b_scale):
        if batch > 1 and tensor.stride(0) == 1 and tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
            raise ValueError("batch-only contiguous storage does not provide a contiguous matrix axis")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if out_dtype is not None and out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("batched_gemm_a16w4 received an unsupported output dtype")
    output_shape = (m, batch, n) if transpose_bm else (batch, m, n)
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty(output_shape, dtype=out_dtype, device=a.device)
        )
        if (
            tuple(out.shape) != output_shape
            or out.dtype != out_dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not batch or not m or (not n):
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if device.arch != "gfx946":
                raise NotImplementedError("native FP4 is not supported on this target")
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                config, _ = get_gemm_config(
                    "BATCHED_GEMM-A16W4",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.batched_gemm_a16w4 import (
                _batched_gemm_a16w4_kernel,
                _batched_gemm_a16w4_reduce_kernel,
            )
            from .triton.common_utils import get_splitk

            if user_config is None:
                config.setdefault("NUM_KSPLIT", 1)
                config["BLOCK_SIZE_K"] = min(
                    config["BLOCK_SIZE_K"], 1 << (k - 1).bit_length()
                )
                config["SPLITK_BLOCK_SIZE"] = k
            x, w, M, N, K = (a, b, m, n, k // 2)
            B, w_scales = (batch, b_scale)
            if config["NUM_KSPLIT"] > 1:
                SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
                    K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
                )
                config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
                config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
                config["NUM_KSPLIT"] = NUM_KSPLIT
                y_pp = torch.empty(
                    (B, config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=out.device
                )
            else:
                config["SPLITK_BLOCK_SIZE"] = 2 * K
                y_pp = None
            if config["BLOCK_SIZE_K"] >= 2 * K:
                config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
                config["SPLITK_BLOCK_SIZE"] = 2 * K
            if config["NUM_KSPLIT"] == 1:
                stride_ck = 0
                stride_cn = out.stride(2)
                stride_cb = out.stride(1 if transpose_bm else 0)
                stride_cm = out.stride(0 if transpose_bm else 1)
            else:
                stride_cb = y_pp.stride(0)
                stride_ck = y_pp.stride(1)
                stride_cm = y_pp.stride(2)
                stride_cn = y_pp.stride(3)
            grid = lambda META: (
                B,
                META["NUM_KSPLIT"]
                * triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )
            _batched_gemm_a16w4_kernel[grid](
                x,
                w,
                out if config["NUM_KSPLIT"] == 1 else y_pp,
                w_scales,
                out_scale,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                w.stride(0),
                w.stride(1),
                w.stride(2),
                stride_cb,
                stride_ck,
                stride_cm,
                stride_cn,
                w_scales.stride(0),
                w_scales.stride(1),
                w_scales.stride(2),
                PRE_QUANT=True,
                HAVE_Y_SCALE=(out_scale is not None),
                **config,
            )
            if config["NUM_KSPLIT"] > 1:
                REDUCE_BLOCK_SIZE_M = 16
                REDUCE_BLOCK_SIZE_N = 64
                ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"] // 2)
                grid_reduce = (
                    B,
                    triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
                    triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
                )
                _batched_gemm_a16w4_reduce_kernel[grid_reduce](
                    y_pp,
                    out,
                    M,
                    N,
                    y_pp.stride(0),
                    y_pp.stride(1),
                    y_pp.stride(2),
                    y_pp.stride(3),
                    out.stride(1 if transpose_bm else 0),
                    out.stride(0 if transpose_bm else 1),
                    out.stride(2),
                    REDUCE_BLOCK_SIZE_M,
                    REDUCE_BLOCK_SIZE_N,
                    ACTUAL_KSPLIT,
                    config["NUM_KSPLIT"],
                )
        elif backend == "gluon":
            raise NotImplementedError(
                "batched_gemm_a16w4: no Gluon kernel is implemented"
            )
        elif backend == "tilelang":
            raise NotImplementedError(
                "batched_gemm_a16w4: no TileLang kernel is implemented"
            )
    return out
