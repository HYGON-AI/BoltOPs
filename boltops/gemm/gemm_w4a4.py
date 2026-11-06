# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Prequantized E2M1 GEMM with E8M0 K32 scales."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_w4a4(
    x,
    w,
    x_scales,
    w_scales,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    backend=None,
):
    """Compute native MXFP4 GEMM from prequantized E2M1 A/B operands.

    Both operands already include packed FP4 values and E8M0 K32
    scales. No activation quantization is performed by this interface.

    Args:
        x (torch.Tensor): uint8 GPU activation [M,K/2], with positive stored byte
            strides. Earlier/later logical K elements occupy the low/high nibble
            of each byte. The carrier may be transposed or strided physically;
            logical K must be positive and divisible by 32.
        w (torch.Tensor): uint8 GPU weights [N,K/2] on x.device with positive stored
            byte strides. Packing matches x; physical transpose/padded views
            are accepted and the logical weight transpose is internal.
        x_scales (torch.Tensor): uint8 [M,K/32] E8M0-encoded GPU scales on x.device,
            with positive matrix strides. Each scale covers 32 logical K elements
            in one activation row; scale bytes are not packed FP4 nibbles.
        w_scales (torch.Tensor): uint8 [N,K/32] E8M0-encoded GPU scales on x.device,
            with positive matrix strides. Each scale covers 32 logical K elements
            in one weight row.
        dtype (torch.dtype | None): Output float16 or bfloat16. Both the default
            and None select bfloat16.
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
        Currently requires gfx946 native FP4 support. Unsupported instructions
        are reported as compiler/runtime failures; no software FP4 GEMM
        replacement is used.
    """
    a, b, a_scale, b_scale, out_dtype = (x, w, x_scales, w_scales, dtype)
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("MXFP4 expects packed A[M,K/2] and B[N,K/2]")
    m, stored_k = a.shape
    n, k = (b.shape[0], stored_k * 2)
    if (
        k <= 0
        or k % 32
        or any((t.dtype != torch.uint8 for t in (a, b, a_scale, b_scale)))
    ):
        raise ValueError("MXFP4 requires K%32=0 and uint8 E2M1/E8M0 storage")
    if tuple(a_scale.shape) != (m, k // 32) or tuple(b_scale.shape) != (n, k // 32):
        raise ValueError("MXFP4 scales must have shape [M|N,K/32]")
    for tensor in (a, b, a_scale, b_scale):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("inputs must have positive strides")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype is not None and out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("gemm_w4a4 received an unsupported output dtype")
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
                    "GEMM-W4A4",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.gemm_w4a4 import (
                _gemm_w4a4_kernel as _triton_gemm_w4a4_kernel,
            )
            from .triton.splitk_reduce import _gemm_splitk_reduce_kernel
            from .triton.common_utils import get_splitk

            if user_config is None:
                config.setdefault("NUM_KSPLIT", 1)
                config["BLOCK_SIZE_K"] = min(
                    config["BLOCK_SIZE_K"], 1 << (k - 1).bit_length()
                )
                config["SPLITK_BLOCK_SIZE"] = k
            x, w, M, N, K = (a, b, m, n, k // 2)
            w, x_scales, w_scales = (w.T, a_scale, b_scale)
            if config["NUM_KSPLIT"] > 1:
                SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
                    K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
                )
                config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
                config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
                config["NUM_KSPLIT"] = NUM_KSPLIT
            if config["BLOCK_SIZE_K"] >= 2 * K:
                config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
                config["SPLITK_BLOCK_SIZE"] = 2 * K
                config.setdefault("NUM_KSPLIT", 1)
            config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)
            if config["NUM_KSPLIT"] > 1:
                y_pp = torch.empty(
                    (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
                )
            else:
                config["SPLITK_BLOCK_SIZE"] = 2 * K
                y_pp = None
            grid = lambda META: (
                META["NUM_KSPLIT"]
                * triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )
            _triton_gemm_w4a4_kernel[grid](
                x,
                w,
                out if config["NUM_KSPLIT"] == 1 else y_pp,
                x_scales,
                w_scales,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
                out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
                out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
                x_scales.stride(0),
                x_scales.stride(1),
                w_scales.stride(0),
                w_scales.stride(1),
                **config,
            )
            if config["NUM_KSPLIT"] > 1:
                REDUCE_BLOCK_SIZE_M = 16
                REDUCE_BLOCK_SIZE_N = 64
                ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"] // 2)
                grid_reduce = (
                    triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
                    triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
                )
                _gemm_splitk_reduce_kernel[grid_reduce](
                    y_pp,
                    out,
                    None,
                    M,
                    N,
                    y_pp.stride(0),
                    y_pp.stride(1),
                    y_pp.stride(2),
                    out.stride(0),
                    out.stride(1),
                    REDUCE_BLOCK_SIZE_M,
                    REDUCE_BLOCK_SIZE_N,
                    ACTUAL_KSPLIT,
                    triton.next_power_of_2(config["NUM_KSPLIT"]),
                    ADD_BIAS=False,
                    activation="",
                    use_activation=False,
                    KERNEL_NAME="_gemm_w4a4_reduce_kernel",
                )
        elif backend == "gluon":
            raise NotImplementedError("gemm_w4a4: no Gluon kernel is implemented")
        elif backend == "tilelang":
            raise NotImplementedError("gemm_w4a4: no TileLang kernel is implemented")
    return out
