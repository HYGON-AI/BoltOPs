# Copyright (c) 2026 Hygon Information Technology Co., Ltd. (modifications).
# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source before relocation: https://github.com/HYGON-AI/BoltOPs/blob/4a179c81d1aa9f12fe5b63bb3caa800ea226c57f/boltops/gemm/triton/triton_scaled_mm.py
"""Triton device kernels for gemm_a8w8."""

from copy import deepcopy

import torch
import triton
import triton.language as tl

from boltops.utility.arch_info import DeviceInfo
from ..config import get_gemm_config
from ..validation import validate_output


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_a,
    stride_scale_b,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am * stride_scale_a
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn * stride_scale_b

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]


def gemm_a8w8(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
    a_scale: torch.Tensor, b_scale: torch.Tensor,
    bias: torch.Tensor | None,
    device: DeviceInfo, config: dict[str, object] | None = None,
) -> None:
    m, k = a.shape
    n = b.shape[0]
    validate_output(out, (m, n), a.device, (torch.float16, torch.bfloat16, torch.float32))
    try:
        a_scale = a_scale.view(-1, 1)
        b_scale = b_scale.view(-1, 1)
    except RuntimeError:
        raise ValueError("whole-K scales must be scalar or flattenable vector views") from None
    user_config = config
    if user_config is not None:
        config = deepcopy(user_config)
    else:
        config, _ = get_gemm_config(
            "GEMM-A8W8",
            m,
            n,
            k,
            backend="triton",
            arch=device.arch,
            cu_num=device.cu_count,
        )

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
