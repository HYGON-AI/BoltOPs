# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_bf16.py
"""Vendored aiter Triton batched BF16/FP16 GEMM.

Upstream: ROCm/aiter ``aiter.ops.triton.batched_gemm_bf16``.

Computes ``Y[i] = X[i] @ W[i]^T`` with ``W`` shaped ``[B, N, K]``.


The backend entry resolves the launch configuration.
"""

from __future__ import annotations


from copy import deepcopy

import torch
import triton
import triton.language as tl

from boltops.utility.arch_info import DeviceInfo
from ..config import get_gemm_config
from ..validation import validate_output


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _batched_gemm_a16w16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    BATCH_LAST_GRID: tl.constexpr = False,
):
    """Batched matmul ``C[i] = A[i] @ B[i]`` (+ optional bias). Not for direct use."""

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    # Tensile launches this workload with the tile coordinate in grid.x and
    # batch in grid.z.  Keep the legacy ordering available for existing config
    # files, while allowing tuned shapes to use the same hardware dispatch
    # order as rocBLAS.
    if BATCH_LAST_GRID:
        pid = tl.program_id(axis=0)
        batch_id = tl.program_id(axis=2)
    else:
        batch_id = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        accumulator += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def batched_gemm_bf16(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
    bias: torch.Tensor | None,
    device: DeviceInfo, config: dict[str, object] | None = None,
) -> None:
    batch, m, k = a.shape
    n = b.shape[1]
    validate_output(out, (batch, m, n), a.device, (torch.float16, torch.bfloat16))
    out_dtype = out.dtype
    user_config = config
    if user_config is not None:
        config = deepcopy(user_config)
    else:
        layout = "TN" if b.stride(-1) == 1 else "TT"
        use_tuned_shape = (a.dtype == out_dtype and a.stride(-1) == 1
                          and (b.stride(-1) == 1 or b.stride(-2) == 1))
        config, _ = get_gemm_config(
            "BATCHED_GEMM-A16W16",
            m,
            n if use_tuned_shape else None,
            k if use_tuned_shape else None,
            B=batch if use_tuned_shape else None,
            layout=layout,
            backend="triton",
            arch=device.arch,
            cu_num=device.cu_count,
        )
    config.setdefault("BATCH_LAST_GRID", False)

    for tensor in (a, b):
        if batch > 1 and tensor.stride(0) == 1 and tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
            raise ValueError("batch-only contiguous storage does not provide a contiguous matrix axis")
    b = b.transpose(1, 2)

    def grid(meta):
        tiles = triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(
            n, meta["BLOCK_SIZE_N"]
        )
        return (
            (tiles, 1, batch)
            if meta.get("BATCH_LAST_GRID", False)
            else (batch, tiles)
        )

    _batched_gemm_a16w16_kernel[grid](
        a,
        b,
        out,
        bias if bias is not None else out,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        bias.stride(0) if bias is not None else 1,
        bias is not None,
        **config,
    )
