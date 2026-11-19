# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from copy import deepcopy

import torch
import triton
import triton.language as tl

from boltops.utility.arch_info import DeviceInfo
from ..config import get_gemm_config
from ..validation import validate_output

from boltops.gemm.triton.kernel_repr import make_kernel_repr
from boltops.gemm.triton.pid_preprocessing import pid_grid, remap_xcd

_gemm_w4a4_repr = make_kernel_repr(
    "_gemm_w4a4_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@triton.jit(repr=_gemm_w4a4_repr)
def _gemm_w4a4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scale_ptrs = (
            a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load scales, masking OOB when K is not aligned to BLOCK_SIZE_K.
            # OOB scales could contain 0xFF (NaN in e8m0) which propagates
            # through dot_scaled even when the corresponding data is zero.
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs)
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                scale_mask = offs_scale_k[None, :] < (
                    2 * K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(a_scale_ptrs, mask=scale_mask, other=127)
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=scale_mask,
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_gemm_w4a4_preshuffle_scales_repr = make_kernel_repr(
    "_gemm_w4a4_preshuffle_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@triton.jit(repr=_gemm_w4a4_preshuffle_scales_repr)
def _gemm_w4a4_kernel_preshuffle_scales(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales

        offs_asn = (
            pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, (BLOCK_SIZE_N // 32))
        ) % N
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr
            + offs_asn[:, None] * stride_bsn
            + offs_ks[None, :] * stride_bsk
        )

        if BLOCK_SIZE_M < 32:
            offs_ks_non_shufl = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
            ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            a_scale_ptrs = (
                a_scales_ptr
                + offs_am[:, None] * stride_asm
                + offs_ks_non_shufl[None, :] * stride_ask
            )
        else:
            offs_asm = (
                pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, (BLOCK_SIZE_M // 32))
            ) % M
            a_scale_ptrs = (
                a_scales_ptr
                + offs_asm[:, None] * stride_asm
                + offs_ks[None, :] * stride_ask
            )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
        offs_shuffled_scale_k = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load scales, masking OOB when K is not aligned to BLOCK_SIZE_K.
            # OOB scales could contain 0xFF (NaN in e8m0) which propagates
            # through dot_scaled even when the corresponding data is zero.
            if EVEN_K:
                if BLOCK_SIZE_M < 32:
                    a_scales = tl.load(a_scale_ptrs)
                else:
                    a_scales = (
                        tl.load(a_scale_ptrs)
                        .reshape(
                            BLOCK_SIZE_M // 32,
                            BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                            4,
                            16,
                            2,
                            2,
                            1,
                        )
                        .permute(0, 5, 3, 1, 4, 2, 6)
                        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                    )
                b_scales = (
                    tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            else:
                shuffled_scale_mask = offs_shuffled_scale_k[None, :] < (
                    2 * K - k * BLOCK_SIZE_K
                )
                if BLOCK_SIZE_M < 32:
                    a_scale_mask = offs_scale_k[None, :] < (
                        2 * K // SCALE_GROUP_SIZE
                        - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                    )
                    a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)
                else:
                    a_scales = (
                        tl.load(a_scale_ptrs, mask=shuffled_scale_mask, other=127)
                        .reshape(
                            BLOCK_SIZE_M // 32,
                            BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                            4,
                            16,
                            2,
                            2,
                            1,
                        )
                        .permute(0, 5, 3, 1, 4, 2, 6)
                        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                    )
                b_scales = (
                    tl.load(
                        b_scale_ptrs,
                        mask=shuffled_scale_mask,
                        other=127,
                        cache_modifier=cache_modifier,
                    )
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2), other=0
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            if BLOCK_SIZE_M < 32:
                a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            else:
                a_scale_ptrs += BLOCK_SIZE_K * stride_ask
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


_gemm_w4a4_preshuffle_repr = make_kernel_repr(
    "_gemm_w4a4_preshuffle_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@triton.jit(repr=_gemm_w4a4_preshuffle_repr)
def _gemm_w4a4_preshuffle_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_shuffle_arr = tl.arange(0, (BLOCK_SIZE_K // 2) * 16)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_k_shuffle = pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_shuffle_arr

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )

        # Create pointers for the first block of A and B scales
        offs_asn = (
            pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, (BLOCK_SIZE_N // 32))
        ) % N
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr
            + offs_asn[:, None] * stride_bsn
            + offs_ks[None, :] * stride_bsk
        )

        if BLOCK_SIZE_M < 32:
            offs_ks_non_shufl = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
            ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            a_scale_ptrs = (
                a_scales_ptr
                + offs_am[:, None] * stride_asm
                + offs_ks_non_shufl[None, :] * stride_ask
            )
        else:
            offs_asm = (
                pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, (BLOCK_SIZE_M // 32))
            ) % M
            a_scale_ptrs = (
                a_scales_ptr
                + offs_asm[:, None] * stride_asm
                + offs_ks[None, :] * stride_ask
            )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if BLOCK_SIZE_M < 32:
                a_scales = tl.load(a_scale_ptrs)
            else:
                a_scales = (
                    tl.load(a_scale_ptrs)
                    .reshape(
                        BLOCK_SIZE_M // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )

            b_scales = (
                tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
                .reshape(
                    BLOCK_SIZE_N // 32,
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                    4,
                    16,
                    2,
                    2,
                    1,
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)

            b = (
                b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 64,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                .trans(1, 0)
            )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * 16 * stride_bk
            if BLOCK_SIZE_M < 32:
                a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            else:
                a_scale_ptrs += BLOCK_SIZE_K * stride_ask
            b_scale_ptrs += BLOCK_SIZE_K * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def gemm_w4a4(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
    a_scale: torch.Tensor, b_scale: torch.Tensor,
    device: DeviceInfo, config: dict[str, object] | None = None,
) -> None:
    m, stored_k = a.shape
    n, k = b.shape[0], stored_k * 2
    validate_output(out, (m, n), a.device, (torch.float16, torch.bfloat16))
    user_config = config
    if device.arch != "gfx946":
        raise NotImplementedError("native FP4 is not supported on this target")
    if user_config is not None:
        config = deepcopy(user_config)
    else:
        config, _ = get_gemm_config(
            "GEMM-W4A4",
            m,
            n,
            k,
            backend="triton",
            arch=device.arch,
            cu_num=device.cu_count,
        )
    from .splitk_reduce import _gemm_splitk_reduce_kernel
    from .common_utils import get_splitk

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
    _gemm_w4a4_kernel[grid](
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
