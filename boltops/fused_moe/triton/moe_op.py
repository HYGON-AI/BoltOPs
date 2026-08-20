# SPDX-License-Identifier: MIT

import functools
import json
import os
import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, List
from boltops.utility import logger
import boltops.utility.arch_info as arch_info
from boltops.fused_moe.triton.moe_config_utils import get_optimal_moe_config_func

@functools.lru_cache()
def distinguish_moe_kernel_name():
    training_mode = os.environ.get("TRAINING_MODE", "0") == "1"
    return not training_mode

@functools.lru_cache()
def support_mls():
    return arch_info.get_arch() in ("gfx938")

capMLS = support_mls()
splitk_size = int(os.environ.get("SPLITK_SIZE", "0"))


# def get_splitk_reduce_config(M, N, SPLIT_K):
#     """Get optimal configuration for splitk reduce kernel"""
#     if M < 32:
#         config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, "num_warps": 4}
#     elif M < 128:
#         config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, "num_warps": 8}
#     else:
#         config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, "num_warps": 8}
#     return config

# def generate_splitk_reduce_configs():
#     configs = []
#     for block_m in [1, 2, 4, 16]:
#         for block_n in [64, 128, 256]:
#             for num_warps in [2, 4]:
#                 for num_stages in [1, 2]:
#                     config = triton.Config({
#                         'BLOCK_SIZE_M': block_m,
#                         'BLOCK_SIZE_N': block_n,
#                     }, num_warps=num_warps, num_stages=num_stages)
#                 configs.append(config)
#     return configs

# @triton.autotune(
#     key=['M', 'N', 'top_k','compute_type'],
#     configs=generate_splitk_reduce_configs(),
#     perf_debug=True,
# )
# @triton.heuristics({
#     'block_m_dividable': lambda nargs: nargs['M'] % nargs['BLOCK_SIZE_M'] == 0,
#     'block_n_dividable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
# })
# @triton.jit
# def splitk_reduce_kernel(
#     # Pointers to matrices
#     output_ptr,             # [M, N]
#     input_ptr,              # [SPLIT_K, M, N]
#     # Matrix dimensions
#     M,
#     N: tl.constexpr,
#     SPLIT_K: tl.constexpr,
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     stride_k,
#     stride_m,
#     stride_n,
#     compute_type: tl.constexpr,
#     block_m_dividable: tl.constexpr,
#     block_n_dividable: tl.constexpr,
# ):
#     """
#     Reduce splitk_cache along the first dimension (SPLIT_K dimension).

#     Args:
#         output_ptr: shape [M, N]
#         input_ptr: shape [SPLIT_K, M, N]
#     """
#     tl.assume(stride_k >= 0)
#     tl.assume(stride_m >= 0)
#     tl.assume(stride_n >= 0)

#     pid = tl.program_id(axis=0)

#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#     mask_m = offs_m < M
#     mask_n = offs_n < N

#     # Accumulate in float32 for higher precision
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     # Sum over SPLIT_K dimension
#     for k in range(SPLIT_K):
#         input_ptrs = input_ptr + (k * stride_k +
#                                  offs_m[:, None] * stride_m +
#                                  offs_n[None, :] * stride_n)
#         if block_m_dividable and block_n_dividable:
#             x = tl.load(input_ptrs)
#         elif block_n_dividable:
#             x = tl.load(input_ptrs, mask=mask_n[None, :], other=0.0)
#         elif block_m_dividable:
#             x = tl.load(input_ptrs, mask=mask_m[:, None], other=0.0)
#         else:
#             x = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
#         acc += x.to(tl.float32)

#     # Convert to target compute type
#     acc = acc.to(compute_type)

#     output_ptrs = output_ptr + (offs_m[:, None] * stride_m +
#                                 offs_n[None, :] * stride_n)

#     if block_m_dividable and block_n_dividable:
#         tl.store(output_ptrs, acc)
#     elif block_n_dividable:
#         tl.store(output_ptrs, acc, mask=mask_n[None, :])
#     elif block_m_dividable:
#         tl.store(output_ptrs, acc, mask=mask_m[:, None])
#     else:
#         tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# def triton_splitk_reduce(input_tensor, output_tensor):
#     """
#     High-performance triton kernel for reducing splitk_cache.

#     Args:
#         input_tensor: [SPLIT_K, M, N] - splitk_cache
#         output_tensor: [M, N] - output tensor C
#     """
#     SPLIT_K, M, N = input_tensor.shape

#     config = get_splitk_reduce_config(M, N, SPLIT_K)
#     grid = (triton.cdiv(M, config['BLOCK_SIZE_M']) * triton.cdiv(N, config['BLOCK_SIZE_N']),)

#     grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
#                                      N, META['BLOCK_SIZE_N']), )

#     # Check constraints
#     assert output_tensor.dtype == torch.float16 or \
#            output_tensor.dtype == torch.bfloat16 or \
#            output_tensor.dtype == torch.float32

#     if output_tensor.dtype == torch.float16:
#         compute_type = tl.float16
#     elif output_tensor.dtype == torch.bfloat16:
#         compute_type = tl.bfloat16
#     elif output_tensor.dtype == torch.float32:
#         compute_type = tl.float32
#     else:
#         compute_type = tl.float32  # Default to float32

#     assert input_tensor.is_contiguous()
#     assert output_tensor.is_contiguous()
#     assert input_tensor.shape[1] == output_tensor.shape[0]
#     assert input_tensor.shape[2] == output_tensor.shape[1]

#     splitk_reduce_kernel[grid](
#         output_tensor,
#         input_tensor,
#         M,
#         N,
#         SPLIT_K,
#         stride_k = input_tensor.stride(0),
#         stride_m = input_tensor.stride(1),
#         stride_n = input_tensor.stride(2),
#         compute_type=compute_type,
#         **config,
#     )

#     return output_tensor


@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
        None, :].to(tl.int64))
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 4):
    ## pid remapping on xcds

    # Optimization: when NUM_XCDS=1, no remapping is needed
    if NUM_XCDS == 1:
        return pid

    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def decode_sorted_token_id(
    encoded_token_id,
    ck_sorting: tl.constexpr,
    ck_topk: tl.constexpr,
):
    """Return the flattened route id encoded by the selected sorter.

    CK sorting stores the token id in the low 24 bits and the route id in
    the high 8 bits.  For topK=1 the route id is necessarily zero, so avoid
    extracting and combining a value that cannot affect the result.
    """
    if ck_sorting:
        token_id = encoded_token_id & 0x00FFFFFF
        if ck_topk == 1:
            return token_id
        route_id = (encoded_token_id >> 24) & 0xFF
        return token_id * ck_topk + route_id
    return encoded_token_id


@triton.jit
def source_token_id(flattened_route_id, top_k: tl.constexpr):
    """Map a flattened route id to its source activation row."""
    if top_k == 1:
        return flattened_route_id
    return flattened_route_id // top_k


def _fused_moe_kernel_gptq_awq_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return "fused_moe_kernel_gptq_awq_bot" if mul_routed_weight else "fused_moe_kernel_gptq_awq"
    else:
        return "fused_moe_kernel_gptq_awq"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
    "num_groups": lambda nargs: triton.cdiv(nargs["BLOCK_SIZE_K"], nargs["group_size"]),
    'group_size_divisible': lambda nargs: nargs['BLOCK_SIZE_K'] % nargs['group_size'] == 0,
})
@triton.jit(repr=_fused_moe_kernel_gptq_awq_repr)
def fused_moe_kernel_gptq_awq(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        sorted_weights_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        group_size: tl.constexpr,
        group_size_divisible: tl.constexpr,
        num_groups: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        COMBINE_SCALE_LOAD: tl.constexpr,
        USE_MLS_LOAD: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_ADDR_OFFSET_INT64_A: tl.constexpr,
        USE_ADDR_OFFSET_INT64_B: tl.constexpr,
        USE_ADDR_OFFSET_INT64_C: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        ck_sorting: tl.constexpr,
        ck_topk: tl.constexpr,
        NUM_XCDS: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)
    tl.assume(stride_bze >= 0)
    tl.assume(stride_bzk >= 0)
    tl.assume(stride_bzn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    tl.static_assert(COMBINE_SCALE_LOAD == False, "COMBINE_SCALE_LOAD not support for awq!")
    tl.static_assert(USE_MLS_LOAD == False, "USE_MLS_LOAD not support yet for awq!")

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
    source_token = source_token_id(offs_token, top_k)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.

        # c_ptr will be a zero inited buffer, no need to write zero explicitly
        # write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
        #                       offs_token, token_mask, BLOCK_SIZE_M,
        #                       BLOCK_SIZE_N, compute_type)
        return

    tl.assume(off_experts >= 0)

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N)).to(tl.int32) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int32)


    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (source_token[:, None].to(tl.int64) * stride_am +
                        offs_k[None, :].to(tl.int64) * stride_ak)
    else:
        a_ptrs = a_ptr + (source_token[:, None] * stride_am +
                        offs_k[None, :] * stride_ak).to(tl.int32)

    if use_int4_w4a16:
        if group_size_divisible and has_zp:
            offs_k_continue = tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int32)
            if USE_ADDR_OFFSET_INT64_B:
                b_ptrs = b_ptr + (
                    off_experts.to(tl.int64) * stride_be
                    + offs_bn[:, None].to(tl.int64) * stride_bn
                    + offs_k_continue[None, :].to(tl.int64) * stride_bk
                )
            else:
                b_ptrs = b_ptr + (
                    off_experts * stride_be
                    + offs_bn[:, None] * stride_bn
                    + offs_k_continue[None, :] * stride_bk
                ).to(tl.int32)
        else:
            if USE_ADDR_OFFSET_INT64_B:
                b_ptrs = b_ptr + (
                    off_experts.to(tl.int64) * stride_be
                    + (offs_k[:, None].to(tl.int64) // 2) * stride_bk
                    + offs_bn[None, :].to(tl.int64) * stride_bn
                )
            else:
                b_ptrs = b_ptr + (
                    off_experts * stride_be
                    + (offs_k[:, None] // 2) * stride_bk
                    + offs_bn[None, :] * stride_bn
                ).to(tl.int32)
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        if USE_ADDR_OFFSET_INT64_B:
            b_ptrs = b_ptr + (
                off_experts.to(tl.int64) * stride_be
                + offs_k[:, None].to(tl.int64) * stride_bk
                + offs_bn[None, :].to(tl.int64) * stride_bn
            )
        else:
            b_ptrs = b_ptr + (
                off_experts * stride_be
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            ).to(tl.int32)

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # w4a16 deepseek case
    if use_int4_w4a16 and (group_size_divisible and has_zp):
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            offs_szk = k * num_groups + tl.arange(0, num_groups)

            if not block_k_diviable:
                k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
                k_other = 0.0

                masks_szk = offs_szk[:, None] < K // group_size
                masks_a   = token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
            else:
                k_mask = None
                k_other = None

                masks_szk = None
                masks_a   = token_mask[:, None]

            a = tl.load(a_ptrs,
                        mask=masks_a,
                        other=0.0)
            b = tl.load(b_ptrs)
            if use_int4_w4a16:
                b = tl.interleave(b, b)
                b = tl.trans(b)

                b = (b >> b_shifter) & 0xF

            b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + \
                offs_bn[None, :] * stride_bsn + \
                offs_szk[:, None] * stride_bsk).to(tl.int32)
            if not block_k_diviable:
                b_scale = tl.load(b_scale_ptrs, mask=masks_szk, other=k_other)
            else:
                b_scale = tl.load(b_scale_ptrs)
            b_scale = b_scale.to(tl.float32)

            b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                (offs_bn[None, :]//2) * stride_bzn + \
                offs_szk[:, None] * stride_bzk).to(tl.int32)
            if not block_k_diviable:
                b_zp = tl.load(b_zp_ptrs, mask=masks_szk, other=k_other)
            else:
                b_zp = tl.load(b_zp_ptrs)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)


            if num_groups == 1:
                # Original efficient implementation for single group
                b_scale = tl.broadcast_to(b_scale, (BLOCK_SIZE_K, BLOCK_SIZE_N))
                b_zp = tl.broadcast_to(b_zp, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            else:
                # Reshape to (num_groups, 1, N) then broadcast to (num_groups, group_size_in_block, N)
                b_scale = tl.broadcast_to(b_scale[:, None, :], (num_groups, group_size, BLOCK_SIZE_N))
                b_scale = tl.reshape(b_scale, (BLOCK_SIZE_K, BLOCK_SIZE_N))
                b_zp = tl.broadcast_to(b_zp[:, None, :], (num_groups, group_size, BLOCK_SIZE_N))
                b_zp = tl.reshape(b_zp, (BLOCK_SIZE_K, BLOCK_SIZE_N))

            # We accumulate along the K dimension.
            if has_zp:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)

            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            if use_int4_w4a16:
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            else:
                b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.

            if not block_k_diviable:
                # Keep the same partial-K masking semantics as SGLang gptq_awq kernel.
                k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
                k_other = 0.0
            else:
                k_mask = None
                k_other = None

            a = tl.load(a_ptrs,
                        mask=token_mask[:, None] &
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0)
            b = tl.load(b_ptrs)
            if use_int4_w4a16:
                b = (b >> b_shifter) & 0xF

            b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + \
                offs_bn[None, :] * stride_bsn + \
                ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * \
                    stride_bsk).to(tl.int32)
            if not block_k_diviable:
                b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
            else:
                b_scale = tl.load(b_scale_ptrs)
            b_scale = b_scale.to(tl.float32)

            if has_zp and use_int4_w4a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                    (offs_bn[None, :] // 2) * stride_bzn + \
                    offs_k_true * stride_bzk).to(tl.int32)
                if not block_k_diviable:
                    b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                else:
                    b_zp = tl.load(b_zp_ptrs)
                b_zp = ((b_zp >> b_zp_shifter) & 0xF)
                b_zp = b_zp.to(tl.float32)
            elif has_zp and use_int8_w8a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                    offs_bn[None, :] * stride_bzn + \
                    offs_k_true * stride_bzk).to(tl.int32)
                if not block_k_diviable:
                    b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                else:
                    b_zp = tl.load(b_zp_ptrs)
                b_zp = b_zp.to(tl.float32)

            # We accumulate along the K dimension.
            if has_zp:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)

            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            if use_int4_w4a16:
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            else:
                b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if USE_ADDR_OFFSET_INT64_C:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
            None, :].to(tl.int64))
    else:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int32)

    if block_n_diviable:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _fused_moe_kernel_gptq_awq_w4a8_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return "fused_moe_kernel_gptq_awq_w4a8_bot" if mul_routed_weight else "fused_moe_kernel_gptq_awq_w4a8"
    else:
        return "fused_moe_kernel_gptq_awq_w4a8"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
    "num_groups": lambda nargs: triton.cdiv(nargs["BLOCK_SIZE_K"], nargs["group_size"]),
    'group_size_divisible': lambda nargs: nargs['BLOCK_SIZE_K'] % nargs['group_size'] == 0,
})
@triton.jit(repr=_fused_moe_kernel_gptq_awq_w4a8_repr)
def fused_moe_kernel_gptq_awq_w4a8(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        sorted_weights_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        group_k: tl.constexpr,                 # a quant group size: ie. 128
        block_k_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        group_size: tl.constexpr,              # b w4 group size: ie. 64
        group_size_divisible: tl.constexpr,
        num_groups: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        COMBINE_SCALE_LOAD: tl.constexpr,
        USE_MLS_LOAD: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_ADDR_OFFSET_INT64_A: tl.constexpr,
        USE_ADDR_OFFSET_INT64_C: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int4_w4a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        ck_sorting: tl.constexpr,
        ck_topk: tl.constexpr,
        NUM_XCDS: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_asm >= 0)
    tl.assume(stride_ask >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)
    tl.assume(stride_bze >= 0)
    tl.assume(stride_bzk >= 0)
    tl.assume(stride_bzn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    tl.static_assert(use_int4_w4a8 == True, "Must use int4_w4a8")
    tl.static_assert(has_zp == True, "only for deepseek w4a8 case.")
    tl.static_assert(group_k > 0, "group_k must be greater than 0 for deepseek w4a8 case.")
    tl.static_assert(block_k_diviable == True, "block_k_diviable must be True for deepseek w4a8 case.")
    tl.static_assert(BLOCK_SIZE_K <= group_k and group_k % BLOCK_SIZE_K == 0,
        "BLOCK_SIZE_K must be divisible by GROUP_SIZE_K")
    tl.static_assert(group_size_divisible == True, "group_size_divisible must be True for deepseek w4a8 case.")
    tl.static_assert(BLOCK_SIZE_K == group_size, "BLOCK_SIZE_K must be equal to group_size for deepseek w4a8 case.")
    tl.static_assert(group_size == group_k, "group_size must be equal to group_k for deepseek w4a8 case.")
    tl.static_assert(USE_MLS_LOAD == False, "USE_MLS_LOAD must be False due to not supported yet.")

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
    source_token = source_token_id(offs_token, top_k)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        # c_ptr will be a zero inited buffer, no need to write zero explicitly
        # write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
        #                       offs_token, token_mask, BLOCK_SIZE_M,
        #                       BLOCK_SIZE_N, compute_type)
        return

    tl.assume(off_experts >= 0)

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N)).to(tl.int32) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int32)


    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (source_token[:, None].to(tl.int64) * stride_am +
                        offs_k[None, :].to(tl.int64) * stride_ak)
    else:
        a_ptrs = a_ptr + (source_token[:, None] * stride_am +
                        offs_k[None, :] * stride_ak).to(tl.int32)

    offs_k_continue = tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int32)
    b_ptrs = b_ptr + (off_experts * stride_be + \
        offs_bn[:, None] * stride_bn + offs_k_continue[None, :] * \
            stride_bk).to(tl.int32)   #[N, K//2]
    b_shifter = (offs_k[:, None] % 2) * 4  #[K]

    b_zp_shifter = (offs_bn[None, :] % 2) * 4

    if COMBINE_SCALE_LOAD:
        a_scale_ptrs = a_scale_ptr + source_token[:, None] * stride_asm
    else:
        a_scale_ptrs = a_scale_ptr + source_token * stride_asm

    # w4a8 deepseek case

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if COMBINE_SCALE_LOAD:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            offs_szk = k + tl.arange(0, 2)

            a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b0 = tl.load(b_ptrs)

            b0 = tl.interleave(b0, b0)  #[N, K]
            b0 = tl.trans(b0)  #[K, N]
            b0 = (b0 >> b_shifter) & 0xF
            b0 = b0.to(tl.int32)

            offs_ks = k + tl.arange(0, 2)
            a_scale = tl.load(a_scale_ptrs + offs_ks[None, :] * stride_ask,
                              mask=token_mask[:, None],
                              other=0.0) # [M, 2]
            a_scale0, a_scale1 = tl.split(a_scale) # [M]

            # b_scale shape: [N, K] = [N, 2]
            b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + \
                offs_bn[:, None] * stride_bsn + \
                offs_szk[None, :] * stride_bsk).to(tl.int32)
            b_scale = tl.load(b_scale_ptrs)
            b_scale = b_scale.to(tl.float32)
            b_scale0, b_scale1 = tl.split(b_scale) # [N]


            b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                (offs_bn[:, None]//2) * stride_bzn + \
                offs_szk[None, :] * stride_bzk).to(tl.int32)   #[N, 2]
            b_zp = tl.load(b_zp_ptrs)
            b_zp0, b_zp1 = tl.split(b_zp) # [N]

            b_zp0 = ((b_zp0 >> b_zp_shifter) & 0xF)
            b_zp1 = ((b_zp1 >> b_zp_shifter) & 0xF)
            b_zp0 = b_zp0.to(tl.int32)
            b_zp1 = b_zp1.to(tl.int32)

            a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak, mask=token_mask[:, None], other=0.0)
            b1 = tl.load(b_ptrs + (BLOCK_SIZE_K // 2) * stride_bk)
            b1 = tl.interleave(b1, b1)
            b1 = tl.trans(b1)
            b1 = (b1 >> b_shifter) & 0xF
            b1 = b1.to(tl.int32)


            b_scale0 = tl.reshape(b_scale0, (1, BLOCK_SIZE_N))
            b_scale1 = tl.reshape(b_scale1, (1, BLOCK_SIZE_N))
            b_scale0 = tl.broadcast_to(b_scale0, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            b_scale1 = tl.broadcast_to(b_scale1, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            b_zp0 = tl.reshape(b_zp0, (1, BLOCK_SIZE_N))
            b_zp1 = tl.reshape(b_zp1, (1, BLOCK_SIZE_N))
            b_zp0 = tl.broadcast_to(b_zp0, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            b_zp1 = tl.broadcast_to(b_zp1, (BLOCK_SIZE_K, BLOCK_SIZE_N))


            # We accumulate along the K dimension.
            b0 = (b0 - b_zp0).to(tl.int8)
            # accumulator += tl.dot(a0, b0) * a_scale0[:, None] * b_scale0[None, :]
            accumulator += tl.dot(a0, b0) * a_scale0[:, None] * b_scale0

            # We accumulate along the K dimension.
            b1 = (b1 - b_zp1).to(tl.int8)
            accumulator += tl.dot(a1, b1) * a_scale1[:, None] * b_scale1

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak * 2
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk * 2
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            tl.static_assert(num_groups == 1, "num_groups must be 1")
            offs_szk = k * num_groups

            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)

            b = tl.interleave(b, b)
            b = tl.trans(b)
            b = (b >> b_shifter) & 0xF
            b = b.to(tl.int32)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                            mask=token_mask,
                            other=0.0)

            b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + \
                offs_bn[None, :] * stride_bsn + \
                offs_szk * stride_bsk).to(tl.int32)
            b_scale = tl.load(b_scale_ptrs)
            b_scale = b_scale.to(tl.float32)

            b_zp_ptrs = b_zp_ptr + (off_experts * stride_bze + \
                (offs_bn[None, :]//2) * stride_bzn + \
                offs_szk[:, None] * stride_bzk).to(tl.int32)
            b_zp = tl.load(b_zp_ptrs)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.int32)


            b_scale = tl.broadcast_to(b_scale, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            b_zp = tl.broadcast_to(b_zp, (BLOCK_SIZE_K, BLOCK_SIZE_N))

            # We accumulate along the K dimension.
            b = (b - b_zp).to(tl.int8)
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if USE_ADDR_OFFSET_INT64_C:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
            None, :].to(tl.int64))
    else:
        c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
            None, :]).to(tl.int32)

    if block_n_diviable:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _fused_moe_kernel_gptq_awq_w4a8_channelwise_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return (
            "fused_moe_kernel_gptq_awq_w4a8_channelwise_bot"
            if mul_routed_weight else
            "fused_moe_kernel_gptq_awq_w4a8_channelwise"
        )
    else:
        return "fused_moe_kernel_gptq_awq_w4a8_channelwise"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
})
@triton.jit(repr=_fused_moe_kernel_gptq_awq_w4a8_channelwise_repr)
def fused_moe_kernel_gptq_awq_w4a8_channelwise(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        sorted_weights_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        block_k_diviable: tl.constexpr,
        block_n_diviable: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        COMBINE_SCALE_LOAD: tl.constexpr,
        USE_MLS_LOAD: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_ADDR_OFFSET_INT64_A: tl.constexpr,
        USE_ADDR_OFFSET_INT64_B: tl.constexpr,
        USE_ADDR_OFFSET_INT64_C: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_int4_w4a8: tl.constexpr,
        ck_sorting: tl.constexpr,
        ck_topk: tl.constexpr,
        NUM_XCDS: tl.constexpr):
    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_asm >= 0)
    tl.assume(stride_ask >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsn >= 0)
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    tl.static_assert(use_int4_w4a8 == True, "Must use int4_w4a8")
    tl.static_assert(BLOCK_SIZE_K % 2 == 0, "BLOCK_SIZE_K must be even for packed int4 weights.")
    tl.static_assert(COMBINE_SCALE_LOAD == False, "Channel-wise w4a8 does not use combined scale loads.")
    tl.static_assert(USE_MLS_LOAD == False, "Channel-wise w4a8 does not support MLS loads.")

    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
    source_token = source_token_id(offs_token, top_k)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int32) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int32)
    offs_k_continue = tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int32)

    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (
            source_token[:, None].to(tl.int64) * stride_am
            + offs_k[None, :].to(tl.int64) * stride_ak
        )
    else:
        a_ptrs = a_ptr + (
            source_token[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        ).to(tl.int32)

    if USE_ADDR_OFFSET_INT64_B:
        b_ptrs = b_ptr + (
            off_experts.to(tl.int64) * stride_be
            + offs_bn[:, None].to(tl.int64) * stride_bn
            + offs_k_continue[None, :].to(tl.int64) * stride_bk
        )
    else:
        b_ptrs = b_ptr + (
            off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_k_continue[None, :] * stride_bk
        ).to(tl.int32)
    # HIPC contiguous K-pack (no shuffle): even-k in high nibble, odd-k in low.
    # even -> shift 4, odd -> shift 0.
    b_shifter = ((offs_k[:, None] + 1) % 2) * 4

    a_scale_ptrs = a_scale_ptr + source_token * stride_asm
    b_scale_ptrs = b_scale_ptr + (off_experts * stride_bse + offs_bn[None, :] * stride_bsn).to(tl.int32)
    b_scale = tl.load(b_scale_ptrs).to(tl.float32)

    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k * BLOCK_SIZE_K
        if not block_k_diviable:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < remaining_k),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k_continue[None, :] < tl.cdiv(remaining_k, 2),
                other=0,
            )
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)

        b = tl.interleave(b, b)
        b = tl.trans(b)
        b = ((b >> b_shifter) & 0xF).to(tl.int32)
        b = tl.where(b >= 8, b - 16, b).to(tl.int8)

        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if USE_ADDR_OFFSET_INT64_C:
        c_ptrs = c_ptr + (
            stride_cm * offs_token[:, None].to(tl.int64)
            + stride_cn * offs_cn[None, :].to(tl.int64)
        )
    else:
        c_ptrs = c_ptr + (
            stride_cm * offs_token[:, None]
            + stride_cn * offs_cn[None, :]
        ).to(tl.int32)

    if block_n_diviable:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _fused_moe_kernel_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return "fused_moe_kernel_bot" if mul_routed_weight else "fused_moe_kernel"
    else:
        return "fused_moe_kernel"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
})
@triton.jit(repr=_fused_moe_kernel_repr)
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    num_tokens_post_padded_ptr,
    expert_ids_ptr,
    sorted_token_ids_ptr,
    sorted_weights_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_bias_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    total_tokens,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    block_k_diviable: tl.constexpr,
    block_n_diviable: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMBINE_SCALE_LOAD: tl.constexpr,
    USE_MLS_LOAD: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    USE_ADDR_OFFSET_INT64_A: tl.constexpr,
    USE_ADDR_OFFSET_INT64_B: tl.constexpr,
    USE_ADDR_OFFSET_INT64_C: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    c_sorted: tl.constexpr,
    bottom_a_use_mls_load: tl.constexpr,
    ck_sorting: tl.constexpr,
    ck_topk: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    SCALE_BIAS_WITH_ROUTED_WEIGHT: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    if group_k > 0:
        tl.static_assert(BLOCK_SIZE_K <= group_k and group_k % BLOCK_SIZE_K == 0,
            "BLOCK_SIZE_K must be divisible by GROUP_SIZE_K")
    if COMBINE_SCALE_LOAD: # used for use_int8_w8a8
        tl.static_assert(stride_ask == 1,
            "COMBINE_SCALE_LOAD implictly stride_ask == 1!")
        tl.static_assert(MUL_ROUTED_WEIGHT == False,
            "COMBINE_SCALE_LOAD and MUL_ROUTED_WEIGHT cannot be both true due to w1_scale and w2_scale diff layout!")
        tl.static_assert(block_k_diviable == True and BLOCK_SIZE_K == group_k,
            "COMBINE_SCALE_LOAD only add and verify on block_k_diviable!")
        tl.static_assert(use_int8_w8a8 or use_fp8_w8a8,
            "COMBINE_SCALE_LOAD only add and verify on use_int8_w8a8 or use_fp8_w8a8!")
    if USE_MLS_LOAD:
        tl.static_assert(block_k_diviable == True and block_n_diviable == True,
            "USE_MLS_LOAD must require block_k_diviable and block_n_diviable(maybe exceed 2M Page)!")
    if bottom_a_use_mls_load:
        tl.static_assert(MUL_ROUTED_WEIGHT == True and c_sorted == False,
                         "bottom_a_use_mls_load true must when MUL_ROUTED_WEIGHT == True and c_sorted == False!")


    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int32)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        # c_ptr will be a zero inited buffer, no need to write zero explicitly
        # write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
        #                       offs_token, token_mask, BLOCK_SIZE_M,
        #                       BLOCK_SIZE_N, compute_type)
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
    source_token = source_token_id(offs_token, top_k)

    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int32)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (source_token[:, None].to(tl.int64) * stride_am +
                 offs_k[None, :].to(tl.int64) * stride_ak)
    else:
        a_ptrs = a_ptr + (source_token[:, None] * stride_am +
                 offs_k[None, :] * stride_ak).to(tl.int32)

    if USE_ADDR_OFFSET_INT64_B:
        b_expert_offset = off_experts.to(tl.int64) * stride_be
        b_ptrs = b_ptr + (
            b_expert_offset
            + offs_k[:, None].to(tl.int64) * stride_bk
            + offs_bn[None, :].to(tl.int64) * stride_bn
        )
    else:
        b_expert_offset = off_experts * stride_be
        b_ptrs = b_ptr + (
            b_expert_offset
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        ).to(tl.int32)
    b_base_ptr = b_ptr + b_expert_offset

    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if COMBINE_SCALE_LOAD:
                a_scale_ptrs = a_scale_ptr + source_token[:, None] * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn[:, None] * stride_bsn)
            else:
                if bottom_a_use_mls_load: # top_k is 1
                    a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
                else:
                    a_scale_ptrs = a_scale_ptr + source_token * stride_asm

                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            if bottom_a_use_mls_load:  # top_k is 1 and A is laid out in sorted MLS order
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + source_token * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mls_offs_k = 0
    if COMBINE_SCALE_LOAD:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a0 = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b0 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            else:
                a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                if not USE_MLS_LOAD:
                    b0 = tl.load(b_ptrs)
                else:
                    b0 = tl.matrix_load(
                            b_base_ptr,
                            shape=[K, N],
                            strides=[stride_bk, stride_bn],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a0, b0.to(compute_type), acc=accumulator)
                tl.static_assert(False, "Not implemented")
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k + tl.arange(0, 2)
                    a_scale = tl.load(a_scale_ptrs + offs_ks[None, :] * stride_ask,
                                    mask=token_mask[:, None],
                                    other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks[None, :] * stride_bsk)
                    a_scale_0, a_scale_1 = tl.split(a_scale)
                    b_scale_0, b_scale_1 = tl.split(b_scale)

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a0, b0) * a_scale_0[:, None] * b_scale_0[None, :]
                    else:
                        accumulator += tl.dot(a0, b0) * (a_scale_0[:, None] * b_scale_0)

                    if not block_k_diviable:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak,
                                    mask=token_mask[:, None] & (offs_k[None, :] < K - (k + 1) * BLOCK_SIZE_K),
                                    other=0.0)
                        b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk,
                                    mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K, other=0.0)
                    else:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak, mask=token_mask[:, None], other=0.0)
                        if not USE_MLS_LOAD:
                            b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk)
                        else:
                            b1 = tl.matrix_load(
                                    b_base_ptr,
                                    shape=[K, N],
                                    strides=[stride_bk, stride_bn],
                                    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                    offsets=[mls_offs_k + BLOCK_SIZE_K, (pid_n * BLOCK_SIZE_N) % N])

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a1, b1) * a_scale_1[:, None] * b_scale_1[None, :]
                    else:
                        accumulator += tl.dot(a1, b1) * (a_scale_1[:, None] * b_scale_1)
                else:
                    if use_fp8_w8a8:
                        # acc used to enable fp8_fast_accum
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
                    tl.static_assert(False, "Not implemented")
            else:
                accumulator += tl.dot(a, b)
                tl.static_assert(False, "Not implemented")

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak * 2
            b_ptrs += BLOCK_SIZE_K * stride_bk * 2
            mls_offs_k += BLOCK_SIZE_K * 2

    else: # non-COMBINE_SCALE_LOAD

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            else:
                if not bottom_a_use_mls_load:
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                else:
                    a = tl.matrix_load(
                            a_ptr,
                            shape=[total_tokens, K],
                            strides=[stride_am, stride_ak],
                            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                            offsets=[pid_m * BLOCK_SIZE_M, mls_offs_k])
                    # mask_a_mls = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < total_tokens
                    # a = tl.where(mask_a_mls[:, None], a, 0)

                if not USE_MLS_LOAD:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.matrix_load(
                            b_base_ptr,
                            shape=[K, N],
                            strides=[stride_bk, stride_bn],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                      mask=token_mask,
                                      other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                    else:
                        accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
                else:
                    if use_fp8_w8a8:
                        # acc used to enable fp8_fast_accum
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
            else:
                accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            mls_offs_k += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        # Both of them can work well.
        if ck_sorting:
            moe_weight = tl.load(sorted_weights_ptr + offs_token_id,
                                mask=token_mask,
                                other=0)
        else:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                mask=token_mask,
                                other=0)
        accumulator = accumulator * moe_weight[:, None]

    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    if ADD_BIAS:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_ptrs = b_bias_ptr + (
            off_experts.to(tl.int64) * stride_bbe
            + offs_cn[None, :].to(tl.int64) * stride_bbn
        )
        if block_n_diviable:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(bias_ptrs, mask=offs_cn[None, :] < N, other=0.0)
        if SCALE_BIAS_WITH_ROUTED_WEIGHT:
            if ck_sorting:
                moe_weight = tl.load(sorted_weights_ptr + offs_token_id, mask=token_mask, other=0)
            else:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
            bias = bias * moe_weight[:, None]
        accumulator = accumulator + bias.to(accumulator.dtype)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if USE_ADDR_OFFSET_INT64_C:
        if c_sorted:
            c_ptrs = c_ptr + (stride_cm * offs_token_id[:, None].to(tl.int64) + stride_cn * offs_cn[
                None, :].to(tl.int64))
        else:
            c_ptrs = c_ptr + (stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
                None, :].to(tl.int64))
    else:
        if c_sorted:
            c_ptrs = c_ptr + (stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[
                None, :]).to(tl.int32)
        else:
            c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
                None, :]).to(tl.int32)
    if not block_n_diviable:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    else:
        c_mask = token_mask[:, None]

    tl.store(c_ptrs, accumulator, mask=c_mask)



def _fused_moe_splitk_kernel_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return "fused_moe_splitk_kernel_bot" if mul_routed_weight else "fused_moe_splitk_kernel"
    else:
        return "fused_moe_splitk_kernel"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
    'k_per_split': lambda nargs: nargs['K'] // nargs['SPLIT_K'],
})
@triton.jit(repr=_fused_moe_splitk_kernel_repr)
def fused_moe_splitk_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    num_tokens_post_padded_ptr,
    expert_ids_ptr,
    sorted_token_ids_ptr,
    sorted_weights_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    total_tokens,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    block_k_diviable: tl.constexpr,
    block_n_diviable: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMBINE_SCALE_LOAD: tl.constexpr,
    USE_MLS_LOAD: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    k_per_split: tl.constexpr,
    USE_ADDR_OFFSET_INT64_A: tl.constexpr,
    USE_ADDR_OFFSET_INT64_B: tl.constexpr,
    USE_ADDR_OFFSET_INT64_C: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    c_sorted: tl.constexpr,
    bottom_a_use_mls_load: tl.constexpr,
    ck_sorting: tl.constexpr,
    ck_topk: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_ck >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    if group_k > 0:
        tl.static_assert(BLOCK_SIZE_K <= group_k and group_k % BLOCK_SIZE_K == 0,
            "BLOCK_SIZE_K must be divisible by GROUP_SIZE_K")
    if COMBINE_SCALE_LOAD: # used for use_int8_w8a8
        tl.static_assert(stride_ask == 1,
            "COMBINE_SCALE_LOAD implictly stride_ask == 1!")
        tl.static_assert(MUL_ROUTED_WEIGHT == False,
            "COMBINE_SCALE_LOAD and MUL_ROUTED_WEIGHT cannot be both true due to w1_scale and w2_scale diff layout!")
        tl.static_assert(block_k_diviable == True and BLOCK_SIZE_K == group_k,
            "COMBINE_SCALE_LOAD only add and verify on block_k_diviable!")
        tl.static_assert(use_int8_w8a8 or use_fp8_w8a8,
            "COMBINE_SCALE_LOAD only add and verify on use_int8_w8a8 or use_fp8_w8a8!")
    if USE_MLS_LOAD:
        tl.static_assert(block_k_diviable == True and block_n_diviable == True,
            "USE_MLS_LOAD must require block_k_diviable and block_n_diviable(maybe exceed 2M Page)!")
    if bottom_a_use_mls_load:
        tl.static_assert(MUL_ROUTED_WEIGHT == True and c_sorted == False,
                         "bottom_a_use_mls_load true must when MUL_ROUTED_WEIGHT == True and c_sorted == False!")

    if SPLIT_K != 1:
        tl.static_assert((use_int8_w8a8 == False and use_int8_w8a16 == False) and use_fp8_w8a8 == False,
            "SPLIT_K only add and verify on use_int8_w8a8 == False and use_fp8_w8a8 == False and use_int8_w8a16 == False!")
        tl.static_assert(MUL_ROUTED_WEIGHT == False,
                         "SPLIT_K can only work on gemm1 case!")

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    splitk_idx = 0 if SPLIT_K == 1 else tl.program_id(axis=1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)


    k_start = splitk_idx * k_per_split
    k_end = k_start + k_per_split

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int32)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        # c_ptr will be a zero inited buffer, no need to write zero explicitly
        # write_zeros_to_output(c_ptr, pid_k, stride_ck, stride_cm, stride_cn, pid_n, N,
        #                       offs_token, token_mask, BLOCK_SIZE_M,
        #                       BLOCK_SIZE_N, compute_type)
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int32)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
    source_token = source_token_id(offs_token, top_k)

    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int32)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if USE_ADDR_OFFSET_INT64_A:
        a_ptrs = a_ptr + (source_token[:, None].to(tl.int64) * stride_am +
                 (offs_k[None, :] + k_start).to(tl.int64) * stride_ak)
    else:
        a_ptrs = a_ptr + (source_token[:, None] * stride_am +
                 (offs_k[None, :] + k_start) * stride_ak).to(tl.int32)

    if USE_ADDR_OFFSET_INT64_B:
        b_expert_offset = off_experts.to(tl.int64) * stride_be
        b_ptrs = b_ptr + (
            b_expert_offset
            + (offs_k[:, None] + k_start).to(tl.int64) * stride_bk
            + offs_bn[None, :].to(tl.int64) * stride_bn
        )
    else:
        b_expert_offset = off_experts * stride_be
        b_ptrs = b_ptr + (
            b_expert_offset
            + (offs_k[:, None] + k_start) * stride_bk
            + offs_bn[None, :] * stride_bn
        ).to(tl.int32)
    b_base_ptr = b_ptr + b_expert_offset

    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if COMBINE_SCALE_LOAD:
                a_scale_ptrs = a_scale_ptr + source_token[:, None] * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn[:, None] * stride_bsn)
            else:
                if bottom_a_use_mls_load: # top_k is 1
                    a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
                else:
                    a_scale_ptrs = a_scale_ptr + source_token * stride_asm

                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            if bottom_a_use_mls_load:  # top_k is 1 and A is laid out in sorted MLS order
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + source_token * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mls_offs_k = k_start
    if COMBINE_SCALE_LOAD:
        tl.static_assert(SPLIT_K == 1, "COMBINE_SCALE_LOAD only add and verify on SPLIT_K == 1!")

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a0 = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b0 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            else:
                a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                if not USE_MLS_LOAD:
                    b0 = tl.load(b_ptrs)
                else:
                    b0 = tl.matrix_load(
                            b_base_ptr,
                            shape=[K, N],
                            strides=[stride_bk, stride_bn],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a0, b0.to(compute_type), acc=accumulator)
                tl.static_assert(False, "Not implemented")
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k + tl.arange(0, 2)
                    a_scale = tl.load(a_scale_ptrs + offs_ks[None, :] * stride_ask,
                                    mask=token_mask[:, None],
                                    other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks[None, :] * stride_bsk)
                    a_scale_0, a_scale_1 = tl.split(a_scale)
                    b_scale_0, b_scale_1 = tl.split(b_scale)

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a0, b0) * a_scale_0[:, None] * b_scale_0[None, :]
                    else:
                        accumulator += tl.dot(a0, b0) * (a_scale_0[:, None] * b_scale_0)

                    if not block_k_diviable:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak,
                                    mask=token_mask[:, None] & (offs_k[None, :] < K - (k + 1) * BLOCK_SIZE_K),
                                    other=0.0)
                        b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk,
                                    mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K, other=0.0)
                    else:
                        a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak, mask=token_mask[:, None], other=0.0)
                        if not USE_MLS_LOAD:
                            b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk)
                        else:
                            b1 = tl.matrix_load(
                                    b_base_ptr,
                                    shape=[K, N],
                                    strides=[stride_bk, stride_bn],
                                    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                    offsets=[mls_offs_k + BLOCK_SIZE_K, (pid_n * BLOCK_SIZE_N) % N])

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a1, b1) * a_scale_1[:, None] * b_scale_1[None, :]
                    else:
                        accumulator += tl.dot(a1, b1) * (a_scale_1[:, None] * b_scale_1)
                else:
                    if use_fp8_w8a8:
                        # acc used to enable fp8_fast_accum
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
                    tl.static_assert(False, "Not implemented")
            else:
                accumulator += tl.dot(a, b)
                tl.static_assert(False, "Not implemented")

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak * 2
            b_ptrs += BLOCK_SIZE_K * stride_bk * 2
            mls_offs_k += BLOCK_SIZE_K * 2

    else: # non-COMBINE_SCALE_LOAD

        for k in range(0, tl.cdiv(k_per_split, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.
            if not block_k_diviable:
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < (k_per_split) - k * BLOCK_SIZE_K),
                            other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < (k_per_split) - k * BLOCK_SIZE_K, other=0.0)
            else:
                if not bottom_a_use_mls_load:
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                else:
                    a = tl.matrix_load(
                            a_ptr,
                            shape=[total_tokens, K],
                            strides=[stride_am, stride_ak],
                            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                            offsets=[pid_m * BLOCK_SIZE_M, mls_offs_k])

                if not USE_MLS_LOAD:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.matrix_load(
                            b_base_ptr,
                            shape=[K, N],
                            strides=[stride_bk, stride_bn],
                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                            offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

            # We accumulate along the K dimension.
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                      mask=token_mask,
                                      other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                    if BLOCK_SIZE_N > group_n:
                        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                    else:
                        accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
                else:
                    if use_fp8_w8a8:
                        # acc used to enable fp8_fast_accum
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
            else:
                accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            mls_offs_k += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        # Both of them can work well.
        if ck_sorting:
            moe_weight = tl.load(sorted_weights_ptr + offs_token_id,
                                mask=token_mask,
                                other=0)
        else:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                mask=token_mask,
                                other=0)
        accumulator = accumulator * moe_weight[:, None]

    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if USE_ADDR_OFFSET_INT64_C:
        if c_sorted:
            c_ptrs = c_ptr + (stride_ck * splitk_idx.to(tl.int64) + stride_cm * offs_token_id[:, None].to(tl.int64) + stride_cn * offs_cn[
                None, :].to(tl.int64))
        else:
            c_ptrs = c_ptr + (stride_ck * splitk_idx.to(tl.int64) + stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
                None, :].to(tl.int64))
    else:
        if c_sorted:
            c_ptrs = c_ptr + (stride_ck * splitk_idx + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[
                None, :]).to(tl.int32)
        else:
            c_ptrs = c_ptr + (stride_ck * splitk_idx + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
                None, :]).to(tl.int32)
    if not block_n_diviable:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    else:
        c_mask = token_mask[:, None]

    tl.store(c_ptrs, accumulator, mask=c_mask)



def _fused_moe_persistent_kernel_repr(specialization):
    if distinguish_moe_kernel_name():
        constants = specialization.constants
        mul_routed_weight = constants.get('MUL_ROUTED_WEIGHT', False)
        return "fused_moe_persistent_kernel_bot" if mul_routed_weight else "fused_moe_persistent_kernel"
    else:
        return "fused_moe_persistent_kernel"


@triton.heuristics(values={
    'block_k_diviable': lambda nargs: nargs['K'] % (nargs['BLOCK_SIZE_K']) == 0,
    'block_n_diviable': lambda nargs: nargs['N'] % nargs['BLOCK_SIZE_N'] == 0,
})
@triton.jit(repr=_fused_moe_persistent_kernel_repr)
def fused_moe_persistent_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    num_tokens_post_padded_ptr,
    expert_ids_ptr,
    sorted_token_ids_ptr,
    sorted_weights_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    total_tokens,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    block_k_diviable: tl.constexpr,
    block_n_diviable: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMBINE_SCALE_LOAD: tl.constexpr,
    USE_MLS_LOAD: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    USE_ADDR_OFFSET_INT64_A: tl.constexpr,
    USE_ADDR_OFFSET_INT64_B: tl.constexpr,
    USE_ADDR_OFFSET_INT64_C: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    c_sorted: tl.constexpr,
    bottom_a_use_mls_load: tl.constexpr,
    ck_sorting: tl.constexpr,
    ck_topk: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    This is the persistent version of the fused_moe kernel.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_be >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_bse >= 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bsn >= 0)

    # to notify the compiler that sorted_token_ids_ptr is a pointer to the memory,
    # and all value in the memory is non-negative.
    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)

    if group_k > 0:
        tl.static_assert(BLOCK_SIZE_K <= group_k and group_k % BLOCK_SIZE_K == 0,
            "BLOCK_SIZE_K must be divisible by GROUP_SIZE_K")
    if COMBINE_SCALE_LOAD: # used for use_int8_w8a8
        tl.static_assert(stride_ask == 1,
            "COMBINE_SCALE_LOAD implictly stride_ask == 1!")
        tl.static_assert(MUL_ROUTED_WEIGHT == False,
            "COMBINE_SCALE_LOAD and MUL_ROUTED_WEIGHT cannot be both true due to w1_scale and w2_scale diff layout!")
        tl.static_assert(block_k_diviable == True and BLOCK_SIZE_K == group_k,
            "COMBINE_SCALE_LOAD only add and verify on block_k_diviable!")
        tl.static_assert(use_int8_w8a8 or use_fp8_w8a8,
            "COMBINE_SCALE_LOAD only add and verify on use_int8_w8a8 or use_fp8_w8a8!")
    if USE_MLS_LOAD:
        tl.static_assert(block_k_diviable == True and block_n_diviable == True,
            "USE_MLS_LOAD must require block_k_diviable and block_n_diviable(maybe exceed 2M Page)!")
    if bottom_a_use_mls_load:
        tl.static_assert(MUL_ROUTED_WEIGHT == True and c_sorted == False,
                         "bottom_a_use_mls_load true must when MUL_ROUTED_WEIGHT == True and c_sorted == False!")


    # -----------------------------------------------------------
    # Simply compute how many iterations each persistent block needs to do
    start_pid = tl.program_id(axis=0)

    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid

    num_tiles = num_pid_m * num_pid_n

    # Compute how many tiles are outside the padding region
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)

    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int32)
        if off_experts == -1:
            # -----------------------------------------------------------
            # Write back zeros to the output when the expert is not
            # in the current expert parallel rank.
            # c_ptr will be a zero inited buffer, no need to write zero explicitly
            # write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
            #                       offs_token, token_mask, BLOCK_SIZE_M,
            #                       BLOCK_SIZE_N, compute_type)
            pass
        else:
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int32)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

            offs_token = decode_sorted_token_id(offs_token, ck_sorting, ck_topk)
            source_token = source_token_id(offs_token, top_k)

            token_mask = offs_token < num_valid_tokens


            offs_bn = (pid_n * BLOCK_SIZE_N +
                    tl.arange(0, BLOCK_SIZE_N).to(tl.int32)) % N

            offs_k = tl.arange(0, BLOCK_SIZE_K)
            if USE_ADDR_OFFSET_INT64_A:
                a_ptrs = a_ptr + (source_token[:, None].to(tl.int64) * stride_am +
                        offs_k[None, :].to(tl.int64) * stride_ak)
            else:
                a_ptrs = a_ptr + (source_token[:, None] * stride_am +
                        offs_k[None, :] * stride_ak).to(tl.int32)

            if USE_ADDR_OFFSET_INT64_B:
                b_expert_offset = off_experts.to(tl.int64) * stride_be
                b_ptrs = b_ptr + (
                    b_expert_offset
                    + offs_k[:, None].to(tl.int64) * stride_bk
                    + offs_bn[None, :].to(tl.int64) * stride_bn
                )
            else:
                b_expert_offset = off_experts * stride_be
                b_ptrs = b_ptr + (
                    b_expert_offset
                    + offs_k[:, None] * stride_bk
                    + offs_bn[None, :] * stride_bn
                ).to(tl.int32)
            b_base_ptr = b_ptr + b_expert_offset

            if use_int8_w8a16:
                b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                    None, :] * stride_bsn
                b_scale = tl.load(b_scale_ptrs)

            if use_fp8_w8a8 or use_int8_w8a8:
                # block-wise
                if group_k > 0 and group_n > 0:
                    if COMBINE_SCALE_LOAD:
                        a_scale_ptrs = a_scale_ptr + source_token[:, None] * stride_asm
                        offs_bsn = offs_bn // group_n
                        b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                        offs_bsn[:, None] * stride_bsn)
                    else:
                        if bottom_a_use_mls_load: # top_k is 1
                            a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
                        else:
                            a_scale_ptrs = a_scale_ptr + source_token * stride_asm

                        offs_bsn = offs_bn // group_n
                        b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                                        offs_bsn * stride_bsn)
                # channel-wise
                elif per_channel_quant:
                    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                        None, :] * stride_bsn
                    b_scale = tl.load(b_scale_ptrs)
                    # Load per-token scale for activations
                    if bottom_a_use_mls_load:  # top_k is 1 and A is laid out in sorted MLS order
                        a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
                    else:
                        a_scale_ptrs = a_scale_ptr + source_token * stride_asm
                    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                                None]
                # tensor-wise
                else:
                    a_scale = tl.load(a_scale_ptr)
                    b_scale = tl.load(b_scale_ptr + off_experts)

            # -----------------------------------------------------------
            # Iterate to compute a block of the C matrix.
            # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
            # of fp32 values for higher accuracy.
            # `accumulator` will be converted back to fp16 after the loop.
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            mls_offs_k = 0
            if COMBINE_SCALE_LOAD:
                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K), 2):
                    # Load the next block of A and B, generate a mask by checking the
                    # K dimension.
                    if not block_k_diviable:
                        a0 = tl.load(a_ptrs,
                                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                                    other=0.0)
                        b0 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                    else:
                        a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                        if not USE_MLS_LOAD:
                            b0 = tl.load(b_ptrs)
                        else:
                            b0 = tl.matrix_load(
                                    b_base_ptr,
                                    shape=[K, N],
                                    strides=[stride_bk, stride_bn],
                                    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                    offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

                    # We accumulate along the K dimension.
                    if use_int8_w8a16:
                        accumulator = tl.dot(a0, b0.to(compute_type), acc=accumulator)
                        tl.static_assert(False, "Not implemented")
                    elif use_fp8_w8a8 or use_int8_w8a8:
                        if group_k > 0 and group_n > 0:
                            k_start = k * BLOCK_SIZE_K
                            offs_ks = k_start // group_k + tl.arange(0, 2)
                            a_scale = tl.load(a_scale_ptrs + offs_ks[None, :] * stride_ask,
                                            mask=token_mask[:, None],
                                            other=0.0)
                            b_scale = tl.load(b_scale_ptrs + offs_ks[None, :] * stride_bsk)
                            a_scale_0, a_scale_1 = tl.split(a_scale)
                            b_scale_0, b_scale_1 = tl.split(b_scale)

                            if BLOCK_SIZE_N > group_n:
                                accumulator += tl.dot(a0, b0) * a_scale_0[:, None] * b_scale_0[None, :]
                            else:
                                accumulator += tl.dot(a0, b0) * (a_scale_0[:, None] * b_scale_0)

                            if not block_k_diviable:
                                a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak,
                                            mask=token_mask[:, None] & (offs_k[None, :] < K - (k + 1) * BLOCK_SIZE_K),
                                            other=0.0)
                                b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk,
                                            mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K, other=0.0)
                            else:
                                a1 = tl.load(a_ptrs + BLOCK_SIZE_K * stride_ak, mask=token_mask[:, None], other=0.0)
                                if not USE_MLS_LOAD:
                                    b1 = tl.load(b_ptrs + BLOCK_SIZE_K * stride_bk)
                                else:
                                    b1 = tl.matrix_load(
                                            b_base_ptr,
                                            shape=[K, N],
                                            strides=[stride_bk, stride_bn],
                                            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                            offsets=[mls_offs_k + BLOCK_SIZE_K, (pid_n * BLOCK_SIZE_N) % N])

                            if BLOCK_SIZE_N > group_n:
                                accumulator += tl.dot(a1, b1) * a_scale_1[:, None] * b_scale_1[None, :]
                            else:
                                accumulator += tl.dot(a1, b1) * (a_scale_1[:, None] * b_scale_1)
                        else:
                            if use_fp8_w8a8:
                                # acc used to enable fp8_fast_accum
                                accumulator = tl.dot(a, b, acc=accumulator)
                            else:
                                accumulator += tl.dot(a, b)
                            tl.static_assert(False, "Not implemented")
                    else:
                        accumulator += tl.dot(a, b)
                        tl.static_assert(False, "Not implemented")

                    # Advance the ptrs to the next K block.
                    a_ptrs += BLOCK_SIZE_K * stride_ak * 2
                    b_ptrs += BLOCK_SIZE_K * stride_bk * 2
                    mls_offs_k += BLOCK_SIZE_K * 2

            else: # non-COMBINE_SCALE_LOAD

                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

                    # Load the next block of A and B, generate a mask by checking the
                    # K dimension.
                    if not block_k_diviable:
                        a = tl.load(a_ptrs,
                                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                                    other=0.0)
                        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                    else:
                        if not bottom_a_use_mls_load:
                            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                        else:
                            a = tl.matrix_load(
                                    a_ptr,
                                    shape=[total_tokens, K],
                                    strides=[stride_am, stride_ak],
                                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                                    offsets=[pid_m * BLOCK_SIZE_M, mls_offs_k])
                            # mask_a_mls = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < total_tokens
                            # a = tl.where(mask_a_mls[:, None], a, 0)

                        if not USE_MLS_LOAD:
                            b = tl.load(b_ptrs)
                        else:
                            b = tl.matrix_load(
                                    b_base_ptr,
                                    shape=[K, N],
                                    strides=[stride_bk, stride_bn],
                                    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                                    offsets=[mls_offs_k, (pid_n * BLOCK_SIZE_N) % N])

                    # We accumulate along the K dimension.
                    if use_int8_w8a16:
                        accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
                    elif use_fp8_w8a8 or use_int8_w8a8:
                        if group_k > 0 and group_n > 0:
                            k_start = k * BLOCK_SIZE_K
                            offs_ks = k_start // group_k
                            a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                            mask=token_mask,
                                            other=0.0)
                            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                            if BLOCK_SIZE_N > group_n:
                                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                            else:
                                accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
                        else:
                            if use_fp8_w8a8:
                                # acc used to enable fp8_fast_accum
                                accumulator = tl.dot(a, b, acc=accumulator)
                            else:
                                accumulator += tl.dot(a, b)
                    else:
                        accumulator += tl.dot(a, b)
                    # Advance the ptrs to the next K block.
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk
                    mls_offs_k += BLOCK_SIZE_K

            if MUL_ROUTED_WEIGHT:
                # Both of them can work well.
                if ck_sorting:
                    moe_weight = tl.load(sorted_weights_ptr + offs_token_id,
                                        mask=token_mask,
                                        other=0)
                else:
                    moe_weight = tl.load(topk_weights_ptr + offs_token,
                                        mask=token_mask,
                                        other=0)
                accumulator = accumulator * moe_weight[:, None]

            if use_int8_w8a16:
                accumulator = (accumulator * b_scale).to(compute_type)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    accumulator = accumulator.to(compute_type)
                else:
                    accumulator = (accumulator * a_scale * b_scale).to(compute_type)
            else:
                accumulator = accumulator.to(compute_type)
            # -----------------------------------------------------------
            # Write back the block of the output
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            if USE_ADDR_OFFSET_INT64_C:
                if c_sorted:
                    c_ptrs = c_ptr + (stride_cm * offs_token_id[:, None].to(tl.int64) + stride_cn * offs_cn[
                        None, :].to(tl.int64))
                else:
                    c_ptrs = c_ptr + (stride_cm * offs_token[:, None].to(tl.int64) + stride_cn * offs_cn[
                        None, :].to(tl.int64))
            else:
                if c_sorted:
                    c_ptrs = c_ptr + (stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[
                        None, :]).to(tl.int32)
                else:
                    c_ptrs = c_ptr + (stride_cm * offs_token[:, None] + stride_cn * offs_cn[
                        None, :]).to(tl.int32)
            if not block_n_diviable:
                c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            else:
                c_mask = token_mask[:, None]

            tl.store(c_ptrs, accumulator, mask=c_mask)

        tile_id += NUM_SMS


_USE_MOE_PERSISTENT_KERNEL = False

def moe_set_use_persistent_kernel(value: bool):
    global _USE_MOE_PERSISTENT_KERNEL
    _USE_MOE_PERSISTENT_KERNEL = value

def fused_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: Optional[torch.Tensor],
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int4_w4a8: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    c_sorted: bool = False,
    bottom_a_use_mls_load: bool = False,
    ck_sorting: bool = False,
    ck_topk: int = 8,
    scale_bias_with_routed_weight: bool = False,
    B_bias: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert (block_shape is None
                or triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2))
        assert (block_shape is None
                or triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1))

    elif use_int8_w8a16 or use_int4_w4a16 or use_int4_w4a8:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    total_tokens = A.size(0)
    num_tokens = topk_ids.numel()

    if config is None:
        moe_config_func = get_optimal_moe_config_func(
            A, B, topk_ids,
            use_int8_w8a16=use_int8_w8a16,
            use_int8_w8a8=use_int8_w8a8,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
            use_int4_w4a8=use_int4_w4a8,
            use_mxfp4_w4a4=use_mxfp4_w4a4,
            block_shape=block_shape,
            is_bottom=mul_routed_weight)
        config = moe_config_func(M)

    if "USE_MLS_LOAD" not in config:
        config["USE_MLS_LOAD"] = False
    if config["USE_MLS_LOAD"] == True and capMLS == False:
        logger.warning("USE_MLS_LOAD is not supported for this architecture!!!")
        config["USE_MLS_LOAD"] = False

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )

    input_dtype = str(A.dtype).split('.')[-1]

    if (use_int8_w8a16 or use_int4_w4a16 or use_int4_w4a8) and \
            block_shape is not None and block_shape[1] > 0:
        if B_bias is not None and use_int4_w4a8:
            raise ValueError("B_bias is not supported in fused_moe_kernel_gptq_awq_w4a8 yet")
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        offset_max = 2**32 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_b = B.numel() * B.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max

        if use_int4_w4a8:
            return fused_moe_kernel_gptq_awq_w4a8[grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                sorted_weights,
                expert_ids,
                num_tokens_post_padded,
                B.size(1),
                A.size(1),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0),
                B_scale.stride(2),
                B_scale.stride(1),
                B_zp.stride(0) if B_zp is not None else 0,
                B_zp.stride(2) if B_zp is not None else 0,
                B_zp.stride(1) if B_zp is not None else 0,
                group_k=block_shape[1],
                group_size=block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                has_zp=B_zp is not None,
                use_int4_w4a16=use_int4_w4a16,
                use_int4_w4a8=use_int4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                ck_sorting=ck_sorting,
                ck_topk=ck_topk,
                NUM_XCDS=1,
                **config
            )

        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            expert_ids,
            num_tokens_post_padded,
            B.size(1),
            A.size(1),
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
            USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
            USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            ck_sorting=ck_sorting,
            ck_topk=ck_topk,
            NUM_XCDS=1,
            **config,
        )
    else:
        offset_max = 2**32 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max
        use_addr_offset_int64_b = B.numel() * B.element_size() >= offset_max

        config = config.copy()
        BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
        if block_shape is not None:
            BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0],
                                                 block_shape[1]))

        if use_int4_w4a8 and per_channel_quant:
            if B_bias is not None:
                raise ValueError("B_bias is not supported in fused_moe_kernel_gptq_awq_w4a8_channelwise yet")
            assert B_scale is not None and B_scale.ndim in (2, 3)
            assert B_zp is None
            channelwise_config = config.copy()
            channelwise_config.pop("USE_MLS_LOAD", None)
            channelwise_config.pop("COMBINE_SCALE_LOAD", None)
            channelwise_config["USE_MLS_LOAD"] = False
            channelwise_config["COMBINE_SCALE_LOAD"] = False
            w4a8_grid = lambda META: (
                triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
            )
            return fused_moe_kernel_gptq_awq_w4a8_channelwise[w4a8_grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                sorted_weights,
                expert_ids,
                num_tokens_post_padded,
                B.size(1),
                A.size(1),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0),
                B_scale.stride(1),
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_int4_w4a8=use_int4_w4a8,
                ck_sorting=ck_sorting,
                ck_topk=ck_topk,
                NUM_XCDS=1,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                **channelwise_config,
            )

        SPLIT_K = config.pop("SPLIT_K", splitk_size)
        if mul_routed_weight:
            SPLIT_K = 0

        if _USE_MOE_PERSISTENT_KERNEL:
            # Note: the sms count is irrelevant to occupancy(lds and regs).
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2

            grid = lambda META: (
                min(
                    NUM_SMS,
                    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
                    * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
                ),
            )

            fused_moe_persistent_kernel[grid](
                A,
                B,
                C,
                topk_weights,
                num_tokens_post_padded,
                expert_ids,
                sorted_token_ids,
                sorted_weights,
                A_scale,
                B_scale,
                B.size(1),
                B.size(2),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2)
                if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                A.size(0),
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                c_sorted=c_sorted,
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=ck_sorting,
                ck_topk=ck_topk,
                NUM_SMS=NUM_SMS,
                NUM_XCDS=1,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                COMBINE_SCALE_LOAD=config.pop("COMBINE_SCALE_LOAD", None),
                **config,
            )
        elif SPLIT_K > 1:
            if B_bias is not None:
                raise ValueError("B_bias is not supported in fused_moe_splitk_kernel yet")

            grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
                    B.shape[1], META['BLOCK_SIZE_N']), SPLIT_K)
            assert B.size(2) % (BLOCK_SIZE_K * SPLIT_K) == 0, "B.size(2) must be divisible by BLOCK_SIZE_K * SPLIT_K"

            splitk_cache = torch.zeros((SPLIT_K,) + C.shape, device=C.device,dtype=C.dtype)

            use_addr_offset_int64_c = C.numel() * C.element_size() * SPLIT_K >= offset_max

            fused_moe_splitk_kernel[grid](
                A,
                B,
                splitk_cache,
                topk_weights,
                num_tokens_post_padded,
                expert_ids,
                sorted_token_ids,
                sorted_weights,
                A_scale,
                B_scale,
                B.size(1),
                B.size(2),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                splitk_cache.stride(-3),
                splitk_cache.stride(-2),
                splitk_cache.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2)
                if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                A.size(0),
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                SPLIT_K=SPLIT_K,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                c_sorted=c_sorted,
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=ck_sorting,
                ck_topk=ck_topk,
                NUM_XCDS=1,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                COMBINE_SCALE_LOAD=config.pop("COMBINE_SCALE_LOAD", None),
                **config,
            )
            torch.sum(splitk_cache, dim=0, out=C)
            # # C.copy_(torch.sum(splitk_cache.to(torch.float32), dim=0).to(C.dtype))
            # triton_splitk_reduce(splitk_cache, C)
        else:
            fused_moe_kernel[grid](
                A,
                B,
                C,
                topk_weights,
                num_tokens_post_padded,
                expert_ids,
                sorted_token_ids,
                sorted_weights,
                A_scale,
                B_scale,
                (B_bias if B_bias is not None else B),
                B.size(1),
                B.size(2),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2)
                if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_bias.stride(0) if B_bias is not None else B.stride(0),
                B_bias.stride(1) if B_bias is not None else B.stride(1),
                A.size(0),
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                c_sorted=c_sorted,
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=ck_sorting,
                ck_topk=ck_topk,
                NUM_XCDS=1,
                SCALE_BIAS_WITH_ROUTED_WEIGHT=scale_bias_with_routed_weight,
                ADD_BIAS=B_bias is not None,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                COMBINE_SCALE_LOAD=config.pop("COMBINE_SCALE_LOAD", None),
                **config,
            )
