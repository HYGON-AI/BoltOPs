# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from boltops.fused_moe.triton.moe_activation import (
    _normalize_activation_and_gate,
    adjust_N_for_activation,
    _apply_activation,
)

from boltops.fused_moe.triton.moe_op import fused_moe as invoke_fused_moe_kernel, support_mls
# from vllm import _custom_ops as ops
from boltops.fused_moe.triton.moe_config_utils import get_optimal_moe_config_func
from boltops.fused_moe.triton.moe_compat import (
    moe_align_block_size as moe_align_block_size_aiter,
    per_block_quant_wrapper,
    per_token_quant_hip,
    sgl_moe_align_block_size as sgl_moe_align_block_size_aiter,
)
from boltops.utility.torch_guard import torch_compile_guard
from boltops.fused_moe.triton.moe_sorting_hipc import moe_sorting_fwd
from boltops.utility import dtypes


def get_moe_sum_config(M, topk, N):
    if M < 32:
        return {"BLOCK_SIZE": 128, "num_warps": 1}
    else:
        return {"BLOCK_SIZE": 512, "num_warps": 4}

# def generate_sum_configs():
#     configs = []
#     for block_n in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
#         for num_warps in [1, 2, 4]:
#             for num_stages in [1, 2]:
#                 config = triton.Config({
#                     'BLOCK_SIZE': block_n,
#                 }, num_warps=num_warps, num_stages=num_stages)
#                 configs.append(config)
#     return configs

# @triton.autotune(
#     key=['M', 'N', 'topk','compute_type'],
#     configs=generate_sum_configs(),
#     # configs = [
#     #             triton.Config({'BLOCK_SIZE': 64  }, num_warps=1),
#     #             triton.Config({'BLOCK_SIZE': 128 }, num_warps=1),
#     #             triton.Config({'BLOCK_SIZE': 256 }, num_warps=4),
#     #             triton.Config({'BLOCK_SIZE': 512 }, num_warps=4),
#     #           ],
#     perf_debug=True,
# )
@triton.heuristics({
    "n_dividable": lambda args: (args["N"] % args["BLOCK_SIZE"]) == 0,
})
@triton.jit
def moe_sum_kernel(
    output_ptr,             # [M, N]
    input_ptr,              # [M, topk, N]
    M,
    N: tl.constexpr,
    topk: tl.constexpr,
    routed_scaling_factor,
    BLOCK_SIZE: tl.constexpr,
    stride_output_m,
    stride_output_n,
    stride_input_m,
    stride_input_k,
    stride_input_n,
    compute_type: tl.constexpr,
    n_dividable: tl.constexpr,
):
    tl.assume(stride_output_m >= 0)
    tl.assume(stride_output_n >= 0)
    tl.assume(stride_input_m >= 0)
    tl.assume(stride_input_k >= 0)
    tl.assume(stride_input_n >= 0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    pid = tl.program_id(axis=0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_row_ptr = input_ptr + pid_m.to(tl.int64) * stride_input_m
    for k in range(topk):
        input_ptrs = input_row_ptr + (
            k * stride_input_k
            + offs_n * stride_input_n
        )
        if n_dividable:
            x = tl.load(input_ptrs)
        else:
            x = tl.load(input_ptrs, mask=mask_n, other=0.0)
        acc += x.to(tl.float32)

    acc *= routed_scaling_factor
    acc = acc.to(compute_type)
    output_ptrs = output_ptr + (pid_m.to(tl.int64) * stride_output_m + offs_n * stride_output_n)
    if n_dividable:
        tl.store(output_ptrs, acc)
    else:
        tl.store(output_ptrs, acc, mask=mask_n)


def triton_moe_sum(input_tensor,
                   output_tensor,
                   routed_scaling_factor: float = 1.0):
    """
    1D tile version of moe_sum.

    Args:
        input_tensor: [M, topk, N]
        output_tensor: [M, N]
    """
    M, topk, N = input_tensor.shape

    assert output_tensor.dtype == torch.float16 or \
           output_tensor.dtype == torch.bfloat16 or \
           output_tensor.dtype == torch.float32

    if output_tensor.dtype == torch.float16:
        compute_type = tl.float16
    elif output_tensor.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif output_tensor.dtype == torch.float32:
        compute_type = tl.float32

    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()
    assert input_tensor.shape[0] == output_tensor.shape[0]
    assert input_tensor.shape[2] == output_tensor.shape[1]

    # 计算grid
    config = get_moe_sum_config(M, topk, N)
    grid = (M * triton.cdiv(N, config["BLOCK_SIZE"]),)
    # grid = lambda META: (M * triton.cdiv(N, META['BLOCK_SIZE']), )

    moe_sum_kernel[grid](
        output_tensor,
        input_tensor,
        M,
        N,
        topk,
        routed_scaling_factor,
        stride_output_m=output_tensor.stride(0),
        stride_output_n=output_tensor.stride(1),
        stride_input_m=input_tensor.stride(0),
        stride_input_k=input_tensor.stride(1),
        stride_input_n=input_tensor.stride(2),
        compute_type=compute_type,
        **config,
    )

    return output_tensor


def ceil_div(a, b):
    return (a + b - 1) // b

@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread,
                                         numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# Triton implementation based on:
# https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts),
                              dtype=torch.int32,
                              device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1, ),
                         dtype=torch.int32,
                         device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1, )](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, topk] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'topk' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    if num_experts >= 224:
        if num_experts != 256:
            moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
        else:
            # Currently requires num_experts=256
            # Note: sgl_moe_align_block_size has 10x performance compared to moe_align_block_size,
            # but this functionis removed since vllm.0.9.2. vLLM will check this issue.
            # if hasattr(ops, 'sgl_moe_align_block_size'):
            #     # ops.sgl_moe_align_block_size(
            #         topk_ids,
            #         num_experts,
            #         block_size,
            #         sorted_ids,
            #         expert_ids,
            #         num_tokens_post_pad,
            #     )
            # else:
            #     ops.moe_align_block_size(
            #         topk_ids,
            #         num_experts,
            #         block_size,
            #         sorted_ids,
            #         expert_ids,
            #         num_tokens_post_pad,
            #     )
            sgl_moe_align_block_size_aiter(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
    else:
        # ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
        #                          expert_ids, num_tokens_post_pad)
        moe_align_block_size_aiter(topk_ids, num_experts, block_size, sorted_ids,
                                   expert_ids, num_tokens_post_pad)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad

def moe_sorting_ck(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=32,
    expert_mask=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    tokens_positions_per_expert = torch.empty(
        (num_experts*2,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((1), dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    # for now, moe_sorting_fwd only support int32 topk_ids
    if topk_ids.dtype != dtypes.i32:
        topk_ids = topk_ids.to(dtypes.i32)

    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_positions_per_expert,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, tokens_positions_per_expert, moe_buf


def inplace_fused_experts(hidden_states: torch.Tensor,
                          w1: torch.Tensor,
                          w2: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          activation: Optional[str] = None,
                          b1: Optional[torch.Tensor] = None,
                          b2: Optional[torch.Tensor] = None,
                          is_gated: Optional[bool] = None,
                          apply_router_weight_on_input: bool = False,
                          use_fp8_w8a8: bool = False,
                          use_int8_w8a8: bool = False,
                          use_int8_w8a16: bool = False,
                          use_int4_w4a16: bool = False,
                          use_int4_w4a8: bool = False,
                          per_channel_quant: bool = False,
                          global_num_experts: int = -1,
                          expert_map: Optional[torch.Tensor] = None,
                          w1_scale: Optional[torch.Tensor] = None,
                          w2_scale: Optional[torch.Tensor] = None,
                          w1_zp: Optional[torch.Tensor] = None,
                          w2_zp: Optional[torch.Tensor] = None,
                          a1_scale: Optional[torch.Tensor] = None,
                          a2_scale: Optional[torch.Tensor] = None,
                          block_shape: Optional[List[int]] = None,
                          output_dtype: Optional[torch.dtype] = None,
                          no_combine: bool = False,
                          routed_scaling_factor: Optional[float] = 1.0,
                          gemm1_alpha: Optional[float] = None,
                          gemm1_limit: Optional[float] = None,
                          activation_situ_beta: Optional[float] = None,
                          activation_situ_linear_beta: Optional[float] = None) -> None:
    if activation is None:
        activation = "silu"
    fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output_dtype=output_dtype,
        inplace=True,
        activation=activation,
        is_gated=is_gated,
        b1=b1,
        b2=b2,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )

def outplace_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: Optional[str] = None,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        is_gated: Optional[bool] = None,
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_int4_w4a8: bool = False,
        per_channel_quant: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        output_dtype: Optional[torch.dtype] = None,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = 1.0,
        gemm1_alpha: Optional[float] = None,
        gemm1_limit: Optional[float] = None,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None) -> torch.Tensor:
    if activation is None:
        activation = "silu"
    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output_dtype=output_dtype,
        inplace=False,
        activation=activation,
        is_gated=is_gated,
        b1=b1,
        b2=b2,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )

def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  activation: Optional[str] = None,
                  b1: Optional[torch.Tensor] = None,
                  b2: Optional[torch.Tensor] = None,
                  is_gated: Optional[bool] = None,
                  apply_router_weight_on_input: bool = False,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  use_int4_w4a16: bool = False,
                  use_int4_w4a8: bool = False,
                  per_channel_quant: bool = False,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  w1_zp: Optional[torch.Tensor] = None,
                  w2_zp: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None,
                  output_dtype: Optional[torch.dtype] = None,
                  no_combine: bool = False,
                  routed_scaling_factor: Optional[float] = 1.0,
                  gemm1_alpha: Optional[float] = None,
                  gemm1_limit: Optional[float] = None,
                  activation_situ_beta: Optional[float] = None,
                  activation_situ_linear_beta: Optional[float] = None) -> torch.Tensor:
    if activation is None:
        activation = 'silu'
    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output_dtype=output_dtype,
        inplace=inplace,
        activation=activation,
        is_gated=is_gated,
        b1=b1,
        b2=b2,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )

def fused_moe_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        inplace: bool = False,
        activation: str = "silu",
        is_gated: Optional[bool] = None,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_int4_w4a8: bool = False,
        per_channel_quant: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = 1.0,
        gemm1_alpha: Optional[float] = None,
        gemm1_limit: Optional[float] = None,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
        fn_key: Optional[str] = None,
) -> torch.Tensor:
    device = topk_ids.device
    M, topk = topk_ids.shape
    dtype = (torch.bfloat16 if hidden_states.dtype == torch.bfloat16 else torch.float16) if output_dtype is None else output_dtype
    # E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    # FIXME: W2.size must be same as hidden_dim
    output_shape = (M, topk, w2.shape[1]) if no_combine else hidden_states.shape
    moe_buf = torch.empty(output_shape, dtype=dtype, device=device)
    return moe_buf
@functools.lru_cache()
def _bottom_moe_use_mls():
    return support_mls()


@torch_compile_guard(gen_fake=fused_moe_fake)
def fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       output_dtype: Optional[torch.dtype] = None,
                       inplace: bool = False,
                       activation: str = "silu",
                       is_gated: Optional[bool] = None,
                       b1: Optional[torch.Tensor] = None,
                       b2: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       use_int4_w4a8: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       no_combine: bool = False,
                       routed_scaling_factor: Optional[float] = 1.0,
                       gemm1_alpha: Optional[float] = None,
                       gemm1_limit: Optional[float] = None,
                       activation_situ_beta: Optional[float] = None,
                       activation_situ_linear_beta: Optional[float] = None)-> torch.Tensor:
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)
    activation_out_dim = adjust_N_for_activation(w1.shape[1], is_gated)

    if output_dtype is None:
        output_dtype = hidden_states.dtype

    # Check constraints.
    if use_int4_w4a16 or use_int4_w4a8:
        if is_gated:
            assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
        else:
            assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16, torch.int8,  torch.float8_e4m3fn
    ]
    assert output_dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported output_dtype"

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    topk = topk_ids.shape[1]

    if output_dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif output_dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = int(os.environ.get("TRITON_FUSED_MOE_CHUNK_SIZE", "16384"))
    M = min(num_tokens, CHUNK_SIZE)

    moe_config_func = get_optimal_moe_config_func(
                        hidden_states, w1, topk_ids,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int8_w8a8=use_int8_w8a8,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int4_w4a16=use_int4_w4a16,
                        use_int4_w4a8=use_int4_w4a8,
                        use_mxfp4_w4a4=False, #always false in wna16
                        block_shape=block_shape,
                        is_bottom=False,
                        is_gated=is_gated)
    moe_config_func2 = get_optimal_moe_config_func(
                        hidden_states, w2, topk_ids,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int8_w8a8=use_int8_w8a8,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int4_w4a16=use_int4_w4a16,
                        use_int4_w4a8=use_int4_w4a8,
                        use_mxfp4_w4a4=False, #always false in wna16
                        block_shape=block_shape,
                        is_bottom=True)
    config = moe_config_func(M)
    config2, max_block_m = moe_config_func2(M)
    if config["BLOCK_SIZE_M"] != config2["BLOCK_SIZE_M"]:
        raise ValueError(
            "Top and bottom MoE configs must use the same BLOCK_SIZE_M: "
            f"top={config['BLOCK_SIZE_M']}, bottom={config2['BLOCK_SIZE_M']}"
        )

    bottom_moe_a_use_mls = (
        _bottom_moe_use_mls()
        and not use_int4_w4a8
        and config2 is not None
        and config2.get("USE_MLS_LOAD", False))

    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if bottom_moe_a_use_mls else 0
    )
    total_tokens = M * topk + max_padded_tokens

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    if expert_map is not None:
        cache13 = torch.zeros(total_tokens * max(N, w2.shape[1]),
                            device=hidden_states.device,
                            dtype=output_dtype)
    else:
        cache13 = torch.empty(total_tokens * max(N, w2.shape[1]),
                            device=hidden_states.device,
                            dtype=output_dtype)
    intermediate_cache3 = cache13[:M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]))

    if no_combine:
        assert not inplace, "no_combine + inplace is not supported"
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=output_dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=output_dtype)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config = moe_config_func(tokens_in_chunk)
            config2, max_block_m = moe_config_func2(tokens_in_chunk)
            if config["BLOCK_SIZE_M"] != config2["BLOCK_SIZE_M"]:
                raise ValueError(
                    "Top and bottom MoE configs must use the same BLOCK_SIZE_M: "
                    f"top={config['BLOCK_SIZE_M']}, bottom={config2['BLOCK_SIZE_M']}"
                )
            bottom_moe_a_use_mls = (
                _bottom_moe_use_mls()
                and config2 is not None
                and config2.get("USE_MLS_LOAD", False)
                and (block_shape is not None and (use_int8_w8a8 or use_fp8_w8a8)))
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if bottom_moe_a_use_mls
            else 0
        )
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache13[: total_tokens * N].view(
            (total_tokens, N),
        )
        if expert_map is not None:
            intermediate_cache2 = torch.zeros(
                (total_tokens, activation_out_dim),
                device=hidden_states.device,
                dtype=output_dtype)
        else:
            intermediate_cache2 = torch.empty(
                (total_tokens, activation_out_dim),
                device=hidden_states.device,
                dtype=output_dtype)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        ck_sorting = True
        sorted_weights = None
        if not ck_sorting:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        else:
            # Convert expert_map (global->local mapping, -1 for inactive) to
            # expert_mask (binary 0/1 mask) expected by moe_sorting_ck's C kernel.
            expert_mask = (expert_map >= 0).to(torch.int32) if expert_map is not None else None
            sorted_token_ids, sorted_weights, expert_ids, num_tokens_post_padded, \
                _tokens_positions_per_expert, _moe_buf = (
                    moe_sorting_ck(curr_topk_ids, curr_topk_weights, global_num_experts,
                                w2.shape[1], output_dtype, config["BLOCK_SIZE_M"], expert_mask)
            )

        if (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8) and per_channel_quant:
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype==torch.bfloat16:
                input_q,input_scale = per_token_quant_hip(curr_hidden_states,quant_dtype=quant_dtype)
            else:
                input_q,input_scale = curr_hidden_states,a1_scale
            invoke_fused_moe_kernel(input_q,
                                    w1,
                                    intermediate_cache1,
                                    input_scale,
                                    w1_scale,
                                    w1_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    apply_router_weight_on_input,
                                    topk,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    c_sorted=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    B_bias=b1,
                                    config=config)
        elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype==torch.bfloat16:
                input_q, input_scale = per_block_quant_wrapper((1,block_shape[1]))(per_token_quant_hip)(curr_hidden_states,quant_dtype=quant_dtype)
            else:
                input_q, input_scale = curr_hidden_states,a1_scale
            invoke_fused_moe_kernel(input_q,
                                    w1,
                                    intermediate_cache1,
                                    input_scale,
                                    w1_scale,
                                    w1_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    apply_router_weight_on_input,
                                    topk,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    c_sorted=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    B_bias=b1,
                                    config=config)
        else:
            invoke_fused_moe_kernel(curr_hidden_states,
                                    w1,
                                    intermediate_cache1,
                                    a1_scale,
                                    w1_scale,
                                    w1_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    apply_router_weight_on_input,
                                    topk,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    c_sorted=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    B_bias=b1,
                                    config=config)
        _apply_activation(
            activation=activation,
            is_gated=is_gated,
            activated_out=intermediate_cache2,
            ffn1_out_2d=intermediate_cache1.view(-1, N),
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        if expert_map != None:
            # for EP mode, intermediate_cache1 and intermediate_cache3 need be zeros inited
            # since intermediate_cache1 and intermediate_cache3 shared same buffer,
            # to make sure intermediate_cache3 is zeros inited,
            # intermediate_cache1 need inited to zeros after silu_and_mul
            intermediate_cache1.fill_(0)

        if (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8) and per_channel_quant:
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            bridge_q, bridge_scale = per_token_quant_hip(intermediate_cache2, quant_dtype=quant_dtype)
            invoke_fused_moe_kernel(bridge_q,
                                    w2,
                                    intermediate_cache3,
                                    bridge_scale,
                                    w2_scale,
                                    w2_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    (not apply_router_weight_on_input) and (not no_combine),
                                    1,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    bottom_a_use_mls_load=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                                    B_bias=b2,
                                    config=config2)
        elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            bridge_q, bridge_scale = per_block_quant_wrapper((1,block_shape[1]))(per_token_quant_hip)(intermediate_cache2, quant_dtype=quant_dtype)
            invoke_fused_moe_kernel(bridge_q,
                                    w2,
                                    intermediate_cache3,
                                    bridge_scale,
                                    w2_scale,
                                    w2_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    (not apply_router_weight_on_input) and (not no_combine),
                                    1,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    bottom_a_use_mls_load=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                                    B_bias=b2,
                                    config=config2)
        else:
            invoke_fused_moe_kernel(intermediate_cache2,
                        w2,
                        intermediate_cache3,
                        a2_scale,
                        w2_scale,
                        w2_zp,
                        curr_topk_weights,
                        curr_topk_ids,
                        sorted_token_ids,
                        sorted_weights,
                        expert_ids,
                        num_tokens_post_padded,
                        (not apply_router_weight_on_input) and (not no_combine),
                        1,
                        compute_type=compute_type,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int8_w8a8=use_int8_w8a8,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int4_w4a16=use_int4_w4a16,
                        use_int4_w4a8=use_int4_w4a8,
                        per_channel_quant=per_channel_quant,
                        block_shape=block_shape,
                        bottom_a_use_mls_load=bottom_moe_a_use_mls,
                        ck_sorting=ck_sorting,
                        ck_topk=topk,
                        scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                        B_bias=b2,
                        config=config2)
        if no_combine:
            out_hidden_states[begin_chunk_idx:end_chunk_idx].copy_(intermediate_cache3)
        else:
            triton_moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor=routed_scaling_factor,
            )
        if end_chunk_idx < num_tokens and expert_map != None:
            # if has next chunk, intermediate_cache3 need init to zeros
            intermediate_cache3.fill_(0)
    return out_hidden_states
