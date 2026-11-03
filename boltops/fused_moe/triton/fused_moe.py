# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/vllm/model_executor/layers/fused_moe/fused_moe.py
"""Compatible fused MoE entry and ordinary expert orchestration."""
import functools
import inspect as _inspect
import os
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from . import mxfp4, mxfp8
from .mxfp4_quant import (
    per_token_group_quant_fp8_mxfp4,
    silu_mul_quant_fp8_packed_triton,
)
from .quant import static_per_tensor_quant_fp8_i8

from boltops.fused_moe.triton.moe_activation import (
    _normalize_activation_and_gate,
    adjust_N_for_activation,
    _apply_activation,
    is_fused_activation_quant_supported,
    triton_activation_quant,
)

from boltops.fused_moe.triton.moe_op import fused_moe as invoke_fused_moe_kernel, support_mls
# from vllm import _custom_ops as ops
from boltops.fused_moe.triton.moe_config_utils import get_optimal_moe_config_func
from boltops.fused_moe.triton.moe_compat import (
    moe_align_block_size as moe_align_block_size_aiter,
    per_block_quant_wrapper,
    per_token_quant_triton,
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
    HAS_ROUTER_WEIGHTS: tl.constexpr = False,  # Preserve the original unweighted sum by default.
    router_weights_ptr=None,  # Optional FP32 [M,topk]; omitted by legacy callers.
    stride_router_m=0,
    stride_router_k=0,
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
        x = x.to(tl.float32)
        if HAS_ROUTER_WEIGHTS:
            x *= tl.load(router_weights_ptr + pid_m * stride_router_m + k * stride_router_k)
        acc += x

    acc *= routed_scaling_factor
    acc = acc.to(compute_type)
    output_ptrs = output_ptr + (pid_m.to(tl.int64) * stride_output_m + offs_n * stride_output_n)
    if n_dividable:
        tl.store(output_ptrs, acc)
    else:
        tl.store(output_ptrs, acc, mask=mask_n)


def triton_moe_sum(input_tensor,
                   output_tensor,
                   routed_scaling_factor: float = 1.0,
                   router_weights: Optional[torch.Tensor] = None):
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
        router_weights_ptr=router_weights,
        HAS_ROUTER_WEIGHTS=router_weights is not None,
        stride_router_m=router_weights.stride(0) if router_weights is not None else 0,
        stride_router_k=router_weights.stride(1) if router_weights is not None else 0,
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
            if (idx >= 0) & (idx < num_experts):
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
        if (expert_id >= 0) & (expert_id < num_experts):
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
                          activation_situ_linear_beta: Optional[float] = None,
                          use_mxfp4_w4a8: bool = False,
                          use_ue8m0: bool = False,
                          use_mxfp8: bool = False,
                          gemm1_beta: Optional[float] = None,  # Up offset; None selects SiLU=0 or SwiGLU-OAI=1.
) -> None:
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
        use_mxfp4_w4a8=use_mxfp4_w4a8,
        use_ue8m0=use_ue8m0,
        use_mxfp8=use_mxfp8,
        gemm1_beta=gemm1_beta,
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
        activation_situ_linear_beta: Optional[float] = None,
        use_mxfp4_w4a8: bool = False,
        use_ue8m0: bool = False,
        use_mxfp8: bool = False,
        gemm1_beta: Optional[float] = None,  # Up offset; None selects SiLU=0 or SwiGLU-OAI=1.
) -> torch.Tensor:
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
        use_mxfp4_w4a8=use_mxfp4_w4a8,
        use_ue8m0=use_ue8m0,
        use_mxfp8=use_mxfp8,
        gemm1_beta=gemm1_beta,
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
                  activation_situ_linear_beta: Optional[float] = None,
                  use_mxfp4_w4a8: bool = False,
                  use_ue8m0: bool = False,
                  use_mxfp8: bool = False,
                  gemm1_beta: Optional[float] = None,  # Up offset; None selects SiLU=0 or SwiGLU-OAI=1.
) -> torch.Tensor:
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
        use_mxfp4_w4a8=use_mxfp4_w4a8,
        use_ue8m0=use_ue8m0,
        use_mxfp8=use_mxfp8,
        gemm1_beta=gemm1_beta,
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
        use_mxfp4_w4a8: bool = False,
        use_ue8m0: bool = False,
        use_mxfp8: bool = False,
        gemm1_beta: Optional[float] = None,  # Up-branch offset; carried for the real operator signature.
) -> torch.Tensor:
    device = topk_ids.device
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype if output_dtype is None else output_dtype
    # E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    # FIXME: W2.size must be same as hidden_dim
    output_shape = (M, topk, w2.shape[1]) if no_combine else hidden_states.shape
    moe_buf = torch.empty(output_shape, dtype=dtype, device=device)
    return moe_buf
@functools.lru_cache()
def _bottom_moe_use_mls():
    return support_mls()


def _bottom_a_uses_sorted_layout(config, use_int4_w4a8):
    """Whether GEMM1/GEMM2 use the padded expert-sorted bridge layout.

    The historical config key mentions MLS because that was the first consumer
    of this layout.  The layout itself does not require MLS: architectures
    without ``tl.matrix_load`` can consume the same contiguous rows with a
    regular ``tl.load``.
    """
    if use_int4_w4a8 or config is None:
        return False
    return config.get("a_sorted", config.get("bottom_a_use_mls_load", False))


def _bottom_a_uses_mls(config, use_int4_w4a8):
    return (
        _bottom_moe_use_mls()
        and _bottom_a_uses_sorted_layout(config, use_int4_w4a8)
        and config.get("bottom_a_use_mls_load", False)
    )


def _should_use_naive_block_assignment(
    *,
    num_routes: int,
    global_num_experts: int,
    has_expert_map: bool,
    bottom_a_sorted: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_int4_w4a8: bool,
    use_block_quant: bool,
) -> bool:
    """Return whether each route should own one GEMM tile without metadata."""
    min_experts_per_route = 8 if use_block_quant else 4
    return (
        not has_expert_map
        and not bottom_a_sorted
        and not use_int8_w8a16
        and not use_int4_w4a16
        and not use_int4_w4a8
        and num_routes * min_experts_per_route <= global_num_experts
    )


def _should_use_fused_activation_quant(
    *,
    use_int8_w8a8: bool,
    use_fp8_w8a8: bool,
    use_int4_w4a8: bool,
    per_channel_quant: bool,
    has_expert_map: bool,
    activation: str,
    is_gated: bool,
    output_dim: int,
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    activation_situ_beta: Optional[float],
    activation_situ_linear_beta: Optional[float],
) -> bool:
    """Return whether activation and quantization may use the fused kernel."""
    # INT4 W4A8 also uses INT8 activations, but its fused path is deferred
    # until that weight path has dedicated correctness and performance tests.
    w8a8_mode_count = int(use_int8_w8a8) + int(use_fp8_w8a8)
    if (
        w8a8_mode_count != 1
        or use_int4_w4a8
        or not per_channel_quant
        or has_expert_map
    ):
        return False
    quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
    return is_fused_activation_quant_supported(
        activation,
        is_gated,
        quant_dtype,
        output_dim,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


def _should_fold_router_weight(
    *,
    fuse_activation_quant: bool,
    needs_router_weight: bool,
) -> bool:
    """Return whether router weights can move into activation quant scales.

    Before folding, every GEMM2 output element is multiplied by the route's
    router weight::

        route_output = (activation_q @ w2) * activation_scale * weight_scale
        route_output *= router_weight

    Fold the per-route scalar into the per-token activation scale instead::

        folded_scale = activation_scale * router_weight
        route_output = (activation_q @ w2) * folded_scale * weight_scale

    A router weight is one scalar shared by the route's entire output row.
    Folding adds one scalar multiply where activation quantization already
    produces the per-token scale, while removing the router-weight load and
    multiply from every GEMM2 output element. The saving therefore grows with
    the GEMM2 output width. GEMM2 bias, when present, is still multiplied by
    the router weight. Layout-specific row alignment is handled separately by
    the caller.
    """
    return fuse_activation_quant and needs_router_weight


def _fused_experts_fake(*args, **kwargs):
    # Bind the real operator's positional arguments by name while retaining
    # the legacy fn_key position on the exported fused_moe_fake helper.
    signature = _inspect.signature(fused_moe_fake)
    parameters = [p for p in signature.parameters.values() if p.name != "fn_key"]
    bound = signature.replace(parameters=parameters).bind(*args, **kwargs)
    return fused_moe_fake(**bound.arguments)


# MXFP algorithm references: vLLM 77f58e530450416d217e4816727a35910bf3d9bf,
# experts/mxfp8_native_moe.py, DeepGemmFP4Experts in experts/deep_gemm_moe.py,
# and experts/aiter_mxfp4_w4a8_moe.py. Cross-checked with SGLang's native
# MXFP8 and DeepGEMM runners; the parent entry alone registers the custom op.
def _get_mxfp_configs(num_tokens, *, use_mxfp8, mxfp4_static, E, N, K, topk):
    if use_mxfp8:
        config = mxfp8.get_mxfp8_moe_config(num_tokens, E=E, N=N, K=K, top_k=topk)
        config2, _ = mxfp8.get_mxfp8_moe_config(
            num_tokens, is_bottom=True, E=E, N=N, K=K, top_k=topk)
    else:
        scale_recipe = 'static-a-tensor-b1x32-e8m0' if mxfp4_static else 'deepgemm-a128-b1x32'
        config = mxfp4.get_mxfp4_moe_config(
            num_tokens, E=E, N=N, K=K, top_k=topk, scale_recipe=scale_recipe)
        config2, _ = mxfp4.get_mxfp4_moe_config(
            num_tokens, is_bottom=True, E=E, N=N, K=K, top_k=topk,
            scale_recipe=scale_recipe)
    if config['BLOCK_SIZE_M'] != config2['BLOCK_SIZE_M']:
        raise ValueError('Top and bottom MoE configs must use the same BLOCK_SIZE_M: '
                         f"top={config['BLOCK_SIZE_M']}, bottom={config2['BLOCK_SIZE_M']}")
    return config, config2


def fused_experts_mxfp_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    activation: str = 'silu',
    is_gated: Optional[bool] = None,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = 1.0,
    gemm1_alpha: Optional[float] = None,
    gemm1_beta: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
    use_mxfp4_w4a8: bool = False,  # FP8(E4M3) A x packed FP4(E2M1) W; A scale: dynamic K128 or static tensor, W scale: K32.
    use_mxfp8: bool = False,  # FP8(E4M3) A x FP8(E4M3) W; both scales: K32 E8M0. Mutually exclusive with MXFP4.
) -> torch.Tensor:
    """Execute native MXFP8, dynamic A128/B32 MXFP4, or static FP8 W4A8.

    Each expert GEMM computes C[m,n] = sum_k (Aq[m,k] * sA) * (Wq[n,k] * sW).
    MXFP4: sW is E8M0 per 32 K elements; sA is a power-of-two scale per
    128 K elements, or a scalar FP32 scale when a2_scale selects static FP8.
    MXFP8: sA and sW are both E8M0 per 32 K elements.
    Here sA/sW denote decoded multipliers, indexed by the row and K group.

    Caller/checkpoint layouts and rounding contracts are the same as the
    parent entry. use_ue8m0 is absent: each MX recipe defines its own scales.
    """
    if not (use_mxfp4_w4a8 or use_mxfp8):
        raise ValueError('MXFP experts require use_mxfp4_w4a8 or use_mxfp8')
    # vLLM 77f58e530450, experts/aiter_mxfp4_w4a8_moe.py:
    # static FP8 W4A8 has two tensor scales, not dynamic A32/A128 scales.
    mxfp4_static = use_mxfp4_w4a8 and a2_scale is not None
    if mxfp4_static:
        for scale in (a1_scale, a2_scale):
            if (scale is None or scale.numel() != 1 or scale.dtype != torch.float32
                    or scale.device != hidden_states.device or not scale.is_contiguous()):
                raise ValueError("static MXFP4 requires a1_scale and a2_scale as one FP32 value on the input device")
        a1_scale, a2_scale = a1_scale.view(-1), a2_scale.view(-1)
    if use_mxfp8 and use_mxfp4_w4a8:
        raise ValueError("MXFP8 and MXFP4 are mutually exclusive")
    if use_mxfp8:
        # vLLM Mxfp8NativeTritonExperts expects unquantized inputs. Its native
        # entry has no bias or input-router-weight implementation; SGLang's
        # native wrapper explicitly rejects both features.
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("native MXFP8 requires BF16/FP16 hidden_states")
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("native MXFP8 quantizes activations internally; a1_scale/a2_scale must be None")
        if b1 is not None or b2 is not None or apply_router_weight_on_input:
            raise ValueError("native MXFP8 does not support bias or input routing weights")
        if output_dtype is not None and output_dtype != hidden_states.dtype:
            raise ValueError("native MXFP8 output_dtype must match hidden_states.dtype")
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise ValueError("MXFP8 weights must be E4M3; FNUZ is not supported")
        if block_shape is not None and tuple(block_shape) != (1, 32):
            raise ValueError("native MXFP8 requires weight block_shape=[1,32]")
        for scale in (w1_scale, w2_scale):
            if scale is None or scale.dtype != torch.uint8:
                raise ValueError("native MXFP8 requires uint8 E8M0 weight scales")
    mx_activation_group = 128 if use_mxfp4_w4a8 and not mxfp4_static else 32
    mx_name = "MXFP8" if use_mxfp8 else "MXFP4"
    if w1_scale is None or w2_scale is None:
        raise ValueError(f"{mx_name} requires UE8M0 weight scales and no zero points")
    if use_mxfp4_w4a8 and block_shape is not None and list(block_shape) != [1, 32]:
        raise ValueError(f"{mx_name} weight block_shape must be [1, 32]")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError(f"{mx_name} experts require BF16/FP16 or prequantized E4M3 activations")
    if use_mxfp4_w4a8 and not mxfp4_static and (b1 is not None or b2 is not None or gemm1_alpha is not None):
        raise ValueError("dynamic MXFP4 follows DeepGEMM and does not support bias or custom alpha")
    if apply_router_weight_on_input and not mxfp4_static:
        if topk_ids.shape[1] != 1:
            raise ValueError(f"{mx_name} input routing weights require topk=1, as in vLLM prepare")
        if hidden_states.dtype == torch.float8_e4m3fn:
            raise ValueError(f"{mx_name} input routing weights must be applied before FP8 quantization; "
                             "provide BF16/FP16 hidden_states")
    if hidden_states.dtype == torch.float8_e4m3fn:
        if a1_scale is None or output_dtype is None:
            raise ValueError(f"prequantized {mx_name} activations require a1_scale and output_dtype")
    elif a1_scale is not None and not mxfp4_static:
        raise ValueError(f"a1_scale is only used with prequantized {mx_name} activations")
    if a2_scale is not None and not mxfp4_static:
        raise ValueError(f"{mx_name} computes GEMM2 activation scales dynamically; a2_scale must be None")
    resolved_output_dtype = hidden_states.dtype if output_dtype is None else output_dtype
    if resolved_output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{mx_name} output_dtype must be BF16, FP16 or FP32")
    if inplace and resolved_output_dtype != hidden_states.dtype:
        raise ValueError(f"inplace {mx_name} requires output_dtype to match hidden_states.dtype")
    if use_mxfp4_w4a8 and hidden_states.dtype == torch.float8_e4m3fn and not mxfp4_static:
        # AITER x_scales and DeepGEMM scales carry distinct group layouts.
        # Retain that recipe through GEMM2; block_shape describes W only.
        mx_activation_group = mxfp4.activation_scale_group_size(hidden_states, a1_scale)
        if mx_activation_group != 128:
            raise ValueError("dynamic MXFP4 fused_experts requires G128 activation scales")
    # Keep weight groups independent of the activation recipe.
    block_shape = [1, 32] if block_shape is None else block_shape
    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)
    if mxfp4_static:
        allowed_activations = ('swigluoai',)
        if gemm1_beta is not None and gemm1_beta != 1.:
            raise ValueError('static MXFP4 SwiGLU-OAI requires gemm1_beta=1')
        gemm1_alpha = 1.702 if gemm1_alpha is None else gemm1_alpha
        gemm1_limit = 7. if gemm1_limit is None else gemm1_limit
    elif use_mxfp8:
        allowed_activations = ('silu', 'swigluoai')
    else:
        allowed_activations = ('silu', 'swiglustep', 'situ')
    if not is_gated or activation not in allowed_activations:
        raise ValueError(f'{mx_name} requires gated {allowed_activations}')
    if gemm1_beta is not None and not (mxfp4_static or use_mxfp8):
        raise ValueError('gemm1_beta requires native MXFP8 or static MXFP4')
    if activation == 'swiglustep':
        # vLLM's unmasked SWIGLUSTEP always uses 7, ignoring the SiLU clamp.
        gemm1_limit = 7.

    output_dtype = resolved_output_dtype
    routed_scaling_factor = 1. if routed_scaling_factor is None else routed_scaling_factor
    assert hidden_states.shape[1] == w1.shape[2] * (2 if use_mxfp4_w4a8 else 1), 'Hidden size mismatch'
    assert topk_weights.shape == topk_ids.shape, 'topk shape mismatch'
    assert hidden_states.is_contiguous(), 'Hidden_states must be contiguous'
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1, 'Stride of last dimension must be 1'
    if no_combine:
        assert not inplace, 'no_combine + inplace is not supported'

    num_tokens, hidden = hidden_states.shape
    E, N, _ = w1.shape
    intermediate = N // 2
    topk = topk_ids.shape[1]
    if use_mxfp8 and expert_map is None:
        global_num_experts = E  # Include appended shared experts, as in vLLM native.
    elif global_num_experts == -1:
        global_num_experts = expert_map.numel() if expert_map is not None else E
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
                    torch.float32: tl.float32}[output_dtype]
    chunk_size = int(os.environ.get('TRITON_FUSED_MOE_CHUNK_SIZE', '32768'))
    if chunk_size <= 0:
        raise ValueError('TRITON_FUSED_MOE_CHUNK_SIZE must be positive')
    M = min(num_tokens, chunk_size)
    config, config2 = _get_mxfp_configs(
        M, use_mxfp8=use_mxfp8, mxfp4_static=mxfp4_static,
        E=E, N=intermediate, K=hidden, topk=topk)

    # vLLM native MXFP8: GEMM1 rounds to input dtype, GEMM2 remains FP32.
    # DeepGEMM FP4: both GEMMs round to output dtype before route reduction.
    cache_dtype = torch.float32 if use_mxfp8 else output_dtype
    gemm1_dtype = hidden_states.dtype if use_mxfp8 else output_dtype
    cache_width = w2.shape[1] if mxfp4_static else max(N, w2.shape[1])
    if expert_map is not None:
        cache = torch.zeros(M * topk * cache_width, dtype=cache_dtype, device=hidden_states.device)
    else:
        cache = torch.empty(M * topk * cache_width, dtype=cache_dtype, device=hidden_states.device)
    routes = cache[:M * topk * w2.shape[1]].view(M, topk, w2.shape[1])
    if inplace:
        output = hidden_states
    else:
        shape = (num_tokens, topk, w2.shape[1]) if no_combine else hidden_states.shape
        output = torch.empty(shape, dtype=output_dtype, device=hidden_states.device)

    for begin in range(0, num_tokens, chunk_size):
        end = min(begin + chunk_size, num_tokens)
        x = hidden_states[begin:end]
        tokens = end - begin
        if begin and tokens < chunk_size:
            config, config2 = _get_mxfp_configs(
                tokens, use_mxfp8=use_mxfp8, mxfp4_static=mxfp4_static,
                E=E, N=intermediate, K=hidden, topk=topk)
        current_routes = routes[:tokens]
        ids = topk_ids[begin:end].contiguous()
        weights = topk_weights[begin:end].to(torch.float32).contiguous()
        if apply_router_weight_on_input and not mxfp4_static:
            # vLLM prepare multiplies in input dtype before dynamic quantization.
            x = x * topk_weights[begin:end].to(x.dtype)
        ck_sorting = expert_map is None
        sorted_weights = None
        if ck_sorting:
            sorted_ids, sorted_weights, experts, padded, _, _ = moe_sorting_ck(
                ids, weights, global_num_experts, w2.shape[1], output_dtype,
                config['BLOCK_SIZE_M'])
        else:
            sorted_ids, experts, padded = moe_align_block_size(
                ids, config['BLOCK_SIZE_M'], global_num_experts, expert_map)
        gemm1 = None if mxfp4_static else cache.view(gemm1_dtype)[:tokens * topk * N].view(-1, N)
        if mxfp4_static:
            if x.dtype == torch.float8_e4m3fn:
                input_q = x
            else:
                input_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                static_per_tensor_quant_fp8_i8(input_q, x, a1_scale)
            if expert_map is not None:
                activation_q = torch.zeros((tokens * topk, intermediate),
                                           dtype=torch.float8_e4m3fn, device=x.device)
            else:
                activation_q = torch.empty((tokens * topk, intermediate),
                                           dtype=torch.float8_e4m3fn, device=x.device)
            # vLLM/AITER: scale -> bias -> SwiGLU-OAI -> optional gamma -> quant.
            mxfp4.moe_gemm_mxfp4(
                input_q, w1, activation_q, a1_scale, w1_scale, ids,
                sorted_ids, experts, padded, top_k=topk,
                ck_sorting=ck_sorting, ck_topk=topk,
                block_size_m=config['BLOCK_SIZE_M'], block_size_n=config['BLOCK_SIZE_N'],
                topk_weights=weights, mul_routed_weight=apply_router_weight_on_input,
                bias=b1, fuse_swiglu=True, quant_scale=a2_scale,
                alpha=gemm1_alpha, limit=gemm1_limit)
            activation_scale = a2_scale
        else:
            if x.dtype == torch.float8_e4m3fn:
                input_q, input_scale = x, a1_scale[begin:end]
            elif use_mxfp8:
                input_q, input_scale = mxfp8.mxfp8_e4m3_quantize(x)
            else:
                input_q, input_scale = per_token_group_quant_fp8_mxfp4(
                    x, group_size=mx_activation_group)
            invoke_fused_moe_kernel(
                input_q, w1, gemm1, input_scale, w1_scale, None,
                weights, ids, sorted_ids, sorted_weights, experts, padded,
                False, topk, compute_type=compute_type,
                use_mxfp4_w4a8=use_mxfp4_w4a8, use_mxfp8=use_mxfp8,
                block_shape=block_shape, ck_sorting=ck_sorting, ck_topk=topk,
                config=config)
            if use_mxfp8:
                oai = activation == 'swigluoai'
                activation_q, activation_scale = mxfp8.mxfp8_e4m3_quantize(
                    gemm1, activation=activation,
                    alpha=(1.702 if oai else 1.) if gemm1_alpha is None else gemm1_alpha,
                    beta=(1. if oai else 0.) if gemm1_beta is None else gemm1_beta,
                    limit=gemm1_limit)
            elif activation == 'silu':
                activation_q, activation_scale = silu_mul_quant_fp8_packed_triton(
                    gemm1, group_size=mx_activation_group, clamp_limit=gemm1_limit)
            else:
                if expert_map is not None:
                    activated = torch.zeros((tokens * topk, intermediate), dtype=gemm1_dtype, device=x.device)
                else:
                    activated = torch.empty((tokens * topk, intermediate), dtype=gemm1_dtype, device=x.device)
                _apply_activation(
                    activation=activation, is_gated=True, activated_out=activated,
                    ffn1_out_2d=gemm1, gemm1_alpha=gemm1_alpha, gemm1_limit=gemm1_limit,
                    activation_situ_beta=activation_situ_beta,
                    activation_situ_linear_beta=activation_situ_linear_beta)
                activation_q, activation_scale = per_token_group_quant_fp8_mxfp4(
                    activated, group_size=mx_activation_group)
            if expert_map is not None:
                gemm1.zero_()  # Restore the overlapping GEMM2 cache for non-local routes.
        needs_weight = not (apply_router_weight_on_input or no_combine)
        invoke_fused_moe_kernel(
            activation_q, w2, current_routes, activation_scale, w2_scale, None,
            weights, ids, sorted_ids, sorted_weights, experts, padded,
            needs_weight and (use_mxfp8 or mxfp4_static), 1,
            compute_type=compute_type,
            use_mxfp4_w4a8=use_mxfp4_w4a8, use_mxfp8=use_mxfp8,
            block_shape=block_shape, ck_sorting=ck_sorting, ck_topk=topk,
            B_bias=b2, config=config2)
        if no_combine:
            output[begin:end].copy_(current_routes)
        else:
            triton_moe_sum(
                current_routes, output[begin:end], routed_scaling_factor=routed_scaling_factor,
                router_weights=weights if needs_weight and not (use_mxfp8 or mxfp4_static) else None)
        if end < num_tokens and expert_map is not None:
            current_routes.zero_()
    return output


@torch_compile_guard(gen_fake=_fused_experts_fake)
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
                       a1_scale: Optional[torch.Tensor] = None,  # Prequantized A1 scales; scalar FP32 for static MXFP4.
                       a2_scale: Optional[torch.Tensor] = None,  # Scalar FP32 selects static MXFP4; dynamic MX requires None.
                       block_shape: Optional[List[int]] = None,
                       no_combine: bool = False,
                       routed_scaling_factor: Optional[float] = 1.0,
                       gemm1_alpha: Optional[float] = None,
                       gemm1_limit: Optional[float] = None,
                       activation_situ_beta: Optional[float] = None,
                       activation_situ_linear_beta: Optional[float] = None,
                       use_mxfp4_w4a8: bool = False,  # FP4(E2M1) weights + FP8(E4M3) activations; weight scale: E8M0 per 32 K elements.
                       use_ue8m0: bool = False,  # Ordinary block FP8: round activation scales up to powers of two; store them as FP32.
                       use_mxfp8: bool = False,  # FP8(E4M3) weights/activations; A/B scales: E8M0 per 32 K elements (dot_scaled).
                       gemm1_beta: Optional[float] = None,  # Up offset; None selects SiLU=0 or SwiGLU-OAI=1.
) -> torch.Tensor:
    if use_mxfp4_w4a8 or use_mxfp8:
        mx_name = "MXFP8" if use_mxfp8 else "MXFP4"
        if any((use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                use_int4_w4a16, use_int4_w4a8, per_channel_quant)):
            raise ValueError(f"{mx_name} cannot be combined with other quantization flags")
        if w1_zp is not None or w2_zp is not None:
            raise ValueError(f"{mx_name} requires UE8M0 weight scales and no zero points")
        return fused_experts_mxfp_impl(
            hidden_states, w1, w2, topk_weights, topk_ids,
            output_dtype=output_dtype, inplace=inplace,
            activation=activation, is_gated=is_gated, b1=b1, b2=b2,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts, expert_map=expert_map,
            w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=a1_scale, a2_scale=a2_scale, block_shape=block_shape,
            no_combine=no_combine, routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha, gemm1_beta=gemm1_beta, gemm1_limit=gemm1_limit,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            use_mxfp4_w4a8=use_mxfp4_w4a8, use_mxfp8=use_mxfp8,
        )

    if use_fp8_w8a8:
        # Ordinary Triton GEMMs consume decoded floating-point multipliers.
        # Encoded E8M0 scales must be decoded during weight preparation.
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        for name, scale in (("w1_scale", w1_scale), ("w2_scale", w2_scale),
                            ("a1_scale", a1_scale), ("a2_scale", a2_scale)):
            if scale is not None and (not scale.is_floating_point()
                                      or scale.dtype == e8m0_dtype):
                raise ValueError(
                    f"{name} must be a decoded floating-point multiplier for FP8; "
                    "convert checkpoint UE8M0 scales during weight preparation "
                    "with moe_compat._upcast_e8m0_to_fp32")
    if use_ue8m0 and (not use_fp8_w8a8 or block_shape is None or per_channel_quant):
        raise ValueError("use_ue8m0 requires use_fp8_w8a8=True, block_shape, and per_channel_quant=False")
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)
    if gemm1_beta is not None:
        raise ValueError("gemm1_beta requires native MXFP8 or static MXFP4")
    activation_out_dim = adjust_N_for_activation(w1.shape[1], is_gated)

    if output_dtype is None:
        output_dtype = hidden_states.dtype

    # Check constraints.
    if use_int4_w4a16 or use_int4_w4a8:
        assert hidden_states.shape[1] == 2 * w1.shape[2], "Hidden size mismatch"
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
        global_num_experts = expert_map.numel() if expert_map is not None else E
    topk = topk_ids.shape[1]

    if output_dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif output_dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = int(os.environ.get("TRITON_FUSED_MOE_CHUNK_SIZE", "32768"))
    if CHUNK_SIZE <= 0:
        raise ValueError("TRITON_FUSED_MOE_CHUNK_SIZE must be positive")
    M = min(num_tokens, CHUNK_SIZE)
    block_quantizer = None
    if block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
        activation_block_k = block_shape[1]
        block_quantizer = per_block_quant_wrapper(
            (1, activation_block_k),
            scale_ue8m0=use_ue8m0,
        )(per_token_quant_triton)

    moe_config_func = get_optimal_moe_config_func(
        hidden_states, w1, topk_ids,
        use_int8_w8a16=use_int8_w8a16,
        use_int8_w8a8=use_int8_w8a8,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        use_mxfp4_w4a4=False,
        block_shape=block_shape,
        is_bottom=False,
        is_gated=is_gated,
    )
    moe_config_func2 = get_optimal_moe_config_func(
        hidden_states, w2, topk_ids,
        use_int8_w8a16=use_int8_w8a16,
        use_int8_w8a8=use_int8_w8a8,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        use_mxfp4_w4a4=False,
        block_shape=block_shape,
        is_bottom=True,
    )
    config = moe_config_func(M)
    config2, max_block_m = moe_config_func2(M)
    if config["BLOCK_SIZE_M"] != config2["BLOCK_SIZE_M"]:
        raise ValueError(
            "Top and bottom MoE configs must use the same BLOCK_SIZE_M: "
            f"top={config['BLOCK_SIZE_M']}, bottom={config2['BLOCK_SIZE_M']}"
        )

    bottom_moe_a_sorted = _bottom_a_uses_sorted_layout(config2, use_int4_w4a8)
    bottom_moe_a_use_mls = _bottom_a_uses_mls(config2, use_int4_w4a8)
    config2_kernel = config2.copy()
    config2_kernel.pop("bottom_a_use_mls_load", None)
    config2_kernel.pop("a_sorted", None)

    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if bottom_moe_a_sorted else 0
    )
    total_tokens = M * topk + max_padded_tokens

    cache_dtype = output_dtype
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache_width = max(N, w2.shape[1])
    if expert_map is not None:
        cache13 = torch.zeros(total_tokens * cache_width,
                            device=hidden_states.device,
                            dtype=cache_dtype)
    else:
        cache13 = torch.empty(total_tokens * cache_width,
                            device=hidden_states.device,
                            dtype=cache_dtype)
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
            bottom_moe_a_sorted = _bottom_a_uses_sorted_layout(
                config2, use_int4_w4a8
            )
            bottom_moe_a_use_mls = _bottom_a_uses_mls(
                config2, use_int4_w4a8
            )
            config2_kernel = config2.copy()
            config2_kernel.pop("bottom_a_use_mls_load", None)
            config2_kernel.pop("a_sorted", None)
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if bottom_moe_a_sorted
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
                dtype=cache_dtype)
        else:
            intermediate_cache2 = torch.empty(
                (total_tokens, activation_out_dim),
                device=hidden_states.device,
                dtype=cache_dtype)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        ck_sorting = True
        sorted_weights = None
        num_routes = tokens_in_chunk * topk
        naive_block_assignment = _should_use_naive_block_assignment(
            num_routes=num_routes,
            global_num_experts=global_num_experts,
            has_expert_map=expert_map is not None,
            bottom_a_sorted=bottom_moe_a_sorted,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_int4_w4a8=use_int4_w4a8,
            use_block_quant=block_shape is not None,
        )
        if naive_block_assignment:
            assert curr_topk_weights.is_contiguous(), (
                "Top-k weights must be contiguous for naive block assignment"
            )
            route_ids = curr_topk_ids.contiguous().view(-1)
            sorted_token_ids = route_ids
            expert_ids = route_ids
            num_tokens_post_padded = route_ids
            ck_sorting = False
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
                input_q,input_scale = per_token_quant_triton(curr_hidden_states,quant_dtype=quant_dtype)
            else:
                input_q = curr_hidden_states
                input_scale = (
                    a1_scale[begin_chunk_idx:end_chunk_idx]
                    if a1_scale is not None and a1_scale.ndim == 2 else a1_scale
                )
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
                                    c_sorted=bottom_moe_a_sorted,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    naive_block_assignment=naive_block_assignment,
                                    B_bias=b1,
                                    config=config)
        elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype==torch.bfloat16:
                input_q, input_scale = block_quantizer(curr_hidden_states, quant_dtype=quant_dtype)
            else:
                input_q, input_scale = curr_hidden_states, (
                    None if a1_scale is None else a1_scale[begin_chunk_idx:end_chunk_idx]
                )
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
                                    c_sorted=bottom_moe_a_sorted,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    naive_block_assignment=naive_block_assignment,
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
                                    c_sorted=bottom_moe_a_sorted,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    naive_block_assignment=naive_block_assignment,
                                    B_bias=b1,
                                    config=config)
        per_channel_a8 = (
            (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8)
            and per_channel_quant
        )
        activation_quant_dtype = (
            torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
        )
        fuse_activation_quant = (
            _should_use_fused_activation_quant(
                use_int8_w8a8=use_int8_w8a8,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int4_w4a8=use_int4_w4a8,
                per_channel_quant=per_channel_quant,
                has_expert_map=expert_map is not None,
                activation=activation,
                is_gated=is_gated,
                output_dim=activation_out_dim,
                gemm1_alpha=gemm1_alpha,
                gemm1_limit=gemm1_limit,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        )
        needs_router_weight = not (
            apply_router_weight_on_input or no_combine
        )
        fold_router_weight = _should_fold_router_weight(
            fuse_activation_quant=fuse_activation_quant,
            needs_router_weight=needs_router_weight,
        )
        activation_input = intermediate_cache1.view(-1, N)
        row_multiplier = None
        if fold_router_weight:
            if bottom_moe_a_sorted:
                if sorted_weights is None:
                    raise RuntimeError(
                        "sorted activation layout requires sorted router weights"
                    )
                # Both buffers conservatively cover every valid padded route,
                # but their allocation upper bounds are not always identical.
                # Quantize only their common valid capacity.
                rows = min(activation_input.shape[0], sorted_weights.numel())
                activation_input = activation_input[:rows]
                row_multiplier = sorted_weights[:rows]
            else:
                row_multiplier = curr_topk_weights.contiguous().view(-1)
        if fuse_activation_quant:
            activation_q, activation_scale = triton_activation_quant(
                activation_input,
                activation,
                is_gated,
                activation_quant_dtype,
                gemm1_alpha=gemm1_alpha,
                gemm1_limit=gemm1_limit,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
                row_multiplier=row_multiplier,
            )
        else:
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
        if expert_map is not None:
            # for EP mode, intermediate_cache1 and intermediate_cache3 need be zeros inited
            # since intermediate_cache1 and intermediate_cache3 shared same buffer,
            # to make sure intermediate_cache3 is zeros inited,
            # intermediate_cache1 need inited to zeros after silu_and_mul
            intermediate_cache1.fill_(0)

        gemm2_mul_router_weight = (
            needs_router_weight and not fold_router_weight
        )

        if per_channel_a8:
            if not fuse_activation_quant:
                activation_q, activation_scale = per_token_quant_triton(
                    intermediate_cache2, quant_dtype=activation_quant_dtype
                )
            invoke_fused_moe_kernel(activation_q,
                                    w2,
                                    intermediate_cache3,
                                    activation_scale,
                                    w2_scale,
                                    w2_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    gemm2_mul_router_weight,
                                    1,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    a_sorted=bottom_moe_a_sorted,
                                    bottom_a_use_mls_load=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    naive_block_assignment=naive_block_assignment,
                                    scale_bias_with_routed_weight=needs_router_weight,
                                    B_bias=b2,
                                    config=config2_kernel)
        elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
            quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
            activation_q, activation_scale = block_quantizer(intermediate_cache2, quant_dtype=quant_dtype)
            invoke_fused_moe_kernel(activation_q,
                                    w2,
                                    intermediate_cache3,
                                    activation_scale,
                                    w2_scale,
                                    w2_zp,
                                    curr_topk_weights,
                                    curr_topk_ids,
                                    sorted_token_ids,
                                    sorted_weights,
                                    expert_ids,
                                    num_tokens_post_padded,
                                    gemm2_mul_router_weight,
                                    1,
                                    compute_type=compute_type,
                                    use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16,
                                    use_int4_w4a8=use_int4_w4a8,
                                    per_channel_quant=per_channel_quant,
                                    block_shape=block_shape,
                                    a_sorted=bottom_moe_a_sorted,
                                    bottom_a_use_mls_load=bottom_moe_a_use_mls,
                                    ck_sorting=ck_sorting,
                                    ck_topk=topk,
                                    naive_block_assignment=naive_block_assignment,
                                    scale_bias_with_routed_weight=needs_router_weight,
                                    B_bias=b2,
                                    config=config2_kernel)
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
                        gemm2_mul_router_weight,
                        1,
                        compute_type=compute_type,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int8_w8a8=use_int8_w8a8,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int4_w4a16=use_int4_w4a16,
                        use_int4_w4a8=use_int4_w4a8,
                        per_channel_quant=per_channel_quant,
                        block_shape=block_shape,
                        a_sorted=bottom_moe_a_sorted,
                        bottom_a_use_mls_load=bottom_moe_a_use_mls,
                        ck_sorting=ck_sorting,
                        ck_topk=topk,
                        naive_block_assignment=naive_block_assignment,
                        scale_bias_with_routed_weight=needs_router_weight,
                        B_bias=b2,
                        config=config2_kernel)
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
