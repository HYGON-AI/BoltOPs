# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Includes AITER-derived portions (MIT).
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/77f58e530450416d217e4816727a35910bf3d9bf/vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py
# Upstream reference: https://github.com/ROCm/aiter/blob/b0043b6c778f0378e8892311e575e1ff898039a6/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py
"""Packed FP8 x MXFP4 MoE expert GEMM for AITER and DeepGEMM recipes.

The production path uses native dot_scaled on packed E2M1 with A g32
(AITER kernel), g128 (DeepGEMM), or static tensor FP8 (vLLM/AITER), and
B g32 UE8M0 scales. Packed weights are consumed directly without expanding
them into another numeric format.
"""
import torch
import triton
import triton.language as tl

from . import moe_config_utils


def get_mxfp4_moe_config(num_tokens, *, is_bottom=False, E=None, N=None,
                        K=None, top_k=None, scale_recipe="deepgemm-a128-b1x32"):
    return moe_config_utils.get_native_mx_moe_config(
        num_tokens, is_bottom=is_bottom, E=E, N=N, K=K, top_k=top_k,
        dtype="mxfp4_w4a8", scale_recipe=scale_recipe)


# Algorithm references:
# AITER b0043b6c778f0378e8892311e575e1ff898039a6,
# _triton_kernels/moe/moe_op_gemm_a8w4.py: native g32 activation/weight scales.
# vLLM selects DeepGemmFP4Experts for dynamic g128 FP8:
# vllm ab0e5ef7954190de2a74a6715d1382b08f5accc4, experts/deep_gemm_moe.py
# DeepGEMM 5081f8513af57e61696de4f4f04b0e707ffb503e,
# impls/sm100_fp8_fp4_gemm_1d1d.cuh: BLOCK_K=128, UMMA_K=32,
# Algorithm reference (MIT), Copyright (c) 2025 DeepSeek:
# https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh
# block-scaled FP8 x packed FP4, one FP32 accumulator across K blocks.
# Static W4A8 wrapper reference:
# https://github.com/vllm-project/vllm/blob/77f58e530450416d217e4816727a35910bf3d9bf/vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py
# SwiGLU/static quant epilogue reference:
# https://github.com/ROCm/aiter/blob/b0043b6c778f0378e8892311e575e1ff898039a6/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py
# CUDA scheduling/TMA is adapted to Triton/HCU; the arithmetic is not replaced
# with an expanded-FP8 partial-dot algorithm.
@triton.jit
def _load_native_mx_scale(P, row, group, sr: tl.constexpr, sk: tl.constexpr,
                          mask, FORMAT: tl.constexpr):
    if FORMAT == 'packed_e8m0':
        value = tl.load(P + row * sr + group // 4 * sk, mask, other=0x7f7f7f7f)
        return ((value.to(tl.uint32) >> ((group % 4) * 8)) & 255).to(tl.uint8)
    if FORMAT == 'fp32_e8m0':
        value = tl.load(P + row * sr + group * sk, mask, other=1.)
        # Contract: already quantized positive normal powers of two. Never
        # round an arbitrary FP32 scale implicitly inside GEMM.
        return ((value.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.uint8)
    return tl.load(P + row * sr + group * sk, mask, other=127).to(tl.uint8)


@triton.jit
def _moe_gemm_mxfp4_kernel(A, B, SA, SB, C, Sorted, Experts, Padded,
                          K, N, ROUTES, stride_am: tl.constexpr,
                          stride_be: tl.constexpr, stride_bn: tl.constexpr,
                          stride_sam: tl.constexpr, stride_sak: tl.constexpr,
                          stride_sbe: tl.constexpr, stride_sbn: tl.constexpr,
                          stride_sbk: tl.constexpr,
                          TOP_K: tl.constexpr, CK: tl.constexpr, CK_TOPK: tl.constexpr,
                          A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr,
                          BM: tl.constexpr = 16, BN: tl.constexpr = 64,
                          A_GROUP_SIZE: tl.constexpr = 32,  # Dynamic A groups: K32 or K128; unused for static A.
                          Router=None, MUL_ROUTED_WEIGHT: tl.constexpr = False,
                          ROUTER_TOPK: tl.constexpr = 1,
                          stride_rwm: tl.constexpr = 0, stride_rwk: tl.constexpr = 0,
                          Bias=None,  # FP32 [E,N], added before activation or routing weights.
                          STATIC_A: tl.constexpr = False,  # SA is a scalar FP32 multiplier; A microscale is 1.
                          FUSE_SWIGLU: tl.constexpr = False,  # Static GEMM1 emits split-gate/up SwiGLU-OAI as FP8.
                          QuantScale=None,  # Scalar FP32 output scale; divide after optional gamma.
                          ALPHA: tl.constexpr = 1.702,  # Sigmoid coefficient in g*sigmoid(alpha*g)*(up+1).
                          LIMIT: tl.constexpr = 7.,  # Gate upper clamp and symmetric up clamp.
):
    tl.static_assert(A_GROUP_SIZE == 32 or A_GROUP_SIZE == 128)
    block = tl.program_id(0)
    if block * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + block)
    if expert < 0:
        return  # The caller zero-initializes non-local routes.
    slot = block * BM + tl.arange(0, BM)
    route = tl.load(Sorted + slot)
    if CK:
        token = route & 0x00ffffff
        rank = (route >> 24) & 255
        route = token * CK_TOPK + rank
    valid = (route >= 0) & (route < ROUTES)
    row = route // TOP_K
    if FUSE_SWIGLU:
        # BoltOPs stores gate/up as two halves. AITER's prepared weights are
        # interleaved; gather pairs here instead of repacking at invocation.
        output_n = tl.program_id(1) * BN + tl.arange(0, 2 * BN) // 2
        n = output_n + (tl.arange(0, 2 * BN) % 2) * (N // 2)
        valid_n = output_n < N // 2
    else:
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        valid_n = n < N
    rk = tl.arange(0, 128)
    pk = tl.arange(0, 64)
    g32 = tl.arange(0, 4)
    acc = tl.zeros((BM, 2 * BN if FUSE_SWIGLU else BN), tl.float32)
    for kb in range(tl.cdiv(K, 128)):
        a = tl.load(A + row[:, None].to(tl.int64) * stride_am + (kb * 128 + rk)[None, :],
                    valid[:, None] & ((kb * 128 + rk)[None, :] < K), other=0.)
        b = tl.load(B + expert.to(tl.int64) * stride_be + n[None, :].to(tl.int64) * stride_bn
                    + (kb * 64 + pk)[:, None],
                    valid_n[None, :] & ((kb * 64 + pk)[:, None] < K // 2), other=0)
        # Map original activation groups to the four K32 hardware groups.
        # AITER consumes four independent g32 scales; DeepGEMM broadcasts g128.
        a_group = (kb * 4 + g32) // (A_GROUP_SIZE // 32)
        if STATIC_A:
            sa = tl.full((BM, 4), 127, tl.uint8)
        else:
            sa = _load_native_mx_scale(SA, row[:, None], a_group[None, :],
                                       stride_sam, stride_sak,
                                       valid[:, None] & (a_group[None, :] < K // A_GROUP_SIZE), A_FORMAT)
        sb = _load_native_mx_scale(SB + expert.to(tl.int64) * stride_sbe,
                                  n[:, None], (kb * 4 + g32)[None, :],
                                  stride_sbn, stride_sbk,
                                  valid_n[:, None] & ((kb * 4 + g32)[None, :] < K // 32),
                                  B_FORMAT)
        acc = tl.dot_scaled(a, sa, 'e4m3', b, sb, 'e2m1', acc=acc)
    # vLLM 77f58e530450, experts/aiter_mxfp4_w4a8_moe.py and
    # AITER b0043b6c, _triton_kernels/moe/moe_op_gemm_a8w4.py:
    # static A scale -> bias -> SwiGLU -> gamma -> static FP8 quant.
    if STATIC_A:
        acc *= tl.load(SA)
    if Bias is not None:
        bias = tl.load(Bias + expert.to(tl.int64) * N + n, valid_n, other=0.)
        acc += bias[None, :]
    if FUSE_SWIGLU:
        gate, up = tl.split(acc.reshape(BM, BN, 2))
        gate = tl.minimum(gate, LIMIT)
        up = tl.clamp(up, -LIMIT, LIMIT)
        sigmoid_gate = gate / (1. + tl.exp2(-1.44269504089 * ALPHA * gate))
        acc = tl.fma(sigmoid_gate, up, sigmoid_gate)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        N = N // 2
        valid_n = n < N
    if MUL_ROUTED_WEIGHT:
        # AITER applies gamma to the FP32 accumulator before output casting.
        weight = tl.load(Router + (route // ROUTER_TOPK) * stride_rwm
                         + (route % ROUTER_TOPK) * stride_rwk, valid, other=0.).to(tl.float32)
        acc *= weight[:, None]
    if QuantScale is not None:
        acc = acc / tl.load(QuantScale)
    # DeepGEMM leaves routing to combine, after this output conversion.
    tl.store(C + route[:, None].to(tl.int64) * N + n[None, :], acc,
             valid[:, None] & valid_n[None, :])


def _scale_format(scale, shape):
    if scale is None:
        raise ValueError('MXFP4 GEMM requires A/B scales')
    formats = {torch.uint8: 'e8m0', torch.int32: 'packed_e8m0', torch.float32: 'fp32_e8m0'}
    if scale.dtype not in formats:
        raise ValueError('scales must be UE8M0 bytes, packed int32 or power-of-two FP32')
    expected = (*shape[:-1], triton.cdiv(shape[-1], 4)) if scale.dtype == torch.int32 else shape
    if tuple(scale.shape) != expected:
        raise ValueError(f'scale shape {tuple(scale.shape)} != {expected}')
    return formats[scale.dtype]


def activation_scale_group_size(a, scale):
    """Read the activation recipe from its explicit scale layout.

    AITER uses uint8 [rows,K/32]. Preserve the existing g128 byte/FP32 and
    packed-int32 contracts. G32 scales use unpacked bytes or FP32: a packed
    word can hide whether it contains one g128 scale or four g32 scales.
    """
    if a.ndim != 2 or a.shape[1] % 32:
        raise ValueError('MXFP4 activations must be [rows,K], K divisible by 32')
    if scale is None or scale.ndim != 2:
        raise ValueError('MXFP4 activation scales must be a [rows,groups] tensor')
    groups = (128,) if scale.dtype == torch.int32 else (32, 128)
    for group in groups:
        if a.shape[1] % group:
            continue
        count = a.shape[1] // group
        width = triton.cdiv(count, 4) if scale.dtype == torch.int32 else count
        if tuple(scale.shape) == (a.shape[0], width):
            _scale_format(scale, (a.shape[0], count))
            return group
    raise ValueError('MXFP4 activation scales must describe g128 or unpacked g32')


def moe_gemm_mxfp4(a, b, c, a_scale, b_scale, topk_ids,
                   sorted_token_ids, expert_ids, num_tokens_post_padded,
                   *, top_k, ck_sorting=False, ck_topk=1,
                   block_size_m=16, block_size_n=64,
                   topk_weights=None, mul_routed_weight=False,
                   bias=None,  # FP32 expert bias [E,N]; supported by the static recipe.
                   fuse_swiglu=False,  # Fuse static GEMM1 activation and FP8 output quantization.
                   quant_scale=None,  # Scalar FP32 divisor for fused output quantization.
                   alpha=1.702,  # SwiGLU-OAI sigmoid coefficient; up always has +1.
                   limit=7.,  # Clamp gate above and up on both sides before activation.
):
    """Packed W4A8 GEMM using DeepGEMM g128 or AITER g32 activation scales.

    Static SA is one FP32 multiplier; fused SwiGLU-OAI emits E4M3 directly
    using quant_scale. Bias is FP32 [E,N] before activation/routing.
    A=[rows,K] E4M3; B=[E,N,K/2] uint8/int8; dynamic SA=[rows,K/128] or
    AITER [rows,K/32]; SB=[E,N,K/32]. Scales may be UE8M0 uint8, packed int32
    (four consecutive exponents per word, g128 A only), or power-of-two FP32.
    FP32 power-of-two validity is a caller contract, as in the upstream SF
    transform; this wrapper does not silently round data or copy it to CPU.
    """
    if topk_ids.ndim != 2 or topk_ids.shape[1] == 0:
        raise ValueError('topk_ids must be [tokens, topk] with topk > 0')
    if top_k not in (1, topk_ids.shape[1]):
        raise ValueError('top_k must be 1 for GEMM2 or the routing topk for GEMM1')
    if ck_sorting and ck_topk != topk_ids.shape[1]:
        raise ValueError('ck_topk must match the routing topk')
    if num_tokens_post_padded.numel() != 1:
        raise ValueError('num_tokens_post_padded must contain one count')
    for metadata in (sorted_token_ids, expert_ids, num_tokens_post_padded):
        if metadata.dtype != torch.int32 or metadata.ndim != 1 or not metadata.is_contiguous():
            raise ValueError('routing metadata must be contiguous int32 vectors')
    if a.dtype != torch.float8_e4m3fn or b.dtype not in (torch.uint8, torch.int8):
        raise ValueError('packed MXFP4 GEMM requires E4M3 A and uint8/int8 packed B')
    if a.ndim != 2 or b.ndim != 3 or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError('packed MXFP4 GEMM requires contiguous A[rows,K], B[E,N,K/2]')
    if a.shape[0] * top_k < topk_ids.numel():
        raise ValueError('A does not contain all routed activation rows')
    k = a.shape[1]
    if k % 32 or b.shape[2] * 2 != k:
        raise ValueError('MXFP4 recipe requires packed weights and K divisible by 32')
    output_n = b.shape[1] // 2 if fuse_swiglu else b.shape[1]
    if c.numel() != topk_ids.numel() * output_n or not c.is_contiguous():
        raise ValueError('C must provide contiguous [routes,N] storage')
    static_a = a_scale is not None and a_scale.numel() == 1 and a_scale.ndim <= 1
    if static_a:
        if a_scale.dtype != torch.float32:
            raise ValueError('static MXFP4 A scale must be FP32')
        a_group_size, af = 32, 'static_fp32'
    else:
        a_group_size = activation_scale_group_size(a, a_scale)
        af = _scale_format(a_scale, (a.shape[0], k // a_group_size))
    if fuse_swiglu:
        if not static_a or quant_scale is None or c.dtype != torch.float8_e4m3fn:
            raise ValueError('fused static SwiGLU requires scalar A/output scales and E4M3 C')
        if b.shape[1] % 2 or quant_scale.numel() != 1 or quant_scale.dtype != torch.float32:
            raise ValueError('static SwiGLU requires paired gate/up and one FP32 output scale')
    elif quant_scale is not None or c.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('C must be BF16, FP16 or FP32 without fused static SwiGLU')
    if bias is not None and (not static_a or bias.dtype != torch.float32
                             or tuple(bias.shape) != tuple(b.shape[:2]) or not bias.is_contiguous()):
        raise ValueError('static MXFP4 bias must be contiguous FP32 [E,N]')
    bf = _scale_format(b_scale, (b.shape[0], b.shape[1], k // 32))
    if mul_routed_weight:
        if a_group_size != 32:
            raise ValueError('DeepGEMM g128 applies routing after output conversion')
        if (topk_weights is None or topk_weights.shape != topk_ids.shape
                or topk_weights.device != a.device):
            raise ValueError('AITER g32 routing requires topk_weights matching topk_ids')
    tensors = (b, c, a_scale, b_scale, topk_ids, sorted_token_ids, expert_ids,
               num_tokens_post_padded, bias, quant_scale)
    if a.device.type != 'cuda' or any(x is not None and x.device != a.device for x in tensors):
        raise ValueError('all MXFP4 GEMM tensors must share a CUDA/HIP device')
    if not c.numel():
        return c
    bm, bn = block_size_m, block_size_n
    if bm not in (16, 32, 64, 128):
        raise ValueError('MXFP4 sorting/GEMM block size must be 16, 32, 64 or 128')
    if bn not in (64, 128):
        raise ValueError('MXFP4 GEMM column tile must be 64 or 128')
    _moe_gemm_mxfp4_kernel[(triton.cdiv(sorted_token_ids.numel(), bm), triton.cdiv(output_n, bn))](
        a, b, a_scale, b_scale, c, sorted_token_ids, expert_ids, num_tokens_post_padded,
        k, b.shape[1], topk_ids.numel(), a.stride(0), b.stride(0), b.stride(1),
        0 if static_a else a_scale.stride(0), 0 if static_a else a_scale.stride(1), *b_scale.stride(),
        top_k, ck_sorting, ck_topk, af, bf, bm, bn,
        A_GROUP_SIZE=a_group_size, Router=topk_weights,
        MUL_ROUTED_WEIGHT=mul_routed_weight, ROUTER_TOPK=topk_ids.shape[1],
        stride_rwm=topk_weights.stride(0) if mul_routed_weight else 0,
        stride_rwk=topk_weights.stride(1) if mul_routed_weight else 0,
        Bias=bias, STATIC_A=static_a, FUSE_SWIGLU=fuse_swiglu,
        QuantScale=quant_scale, ALPHA=alpha, LIMIT=limit,
        num_warps=4, num_stages=1)
    return c
