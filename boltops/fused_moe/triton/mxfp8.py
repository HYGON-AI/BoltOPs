# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/77f58e530450416d217e4816727a35910bf3d9bf/vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py
"""E4M3 x E4M3 MoE with native E8M0 scales.

Algorithm references (vLLM 77f58e530450416d217e4816727a35910bf3d9bf):
- model_executor/layers/fused_moe/experts/mxfp8_native_moe.py:
  native K32 scaled-dot flow, GEMM1 output cast and FP32 weighted reduction.
- model_executor/layers/quantization/utils/mxfp8_utils.py:
  _mxfp8_quant_triton_kernel, ceil-scale input quantization.
- models/minimax_m3/amd/ops/swiglu_oai.py:
  _swiglu_oai_quant_kernel, split gate/up activation fused with quantization.
Local adaptations reuse the BoltOPs route ABI, expert orchestration and JSON
tuning in the ordinary MoE kernel's non-COMBINE_SCALE_LOAD branch.
A/B scales are uint8 E8M0 per K32, as in upstream native MXFP8 MoE.
No FNUZ, G128/block broadcast, or alternative scale storage is supported.
"""
import torch
import triton
import triton.language as tl

from . import moe_config_utils


SCALE_RECIPE = 'vllm-a32-b1x32-e8m0-e8m0'


def get_mxfp8_moe_config(num_tokens, **selectors):
    return moe_config_utils.get_native_mx_moe_config(
        num_tokens, dtype='mxfp8_w8a8', scale_recipe=SCALE_RECIPE, **selectors)


def moe_gemm_mxfp8(a, b, c, a_scale, b_scale, topk_ids,
                   sorted_token_ids, expert_ids, num_tokens_post_padded,
                   *, kernel, top_k, block_shape=(1, 32), ck_sorting=False, ck_topk=1,
                   block_size_m=16, block_size_n=64,
                   topk_weights=None, mul_routed_weight=False, sorted_weights=None):
    """Consume original E4M3 values and multiplicative E8M0 scales.

    A [rows,K], B [E,N,K]; uint8 E8M0 SA [rows,K/32], SB [E,N,K/32].
    Scale bytes 0..254 encode powers of two; 255 denotes NaN. Checkpoint
    layout preparation stays outside experts, as in the upstream backend.
    """
    if (a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn
            or a.ndim != 2 or b.ndim != 3
            or not a.is_contiguous() or not b.is_contiguous()):
        raise ValueError('MXFP8 requires contiguous E4M3 A[rows,K], B[E,N,K]')
    if tuple(block_shape) != (1, 32):
        raise ValueError('native MXFP8 requires weight block_shape=[1,32]')
    if a.shape[1] != b.shape[2] or a.shape[1] % 32:
        raise ValueError('MXFP8 requires matching K divisible by 32')
    for name, scale, shape in (
        ('A', a_scale, (a.shape[0], a.shape[1] // 32)),
        ('B', b_scale, (b.shape[0], b.shape[1], a.shape[1] // 32)),
    ):
        if scale is None or scale.dtype != torch.uint8 or tuple(scale.shape) != shape:
            raise ValueError(f'native MXFP8 {name} scale must be uint8 E8M0 with shape {shape}')
    if (topk_ids.ndim != 2 or topk_ids.shape[1] == 0
            or top_k not in (1, topk_ids.shape[1])
            or a.shape[0] * top_k < topk_ids.numel()):
        raise ValueError('MXFP8 requires valid topk routes and sufficient A rows')
    if ck_sorting and ck_topk != topk_ids.shape[1]:
        raise ValueError('ck_topk must match routing topk')
    for metadata in (sorted_token_ids, expert_ids, num_tokens_post_padded):
        if metadata.dtype != torch.int32 or metadata.ndim != 1 or not metadata.is_contiguous():
            raise ValueError('routing metadata must be contiguous int32 vectors')
    if num_tokens_post_padded.numel() != 1:
        raise ValueError('num_tokens_post_padded must contain one count')
    if (c.numel() != topk_ids.numel() * b.shape[1] or not c.is_contiguous()
            or c.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        raise ValueError('MXFP8 C must provide contiguous BF16/FP16/FP32 [routes,N] storage')
    tensors = (b, c, a_scale, b_scale, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded)
    if a.device.type != 'cuda' or any(x.device != a.device for x in tensors):
        raise ValueError('all MXFP8 tensors must share a CUDA/HIP device')
    if mul_routed_weight and ck_sorting:
        if (sorted_weights is None or sorted_weights.device != a.device
                or sorted_weights.numel() < sorted_token_ids.numel()):
            raise ValueError('CK routing requires sorted_weights matching sorted_token_ids')
        sorted_weights = sorted_weights.to(torch.float32).contiguous()
    if mul_routed_weight and (topk_weights is None or topk_weights.shape != topk_ids.shape
                             or topk_weights.device != a.device):
        raise ValueError('MXFP8 routing weights must match topk_ids')
    if mul_routed_weight and not ck_sorting:
        # vLLM native passes topk_weights.reshape(-1).to(float32). The regular
        # kernel addresses flattened routes, so preserve that contiguous ABI.
        topk_weights = topk_weights.to(torch.float32).contiguous()
    if block_size_m not in (16, 32, 64, 128) or block_size_n not in (64, 128):
        raise ValueError('MXFP8 requires BM=16/32/64/128 and BN=64/128')
    if not c.numel():
        return c
    grid = lambda meta: (triton.cdiv(sorted_token_ids.numel(), meta['BLOCK_SIZE_M'])
                         * triton.cdiv(b.shape[1], meta['BLOCK_SIZE_N']),)
    kernel[grid](
        a, b, c, topk_weights, num_tokens_post_padded, expert_ids,
        sorted_token_ids, sorted_weights, a_scale, b_scale, b,
        b.shape[1], a.shape[1], sorted_token_ids.numel(), topk_ids.numel(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
        b.shape[1], 1, *a_scale.stride(),
        b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
        0, 0, a.shape[0],
        group_n=1, group_k=32,
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=128, GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
        USE_MLS_LOAD=False, MUL_ROUTED_WEIGHT=mul_routed_weight,
        USE_ADDR_OFFSET_INT64_A=True, USE_ADDR_OFFSET_INT64_B=True,
        USE_ADDR_OFFSET_INT64_C=True, top_k=top_k,
        compute_type=tl.float32 if c.dtype == torch.float32 else (
            tl.bfloat16 if c.dtype == torch.bfloat16 else tl.float16),
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        per_channel_quant=False, c_sorted=False, a_sorted=False,
        bottom_a_use_mls_load=False, ck_sorting=ck_sorting, ck_topk=ck_topk,
        naive_block_assignment=False, NUM_XCDS=1,
        SCALE_BIAS_WITH_ROUTED_WEIGHT=False, ADD_BIAS=False,
        use_mxfp8=True,
        num_warps=4, num_stages=1)
    return c


# Upstream reference: https://github.com/vllm-project/vllm/blob/77f58e530450416d217e4816727a35910bf3d9bf/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
# Upstream reference: https://github.com/vllm-project/vllm/blob/77f58e530450416d217e4816727a35910bf3d9bf/vllm/models/minimax_m3/amd/ops/swiglu_oai.py
@triton.jit
def _quantize_mxfp8_kernel(X, Q, S, ROWS, COLS: tl.constexpr,
                           INPUT_STRIDE: tl.constexpr,
                           GATED: tl.constexpr,  # X holds split [gate,up]; COLS is the output width.
                           ALPHA: tl.constexpr,  # Sigmoid coefficient; 1 for SiLU, typically 1.702 for OAI.
                           BETA: tl.constexpr,  # Up-branch offset; 0 for SiLU, 1 for OAI.
                           HAS_LIMIT: tl.constexpr,  # Apply pre-sigmoid gate/up clamp only when configured.
                           LIMIT: tl.constexpr,  # Gate upper bound and symmetric up bound.
                           BM: tl.constexpr = 16):
    # vLLM mxfp8_utils._mxfp8_quant_triton_kernel and
    # minimax_m3/amd/ops/swiglu_oai._swiglu_oai_quant_kernel.
    row = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    col = tl.program_id(1) * 32 + tl.arange(0, 32)
    value = tl.load(X + row[:, None] * INPUT_STRIDE + col[None, :],
                    row[:, None] < ROWS, other=0.).to(tl.float32)
    if GATED:
        up = tl.load(X + row[:, None] * INPUT_STRIDE + COLS + col[None, :],
                     row[:, None] < ROWS, other=0.).to(tl.float32)
        if HAS_LIMIT:
            value = tl.minimum(value, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        value = value * tl.sigmoid(ALPHA * value) * (up + BETA)
    amax = tl.maximum(tl.max(tl.abs(value), axis=1), 1.1754943508222875e-38)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.)) + 127., 0.), 254.)
    # The byte-0 divisor is subnormal; use the finite reciprocal, as upstream.
    q = (value * tl.exp2(127. - exponent)[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + row[:, None] * COLS + col[None, :], q, row[:, None] < ROWS)
    tl.store(S + row * (COLS // 32) + tl.program_id(1), exponent.to(tl.uint8), row < ROWS)


def mxfp8_e4m3_quantize(x, *, quant_dtype=torch.float8_e4m3fn,
                        activation=None, alpha=1., beta=0., limit=None):
    """vLLM K32 ceil-scale quantization; optionally fuse split gate/up activation.

    With activation='swigluoai' the formula is
    clamp(gate,max=limit) * sigmoid(alpha*gate) * (clamp(up,-limit,limit)+beta).
    activation='silu' is its alpha=1,beta=0 specialization. No intermediate
    activation cast is inserted. Inputs and checkpoint weights stay separate.
    """
    if x.ndim != 2 or not x.is_contiguous() or quant_dtype != torch.float8_e4m3fn:
        raise ValueError('MXFP8 quantization requires contiguous [rows,cols] and E4M3 output')
    if activation not in (None, 'silu', 'swigluoai'):
        raise ValueError('fused MXFP8 activation must be silu or swigluoai')
    cols = x.shape[1] // 2 if activation is not None else x.shape[1]
    if cols % 32 or (activation is not None and x.shape[1] % 2):
        raise ValueError('MXFP8 quantization requires output columns divisible by 32')
    q = torch.empty((x.shape[0], cols), dtype=quant_dtype, device=x.device)
    scale = torch.empty((x.shape[0], cols // 32), dtype=torch.uint8, device=x.device)
    if x.numel():
        _quantize_mxfp8_kernel[(triton.cdiv(x.shape[0], 16), cols // 32)](
            x, q, scale, x.shape[0], cols, x.stride(0), activation is not None,
            float(alpha), float(beta), limit is not None, 0. if limit is None else float(limit),
            num_warps=4, num_stages=1)
    return q, scale
