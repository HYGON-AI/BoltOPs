# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Includes an AITER-derived quantization helper (MIT); attribution below.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/ab0e5ef7954190de2a74a6715d1382b08f5accc4/vllm/model_executor/layers/quantization/utils/fp8_utils.py
"""vLLM FP8/UE8M0 activation quantization for the MXFP4 recipe.

Source: vllm ab0e5ef7954190de2a74a6715d1382b08f5accc4,
vllm/model_executor/layers/quantization/utils/fp8_utils.py.
The g128 path retains the vLLM quantization recipe and launch policy.
The g32 path uses AITER MXFP8 scale selection and unpacked UE8M0 storage.
"""
import torch
import triton
import triton.language as tl


# The MXFP8 scale helper below is adapted from AITER (MIT).
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# AITER b0043b6c778f0378e8892311e575e1ff898039a6:
# aiter/ops/triton/_triton_kernels/quant/quant.py::_mxfp8_quant_op.
# Upstream reference: https://github.com/ROCm/aiter/blob/b0043b6c778f0378e8892311e575e1ff898039a6/aiter/ops/triton/_triton_kernels/quant/quant.py
@triton.jit
def _mxfp8_quant_op(x_grouped, QUANT_AXIS: tl.constexpr):
    """AITER g32 scale selection; distinct from DeepGEMM's ceil(amax/448)."""
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale


@triton.jit
def _per_group_fp8_mxfp4_quant_kernel(X, Q, S, GROUP_SIZE: tl.constexpr = 32):
    group = tl.program_id(0).to(tl.int64)
    offsets = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    x = tl.load(X + offsets).to(tl.float32)
    if GROUP_SIZE == 32:
        exponent, quant_scale = _mxfp8_quant_op(x, QUANT_AXIS=0)
        q = (x * quant_scale).to(Q.dtype.element_ty)
        exponent = tl.reshape(exponent, ())
    else:
        maximum = tl.maximum(tl.max(tl.abs(x)), 1.e-10)
        raw = tl.maximum(tl.div_rn(maximum, 448.), 1.e-10)
        bits = raw.to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(tl.uint32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(x * (1. / scale), -448., 448.).to(Q.dtype.element_ty)
    tl.store(Q + offsets, q)
    tl.store(S + group, exponent.to(tl.uint8))


def per_token_group_quant_fp8_mxfp4(x, *, quant_dtype=torch.float8_e4m3fn, group_size=32):
    """Quantize with DeepGEMM g128 or AITER g32 UE8M0 scales.

    Reference: vLLM 77f58e530450416d217e4816727a35910bf3d9bf,
    csrc/quantization/w8a8/fp8/per_token_group_quant.cu,
    per_token_group_quant_8bit_packed_register_kernel. Keep its scale floor,
    integer exponent rounding and reciprocal multiplication. HCU consumes
    contiguous exponent bytes directly instead of DeepGEMM's packed TMA SF.
    """
    if group_size not in (32, 128):
        raise ValueError('MXFP4 activation groups must be 32 (AITER) or 128 (DeepGEMM)')
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % group_size:
        raise ValueError('MXFP4 activation quantization requires contiguous '
                         '[rows,K], K divisible by group_size')
    if quant_dtype != torch.float8_e4m3fn:
        raise ValueError('MXFP4 activation quantization requires E4M3')
    q = torch.empty_like(x, dtype=quant_dtype)
    scales = torch.empty((x.shape[0], x.shape[1] // group_size), dtype=torch.uint8, device=x.device)
    if x.numel():
        _per_group_fp8_mxfp4_quant_kernel[(x.numel() // group_size,)](
            x, q, scales, group_size, num_warps=4)
    return q, scales


@triton.jit
def _silu_mul_quant_fp8_packed_kernel(
    input_ptr,
    output_q_ptr,
    output_scale_ptr,
    expert_ends_ptr,
    M,
    input_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    clamp_limit,
    alpha,
    beta,
    N: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    PACKS_PER_ROW: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKS_PER_CTA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    EXPERT_ALIGNMENT: tl.constexpr,
    USE_MXFP8_QUANT: tl.constexpr,
):
    GROUPS_PER_PACK: tl.constexpr = 4
    hidden_size: tl.constexpr = N // 2

    pack_tile = tl.program_id(0)
    row_start = tl.program_id(1).to(tl.int64) * BLOCK_M
    row_step = tl.num_programs(1).to(tl.int64) * BLOCK_M

    if EXPERT_ALIGNMENT:
        expert = tl.program_id(2)
        previous_end = tl.load(expert_ends_ptr + expert - 1, expert > 0, other=0)
        expert_start = tl.cdiv(previous_end, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT
        row_start += expert_start.to(tl.int64)
        M = tl.load(expert_ends_ptr + expert)

    groups_per_cta: tl.constexpr = PACKS_PER_CTA * GROUPS_PER_PACK
    elems_per_cta: tl.constexpr = groups_per_cta * GROUP_SIZE
    col_start = pack_tile * elems_per_cta
    col_offsets = tl.arange(0, elems_per_cta)
    row_offsets = tl.arange(0, BLOCK_M)
    pack_offsets = tl.arange(0, PACKS_PER_CTA)

    col_mask = (col_start + col_offsets) < (GROUPS_PER_ROW * GROUP_SIZE)

    # persistent with grid_m-stride loop
    while row_start < M:
        rows = row_start + row_offsets
        row_mask = rows < M
        input_row_start = rows[:, None] * input_stride_m
        output_row_start = rows[:, None] * output_q_stride_m

        gate_flat = tl.load(
            input_ptr + input_row_start + col_start + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        up_flat = tl.load(
            input_ptr
            + input_row_start
            + hidden_size
            + col_start
            + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        gate = tl.reshape(gate_flat, (BLOCK_M, groups_per_cta, GROUP_SIZE)).to(
            tl.float32
        )
        up = tl.reshape(up_flat, (BLOCK_M, groups_per_cta, GROUP_SIZE)).to(tl.float32)

        if HAS_CLAMP:
            gate = tl.minimum(gate, clamp_limit)
            up = tl.clamp(up, -clamp_limit, clamp_limit)

        # Unified gated activation: silu == swigluoai with alpha=1, beta=0.
        #   glu = gate * sigmoid(alpha * gate); y = (up + beta) * glu
        glu = gate / (1.0 + tl.exp(-gate * alpha))
        y = (up + beta) * glu
        if USE_MXFP8_QUANT:
            # Match AITER's GEMM writeback: quantize the FP32 activation
            # directly, without the BF16 intermediate of vLLM's g128 path.
            scale_byte, quant_scale = _mxfp8_quant_op(y, QUANT_AXIS=2)
            scale_byte = tl.reshape(scale_byte, (BLOCK_M, groups_per_cta))
            y_q = y * quant_scale
        else:
            # Round through bf16 to match the upstream vLLM unfused path.
            y = y.to(tl.bfloat16).to(tl.float32)
            absmax = tl.max(tl.abs(y), axis=2)
            scale_raw = tl.maximum(absmax / fp8_max, 1e-10)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            y_q = tl.clamp(y / scale[:, :, None], fp8_min, fp8_max)
            scale_byte = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)

        y_q_flat = tl.reshape(y_q, (BLOCK_M, elems_per_cta))
        tl.store(
            output_q_ptr + output_row_start + col_start + col_offsets[None, :],
            y_q_flat.to(output_q_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

        if USE_MXFP8_QUANT:
            scale_group = pack_tile * groups_per_cta + tl.arange(0, groups_per_cta)
            scale_ptrs = (output_scale_ptr + scale_group[None, :] * output_scale_stride_k
                          + rows[:, None])
            tl.store(scale_ptrs, scale_byte,
                     row_mask[:, None] & (scale_group[None, :] < GROUPS_PER_ROW))
        else:
            scale_bytes = tl.reshape(scale_byte, (BLOCK_M, PACKS_PER_CTA, GROUPS_PER_PACK))
            shifts = tl.arange(0, GROUPS_PER_PACK) * 8
            packed_scale = tl.sum(scale_bytes << shifts[None, None, :], axis=2)
            scale_pack = pack_tile * PACKS_PER_CTA + pack_offsets
            scale_ptrs = (output_scale_ptr + scale_pack[None, :] * output_scale_stride_k
                          + rows[:, None])
            tl.store(scale_ptrs, packed_scale,
                     row_mask[:, None] & (scale_pack[None, :] < PACKS_PER_ROW))
        row_start += row_step


def silu_mul_quant_fp8_packed_triton(
    input: torch.Tensor,
    group_size: int = 32,
    output_q: torch.Tensor | None = None,
    clamp_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    *,
    expert_ends: torch.Tensor | None = None,
    expert_alignment: int = 0,
    use_mxfp8_quant: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse activation and FP8 quantization, optionally skipping expert padding.

    ``expert_ends`` uses DeepGEMM's prefix-sum layout: each entry is the
    exclusive end of an expert's live rows, including earlier alignment gaps.
    An expert starts at the preceding end rounded up to ``expert_alignment``.
    Padding outputs are left unwritten and must not be consumed as live rows.
    By default g32 selects AITER rounding and unpacked uint8 scales; g128
    preserves vLLM rounding and packed int32 storage. Explicit False retains
    the original vLLM quantizer for callers requesting other group sizes.
    Without ``expert_ends``, the original dense launch configuration is used
    and the metadata loads are compiled out.
    """
    if use_mxfp8_quant is None:
        use_mxfp8_quant = group_size == 32
    if use_mxfp8_quant and group_size != 32:
        raise ValueError("AITER MXFP8 quantization requires group_size=32")
    assert input.dim() == 2
    assert input.is_contiguous()

    M, N = input.shape
    hidden_size = N // 2

    assert hidden_size % group_size == 0

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    groups_per_row = hidden_size // group_size
    groups_per_pack = 4  # pack 4 UE8M0 scales to a single INT32
    packs_per_row = triton.cdiv(groups_per_row, groups_per_pack)

    if output_q is None:
        output_q = torch.empty((M, hidden_size), dtype=fp8_dtype, device=input.device)

    aligned_m = triton.cdiv(M, 4) * 4
    output_scale_packed = torch.empty(
        (groups_per_row if use_mxfp8_quant else packs_per_row, aligned_m),
        dtype=torch.uint8 if use_mxfp8_quant else torch.int32,
        device=input.device,
    ).T[:M, :]

    # Tuned for group_size=32 (MXFP8) and group_size=128 (DeepSeek-V4)
    num_warps = 4
    num_stages = 2
    if group_size < 128:
        BM = 1
        packs_per_cta = 8
    else:
        BM = 1 if M < 512 else 4
        packs_per_cta = 2 if M < 512 else 1

    grid_n = triton.cdiv(packs_per_row, packs_per_cta)
    grid_m = min(triton.cdiv(M, BM), 4096)
    grid: tuple[int, ...] = (grid_n, grid_m)
    if expert_ends is not None:
        assert expert_ends.ndim == 1 and expert_ends.numel() > 0
        assert expert_ends.dtype == torch.int32
        assert expert_ends.device == input.device and expert_ends.is_contiguous()
        assert expert_alignment > 0
        num_experts = expert_ends.numel()
        # Bound empty CTAs while retaining enough parallelism for full experts.
        max_ctas = 4096 if group_size < 128 else 8192
        grid_m = min(
            triton.cdiv(M, BM * num_experts),
            max(1, triton.cdiv(max_ctas, grid_n * num_experts)),
        )
        grid = (grid_n, grid_m, num_experts)
    else:
        assert expert_alignment == 0

    has_clamp = clamp_limit is not None
    _silu_mul_quant_fp8_packed_kernel[grid](
        input,
        output_q,
        output_scale_packed,
        expert_ends,
        M,
        input.stride(0),
        output_q.stride(0),
        output_scale_packed.stride(1),
        clamp_limit if has_clamp else 0.0,
        alpha,
        beta,
        N=N,
        GROUPS_PER_ROW=groups_per_row,
        PACKS_PER_ROW=packs_per_row,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        GROUP_SIZE=group_size,
        PACKS_PER_CTA=packs_per_cta,
        BLOCK_M=BM,
        HAS_CLAMP=has_clamp,
        EXPERT_ALIGNMENT=expert_alignment,
        USE_MXFP8_QUANT=use_mxfp8_quant,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_q, output_scale_packed
