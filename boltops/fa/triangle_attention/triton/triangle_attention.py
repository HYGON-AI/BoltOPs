# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BF16 Triangle Attention forward for pre-scaled ``[B, H, N, D]`` inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _repack_bias_kernel(
    bias_ptr,
    output_ptr,
    H,
    N,
    N_PAD,
    stride_bias_batch,
    stride_bias_head,
    stride_bias_query,
    stride_bias_key,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_index = tl.program_id(0)
    key_block = tl.program_id(1)
    batch_index = tl.program_id(2).to(tl.int64)
    head_indices = tl.arange(0, BLOCK_H)
    key_indices = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    head_mask = head_indices < H

    source_ptrs = (
        bias_ptr
        + batch_index * stride_bias_batch
        + head_indices[:, None] * stride_bias_head
        + query_index * stride_bias_query
        + key_indices[None, :] * stride_bias_key
    )
    bias_values = tl.load(
        source_ptrs,
        mask=head_mask[:, None] & (key_indices < N)[None, :],
        other=-float("inf"),
    )
    output_ptrs = (
        output_ptr
        + ((batch_index * H + head_indices[:, None]) * N + query_index) * N_PAD
        + key_indices[None, :]
    )
    tl.store(
        output_ptrs,
        bias_values,
        mask=head_mask[:, None] & (key_indices < N_PAD)[None, :],
    )


@triton.jit
def _attention_block(
    accumulator,
    row_sum,
    row_max,
    query,
    key_ptrs,
    value_ptrs,
    bias_ptrs,
    mask_ptrs,
    key_indices,
    N,
    dim_mask,
    MASK_K: tl.constexpr,
    BIAS_PADDED: tl.constexpr,
    FAST_P: tl.constexpr,
    FIRST_BLOCK: tl.constexpr = False,
):
    key_mask = key_indices < N
    if MASK_K:
        key = tl.load(key_ptrs, mask=key_mask[:, None] & dim_mask[None, :], other=0.0)
        mask_bias = tl.load(mask_ptrs, mask=key_mask, other=0.0)
        if BIAS_PADDED:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(
                bias_ptrs,
                mask=key_mask[None, :],
                other=-float("inf"),
            )
    else:
        key = tl.load(key_ptrs, mask=dim_mask[None, :], other=0.0)
        mask_bias = tl.load(mask_ptrs)
        bias = tl.load(bias_ptrs)

    # Fuse both additive biases into the QK dot accumulator.
    scores = tl.dot(
        query,
        tl.trans(key),
        bias.to(tl.float32) + mask_bias[None, :].to(tl.float32),
    )
    scaled_scores = scores * 1.4426950408889634

    if FIRST_BLOCK:
        new_row_max = tl.max(scaled_scores, axis=1)
    else:
        new_row_max = tl.maximum(row_max, tl.max(scaled_scores, axis=1))
        rescale = tl.math.exp2(row_max - new_row_max)
    probabilities = tl.math.exp2(scaled_scores - new_row_max[:, None])
    if FAST_P:
        # Round-half-up assumes finite probabilities; NaN/Inf handling differs.
        rounded_bits = probabilities.to(tl.uint32, bitcast=True) + 0x8000
        pv_probabilities = (rounded_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        pv_probabilities = probabilities.to(value_ptrs.dtype.element_ty)

    if FIRST_BLOCK:
        row_sum = tl.sum(probabilities, axis=1)
    else:
        row_sum = row_sum * rescale + tl.sum(probabilities, axis=1)
    # Load V after softmax to shorten its live range.
    if MASK_K:
        value = tl.load(value_ptrs, mask=key_mask[:, None] & dim_mask[None, :], other=0.0)
    else:
        value = tl.load(value_ptrs, mask=dim_mask[None, :], other=0.0)
    if FIRST_BLOCK:
        accumulator = tl.dot(pv_probabilities, value)
    else:
        accumulator = tl.dot(pv_probabilities, value, accumulator * rescale[:, None])
    return accumulator, row_sum, new_row_max


@triton.jit
def _triangle_attention_fwd_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    bias_ptr,
    mask_bias_ptr,
    output_ptr,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_bias_batch,
    stride_bias_head,
    stride_bias_query,
    stride_bias_key,
    stride_mask_batch,
    stride_mask_key,
    stride_out_batch,
    stride_out_head,
    stride_out_seq,
    stride_out_dim,
    B,
    H,
    N,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    BIAS_PADDED: tl.constexpr,
    FAST_P: tl.constexpr,
    PEEL_FIRST: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if HEAD_MAJOR:
        head_index = batch_head // B
        batch_index = batch_head - head_index * B
    else:
        batch_index = batch_head // H
        head_index = batch_head - batch_index * H
    batch_index = batch_index.to(tl.int64)

    query_indices = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    safe_query_indices = tl.minimum(query_indices, N - 1)
    key_offsets = tl.arange(0, BLOCK_N)
    dim_indices = tl.arange(0, BLOCK_D)
    dim_mask = dim_indices < D

    query_ptrs = (
        query_ptr
        + batch_index * stride_q_batch
        + head_index * stride_q_head
        + safe_query_indices[:, None] * stride_q_seq
        + dim_indices[None, :] * stride_q_dim
    )
    query = tl.load(query_ptrs, mask=dim_mask[None, :], other=0.0)
    key_ptrs = (
        key_ptr
        + batch_index * stride_k_batch
        + head_index * stride_k_head
        + key_offsets[:, None] * stride_k_seq
        + dim_indices[None, :] * stride_k_dim
    )
    value_ptrs = (
        value_ptr
        + batch_index * stride_v_batch
        + head_index * stride_v_head
        + key_offsets[:, None] * stride_v_seq
        + dim_indices[None, :] * stride_v_dim
    )
    # The batch stride is zero when bias is shared across batches.
    bias_ptrs = (
        bias_ptr
        + batch_index * stride_bias_batch
        + head_index * stride_bias_head
        + safe_query_indices[:, None] * stride_bias_query
        + key_offsets[None, :] * stride_bias_key
    )
    mask_ptrs = mask_bias_ptr + batch_index * stride_mask_batch + key_offsets * stride_mask_key

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    full_key_count = (N // BLOCK_N) * BLOCK_N

    if PEEL_FIRST and full_key_count > 0:
        accumulator, row_sum, row_max = _attention_block(
            accumulator,
            row_sum,
            row_max,
            query,
            key_ptrs,
            value_ptrs,
            bias_ptrs,
            mask_ptrs,
            key_offsets,
            N,
            dim_mask,
            False,
            BIAS_PADDED,
            FAST_P,
            FIRST_BLOCK=True,
        )
        key_ptrs += BLOCK_N * stride_k_seq
        value_ptrs += BLOCK_N * stride_v_seq
        bias_ptrs += BLOCK_N * stride_bias_key
        mask_ptrs += BLOCK_N * stride_mask_key

    first_regular_key = BLOCK_N if PEEL_FIRST else 0
    for key_start in range(first_regular_key, full_key_count, BLOCK_N):
        accumulator, row_sum, row_max = _attention_block(
            accumulator,
            row_sum,
            row_max,
            query,
            key_ptrs,
            value_ptrs,
            bias_ptrs,
            mask_ptrs,
            key_start + key_offsets,
            N,
            dim_mask,
            False,
            BIAS_PADDED,
            FAST_P,
        )
        key_ptrs += BLOCK_N * stride_k_seq
        value_ptrs += BLOCK_N * stride_v_seq
        bias_ptrs += BLOCK_N * stride_bias_key
        mask_ptrs += BLOCK_N * stride_mask_key

    if full_key_count < N:
        accumulator, row_sum, row_max = _attention_block(
            accumulator,
            row_sum,
            row_max,
            query,
            key_ptrs,
            value_ptrs,
            bias_ptrs,
            mask_ptrs,
            full_key_count + key_offsets,
            N,
            dim_mask,
            True,
            BIAS_PADDED,
            FAST_P,
        )

    output = accumulator * (1.0 / row_sum)[:, None]
    output_ptrs = (
        output_ptr
        + batch_index * stride_out_batch
        + head_index * stride_out_head
        + query_indices[:, None] * stride_out_seq
        + dim_indices[None, :] * stride_out_dim
    )
    tl.store(
        output_ptrs,
        output.to(output_ptr.dtype.element_ty),
        mask=(query_indices < N)[:, None] & dim_mask[None, :],
    )


def _prepare_bias(bias, block_n):
    bias_batch, num_heads, seq_len, _ = bias.shape
    padded_seq_len = triton.cdiv(seq_len, block_n) * block_n

    prepared_bias = torch.empty(
        (bias_batch, num_heads, seq_len, padded_seq_len),
        device=bias.device,
        dtype=bias.dtype,
    )

    block_k = 64 if num_heads <= 4 else 32
    grid = (seq_len, triton.cdiv(padded_seq_len, block_k), bias_batch)
    _repack_bias_kernel[grid](
        bias,
        prepared_bias,
        num_heads,
        seq_len,
        padded_seq_len,
        bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        BLOCK_H=triton.next_power_of_2(num_heads),
        BLOCK_K=block_k,
        num_warps=1,
    )
    return prepared_bias


_LAUNCH_CONFIG_KEYS = ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages")
_DEFAULT_VARIANT_CONFIG = {"BIAS_PREP": 2, "HEAD_MAJOR": 1, "FAST_P": 1}


def _get_triangle_config(batch, heads, seq_len, head_dim, device=None, mask_layout="contiguous"):
    from . import config

    kernel_config = config.get_kernel_config(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        mask_layout=mask_layout,
        device=device,
    )
    if torch.version.hip is not None and config.get_device_info(device).arch == "gfx936":
        kernel_config = dict(kernel_config, kpack=2)
    return kernel_config


def triangle_attention_fwd(q, k, v, biases, *, config=None):
    """Forward attention for production ``q/k/v=[B, H, N, D]``.

    ``q`` must already be scaled by ``1/sqrt(D)``.
    Q/K/V must be BF16. Floating-point biases retain their input dtype and
    are converted to FP32 for accumulation. ``mask_bias`` uses ``0`` for
    visible keys and finite ``-1e9`` for masked keys, stored as BF16, FP32,
    or FP64, with at least one visible key per query.
    Negative-infinity input masks are unsupported.
    ``BIAS_PREP``: 0 = use the bias as given, 1 = ``.contiguous()`` copy,
    2 = padded input-dtype repack. Bias and additive mask retain their dtype.
    FP32-scaled mode 3 is unsupported.
    """
    mask_bias, triangle_bias = biases
    batch_size, num_heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)

    kernel_config = (
        dict(config)
        if config is not None
        else _get_triangle_config(
            batch_size, num_heads, seq_len, head_dim, q.device,
            mask_layout="contiguous" if mask_bias.stride(-1) == 1 else "strided",
        )
    )
    variant_config = {
        key: kernel_config.pop(key, default)
        for key, default in _DEFAULT_VARIANT_CONFIG.items()
    }
    bias_prep_mode = variant_config["BIAS_PREP"]
    if bias_prep_mode not in (0, 1, 2):
        raise ValueError("BIAS_PREP must be 0, 1, or 2 to preserve bias dtype")

    if bias_prep_mode == 1 and triangle_bias.stride(-1) != 1:
        triangle_bias = triangle_bias.contiguous()
    elif bias_prep_mode == 2:
        triangle_bias = _prepare_bias(triangle_bias, kernel_config["BLOCK_N"])

    stride_bias_batch = triangle_bias.stride(0) if triangle_bias.shape[0] != 1 else 0
    block_dim = triton.next_power_of_2(head_dim)
    launch_grid = (triton.cdiv(seq_len, kernel_config["BLOCK_M"]), batch_size * num_heads)
    compiler_options = {
        key: value for key, value in kernel_config.items() if key not in _LAUNCH_CONFIG_KEYS
    }

    _triangle_attention_fwd_kernel[launch_grid](
        q,
        k,
        v,
        triangle_bias,
        mask_bias,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        stride_bias_batch,
        triangle_bias.stride(1),
        triangle_bias.stride(2),
        triangle_bias.stride(3),
        mask_bias.stride(0),
        mask_bias.stride(-1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_M=kernel_config["BLOCK_M"],
        BLOCK_N=kernel_config["BLOCK_N"],
        BLOCK_D=block_dim,
        HEAD_MAJOR=bool(variant_config["HEAD_MAJOR"]),
        BIAS_PADDED=bias_prep_mode >= 2,
        FAST_P=bool(variant_config["FAST_P"]),
        PEEL_FIRST=mask_bias.stride(-1) != 1,
        num_warps=kernel_config["num_warps"],
        num_stages=kernel_config["num_stages"],
        **compiler_options,
    )
    return out
