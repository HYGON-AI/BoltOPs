# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Sparse GQA prefill attention over selected regions (scalar and query-tiled).

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``sparse_attn`` prefill).
Kernels, tile-metadata builders, and launchers live in this module. The padded
fallback kernels are shared by BF16 and FP8. The 192-dim kernels stay separate:
BF16 casts gathered K/V to the query dtype, and FP8 widens them through fp32.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import (
    _cache_is_fp8,
    _choose_kv_splits,
    _clamp_auto_kv_splits,
    _config_match_score,
    _device_name_for_config,
    _fit_block_regions,
    _load_device_configs,
    _occupancy_dtype,
    _portable_launch_options,
    _require_contiguous_last_dim,
    _select_config,
)
from boltops.utility.triton_capability import get_triton_capabilities

# Retained for source-tree config verification callers.
_prefill_config_match_score = _config_match_score

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)


# Shared BF16/FP8 padded fallbacks. Gathered K/V cast with .to(queries.dtype).
@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Attention for one query against one KV group's selected regions.

    Online softmax over region tiles. The accumulator is per (head, dim) and the running
    max is per head, exactly as in dense flash attention -- the only difference is that
    the K/V rows come from a gather driven by the packed region list instead of a
    contiguous range.

    Split windows are a partition of the packed row. The last ``BLOCK_R`` tile is
    clamped with ``slots < window_end`` so a window that is not a multiple of
    ``BLOCK_R`` cannot walk into the next split.

    The cache may hold ``float8_e4m3`` rather than the query's dtype. Both dequantization
    scales stay out of the fp8 operands: ``k_scale`` is pre-folded into ``softmax_scale``
    by the wrapper, so it lands on the fp32 score tile, and ``v_scale`` multiplies the
    fp32 accumulator at the end. Scaling the operands instead would re-quantize them.
    """
    query = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    row_end = tl.minimum(window_end, tl.minimum(count, topk))
    for base in tl.range(window_begin, row_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = meta >> REGION_VALID_SHIFT
        # A slot is live only if it was within the count, not a -1 sentinel, and actually
        # exposes tokens. The third condition is not redundant: a region whose first token
        # is past the query position packs a valid id with zero visible tokens.
        live = slot_ok & (meta >= 0) & (valid > 0)

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )
        safe_tokens = tl.where(token_ok, tokens, 0)

        keys = tl.load(
            k_ptr
            + safe_tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        # Widen to the compute dtype before either dot. This is what keeps the later
        # ``probabilities.to(values.dtype)`` honest: against an fp8 cache that cast would
        # otherwise quantize the softmax probabilities to e4m3.
        keys = keys.to(queries.dtype)
        values = values.to(queries.dtype)

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        # A tile can be entirely masked -- every slot exhausted, or a row with no
        # selection at all. Then both maxima are -inf and ``m - m`` is NaN, which would
        # poison the accumulator for the rest of the loop. Substituting a finite shift
        # keeps the arithmetic well defined and leaves the running state untouched.
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0) * v_scale
    tl.store(
        out_ptr
        + split * stride_out_split
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + query * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_prefill_tiled_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    list_ptr,
    logical_list_ptr,
    lengths_ptr,
    mask_ptr,
    q_start_ptr,
    q_count_ptr,
    q_positions_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_list_tile,
    stride_list_slot,
    stride_logical_tile,
    stride_logical_slot,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    num_kv_groups,
    num_splits,
    slots_per_split,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Query-tiled sparse prefill over a per-tile union of selected regions.

    The selector produces one region list per query, but neighbouring queries usually
    select nearly the same regions. This kernel loads each union region once per
    (query-tile, KV group, head, split) and uses a bitset to decide which query owns
    each token. Tiles are never allowed to cross a request boundary by
    :func:`_build_prefill_tiles`, so a logical region id maps to one physical cache
    region within a tile.

    Grid z is ``head * num_splits + split`` so ``num_splits=1`` keeps the original
    ``(tile, group, head)`` mapping. Split windows partition the union list; the last
    ``BLOCK_R`` tile is clamped with ``slots < window_end``.

    K is loaded as ``[BLOCK_D, tokens]`` so QK is ``tl.dot(Q, K)`` without
    ``tl.trans``. V stays ``[tokens, BLOCK_D]`` for ``P @ V``. Online softmax is
    log2; the epilogue shares one reciprocal per query. ``v_scale`` and the
    ``k_scale`` already folded into ``softmax_scale`` dequantize an fp8 cache on
    the fp32 side; see :func:`_sparse_attn_prefill_kernel`.
    """
    tile = tl.program_id(0)
    group = tl.program_id(1)
    combo = tl.program_id(2)
    head = combo // num_splits
    split = combo % num_splits
    group_tile = tile * num_kv_groups + group

    q_local = tl.arange(0, BLOCK_Q)
    q_start = tl.load(q_start_ptr + tile).to(tl.int32)
    q_count = tl.load(q_count_ptr + tile).to(tl.int32)
    q_index = q_start + q_local
    q_ok = (q_local < q_count) & (q_index < total_q)

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim

    queries = tl.load(
        q_ptr
        + q_index[:, None] * stride_q_token
        + (group * heads_per_group + head) * stride_q_head
        + dims[None, :],
        mask=q_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )

    scale = softmax_scale * LOG2E
    running_max = tl.full((BLOCK_Q,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    positions = tl.load(
        q_positions_ptr + q_index,
        mask=q_ok,
        other=-1,
    ).to(tl.int32)
    words_of_q = q_local // 32
    bits_of_q = q_local % 32
    offsets_in_region = tl.arange(0, region_size)
    union_count = tl.load(lengths_ptr + group_tile).to(tl.int32)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < union_count) & (slots < window_end)
        physical = tl.load(
            list_ptr + group_tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr
            + group_tile * stride_logical_tile
            + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + group_tile * stride_mask_tile
            + words_of_q[:, None] * stride_mask_word
            + slots[None, :] * stride_mask_slot,
            mask=q_ok[:, None] & slot_ok[None, :],
            other=0,
        )
        selected = ((words >> bits_of_q[:, None]) & 1) == 1

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_pos = tl.reshape(
            logical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_live = tl.reshape(
            tl.broadcast_to(slot_ok[:, None], (BLOCK_R, region_size)),
            (BLOCK_R * region_size,),
        )
        causal = token_pos[None, :] <= positions[:, None]
        keep = causal & token_live[None, :] & q_ok[:, None]
        keep = keep & tl.reshape(
            tl.broadcast_to(selected[:, :, None], (BLOCK_Q, BLOCK_R, region_size)),
            (BLOCK_Q, BLOCK_R * region_size),
        )

        safe_tokens = tl.where(token_live, tokens, 0)
        k_mask = token_live[None, :] & dim_ok[:, None]
        keys = tl.load(
            k_ptr
            + safe_tokens[None, :] * stride_k_token
            + group * stride_k_group
            + dims[:, None],
            mask=k_mask,
            other=0.0,
        ).to(queries.dtype)
        scores = tl.dot(queries, keys, out_dtype=tl.float32) * scale
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)

        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_live[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(queries.dtype)
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected_any = running_sum > 0
    inv_sum = tl.div_rn(1.0, tl.where(selected_any, running_sum, 1.0))
    output = (
        tl.where(selected_any[:, None], accumulator * inv_sum[:, None], 0.0) * v_scale
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + q_index[:, None] * stride_out_token
        + (group * heads_per_group + head) * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None] & dim_ok[None, :],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + q_index * stride_lse_token
        + group * heads_per_group
        + head,
        tl.where(
            selected_any,
            running_max * LN2 + tl.log(running_sum),
            NEGATIVE_INFINITY,
        ),
        mask=q_ok,
    )


# BF16 192-dim kernels. Direct cast of gathered K/V to the query dtype.
@triton.jit
def _sparse_attn_prefill_dsplit_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Three 64-wide MMAC chunks with the HEAD BF16 online-softmax loop."""
    tl.static_assert(BLOCK_D1 + BLOCK_D2 + BLOCK_D3 == head_dim)
    query = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims0 = tl.arange(0, BLOCK_D1)
    dims1 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    dims2 = BLOCK_D1 + BLOCK_D2 + tl.arange(0, BLOCK_D3)
    head_ok = heads < heads_per_group
    q_base = (
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
    )
    queries0 = tl.load(q_base + dims0[None, :], mask=head_ok[:, None], other=0.0)
    queries1 = tl.load(q_base + dims1[None, :], mask=head_ok[:, None], other=0.0)
    queries2 = tl.load(q_base + dims2[None, :], mask=head_ok[:, None], other=0.0)

    scale = softmax_scale * LOG2E
    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, BLOCK_D1), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_H, BLOCK_D2), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_H, BLOCK_D3), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    # Skip tiles after this row's last selected region; preserve the split boundary.
    row_end = tl.minimum(window_end, tl.minimum(count, topk))
    for base in tl.range(window_begin, row_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = meta >> REGION_VALID_SHIFT
        live = slot_ok & (meta >= 0) & (valid > 0)

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )
        safe_tokens = tl.where(token_ok, tokens, 0)
        token_head_ok = token_ok[None, :] & head_ok[:, None]
        k_base = (
            k_ptr
            + safe_tokens[None, :] * stride_k_token
            + group * stride_k_group
        )
        k_mask = token_ok[None, :]
        keys0 = tl.load(k_base + dims0[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        keys1 = tl.load(k_base + dims1[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        keys2 = tl.load(k_base + dims2[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        scores = (
            tl.dot(queries0, keys0, out_dtype=tl.float32)
            + tl.dot(queries1, keys1, out_dtype=tl.float32)
            + tl.dot(queries2, keys2, out_dtype=tl.float32)
        ) * scale
        scores = tl.where(token_head_ok, scores, NEGATIVE_INFINITY)
        v_base = (
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
        )
        v_mask = token_ok[:, None]
        values0 = tl.load(v_base + dims0[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        values1 = tl.load(v_base + dims1[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        values2 = tl.load(v_base + dims2[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        p = probabilities.to(queries0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(p, values0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(p, values1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(p, values2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    inv_sum = tl.div_rn(1.0, tl.where(selected, running_sum, 1.0))
    output0 = tl.where(selected[:, None], acc0 * inv_sum[:, None], 0.0) * v_scale
    output1 = tl.where(selected[:, None], acc1 * inv_sum[:, None], 0.0) * v_scale
    output2 = tl.where(selected[:, None], acc2 * inv_sum[:, None], 0.0) * v_scale
    out_base = (
        out_ptr
        + split * stride_out_split
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
    )
    tl.store(
        out_base + dims0[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_base + dims1[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_base + dims2[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + query * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(
            selected,
            running_max * LN2 + tl.log(running_sum),
            NEGATIVE_INFINITY,
        ),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_prefill_tiled_dsplit_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    list_ptr,
    logical_list_ptr,
    lengths_ptr,
    mask_ptr,
    q_start_ptr,
    q_count_ptr,
    q_positions_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_list_tile,
    stride_list_slot,
    stride_logical_tile,
    stride_logical_slot,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    num_kv_groups,
    num_splits,
    slots_per_split,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Tiled prefill specialized for an exact three-chunk head dimension.

    ``head_dim=192`` launches as ``64+64+64`` MMACs instead of a 256-padded tile.
    Inner-loop schedule: cluster the three K gathers, QK, then the three V
    gathers, then softmax / PV. K is loaded already transposed for ``tl.dot``.
    Online softmax is log2; the epilogue shares one reciprocal per query. Split
    windows partition the union list the same way as
    :func:`_sparse_attn_prefill_tiled_kernel`.
    """
    tl.static_assert(BLOCK_D1 + BLOCK_D2 + BLOCK_D3 == head_dim)
    tile = tl.program_id(0)
    group = tl.program_id(1)
    combo = tl.program_id(2)
    head = combo // num_splits
    split = combo % num_splits
    group_tile = tile * num_kv_groups + group

    q_local = tl.arange(0, BLOCK_Q)
    q_start = tl.load(q_start_ptr + tile).to(tl.int32)
    q_count = tl.load(q_count_ptr + tile).to(tl.int32)
    q_index = q_start + q_local
    q_ok = (q_local < q_count) & (q_index < total_q)

    dims0 = tl.arange(0, BLOCK_D1)
    dims1 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    dims2 = BLOCK_D1 + BLOCK_D2 + tl.arange(0, BLOCK_D3)
    q_base = (
        q_ptr
        + q_index[:, None] * stride_q_token
        + (group * heads_per_group + head) * stride_q_head
    )
    queries0 = tl.load(q_base + dims0[None, :], mask=q_ok[:, None], other=0.0)
    queries1 = tl.load(q_base + dims1[None, :], mask=q_ok[:, None], other=0.0)
    queries2 = tl.load(q_base + dims2[None, :], mask=q_ok[:, None], other=0.0)

    scale = softmax_scale * LOG2E
    running_max = tl.full((BLOCK_Q,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_Q, BLOCK_D1), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_Q, BLOCK_D2), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_Q, BLOCK_D3), dtype=tl.float32)

    positions = tl.load(
        q_positions_ptr + q_index,
        mask=q_ok,
        other=-1,
    ).to(tl.int32)
    words_of_q = q_local // 32
    bits_of_q = q_local % 32
    offsets_in_region = tl.arange(0, region_size)
    union_count = tl.load(lengths_ptr + group_tile).to(tl.int32)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < union_count) & (slots < window_end)
        physical = tl.load(
            list_ptr + group_tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr
            + group_tile * stride_logical_tile
            + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + group_tile * stride_mask_tile
            + words_of_q[:, None] * stride_mask_word
            + slots[None, :] * stride_mask_slot,
            mask=q_ok[:, None] & slot_ok[None, :],
            other=0,
        )
        selected = ((words >> bits_of_q[:, None]) & 1) == 1

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_pos = tl.reshape(
            logical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_live = tl.reshape(
            tl.broadcast_to(slot_ok[:, None], (BLOCK_R, region_size)),
            (BLOCK_R * region_size,),
        )
        causal = token_pos[None, :] <= positions[:, None]
        keep = causal & token_live[None, :] & q_ok[:, None]
        keep = keep & tl.reshape(
            tl.broadcast_to(selected[:, :, None], (BLOCK_Q, BLOCK_R, region_size)),
            (BLOCK_Q, BLOCK_R * region_size),
        )

        safe_tokens = tl.where(token_live, tokens, 0)
        k_base = (
            k_ptr + safe_tokens[None, :] * stride_k_token + group * stride_k_group
        )
        k_mask = token_live[None, :]
        keys0 = tl.load(k_base + dims0[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        keys1 = tl.load(k_base + dims1[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        keys2 = tl.load(k_base + dims2[:, None], mask=k_mask, other=0.0).to(
            queries0.dtype
        )
        scores = (
            tl.dot(queries0, keys0, out_dtype=tl.float32)
            + tl.dot(queries1, keys1, out_dtype=tl.float32)
            + tl.dot(queries2, keys2, out_dtype=tl.float32)
        ) * scale
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)

        v_base = (
            v_ptr + safe_tokens[:, None] * stride_v_token + group * stride_v_group
        )
        v_mask = token_live[:, None]
        values0 = tl.load(v_base + dims0[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        values1 = tl.load(v_base + dims1[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        values2 = tl.load(v_base + dims2[None, :], mask=v_mask, other=0.0).to(
            queries0.dtype
        )
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        p = probabilities.to(queries0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(p, values0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(p, values1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(p, values2, out_dtype=tl.float32)
        running_max = tile_max

    selected_any = running_sum > 0
    inv_sum = tl.div_rn(1.0, tl.where(selected_any, running_sum, 1.0))
    output0 = tl.where(selected_any[:, None], acc0 * inv_sum[:, None], 0.0) * v_scale
    output1 = tl.where(selected_any[:, None], acc1 * inv_sum[:, None], 0.0) * v_scale
    output2 = tl.where(selected_any[:, None], acc2 * inv_sum[:, None], 0.0) * v_scale
    out_base = (
        out_ptr
        + split * stride_out_split
        + q_index[:, None] * stride_out_token
        + (group * heads_per_group + head) * stride_out_head
    )
    tl.store(
        out_base + dims0[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_base + dims1[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_base + dims2[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + q_index * stride_lse_token
        + group * heads_per_group
        + head,
        tl.where(
            selected_any,
            running_max * LN2 + tl.log(running_sum),
            NEGATIVE_INFINITY,
        ),
        mask=q_ok,
    )


# FP8 192-dim kernels. Gathered K/V are widened through fp32.
@triton.jit
def _widen_fp8_kv(raw, ref):
    """Widen a gathered fp8 K/V tile to the query dtype through fp32.

    A direct cast of fp8 lowers to a software byte unpack. Going through fp32
    emits ``v_cvt_pk_f32_fp8`` and then packs the query dtype.
    """
    return raw.to(tl.float32).to(ref.dtype)


@triton.jit
def _sparse_attn_prefill_dsplit_fp8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """FP8-cache prefill specialized for an exact three-chunk head dimension.

    ``head_dim=192`` launches as ``64+64+64`` MMACs (not a 256-padded tile).
    K gathers and QK dot precede the V gathers and softmax/PV dot. The first
    tile initializes the online softmax accumulators; subsequent tiles use a
    two-way unrolled loop. Padded heads have masked query loads and stores.
    Split windows partition the packed row; ``slots < window_end`` keeps the last
    ``BLOCK_R`` tile out of the next split.

    ``v_scale`` and the ``k_scale`` already folded into ``softmax_scale``
    dequantize the FP8 cache on the fp32 side. K/V widen through fp32 for
    packed hardware conversion.
    """
    tl.static_assert(BLOCK_D1 + BLOCK_D2 + BLOCK_D3 == head_dim)
    query = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims0 = tl.arange(0, BLOCK_D1)
    dims1 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    dims2 = BLOCK_D1 + BLOCK_D2 + tl.arange(0, BLOCK_D3)
    head_ok = heads < heads_per_group
    q_base = (
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
    )
    queries0 = tl.load(q_base + dims0[None, :], mask=head_ok[:, None], other=0.0)
    queries1 = tl.load(q_base + dims1[None, :], mask=head_ok[:, None], other=0.0)
    queries2 = tl.load(q_base + dims2[None, :], mask=head_ok[:, None], other=0.0)

    scale = softmax_scale * LOG2E
    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split

    # The first tile initializes the online softmax without rescaling an all-zero
    # accumulator. Even an empty split executes this masked tile and produces
    # zero output and -inf LSE.
    base = window_begin
    slots = base + tl.arange(0, BLOCK_R)
    slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
    meta = tl.load(
        packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
    )
    physical = meta & REGION_ID_MASK
    valid = meta >> REGION_VALID_SHIFT
    live = slot_ok & (meta >= 0) & (valid > 0)
    tokens = tl.reshape(
        physical[:, None] * region_size + offsets_in_region[None, :],
        (BLOCK_R * region_size,),
    )
    token_ok = tl.reshape(
        live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
        (BLOCK_R * region_size,),
    )
    safe_tokens = tl.where(token_ok, tokens, 0)
    token_head_ok = token_ok[None, :]
    k_base = k_ptr + safe_tokens[None, :] * stride_k_token + group * stride_k_group
    keys0 = _widen_fp8_kv(
        tl.load(k_base + dims0[:, None], mask=token_ok[None, :], other=0.0),
        queries0,
    )
    keys1 = _widen_fp8_kv(
        tl.load(k_base + dims1[:, None], mask=token_ok[None, :], other=0.0),
        queries1,
    )
    keys2 = _widen_fp8_kv(
        tl.load(k_base + dims2[:, None], mask=token_ok[None, :], other=0.0),
        queries2,
    )
    scores = (
        tl.dot(queries0, keys0, out_dtype=tl.float32)
        + tl.dot(queries1, keys1, out_dtype=tl.float32)
        + tl.dot(queries2, keys2, out_dtype=tl.float32)
    ) * scale
    scores = tl.where(token_head_ok, scores, NEGATIVE_INFINITY)
    v_base = v_ptr + safe_tokens[:, None] * stride_v_token + group * stride_v_group
    values0 = _widen_fp8_kv(
        tl.load(v_base + dims0[None, :], mask=token_ok[:, None], other=0.0),
        queries0,
    )
    values1 = _widen_fp8_kv(
        tl.load(v_base + dims1[None, :], mask=token_ok[:, None], other=0.0),
        queries1,
    )
    values2 = _widen_fp8_kv(
        tl.load(v_base + dims2[None, :], mask=token_ok[:, None], other=0.0),
        queries2,
    )
    running_max = tl.max(scores, axis=1)
    empty = running_max == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, running_max)
    probabilities = tl.math.exp2(scores - shift[:, None])
    running_sum = tl.sum(probabilities, axis=1)
    p = probabilities.to(queries0.dtype)
    acc0 = tl.dot(p, values0, out_dtype=tl.float32)
    acc1 = tl.dot(p, values1, out_dtype=tl.float32)
    acc2 = tl.dot(p, values2, out_dtype=tl.float32)

    # Short windows can have fewer iterations than the unroll factor after
    # the first tile; the HCU loop-unroll pass rejects those specializations.
    # Odd iterations leave an epilogue that spills registers. Keep two-way
    # unrolling only when the remaining loop iterations are even and >= 2.
    # Round the dynamic stop to full pairs: the HCU unroller requires at least
    # two remaining iterations, and masked slots in the rounded-up pair are inert.
    row_end = tl.minimum(window_end, tl.minimum(count, topk))
    if ((slots_per_split - 1) // BLOCK_R) >= 2 and ((slots_per_split - 1) // BLOCK_R) % 2 == 0:
        remaining = tl.maximum(0, row_end - (window_begin + BLOCK_R))
        pairs = tl.maximum(1, (remaining + 2 * BLOCK_R - 1) // (2 * BLOCK_R))
        row_end = tl.minimum(window_end, window_begin + BLOCK_R + pairs * 2 * BLOCK_R)
    for base in tl.range(
        window_begin + BLOCK_R,
        row_end,
        BLOCK_R,
        loop_unroll_factor=(
            2
            if ((slots_per_split - 1) // BLOCK_R) >= 2
            and ((slots_per_split - 1) // BLOCK_R) % 2 == 0
            else 1
        ),
    ):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = meta >> REGION_VALID_SHIFT
        live = slot_ok & (meta >= 0) & (valid > 0)

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )
        safe_tokens = tl.where(token_ok, tokens, 0)
        # Padded heads are masked on query loads and output stores.
        token_head_ok = token_ok[None, :]
        k_base = (
            k_ptr
            + safe_tokens[None, :] * stride_k_token
            + group * stride_k_group
        )
        keys0 = _widen_fp8_kv(
            tl.load(k_base + dims0[:, None], mask=token_ok[None, :], other=0.0),
            queries0,
        )
        keys1 = _widen_fp8_kv(
            tl.load(k_base + dims1[:, None], mask=token_ok[None, :], other=0.0),
            queries1,
        )
        keys2 = _widen_fp8_kv(
            tl.load(k_base + dims2[:, None], mask=token_ok[None, :], other=0.0),
            queries2,
        )
        scores = (
            tl.dot(queries0, keys0, out_dtype=tl.float32)
            + tl.dot(queries1, keys1, out_dtype=tl.float32)
            + tl.dot(queries2, keys2, out_dtype=tl.float32)
        ) * scale
        scores = tl.where(token_head_ok, scores, NEGATIVE_INFINITY)
        v_base = (
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
        )
        values0 = _widen_fp8_kv(
            tl.load(v_base + dims0[None, :], mask=token_ok[:, None], other=0.0),
            queries0,
        )
        values1 = _widen_fp8_kv(
            tl.load(v_base + dims1[None, :], mask=token_ok[:, None], other=0.0),
            queries1,
        )
        values2 = _widen_fp8_kv(
            tl.load(v_base + dims2[None, :], mask=token_ok[:, None], other=0.0),
            queries2,
        )
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        p = probabilities.to(queries0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(p, values0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(p, values1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(p, values2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    inv_sum = tl.div_rn(1.0, tl.where(selected, running_sum, 1.0))
    norm = inv_sum * v_scale
    output0 = tl.where(selected[:, None], acc0 * norm[:, None], 0.0)
    output1 = tl.where(selected[:, None], acc1 * norm[:, None], 0.0)
    output2 = tl.where(selected[:, None], acc2 * norm[:, None], 0.0)
    out_base = (
        out_ptr
        + split * stride_out_split
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
    )
    tl.store(
        out_base + dims0[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_base + dims1[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_base + dims2[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + query * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(
            selected,
            (running_max + tl.math.log2(running_sum)) * LN2,
            NEGATIVE_INFINITY,
        ),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_prefill_tiled_dsplit_fp8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    list_ptr,
    logical_list_ptr,
    lengths_ptr,
    mask_ptr,
    q_start_ptr,
    q_count_ptr,
    q_positions_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_list_tile,
    stride_list_slot,
    stride_logical_tile,
    stride_logical_slot,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    total_q,
    num_kv_groups,
    num_splits,
    slots_per_split,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Tiled prefill specialized for an exact three-chunk head dimension.

    ``head_dim=192`` launches as ``64+64+64`` MMACs instead of a 256-padded tile.
    Inner-loop schedule: cluster the three K gathers, QK, then the three V
    gathers, then softmax / PV. K is loaded already transposed for ``tl.dot``.
    Online softmax is log2; the epilogue shares one reciprocal per query. Split
    windows partition the union list the same way as
    :func:`_sparse_attn_prefill_tiled_kernel`.
    """
    tl.static_assert(BLOCK_D1 + BLOCK_D2 + BLOCK_D3 == head_dim)
    tile = tl.program_id(0)
    group = tl.program_id(1)
    combo = tl.program_id(2)
    head = combo // num_splits
    split = combo % num_splits
    group_tile = tile * num_kv_groups + group

    q_local = tl.arange(0, BLOCK_Q)
    q_start = tl.load(q_start_ptr + tile).to(tl.int32)
    q_count = tl.load(q_count_ptr + tile).to(tl.int32)
    q_index = q_start + q_local
    q_ok = (q_local < q_count) & (q_index < total_q)

    dims0 = tl.arange(0, BLOCK_D1)
    dims1 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    dims2 = BLOCK_D1 + BLOCK_D2 + tl.arange(0, BLOCK_D3)
    q_base = (
        q_ptr
        + q_index[:, None] * stride_q_token
        + (group * heads_per_group + head) * stride_q_head
    )
    queries0 = tl.load(q_base + dims0[None, :], mask=q_ok[:, None], other=0.0)
    queries1 = tl.load(q_base + dims1[None, :], mask=q_ok[:, None], other=0.0)
    queries2 = tl.load(q_base + dims2[None, :], mask=q_ok[:, None], other=0.0)

    scale = softmax_scale * LOG2E
    running_max = tl.full((BLOCK_Q,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_Q, BLOCK_D1), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_Q, BLOCK_D2), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_Q, BLOCK_D3), dtype=tl.float32)

    positions = tl.load(
        q_positions_ptr + q_index,
        mask=q_ok,
        other=-1,
    ).to(tl.int32)
    words_of_q = q_local // 32
    bits_of_q = q_local % 32
    offsets_in_region = tl.arange(0, region_size)
    union_count = tl.load(lengths_ptr + group_tile).to(tl.int32)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < union_count) & (slots < window_end)
        physical = tl.load(
            list_ptr + group_tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr
            + group_tile * stride_logical_tile
            + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + group_tile * stride_mask_tile
            + words_of_q[:, None] * stride_mask_word
            + slots[None, :] * stride_mask_slot,
            mask=q_ok[:, None] & slot_ok[None, :],
            other=0,
        )
        selected = ((words >> bits_of_q[:, None]) & 1) == 1

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_pos = tl.reshape(
            logical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_live = tl.reshape(
            tl.broadcast_to(slot_ok[:, None], (BLOCK_R, region_size)),
            (BLOCK_R * region_size,),
        )
        causal = token_pos[None, :] <= positions[:, None]
        keep = causal & token_live[None, :] & q_ok[:, None]
        keep = keep & tl.reshape(
            tl.broadcast_to(selected[:, :, None], (BLOCK_Q, BLOCK_R, region_size)),
            (BLOCK_Q, BLOCK_R * region_size),
        )

        safe_tokens = tl.where(token_live, tokens, 0)
        k_base = (
            k_ptr + safe_tokens[None, :] * stride_k_token + group * stride_k_group
        )
        k_mask = token_live[None, :]
        keys0 = _widen_fp8_kv(
            tl.load(k_base + dims0[:, None], mask=k_mask, other=0.0), queries0
        )
        keys1 = _widen_fp8_kv(
            tl.load(k_base + dims1[:, None], mask=k_mask, other=0.0), queries0
        )
        keys2 = _widen_fp8_kv(
            tl.load(k_base + dims2[:, None], mask=k_mask, other=0.0), queries0
        )
        scores = (
            tl.dot(queries0, keys0, out_dtype=tl.float32)
            + tl.dot(queries1, keys1, out_dtype=tl.float32)
            + tl.dot(queries2, keys2, out_dtype=tl.float32)
        ) * scale
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)

        v_base = (
            v_ptr + safe_tokens[:, None] * stride_v_token + group * stride_v_group
        )
        v_mask = token_live[:, None]
        values0 = _widen_fp8_kv(
            tl.load(v_base + dims0[None, :], mask=v_mask, other=0.0), queries0
        )
        values1 = _widen_fp8_kv(
            tl.load(v_base + dims1[None, :], mask=v_mask, other=0.0), queries0
        )
        values2 = _widen_fp8_kv(
            tl.load(v_base + dims2[None, :], mask=v_mask, other=0.0), queries0
        )
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        p = probabilities.to(queries0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(p, values0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(p, values1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(p, values2, out_dtype=tl.float32)
        running_max = tile_max

    selected_any = running_sum > 0
    inv_sum = tl.div_rn(1.0, tl.where(selected_any, running_sum, 1.0))
    output0 = tl.where(selected_any[:, None], acc0 * inv_sum[:, None], 0.0) * v_scale
    output1 = tl.where(selected_any[:, None], acc1 * inv_sum[:, None], 0.0) * v_scale
    output2 = tl.where(selected_any[:, None], acc2 * inv_sum[:, None], 0.0) * v_scale
    out_base = (
        out_ptr
        + split * stride_out_split
        + q_index[:, None] * stride_out_token
        + (group * heads_per_group + head) * stride_out_head
    )
    tl.store(
        out_base + dims0[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_base + dims1[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        out_base + dims2[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None],
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + q_index * stride_lse_token
        + group * heads_per_group
        + head,
        tl.where(
            selected_any,
            running_max * LN2 + tl.log(running_sum),
            NEGATIVE_INFINITY,
        ),
        mask=q_ok,
    )


# Query-tile metadata. Four kernels: Triton has no grid-wide barrier.
@triton.jit
def _prefill_tile_scan_kernel(
    request_ptr,
    req_start_ptr,
    q_start_ptr,
    q_count_ptr,
    tile_of_q_ptr,
    num_tiles_ptr,
    overflow_ptr,
    total_q,
    BLOCK_Q: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    tl.store(overflow_ptr, 0)
    n_new = tl.zeros((), dtype=tl.int32)
    for base in tl.range(0, MAX_Q, BLOCK):
        idx = base + tl.arange(0, BLOCK)
        q_ok = idx < total_q
        req = tl.load(request_ptr + idx, mask=q_ok, other=0)
        req_prev = tl.load(
            request_ptr + (idx - 1), mask=q_ok & (idx > 0), other=0
        )
        is_new = q_ok & ((idx == 0) | (req != req_prev))
        flag = is_new.to(tl.int32)
        req_id = n_new + tl.cumsum(flag, axis=0) - 1
        tl.store(req_start_ptr + req_id, idx, mask=is_new)
        n_new = n_new + tl.sum(flag)

    tile_cursor = tl.zeros((), dtype=tl.int32)
    qoff = tl.arange(0, BLOCK_Q)
    for r in tl.range(0, n_new):
        rst = tl.load(req_start_ptr + r)
        rend = tl.load(req_start_ptr + r + 1, mask=(r + 1) < n_new, other=total_q)
        n_t = (rend - rst + BLOCK_Q - 1) // BLOCK_Q
        for t in tl.range(0, n_t):
            qs = rst + t * BLOCK_Q
            qc = tl.minimum(BLOCK_Q, rend - qs)
            tl.store(q_start_ptr + tile_cursor, qs)
            tl.store(q_count_ptr + tile_cursor, qc)
            tl.store(tile_of_q_ptr + qs + qoff, tile_cursor, mask=qoff < qc)
            tile_cursor = tile_cursor + 1
    tl.store(num_tiles_ptr, tile_cursor)


@triton.jit
def _prefill_tile_mark_kernel(
    logical_ptr,
    packed_ptr,
    tile_of_q_ptr,
    present_ptr,
    phys_tbl_ptr,
    overflow_ptr,
    total_q,
    num_kv_groups,
    slots,
    n_pairs,
    stride_lr,
    stride_ls,
    stride_pr,
    stride_ps,
    stride_present,
    BLOCK: tl.constexpr,
    MAX_REGIONS: tl.constexpr,
):
    wid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = wid < n_pairs
    row = wid // slots
    s = wid % slots
    q = row % total_q
    gt = tl.load(tile_of_q_ptr + q, mask=take, other=0) * num_kv_groups + (row // total_q)
    logical = tl.load(logical_ptr + row * stride_lr + s * stride_ls, mask=take, other=-1)
    packed = tl.load(packed_ptr + row * stride_pr + s * stride_ps, mask=take, other=-1)
    live = take & (logical >= 0) & (packed >= 0)
    tl.atomic_max(overflow_ptr, tl.max((live & (logical >= MAX_REGIONS)).to(tl.int32)))
    in_tbl = live & (logical < MAX_REGIONS)
    log_id = tl.where(in_tbl, logical, 0)
    tl.atomic_max(present_ptr + gt * stride_present + log_id, 1, mask=in_tbl)
    tl.store(
        phys_tbl_ptr + gt * stride_present + log_id, packed & REGION_ID_MASK, mask=in_tbl
    )


@triton.jit
def _prefill_tile_compact_kernel(
    present_ptr,
    phys_tbl_ptr,
    slot_of_ptr,
    tile_phys_ptr,
    tile_logical_ptr,
    lengths_ptr,
    num_tiles_ptr,
    overflow_ptr,
    num_kv_groups,
    stride_present,
    stride_out,
    BLOCK: tl.constexpr,
    MAX_REGIONS: tl.constexpr,
    UNION_CAP: tl.constexpr,
):
    gt = tl.program_id(0)
    n_gt = tl.load(num_tiles_ptr) * num_kv_groups
    if gt >= n_gt:
        return
    n_union = tl.zeros((), dtype=tl.int32)
    ov = tl.zeros((), dtype=tl.int32)
    for base in tl.range(0, MAX_REGIONS, BLOCK):
        lid = base + tl.arange(0, BLOCK)
        keep = lid < MAX_REGIONS
        flag = tl.load(present_ptr + gt * stride_present + lid, mask=keep, other=0)
        phys = tl.load(phys_tbl_ptr + gt * stride_present + lid, mask=keep, other=0)
        slot = n_union + tl.cumsum(flag, axis=0) - flag
        can = keep & (flag != 0) & (slot < UNION_CAP)
        tl.store(tile_phys_ptr + gt * stride_out + slot, phys, mask=can)
        tl.store(tile_logical_ptr + gt * stride_out + slot, lid, mask=can)
        tl.store(
            slot_of_ptr + gt * stride_present + lid,
            tl.where(flag != 0, slot, -1),
            mask=keep,
        )
        grown = n_union + tl.sum(flag)
        ov = tl.maximum(ov, (grown > UNION_CAP).to(tl.int32))
        n_union = grown
    for base in tl.range(0, UNION_CAP, BLOCK):
        pad = base + tl.arange(0, BLOCK)
        dead = (pad >= n_union) & (pad < UNION_CAP)
        tl.store(tile_phys_ptr + gt * stride_out + pad, 0, mask=dead)
        tl.store(tile_logical_ptr + gt * stride_out + pad, 0, mask=dead)
    tl.store(lengths_ptr + gt, n_union)
    tl.atomic_max(overflow_ptr, ov)


@triton.jit
def _prefill_tile_mask_kernel(
    logical_ptr,
    packed_ptr,
    tile_of_q_ptr,
    q_start_ptr,
    slot_of_ptr,
    mask_ptr,
    total_q,
    num_kv_groups,
    slots,
    n_pairs,
    stride_lr,
    stride_ls,
    stride_pr,
    stride_ps,
    stride_present,
    stride_mask_tile,
    stride_mask_word,
    stride_mask_slot,
    BLOCK: tl.constexpr,
    MAX_REGIONS: tl.constexpr,
    UNION_CAP: tl.constexpr,
):
    wid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = wid < n_pairs
    row = wid // slots
    s = wid % slots
    q = row % total_q
    tile = tl.load(tile_of_q_ptr + q, mask=take, other=0)
    gt = tile * num_kv_groups + (row // total_q)
    logical = tl.load(logical_ptr + row * stride_lr + s * stride_ls, mask=take, other=-1)
    packed = tl.load(packed_ptr + row * stride_pr + s * stride_ps, mask=take, other=-1)
    live = take & (logical >= 0) & (packed >= 0) & (logical < MAX_REGIONS)
    u_slot = tl.load(
        slot_of_ptr + gt * stride_present + tl.where(live, logical, 0),
        mask=live,
        other=-1,
    )
    ok = live & (u_slot >= 0) & (u_slot < UNION_CAP)
    q_local = q - tl.load(q_start_ptr + tile, mask=take, other=0)
    tl.atomic_or(
        mask_ptr
        + gt * stride_mask_tile
        + (q_local // 32) * stride_mask_word
        + u_slot * stride_mask_slot,
        tl.full((BLOCK,), 1, dtype=tl.int32) << (q_local % 32),
        mask=ok,
    )


def _require_matching_kv_dtypes(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> None:
    """Reject K/V caches that are not the same dtype."""
    if key_cache.dtype != value_cache.dtype:
        raise ValueError(
            "key/value cache dtype mismatch: "
            f"{key_cache.dtype} vs {value_cache.dtype}"
        )


def _prefill_head_dim_split(head_dim: int) -> tuple[int, int, int] | None:
    """``(D1, D2, D3)`` pow2 MMAC chunks covering ``head_dim``, or ``None``.

    ``head_dim=192`` is launched as ``64+64+64`` rather than ``128+64``: three
    64-wide MMACs use fewer VGPRs than one 128-wide tile plus one 64-wide tile.
    """
    if head_dim <= 0 or head_dim & (head_dim - 1) == 0:
        return None
    d1 = 1 << (head_dim.bit_length() - 1)
    d2 = head_dim - d1
    if d1 == 128 and d2 == 64:
        return 64, 64, 64
    return None


def _as_query_space(
    values: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    name: str,
) -> torch.Tensor:
    """Accept length ``Q`` or group-major ``G*Q`` (identical per group) as query ids."""
    count = int(values.numel())
    expected_rows = num_kv_groups * total_q
    if count == total_q:
        return values.reshape(total_q)
    if count != expected_rows:
        raise ValueError(f"expected {total_q} or {expected_rows} {name}, got {count}")
    viewed = values.reshape(num_kv_groups, total_q)
    if num_kv_groups > 1 and not torch.equal(viewed, viewed[0].expand_as(viewed)):
        raise ValueError(
            f"{name} must repeat the same query-space values across KV groups"
        )
    return viewed[0].contiguous()


# Device tile builder: Q<=4096, logical<4096, union<=1024, block_q<=256.
# CPU / oversized / overflow keep the PyTorch path below.
_TILE_BUILD_MAX_Q = 4096
_TILE_BUILD_MAX_REGIONS = 4096
_TILE_BUILD_UNION_CAP = 1024
_TILE_BUILD_BLOCK = 512
_TILE_BUILD_MASK_WORDS = 8
_TILE_BUILD_WARPS = 16


class _PrefillTileWorkspace:
    def __init__(self, device: torch.device, max_q: int, max_groups: int):
        max_gt = max_q * max_groups
        self.max_q = max_q
        self.max_groups = max_groups
        self.req_start = torch.empty(max_q, device=device, dtype=torch.int32)
        self.q_starts = torch.empty(max_q, device=device, dtype=torch.int32)
        self.q_counts = torch.empty(max_q, device=device, dtype=torch.int32)
        self.tile_of_q = torch.empty(max_q, device=device, dtype=torch.int32)
        self.present = torch.empty(
            (max_gt, _TILE_BUILD_MAX_REGIONS), device=device, dtype=torch.int32
        )
        self.phys_tbl = torch.empty_like(self.present)
        self.slot_of = torch.empty_like(self.present)
        self.tile_phys = torch.empty(
            (max_gt, _TILE_BUILD_UNION_CAP), device=device, dtype=torch.int32
        )
        self.tile_logical = torch.empty_like(self.tile_phys)
        self.lengths = torch.empty(max_gt, device=device, dtype=torch.int32)
        self.mask = torch.empty(
            (max_gt, _TILE_BUILD_MASK_WORDS, _TILE_BUILD_UNION_CAP),
            device=device,
            dtype=torch.int32,
        )
        self.num_tiles = torch.empty((), device=device, dtype=torch.int32)
        self.overflow = torch.empty((), device=device, dtype=torch.int32)


_PREFILL_TILE_WS: dict[tuple[str, int], _PrefillTileWorkspace] = {}


def _prefill_tile_workspace(
    device: torch.device, total_q: int, num_kv_groups: int
) -> _PrefillTileWorkspace:
    key = (str(device), int(num_kv_groups))
    ws = _PREFILL_TILE_WS.get(key)
    if ws is None or ws.max_q < total_q:
        cap = min(_TILE_BUILD_MAX_Q, triton.next_power_of_2(max(int(total_q), 1)))
        ws = _PrefillTileWorkspace(device, cap, num_kv_groups)
        _PREFILL_TILE_WS[key] = ws
    return ws


def _build_prefill_tiles_device(
    logical_regions: torch.Tensor,
    packed_regions: torch.Tensor,
    request_query: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    block_q: int,
) -> tuple[torch.Tensor, ...] | None:
    rows, slots = logical_regions.shape
    words = (block_q + 31) // 32
    block = _TILE_BUILD_BLOCK
    n_pairs = rows * slots
    n_gt_cap = total_q * num_kv_groups
    mark_grid = (triton.cdiv(n_pairs, block),)
    launch = {"num_warps": _TILE_BUILD_WARPS, "num_stages": 1}
    request = request_query.to(dtype=torch.int32).contiguous()
    ws = _prefill_tile_workspace(logical_regions.device, total_q, num_kv_groups)
    ws.present[:n_gt_cap].zero_()
    ws.mask[:n_gt_cap].zero_()
    _prefill_tile_scan_kernel[(1,)](
        request,
        ws.req_start,
        ws.q_starts,
        ws.q_counts,
        ws.tile_of_q,
        ws.num_tiles,
        ws.overflow,
        total_q,
        BLOCK_Q=block_q,
        BLOCK=block,
        MAX_Q=_TILE_BUILD_MAX_Q,
        **launch,
    )
    _prefill_tile_mark_kernel[mark_grid](
        logical_regions,
        packed_regions,
        ws.tile_of_q,
        ws.present,
        ws.phys_tbl,
        ws.overflow,
        total_q,
        num_kv_groups,
        slots,
        n_pairs,
        logical_regions.stride(0),
        logical_regions.stride(1),
        packed_regions.stride(0),
        packed_regions.stride(1),
        ws.present.stride(0),
        BLOCK=block,
        MAX_REGIONS=_TILE_BUILD_MAX_REGIONS,
        **launch,
    )
    _prefill_tile_compact_kernel[(n_gt_cap,)](
        ws.present,
        ws.phys_tbl,
        ws.slot_of,
        ws.tile_phys,
        ws.tile_logical,
        ws.lengths,
        ws.num_tiles,
        ws.overflow,
        num_kv_groups,
        ws.present.stride(0),
        ws.tile_phys.stride(0),
        BLOCK=block,
        MAX_REGIONS=_TILE_BUILD_MAX_REGIONS,
        UNION_CAP=_TILE_BUILD_UNION_CAP,
        **launch,
    )
    _prefill_tile_mask_kernel[mark_grid](
        logical_regions,
        packed_regions,
        ws.tile_of_q,
        ws.q_starts,
        ws.slot_of,
        ws.mask,
        total_q,
        num_kv_groups,
        slots,
        n_pairs,
        logical_regions.stride(0),
        logical_regions.stride(1),
        packed_regions.stride(0),
        packed_regions.stride(1),
        ws.slot_of.stride(0),
        ws.mask.stride(0),
        ws.mask.stride(1),
        ws.mask.stride(2),
        BLOCK=block,
        MAX_REGIONS=_TILE_BUILD_MAX_REGIONS,
        UNION_CAP=_TILE_BUILD_UNION_CAP,
        **launch,
    )
    n_tiles = int(ws.num_tiles.item())
    if n_tiles <= 0 or int(ws.overflow.item()) != 0:
        return None
    n_gt = n_tiles * num_kv_groups
    union = max(1, int(ws.lengths[:n_gt].max().item())) if n_gt else 1
    return (
        ws.q_starts[:n_tiles].contiguous(),
        ws.q_counts[:n_tiles].contiguous(),
        ws.tile_phys[:n_gt, :union].contiguous(),
        ws.tile_logical[:n_gt, :union].contiguous(),
        ws.lengths[:n_gt].contiguous(),
        ws.mask[:n_gt, :words, :union].contiguous(),
    )


def _build_prefill_tiles_eager(
    logical_regions: torch.Tensor,
    packed_regions: torch.Tensor,
    request_query: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    block_q: int,
) -> tuple[torch.Tensor, ...]:
    rows, slots = logical_regions.shape
    request_cpu = request_query.to(device="cpu", dtype=torch.int64).tolist()
    q_starts: list[int] = []
    q_counts: list[int] = []
    run_start = 0
    while run_start < total_q:
        request = request_cpu[run_start]
        run_end = run_start + 1
        while run_end < total_q and request_cpu[run_end] == request:
            run_end += 1
        for start in range(run_start, run_end, block_q):
            q_starts.append(start)
            q_counts.append(min(block_q, run_end - start))
        run_start = run_end
    if not q_starts:
        raise ValueError("cannot build prefill tiles for an empty query batch")

    device = logical_regions.device
    num_tiles = len(q_starts)
    num_group_tiles = num_tiles * num_kv_groups
    q_starts_d = torch.tensor(q_starts, device=device, dtype=torch.int32)
    q_counts_d = torch.tensor(q_counts, device=device, dtype=torch.int32)
    tile_ids = torch.arange(num_tiles, device=device, dtype=torch.int64)
    tile_of_query = torch.repeat_interleave(tile_ids, q_counts_d.to(torch.int64))
    if tile_of_query.numel() != total_q:
        raise ValueError("request boundaries do not cover all prefill queries")

    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    query_of_row = row_ids % total_q
    group_of_row = torch.div(row_ids, total_q, rounding_mode="floor")
    tile_of_row = tile_of_query[query_of_row]
    group_tile = tile_of_row * num_kv_groups + group_of_row

    live = (logical_regions >= 0) & (packed_regions >= 0)
    tile_of = group_tile[:, None].expand(rows, slots)[live]
    q_local = (
        query_of_row - q_starts_d[tile_of_row].to(torch.int64)
    )[:, None].expand(rows, slots)[live]
    logical = logical_regions[live].to(torch.int64)
    physical = (packed_regions[live].to(torch.int64) & ((1 << 24) - 1)).to(torch.int32)

    num_regions = int(logical.max().item()) + 1 if logical.numel() else 1
    present = torch.zeros((num_group_tiles, num_regions), device=device, dtype=torch.bool)
    present[tile_of, logical] = True
    lengths = present.sum(dim=1).to(torch.int32)
    union = max(1, int(lengths.max().item()))

    # Mapping through logical ids is safe because a tile never crosses a request. It also
    # keeps the temporary bitmap proportional to sequence length rather than to the
    # allocator's (potentially very sparse) physical block ids.
    physical_by_logical = torch.zeros(
        (num_group_tiles, num_regions), device=device, dtype=torch.int32
    )
    physical_by_logical[tile_of, logical] = physical
    tile, logical_slot = present.nonzero(as_tuple=True)
    slot_of = present.cumsum(dim=1).to(torch.int64) - 1
    tile_phys = torch.zeros((num_group_tiles, union), device=device, dtype=torch.int32)
    tile_logical = torch.zeros((num_group_tiles, union), device=device, dtype=torch.int32)
    tile_phys[tile, slot_of[tile, logical_slot]] = physical_by_logical[
        tile, logical_slot
    ]
    tile_logical[tile, slot_of[tile, logical_slot]] = logical_slot.to(torch.int32)

    words = (block_q + 31) // 32
    mask64 = torch.zeros((num_group_tiles, words, union), device=device, dtype=torch.int64)
    entry_slots = slot_of[tile_of, logical]
    mask64.index_put_(
        (tile_of, q_local // 32, entry_slots),
        torch.ones_like(q_local, dtype=torch.int64) << (q_local % 32),
        accumulate=True,
    )
    mask = (mask64 - (mask64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
    return q_starts_d, q_counts_d, tile_phys, tile_logical, lengths, mask


def _build_prefill_tiles(
    logical_regions: torch.Tensor,
    packed_regions: torch.Tensor,
    q_positions: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    total_q: int,
    num_kv_groups: int,
    region_size: int,
    block_q: int,
) -> tuple[torch.Tensor, ...]:
    """Build physical union lists and per-query membership masks for tiled prefill.

    Tiles are cut in query space: at every request boundary, then into ``block_q``
    queries. Union lists / bitsets are stored group-major inside each tile as
    ``group_tile = tile * num_kv_groups + group``, matching the kernel grid
    ``(num_tiles, num_kv_groups, heads_per_group)``. A logical region number is
    meaningful only within one request, while the physical cache region is
    request-specific. The returned tensors are all on the input device and can be
    passed directly to the Triton kernel.

    CUDA runs a four-kernel builder (chunked request scan, mark, per-tile compact,
    mask) for ``Q<=4096``. CPU, larger batches, ``block_q>256``, or union overflow
    keep the PyTorch path.
    """
    del region_size
    rows = int(logical_regions.shape[0])
    expected_rows = num_kv_groups * total_q
    if packed_regions.shape != logical_regions.shape:
        raise ValueError(
            "logical and packed prefill metadata must have the same shape, got "
            f"{tuple(logical_regions.shape)} and {tuple(packed_regions.shape)}"
        )
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} metadata rows, got {rows}")
    if q_positions.numel() not in (total_q, expected_rows):
        raise ValueError(
            f"expected {total_q} or {expected_rows} query positions, "
            f"got {q_positions.numel()}"
        )
    if block_q <= 0 or block_q & (block_q - 1):
        raise ValueError(f"block_q must be a positive power of two, got {block_q}")

    # vLLM lays prefill queries out contiguously by request. Request ids may be
    # length Q or a group-major G*Q repeat; tile cuts always follow query space.
    request_query = _as_query_space(
        request_indices,
        total_q=total_q,
        num_kv_groups=num_kv_groups,
        name="request indices",
    )
    use_device = (
        logical_regions.is_cuda
        and 0 < total_q <= _TILE_BUILD_MAX_Q
        and block_q <= _TILE_BUILD_MASK_WORDS * 32
    )
    if use_device:
        built = _build_prefill_tiles_device(
            logical_regions,
            packed_regions,
            request_query,
            total_q=total_q,
            num_kv_groups=num_kv_groups,
            block_q=block_q,
        )
        if built is not None:
            return built
    return _build_prefill_tiles_eager(
        logical_regions,
        packed_regions,
        request_query,
        total_q=total_q,
        num_kv_groups=num_kv_groups,
        block_q=block_q,
    )


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "sparse_attn_prefill",
)
# Occupancy JSON keys consumed by this file:
#   AUTO_KV_SPLITS                         – host only (grid split count; tiled
#                                            and per-query). Tiled splits the
#                                            union list; per-query splits packed
#                                            slots. Both merge with merge_split_states.
#   BLOCK_R, BLOCK_H, BLOCK_D, BLOCK_Q     – tiles
#   num_warps, num_stages                  – standard launch
#   remaining keys                         – HIP compiler options if the backend supports them
# Dsplit path is fixed: 64+64+64, log2 softmax, shared reciprocal, transposed K,
# hybrid K-before-QK / V-after-QK schedule. Tiled uses the same recipe when
# head_dim splits; otherwise the padded tiled kernel still gets log2 / reciprocal
# / transposed K / union splits.
_HOST_CONFIG_KEYS = frozenset({"AUTO_KV_SPLITS"})
_KERNEL_CONFIG_KEYS = frozenset(
    {
        "BLOCK_R",
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_Q",
    }
)


def _build_launch_options(
    config: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build portable Triton launch options from a matched JSON config.

    Kernel constexpr fields are consumed separately. Standard launch fields are
    always accepted; HIP-specific compiler fields are forwarded only when the
    installed backend declares support for them. Unknown and unsupported fields
    are ignored so configs remain portable across Triton versions.
    """
    supports = get_triton_capabilities().supports if torch.version.hip is not None else None
    return _portable_launch_options(
        config,
        consumed_keys=_KERNEL_CONFIG_KEYS | _HOST_CONFIG_KEYS,
        supports=supports,
        defaults=defaults,
    )


def _prefill_config_filename(device: str) -> str:
    """Return ``sparse_attn_prefill_{device}.json``."""
    return f"sparse_attn_prefill_{device}.json"


@functools.lru_cache(maxsize=None)
def _load_prefill_device_configs(device: str) -> dict[str, Any]:
    """Load ``sparse_attn_prefill_{device}.json`` and return its parsed dict."""
    return _load_device_configs(_CONFIG_DIR, _prefill_config_filename, device)


def _load_sparse_attn_prefill_config(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_size: int,
) -> dict[str, Any] | None:
    """Select a per-query config by unique ``sparse_attention_prefill`` inputs.

    Duplicate extents are recorded once:

    * ``s_q`` / ``h_q`` / ``d`` from ``query``
      (``[total_q, num_heads, head_dim]``). ``d`` also covers K/V head dim.
    * ``s_k`` / ``h_kv`` / ``dtype`` from ``key_cache``
      (``[tokens, num_kv_groups, head_dim]``). ``value_cache`` must equal
      ``key_cache``; ``num_kv_groups`` is ``h_kv``. Query stays bfloat16 on
      the FP8 serving path; ``float8_e4m3fn`` vs ``bfloat16`` is the cache.
    * ``topk`` from ``packed_regions.shape[1]``. Row count is ``h_kv * s_q``.
    * ``region_size`` from the keyword argument.

    Not match keys: ``region_counts`` (same rows as ``packed_regions``);
    ``softmax_scale`` (defaults from ``d``); ``block_regions`` (fallback
    ``BLOCK_R`` only); ``logical_regions`` / ``q_positions`` /
    ``request_indices`` / ``block_q`` (tiled JSON).
    """
    device_tag = _device_name_for_config()
    data = _load_prefill_device_configs(device_tag)
    shape = {
        "s_q": int(query.shape[0]),
        "h_q": int(query.shape[1]),
        "d": int(query.shape[2]),
        "s_k": int(key_cache.shape[0]),
        "h_kv": int(key_cache.shape[1]),
        "topk": int(packed_regions.shape[1]),
        "region_size": int(region_size),
        "dtype": _occupancy_dtype(key_cache.dtype),
    }
    return _select_config(data, shape)


def _per_query_tile_defaults(
    *,
    heads_per_group: int,
    head_dim: int,
    topk: int,
    block_regions: int,
    region_size: int,
    element_size: int,
) -> tuple[int, int, int]:
    block_h = max(16, triton.next_power_of_2(heads_per_group))
    block_d = triton.next_power_of_2(head_dim)
    block_r = _fit_block_regions(
        min(block_regions, triton.next_power_of_2(topk)),
        region_size=region_size,
        block_d=block_d,
        element_size=element_size,
    )
    return block_h, block_d, block_r


def choose_prefill_kv_splits(
    total_q: int,
    num_kv_groups: int,
    packed_width: int,
    *,
    target_programs: int | None = None,
    min_slots_per_split: int = 8,
    device: torch.device | None = None,
) -> int:
    """Pick a split count that fills CUs without emptying each split's slot window.

    Size-aware occupancy only: uses ``Q * G`` and device CU count. Caps splits so
    ``ceil(packed_width / splits) >= min_slots_per_split`` when the row is full.
    Extra-wave occupancy beyond CU-fill is pinned per match by ``AUTO_KV_SPLITS``
    in the occupancy JSON. The chooser itself only covers ``Q * G < CUs``.
    """
    return _choose_kv_splits(
        total_q,
        num_kv_groups,
        packed_width,
        target_programs=target_programs,
        min_slots_per_split=min_slots_per_split,
        device=device,
    )


def _resolve_prefill_kv_splits(
    launch: dict[str, Any] | None,
    *,
    total_q: int,
    num_kv_groups: int,
    packed_width: int,
    device: torch.device | None = None,
) -> int:
    """Pinned ``AUTO_KV_SPLITS`` if present, otherwise CU-fill occupancy."""
    pinned = launch.get("AUTO_KV_SPLITS") if launch else None
    if pinned is not None:
        return _clamp_auto_kv_splits(int(pinned), packed_width)
    return choose_prefill_kv_splits(
        total_q,
        num_kv_groups,
        packed_width,
        device=device,
    )


def sparse_attn_prefill_tiled_union_bucket(union: int) -> int:
    """Bucket a tile-union width with ``next_power_of_2`` for config matching."""
    return int(triton.next_power_of_2(max(1, int(union))))


def _prefill_tiled_config_filename(device: str) -> str:
    """Return ``sparse_attn_prefill_tiled_{device}.json``."""
    return f"sparse_attn_prefill_tiled_{device}.json"


@functools.lru_cache(maxsize=None)
def _load_prefill_tiled_device_configs(device: str) -> dict[str, Any]:
    """Load ``sparse_attn_prefill_tiled_{device}.json`` and return its parsed dict."""
    return _load_device_configs(_CONFIG_DIR, _prefill_tiled_config_filename, device)


def _load_sparse_attn_prefill_tiled_config(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    q_starts: torch.Tensor,
    tile_phys: torch.Tensor,
    packed_width: int,
    region_size: int,
    block_q: int,
) -> dict[str, Any] | None:
    """Select the best matching tiled-prefill config for the runtime shape.

    The fields mirror MLA's tensor-shape based matching while avoiding
    equivalent dimensions from key/value cache and tile metadata:

    * ``s_q``          – number of query tokens          (``query.shape[0]``)
    * ``h_q``          – total query heads               (``query.shape[1]``)
    * ``d``            – common head dimension           (``query.shape[2]``)
    * ``s_k``          – KV-cache tokens                 (``key_cache.shape[0]``)
    * ``h_kv``         – KV-cache head groups            (``key_cache.shape[1]``)
    * ``num_tiles``    – number of query tiles           (``q_starts.numel()``)
    * ``topk``         – packed selected-region width    (``packed_width``)
    * ``union_bucket`` – bucketed tile-union width       (``tile_phys.shape[1]``)
    * ``region_size``  – tokens per region               (runtime argument)
    * ``block_q``      – query tile width                (runtime argument)
    * ``dtype``        – KV-cache element type           (``key_cache.dtype``)

    ``block_regions`` is intentionally excluded: it only determines the
    fallback ``BLOCK_R`` when no tuned config matches. A matched config supplies
    ``BLOCK_R`` directly. ``value_cache`` is also excluded because its shape is
    required to equal ``key_cache.shape`` by the public entry point.
    """
    device_tag = _device_name_for_config()
    data = _load_prefill_tiled_device_configs(device_tag)
    union_bucket = sparse_attn_prefill_tiled_union_bucket(int(tile_phys.shape[1]))
    shape = {
        "s_q": int(query.shape[0]),
        "h_q": int(query.shape[1]),
        "d": int(query.shape[2]),
        "s_k": int(key_cache.shape[0]),
        "h_kv": int(key_cache.shape[1]),
        "num_tiles": int(q_starts.numel()),
        "topk": int(packed_width),
        "union_bucket": union_bucket,
        "region_size": int(region_size),
        "block_q": int(block_q),
        "dtype": _occupancy_dtype(key_cache.dtype),
    }
    return _select_config(data, shape)


def _run_tiled_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    q_starts: torch.Tensor,
    q_counts: torch.Tensor,
    tile_phys: torch.Tensor,
    tile_logical: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
    q_positions: torch.Tensor,
    *,
    packed_width: int,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float,
    block_q: int,
    block_regions: int = 4,
    config: dict[str, Any] | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the query-tiled kernel on already-built union lists.

    Callers that need end-to-end serving cost should use
    :func:`sparse_attention_prefill`, which rebuilds tiles on every call.
    Bench / xprof / tuner pass prebuilt tiles so CUDA-event timing matches
    the Triton kernel in isolation. ``v_scale`` dequantizes an fp8 V cache on
    the fp32 output; ``k_scale`` must already be folded into ``softmax_scale``.

    ``AUTO_KV_SPLITS`` in the launch JSON partitions the union list across extra
    CTAs and merges with :func:`merge_split_states`. ``head_dim=192`` selects
    a BF16 or FP8 64+64+64 dsplit kernel from the actual KV-cache dtype;
    other head dimensions share the original padded tiled fallback.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    _require_matching_kv_dtypes(key_cache, value_cache)
    _require_contiguous_last_dim(query, "query")
    _require_contiguous_last_dim(key_cache, "key_cache")
    _require_contiguous_last_dim(value_cache, "value_cache")
    heads_per_group = num_heads // num_kv_groups
    if out is None:
        out = torch.empty_like(query)
    _require_contiguous_last_dim(out, "out")
    if lse is None:
        lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    num_tiles = int(q_starts.numel())
    group_tiles = num_tiles * num_kv_groups
    if int(tile_phys.shape[0]) != group_tiles:
        raise ValueError(
            "tile union lists are laid out as (query_tile, kv_group); expected "
            f"{group_tiles} rows, got {tile_phys.shape[0]}"
        )
    if q_positions.numel() == num_kv_groups * total_q:
        q_positions = q_positions.reshape(num_kv_groups, total_q)[0].contiguous()
    elif q_positions.numel() != total_q:
        raise ValueError(
            f"expected {total_q} or {num_kv_groups * total_q} query positions, "
            f"got {q_positions.numel()}"
        )
    else:
        q_positions = q_positions.reshape(total_q).contiguous()
    union = int(tile_phys.shape[1])
    launch = config
    if launch is None:
        launch = _load_sparse_attn_prefill_tiled_config(
            query,
            key_cache,
            q_starts,
            tile_phys,
            packed_width,
            region_size,
            block_q,
        )
    default_r = _fit_block_regions(
        min(block_regions, triton.next_power_of_2(union)),
        region_size=region_size,
        block_d=block_d,
        element_size=query.element_size(),
    )
    extra = {"num_warps": 4, "num_stages": 2}
    if launch:
        block_d = int(launch.get("BLOCK_D", block_d))
        block_r = int(launch.get("BLOCK_R", default_r))
        extra = _build_launch_options(launch, defaults=extra)
        pinned = launch.get("AUTO_KV_SPLITS")
        num_splits = (
            _clamp_auto_kv_splits(int(pinned), union) if pinned is not None else 1
        )
    else:
        block_r = default_r
        num_splits = 1
    slots_per_split = -(-union // num_splits)
    if num_splits > 1:
        kernel_out = torch.empty(
            (num_splits, total_q, num_heads, head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        _require_contiguous_last_dim(kernel_out, "out")
        kernel_lse = torch.empty(
            (num_splits, total_q, num_heads),
            device=query.device,
            dtype=torch.float32,
        )
        stride_out_split = kernel_out.stride(0)
        stride_out_token = kernel_out.stride(1)
        stride_out_head = kernel_out.stride(2)
        stride_lse_split = kernel_lse.stride(0)
        stride_lse_token = kernel_lse.stride(1)
    else:
        kernel_out = out
        kernel_lse = lse
        stride_out_split = 0
        stride_out_token = kernel_out.stride(0)
        stride_out_head = kernel_out.stride(1)
        stride_lse_split = 0
        stride_lse_token = kernel_lse.stride(0)
    common_args = (
        query,
        key_cache,
        value_cache,
        tile_phys,
        tile_logical,
        lengths,
        mask,
        q_starts,
        q_counts,
        q_positions,
        kernel_out,
        kernel_lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        tile_phys.stride(0),
        tile_phys.stride(1),
        tile_logical.stride(0),
        tile_logical.stride(1),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        stride_out_split,
        stride_out_token,
        stride_out_head,
        stride_lse_split,
        stride_lse_token,
        softmax_scale,
        v_scale,
        total_q,
        num_kv_groups,
        num_splits,
        slots_per_split,
    )
    common_meta = dict(
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        BLOCK_Q=block_q,
        BLOCK_R=block_r,
    )
    grid = (num_tiles, num_kv_groups, heads_per_group * num_splits)
    dim_split = _prefill_head_dim_split(head_dim)
    fp8_cache = _cache_is_fp8(key_cache.dtype)
    if dim_split is not None:
        d1, d2, d3 = dim_split
        kernel = (
            _sparse_attn_prefill_tiled_dsplit_fp8_kernel
            if fp8_cache
            else _sparse_attn_prefill_tiled_dsplit_bf16_kernel
        )
        kernel[grid](
            *common_args,
            BLOCK_D1=d1,
            BLOCK_D2=d2,
            BLOCK_D3=d3,
            **common_meta,
            **extra,
        )
    else:
        # The original padded tiled kernel is shared by BF16 and FP8.
        _sparse_attn_prefill_tiled_kernel[grid](
            *common_args,
            BLOCK_D=block_d,
            **common_meta,
            **extra,
        )
    if num_splits == 1:
        return kernel_out, kernel_lse
    from boltops.step4_dsa.triton.merge import merge_split_states

    return merge_split_states(kernel_out, kernel_lse, out=out, lse=lse)


def sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    block_regions: int = 16,
    logical_regions: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    request_indices: torch.Tensor | None = None,
    block_q: int = 128,
    config: dict[str, Any] | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over per-query selected regions.

    Args:
        query: ``[total_q, num_heads, head_dim]``. Last dim must be contiguous
            (``stride(-1) == 1``); token/head strides are passed through. The
            kernel addresses ``head_dim`` as ``base + d``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, i.e. the paged
            cache already flattened across pages. Flattening is safe because a region's
            tokens are contiguous in the physical token space by construction.
            Same last-dim rule as ``query``. Allocated ``out`` (and a tiled
            launcher ``out`` buffer) is checked the same way. May hold
            ``float8_e4m3`` instead of the query dtype; see ``k_scale``/``v_scale``.
        packed_regions: ``[num_kv_groups * total_q, topk]`` int32 from the selector,
            each slot ``phys_region | (valid_tokens << 24)``, ``-1`` padding the tail.
        region_counts: ``[num_kv_groups * total_q]`` int32, how many slots to read. This
            truncates *positionally* -- it does not re-select by score.
        logical_regions: optional logical ids parallel to ``packed_regions``. When supplied,
            the query-tiled implementation is used and ``q_positions`` is required.
        q_positions: absolute positions of each query (length ``total_q``), or the
            same vector repeated group-major as ``[G * total_q]``.
        request_indices: request id per query (length ``total_q``), or the same ids
            repeated group-major. Tiles are split at changes in this vector so
            per-request logical ids never alias.
        block_q: query tile width for the tiled implementation; must be a power of two.
        config: launch override. ``None`` loads a device-specific JSON and selects
            the most specific matching entry. Per-query prefill reads
            ``sparse_attn_prefill_{device}.json``; tiled prefill reads
            ``sparse_attn_prefill_tiled_{device}.json`` after tile metadata is built.
            Missing tile fields use the built-in defaults; supported Triton compiler
            options are forwarded according to the active backend. An empty mapping
            skips lookup and keeps all built-in defaults.
            On the per-query path, ``AUTO_KV_SPLITS`` pins extra-wave KV splits so
            ``Q * G`` occupancy can hide gather latency; if absent,
            :func:`choose_prefill_kv_splits` fills CUs (``ceil(CUs / (Q*G))``,
            usually 1 at production ``Q=128``). On the tiled path the same key
            partitions the per-tile union list (production JSON uses 18) so
            ``tiles * G * heads`` can fill CUs; absent means one split. The key
            is host-only. Partials are always merged with :func:`merge_split_states`.
        k_scale, v_scale: per-tensor dequantization scales for an fp8 cache, i.e. the
            true tensor is ``cache * scale``. Both default to 1.0. ``k_scale`` is
            folded into ``softmax_scale`` on the host; ``v_scale`` multiplies the
            fp32 output. Staging uses the query element size, because K/V are widened
            to the query dtype immediately after the load.

    Returns:
        ``(out, lse)`` shaped ``[total_q, num_heads, head_dim]`` and
        ``[total_q, num_heads]``. Rows with no selected region get a zero output and
        ``lse = -inf``.

    The caller must guarantee the selected regions are distinct. Our selector does (the
    stage 3 gate checks it), but a union-with-sliding-window path would not, and duplicate
    regions would silently double-weight those tokens in the softmax rather than fail.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * total_q:
        raise ValueError(
            f"expected {num_kv_groups * total_q} metadata rows, got {packed_regions.shape[0]}"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {tuple(key_cache.shape)} vs {tuple(value_cache.shape)}"
        )
    _require_matching_kv_dtypes(key_cache, value_cache)
    _require_contiguous_last_dim(query, "query")
    _require_contiguous_last_dim(key_cache, "key_cache")
    _require_contiguous_last_dim(value_cache, "value_cache")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    # k_scale rides here rather than on the loaded keys: the product lands on the fp32
    # score tile, where it costs nothing and cannot re-quantize an fp8 operand.
    scale = scale * float(k_scale)
    v_scale = float(v_scale)

    if logical_regions is not None:
        if q_positions is None:
            raise ValueError("q_positions is required for tiled sparse prefill")
        if request_indices is None:
            request_indices = torch.zeros(
                (total_q,), device=query.device, dtype=torch.int32
            )
        q_positions = _as_query_space(
            q_positions.to(device=query.device, dtype=torch.int32).contiguous(),
            total_q=total_q,
            num_kv_groups=num_kv_groups,
            name="query positions",
        )
        request_indices = request_indices.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        if logical_regions.shape != packed_regions.shape:
            raise ValueError(
                "logical_regions and packed_regions must have identical shapes"
            )
        q_starts, q_counts, tile_phys, tile_logical, lengths, mask = _build_prefill_tiles(
            logical_regions.contiguous(),
            packed_regions.contiguous(),
            q_positions,
            request_indices,
            total_q=total_q,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            block_q=block_q,
        )
        out = torch.empty_like(query)
        _require_contiguous_last_dim(out, "out")
        lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
        return _run_tiled_prefill(
            query,
            key_cache,
            value_cache,
            q_starts,
            q_counts,
            tile_phys,
            tile_logical,
            lengths,
            mask,
            q_positions,
            packed_width=topk,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            softmax_scale=scale,
            block_q=block_q,
            block_regions=block_regions,
            config=config,
            out=out,
            lse=lse,
            v_scale=v_scale,
        )

    from boltops.step4_dsa.triton.merge import merge_split_states

    launch = config
    if launch is None:
        launch = _load_sparse_attn_prefill_config(
            query,
            key_cache,
            packed_regions,
            region_size,
        )
    num_splits = _resolve_prefill_kv_splits(
        launch,
        total_q=total_q,
        num_kv_groups=num_kv_groups,
        packed_width=topk,
        device=query.device,
    )
    slots_per_split = -(-topk // num_splits)

    out = torch.empty(
        (num_splits, total_q, num_heads, head_dim),
        device=query.device,
        dtype=query.dtype,
    )
    _require_contiguous_last_dim(out, "out")
    lse = torch.empty(
        (num_splits, total_q, num_heads), device=query.device, dtype=torch.float32
    )
    default_h, default_d, default_r = _per_query_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        topk=slots_per_split,
        block_regions=block_regions,
        region_size=region_size,
        element_size=query.element_size(),
    )
    extra: dict[str, Any] = {}
    if launch:
        block_h = int(launch.get("BLOCK_H", default_h))
        block_d = int(launch.get("BLOCK_D", default_d))
        block_r = int(launch.get("BLOCK_R", default_r))
        extra = _build_launch_options(launch)
    else:
        block_h, block_d, block_r = default_h, default_d, default_r

    common_args = (
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        scale,
        v_scale,
        total_q,
    )
    common_meta = dict(
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
    )
    dim_split = _prefill_head_dim_split(head_dim)
    if dim_split is not None:
        d1, d2, d3 = dim_split
        kernel = (
            _sparse_attn_prefill_dsplit_fp8_kernel
            if _cache_is_fp8(key_cache.dtype)
            else _sparse_attn_prefill_dsplit_bf16_kernel
        )
        kernel[(total_q, num_kv_groups, num_splits)](
            *common_args,
            BLOCK_D1=d1,
            BLOCK_D2=d2,
            BLOCK_D3=d3,
            **common_meta,
            **extra,
        )
    else:
        # Other head dims stay on the generic padded fallback for both dtypes.
        _sparse_attn_prefill_kernel[(total_q, num_kv_groups, num_splits)](
            *common_args,
            **common_meta,
            BLOCK_D=block_d,
            **extra,
        )
    if num_splits == 1:
        return out[0], lse[0]
    return merge_split_states(out, lse)


# Previous landing name; same function.
sparse_attention_prefill_optimized = sparse_attention_prefill
