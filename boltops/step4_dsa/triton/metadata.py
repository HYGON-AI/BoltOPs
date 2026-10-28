# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Prefill / decode sparse-attention metadata construction.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``dsa_meta``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import REGION_BLOCK_SIZE, shared_table_stride
from boltops.step4_dsa.triton.topk import region_topk_ids, region_topk_pack

PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
DECODE_PHYS_SHIFT = tl.constexpr(32)
# Larger than any real packed word, so invalid slots sort to the tail.
DECODE_SORT_SENTINEL = tl.constexpr(1 << 62)


@triton.jit
def _append_current_region_kernel(
    history_ptr,
    history_ids_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    ids_out_ptr,
    counts_ptr,
    stride_history_row,
    stride_history_ids_row,
    stride_table_row,
    stride_out_row,
    stride_ids_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    WRITE_LOGICAL_IDS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Widen prefill history metadata to ``topk + 1`` slots and append the current region.

    The history is copied verbatim -- including its ``-1`` slots, which stay *in place*
    rather than being compacted, because prefill's ``region_counts`` is positional
    (``min(current_region, topk) + 1``) and not a count of live entries. The appended slot
    lands at exactly the index the selector stopped filling.

    The two stores are kept **disjoint** (``slots != history_count``) rather than letting
    the second overwrite the first. Writing the whole row and then patching one slot is a
    cross-thread race: the row store is a vector store spread over lanes, the patch is a
    scalar store from one lane, and nothing orders them. It loses roughly one row in a
    thousand, so it survives a small gate and fails a large one. The upstream kernel
    carries a comment about this exact hazard, which suggests it was found the same way.
    """
    row = tl.program_id(0)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)
    current_region = tl.maximum(q_position // region_size, 0)
    history_count = tl.minimum(current_region, topk)

    slots = tl.arange(0, BLOCK_K)
    history = tl.load(
        history_ptr + row * stride_history_row + slots, mask=slots < topk, other=-1
    )
    tl.store(
        out_ptr + row * stride_out_row + slots,
        history,
        mask=(slots < topk + 1) & (slots != history_count),
    )

    if WRITE_LOGICAL_IDS:
        history_ids = tl.load(
            history_ids_ptr + row * stride_history_ids_row + slots,
            mask=slots < topk,
            other=-1,
        )
        tl.store(
            ids_out_ptr + row * stride_ids_out_row + slots,
            history_ids,
            mask=(slots < topk + 1) & (slots != history_count),
        )

    page = tl.load(
        block_table_ptr + request * stride_table_row + current_region // regions_per_page
    )
    physical = page * regions_per_page + current_region % regions_per_page
    valid_tokens = tl.minimum(
        tl.maximum(q_position + 1 - current_region * region_size, 0), region_size
    )
    packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
    tl.store(
        out_ptr + row * stride_out_row + history_count,
        tl.where(page >= 0, packed, -1),
    )
    if WRITE_LOGICAL_IDS:
        tl.store(
            ids_out_ptr + row * stride_ids_out_row + history_count,
            tl.where(page >= 0, current_region, -1),
        )
    tl.store(counts_ptr + row, history_count + 1)


@triton.jit
def _decode_pack_kernel(
    ids_ptr,
    seqlens_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    counts_ptr,
    stride_ids_row,
    stride_table_row,
    stride_out_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Pack selected logical regions into the int64 decode word.

    One row fits in one tile, so the compaction is a sort rather than a scatter: invalid
    slots are mapped to a sentinel above every real word, and sorting by the packed value
    both drops them to the tail and produces the physical-major order the upstream
    converter emits. The current region is then written at the surviving count -- after the
    sort, so it stays last, which is also where the reference puts it.
    """
    row = tl.program_id(0)
    request = tl.load(request_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int32)
    valid_regions = (kv_seqlen + region_size - 1) // region_size

    slots = tl.arange(0, BLOCK_K)
    ids = tl.load(ids_ptr + row * stride_ids_row + slots, mask=slots < topk, other=-1)

    live = (slots < topk) & (ids >= 0) & (ids < valid_regions)
    safe_ids = tl.where(live, ids, 0)
    pages = tl.load(
        block_table_ptr + request * stride_table_row + safe_ids // regions_per_page,
        mask=live,
        other=-1,
    )
    live = live & (pages >= 0)

    physical = pages * regions_per_page + safe_ids % regions_per_page
    packed = (safe_ids * region_size).to(tl.int64) | (
        physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    packed = tl.where(live, packed, DECODE_SORT_SENTINEL)
    packed = tl.sort(packed, descending=False)

    count = tl.sum(live.to(tl.int32))
    tl.store(
        out_ptr + row * stride_out_row + slots,
        tl.where(slots < count, packed, 0),
        # Slot ``count`` is left to the scalar store below. Overlapping the two would be a
        # cross-thread race between a vector store and a scalar one -- see
        # :func:`_append_current_region_kernel`.
        mask=(slots < topk + 1) & (slots != count),
    )

    current_region = tl.maximum((kv_seqlen - 1) // region_size, 0)
    current_page = tl.load(
        block_table_ptr + request * stride_table_row + current_region // regions_per_page
    )
    current_live = (current_region < valid_regions) & (current_page >= 0)
    current_physical = (
        current_page * regions_per_page + current_region % regions_per_page
    )
    current_packed = (current_region * region_size).to(tl.int64) | (
        current_physical.to(tl.int64) << DECODE_PHYS_SHIFT
    )
    # Written unconditionally so the slot is defined even when the current region is
    # unmapped, in which case it becomes ordinary zero padding.
    tl.store(
        out_ptr + row * stride_out_row + count,
        tl.where(current_live, current_packed, 0),
    )
    tl.store(counts_ptr + row, count + current_live.to(tl.int32))


def prefill_sparse_meta(
    logits: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    return_logical_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select regions and build prefill sparse metadata.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed,
            group-major (``rows = num_kv_groups * total_q``). The scores are weighted
            sums of per-head ReLU terms and unconstrained signed indexer weights.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page, which yields a ``-1`` slot rather
            than an error.
        request_indices: ``[rows]`` int32, which block-table row each score row uses.
            Defaults to all-zero, i.e. every row shares one table.

    Returns:
        ``(packed [rows, topk + 1] int32, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_prefill`. When
        ``return_logical_ids`` is true, a third ``[rows, topk + 1]`` int32 tensor is
        returned; it contains the logical region id for each live slot and ``-1`` padding.

    The history length is derived here as ``q_position // region_size`` rather than taken
    from the caller, because getting it wrong by one is the difference between the current
    region being force-included and it competing for a slot.
    """
    rows = int(logits.shape[0])
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    history_lengths = torch.div(
        q_positions.to(torch.int32), region_size, rounding_mode="floor"
    ).contiguous()

    if return_logical_ids:
        history, logical_history = region_topk_pack(
            logits,
            history_lengths,
            q_positions,
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
            _logical_out=torch.empty(
                (rows, int(topk)), device=logits.device, dtype=torch.int32
            ),
        )
    else:
        history = region_topk_pack(
            logits,
            history_lengths,
            q_positions,
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
        )
        logical_history = None

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int32)
    logical_ids = (
        torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int32)
        if return_logical_ids
        else None
    )
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _append_current_region_kernel[(rows,)](
        history,
        history if logical_history is None else logical_history,
        q_positions,
        block_table,
        request_indices,
        packed,
        packed if logical_ids is None else logical_ids,
        counts,
        history.stride(0),
        history.stride(0) if logical_history is None else logical_history.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        packed.stride(0) if logical_ids is None else logical_ids.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        WRITE_LOGICAL_IDS=return_logical_ids,
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    if return_logical_ids:
        return packed, counts, logical_ids
    return packed, counts


def decode_sparse_meta(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build decode sparse metadata.

    Args:
        logits: ``[num_kv_groups * num_reqs, seq_regions]`` float32 indexer scores,
            which may be signed, group-major. The selector orders them numerically rather
            than by the magnitude encoded in their IEEE-754 bit pattern.
        kv_seqlens: ``[num_reqs]`` int32, context length *including* the current token.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page.
        request_indices: ``[rows]`` int32. Defaults to the group-major layout implied by
            the row count, ``arange(num_reqs)`` tiled once per KV group.

    Returns:
        ``(packed [rows, topk + 1] int64, region_counts [rows] int32)``, ready for
        :func:`inference.sparse_attn.sparse_attention_decode`.

    ``region_counts`` here is a count of *surviving* entries, so it varies with how much of
    the block table is mapped. The tail beyond it is zero, and a zero word is not
    self-identifying -- the count is the only bound.
    """
    rows, _ = logits.shape
    num_reqs = int(kv_seqlens.shape[0])
    if rows % num_reqs:
        raise ValueError(f"{rows} score rows do not divide into {num_reqs} requests")

    if request_indices is None:
        request_indices = (
            torch.arange(num_reqs, device=logits.device, dtype=torch.int32)
            .repeat(rows // num_reqs)
            .contiguous()
        )
    # Decode's query position is the last token, so the history is everything strictly
    # before the current region.
    history_lengths = torch.div(
        (kv_seqlens.to(torch.int32) - 1).clamp_min(0), region_size, rounding_mode="floor"
    )[request_indices.to(torch.long)].contiguous()

    ids = region_topk_ids(logits, history_lengths, topk=topk)

    packed = torch.empty((rows, int(topk) + 1), device=logits.device, dtype=torch.int64)
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _decode_pack_kernel[(rows,)](
        ids,
        kv_seqlens,
        block_table,
        request_indices,
        packed,
        counts,
        ids.stride(0),
        shared_table_stride(block_table),
        packed.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        BLOCK_K=triton.next_power_of_2(int(topk) + 1),
    )
    return packed, counts
