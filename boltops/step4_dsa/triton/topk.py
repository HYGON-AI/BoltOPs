"""Signed region top-k (logical ids) and fused prefill pack.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``dsa_kernels`` selector).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import REGION_BLOCK_SIZE, shared_table_stride

# Prefill selection packs one entry as ``phys_region | (valid_tokens << 24)``.
# These are ``tl.constexpr`` because Triton kernels cannot read plain module globals.
PREFILL_REGION_VALID_SHIFT = tl.constexpr(24)
RADIX_BITS = tl.constexpr(8)
RADIX_BUCKETS = tl.constexpr(256)
RADIX_PASSES = tl.constexpr(4)


@triton.jit
def _ordered_float_key(scores):
    """Map fp32 values to monotonically ordered unsigned-32 keys held in int64.

    Negative IEEE-754 values reverse their bit ordering, while non-negative values
    become ordered after setting the sign bit. Keeping the resulting 32-bit pattern
    in a non-negative int64 lets ordinary Triton integer comparisons implement the
    required unsigned ordering. Branching on ``scores < 0`` deliberately maps
    ``-0.0`` and ``+0.0`` to the same key, matching numeric equality and CuTe.

    NaNs follow CuTe's raw ordered-float behavior and sort above ``+inf``. Production
    indexer scores are expected to be finite; this helper does not add a costly
    full-tensor finiteness scan to the hot path.
    """
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(
        scores < 0.0,
        (~bits) & 0xFFFFFFFF,
        bits | 0x80000000,
    )


@triton.jit
def _radix_topk_threshold(
    logits_ptr,
    row,
    stride_logits_row,
    visible,
    budget,
    seq_regions,
    BLOCK_R: tl.constexpr,
):
    """Find the exact score threshold separating the top ``budget`` entries of one row.

    Returns ``(threshold, remaining)``: the fp32 bit pattern of the ``budget``-th largest
    score, and how many entries *at* that exact key still belong in the selection. A
    threshold plus a tie count is what lets the caller take a **prefix** of a tied group
    rather than all or none of it, without ever materializing a sort.

    Scores may be signed. :func:`_ordered_float_key` converts their IEEE-754 bit
    patterns into unsigned-monotone keys before the radix passes.
    """
    threshold = budget.to(tl.int64) * 0
    remaining = budget
    bucket_ids = tl.arange(0, RADIX_BUCKETS)

    # Narrow the threshold one byte at a time, most significant first.
    for radix_pass in tl.static_range(RADIX_PASSES):
        shift = 32 - RADIX_BITS * (radix_pass + 1)
        histogram = tl.zeros((RADIX_BUCKETS,), dtype=tl.int32)
        for base in tl.range(0, seq_regions, BLOCK_R):
            regions = base + tl.arange(0, BLOCK_R)
            live = regions < visible
            scores = tl.load(
                logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
            )
            keys = _ordered_float_key(scores)
            if radix_pass == 0:
                # No high bits are fixed yet. Testing them would need a shift by 32,
                # which is undefined.
                matches = live
            else:
                high = shift + RADIX_BITS
                matches = live & ((keys >> high) == (threshold >> high))
            # Park non-matching lanes in bucket 0 and subtract them back out. Relying on
            # tl.histogram to drop an out-of-range sentinel would corrupt the bucket
            # totals if it clamps instead -- and the totals are what pick the cut, so the
            # failure would look like a wrong threshold rather than a bad histogram.
            buckets = tl.where(
                matches,
                ((keys >> shift) & (RADIX_BUCKETS - 1)).to(tl.int32),
                0,
            )
            counts = tl.histogram(buckets, RADIX_BUCKETS)
            parked = tl.sum((~matches).to(tl.int32))
            histogram += counts - tl.where(bucket_ids == 0, parked, 0)

        total = tl.sum(histogram)
        above = total - tl.cumsum(histogram, axis=0)
        # ``above`` is non-increasing, so the buckets satisfying above < remaining form a
        # suffix; the smallest of them is the one holding the cut.
        chosen = tl.min(tl.where(above < remaining, bucket_ids, RADIX_BUCKETS))
        remaining -= tl.sum(tl.where(bucket_ids > chosen, histogram, 0))
        # Cast before shifting the MSB byte. An int32 bucket >= 128 would otherwise
        # become negative first and sign-extend into the int64 threshold.
        threshold |= chosen.to(tl.int64) << shift

    return threshold, remaining


@triton.jit
def _region_topk_pack_kernel(
    logits_ptr,
    lengths_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    logical_out_ptr,
    seq_regions,
    stride_logits_row,
    stride_table_row,
    stride_out_row,
    stride_logical_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    WRITE_LOGICAL_IDS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions of one row and pack them.

    Selection is ``torch.topk`` with ties broken by ascending region id: order the
    visible scores by ``(-score, id)`` and keep the first ``min(topk, visible)``. Ties at
    the cut take a *prefix* of the tied group, they do not drop it -- and that case is
    common rather than exotic, because post-ReLU indexer scores are exactly zero about
    6.25% of the time, so the group sitting at the cut is usually the zero group.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)
    q_position = tl.load(q_positions_ptr + row).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if WRITE_LOGICAL_IDS:
        tl.store(
            logical_out_ptr + row * stride_logical_row + slots,
            -1,
            mask=slots < topk,
        )
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr, row, stride_logits_row, visible, budget, seq_regions, BLOCK_R=BLOCK_R
    )

    # Emit in ascending region order. The CuTe selector emits in radix-bucket order, but
    # that order is not part of the contract -- consumers compare sets -- and ascending
    # is both cheaper here and what the standalone metadata converter produces.
    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        safe_regions = tl.where(selected, regions, 0)
        pages = tl.load(
            block_table_ptr
            + request * stride_table_row
            + safe_regions // regions_per_page,
            mask=selected,
            other=0,
        )
        physical = pages * regions_per_page + safe_regions % regions_per_page
        valid_tokens = tl.minimum(
            tl.maximum(q_position + 1 - safe_regions * region_size, 0), region_size
        )
        packed = physical | (valid_tokens << PREFILL_REGION_VALID_SHIFT)
        # An unmapped page is a legal block-table state, not an error. It has to become a
        # ``-1`` slot rather than an address computed from a negative page, which would
        # read from before the cache.
        packed = tl.where(pages >= 0, packed, -1)
        tl.store(
            out_ptr + row * stride_out_row + positions,
            packed,
            mask=selected & (positions < topk),
        )
        if WRITE_LOGICAL_IDS:
            tl.store(
                logical_out_ptr + row * stride_logical_row + positions,
                regions,
                mask=selected & (positions < topk),
            )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


@triton.jit
def _region_topk_ids_kernel(
    logits_ptr,
    lengths_ptr,
    out_ptr,
    seq_regions,
    stride_logits_row,
    stride_out_row,
    topk: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select the top-``topk`` visible regions and emit **logical ids**, ascending.

    The same selection as :func:`_region_topk_pack_kernel` with the packing left off.
    Prefill and decode pack the identical selection into two different words, and the
    decode word needs the logical region id, which the prefill word does not carry -- so
    decode metadata cannot be derived from prefill metadata, only from the raw ids.
    """
    row = tl.program_id(0)
    visible = tl.load(lengths_ptr + row).to(tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), seq_regions)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(out_ptr + row * stride_out_row + slots, -1, mask=slots < topk)
    if visible <= 0:
        return

    budget = tl.minimum(topk, visible)
    threshold, remaining = _radix_topk_threshold(
        logits_ptr, row, stride_logits_row, visible, budget, seq_regions, BLOCK_R=BLOCK_R
    )

    cursor = 0
    ties_taken = 0
    for base in tl.range(0, seq_regions, BLOCK_R):
        regions = base + tl.arange(0, BLOCK_R)
        live = regions < visible
        scores = tl.load(
            logits_ptr + row * stride_logits_row + regions, mask=live, other=0.0
        )
        keys = _ordered_float_key(scores)
        greater = live & (keys > threshold)
        tied = live & (keys == threshold)
        tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
        selected = greater | (tied & (tied_rank < remaining))

        positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(
            out_ptr + row * stride_out_row + positions,
            regions,
            mask=selected & (positions < topk),
        )
        cursor += tl.sum(selected.to(tl.int32))
        ties_taken += tl.sum(tied.to(tl.int32))


def region_topk_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int,
    block_r: int = 1024,
) -> torch.Tensor:
    """Top-k region selection, returning logical region ids in ascending order.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed.
        lengths: ``[rows]`` int32, candidate region count per row. For DSA this is the
            **history** length, ``q_position // region_size``, which excludes the region
            containing the query itself -- that one is appended by the metadata builder
            rather than made to compete.

    Returns:
        ``[rows, topk]`` int32 logical region ids, ``-1`` padding the tail.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")

    rows, seq_regions = logits.shape
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    _region_topk_ids_kernel[(rows,)](
        logits,
        lengths,
        out,
        seq_regions,
        logits.stride(0),
        out.stride(0),
        topk=int(topk),
        BLOCK_R=min(block_r, triton.next_power_of_2(seq_regions)),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
    )
    return out


def region_topk_pack(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor | None = None,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
    block_r: int = 1024,
    _logical_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Top-k region selection fused with prefill metadata packing.

    Args:
        logits: ``[rows, seq_regions]`` float32 indexer scores, which may be signed, where
            ``rows = groups * seq_q`` in group-major order.
        lengths: ``[rows]`` int32, causally visible region count per row.
        q_positions: ``[rows]`` int32, absolute token position of each row's query.
        block_table: ``[num_reqs, pages]`` int32, logical page -> physical page, one row
            per request. ``-1`` marks an unmapped page. A table with a single row is
            treated as shared by every request, regardless of ``request_indices``.
        request_indices: ``[rows]`` int32, which block-table row each score row belongs
            to. Defaults to all-zero.

    Returns:
        ``[rows, topk]`` int32, each valid slot holding
        ``phys_region | (valid_tokens << 24)``, with ``-1`` padding the tail. A row fills
        exactly ``min(topk, visible)`` slots.
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )

    rows, seq_regions = logits.shape
    if request_indices is None:
        request_indices = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    out = torch.empty((rows, int(topk)), device=logits.device, dtype=torch.int32)
    _region_topk_pack_kernel[(rows,)](
        logits,
        lengths,
        q_positions,
        block_table,
        request_indices,
        out,
        out if _logical_out is None else _logical_out,
        seq_regions,
        logits.stride(0),
        shared_table_stride(block_table),
        out.stride(0),
        out.stride(0) if _logical_out is None else _logical_out.stride(0),
        topk=int(topk),
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        WRITE_LOGICAL_IDS=_logical_out is not None,
        BLOCK_R=min(block_r, triton.next_power_of_2(seq_regions)),
        BLOCK_TOPK=triton.next_power_of_2(int(topk)),
    )
    return (out, _logical_out) if _logical_out is not None else out
