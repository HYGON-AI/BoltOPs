# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Stable Wave64 MoE routing using Gluon 3.6.

Inputs and outputs
------------------
T = tokens, K = topk (not GEMM K), E = experts, U = unit_size, R = T*K routes.
Inputs are contiguous INT32 topk_ids[T, K] and FP32 topk_weights[T, K] on GPU.
Each token selects K distinct expert IDs in [0, E); T < 2**24, 0 < K < 256.
An optional local_expert_mask[E] enables experts with nonzero entries and maps
their global IDs to compact local IDs in enabled-expert order.

moe_sorting_fwd fills five caller-allocated outputs, returning no new tensors:
  sorted_token_ids[C]              INT32, token | (topk_slot << 24)
  sorted_weights[C]                FP32, original weight bits; zero for padding
  sorted_expert_ids[ceil(C/U)]     INT32, one expert ID per U consecutive routes
  tokens_positions_per_expert[2*E] INT32, E counts followed by E start offsets
  num_valid_ids[1]                 INT32, enabled experts' total padded length
Only the first num_valid_ids routes and their expert tiles are defined. Counts
include disabled experts; empty/disabled experts have start offset -1. A
sufficient route capacity is C = R + E*(U-1). The public MoE wrapper returns
(IDs, weights, experts, valid count, positions).

Example: T=3, K=2, E=3, U=2, no mask; (t,s) denotes token and topk slot.

    topk_ids        topk_weights       Flattened in (token, slot) order
    [2, 0]          [a, b]             (0,0):e2/a  (0,1):e0/b
    [0, 2]          [c, d]      -->    (1,0):e0/c  (1,1):e2/d
    [2, 0]          [e, f]             (2,0):e2/e  (2,1):e0/f
                                               |
                              Stable grouping by expert; pad each to U
                                               v
    expert 0:   (0,1)  (1,0)  (2,1)  PAD     weights: b c f 0
    expert 1:   empty
    expert 2:   (0,0)  (1,1)  (2,0)  PAD     weights: a d e 0

    experts   = [0, 0, 2, 2], one entry per U=2 consecutive routes
    positions = [3, 0, 3 | 0, -1, 4], valid = 8

PAD is packed as T | (K << 24), an invalid token/slot pair. With mask [1,0,1],
the same routes are emitted but expert tiles become [0,0,1,1] after local-ID
mapping. Disabled experts contribute no output or padding.

Small inputs: R <= 2048
----------------------
One CTA per expert, using a histogram and a stable prefix scan:

    All IDs --> histogram --> mask / round counts to U --> expert start
       |
       +--> select this expert's routes --> prefix scan --> stable rank
                                                               |
                          output offset = expert start + rank <-+

Each CTA writes its expert's routes, weights, padding and metadata. CTA 0 also
writes the total padded length. Buffer load/store reduces address calculation
when all byte offsets fit its 32-bit range; larger ranges use global I/O.

Large inputs: R > 2048
---------------------
Split the input into NB ordered chunks, each using four waves.
B = max(256, next_power_of_2(ceil(R/64))), NB = ceil(R/B) <= 64.

    Pass 1: _chunk_counts, one CTA per chunk

        chunk 0 --> histogram --+       Parts[NB, E]
        chunk 1 --> histogram --+-----> complete count table
           ...                  |
        chunk c --> histogram --+

    ================= same-stream kernel boundary =================

    Pass 2: _large_sort

        Parts[:, e] --> sum all chunks --> count[e]
                                           |
                              mask / round to U --> expert prefix
        Parts[:c, e] --> sum earlier chunks ------> chunk prefix

        chunk c --> bitonic sort(expert*B + original position)
                    --> segmented scan by expert --> stable chunk rank

        output offset = expert prefix + chunk prefix + stable chunk rank
                        |               |              |
                        |               |              +-- within chunk c
                        |               +-- same expert in earlier chunks
                        +-- padded lengths of earlier enabled experts

Sorting keys include original positions; chunk prefixes follow input order.
Together they preserve route order within each expert. The first ceil(E/16)
CTAs also write expert tiles, padding, counts, offsets and total padded length.

Pass 1 overwrites every count in invocation-local torch.empty scratch. Stream
order makes the complete table visible to pass 2, including on HIP Graph
replay. No generation tags, count-table cache, global atomics, cross-CTA polling
or speculative recounting are needed. Weights are copied without arithmetic.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import (
    buffer_load,
    buffer_store,
)


@g.jit
def _add(a, b):
    return a + b


@g.jit
def _load_vector(Ptr, offsets, mask, other, BUFFER: gl.constexpr):
    if BUFFER:
        return buffer_load(Ptr, offsets, mask, other)
    return gl.load(Ptr + offsets, mask, other)


@g.jit
def _store_vector(Ptr, offsets, value, mask, BUFFER: gl.constexpr):
    if BUFFER:
        values, _ = gl.broadcast(gl.cast(value, Ptr.dtype.element_ty), offsets)
        buffer_store(values, Ptr, offsets, mask)
    else:
        gl.store(Ptr + offsets, value, mask)


@g.jit(
    do_not_specialize=[
        "Ids",
        "Weights",
        "Sorted",
        "SortedW",
        "Experts",
        "Positions",
        "Valid",
        "Mask",
    ]
)
def _small_sort(
    Ids,
    Weights,
    Sorted,
    SortedW,
    Experts,
    Positions,
    Valid,
    Mask,
    T: gl.constexpr,
    K: gl.constexpr,
    E: gl.constexpr,
    UNIT: gl.constexpr,
    BE: gl.constexpr,
    BR: gl.constexpr,
    HAS_MASK: gl.constexpr,
    BUFFER: gl.constexpr = False,
):
    expert = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    if expert < E:
        r = gl.arange(0, BR, layout)
        ids = _load_vector(Ids, r, r < T * K, 0, BUFFER)
        counts = gl.histogram(ids, BE, mask=r < T * K, layout=layout)
        es = gl.arange(0, BE, layout)
        enabled = es < E
        if HAS_MASK:
            enabled = enabled & (_load_vector(Mask, es, es < E, 0, BUFFER) != 0)
        padded = gl.where(enabled, gl.cdiv(counts, UNIT) * UNIT, 0)
        start = gl.sum(gl.where(es < expert, padded, 0), 0)
        count = gl.sum(gl.where(es == expert, counts, 0), 0)
        size = gl.sum(gl.where(es == expert, padded, 0), 0)
        local = expert
        if HAS_MASK:
            local = gl.sum((enabled & (es < expert)).to(gl.int32), 0)
        gl.store(Positions + expert, count)
        gl.store(Positions + E + expert, gl.where(size > 0, start, -1))
        if expert == 0:
            gl.store(Valid, gl.sum(padded, 0))
        if size > 0:
            selected = (r < T * K) & (ids == expert)
            rank = gl.associative_scan(selected.to(gl.int32), 0, _add) - 1
            weight = _load_vector(Weights, r, selected, 0, BUFFER)
            _store_vector(
                Sorted,
                start + rank,
                r // K | ((r % K) << 24),
                selected,
                BUFFER,
            )
            _store_vector(SortedW, start + rank, weight, selected, BUFFER)
            pad = gl.arange(0, triton.next_power_of_2(UNIT), layout)
            _store_vector(
                Sorted,
                start + count + pad,
                T | (K << 24),
                count + pad < size,
                BUFFER,
            )
            _store_vector(
                SortedW,
                start + count + pad,
                0.0,
                count + pad < size,
                BUFFER,
            )
            block = gl.arange(
                0, triton.next_power_of_2(max(triton.cdiv(T, UNIT), 1)), layout
            )
            _store_vector(
                Experts,
                start // UNIT + block,
                local,
                block < size // UNIT,
                BUFFER,
            )


@g.jit
def _segment_add(ka, va, kb, vb):
    return kb, gl.where(ka == kb, va + vb, vb)


@g.jit
def _sort_chunk(ids, x, B: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    key = ids * B + x
    for stage in gl.static_range(1, triton.next_power_of_2(B).bit_length()):
        for step in gl.static_range(stage - 1, -1, -1):
            # Each 64-element row belongs to a wave. Keep lane exchanges
            # within that row; larger exchanges use the whole chunk.
            if step < 6:
                neighbor = gl.gather(
                    key.reshape((B // 64, 64)),
                    (x.reshape((B // 64, 64)) % 64) ^ (1 << step),
                    1,
                ).reshape((B,))
            else:
                neighbor = gl.gather(key, x ^ (1 << step), 0)
            low = ((x & (1 << stage)) == 0) == ((x & (1 << step)) == 0)
            key = gl.where(low, gl.minimum(key, neighbor), gl.maximum(key, neighbor))
    sid = key // B
    _, rank = gl.associative_scan(
        (sid, gl.full((B,), 1, gl.int32, layout)), 0, _segment_add
    )
    return key, rank - 1


@g.jit(do_not_specialize=["Ids", "Parts"])
def _chunk_counts(
    Ids,
    Parts,
    R: gl.constexpr,
    E: gl.constexpr,
    BE: gl.constexpr,
    B: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    r = pid * B + gl.arange(0, B, layout)
    ids = gl.load(Ids + r, r < R, other=E)
    counts = gl.histogram(ids, BE, mask=r < R, layout=layout)
    e = gl.arange(0, BE, layout)
    gl.store(Parts + pid * E + e, counts, e < E)


@g.jit
def _ready_counts(Parts, pid, E: gl.constexpr, BE: gl.constexpr, NB: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    tile: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])
    c0 = gl.arange(0, 8, gl.SliceLayout(1, tile))
    e = gl.arange(0, BE, gl.SliceLayout(0, tile))
    counts = gl.full((BE,), 0, gl.int32, layout)
    prefix = gl.full((BE,), 0, gl.int32, layout)
    for row in range(triton.cdiv(NB, 8)):
        c = row * 8 + c0
        hist = gl.load(
            Parts + c[:, None] * E + e[None, :],
            (c[:, None] < NB) & (e[None, :] < E),
            other=0,
        )
        counts += gl.convert_layout(gl.sum(hist, 0), layout)
        prefix += gl.convert_layout(
            gl.sum(gl.where(c[:, None] < pid, hist, 0), 0),
            layout,
        )
    return counts, prefix


@g.jit(
    do_not_specialize=[
        "Ids",
        "Weights",
        "Parts",
        "Sorted",
        "SortedW",
        "Experts",
        "Positions",
        "Valid",
        "Mask",
    ]
)
def _large_sort(
    Ids,
    Weights,
    Parts,
    Sorted,
    SortedW,
    Experts,
    Positions,
    Valid,
    Mask,
    T: gl.constexpr,
    K: gl.constexpr,
    E: gl.constexpr,
    U: gl.constexpr,
    BE: gl.constexpr,
    B: gl.constexpr,
    MASK: gl.constexpr,
):
    NB: gl.constexpr = triton.cdiv(T * K, B)
    WORKERS: gl.constexpr = max(NB, triton.cdiv(E, 16))
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    if pid < WORKERS:
        x = gl.arange(0, B, layout)
        key = gl.full((B,), 0, gl.int32, layout)
        rank = gl.full((B,), 0, gl.int32, layout)
        if pid < NB:
            r = pid * B + x
            ids = gl.load(Ids + r, r < T * K, other=E)
            key, rank = _sort_chunk(ids, x, B)
        counts, prefix = _ready_counts(Parts, pid, E, BE, NB)
        es = gl.arange(0, BE, layout)
        enabled = es < E
        if MASK:
            enabled = enabled & (gl.load(Mask + es, es < E, other=0) != 0)
        sizes = gl.where(enabled, gl.cdiv(counts, U) * U, 0)
        starts = gl.associative_scan(sizes, 0, _add) - sizes
        if pid < NB:
            route = pid * B + key % B
            expert = gl.minimum(key // B, E - 1)
            real = route < T * K
            selected = real & gl.gather(enabled, expert, 0)
            base = gl.gather(starts + prefix, expert, 0)
            weight = gl.load(Weights + route, selected, other=0)
            gl.store(Sorted + base + rank, route // K | ((route % K) << 24), selected)
            gl.store(SortedW + base + rank, weight, selected)
        if pid < triton.cdiv(E, 16):
            eh = pid * 16 + gl.arange(0, 16, layout)
            idx = gl.minimum(eh, E - 1)
            begin = gl.gather(starts, idx, 0)
            length = gl.gather(counts, idx, 0)
            padded = gl.gather(sizes, idx, 0)
            gl.store(Positions + eh, length, eh < E)
            gl.store(Positions + E + eh, gl.where(padded > 0, begin, -1), eh < E)
            if pid == 0:
                gl.store(Valid, gl.sum(sizes, 0))
            local = eh
            if MASK:
                mapped = gl.associative_scan(enabled.to(gl.int32), 0, _add) - 1
                local = gl.gather(mapped, idx, 0)
            tile: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
            rows = gl.convert_layout(eh, gl.SliceLayout(1, tile))
            begin = gl.convert_layout(begin, gl.SliceLayout(1, tile))
            length = gl.convert_layout(length, gl.SliceLayout(1, tile))
            padded = gl.convert_layout(padded, gl.SliceLayout(1, tile))
            local = gl.convert_layout(local, gl.SliceLayout(1, tile))
            pad = gl.arange(0, triton.next_power_of_2(U), gl.SliceLayout(0, tile))
            pad_mask = (rows[:, None] < E) & (
                length[:, None] + pad[None, :] < padded[:, None]
            )
            gl.store(
                Sorted + begin[:, None] + length[:, None] + pad[None, :],
                T | (K << 24),
                pad_mask,
            )
            gl.store(
                SortedW + begin[:, None] + length[:, None] + pad[None, :], 0.0, pad_mask
            )
            blocks = gl.arange(
                0,
                triton.next_power_of_2(max(triton.cdiv(T, U), 1)),
                gl.SliceLayout(0, tile),
            )
            gl.store(
                Experts + begin[:, None] // U + blocks[None, :],
                local[:, None] + 0 * blocks[None, :],
                (rows[:, None] < E) & (blocks[None, :] < padded[:, None] // U),
            )


def moe_sorting_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens_positions_per_expert: torch.Tensor,
    num_valid_ids: torch.Tensor,
    num_experts: int,
    unit_size: int,
    local_expert_mask: torch.Tensor | None = None,
) -> None:
    """Stable routing; token experts must be distinct and in [0, num_experts).

    Only the valid output prefix is defined. Counts include every global expert;
    empty/disabled experts have start=-1. Inputs are contiguous int32/float32.
    """
    tokens, topk = topk_ids.shape
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        raise ValueError("sorting requires int32 IDs and FP32 weights")
    if (
        topk_weights.shape != topk_ids.shape
        or not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
    ):
        raise ValueError(
            "IDs and weights must have the same contiguous [tokens, topk] shape"
        )
    if num_experts <= 0 or unit_size <= 0 or tokens >= (1 << 24) or not 0 < topk < 256:
        raise ValueError("invalid experts, unit_size or packed token/slot dimensions")

    def dispatch():
        experts = 1 << (num_experts - 1).bit_length()
        if tokens * topk <= 2048:
            routes = 1 << (max(tokens * topk, 1) - 1).bit_length()
            warps = (1 if num_experts > 64 else 2) if tokens * topk <= 256 else 4
            _small_sort[(num_experts,)](
                topk_ids,
                topk_weights,
                sorted_token_ids,
                sorted_weights,
                sorted_expert_ids,
                tokens_positions_per_expert,
                num_valid_ids,
                local_expert_mask,
                tokens,
                topk,
                num_experts,
                unit_size,
                experts,
                routes,
                local_expert_mask is not None,
                4 * (tokens * topk + num_experts * (unit_size - 1)) <= (1 << 32) - 8,
                num_warps=warps,
            )
            return

        block = max(256, 1 << (((tokens * topk + 63) // 64) - 1).bit_length())
        chunks = (tokens * topk + block - 1) // block
        # Stream order separates publication from consumption. Every count is
        # overwritten, so neither clearing nor generations/atomic snapshots
        # are needed. Each invocation owns its workspace, including in graphs.
        parts = torch.empty(
            chunks * num_experts, device=topk_ids.device, dtype=torch.int32
        )
        _chunk_counts[(chunks,)](
            topk_ids,
            parts,
            tokens * topk,
            num_experts,
            experts,
            block,
        )
        workers = max(chunks, (num_experts + 15) // 16)
        _large_sort[(workers,)](
            topk_ids,
            topk_weights,
            parts,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            tokens_positions_per_expert,
            num_valid_ids,
            local_expert_mask,
            tokens,
            topk,
            num_experts,
            unit_size,
            experts,
            block,
            local_expert_mask is not None,
        )

    if topk_ids.device.index == torch.cuda.current_device():
        dispatch()
    else:
        with torch.cuda.device(topk_ids.device):
            dispatch()
