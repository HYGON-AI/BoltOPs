"""Sparse GQA prefill attention over selected regions (scalar and query-tiled).

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``sparse_attn`` prefill).
Per-query ``num_splits is None`` chooses occupancy from device CU count; split
windows are disjoint (``slots < window_end``). Tiled prefill is unchanged.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import (
    _device_name_for_config,
    _dtype_tag,
    _fit_block_regions,
)

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)
DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))


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

    ``program_id(2)`` partitions the packed row into disjoint slot windows;
    ``slots < window_end`` keeps adjacent splits from overlapping when
    ``slots_per_split % BLOCK_R != 0``. ``num_splits=1`` is the full packed row.
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
    for base in tl.range(window_begin, window_end, BLOCK_R):
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
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
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
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    softmax_scale,
    total_q,
    num_tiles,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Query-tiled sparse prefill over a per-tile union of selected regions.

    The selector produces one region list per query, but neighbouring queries usually
    select nearly the same regions. This kernel loads each union region once per query
    tile and uses a bitset to decide which query owns each token. Tiles are never allowed
    to cross a request boundary by :func:`_build_prefill_tiles`, so a logical region id
    maps to one physical cache region within a tile.
    """
    tile = tl.program_id(0)
    group = tl.program_id(1)
    head = tl.program_id(2)

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
    union_count = tl.load(lengths_ptr + tile).to(tl.int32)

    for base in tl.range(0, union_count, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < union_count
        physical = tl.load(
            list_ptr + tile * stride_list_tile + slots * stride_list_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        logical = tl.load(
            logical_list_ptr + tile * stride_logical_tile + slots * stride_logical_slot,
            mask=slot_ok,
            other=0,
        ).to(tl.int32)
        words = tl.load(
            mask_ptr
            + tile * stride_mask_tile
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
        keys = tl.load(
            k_ptr
            + safe_tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_live[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_live[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(keep, scores, NEGATIVE_INFINITY)
        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))
        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected_any = running_sum > 0
    output = tl.where(
        selected_any[:, None], accumulator / running_sum[:, None], 0.0
    )
    tl.store(
        out_ptr
        + q_index[:, None] * stride_out_token
        + (group * heads_per_group + head) * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=q_ok[:, None] & dim_ok[None, :],
    )
    tl.store(
        lse_ptr + q_index * stride_lse_token + group * heads_per_group + head,
        tl.where(selected_any, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=q_ok,
    )


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

    A tile is cut at every request boundary before it is split into ``block_q`` queries.
    That invariant is important: a logical region number is meaningful only within one
    request, while the physical cache region is request-specific. The returned tensors are
    all on the input device and can be passed directly to the Triton kernel.
    """
    rows, slots = logical_regions.shape
    expected_rows = num_kv_groups * total_q
    if packed_regions.shape != logical_regions.shape:
        raise ValueError(
            "logical and packed prefill metadata must have the same shape, got "
            f"{tuple(logical_regions.shape)} and {tuple(packed_regions.shape)}"
        )
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} metadata rows, got {rows}")
    if q_positions.numel() == total_q:
        q_positions = q_positions.repeat(num_kv_groups)
    if q_positions.numel() != rows:
        raise ValueError(f"expected {rows} query positions, got {q_positions.numel()}")
    if request_indices.numel() != rows:
        raise ValueError(f"expected {rows} request indices, got {request_indices.numel()}")
    if block_q <= 0 or block_q & (block_q - 1):
        raise ValueError(f"block_q must be a positive power of two, got {block_q}")

    # vLLM lays prefill rows out contiguously by request. Read only this small vector on
    # the host to make tile boundaries; all large metadata construction stays on device.
    request_cpu = request_indices.to(device="cpu", dtype=torch.int64).tolist()
    q_starts: list[int] = []
    q_counts: list[int] = []
    for group in range(num_kv_groups):
        group_start = group * total_q
        group_end = group_start + total_q
        run_start = group_start
        while run_start < group_end:
            request = request_cpu[run_start]
            run_end = run_start + 1
            while run_end < group_end and request_cpu[run_end] == request:
                run_end += 1
            for start in range(run_start, run_end, block_q):
                q_starts.append(start)
                q_counts.append(min(block_q, run_end - start))
            run_start = run_end
    if not q_starts:
        raise ValueError("cannot build prefill tiles for an empty query batch")

    device = logical_regions.device
    num_tiles = len(q_starts)
    q_starts_d = torch.tensor(q_starts, device=device, dtype=torch.int32)
    q_counts_d = torch.tensor(q_counts, device=device, dtype=torch.int32)
    tile_ids = torch.arange(num_tiles, device=device, dtype=torch.int64)
    tile_of_row = torch.repeat_interleave(tile_ids, q_counts_d.to(torch.int64))
    if tile_of_row.numel() != rows:
        raise ValueError("request boundaries do not cover all prefill rows")

    live = (logical_regions >= 0) & (packed_regions >= 0)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    tile_of = tile_of_row[:, None].expand(rows, slots)[live]
    q_local = (
        row_ids - q_starts_d[tile_of_row].to(torch.int64)
    )[:, None].expand(rows, slots)[live]
    logical = logical_regions[live].to(torch.int64)
    physical = (packed_regions[live].to(torch.int64) & ((1 << 24) - 1)).to(torch.int32)

    num_regions = int(logical.max().item()) + 1 if logical.numel() else 1
    present = torch.zeros((num_tiles, num_regions), device=device, dtype=torch.bool)
    present[tile_of, logical] = True
    lengths = present.sum(dim=1).to(torch.int32)
    union = max(1, int(lengths.max().item()))

    # Mapping through logical ids is safe because a tile never crosses a request. It also
    # keeps the temporary bitmap proportional to sequence length rather than to the
    # allocator's (potentially very sparse) physical block ids.
    physical_by_logical = torch.zeros(
        (num_tiles, num_regions), device=device, dtype=torch.int32
    )
    physical_by_logical[tile_of, logical] = physical
    tile, logical_slot = present.nonzero(as_tuple=True)
    slot_of = present.cumsum(dim=1).to(torch.int64) - 1
    tile_phys = torch.zeros((num_tiles, union), device=device, dtype=torch.int32)
    tile_logical = torch.zeros((num_tiles, union), device=device, dtype=torch.int32)
    tile_phys[tile, slot_of[tile, logical_slot]] = physical_by_logical[
        tile, logical_slot
    ]
    tile_logical[tile, slot_of[tile, logical_slot]] = logical_slot.to(torch.int32)

    words = (block_q + 31) // 32
    mask64 = torch.zeros((num_tiles, words, union), device=device, dtype=torch.int64)
    entry_slots = slot_of[tile_of, logical]
    mask64.index_put_(
        (tile_of, q_local // 32, entry_slots),
        torch.ones_like(q_local, dtype=torch.int64) << (q_local % 32),
        accumulate=True,
    )
    mask = (mask64 - (mask64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
    return q_starts_d, q_counts_d, tile_phys, tile_logical, lengths, mask


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "sparse_attn_prefill",
)
_LAUNCH_KEYS = frozenset(
    {
        "BLOCK_R",
        "BLOCK_H",
        "BLOCK_D",
        "num_warps",
        "num_stages",
        "waves_per_eu",
    }
)
_GRID_KEYS = ("num_warps", "num_stages", "waves_per_eu")
_TILED_LAUNCH_KEYS = frozenset(
    {
        "BLOCK_R",
        "BLOCK_Q",
        "BLOCK_D",
        "num_warps",
        "num_stages",
        "waves_per_eu",
    }
)


def sparse_attn_prefill_config_filename(
    packed_width: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype: torch.dtype | str,
    device: str | None = None,
) -> str:
    """Return ``prefill-k{width}-hpg{hpg}-d{dim}-r{region}_{dtype}_{device}.json``."""
    dtype_tag = dtype if isinstance(dtype, str) else _dtype_tag(dtype)
    device_tag = device if device is not None else _device_name_for_config()
    return (
        f"prefill-k{int(packed_width)}-hpg{int(heads_per_group)}"
        f"-d{int(head_dim)}-r{int(region_size)}_{dtype_tag}_{device_tag}.json"
    )


def sparse_attn_prefill_config_path(
    packed_width: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype: torch.dtype | str,
    device: str | None = None,
) -> str:
    return os.path.join(
        _CONFIG_DIR,
        sparse_attn_prefill_config_filename(
            packed_width, heads_per_group, head_dim, region_size, dtype, device
        ),
    )


def _filter_launch_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key in _LAUNCH_KEYS}


@functools.lru_cache
def _load_sparse_attn_prefill_config(
    packed_width: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype_tag: str,
    device: str,
) -> dict[str, Any] | None:
    path = sparse_attn_prefill_config_path(
        packed_width, heads_per_group, head_dim, region_size, dtype_tag, device
    )
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    filtered = _filter_launch_config(payload)
    if "BLOCK_R" not in filtered:
        return None
    return filtered


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


def sparse_attn_prefill_tiled_union_bucket(union: int) -> int:
    """Bucket a tile-union width with ``next_power_of_2`` for config filenames."""
    return int(triton.next_power_of_2(max(1, int(union))))


def sparse_attn_prefill_tiled_config_filename(
    packed_width: int,
    union_bucket: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    block_q: int,
    dtype: torch.dtype | str,
    device: str | None = None,
) -> str:
    """Return ``prefill-tiled-k{width}-u{union}-hpg{hpg}-d{dim}-r{region}-q{bq}_{dtype}_{device}.json``."""
    dtype_tag = dtype if isinstance(dtype, str) else _dtype_tag(dtype)
    device_tag = device if device is not None else _device_name_for_config()
    return (
        f"prefill-tiled-k{int(packed_width)}-u{int(union_bucket)}"
        f"-hpg{int(heads_per_group)}-d{int(head_dim)}-r{int(region_size)}"
        f"-q{int(block_q)}_{dtype_tag}_{device_tag}.json"
    )


def sparse_attn_prefill_tiled_config_path(
    packed_width: int,
    union_bucket: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    block_q: int,
    dtype: torch.dtype | str,
    device: str | None = None,
) -> str:
    return os.path.join(
        _CONFIG_DIR,
        sparse_attn_prefill_tiled_config_filename(
            packed_width,
            union_bucket,
            heads_per_group,
            head_dim,
            region_size,
            block_q,
            dtype,
            device,
        ),
    )


def _filter_tiled_launch_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key in _TILED_LAUNCH_KEYS}


@functools.lru_cache
def _load_sparse_attn_prefill_tiled_config(
    packed_width: int,
    union_bucket: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    block_q: int,
    dtype_tag: str,
    device: str,
) -> dict[str, Any] | None:
    path = sparse_attn_prefill_tiled_config_path(
        packed_width,
        union_bucket,
        heads_per_group,
        head_dim,
        region_size,
        block_q,
        dtype_tag,
        device,
    )
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    filtered = _filter_tiled_launch_config(payload)
    if "BLOCK_R" not in filtered:
        return None
    return filtered


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the query-tiled kernel on already-built union lists.

    Callers that need end-to-end serving cost should use
    :func:`sparse_attention_prefill`, which rebuilds tiles on every call.
    Bench / xprof / tuner pass prebuilt tiles so CUDA-event timing matches
    the Triton kernel in isolation.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    heads_per_group = num_heads // num_kv_groups
    if out is None:
        out = torch.empty_like(query)
    if lse is None:
        lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    num_tiles = int(q_starts.numel())
    union = int(tile_phys.shape[1])
    launch = config
    if launch is None:
        launch = _load_sparse_attn_prefill_tiled_config(
            packed_width,
            sparse_attn_prefill_tiled_union_bucket(union),
            heads_per_group,
            head_dim,
            region_size,
            block_q,
            _dtype_tag(query.dtype),
            _device_name_for_config(),
        )
    default_r = _fit_block_regions(
        min(block_regions, triton.next_power_of_2(union)),
        region_size=region_size,
        block_d=block_d,
        element_size=key_cache.element_size(),
    )
    extra: dict[str, Any] = {"num_warps": 4, "num_stages": 2}
    if launch:
        block_d = int(launch.get("BLOCK_D", block_d))
        block_r = int(launch["BLOCK_R"])
        extra = {key: launch[key] for key in _GRID_KEYS if key in launch}
    else:
        block_r = default_r
    _sparse_attn_prefill_tiled_kernel[
        (num_tiles, num_kv_groups, heads_per_group)
    ](
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
        out,
        lse,
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
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        softmax_scale,
        total_q,
        num_tiles,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        BLOCK_Q=block_q,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        **extra,
    )
    return out, lse


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
    num_splits: int | None = None,
    kv_split_target_programs: int | None = None,
    min_slots_per_split: int = 8,
    return_partials: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over per-query selected regions.

    Args:
        query: ``[total_q, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, i.e. the paged
            cache already flattened across pages. Flattening is safe because a region's
            tokens are contiguous in the physical token space by construction.
        packed_regions: ``[num_kv_groups * total_q, topk]`` int32 from the selector,
            each slot ``phys_region | (valid_tokens << 24)``, ``-1`` padding the tail.
        region_counts: ``[num_kv_groups * total_q]`` int32, how many slots to read. This
            truncates *positionally* -- it does not re-select by score.
        logical_regions: optional logical ids parallel to ``packed_regions``. When supplied,
            the query-tiled implementation is used and ``q_positions`` is required.
        q_positions: absolute positions parallel to the prefill query rows (or one vector
            of length ``total_q`` shared by all KV groups).
        request_indices: request id parallel to metadata rows; tiles are split at changes
            in this vector so per-request logical ids never alias.
        block_q: query tile width for the tiled implementation; must be a power of two.
        config: launch override. ``None`` loads a path-specific JSON. Per-query uses
            ``prefill-k{width}-hpg{hpg}-d{dim}-r{region}_{dtype}_{device}.json``. Tiled
            uses ``prefill-tiled-k{width}-u{union}-hpg{hpg}-d{dim}-r{region}-q{bq}_{dtype}_{device}.json``
            after tiles are built (``u`` is ``next_power_of_2`` of the union width). A
            mapping is used as-is (no shared-memory shrink). An empty mapping skips
            lookup and keeps the built-in defaults.
        num_splits: ``None`` picks a split count from device CU count and ``Q * G``.
            An explicit value is used as-is. With more than one split the default is
            to merge to ``[total_q, heads, dim]``; pass ``return_partials=True``
            to keep the leading split axis for :func:`merge_split_states`. Tiled
            prefill ignores this argument.

    Returns:
        Merged ``(out [total_q, num_heads, head_dim], lse [total_q, num_heads])``
        unless ``return_partials`` is set, in which case both have a leading split
        axis. Rows with no selected region get a zero output and ``lse = -inf``.

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

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    if logical_regions is not None:
        if q_positions is None:
            raise ValueError("q_positions is required for tiled sparse prefill")
        if request_indices is None:
            request_indices = torch.zeros(
                (num_kv_groups * total_q,), device=query.device, dtype=torch.int32
            )
        q_positions = q_positions.to(device=query.device, dtype=torch.int32).contiguous()
        request_indices = request_indices.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        if q_positions.numel() == total_q:
            q_positions = q_positions.repeat(num_kv_groups)
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
        )

    from boltops.step4_dsa.triton.merge import merge_split_states

    if num_splits is None:
        num_splits = choose_prefill_kv_splits(
            total_q,
            num_kv_groups,
            topk,
            target_programs=kv_split_target_programs,
            min_slots_per_split=min_slots_per_split,
            device=query.device,
        )
    num_splits = max(int(num_splits), 1)
    slots_per_split = -(-topk // num_splits)

    out = torch.empty(
        (num_splits, total_q, num_heads, head_dim), device=query.device, dtype=query.dtype
    )
    lse = torch.empty((num_splits, total_q, num_heads), device=query.device, dtype=torch.float32)
    launch = config
    if launch is None:
        launch = _load_sparse_attn_prefill_config(
            topk,
            heads_per_group,
            head_dim,
            region_size,
            _dtype_tag(query.dtype),
            _device_name_for_config(),
        )
    default_h, default_d, default_r = _per_query_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        topk=slots_per_split,
        block_regions=block_regions,
        region_size=region_size,
        element_size=key_cache.element_size(),
    )
    extra: dict[str, Any] = {}
    if launch:
        block_h = int(launch.get("BLOCK_H", default_h))
        block_d = int(launch.get("BLOCK_D", default_d))
        block_r = int(launch["BLOCK_R"])
        extra = {
            key: launch[key]
            for key in _GRID_KEYS
            if key in launch and key != "waves_per_eu"
        }
    else:
        block_h, block_d, block_r = default_h, default_d, default_r

    _sparse_attn_prefill_kernel[(total_q, num_kv_groups, num_splits)](
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
        total_q,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        **extra,
    )
    if num_splits == 1:
        return out[0], lse[0]
    if return_partials:
        return out, lse
    return merge_split_states(out, lse)


def choose_prefill_kv_splits(
    total_q: int,
    num_kv_groups: int,
    packed_width: int,
    *,
    target_programs: int | None = None,
    min_slots_per_split: int = 8,
    device: torch.device | None = None,
) -> int:
    """Pick a packed-slot split count from CU occupancy, not an evidence-shape test.

    Underfills (``Q * G < CUs``) use ``ceil(CUs / (Q*G))`` like decode. When the
    per-query grid already covers the device, keep one split: extra waves plus
    ``merge_split_states`` did not beat the serial packed-row loop on a full
    grid. Callers may still pass ``target_programs`` to request extra waves.
    Cap so a full row keeps ``min_slots_per_split`` slots per window.
    """
    packed_width = max(int(packed_width), 1)
    programs_base = max(int(total_q) * int(num_kv_groups), 1)
    if target_programs is None:
        try:
            index = (
                device.index
                if device is not None and device.index is not None
                else torch.cuda.current_device()
            )
            target_programs = int(torch.cuda.get_device_properties(index).multi_processor_count)
        except Exception:
            target_programs = 1
    target_programs = max(int(target_programs or 1), 1)
    splits = -(-target_programs // programs_base)
    max_splits = max(1, packed_width // max(int(min_slots_per_split), 1))
    return min(max(splits, 1), max_splits, packed_width)


# Previous landing name; same function.
sparse_attention_prefill_optimized = sparse_attention_prefill
