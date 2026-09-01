"""Sparse GQA decode attention over selected regions, with optional split-KV.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``sparse_attn`` decode).
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

DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))

@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
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
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """One decode query against one KV group's selected regions, over one slot window.

    Structurally the same online softmax as prefill, but the metadata contract differs in
    the one way that matters: decode packs ``start_token | (phys_region << 32)`` in int64
    and does **not** carry ``valid_tokens``. The visible length is recomputed here as
    ``clamp(kv_seqlen - start_token, 0, region_size)``, which is why the logical start
    token has to be in the word at all -- the physical region alone cannot say how much of
    the last region exists yet.

    That also disposes of both padding sentinels without a special case. ``-1`` and
    ``0x7FFF_FFFF_FFFF_FFFF`` both leave ``0xFFFF_FFFF`` in the low word, so
    ``kv_seqlen - start_token`` is hugely negative and the slot clamps to zero tokens.
    """
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + request * stride_q_token
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
    for base in tl.range(window_begin, window_begin + slots_per_split, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        # ``physical`` is a sentinel-sized integer on dead slots, so it must be
        # neutralised before it becomes an address -- masking the load is not enough,
        # since the pointer arithmetic itself would overflow.
        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        keys = tl.load(
            k_ptr + tokens[:, None] * stride_k_token + group * stride_k_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr + tokens[:, None] * stride_v_token + group * stride_v_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

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

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "sparse_attn_decode",
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


def sparse_attn_decode_config_filename(
    packed_width: int,
    num_splits: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype: torch.dtype | str,
    device: str | None = None,
    *,
    batch: int | None = None,
    context: int | None = None,
) -> str:
    """Return the launch JSON name.

    With both ``batch`` and ``context``:
    ``decode-k{width}-s{splits}-b{batch}-ctx{context}-hpg{hpg}-d{dim}-r{region}_{dtype}_{device}.json``.
    Otherwise the fallback name without ``b`` / ``ctx``.
    """
    dtype_tag = dtype if isinstance(dtype, str) else _dtype_tag(dtype)
    device_tag = device if device is not None else _device_name_for_config()
    occupancy = ""
    if batch is not None and context is not None:
        occupancy = f"-b{int(batch)}-ctx{int(context)}"
    return (
        f"decode-k{int(packed_width)}-s{int(num_splits)}{occupancy}"
        f"-hpg{int(heads_per_group)}-d{int(head_dim)}-r{int(region_size)}"
        f"_{dtype_tag}_{device_tag}.json"
    )


def sparse_attn_decode_config_path(
    packed_width: int,
    num_splits: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype: torch.dtype | str,
    device: str | None = None,
    *,
    batch: int | None = None,
    context: int | None = None,
) -> str:
    return os.path.join(
        _CONFIG_DIR,
        sparse_attn_decode_config_filename(
            packed_width,
            num_splits,
            heads_per_group,
            head_dim,
            region_size,
            dtype,
            device,
            batch=batch,
            context=context,
        ),
    )


def _filter_launch_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key in _LAUNCH_KEYS}


def _read_launch_config(path: str) -> dict[str, Any] | None:
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


@functools.lru_cache
def _load_sparse_attn_decode_config(
    packed_width: int,
    num_splits: int,
    heads_per_group: int,
    head_dim: int,
    region_size: int,
    dtype_tag: str,
    device: str,
    batch: int | None,
    context: int | None,
) -> dict[str, Any] | None:
    if batch is None or context is None:
        return None
    return _read_launch_config(
        sparse_attn_decode_config_path(
            packed_width,
            num_splits,
            heads_per_group,
            head_dim,
            region_size,
            dtype_tag,
            device,
            batch=batch,
            context=context,
        )
    )


def _decode_tile_defaults(
    *,
    heads_per_group: int,
    head_dim: int,
    slots_per_split: int,
    block_regions: int,
    region_size: int,
    element_size: int,
) -> tuple[int, int, int]:
    block_h = max(16, triton.next_power_of_2(heads_per_group))
    block_d = triton.next_power_of_2(head_dim)
    block_r = _fit_block_regions(
        min(block_regions, triton.next_power_of_2(slots_per_split)),
        region_size=region_size,
        block_d=block_d,
        element_size=element_size,
    )
    return block_h, block_d, block_r


def sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: float | None = None,
    num_splits: int = 1,
    block_regions: int = 16,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention for one query token per request.

    Args:
        query: ``[num_reqs, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, pages flattened.
        packed_regions: ``[num_kv_groups * num_reqs, topk]`` **int64**, each slot
            ``start_token | (phys_region << 32)``. Note this is a different packing from
            prefill's int32 ``phys_region | (valid_tokens << 24)``: decode carries the
            logical start token and derives the visible length from ``kv_seqlens``.
        region_counts: ``[num_kv_groups * num_reqs]`` int32, slots to read per row.
        kv_seqlens: ``[num_reqs]`` int32, context length including the current token.
        num_splits: split the slot window this many ways and return partial states. With
            ``num_splits == 1`` the returned tensors are already the final result;
            otherwise pass them to :func:`merge_split_states`.
        config: launch override. ``None`` loads
            ``config/sparse_attn_decode/decode-k{width}-s{splits}-b{batch}-ctx{context}-hpg{hpg}-d{dim}-r{region}_{dtype}_{device}.json``
            using ``batch=num_reqs`` and ``context=kv_seqlens[0]``. Missing that
            file uses the built-in default tile. A mapping is used as-is (no
            shared-memory shrink). An empty mapping skips lookup and keeps the
            built-in defaults.

    Returns:
        ``num_splits == 1``: ``(out [num_reqs, num_heads, head_dim], lse [num_reqs, num_heads])``.
        Otherwise the same with a leading ``num_splits`` axis.

    As in prefill, the selected regions must be distinct. The upstream decode reference is
    stricter about this than the prefill one: it concatenates the gathered spans with no
    dedup at all, so a duplicated region would be counted twice in the softmax.
    """
    num_reqs, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * num_reqs:
        raise ValueError(
            f"expected {num_kv_groups * num_reqs} metadata rows, got {packed_regions.shape[0]}"
        )
    if packed_regions.dtype != torch.int64:
        raise ValueError(f"decode metadata must be int64, got {packed_regions.dtype}")
    if kv_seqlens.shape[0] != num_reqs:
        raise ValueError(f"expected {num_reqs} kv_seqlens, got {kv_seqlens.shape[0]}")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    # Every split walks the same number of slots so the window bound is a constexpr; the
    # tail split simply finds every slot masked out.
    slots_per_split = -(-topk // num_splits)

    out = torch.empty((num_splits, num_reqs, num_heads, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((num_splits, num_reqs, num_heads), device=query.device, dtype=torch.float32)
    launch = config
    if launch is None:
        launch = _load_sparse_attn_decode_config(
            topk,
            num_splits,
            heads_per_group,
            head_dim,
            region_size,
            _dtype_tag(query.dtype),
            _device_name_for_config(),
            num_reqs,
            int(kv_seqlens[0].item()),
        )
    default_h, default_d, default_r = _decode_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        slots_per_split=slots_per_split,
        block_regions=block_regions,
        region_size=region_size,
        element_size=key_cache.element_size(),
    )
    extra: dict[str, Any] = {}
    if launch:
        block_h = int(launch.get("BLOCK_H", default_h))
        block_d = int(launch.get("BLOCK_D", default_d))
        block_r = int(launch["BLOCK_R"])
        extra = {key: launch[key] for key in _GRID_KEYS if key in launch}
    else:
        block_h, block_d, block_r = default_h, default_d, default_r

    _sparse_attn_decode_kernel[(num_reqs, num_kv_groups, num_splits)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
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
        num_reqs,
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
    return out, lse
