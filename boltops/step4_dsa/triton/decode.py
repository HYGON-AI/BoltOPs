"""Sparse GQA decode attention over selected regions, with optional split-KV.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``sparse_attn`` decode).
Matches the vLLM / handoff caller: ``num_splits==1`` auto-splits to fill CUs and
returns merged states; ``num_splits>1`` returns partials for the caller to merge.
Split windows are disjoint (``slots < window_end``). Tuner / occupancy JSON
measurement opts out with ``auto_kv_split=False``.

On the ``num_splits==1`` path, occupancy JSON may pin the launch grid with
``AUTO_KV_SPLITS``; otherwise :func:`choose_decode_kv_splits` is used.
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
    _fit_block_regions,
    _require_contiguous_last_dim,
)
from boltops.utility.triton_capability import get_triton_capabilities

DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)

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
    BALANCE_LIVE: tl.constexpr = False,
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

    Split windows are a partition of the packed row: the last ``BLOCK_R`` tile is
    clamped with ``slots < window_end``. Without that, ``slots_per_split % BLOCK_R
    != 0`` walks into the next split and the same region is softmaxed twice.

    K is loaded as ``[BLOCK_D, tokens]`` so QK is ``tl.dot(Q, K)`` without
    ``tl.trans``. V stays ``[tokens, BLOCK_D]`` for ``P @ V``. Launch tiles come
    from ``config/sparse_attn_decode``.

    Online softmax is in the log2 domain: ``log2(e)`` is folded into the softmax
    scale and ``exp2`` replaces ``exp``. Stored LSE is converted back to natural
    log (``running_max * ln2 + log(sum)``), so the merge-kernel contract is
    unchanged.
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

    scale = softmax_scale * LOG2E
    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    if BALANCE_LIVE:
        # Balance device-resident row lengths without a host read or metadata copy.
        slots_per_split = tl.cdiv(tl.minimum(count, topk), tl.num_programs(2))
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
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
            k_ptr
            + tokens[None, :] * stride_k_token
            + group * stride_k_group
            + dims[:, None],
            mask=dim_ok[:, None] & token_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr + tokens[:, None] * stride_v_token + group * stride_v_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, keys, out_dtype=tl.float32) * scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

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

    selected = running_sum > 0
    # Reuse one reciprocal per head instead of dividing every output element.
    inv_sum = tl.div_rn(1.0, tl.where(selected, running_sum, 1.0))
    output = tl.where(selected[:, None], accumulator * inv_sum[:, None], 0.0)
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
        tl.where(selected, running_max * LN2 + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


@triton.jit
def _sparse_attn_decode_dsplit_kernel(
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
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BALANCE_LIVE: tl.constexpr = False,
) -> None:
    """Same contract as ``_sparse_attn_decode_kernel``; non-pow2 ``head_dim`` fast path.

    Two differences, both performance-only (bit-level output differs by fp
    rounding order, not semantics):

    - ``head_dim`` is covered by three pow2 chunks ``BLOCK_D1 + BLOCK_D2 +
      BLOCK_D3 == head_dim`` (e.g. 192 = 64 + 64 + 64) instead of one padded
      ``BLOCK_D = next_pow2(head_dim)`` or a 128+64 pair. QK is
      ``dot(q0, k0) + dot(q1, k1) + dot(q2, k2)`` and the PV accumulator is
      split the same way. Three 64-wide MMACs use fewer VGPRs than one 128-wide
      tile plus one 64-wide tile. This also removes the padding share of the
      K/V LDS tiles, the MMACs and the accumulator rescale (25% for 192), and
      the ``dims < head_dim`` mask disappears because the chunks cover the head
      exactly.
    - The online softmax runs in the log2 domain: ``log2(e)`` is folded into
      the softmax scale and ``exp2`` replaces ``exp``. The stored LSE is
      converted back to natural log (``running_max * ln2 + log(sum)``), so the
      merge-kernel contract is unchanged.

    Launched by :func:`sparse_attention_decode` when ``head_dim`` splits into
    three 64-wide chunks (currently ``head_dim=192``); otherwise the original
    kernel runs. Occupancy JSONs omit ``BLOCK_D`` / ``BLOCK_D1`` / ``BLOCK_D2``
    / ``BLOCK_D3`` on this path; the chunks are derived from ``head_dim``. A
    leftover ``BLOCK_D`` is ignored.
    """
    tl.static_assert(BLOCK_D1 + BLOCK_D2 + BLOCK_D3 == head_dim)
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dims0 = tl.arange(0, BLOCK_D1)
    dims1 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    dims2 = BLOCK_D1 + BLOCK_D2 + tl.arange(0, BLOCK_D3)
    head_ok = heads < heads_per_group

    q_base = (
        q_ptr
        + request * stride_q_token
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
    if BALANCE_LIVE:
        # Balance device-resident row lengths without a host read or metadata copy.
        slots_per_split = tl.cdiv(tl.minimum(count, topk), tl.num_programs(2))
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        k_base = k_ptr + tokens[None, :] * stride_k_token + group * stride_k_group
        keys0 = tl.load(k_base + dims0[:, None], mask=token_ok[None, :], other=0.0)
        keys1 = tl.load(k_base + dims1[:, None], mask=token_ok[None, :], other=0.0)
        keys2 = tl.load(k_base + dims2[:, None], mask=token_ok[None, :], other=0.0)
        v_base = v_ptr + tokens[:, None] * stride_v_token + group * stride_v_group
        values0 = tl.load(v_base + dims0[None, :], mask=token_ok[:, None], other=0.0)
        values1 = tl.load(v_base + dims1[None, :], mask=token_ok[:, None], other=0.0)
        values2 = tl.load(v_base + dims2[None, :], mask=token_ok[:, None], other=0.0)

        scores = (
            tl.dot(queries0, keys0, out_dtype=tl.float32)
            + tl.dot(queries1, keys1, out_dtype=tl.float32)
            + tl.dot(queries2, keys2, out_dtype=tl.float32)
        ) * scale
        scores = tl.where(
            token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY
        )

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.math.exp2(running_max - shift))
        probabilities = tl.math.exp2(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        p = probabilities.to(values0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(p, values0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(p, values1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(p, values2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    # Reuse one reciprocal per head instead of dividing every output element.
    inv_sum = tl.div_rn(1.0, tl.where(selected, running_sum, 1.0))
    output0 = tl.where(selected[:, None], acc0 * inv_sum[:, None], 0.0)
    output1 = tl.where(selected[:, None], acc1 * inv_sum[:, None], 0.0)
    output2 = tl.where(selected[:, None], acc2 * inv_sum[:, None], 0.0)
    out_base = (
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
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
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max * LN2 + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


def _decode_head_dim_split(head_dim: int) -> tuple[int, int, int] | None:
    """``(D1, D2, D3)`` pow2 MMAC chunks covering ``head_dim``, or ``None``.

    ``None`` (pow2 ``head_dim``, or a remainder that is not the 192-dim
    ``128+64`` split) keeps the original single-``BLOCK_D`` kernel.

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


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "sparse_attn_decode",
)
# Shared by both kernels. ``BLOCK_D`` is only in the padded-kernel JSON;
# dsplit occupancy files omit it (and never carry ``BLOCK_D1`` / ``BLOCK_D2``
# / ``BLOCK_D3``).
# ``AUTO_KV_SPLITS`` is host-only: pins the s1 auto-path launch grid; never a
# kernel constexpr.
_HOST_CONFIG_KEYS = frozenset({"AUTO_KV_SPLITS"})
_COMMON_KERNEL_CONFIG_KEYS = frozenset({"BLOCK_R", "BLOCK_H", "BALANCE_LIVE"})
_PADDED_KERNEL_CONFIG_KEYS = _COMMON_KERNEL_CONFIG_KEYS | {"BLOCK_D"}
_ALL_KERNEL_CONFIG_KEYS = _PADDED_KERNEL_CONFIG_KEYS | {"BLOCK_D1", "BLOCK_D2", "BLOCK_D3"}
_HOST_OR_KERNEL_CONFIG_KEYS = _ALL_KERNEL_CONFIG_KEYS | _HOST_CONFIG_KEYS
_STANDARD_LAUNCH_KEYS = frozenset({"num_warps", "num_stages"})


def _build_launch_options(config: dict[str, Any]) -> dict[str, Any]:
    """Build portable Triton launch options from a decode config.

    Standard launch fields are always accepted. HIP-specific compiler fields
    are forwarded only when the installed backend declares support for them.
    Kernel constexpr, host-only, unknown, and unsupported fields are omitted.
    """
    options = {
        name: int(config[name])
        for name in _STANDARD_LAUNCH_KEYS
        if name in config
    }
    if torch.version.hip is None:
        return options

    capabilities = get_triton_capabilities()
    consumed = _HOST_OR_KERNEL_CONFIG_KEYS | _STANDARD_LAUNCH_KEYS
    for name, value in config.items():
        if name not in consumed and capabilities.supports(name):
            options[name] = value
    return options


def _decode_config_filename(device: str) -> str:
    """Return ``sparse_attn_decode_{device}.json``."""
    return f"sparse_attn_decode_{device}.json"


def _decode_config_match_score(
    match: dict[str, Any], shape: dict[str, Any]
) -> tuple[int, int, float] | None:
    """Return match specificity, or ``None`` when a shape does not match."""
    exact = 0
    bounds = 0
    widths: dict[str, dict[str, float]] = {}
    for key, expected in match.items():
        if key.endswith("_min") or key.endswith("_max"):
            field, kind = key.rsplit("_", 1)
            actual = shape.get(field)
            if actual is None:
                return None
            if kind == "min" and actual < expected:
                return None
            if kind == "max" and actual > expected:
                return None
            widths.setdefault(field, {})[kind] = expected
            bounds += 1
        elif shape.get(key) != expected:
            return None
        else:
            exact += 1
    total_width = 0.0
    for limits in widths.values():
        if "min" in limits and "max" in limits:
            total_width += limits["max"] - limits["min"]
        else:
            total_width += float("inf")
    return exact, bounds, -total_width


@functools.lru_cache(maxsize=None)
def _load_decode_device_configs(device: str) -> dict[str, Any]:
    path = os.path.join(_CONFIG_DIR, _decode_config_filename(device))
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _filter_launch_config(config: dict[str, Any], *, head_dim: int) -> dict[str, Any]:
    """Keep host, selected-kernel, and supported Triton launch fields.

    ``_sparse_attn_decode_dsplit_kernel``: no ``BLOCK_D`` (chunks come from
    ``head_dim``). Leftover ``BLOCK_D`` / ``BLOCK_D1`` / ``BLOCK_D2`` /
    ``BLOCK_D3`` are dropped.
    ``_sparse_attn_decode_kernel``: keep ``BLOCK_D`` when the JSON has it.
    ``AUTO_KV_SPLITS`` is retained for the s1 auto path and never forwarded to
    the kernel. Other compiler fields are retained only when the current HIP
    backend declares them in ``HIPOptions``.
    """
    dsplit = _decode_head_dim_split(head_dim) is not None
    kernel_keys = _COMMON_KERNEL_CONFIG_KEYS if dsplit else _PADDED_KERNEL_CONFIG_KEYS
    consumed = kernel_keys | _HOST_CONFIG_KEYS
    filtered = {key: value for key, value in config.items() if key in consumed}
    filtered.update(_build_launch_options(config))
    return filtered


def _clamp_auto_kv_splits(value: int, packed_width: int) -> int:
    return min(max(int(value), 1), max(int(packed_width), 1))


def _load_sparse_attn_decode_config(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    kv_seqlens: torch.Tensor,
    num_splits: int,
    region_size: int,
) -> dict[str, Any] | None:
    """Select the most specific decode config for the runtime tensor shape.

    Equivalent dimensions are represented once: ``d`` comes from ``query`` and
    applies to query/key/value, while ``h_kv`` comes from ``key_cache`` and is
    equivalent to the public ``num_kv_groups`` argument. Physical cache capacity
    (``key_cache.shape[0]``) is excluded; actual work is represented by ``s_kv``
    from the maximum runtime sequence length and by ``topk``.
    """
    device_tag = _device_name_for_config()
    data = _load_decode_device_configs(device_tag)
    shape = {
        "s_q": int(query.shape[0]),
        "s_kv": int(kv_seqlens.max().item()),
        "h_q": int(query.shape[1]),
        "h_kv": int(key_cache.shape[1]),
        "topk": int(packed_regions.shape[1]),
        "d": int(query.shape[2]),
        "num_splits": int(num_splits),
        "region_size": int(region_size),
        "dtype": str(query.dtype).replace("torch.", ""),
    }
    selected = None
    selected_score: tuple[int, int, float] | None = None
    for entry in data.get("configs", ()):
        match = entry.get("match", {})
        if not isinstance(match, dict):
            continue
        score = _decode_config_match_score(match, shape)
        if score is not None and (selected_score is None or score > selected_score):
            selected, selected_score = entry, score
    if selected is None:
        return None
    config = selected.get("config", {})
    if not isinstance(config, dict):
        return None
    filtered = _filter_launch_config(config, head_dim=int(query.shape[2]))
    if "BLOCK_R" not in filtered:
        return None
    return filtered


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


def choose_decode_kv_splits(
    num_reqs: int,
    num_kv_groups: int,
    packed_width: int,
    *,
    target_programs: int | None = None,
    min_slots_per_split: int = 8,
    device: torch.device | None = None,
) -> int:
    """Pick a split count that fills CUs without emptying each split's slot window.

    Size-aware occupancy only: uses ``B * G`` and device CU count. Caps splits so
    ``ceil(packed_width / splits) >= min_slots_per_split`` when the row is full
    (``max_splits = (K - 1) // (M - 1)`` for ``M > 1``).
    """
    packed_width = max(int(packed_width), 1)
    programs_base = max(int(num_reqs) * int(num_kv_groups), 1)
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
    min_slots = max(int(min_slots_per_split), 1)
    if min_slots <= 1:
        max_splits = packed_width
    else:
        # ceil(K/S) >= M  <=>  S <= (K - 1) // (M - 1)
        max_splits = max(1, (packed_width - 1) // (min_slots - 1))
    return min(max(splits, 1), max_splits, packed_width)


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
    num_splits: int | None = 1,
    block_regions: int = 16,
    config: dict[str, Any] | None = None,
    kv_split_target_programs: int | None = None,
    min_slots_per_split: int = 8,
    auto_kv_split: bool | None = None,
    return_partials: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention for one query token per request.

    Args:
        query: ``[num_reqs, num_heads, head_dim]``. Last dim must be contiguous
            (``stride(-1) == 1``); token/head strides are passed through. The
            kernel addresses ``head_dim`` as ``base + d``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, pages flattened.
            Same last-dim rule as ``query``. Allocated ``out`` is checked the same way.
        packed_regions: ``[num_kv_groups * num_reqs, topk]`` **int64**, each slot
            ``start_token | (phys_region << 32)``. Note this is a different packing from
            prefill's int32 ``phys_region | (valid_tokens << 24)``: decode carries the
            logical start token and derives the visible length from ``kv_seqlens``.
        region_counts: ``[num_kv_groups * num_reqs]`` int32, slots to read per row.
        kv_seqlens: ``[num_reqs]`` int32, context length including the current token.
        num_splits: vLLM / handoff contract. ``1`` (and ``None``) means "return
            final states": occupancy-split to fill CUs, then merge. ``>1`` is an
            explicit split count and returns partials with a leading split axis
            for :func:`merge_split_states`. Launch JSON is keyed by this
            caller-facing value (``s1`` is the fast path, even if launch is N-way).
        auto_kv_split: ``None`` follows the contract above. ``False`` uses
            ``num_splits`` as-is (true 1-way when it is 1).
        return_partials: ``None`` follows the contract (partials only when the
            caller asked for ``num_splits>1``). ``True`` / ``False`` overrides.
        config: launch override. ``None`` loads
            ``config/sparse_attn_decode/sparse_attn_decode_{device}.json`` and
            selects the most specific entry matching the runtime tensor shape,
            sequence length, and caller-facing split count. If an explicit
            ``num_splits>1`` entry is absent, selection retries with ``num_splits=1``
            before using the built-in default tile. A mapping is filtered to
            kernel fields plus portable Triton launch options (no shared-memory
            shrink). Standard ``num_warps`` / ``num_stages`` fields are accepted;
            additional HIP compiler fields such as ``waves_per_eu`` or
            ``misched_regpressure`` are forwarded only when the current backend's
            ``HIPOptions`` declares them. Unknown or unsupported fields are ignored.
            An empty mapping skips lookup and keeps the built-in defaults.
            On the auto path (``num_splits==1``), ``AUTO_KV_SPLITS`` in the
            occupancy JSON / override pins the launch grid (clamped to
            ``[1, packed_width]``); if absent, :func:`choose_decode_kv_splits`
            is used. Explicit ``num_splits>1`` ignores ``AUTO_KV_SPLITS``.
            Occupancy JSON for ``_sparse_attn_decode_kernel`` includes ``BLOCK_D``;
            the dsplit kernel (non-pow2 ``head_dim``) omits it and derives
            ``BLOCK_D1`` / ``BLOCK_D2`` / ``BLOCK_D3`` from ``head_dim``. Leftover
            ``BLOCK_D`` / ``BLOCK_D1`` / ``BLOCK_D2`` / ``BLOCK_D3`` on that path
            are ignored.
            ``BALANCE_LIVE`` balances valid slots across automatic splits. It is
            ignored for explicit splits or partial outputs to preserve their windows.

    Returns:
        ``num_splits==1``: merged ``(out [num_reqs, num_heads, head_dim],
        lse [num_reqs, num_heads])``. ``num_splits>1``: the same with a leading
        split axis, unless ``return_partials=False``.

    As in prefill, the selected regions must be distinct. The upstream decode reference is
    stricter about this than the prefill one: it concatenates the gathered spans with no
    dedup at all, so a duplicated region would be counted twice in the softmax.
    """
    from boltops.step4_dsa.triton.merge import merge_split_states

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
    _require_contiguous_last_dim(query, "query")
    _require_contiguous_last_dim(key_cache, "key_cache")
    _require_contiguous_last_dim(value_cache, "value_cache")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    if auto_kv_split is None:
        auto_kv_split = num_splits is None or int(num_splits) == 1
    if auto_kv_split:
        config_splits = 1
    else:
        num_splits = max(int(num_splits if num_splits is not None else 1), 1)
        config_splits = num_splits

    # Load occupancy JSON before choosing the auto-path grid so ``AUTO_KV_SPLITS``
    # can pin launch. JSON is named by the caller-facing num_splits (s1 = fast
    # path), not the occupancy launch count.
    launch = config
    if launch is None:
        launch = _load_sparse_attn_decode_config(
            query,
            key_cache,
            packed_regions,
            kv_seqlens,
            config_splits,
            region_size,
        )
        if not launch and config_splits != 1:
            launch = _load_sparse_attn_decode_config(
                query,
                key_cache,
                packed_regions,
                kv_seqlens,
                1,
                region_size,
            )
    elif launch:
        launch = _filter_launch_config(launch, head_dim=head_dim)

    if auto_kv_split:
        pinned = launch.get("AUTO_KV_SPLITS") if launch else None
        if pinned is not None:
            num_splits = _clamp_auto_kv_splits(int(pinned), topk)
        else:
            num_splits = choose_decode_kv_splits(
                num_reqs,
                num_kv_groups,
                topk,
                target_programs=kv_split_target_programs,
                min_slots_per_split=min_slots_per_split,
                device=query.device,
            )
    if return_partials is None:
        return_partials = (not auto_kv_split) and num_splits > 1
    slots_per_split = -(-topk // num_splits)

    out = torch.empty(
        (num_splits, num_reqs, num_heads, head_dim), device=query.device, dtype=query.dtype
    )
    _require_contiguous_last_dim(out, "out")
    lse = torch.empty((num_splits, num_reqs, num_heads), device=query.device, dtype=torch.float32)
    default_h, default_d, default_r = _decode_tile_defaults(
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        slots_per_split=slots_per_split,
        block_regions=block_regions,
        region_size=region_size,
        element_size=key_cache.element_size(),
    )
    extra: dict[str, Any] = {}
    d_split = _decode_head_dim_split(head_dim)
    if launch:
        block_h = int(launch.get("BLOCK_H", default_h))
        block_r = int(launch["BLOCK_R"])
        extra = {
            key: value
            for key, value in launch.items()
            if key not in _HOST_OR_KERNEL_CONFIG_KEYS
        }
    else:
        block_h, block_r = default_h, default_r

    common_args = (
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
    )
    common_meta = dict(
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
        BALANCE_LIVE=bool(launch and launch.get("BALANCE_LIVE", False))
        and auto_kv_split and not return_partials and num_splits > 1,
    )
    if d_split is not None:
        # Non-pow2 head_dim (e.g. 192): padding-free 64+64+64 kernel.
        # Occupancy JSON omits ``BLOCK_D``; chunks come from ``head_dim``.
        _sparse_attn_decode_dsplit_kernel[(num_reqs, num_kv_groups, num_splits)](
            *common_args,
            **common_meta,
            BLOCK_D1=d_split[0],
            BLOCK_D2=d_split[1],
            BLOCK_D3=d_split[2],
            **extra,
        )
    else:
        # Padded-kernel JSON still has ``BLOCK_D``; missing it uses the default tile.
        block_d = int(launch.get("BLOCK_D", default_d)) if launch else default_d
        _sparse_attn_decode_kernel[(num_reqs, num_kv_groups, num_splits)](
            *common_args,
            **common_meta,
            BLOCK_D=block_d,
            **extra,
        )
    if num_splits == 1:
        return out[0], lse[0]
    if return_partials:
        return out, lse
    return merge_split_states(out, lse)
