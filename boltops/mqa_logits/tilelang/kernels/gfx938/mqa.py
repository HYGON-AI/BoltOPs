# SPDX-License-Identifier: MIT
"""gfx938 entry points for TileLang QSA MQA kernels."""

from __future__ import annotations

import math
from typing import Optional

import torch

from boltops.mqa_logits.tilelang.configs.mqa_decode import (
    resolve_get_config_fp8 as resolve_get_decode_config_fp8,
)
from boltops.mqa_logits.tilelang.configs.mqa_prefill import (
    resolve_get_config as resolve_get_prefill_config,
    resolve_get_config_fp8 as resolve_get_prefill_config_fp8,
)
from boltops.utility.arch_info import get_device_info

from ..common import mqa as _common

if _common.HAS_TILELANG:
    from .mqa_decode_fp8 import (
        _tilelang_qsa_mqa_decode_fp8_kernel,
        _tilelang_qsa_mqa_decode_fp8_persistent_kernel,
    )
    from .mqa_prefill_wasp import _tilelang_qsa_mqa_prefill_wasp_kernel
    from .mqa_prefill_wasp_fp8 import _tilelang_qsa_mqa_prefill_wasp_fp8_kernel


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """gfx938 decode: packed FP8 ``D+4`` cache, else BF16 occupancy."""
    packed = _common._is_fp8_packed_decode_cache(q, k_cache)
    if packed:
        return _qsa_mqa_decode_fp8(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    if _common._is_fp8_dtype(q.dtype) or _common._is_fp8_dtype(k_cache.dtype):
        raise ValueError(
            "FP8 QSA decode requires aiter packed k_cache [pages, page_size, 1, D+4]"
        )
    return _common.qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


def _qsa_mqa_decode_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float],
) -> torch.Tensor:
    _common._validate_decode_inputs(
        q, k_cache, page_table, context_lens, allow_packed_fp8=True
    )
    if not q.is_cuda:
        raise ValueError("TileLang QSA decode requires device tensors")
    query_heads, head_dim = q.shape[1:]
    if query_heads not in (4, 8, 16, 32, 64):
        raise ValueError(
            "TileLang QSA decode requires query heads in 4/8/16/32/64, "
            f"got {query_heads}"
        )
    if head_dim != 128:
        raise ValueError(
            f"TileLang QSA decode requires head_dim=128, got {head_dim}"
        )
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.empty(
        (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
    )
    if not q.shape[0] or not max_model_len:
        return logits
    packed_cache_shape = (
        (k_cache.shape[0], 1, page_size * (head_dim + 4))
        if query_heads > 16
        else (k_cache.shape[0], page_size * (head_dim + 4))
    )
    packed_k_cache = k_cache.view(torch.float8_e4m3fn).reshape(
        packed_cache_shape
    )
    device_info = get_device_info(q.device)
    config = resolve_get_decode_config_fp8(device_info.arch)(
        q.shape[0],
        max_model_len,
        device_info.cu_count,
        num_heads=query_heads,
    )
    kernel_factory = (
        _tilelang_qsa_mqa_decode_fp8_persistent_kernel
        if config.use_persistent
        else _tilelang_qsa_mqa_decode_fp8_kernel
    )
    kernel_kwargs = dict(
        heads=query_heads,
        head_dim=head_dim,
        page_size=page_size,
        group_size=config.group_size,
        num_stages=config.num_stages,
        use_async=config.use_async,
    )
    if config.use_persistent:
        kernel_kwargs["persistent_splits"] = config.persistent_splits
    else:
        kernel_kwargs["groups_per_cta"] = config.groups_per_cta
    context_lens_kernel = context_lens.to(
        device=q.device, dtype=torch.int32
    ).contiguous()
    kernel_factory(**kernel_kwargs)(
        q.unsqueeze(1).contiguous(),
        packed_k_cache,
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens_kernel,
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    suffix_splits = _common._get_decode_suffix_splits(q.shape[0], max_model_len)
    _common._tilelang_qsa_mqa_decode_mask_kernel(
        steady_phases=(
            16 if suffix_splits == 1 and max_model_len >= 64 * 1024 else 1
        ),
        suffix_splits=suffix_splits,
        reverse_suffix=max_model_len > 8 * 1024,
    )(logits, context_lens_kernel)
    return logits


def prefill_mask_launch_kwargs(rows: int, keys: int) -> dict[str, int | bool]:
    """Shape heuristic for the gfx938 prefill invalid-range mask.

    Reverse helps half_ramp / ragged suffixes once a full 1K-element round
    exists.  near_full_ramp does not regress.  Phase-16 is only a win at
    64K; 32K stays phase-1 plus reverse.  row_phases=4 is only for the
    large-row 64K case.
    """
    long_key = keys >= 8 * 1024
    very_long_key = keys >= 64 * 1024
    large_rows = rows >= 4096
    return dict(
        steady_phases=16 if very_long_key else 1,
        row_phases=4 if large_rows and very_long_key else 1,
        reverse_suffix=long_key,
    )


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    kv_scales: Optional[torch.Tensor] = None,
    effective_range: Optional[int] = None,
) -> torch.Tensor:
    """Run the gfx938 WASP prefill kernel when its fixed geometry applies.

    ``kv_scales`` selects FP8 WASP and ``get_config_fp8``; ``None`` keeps
    BF16 ``get_config``.  ``effective_range`` is the longest valid key span.
    ``None`` keeps the previous ``num_keys`` heuristic.
    """
    if (
        not _common.HAS_TILELANG
        or q.ndim != 3
        or q.shape[1] not in (4, 8, 32, 64)
        or q.shape[2] != 128
    ):
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 QSA prefill WASP is gfx938-only for H in {4,8,32,64}, D=128"
            )
        return _common.qsa_mqa_prefill(
            q, k, row_starts, row_ends, score_scale, None, effective_range
        )

    _common._validate_q(q)
    _common._validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if q.dtype != k.dtype:
        raise ValueError("QSA query and key dtypes must match")

    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )

    device_info = get_device_info(q.device)
    get_config = (
        resolve_get_prefill_config_fp8(device_info.arch)
        if kv_scales is not None
        else resolve_get_prefill_config(device_info.arch)
    )
    config = get_config(
        rows,
        keys,
        device_info.cu_count,
        effective_range=effective_range,
        num_heads=q.shape[1],
    )

    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    inv_scale = 1.0 / (score_scale or math.sqrt(128))

    if kv_scales is None:
        q_kernel = q.to(torch.bfloat16).contiguous()
        _tilelang_qsa_mqa_prefill_wasp_kernel(
            block_q=config.block_q,
            block_n=config.block_n,
            heads=config.heads,
            threads=config.threads,
            rows_divisible=config.rows_divisible,
            n_tile_phases=config.n_tile_phases,
            split_n=config.split_n,
            apply_scale=inv_scale != 1.0,
        )(
            q_kernel.reshape(-1, 128),
            k[:, 0].to(torch.bfloat16).contiguous(),
            logits,
            starts,
            ends,
            inv_scale,
        )
    else:
        if "float8_e4m3" not in str(q.dtype):
            raise ValueError(
                "kv_scales is set → FP8 QSA prefill requires float8_e4m3* Q/K, "
                f"got {q.dtype}"
            )
        if kv_scales.shape != (keys,) or kv_scales.dtype != torch.float32:
            raise ValueError(
                "kv_scales must be float32 [keys], "
                f"got {tuple(kv_scales.shape)} {kv_scales.dtype}"
            )
        _tilelang_qsa_mqa_prefill_wasp_fp8_kernel(
            block_q=config.block_q,
            block_n=config.block_n,
            heads=config.heads,
            threads=config.threads,
            rows_divisible=config.rows_divisible,
            n_tile_phases=config.n_tile_phases,
            split_n=config.split_n,
            apply_scale=inv_scale != 1.0,
        )(
            q.contiguous().reshape(-1, 128),
            k[:, 0].contiguous(),
            kv_scales.to(device=q.device, dtype=torch.float32).contiguous(),
            logits,
            starts,
            ends,
            inv_scale,
        )
    _common._tilelang_qsa_mqa_mask_kernel(
        **prefill_mask_launch_kwargs(rows, keys)
    )(logits, starts, ends)
    return logits


__all__ = ["qsa_mqa_decode", "qsa_mqa_prefill", "prefill_mask_launch_kwargs"]
