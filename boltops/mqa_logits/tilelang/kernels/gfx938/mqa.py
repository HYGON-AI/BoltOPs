# SPDX-License-Identifier: MIT
"""gfx938 entry points for TileLang QSA MQA kernels."""

from __future__ import annotations

import math
from typing import Optional

import torch

from boltops.mqa_logits.tilelang.configs.mqa_prefill import (
    resolve_get_config as resolve_get_prefill_config,
)
from boltops.utility.arch_info import get_device_info

from ..common import mqa as _common

if _common.HAS_TILELANG:
    from .mqa_prefill_wasp import _tilelang_qsa_mqa_prefill_wasp_kernel


qsa_mqa_decode = _common.qsa_mqa_decode


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
    effective_range: Optional[int] = None,
) -> torch.Tensor:
    """Run the gfx938 WASP prefill kernel when its fixed geometry applies.

    ``effective_range`` is the longest valid key span and is forwarded to
    ``get_config``.  ``None`` keeps the previous ``num_keys`` heuristic.
    """
    if (
        not _common.HAS_TILELANG
        or q.ndim != 3
        or q.shape[1] not in (4, 8, 32, 64)
        or q.shape[2] != 128
    ):
        return _common.qsa_mqa_prefill(
            q, k, row_starts, row_ends, score_scale, effective_range
        )

    _common._validate_q(q)
    _common._validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")

    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )

    device_info = get_device_info(q.device)
    get_config = resolve_get_prefill_config(device_info.arch)
    config = get_config(
        rows,
        keys,
        device_info.cu_count,
        effective_range=effective_range,
        num_heads=q.shape[1],
    )

    # Unlike the common fallback, WASP handles the final partial row CTA in
    # the kernel.  Keep all externally visible tensors at their true row size.
    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    q_kernel = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    inv_scale = 1.0 / (score_scale or math.sqrt(128))

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
    _common._tilelang_qsa_mqa_mask_kernel(
        **prefill_mask_launch_kwargs(rows, keys)
    )(logits, starts, ends)
    return logits


__all__ = ["qsa_mqa_decode", "qsa_mqa_prefill", "prefill_mask_launch_kwargs"]
