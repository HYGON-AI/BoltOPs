# SPDX-License-Identifier: MIT
"""gfx938 policy for the 12-wave MQA logits prefill kernel."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .common import QSAMQAPrefillConfig


_LOGICAL_M_CHOICES = (32, 64, 128, 256)
_SPLIT_N_CHOICES = (1, 2, 4, 8, 16, 32, 64)
_MAX_CTA_STARTUP_N_TILES = 2


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@lru_cache(maxsize=1024)
def get_config(
    num_rows: int,
    num_keys: int,
    cu_count: int,
    effective_range: Optional[int] = None,
    num_heads: int = 4,
) -> QSAMQAPrefillConfig:
    """Pick WASP launch tiles for gfx938.

    ``effective_range`` is the longest valid key span
    ``max_i (row_ends[i] - row_starts[i])``.  Omit it (``None``) to size
    phases and split-N from ``num_keys``.
    """
    if num_rows <= 0:
        raise ValueError(f"QSA MQA prefill requires num_rows > 0, got {num_rows}")
    if num_keys <= 0:
        raise ValueError(f"QSA MQA prefill requires num_keys > 0, got {num_keys}")
    if cu_count <= 0:
        raise ValueError(f"QSA MQA prefill requires cu_count > 0, got {cu_count}")
    if num_heads not in (4, 8, 32, 64):
        raise ValueError(
            "gfx938 WASP requires num_heads in (4, 8, 32, 64), got "
            f"{num_heads}"
        )
    if effective_range is not None and not 0 < effective_range <= num_keys:
        raise ValueError(
            "effective_range must be in (0, num_keys], got "
            f"{effective_range} for num_keys={num_keys}"
        )

    phase_range = effective_range if effective_range is not None else num_keys
    block_q_choices = tuple(
        logical_m // num_heads
        for logical_m in _LOGICAL_M_CHOICES
        if (
            logical_m % num_heads == 0
            and logical_m // num_heads >= 2
            and logical_m // num_heads % 2 == 0
        )
    )
    if not block_q_choices:
        raise ValueError(
            f"no logical-M choice is divisible by num_heads={num_heads}"
        )

    rows_per_cu = _ceil_div(num_rows, cu_count)
    if num_heads == 4:
        # Start with the Q-only occupancy choice, then enforce the measured
        # logical-M floor for longer key ranges.  At small row counts a tiny
        # logical M (for example block_q=8 / logical-M=32) leaves enough Q
        # CTAs, but its GEMM granularity dominates runtime.  Split-N can fill
        # the device after increasing block_q without over-decomposing beyond
        # roughly one CTA round.
        block_q = next(
            (
                candidate
                for candidate in block_q_choices
                if candidate >= rows_per_cu
            ),
            block_q_choices[-1],
        )
        if phase_range >= 32 * 1024:
            preferred_block_q = 64
        elif phase_range >= 8 * 1024:
            preferred_block_q = 32
        else:
            preferred_block_q = block_q_choices[0]
        # Do not introduce additional padded Q rows solely to reach the
        # logical-M floor.  Existing sub-tile row shapes retain their nearest
        # supported block_q and the normal partial-row handling.
        if preferred_block_q > num_rows:
            preferred_block_q = max(
                candidate
                for candidate in block_q_choices
                if candidate <= max(num_rows, block_q_choices[0])
            )
        block_q = max(block_q, preferred_block_q)
    else:
        # For wider head groups, making logical M too small hurts the GEMM
        # wave geometry even when it creates more Q CTAs.  Measurements favor
        # logical-M=128 as the baseline, with the exceptions below.
        preferred_logical_m = 128
        if (
            num_heads == 32
            and phase_range <= 2048
            and num_rows < 4 * cu_count
        ):
            preferred_logical_m = 64
        elif num_heads == 32 and phase_range >= 16 * 1024:
            preferred_logical_m = 256
        elif num_heads == 8 and rows_per_cu > 16:
            preferred_logical_m = 256
        elif num_heads == 64 and (
            num_rows > 2 * cu_count or phase_range >= 32 * 1024
        ):
            preferred_logical_m = 256
        preferred_block_q = preferred_logical_m // num_heads
        block_q = next(
            (
                candidate
                for candidate in block_q_choices
                if candidate >= preferred_block_q
            ),
            block_q_choices[-1],
        )

    # Keep the byte distance between adjacent CTA phases below the observed
    # ~16-KiB conflict period.  Since Logits is fp32, doubling the effective
    # N range doubles the required phase count.  Use num_keys as the safe
    # upper bound when the caller cannot cheaply provide the effective range.
    if phase_range <= 2048:
        n_tile_phases = 1
    elif phase_range < 16 * 1024:
        n_tile_phases = 4
    elif phase_range < 32 * 1024:
        n_tile_phases = 8
    elif phase_range < 64 * 1024:
        n_tile_phases = 16
    else:
        n_tile_phases = 32

    q_ctas = _ceil_div(num_rows, block_q)
    n_tiles = _ceil_div(phase_range, 128)
    if num_heads == 4:
        max_split_n = _SPLIT_N_CHOICES[-1]
        if q_ctas < cu_count:
            max_split_n = max(
                split
                for split in _SPLIT_N_CHOICES
                if q_ctas * split <= cu_count
            )
        # Wave slots * (N tiles per CTA + startup), plus a launched-CTA
        # term so a sparse leftover round can still split, while a nearly
        # full extra round (H4 8192) does not jump to huge split_n.
        split_n = min(
            (
                split
                for split in _SPLIT_N_CHOICES
                if split <= n_tiles and split <= max_split_n
            ),
            key=lambda split: (
                _ceil_div(q_ctas * split, cu_count)
                * (_ceil_div(n_tiles, split) + _MAX_CTA_STARTUP_N_TILES)
                * cu_count
                + q_ctas * split,
                split,
            ),
        )
    elif q_ctas < cu_count:
        # Prefer a large Q tile, then use split-N to fill (but not exceed)
        # one device-wide CTA round.
        split_n = max(
            split
            for split in _SPLIT_N_CHOICES
            if split <= n_tiles and q_ctas * split <= cu_count
        )
    elif phase_range < 16 * 1024:
        split_n = 1
    elif num_heads == 8 and phase_range >= 32 * 1024:
        split_n = 4
    else:
        split_n = 2
    n_tile_phases = max(1, n_tile_phases // split_n)

    return QSAMQAPrefillConfig(
        block_q=block_q,
        block_n=128,
        threads=768,
        n_tile_phases=n_tile_phases,
        split_n=split_n,
        rows_divisible=num_rows % block_q == 0,
        heads=num_heads,
    )
