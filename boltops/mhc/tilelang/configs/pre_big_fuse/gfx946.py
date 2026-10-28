# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""gfx946 launch policy for ``mhc_pre_big_fuse``.

Common kernels target three resident CTAs per CU.  Persistent kernels target
one power-of-two CU wave for up to 256 tokens and three waves for larger
shapes, while keeping token tiles large and split reduction bounded.
"""

from __future__ import annotations

from functools import lru_cache

from .common import PreBigFuseConfig

_TOKEN_BLOCKS = (128, 64, 32)
_COMMON_SPLITS = (1, 2, 4, 8, 16, 32, 64)
_PERSISTENT_SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)
_LDS_PER_CU = 128 * 1024


def _lds_bytes(token_block: int, hidden_block: int, overlap: bool) -> int:
    x_bytes = token_block * hidden_block * 2
    fn_bytes = 32 * hidden_block * 4
    out_bytes = token_block * 32 * 4
    loop_bytes = x_bytes + fn_bytes if overlap else max(x_bytes, fn_bytes)
    return max(loop_bytes, out_bytes)


def _lds_schedule(
    token_block: int, hidden_block: int, occupancy: int
) -> tuple[bool, int]:
    budget = _LDS_PER_CU // occupancy
    overlap_bytes = _lds_bytes(token_block, hidden_block, True)
    if overlap_bytes <= budget:
        return True, overlap_bytes
    return False, _lds_bytes(token_block, hidden_block, False)


def _score_common_candidate(
    num_tokens: int,
    token_block: int,
    split_k: int,
    cu_count: int,
) -> tuple[int, int, int, int]:
    """Rank common kernels by occupancy-3 residency-slot coverage."""
    token_tiles = (num_tokens + token_block - 1) // token_block
    total_ctas = token_tiles * split_k
    unused_resident_slots = 3 * cu_count - total_ctas
    token_padding = token_tiles * token_block - num_tokens
    return (
        unused_resident_slots,
        token_padding,
        -token_block,
        split_k,
    )


def _get_common_config(
    num_tokens: int, mhc_hidden_size: int, cu_count: int
) -> PreBigFuseConfig:
    hidden_block = 64
    hidden_loop = mhc_hidden_size // hidden_block
    candidates = []
    for token_block in _TOKEN_BLOCKS:
        separate, lds_bytes = _lds_schedule(token_block, hidden_block, 3)
        if not separate or lds_bytes > _LDS_PER_CU // 3:
            continue
        for split_k in _COMMON_SPLITS:
            if split_k > hidden_loop or hidden_loop % split_k:
                continue
            total_ctas = (num_tokens + token_block - 1) // token_block * split_k
            rounds = (total_ctas + cu_count - 1) // cu_count
            if rounds > 3:
                continue
            score = _score_common_candidate(num_tokens, token_block, split_k, cu_count)
            candidates.append((score, token_block, split_k))

    if not candidates:
        return PreBigFuseConfig(
            token_block=32,
            hidden_block=hidden_block,
            hidden_loop=hidden_loop,
            n_splits_pre=1,
            use_small_token_splitk=False,
            target_occupancy=3,
            separate_x_fn_smem=True,
        )

    _, token_block, split_k = min(candidates, key=lambda item: item[0])
    return PreBigFuseConfig(
        token_block=token_block,
        hidden_block=hidden_block,
        hidden_loop=hidden_loop,
        n_splits_pre=split_k,
        use_small_token_splitk=split_k > 1,
        target_occupancy=3,
        separate_x_fn_smem=True,
    )


def _score_persistent_candidate(
    *,
    token_block: int,
    split_k: int,
    total_tiles: int,
    cu_count: int,
    token_padding: int,
) -> tuple[int, int, int, int, int]:
    """Rank persistent kernels by CU utilization within the round limit."""
    rounds = (total_tiles + cu_count - 1) // cu_count
    idle_slots = rounds * cu_count - total_tiles
    return (
        idle_slots,
        -rounds,
        token_padding,
        -token_block,
        split_k,
    )


def _get_persistent_config(
    num_tokens: int, mhc_hidden_size: int, cu_count: int
) -> PreBigFuseConfig:
    hidden_block = 64
    hidden_loop = mhc_hidden_size // hidden_block
    small_shape = num_tokens <= 256
    min_rounds = (
        (num_tokens + _TOKEN_BLOCKS[0] - 1) // _TOKEN_BLOCKS[0] + cu_count - 1
    ) // cu_count
    round_limit = 1 if small_shape else max(3, min_rounds)
    candidates = []

    for token_block in _TOKEN_BLOCKS:
        token_blocks = (num_tokens + token_block - 1) // token_block
        for split_k in _PERSISTENT_SPLITS:
            if (
                (small_shape and split_k > 16)
                or mhc_hidden_size % split_k
                or (mhc_hidden_size // split_k) % (2 * hidden_block)
            ):
                continue
            total_tiles = token_blocks * split_k
            rounds = (total_tiles + cu_count - 1) // cu_count
            if rounds > round_limit:
                continue
            token_padding = token_blocks * token_block - num_tokens
            score = _score_persistent_candidate(
                token_block=token_block,
                split_k=split_k,
                total_tiles=total_tiles,
                cu_count=cu_count,
                token_padding=token_padding,
            )
            candidates.append((score, token_block, split_k))

    if not candidates:
        raise ValueError(
            "No valid gfx946 persistent split-K configuration for "
            f"num_tokens={num_tokens}, mhc_hidden_size={mhc_hidden_size}, "
            f"cu_count={cu_count}"
        )

    _, token_block, split_k = min(candidates, key=lambda item: item[0])
    return PreBigFuseConfig(
        token_block=token_block,
        hidden_block=hidden_block,
        hidden_loop=hidden_loop,
        n_splits_pre=split_k,
        use_small_token_splitk=True,
        target_occupancy=1,
        separate_x_fn_smem=False,
    )


@lru_cache(maxsize=1024)
def get_config(
    num_tokens: int, mhc_hidden_size: int, cu_count: int
) -> PreBigFuseConfig:
    if num_tokens <= 128:
        return _get_common_config(num_tokens, mhc_hidden_size, cu_count)
    return _get_persistent_config(num_tokens, mhc_hidden_size, cu_count)
