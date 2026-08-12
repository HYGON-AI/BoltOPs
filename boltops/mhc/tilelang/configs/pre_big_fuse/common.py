"""Default configuration policy for ``mhc_pre_big_fuse``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class PreBigFuseConfig:
    token_block: int
    hidden_block: int
    hidden_loop: int
    n_splits_pre: int
    use_small_token_splitk: bool
    target_occupancy: int = 1
    separate_x_fn_smem: bool = False


@lru_cache(maxsize=1024)
def get_config(
    num_tokens: int, mhc_hidden_size: int, cu_count: int
) -> PreBigFuseConfig:
    """Select the existing Aiter launch policy for a bucketed token count."""
    token_block = 128
    hidden_block = 128
    hidden_loop = mhc_hidden_size // hidden_block
    token_loop = (num_tokens + token_block - 1) // token_block

    if token_loop <= 2:
        token_block = 128 if num_tokens > 128 else (64 if num_tokens > 64 else 32)
        n_splits_pre = 64
        if hidden_loop % n_splits_pre != 0:
            hidden_block = 64
            hidden_loop = mhc_hidden_size // hidden_block
    elif token_loop <= 4:
        n_splits_pre = 32
    elif token_loop <= cu_count // 8:
        n_splits_pre = 16
    elif token_loop <= cu_count * 0.75:
        n_splits_pre = 8
    elif token_loop <= cu_count * 2:
        n_splits_pre = 4
    else:
        n_splits_pre = 1

    final_token_loop = (num_tokens + token_block - 1) // token_block
    use_small_token_splitk = (
        n_splits_pre > 1
        and final_token_loop <= cu_count * 2
        and hidden_loop > 0
        and hidden_loop % n_splits_pre == 0
    )
    if not use_small_token_splitk:
        token_block = 64
        hidden_block = 128

    return PreBigFuseConfig(
        token_block=token_block,
        hidden_block=hidden_block,
        hidden_loop=hidden_loop,
        n_splits_pre=n_splits_pre,
        use_small_token_splitk=use_small_token_splitk,
    )
