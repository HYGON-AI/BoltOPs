# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Default launch policy for ``ihc_pre`` (shared GEMM tile/split heuristic)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from boltops.utility.arch_info import get_device_info, get_lds_capacity_bytes

# Logical IHC N is 8 (pre) or 4 (head). Do not pad N to 32.
BLOCK_N = 16

_GEMM_THREADS = 256
_THREADS_PER_CU = 2048

_HIDDEN_BLOCK = 64
_MAX_SPLIT_K = 128
# Prefer at least 2 hb iters in the GEMM K-loop.
_MIN_K_ITERS = 2
_MAX_WAVES = 4
# Inclusive: T<=64 uses underfill / CU-coverage scoring (tb=32 allowed).
_SMALL_TOKEN_LIMIT = 64
# Inclusive: T>=4096 uses large-tile path (tb/hb=128, sep=False).
_LARGE_TOKEN_LIMIT = 4096


@dataclass(frozen=True)
class PreConfig:
    token_block: int
    hidden_block: int
    split_k: int
    separate_x_fn_smem: bool = True
    est_occupancy: int = 1


def _lds_bytes(token_block: int, hidden_block: int, separate_x_fn_smem: bool) -> int:
    """Live shared bytes in the GEMM K-loop (see ``gemm.py``)."""
    x_bytes = token_block * hidden_block * 2  # bf16
    fn_bytes = BLOCK_N * hidden_block * 4  # fp32
    if separate_x_fn_smem:
        return x_bytes + fn_bytes
    return max(x_bytes, fn_bytes)


def _est_occupancy(
    token_block: int,
    hidden_block: int,
    separate_x_fn_smem: bool,
    lds_cap: int,
) -> int:
    lds = _lds_bytes(token_block, hidden_block, separate_x_fn_smem)
    occ_lds = max(1, lds_cap // lds) if lds > 0 else 1
    occ_thr = max(1, _THREADS_PER_CU // _GEMM_THREADS)
    return max(1, min(occ_lds, occ_thr))


def _legal_split(k_total: int, split_k: int, hidden_block: int) -> bool:
    if split_k < 1 or k_total % split_k != 0:
        return False
    split_size = k_total // split_k
    if split_size % hidden_block != 0:
        return False
    return (split_size // hidden_block) >= _MIN_K_ITERS


def _legal_splits(k_total: int, hidden_block: int) -> tuple[int, ...]:
    """Legal ``split_k`` divisors of ``k_total`` (incl. non-pow2), capped at ``_MAX_SPLIT_K``."""
    if k_total <= 0:
        return (1,)
    out: list[int] = []
    i = 1
    while i * i <= k_total:
        if k_total % i == 0:
            for sk in (i, k_total // i):
                if sk <= _MAX_SPLIT_K and _legal_split(k_total, sk, hidden_block):
                    out.append(sk)
        i += 1
    return tuple(sorted(set(out)))


def _token_blocks_for(num_tokens: int) -> tuple[int, ...]:
    if num_tokens <= 32:
        return (32,)
    if num_tokens <= _SMALL_TOKEN_LIMIT:
        return (32, 64)
    return (64,)


def _score_candidate(
    num_tokens: int,
    token_block: int,
    split_k: int,
    cu_count: int,
    occ: int,
) -> tuple[int, ...]:
    """Lower is better."""
    token_loop = (num_tokens + token_block - 1) // token_block
    grid = token_loop * split_k
    resident = max(1, occ * cu_count)
    waves = (grid + resident - 1) // resident
    token_pad = token_loop * token_block - num_tokens

    if num_tokens <= _SMALL_TOKEN_LIMIT:
        unused_cus = max(0, cu_count - grid)
        return (0, unused_cus, -occ, token_block, split_k, token_pad)

    waste = waves * resident - grid
    if 2 <= waves <= 3:
        wave_band = 0
    elif waves == 1:
        wave_band = 1
    else:
        wave_band = 2 + abs(waves - 3)
    return (1, waste, wave_band, split_k, token_pad)


def _large_token_split_k(num_tokens: int, token_block: int, cu_count: int) -> int:
    """Split-k schedule for the large-tile path (``tb=128``).

    ``split_k=1`` is valid: GEMM launches a 1-wide split grid and the epilogue
    takes the dedicated ``n_splits == 1`` combine path.
    """
    token_loop = (num_tokens + token_block - 1) // token_block
    if token_loop <= 2:
        return 64
    if token_loop <= 4:
        return 32
    if token_loop <= cu_count // 8:
        return 16
    if token_loop <= cu_count * 0.75:
        return 8
    if token_loop <= cu_count * 2:
        return 4
    return 1


def _pick_large_token_config(
    num_tokens: int,
    k_total: int,
    cu_count: int,
    lds_cap: int,
) -> PreConfig:
    """T>=4096: large tiles ``tb/hb=128``, merged smem, low ``split_k``."""
    token_block = 128
    prefer_hb = 128 if k_total % 128 == 0 else 64
    target_sk = _large_token_split_k(num_tokens, token_block, cu_count)

    def try_cfg(hb: int, sk: int, sep: bool) -> PreConfig | None:
        if _lds_bytes(token_block, hb, sep) > lds_cap:
            return None
        if not _legal_split(k_total, sk, hb):
            return None
        return PreConfig(
            token_block=token_block,
            hidden_block=hb,
            split_k=sk,
            separate_x_fn_smem=sep,
            est_occupancy=_est_occupancy(token_block, hb, sep, lds_cap),
        )

    # Prefer merged smem so hb=128 stays at occ>=2.
    for sep in (False, True):
        cfg = try_cfg(prefer_hb, target_sk, sep)
        if cfg is not None:
            return cfg
        if prefer_hb != 64:
            cfg = try_cfg(64, target_sk, sep)
            if cfg is not None:
                return cfg
        legals = _legal_splits(k_total, prefer_hb)
        if legals:
            sk = min(legals, key=lambda s: (abs(s - target_sk), s))
            cfg = try_cfg(prefer_hb, sk, sep)
            if cfg is not None:
                return cfg
        if prefer_hb != 64:
            legals = _legal_splits(k_total, 64)
            if legals:
                sk = min(legals, key=lambda s: (abs(s - target_sk), s))
                cfg = try_cfg(64, sk, sep)
                if cfg is not None:
                    return cfg

    return PreConfig(
        token_block=64,
        hidden_block=64 if k_total % 64 == 0 else 32,
        split_k=1,
        separate_x_fn_smem=True,
        est_occupancy=_est_occupancy(64, 64, True, lds_cap),
    )


@lru_cache(maxsize=1024)
def get_config(
    num_tokens: int,
    hc_hidden_size: int,
    cu_count: int | None = None,
) -> PreConfig:
    """Pick token tile and split_k from LDS occupancy × CU rounds.

    ``hc_hidden_size`` is ``k_total = hc * hidden``.

    - ``T <= 64``: maximize CU coverage (underfill).
    - ``64 < T < 4096``: ``tb=64``, ``sep=True``, minimize packing waste.
    - ``T >= 4096``: ``tb/hb=128``, ``sep=False``, low ``split_k``.
    """
    info = get_device_info()
    if cu_count is None:
        cu_count = info.cu_count
    lds_cap = get_lds_capacity_bytes(info.arch)

    if num_tokens >= _LARGE_TOKEN_LIMIT:
        return _pick_large_token_config(
            num_tokens, hc_hidden_size, cu_count, lds_cap
        )

    hidden_block = _HIDDEN_BLOCK
    if hc_hidden_size % hidden_block != 0:
        hidden_block = 128 if hc_hidden_size % 128 == 0 else 32

    separate = True
    split_candidates = _legal_splits(hc_hidden_size, hidden_block)
    candidates: list[tuple[tuple[int, ...], PreConfig]] = []
    for token_block in _token_blocks_for(num_tokens):
        if _lds_bytes(token_block, hidden_block, separate) > lds_cap:
            continue
        occ = _est_occupancy(token_block, hidden_block, separate, lds_cap)
        for split_k in split_candidates:
            token_loop = (num_tokens + token_block - 1) // token_block
            grid = token_loop * split_k
            resident = max(1, occ * cu_count)
            waves = (grid + resident - 1) // resident
            if waves > _MAX_WAVES:
                continue
            score = _score_candidate(
                num_tokens, token_block, split_k, cu_count, occ
            )
            cfg = PreConfig(
                token_block=token_block,
                hidden_block=hidden_block,
                split_k=split_k,
                separate_x_fn_smem=separate,
                est_occupancy=occ,
            )
            candidates.append((score, cfg))

    if not candidates:
        return PreConfig(
            token_block=32 if num_tokens <= 64 else 64,
            hidden_block=hidden_block,
            split_k=1,
            separate_x_fn_smem=True,
            est_occupancy=_est_occupancy(64, hidden_block, True, lds_cap),
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
