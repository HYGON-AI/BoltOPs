# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BN-tile compatibility entry for FP8 MQA logits.

Forwards to ``_fp8_mqa_logits_bn_tile_kernel`` via ``fp8_mqa_logits`` with
shape-based ``get_bn_tile_dispatch``. Prefer public
``boltops.mqa_logits.fp8_mqa_logits(..., backend="bn"|"auto")``.
"""

from __future__ import annotations

from typing import Optional

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits import fp8_mqa_logits
from boltops.mqa_logits.triton.mqa_config_utils import (
    get_bn_tile_dispatch,
    is_large_prefill,
)


def fp8_mqa_logits_tiled(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    schedule_hint: Optional[str] = None,
    use_block_pingpong: Optional[bool] = None,
    force_row_kernel: bool = False,
    assume_ks_zero: bool = False,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FP8 MQA logits via BN-tile (or row when ``force_row_kernel``)."""
    if force_row_kernel:
        return fp8_mqa_logits(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            block_kv=block_m if block_m is not None else None,
            num_warps=num_warps,
            num_stages=num_stages,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            kpack=kpack,
            schedule_hint=schedule_hint,
            use_block_pingpong=use_block_pingpong,
            assume_ks_zero=assume_ks_zero,
            D_out=D_out,
            auto_bn_tile=False,
        )

    seq_len, num_heads, _ = Q.shape
    seq_len_kv = KV.shape[0]
    disp = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)

    bn = int(block_n) if block_n is not None else int(disp["bn_tile"])
    bm = int(block_m) if block_m is not None else int(disp["block_m_tile"])
    if block_n is not None or block_m is not None:
        sk = 1 if is_large_prefill(seq_len, seq_len_kv) else int(disp["head_split_k"])
    else:
        sk = int(disp["head_split_k"])

    def _pick(name: str, override):
        if override is not None:
            return override
        return disp[name]

    return fp8_mqa_logits(
        Q,
        KV,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        num_warps=_pick("num_warps", num_warps),
        num_stages=_pick("num_stages", num_stages),
        waves_per_eu=_pick("waves_per_eu", waves_per_eu),
        matrix_instr_nonkdim=_pick("matrix_instr_nonkdim", matrix_instr_nonkdim),
        kpack=_pick("kpack", kpack),
        schedule_hint=_pick("schedule_hint", schedule_hint),
        use_block_pingpong=_pick("use_block_pingpong", use_block_pingpong),
        assume_ks_zero=assume_ks_zero,
        D_out=D_out,
        head_split_k=sk,
        bn_tile=bn,
        block_m_tile=bm,
        auto_bn_tile=False,
    )
