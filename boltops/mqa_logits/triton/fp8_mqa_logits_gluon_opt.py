# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Optimal FP8 Gluon MQA logits entry.

Ablations (WPP / QPP / ship vs BN layouts / Q-mask / epilogue) showed that the
**original Gluon kernel already is the winning combination**:

* ship MFMA / Q / shared layouts
* continuous Q LDS double-buffer (QPP)
* W LDS ping-pong (WPP)
* unmasked Q loads + store-layout epilogue

The bn_align kernel can approximate this with
``qpp+wpp+match_ship_layout+no_q_mask+ship_epilogue``, but still trails original
Gluon by ~1–6% under the same launch knobs (extra branchy IR). So ``gluon_opt``
delegates to ``fp8_mqa_logits_gluon(..., force_gluon_kernel=True)``.

Launch: ``fp8_mqa_logits_gluon_opt(...)`` or ``backend="gluon_opt"``.
"""

from __future__ import annotations

from typing import Optional

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
    fp8_mqa_logits_gluon,
    gluon_prefill_available,
)


def fp8_mqa_logits_gluon_opt(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    schedule_hint: Optional[str] = None,
    use_block_pingpong: Optional[bool] = None,
    assume_ks_zero: bool = False,
    kv_tiles_n: Optional[int] = None,
    w_group: Optional[int] = None,
    head_split_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Optimal FP8 Gluon path (= original Gluon forced on)."""
    if not gluon_prefill_available():
        raise RuntimeError("gluon_opt requires triton.experimental.gluon")

    return fp8_mqa_logits_gluon(
        Q,
        KV,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
        kpack=kpack,
        schedule_hint=schedule_hint,
        use_block_pingpong=use_block_pingpong,
        assume_ks_zero=assume_ks_zero,
        kv_tiles_n=kv_tiles_n,
        w_group=w_group,
        head_split_k=head_split_k,
        num_warps=num_warps,
        force_gluon_kernel=True,
        D_out=D_out,
    )
