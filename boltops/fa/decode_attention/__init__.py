# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Decode attention adapters over existing shared kernels."""

from __future__ import annotations


def decode_attention_fwd_4d(
    q, k, v, *, sm_scale, k_scale, v_scale,
    block_table=None, seq_lens=None, num_kv_splits=None,
):
    """Return single-query BF16 MHA output over logical or paged FP8 KV.

    Current contract: Q [B,H,1,128], logical K/V [B,H,S,128] or paged
    [num_pages,page_size,H,128]. Scales are explicit; this is forward-only.
    The backend reuses boltops.mla.decode_attention_fwd_normal's MHA path.
    """
    from .triton.decode_attention import decode_attention_fwd_4d as impl

    return impl(
        q, k, v, sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale,
        block_table=block_table, seq_lens=seq_lens, num_kv_splits=num_kv_splits,
    )


__all__ = ["decode_attention_fwd_4d"]
