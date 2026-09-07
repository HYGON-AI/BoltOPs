"""MLA operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.

Public APIs (aligned with vLLM ``TRITON_MLA`` / ``decode_attention_fwd``):

* docs: ``docs/ops/mla.md``
* tests: ``test/mla/``

Public operators:

* ``flash_mla_sparse_fwd``
* ``decode_attention_fwd``
* ``decode_attention_fwd_grouped``
* ``decode_attention_fwd_normal``
* ``compute_num_kv_splits``
"""

from __future__ import annotations

from typing import Optional

import torch


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashMLA-compatible sparse forward operator."""
    from boltops.mla.triton.flash_mla_sparse import flash_mla_sparse_fwd as impl

    return impl(q, kv, indices, softmax_scale, d_v, attn_sink, topk_length)


def decode_attention_fwd(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int = 1,
    logit_cap: float = 0.0,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    is_mla: bool = False,
) -> None:
    """vLLM Triton paged decode attention (MHA / GQA / MLA)."""
    from boltops.mla.triton.decode_attention import decode_attention_fwd as impl

    return impl(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size=page_size,
        logit_cap=logit_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        is_mla=is_mla,
    )


def decode_attention_fwd_grouped(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    is_mla: bool = False,
) -> None:
    """Grouped (GQA/MQA/MLA) path of ``decode_attention_fwd``."""
    from boltops.mla.triton.decode_attention import (
        decode_attention_fwd_grouped as impl,
    )

    return impl(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap=logit_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        is_mla=is_mla,
    )


def decode_attention_fwd_normal(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    """MHA path of ``decode_attention_fwd``."""
    from boltops.mla.triton.decode_attention import decode_attention_fwd_normal as impl

    return impl(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap=logit_cap,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def compute_num_kv_splits(
    max_seq_len: int,
    sm_count: Optional[int] = None,
) -> int:
    """vLLM ``TRITON_MLA`` split-KV count heuristic."""
    from boltops.mla.triton.num_kv_splits import compute_num_kv_splits as impl

    return impl(max_seq_len, sm_count)


__all__ = [
    "compute_num_kv_splits",
    "decode_attention_fwd",
    "decode_attention_fwd_grouped",
    "decode_attention_fwd_normal",
    "flash_mla_sparse_fwd",
]
