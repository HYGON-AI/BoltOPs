# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""SGLang-compatible ExtendAttention operator family."""

from __future__ import annotations


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    extend_seq_lens_cpu=None,
    identity_kv_indices: bool = False,
):
    """Run SGLang-compatible extend attention into ``o_extend``."""
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale are required; use 1.0 for unscaled KV")
    if skip_prefix and skip_extend:
        raise ValueError("skip_prefix and skip_extend cannot both be True")
    from .triton.extend_attention import extend_attention_fwd as impl

    return impl(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale,
        logit_cap,
        skip_prefix_custom_mask,
        sliding_window_size,
        sinks,
        window_kv_offsets,
        xai_temperature_len,
        lse_extend,
        skip_prefix,
        skip_extend,
        page_size,
        score_mod,
        aux_tensors,
        extend_seq_lens_cpu,
        identity_kv_indices,
    )


def build_unified_kv_indices(
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    bs: int,
):
    """Build the combined prefix/current KV layout used by the unified path."""
    from .triton.extend_attention import build_unified_kv_indices as impl

    return impl(
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        bs,
    )


def extend_attention_fwd_unified(
    q,
    o,
    k_buffer,
    v_buffer,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    max_len_extend,
    custom_mask=None,
    mask_indptr=None,
    sm_scale=None,
    logit_cap=0.0,
    is_causal=True,
    sliding_window_size=-1,
    sinks=None,
    window_start_pos=None,
    xai_temperature_len=-1,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
    """Run the one-stage combined-KV ExtendAttention path."""
    from .triton.extend_attention import extend_attention_fwd_unified as impl

    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale are required; use 1.0 for unscaled KV")
    return impl(
        q, o, k_buffer, v_buffer, k_scale, v_scale, qo_indptr, kv_indptr,
        kv_indices, prefix_lens, max_len_extend, custom_mask, mask_indptr,
        sm_scale, logit_cap, is_causal, sliding_window_size, sinks,
        window_start_pos, xai_temperature_len, page_size, score_mod, aux_tensors,
    )


def can_use_dense_prefill_fp8(q, k, v, *, is_causal: bool, logit_cap: float):
    """Return whether dense prefill may use its narrow FP8 fast path."""
    from .triton.extend_attention import can_use_dense_prefill_fp8 as impl

    return impl(q, k, v, is_causal=is_causal, logit_cap=logit_cap)


def dense_prefill_attention_fwd(
    q,
    k,
    v,
    o,
    qo_indptr,
    kv_indptr,
    max_len_q: int,
    sm_scale=None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    logit_cap: float = 0.0,
    is_causal: bool = True,
    lse=None,
):
    """Run dense variable-length prefill into the caller-owned output."""
    from .triton.extend_attention import dense_prefill_attention_fwd as impl

    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale must be numeric")
    return impl(
        q, k, v, o, qo_indptr, kv_indptr, max_len_q, sm_scale, k_scale,
        v_scale, logit_cap, is_causal, lse,
    )


__all__ = [
    "build_unified_kv_indices",
    "can_use_dense_prefill_fp8",
    "dense_prefill_attention_fwd",
    "extend_attention_fwd",
    "extend_attention_fwd_unified",
]
