"""UnifiedAttention operators for paged KV caches."""

from __future__ import annotations

from typing import Any


def unified_attention(
    q, k, v, out,
    cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale,
    seq_threshold_3D=None, num_par_softmax_segments=None,
    softmax_segm_output=None, softmax_segm_max=None,
    softmax_segm_expsum=None, alibi_slopes=None, output_scale=None,
    qq_bias=None, sinks=None, mm_prefix_range=None,
    rswa_prefix_lens=None, rswa_window=None, use_alibi_sqrt=False,
    kv_quant_mode=None, k_scale_cache=None, v_scale_cache=None,
    chunk_lookback=-1, use_td=False,
    mm_prefix_clamp_sliding_window=False,
) -> None:
    """Write paged-KV attention output into ``out``."""
    from .triton.unified_attention import unified_attention as impl

    return impl(
        q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, softcap, q_descale,
        k_descale, v_descale, seq_threshold_3D, num_par_softmax_segments,
        softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
        alibi_slopes, output_scale, qq_bias, sinks, mm_prefix_range,
        rswa_prefix_lens, rswa_window, use_alibi_sqrt, kv_quant_mode,
        k_scale_cache, v_scale_cache, chunk_lookback, use_td,
        mm_prefix_clamp_sliding_window,
    )


def reshape_and_cache_int4(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    *,
    k_scale_cache,
    v_scale_cache,
) -> None:
    """Quantize K/V to INT4 and write them into the paged KV cache."""
    from .triton.int4_per_token_head import reshape_and_cache_int4 as impl

    return impl(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )


def __getattr__(name: str) -> Any:
    """Load public configuration types without making package import eager."""
    if name == "KVQuantMode":
        from .triton.unified_attention import KVQuantMode

        globals()[name] = KVQuantMode
        return KVQuantMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["unified_attention", "KVQuantMode", "reshape_and_cache_int4"]
