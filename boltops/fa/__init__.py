# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Flash-attention operator families."""

from typing import Any

from .extend_attention import (
    build_unified_kv_indices,
    can_use_dense_prefill_fp8,
    dense_prefill_attention_fwd,
    dense_prefill_attention_fwd_4d,
    dense_prefill_attention_fwd_segmented,
    extend_attention_fwd,
    extend_attention_fwd_unified,
)
from .unified_attention import reshape_and_cache_int4, unified_attention
from .triangle_attention import triangle_attention_fwd
from .decode_attention import decode_attention_fwd_4d


def __getattr__(name: str) -> Any:
    """Load UnifiedAttention configuration types without eager Triton imports."""
    if name == "KVQuantMode":
        from .unified_attention import KVQuantMode

        globals()[name] = KVQuantMode
        return KVQuantMode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "build_unified_kv_indices",
    "can_use_dense_prefill_fp8",
    "dense_prefill_attention_fwd",
    "dense_prefill_attention_fwd_4d",
    "dense_prefill_attention_fwd_segmented",
    "extend_attention_fwd",
    "extend_attention_fwd_unified",
    "KVQuantMode",
    "reshape_and_cache_int4",
    "unified_attention",
    "triangle_attention_fwd",
    "decode_attention_fwd_4d",
]
