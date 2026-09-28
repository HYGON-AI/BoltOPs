"""Flash-attention operator families."""

from typing import Any

from .extend_attention import (
    build_unified_kv_indices,
    can_use_dense_prefill_fp8,
    dense_prefill_attention_fwd,
    extend_attention_fwd,
    extend_attention_fwd_unified,
)
from .unified_attention import reshape_and_cache_int4, unified_attention


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
    "extend_attention_fwd",
    "extend_attention_fwd_unified",
    "KVQuantMode",
    "reshape_and_cache_int4",
    "unified_attention",
]
