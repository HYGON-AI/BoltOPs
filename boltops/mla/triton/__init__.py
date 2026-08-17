"""Triton backend for the MLA family."""

from boltops.mla.triton.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from boltops.mla.triton.num_kv_splits import compute_num_kv_splits

__all__ = [
    "compute_num_kv_splits",
    "decode_attention_fwd",
    "decode_attention_fwd_grouped",
    "decode_attention_fwd_normal",
]
