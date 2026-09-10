"""Qwen Sparse-GQA operators for Hygon DCU."""

from .triton.compact import qwen_sparse_kv_extraction_compact_hcu_fa_triton
from .triton.packed_decode import sparse_gqa_packed_decode_triton


__all__ = [
    "qwen_sparse_kv_extraction_compact_hcu_fa_triton",
    "sparse_gqa_packed_decode_triton",
]
