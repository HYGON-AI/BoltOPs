# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton backend for the MLA family."""

from boltops.mla.triton.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from boltops.mla.triton.num_kv_splits import compute_num_kv_splits
from boltops.mla.triton.flash_mla_sparse import flash_mla_sparse_fwd

__all__ = [
    "compute_num_kv_splits",
    "decode_attention_fwd",
    "decode_attention_fwd_grouped",
    "decode_attention_fwd_normal",
    "flash_mla_sparse_fwd",
]
