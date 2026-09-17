"""Triton backend for MQA logits (prefill).

Public callers should use ``boltops.mqa_logits.triton_mqa_logits``. Low-level
float8 row/BN kernels live in ``fp8_mqa_logits.py`` and are used internally.
"""

from boltops.mqa_logits import triton_mqa_logits

__all__ = [
    "triton_mqa_logits",
]
