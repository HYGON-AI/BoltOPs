"""Triton backend for MQA logits (prefill)."""

from boltops.mqa_logits.triton.fp8_mqa_logits import fp8_mqa_logits

__all__ = [
    "fp8_mqa_logits",
]
