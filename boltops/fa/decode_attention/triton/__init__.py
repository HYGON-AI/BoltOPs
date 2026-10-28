# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton-backed Decode attention adapter."""

from .decode_attention import decode_attention_fwd_4d

__all__ = ["decode_attention_fwd_4d"]
