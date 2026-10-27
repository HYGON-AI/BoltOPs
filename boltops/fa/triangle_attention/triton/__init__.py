# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton backend for triangle attention."""

from .triangle_attention import triangle_attention_fwd

__all__ = ["triangle_attention_fwd"]
