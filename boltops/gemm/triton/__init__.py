# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton backend for the GEMM family."""

from boltops.gemm.triton.gemm_a16w16 import gemm_a16w16
from boltops.gemm.triton.matmul_persistent import matmul_persistent

__all__ = [
    "gemm_a16w16",
    "matmul_persistent",
]
