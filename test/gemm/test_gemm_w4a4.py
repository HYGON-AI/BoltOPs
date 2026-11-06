# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Smoke / skip-gated tests for ``gemm_w4a4``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_w4a4  # noqa: E402
from boltops.utility import arch_info  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
FP4_REQUIRED = pytest.mark.skipif(
    not arch_info.is_fp4_avail(), reason="MXFP4 not available on this GPU"
)


@GPU_REQUIRED
@FP4_REQUIRED
def test_gemm_w4a4_runs():
    """Shape smoke: full numeric golden needs host FP4 pack helpers."""
    M, N, K = 32, 64, 64
    x = torch.zeros((M, K // 2), dtype=torch.uint8, device=DEVICE)
    w = torch.zeros((N, K // 2), dtype=torch.uint8, device=DEVICE)
    x_scales = torch.zeros((M, K // 32), dtype=torch.uint8, device=DEVICE)
    w_scales = torch.zeros((N, K // 32), dtype=torch.uint8, device=DEVICE)
    y = gemm_w4a4(x, w, x_scales, w_scales, dtype=torch.bfloat16)
    assert y.shape == (M, N)
