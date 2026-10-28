# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Smoke / skip-gated tests for ``batched_gemm_a16wfp4``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import batched_gemm_a16wfp4  # noqa: E402
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
def test_batched_gemm_a16wfp4_runs():
    B, M, N, K = 2, 32, 64, 64
    x = torch.randn((B, M, K), dtype=torch.bfloat16, device=DEVICE)
    w = torch.zeros((B, N, K // 2), dtype=torch.uint8, device=DEVICE)
    w_scales = torch.zeros((B, N, K // 32), dtype=torch.uint8, device=DEVICE)
    y = batched_gemm_a16wfp4(x, w, w_scales, dtype=torch.bfloat16)
    assert y.shape == (B, M, N)
