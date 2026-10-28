# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/tests/kernels/quantization/test_awq_triton.py
"""Correctness tests for ``awq_gemm_triton`` (aligned with vLLM test_awq_triton)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import awq_gemm_triton  # noqa: E402
from boltops.gemm.triton.awq_gemm_triton import awq_dequantize_triton  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_awq_gemm_triton_correctness():
    torch.manual_seed(0)
    # Keep small on PMD; larger grids have aborted under gfx946 sim.
    M, N, K, G = 16, 64, 64, 64
    x = torch.rand((M, K), device=DEVICE, dtype=torch.float16) / 10
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (K, N // 8), device=DEVICE, dtype=torch.int32
    )
    scales = torch.rand((K // G, N), device=DEVICE, dtype=torch.float16)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K // G, N // 8),
        device=DEVICE,
        dtype=torch.int32,
    )

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
    }
    out = awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters=1, config=config)
    deq = awq_dequantize_triton(qweight, scales, qzeros)
    golden = torch.matmul(x.cpu().float(), deq.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, atol=1e-1, rtol=1e-1)
