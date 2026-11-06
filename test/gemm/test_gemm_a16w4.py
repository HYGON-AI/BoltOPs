# Copyright (c) 2026 Hygon Information Technology Co., Ltd. (modifications).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/tests/kernels/quantization/test_awq_triton.py
"""Correctness tests for ``gemm_a16w4`` (aligned with vLLM test_awq_triton)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_a16w4  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_gemm_a16w4_correctness():
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
    out = gemm_a16w4(x, qweight, scales, qzeros, config=config)
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7]) * 4
    weights = ((qweight.cpu().unsqueeze(-1) >> shifts) & 15).reshape(K, N)
    zeros = ((qzeros.cpu().unsqueeze(-1) >> shifts) & 15).reshape(K // G, N)
    deq = (
        (weights - zeros.repeat_interleave(G, dim=0))
        * scales.cpu().repeat_interleave(G, dim=0)
    ).half()
    golden = torch.matmul(x.cpu().float(), deq.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, atol=1e-1, rtol=1e-1)
