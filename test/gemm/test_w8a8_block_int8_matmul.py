# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for ``w8a8_block_int8_matmul``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import w8a8_block_int8_matmul  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _golden(a, b, a_s, b_s, block_n, block_k):
    m, k = a.shape
    n, _ = b.shape
    af, bf = a.float(), b.float()
    out = torch.zeros((m, n), dtype=torch.float32)
    for i in range(0, k, block_k):
        ke = min(i + block_k, k)
        aks = i // block_k
        a_blk = af[:, i:ke]
        for j in range(0, n, block_n):
            ne = min(j + block_n, n)
            bns = j // block_n
            b_blk = bf[j:ne, i:ke]
            out[:, j:ne] += (a_blk @ b_blk.T) * a_s[:, aks : aks + 1] * b_s[
                bns : bns + 1, aks : aks + 1
            ]
    return out


@GPU_REQUIRED
def test_w8a8_block_int8_matmul_correctness():
    torch.manual_seed(0)
    M, N, K = 64, 128, 128
    bn = bk = 128
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    b = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    a_s = torch.rand((M, K // bk), dtype=torch.float32) * 0.05 + 0.01
    b_s = torch.rand((N // bn, K // bk), dtype=torch.float32) * 0.05 + 0.01
    golden = _golden(a, b, a_s, b_s, bn, bk)
    out = w8a8_block_int8_matmul(
        a.contiguous().to(DEVICE),
        b.contiguous().to(DEVICE),
        a_s.to(DEVICE),
        b_s.to(DEVICE),
        [bn, bk],
        output_dtype=torch.float16,
    )
    torch.testing.assert_close(out.cpu().float(), golden, atol=5e-2, rtol=5e-2)
