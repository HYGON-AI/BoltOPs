"""Correctness tests for ``w8a8_triton_block_scaled_mm``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import w8a8_triton_block_scaled_mm  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
FP8 = getattr(torch, "float8_e4m3fn", None)
FP8_REQUIRED = pytest.mark.skipif(FP8 is None, reason="float8_e4m3fn not available")


def _block_scale_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    """Match kernel: ``dot(a,b) * a_s * b_s`` per tile (B is ``[N,K]``)."""
    m, k = a.shape
    n, _ = b.shape
    a_f = a.float()
    b_f = b.float()
    out = torch.zeros((m, n), dtype=torch.float32)
    for i in range(0, k, block_k):
        ke = min(i + block_k, k)
        aks = i // block_k
        a_blk = a_f[:, i:ke]
        for j in range(0, n, block_n):
            ne = min(j + block_n, n)
            bns = j // block_n
            b_blk = b_f[j:ne, i:ke]
            out[:, j:ne] += (a_blk @ b_blk.T) * a_s[:, aks : aks + 1] * b_s[
                bns : bns + 1, aks : aks + 1
            ]
    return out


@GPU_REQUIRED
@FP8_REQUIRED
@pytest.mark.parametrize("M,N,K", [(64, 128, 128)])
def test_w8a8_triton_block_scaled_mm_correctness(M: int, N: int, K: int):
    torch.manual_seed(0)
    block_n, block_k = 128, 128
    a_f = (torch.randn((M, K), dtype=torch.float32) * 0.5).clamp(-400, 400)
    b_f = (torch.randn((N, K), dtype=torch.float32) * 0.5).clamp(-400, 400)
    a = a_f.to(FP8)
    b = b_f.to(FP8)
    a_s = torch.rand((M, K // block_k), dtype=torch.float32) * 0.1 + 0.01
    b_s = torch.rand((N // block_n, K // block_k), dtype=torch.float32) * 0.1 + 0.01
    golden = _block_scale_golden(a, b, a_s, b_s, block_n, block_k)

    out = w8a8_triton_block_scaled_mm(
        a.contiguous().to(DEVICE),
        b.contiguous().to(DEVICE),
        a_s.to(DEVICE),
        b_s.to(DEVICE),
        [block_n, block_k],
        output_dtype=torch.float16,
    )
    torch.testing.assert_close(out.cpu().float(), golden, atol=5e-2, rtol=5e-2)
