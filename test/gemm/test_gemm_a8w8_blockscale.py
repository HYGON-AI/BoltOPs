"""Correctness tests for ``gemm_a8w8_blockscale``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from bolt_ops.gemm import gemm_a8w8_blockscale  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _a8w8_blockscale_golden(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_k: int,
    block_n: int,
) -> torch.Tensor:
    """Match ``dot(x,w.T)*x_s*w_s`` per K/N tile; ``w`` is ``[N,K]``."""
    m, k = x.shape
    n, _ = w.shape
    xf, wf = x.float(), w.float()
    out = torch.zeros((m, n), dtype=torch.float32)
    for i in range(0, k, block_k):
        ke = min(i + block_k, k)
        ks = i // block_k
        x_blk = xf[:, i:ke]
        xs = x_scale[:, ks : ks + 1]
        for j in range(0, n, block_n):
            ne = min(j + block_n, n)
            ns = j // block_n
            w_blk = wf[j:ne, i:ke]
            ws = w_scale[ns : ns + 1, ks : ks + 1]
            out[:, j:ne] += (x_blk @ w_blk.T) * xs * ws
    return out


@GPU_REQUIRED
def test_gemm_a8w8_blockscale_correctness():
    torch.manual_seed(0)
    M, N, K = 64, 128, 128
    block_k = block_n = 128
    x = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    x_scale = torch.rand((M, K // block_k), dtype=torch.float32) * 0.05 + 0.01
    w_scale = torch.rand((N // block_n, K // block_k), dtype=torch.float32) * 0.05 + 0.01
    golden = _a8w8_blockscale_golden(x, w, x_scale, w_scale, block_k, block_n)

    config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "NUM_KSPLIT": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    out = gemm_a8w8_blockscale(
        x.to(DEVICE),
        w.to(DEVICE),
        x_scale.to(DEVICE),
        w_scale.to(DEVICE),
        dtype=torch.float16,
        config=config,
        backend="triton",
    )
    torch.testing.assert_close(out.cpu().float(), golden, atol=5e-2, rtol=5e-2)
