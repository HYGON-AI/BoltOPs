"""Correctness tests for ``batched_gemm_bf16`` (aiter Triton batched A16W16)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from boltops.gemm import batched_gemm_bf16  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

# Small CI shapes (B > 1 covered). Keep off aiter's huge get_x_vals grid.
_SHAPES = [
    (2, 32, 64, 64),
    (4, 64, 128, 64),
    (2, 16, 128, 32),
]


def _golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, M, _ = x.shape
    N = weight.shape[1]
    out = torch.empty(B, M, N, dtype=torch.float32, device=x.device)
    for b in range(B):
        b_out = F.linear(x[b].float(), weight[b].float())
        if bias is not None:
            b_out = b_out + bias[b].float()
        out[b] = b_out
    return out.to(dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("B,M,N,K", _SHAPES)
@pytest.mark.parametrize("prealloc", [False, True])
def test_batched_gemm_bf16_correctness(
    dtype: torch.dtype,
    use_bias: bool,
    B: int,
    M: int,
    N: int,
    K: int,
    prealloc: bool,
):
    torch.manual_seed(0)
    x = torch.randint(-20, 20, (B, M, K), dtype=dtype, device=DEVICE)
    w = torch.randint(-20, 20, (B, N, K), dtype=dtype, device=DEVICE)
    bias = (
        (torch.rand((B, 1, N), dtype=dtype, device=DEVICE) * 10)
        if use_bias
        else None
    )
    y = torch.empty((B, M, N), dtype=dtype, device=DEVICE) if prealloc else None

    # Fallback tiles can be large; pin a small launch config for CI.
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    out = batched_gemm_bf16(x, w, bias, dtype=dtype, y=y, config=config)
    golden = _golden(x, w, bias, dtype)
    assert out.shape == (B, M, N)
    _assert_close(out, golden)


@GPU_REQUIRED
def test_batched_gemm_bf16_splitk_rejected():
    x = torch.randn((2, 16, 32), dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn((2, 64, 32), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(AssertionError, match="splitK"):
        batched_gemm_bf16(x, w, dtype=torch.bfloat16, splitK=2)
