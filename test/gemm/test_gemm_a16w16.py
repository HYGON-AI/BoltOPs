"""Correctness and optional performance tests for ``gemm_a16w16``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_a16w16  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.parametrize("M,N,K", [(64, 128, 64), (128, 256, 128)])
def test_gemm_a16w16_correctness(M: int, N: int, K: int):
    """``Y = X @ W^T`` with ``W[N,K]`` — same as vLLM ROCm / aiter."""
    torch.manual_seed(0)
    x_cpu = torch.randn((M, K), device="cpu", dtype=torch.float16)
    w_cpu = torch.randn((N, K), device="cpu", dtype=torch.float16)
    golden = torch.matmul(x_cpu, w_cpu.T)
    out = gemm_a16w16(
        x_cpu.to(DEVICE),
        w_cpu.to(DEVICE),
        dtype=torch.float16,
        backend="triton",
    )
    _assert_close(out, golden)


@GPU_REQUIRED
def test_gemm_a16w16_bias():
    torch.manual_seed(1)
    M, N, K = 64, 128, 64
    x_cpu = torch.randn((M, K), device="cpu", dtype=torch.float16)
    w_cpu = torch.randn((N, K), device="cpu", dtype=torch.float16)
    bias_cpu = torch.randn((N,), device="cpu", dtype=torch.float16)
    golden = torch.matmul(x_cpu, w_cpu.T) + bias_cpu
    out = gemm_a16w16(
        x_cpu.to(DEVICE),
        w_cpu.to(DEVICE),
        bias_cpu.to(DEVICE),
        dtype=torch.float16,
        backend="triton",
    )
    _assert_close(out, golden)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("M,N,K", [(512, 512, 512)])
def test_bench_gemm_a16w16(
    M: int, N: int, K: int, time_ms, bench_warmup: int, bench_repeat: int
):
    torch.manual_seed(0)
    x = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    w = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    gemm_a16w16(x, w, dtype=torch.float16, backend="triton")
    torch.cuda.synchronize()
    ms = time_ms(
        lambda: gemm_a16w16(x, w, dtype=torch.float16, backend="triton"),
        bench_warmup,
        bench_repeat,
    )
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    print(f"[bench] gemm_a16w16 {M}x{N}x{K}: {ms:.3f} ms, {tflops:.2f} TFLOPS")
