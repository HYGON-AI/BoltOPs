# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for ``gemm_a32w32``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_a32w32  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _make_nn(M: int, N: int, K: int, *, seed: int = 0, bias: bool = False):
    """NN layout golden on CPU: A[M,K] @ B[K,N] [+ bias]."""
    torch.manual_seed(seed)
    a_cpu = torch.randn((M, K), device="cpu", dtype=torch.float32)
    b_cpu = torch.randn((N, K), device="cpu", dtype=torch.float32).transpose(0, 1)
    bias_cpu = torch.randn((N,), device="cpu", dtype=torch.float32) if bias else None
    golden = torch.matmul(a_cpu, b_cpu)
    if bias_cpu is not None:
        golden = golden + bias_cpu
    bias_dev = None if bias_cpu is None else bias_cpu.to(DEVICE)
    return a_cpu.to(DEVICE), b_cpu.to(DEVICE), bias_dev, golden


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 256, 64), (96, 160, 128)])
def test_gemm_a32w32_correctness(M: int, N: int, K: int):
    a, b, _, golden = _make_nn(M, N, K)
    _assert_close(gemm_a32w32(a, b.T), golden)


@GPU_REQUIRED
def test_gemm_a32w32_bias():
    a, b, bias, golden = _make_nn(128, 256, 64, bias=True)
    _assert_close(gemm_a32w32(a, b.T, bias), golden)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1024, 1024, 1024)])
def test_bench_gemm_a32w32(
    M: int, N: int, K: int, time_ms, bench_warmup: int, bench_repeat: int
):
    a, b, _, _ = _make_nn(M, N, K)
    gemm_a32w32(a, b.T)
    torch.cuda.synchronize()
    ms = time_ms(lambda: gemm_a32w32(a, b.T), bench_warmup, bench_repeat)
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    print(f"[bench] gemm_a32w32 {M}x{N}x{K}: {ms:.3f} ms, {tflops:.2f} TFLOPS")
