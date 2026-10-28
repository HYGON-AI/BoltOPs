# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for ``batched_gemm_bf16`` (aiter Triton batched A16W16)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")

from boltops.gemm import batched_gemm_bf16  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

# Small CI shapes (B > 1 covered). Keep off aiter's huge get_x_vals grid.
_SHAPES = [
    (2, 32, 96, 64),
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
@pytest.mark.parametrize("layout", ["tn", "tt"])
def test_batched_gemm_bf16_correctness(
    dtype: torch.dtype,
    use_bias: bool,
    B: int,
    M: int,
    N: int,
    K: int,
    prealloc: bool,
    layout: str,
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
    stored_w = (
        w
        if layout == "tn"
        else w.transpose(1, 2).contiguous().transpose(1, 2)
    )
    out = batched_gemm_bf16(
        x, stored_w, bias, dtype=dtype, y=y, config=config
    )
    golden = _golden(x, w, bias, dtype)
    assert out.shape == (B, M, N)
    _assert_close(out, golden)


@GPU_REQUIRED
def test_batched_gemm_bf16_splitk_rejected():
    x = torch.randn((2, 16, 32), dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn((2, 64, 32), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(AssertionError, match="splitK"):
        batched_gemm_bf16(x, w, dtype=torch.bfloat16, splitK=2)


@GPU_REQUIRED
def test_batched_gemm_bf16_batch_last_grid():
    """The rocBLAS-aligned grid order preserves batched GEMM semantics."""
    B, M, N, K = 3, 17, 48, 32
    x = torch.randn((B, M, K), dtype=torch.float16, device=DEVICE)
    w_kn = torch.randn((B, K, N), dtype=torch.float16, device=DEVICE)
    w = w_kn.transpose(1, 2)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "BATCH_LAST_GRID": True,
    }
    out = batched_gemm_bf16(x, w, dtype=torch.float16, config=config)
    torch.testing.assert_close(out, torch.bmm(x, w_kn), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Performance benchmarks (real-model sizes, shipped tuned configs)
# ---------------------------------------------------------------------------

_BENCH_DTYPE = torch.float16
_BENCH_M_VALUES = (
    1, 8, 16, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384,
)
# (B, M, N, K); the 56-batch rows use the measured M=234 size.
_BENCH_SHAPES = (
    [(16, M, 1024, 4096) for M in _BENCH_M_VALUES]
    + [(16, M, 8192, 1024) for M in _BENCH_M_VALUES]
    + [(16, M, 1280, 8192) for M in _BENCH_M_VALUES]
    + [(56, 234, 4928, 1232), (56, 234, 1232, 2464)]
)
_BENCH_CASES = [
    (B, M, N, K, layout)
    for layout in ("tn", "tt")
    for (B, M, N, K) in _BENCH_SHAPES
]


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    "B, M, N, K, layout",
    _BENCH_CASES,
    ids=[
        f"{layout} B={B} M={M} N={N} K={K}"
        for B, M, N, K, layout in _BENCH_CASES
    ],
)
def test_bench_batched_gemm_bf16(
    B: int,
    M: int,
    N: int,
    K: int,
    layout: str,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Time the shipped tuned config on both NT and TT layouts."""
    from boltops.gemm.triton.batched_gemm_bf16 import _get_config

    torch.manual_seed(223)
    x = torch.randn((B, M, K), device=DEVICE, dtype=_BENCH_DTYPE) * 0.05
    if layout == "tn":
        weight = torch.randn((B, N, K), device=DEVICE, dtype=_BENCH_DTYPE) * 0.05
    else:
        weight = (
            torch.randn((B, K, N), device=DEVICE, dtype=_BENCH_DTYPE)
            .transpose(1, 2)
        )
    output = torch.empty((B, M, N), device=DEVICE, dtype=_BENCH_DTYPE)
    config = _get_config(M, N, K, _BENCH_DTYPE, B=B, layout=layout)
    config["BATCH_LAST_GRID"] = True

    def run() -> None:
        batched_gemm_bf16(x, weight, dtype=_BENCH_DTYPE, y=output, config=config)

    run()
    assert output.shape == (B, M, N)

    ms = time_ms(run, bench_warmup, bench_repeat)
    flops = 2.0 * B * M * N * K
    print(
        f"[bench] batched_gemm_bf16 {layout} B={B} M={M} N={N} K={K}: "
        f"{ms:.6f} ms, {flops / (ms * 1e9):.2f} TFLOPS"
    )
