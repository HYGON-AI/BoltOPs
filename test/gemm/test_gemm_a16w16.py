# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for ``gemm_a16w16``."""

from __future__ import annotations

import os
import re
import subprocess
import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_a16w16  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@pytest.fixture(scope="module", autouse=True)
def select_free_gpu():
    try:
        # Execute the 'hy-smi' command
        result = subprocess.run(["hy-smi"], capture_output=True, text=True, check=True)
        output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error executing 'hy-smi': {e}")
        free_hcus = []
    else:
        free_hcus = []
        lines = output.strip().split("\n")

        header_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("HCU"):
                header_line_index = i
                break

        if header_line_index == -1:
            print("Could not find header row in hy-smi output.")
        else:
            vram_col_index = lines[header_line_index].split().index("VRAM%")
            for line in lines[header_line_index + 1 :]:
                # Match lines that start with a digit (the HCU number)
                if re.match(r"^\d", line.strip()):
                    parts = line.split()
                    # Check for 0% VRAM usage. The value might be '0%' or '0.0%'.
                    if float(parts[vram_col_index].strip("%")) == 0.0:
                        free_hcus.append(int(parts[0]))

    if len(free_hcus) == 0:
        raise RuntimeError("No free HCU devices found")

    device_id = free_hcus[0]
    torch.cuda.set_device(device_id)
    print(f"Using HCU device: {device_id}")


# Qwen3-VL-8B shapes
QWEN3_VL_8B_SHAPES = [
    pytest.param(512, 4096, 4096),
    pytest.param(512, 4096, 12288),
    pytest.param(512, 6144, 4096),
    pytest.param(512, 24576, 4096),
    pytest.param(512, 151936, 4096),
    pytest.param(2560, 4096, 4096),
    pytest.param(2560, 4096, 12288),
    pytest.param(2560, 6144, 4096),
    pytest.param(2560, 24576, 4096),
    pytest.param(5684, 4096, 4608),
    pytest.param(5684, 4608, 4608),
    pytest.param(6656, 4096, 4096),
    pytest.param(6656, 4096, 12288),
    pytest.param(6656, 6144, 4096),
    pytest.param(6656, 24576, 4096),
    pytest.param(22736, 1152, 1152),
    pytest.param(22736, 1152, 1536),
    pytest.param(22736, 1152, 4304),
    pytest.param(22736, 3456, 1152),
    pytest.param(22736, 4304, 1152),
]


PERF_BACKENDS = ["triton", "tilelang"]

_SELECTED_BACKEND = os.environ.get("OPTEST_BACKEND", "triton")
if _SELECTED_BACKEND not in PERF_BACKENDS:
    raise pytest.UsageError(f"unsupported GEMM backend: {_SELECTED_BACKEND!r}")

GEMM_BACKENDS = [_SELECTED_BACKEND]


@pytest.fixture(params=[pytest.param(name, id=name) for name in GEMM_BACKENDS])
def gemm_backend(request: pytest.FixtureRequest):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request: pytest.FixtureRequest):
    return request.param


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.parametrize("M,N,K", [(64, 128, 64), (128, 256, 128)])
def test_gemm_a16w16_correctness(M: int, N: int, K: int, gemm_backend):
    """``Y = X @ W^T`` with ``W[N,K]`` — same as vLLM ROCm / aiter."""
    backend_name = gemm_backend
    torch.manual_seed(0)
    x_cpu = torch.randn((M, K), device="cpu", dtype=torch.float16)
    w_cpu = torch.randn((N, K), device="cpu", dtype=torch.float16)
    golden = torch.matmul(x_cpu, w_cpu.T)
    out = gemm_a16w16(
        x_cpu.to(DEVICE),
        w_cpu.to(DEVICE),
        dtype=torch.float16,
        backend=backend_name,
    )
    _assert_close(out, golden)


@GPU_REQUIRED
def test_gemm_a16w16_bias(gemm_backend):
    backend_name = gemm_backend
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
        backend=backend_name,
    )
    _assert_close(out, golden)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("M,N,K", [(512, 512, 512)] + QWEN3_VL_8B_SHAPES)
def test_bench_gemm_a16w16(
    M: int, N: int, K: int, gemm_backend, dtype, time_ms,
    bench_warmup: int, bench_repeat: int
):
    backend_name = gemm_backend
    torch.manual_seed(0)
    x = torch.randn((M, K), device=DEVICE, dtype=dtype)
    w = torch.randn((N, K), device=DEVICE, dtype=dtype)
    gemm_a16w16(x, w, dtype=dtype, backend=backend_name)
    torch.cuda.synchronize()
    ms = time_ms(
        lambda: gemm_a16w16(x, w, dtype=dtype, backend=backend_name),
        bench_warmup,
        bench_repeat,
    )
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    print(
        f"[bench] gemm_a16w16 backend={backend_name}, dtype={dtype}, "
        f"{M}x{N}x{K}: {ms:.3f} ms, {tflops:.2f} TFLOPS"
    )
