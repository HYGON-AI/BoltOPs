# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Recorded Gluon selection and its explicit native input contract."""

from importlib import import_module

import pytest
import torch

from boltops.gemm import gemm_a8w8_blockscale
from boltops.gemm.config import get_gemm_config

PROFILE = "DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128"
MODULE = "gemm_a8w8_blockscale"
KERNEL = "_gemm_a8w8_blockscale_as_fp32_bs_fp32_recipe_a_1x128_recipe_b_128x128_bm128_bn256_bk128_kernel"


def test_48tg_config_is_exact_and_cu_specific():
    params, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE",
        768,
        2048,
        6144,
        profile=PROFILE,
        backend="gluon",
        arch="gfx946",
        cu_num=48,
    )
    assert tuned
    assert (768 // params["BLOCK_SIZE_M"]) * (2048 // params["BLOCK_SIZE_N"]) == 48
    _, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE",
        640,
        2048,
        6144,
        profile=PROFILE,
        backend="gluon",
        arch="gfx946",
        cu_num=48,
    )
    assert not tuned
    with pytest.raises(FileNotFoundError):
        get_gemm_config(
            "GEMM-A8W8_BLOCKSCALE",
            768,
            2048,
            6144,
            backend="gluon",
            arch="gfx946",
            cu_num=192,
        )


@pytest.mark.parametrize(
    "m,scale_dtype,expected_backend",
    [
        (768, torch.float32, "gluon"),
        (640, torch.float32, "triton"),
        (768, torch.float16, "triton"),
    ],
)
@pytest.mark.parametrize("a_column_major", [False, True])
def test_auto_selects_only_the_recorded_gluon_shape(
    monkeypatch, m, scale_dtype, expected_backend, a_column_major
):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    gluon = import_module(f"boltops.gemm.gluon.{MODULE}")
    triton = import_module("boltops.gemm.triton.gemm_a8w8_blockscale_fp8")
    calls = []

    class CaptureLaunch:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(
                    (self.backend, grid(kwargs) if callable(grid) else grid, kwargs)
                )

            return launch

    monkeypatch.setattr(
        gluon,
        KERNEL,
        CaptureLaunch("gluon"),
    )
    monkeypatch.setattr(triton, "_w8a8_triton_block_scaled_mm", CaptureLaunch("triton"))
    a = torch.empty((m, 6144), dtype=torch.float8_e4m3fn, device="cuda")
    if a_column_major:
        a = a.as_strided(a.shape, (1, m))
        expected_backend = "triton"
    b = torch.empty((2048, 6144), dtype=a.dtype, device=a.device)
    sa = torch.empty((m, 48), dtype=scale_dtype, device=a.device)
    sb = torch.empty((16, 48), dtype=torch.float32, device=a.device)
    out = gemm_a8w8_blockscale(a, b, sa, sb)
    assert out.shape == (m, 2048) and out.dtype == torch.bfloat16
    assert len(calls) == 1 and calls[0][0] == expected_backend
    if expected_backend == "gluon":
        assert calls[0][1] == (6, 8)
        assert calls[0][2]["wasp_partition_warps"] == (4, 4, 4, 4)


@pytest.mark.parametrize(
    "k,scale_dtype,out_dtype",
    [
        (128, torch.float32, torch.bfloat16),
        (256, torch.float16, torch.bfloat16),
        (256, torch.float32, torch.float32),
    ],
)
def test_explicit_gluon_rejects_unsupported_combinations(k, scale_dtype, out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    from triton.compiler.errors import CompilationError

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    a = torch.empty((128, k), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((256, k), dtype=a.dtype, device=a.device)
    sa = torch.empty((128, k // 128), dtype=scale_dtype, device=a.device)
    sb = torch.empty((2, k // 128), dtype=scale_dtype, device=a.device)
    with pytest.raises(
        (CompilationError, NotImplementedError),
        match="pipeline requires|no Gluon configuration",
    ):
        gemm_a8w8_blockscale(a, b, sa, sb, dtype=out_dtype, backend="gluon")


def test_native_gluon_blockscale_small():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from boltops.utility.arch_info import get_device_info

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    torch.manual_seed(713)
    a = (torch.randint(-4, 5, (128, 256)).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (256, 256)).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (128, 2)).float() / 8
    sb = torch.randint(1, 5, (2, 2)).float() / 8
    golden = torch.zeros((128, 256), dtype=torch.float32)
    for group in range(2):
        start, end = group * 128, (group + 1) * 128
        golden += (a[:, start:end].float() @ b[:, start:end].float().T) * (
            sa[:, group, None] * sb[:, group].repeat_interleave(128)[None, :]
        )
    out = gemm_a8w8_blockscale(
        a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), backend="gluon"
    )
    torch.testing.assert_close(out.cpu(), golden.bfloat16(), atol=0, rtol=0)


def test_gluon_kernel_and_tile_come_from_config(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    seen = {}

    class CaptureKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.update(grid=grid(kwargs), params=kwargs)

            return launch

    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    kernels = import_module("boltops.gemm.gluon.gemm_a8w8_blockscale")
    monkeypatch.setattr(kernels, "kernel_another_tile", CaptureKernel(), raising=False)
    config = dict(
        kernel="kernel_another_tile",
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_warps=4,
        num_stages=1,
    )
    before = config.copy()
    a = torch.empty((128, 256), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((256, 256), dtype=a.dtype, device=a.device)
    sa = torch.empty((128, 2), dtype=torch.float32, device=a.device)
    sb = torch.empty((2, 2), dtype=torch.float32, device=a.device)
    gemm_a8w8_blockscale(a, b, sa, sb, backend="gluon", config=config)
    assert config == before
    assert seen["grid"] == (2, 2)
    assert tuple(seen["params"]["BLOCK_SIZE_" + dim] for dim in ("M", "N", "K")) == (
        64,
        128,
        64,
    )
    assert seen["params"]["GROUP_N"] == seen["params"]["GROUP_K"] == 128
