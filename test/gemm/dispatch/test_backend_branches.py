# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Each operation owns backend branching and directly launches native kernels."""

from importlib import import_module
import pytest
import torch
from boltops import gemm


@pytest.mark.parametrize(
    "interface,backend",
    [
        (interface, backend)
        for interface in gemm.__all__
        for backend in ("gluon", "tilelang")
        if (interface, backend) != ("gemm_a8w8_blockscale", "gluon")
    ],
)
def test_unimplemented_backend_fails_without_retry(interface, backend):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    dtype = torch.float16
    a = torch.ones((2, 64), device="cuda", dtype=dtype)
    b = torch.ones((16, 64), device="cuda", dtype=dtype)
    kwargs = dict(backend=backend)
    if interface == "gemm_a32w32":
        a, b = a.float(), b.float()
    if interface.startswith("batched_"):
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    if interface in ("gemm_a8w8", "gemm_a8w8_blockscale"):
        a, b = a.to(torch.int8), b.to(torch.int8)
    args = [a, b]
    if interface == "gemm_a8w8_blockscale":
        args += [torch.ones((2, 2), device="cuda"), torch.ones((1, 2), device="cuda")]

    elif interface == "gemm_a8w8":
        args += [torch.ones(2, device="cuda"), torch.ones(16, device="cuda")]
    elif interface == "gemm_a16w4":
        args = [
            a,
            torch.zeros((64, 2), device="cuda", dtype=torch.int32),
            torch.ones((1, 16), device="cuda", dtype=dtype),
        ]

    elif interface == "gemm_w4a4":
        args = [
            torch.full((2, 32), 0x22, device="cuda", dtype=torch.uint8),
            torch.full((16, 32), 0x22, device="cuda", dtype=torch.uint8),
            torch.full((2, 2), 127, device="cuda", dtype=torch.uint8),
            torch.full((16, 2), 127, device="cuda", dtype=torch.uint8),
        ]
    elif interface == "batched_gemm_a16w4":
        args = [
            a,
            torch.full((1, 16, 32), 0x22, device="cuda", dtype=torch.uint8),
            torch.full((1, 16, 2), 127, device="cuda", dtype=torch.uint8),
        ]
    with pytest.raises(
        NotImplementedError,
        match="no " + ("Gluon" if backend == "gluon" else "TileLang"),
    ):
        getattr(gemm, interface)(*args, **kwargs)


def test_public_batched_directly_launches_jit():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    public = import_module("boltops.gemm.batched_gemm_bf16")
    kernels = import_module("boltops.gemm.triton.batched_gemm_a16w16")

    assert not hasattr(kernels, "batched_gemm_a16w16")
    a = torch.ones((2, 3, 32), device="cuda", dtype=torch.float16)
    b = torch.ones((2, 5, 32), device="cuda", dtype=torch.float16)
    out = public.batched_gemm_bf16(a, b, dtype=torch.float16)
    torch.testing.assert_close(
        out.cpu(), torch.full((2, 3, 5), 32.0, dtype=torch.float16), atol=0, rtol=0
    )


def test_batched_aiter_output_and_config(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    a = torch.ones((1, 3, 32), device="cuda", dtype=torch.float16)
    b = torch.ones((1, 5, 32), device="cuda", dtype=torch.float16)
    output = torch.empty((1, 3, 5), device="cuda", dtype=torch.float16)
    params = dict(
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=1,
        BATCH_LAST_GRID=True,
    )
    before = params.copy()
    result = gemm.batched_gemm_bf16(a, b, dtype=torch.float16, out=output, config=params)
    assert result is output and params == before
    torch.testing.assert_close(
        result.cpu(), torch.full((1, 3, 5), 32.0, dtype=torch.float16), atol=0, rtol=0
    )


def test_aiter_default_output_dtype():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    a = torch.ones((1, 3, 32), device="cuda", dtype=torch.float16)
    b = torch.ones((1, 5, 32), device="cuda", dtype=torch.float16)
    assert gemm.batched_gemm_bf16(a, b).dtype == torch.bfloat16
    assert gemm.gemm_a16w16(a[0], b[0]).dtype == torch.bfloat16
