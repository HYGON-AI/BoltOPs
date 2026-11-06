# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Small native tests for the three common GEMM interfaces; no tuning scans."""

from dataclasses import replace

import pytest
import torch
import triton

from boltops.gemm import gemm_a16w16, gemm_a32w32, batched_gemm_bf16
from boltops.gemm.config import get_gemm_config


def require_hcu():
    if (
        not torch.cuda.is_available()
        or triton.runtime.driver.active.get_current_target().arch != "gfx946"
    ):
        pytest.skip("requires gfx946")


def dense_config(m=4096, n=4608, k=4096, cus=48):
    return get_gemm_config("GEMM-A16W16", m, n, k, arch="gfx946", cu_num=cus)[0]


def test_dense_selects_existing_fp16_config():
    assert dense_config()["wasp_partition_regs"] == [16, 244, 244]
    for m in [4095, 4097]:
        assert "use_mls" not in dense_config(m=m)
    assert "use_mls" not in dense_config(cus=96)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dense_native_small(dtype):
    require_hcu()
    torch.manual_seed(317)
    a = (torch.randint(-4, 5, (5, 64)).float() / 4).to(dtype)
    b = (torch.randint(-4, 5, (7, 64)).float() / 4).to(dtype)
    bias = torch.full((7,), 0.25, dtype=dtype)
    output = torch.empty((5, 7), dtype=dtype, device="cuda")
    out = (gemm_a32w32 if dtype == torch.float32 else gemm_a16w16)(
        a.cuda(), b.cuda(), bias=bias.cuda(), dtype=dtype, out=output
    )
    assert out is output
    torch.testing.assert_close(
        out.cpu(), (a.float() @ b.float().T + bias.float()).to(dtype), atol=0, rtol=0
    )


def test_existing_tuned_fp16_recipe_single_cta(monkeypatch):
    require_hcu()
    from importlib import import_module

    module = import_module("boltops.gemm.gemm_a16w16")
    params = dense_config()
    torch.manual_seed(319)
    a = (torch.randint(-2, 3, (256, 256)).float() / 4).half()
    nn_b = (torch.randint(-2, 3, (256, 256)).float() / 4).half()
    out = module.gemm_a16w16(
        a.cuda(), nn_b.cuda().T, dtype=torch.float16, config=params
    )
    torch.testing.assert_close(
        out.cpu(), (a.float() @ nn_b.float()).half(), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_contiguous", [False, True])
def test_grouped_native_equal(dtype, n_contiguous):
    require_hcu()
    a = torch.ones((2, 3, 32), dtype=dtype)
    b = ((torch.arange(2 * 5 * 32).reshape(2, 5, 32) % 5 + 1).float() / 4).to(dtype)
    bias = torch.full((2, 1, 5), 0.25, dtype=dtype)
    weight = b.transpose(1, 2).contiguous().transpose(1, 2) if n_contiguous else b
    out = batched_gemm_bf16(a.cuda(), weight.cuda(), bias=bias.cuda(), dtype=dtype)
    golden = torch.bmm(a.float(), b.float().transpose(1, 2)) + bias.float()
    torch.testing.assert_close(out.cpu(), golden.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize(
    "activation", ["relu", "relu6", "gelu", "gelu_tanh", "silu", "silu_exp2"]
)
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_fused_activation(activation, split_k, dtype):
    require_hcu()
    torch.manual_seed(741)
    a = (torch.randint(-2, 3, (3, 128)).float() / 4).to(dtype)
    b = (torch.randint(-2, 3, (7, 128)).float() / 4).to(dtype)
    bias = torch.linspace(-1.5, 1.5, 7).to(dtype)
    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=1,
        NUM_KSPLIT=split_k,
        SPLITK_BLOCK_SIZE=128 // split_k,
        num_warps=4,
        num_stages=1,
        cache_modifier=None,
    )
    before = config.copy()
    output = torch.empty((3, 7), dtype=torch.float32, device="cuda")
    result = gemm_a16w16(
        a.cuda(),
        b.cuda(),
        bias=bias.cuda(),
        dtype=torch.float32,
        out=output,
        activation=activation,
        config=config,
    )
    value = a.float() @ b.float().T + bias.float()
    if activation in ("relu", "relu6"):
        golden = value.clamp(min=0, max=6 if activation == "relu6" else None)
    elif activation in ("gelu", "gelu_tanh"):
        golden = torch.nn.functional.gelu(
            value, approximate="tanh" if activation == "gelu_tanh" else "none"
        )
    else:
        golden = torch.nn.functional.silu(value)
    assert result is output and config == before
    torch.testing.assert_close(result.cpu(), golden, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "activation", ["relu", "relu6", "gelu", "gelu_tanh", "silu", "silu_exp2"]
)
def test_dense_activation_empty_k(activation):
    require_hcu()
    a = torch.empty((3, 0), dtype=torch.float16, device="cuda")
    b = torch.empty((7, 0), dtype=a.dtype, device=a.device)
    bias = torch.linspace(-2, 8, 7, device=a.device).to(a.dtype)
    output = torch.empty((3, 7), dtype=torch.float32, device=a.device)
    result = gemm_a16w16(
        a, b, bias=bias, dtype=output.dtype, out=output, activation=activation
    )
    value = bias.float().cpu().expand(3, 7)
    if activation in ("relu", "relu6"):
        golden = value.clamp(min=0, max=6 if activation == "relu6" else None)
    elif activation in ("gelu", "gelu_tanh"):
        golden = torch.nn.functional.gelu(
            value, approximate="tanh" if activation == "gelu_tanh" else "none"
        )
    else:
        golden = torch.nn.functional.silu(value)
    assert result is output
    torch.testing.assert_close(result.cpu(), golden, atol=1e-4, rtol=1e-4)


def test_dense_unknown_activation_rejected():
    require_hcu()
    a = torch.empty((3, 32), dtype=torch.float16, device="cuda")
    b = torch.empty((7, 32), dtype=a.dtype, device=a.device)
    with pytest.raises(ValueError, match="unsupported activation"):
        gemm_a16w16(a, b, activation="unknown")
