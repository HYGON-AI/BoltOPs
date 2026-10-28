# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for public MoE activations."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from boltops.fused_moe import (  # noqa: E402
    moe_activation,
    moe_activation_output_size,
    normalize_moe_activation,
)
from boltops.fused_moe.triton.moe_activation import (  # noqa: E402
    FUSED_ACTIVATION_QUANT_ACTIVATIONS,
    _get_fused_activation_quant_config,
    get_triton_situ_and_mul_config,
    is_fused_activation_quant_supported,
    triton_activation_quant,
)
from boltops.fused_moe.triton.moe_compat import (  # noqa: E402
    per_token_quant_hip,
    per_token_quant_triton,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _chunked(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return input.float().chunk(2, dim=-1)


def test_per_token_quant_hip_compat_alias():
    assert per_token_quant_hip is per_token_quant_triton


def _reference(
    input: torch.Tensor,
    activation: str,
    is_gated: bool,
) -> torch.Tensor:
    if is_gated:
        gate, up = _chunked(input)
        if activation == "silu":
            return F.silu(gate) * up
        if activation == "gelu":
            return F.gelu(gate, approximate="none") * up
        if activation in {"gelu_tanh", "gelu_pytorch_tanh"}:
            return F.gelu(gate, approximate="tanh") * up
        raise AssertionError(f"missing gated reference for {activation}")

    x = input.float()
    if activation in {"silu", "silu_no_mul"}:
        return F.silu(x)
    if activation in {"gelu", "gelu_no_mul"}:
        return F.gelu(x, approximate="none")
    if activation in {
        "gelu_tanh",
        "gelu_tanh_no_mul",
        "gelu_pytorch_tanh_no_mul",
    }:
        return F.gelu(x, approximate="tanh")
    if activation in {"relu2", "relu2_no_mul"}:
        return F.relu(x).square()
    raise AssertionError(f"missing non-gated reference for {activation}")


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("activation", "is_gated"),
    [
        ("silu", None),
        ("gelu", True),
        ("gelu_pytorch_tanh", True),
        ("silu", False),
        ("silu_no_mul", None),
        ("gelu_no_mul", None),
        ("gelu_pytorch_tanh_no_mul", None),
        ("relu2", None),
    ],
)
def test_moe_activation_matches_torch(activation, is_gated):
    torch.manual_seed(42)
    normalized, resolved_gated = normalize_moe_activation(activation, is_gated)
    width = 66 if resolved_gated else 33
    input = torch.randn((6, width), device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(
        (input.shape[0], moe_activation_output_size(width, resolved_gated)),
        device=DEVICE,
        dtype=input.dtype,
    )

    moe_activation(out, input, activation, is_gated)
    expected = _reference(input, activation, resolved_gated).to(input.dtype)

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    assert normalized in {
        "silu",
        "gelu",
        "gelu_tanh",
        "silu_no_mul",
        "gelu_no_mul",
        "gelu_tanh_no_mul",
        "relu2",
    }


@GPU_REQUIRED
def test_explicit_output_api_matches_aiter_call_shape():
    """Exercise the keyword contract used by moe_c/moe_asm/moe_triton."""
    input = torch.randn((7, 130), device=DEVICE, dtype=torch.float16)
    out = torch.empty((7, 65), device=DEVICE, dtype=torch.float16)

    returned = moe_activation(
        activation="silu",
        is_gated=True,
        activated_out=out,
        ffn1_out_2d=input,
        gemm1_alpha=None,
        gemm1_limit=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )

    expected = _reference(input, "silu", True).to(input.dtype)
    assert returned is None
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@GPU_REQUIRED
def test_situ_matches_torch():
    input = torch.randn((4, 70), device=DEVICE, dtype=torch.bfloat16)
    beta, linear_beta = 6.0, 5.0
    gate, up = _chunked(input)
    expected = (
        beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        * (linear_beta * torch.tanh(up / linear_beta))
    )

    out = torch.empty((4, 35), device=DEVICE, dtype=input.dtype)
    moe_activation(
        out,
        input,
        activation="situ",
        activation_situ_beta=beta,
        activation_situ_linear_beta=linear_beta,
    )

    torch.testing.assert_close(out, expected.to(input.dtype), rtol=2e-2, atol=2e-2)


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("rows", "width"),
    [(16, 66), (256, 2560), (257, 2562)],
)
def test_gpt_oss_swiglu_matches_torch(rows, width):
    """Cover the small-M path, flat path, and a non-divisible flat tail."""
    torch.manual_seed(42)
    alpha, limit = 1.702, 7.0
    input = torch.empty(
        (rows, width), device=DEVICE, dtype=torch.bfloat16
    ).uniform_(-9.0, 9.0)
    out = torch.empty((rows, width // 2), device=DEVICE, dtype=input.dtype)

    moe_activation(
        out,
        input,
        activation="silu",
        is_gated=True,
        gemm1_alpha=alpha,
        gemm1_limit=limit,
    )

    gate, up = _chunked(input)
    gate = torch.minimum(gate, torch.tensor(limit, device=DEVICE))
    up = torch.clamp(up, min=-limit, max=limit)
    expected = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    torch.testing.assert_close(out, expected.to(input.dtype), rtol=2e-2, atol=2e-2)


@GPU_REQUIRED
def test_situ_validation():
    input = torch.randn((2, 16), dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty((2, 8), dtype=torch.bfloat16, device=DEVICE)

    with pytest.raises(ValueError, match="requires activation_situ_beta"):
        moe_activation(out, input, activation="situ")
    with pytest.raises(ValueError, match="activation_situ_beta must be greater than 0"):
        moe_activation(out, input, activation="situ", activation_situ_beta=0.0)
    with pytest.raises(
        ValueError, match="activation_situ_linear_beta must be greater than 0"
    ):
        moe_activation(
            out,
            input,
            activation="situ",
            activation_situ_beta=1.0,
            activation_situ_linear_beta=0.0,
        )


@pytest.mark.parametrize(
    ("M", "expected"),
    [
        (1, {"BLOCK_SIZE_D": 128, "num_warps": 1}),
        (16, {"BLOCK_SIZE_D": 128, "num_warps": 1}),
        (17, {"BLOCK_SIZE_D": 512, "num_warps": 1}),
        (3394, {"BLOCK_SIZE_D": 512, "num_warps": 1}),
    ],
)
def test_situ_config(M, expected):
    assert get_triton_situ_and_mul_config(M, 1536) == expected


@GPU_REQUIRED
@pytest.mark.parametrize("activation", ["swigluoai", "swiglustep"])
def test_swiglu_variants_match_torch(activation):
    input = torch.randn((4, 74), device=DEVICE, dtype=torch.bfloat16)
    alpha, limit = 1.5, 4.0
    if activation == "swigluoai":
        gate = input[..., 0::2].float()
        up = input[..., 1::2].float()
        gate = torch.minimum(gate, torch.tensor(limit, device=DEVICE))
        up = torch.clamp(up, -limit, limit)
        expected = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    else:
        gate, up = _chunked(input)
        expected = torch.clamp(F.silu(gate), max=limit) * torch.clamp(
            up, -limit, limit
        )

    out = torch.empty((4, 37), device=DEVICE, dtype=input.dtype)
    moe_activation(
        out,
        input,
        activation=activation,
        gemm1_alpha=alpha if activation == "swigluoai" else None,
        gemm1_limit=limit,
    )

    torch.testing.assert_close(out, expected.to(input.dtype), rtol=2e-2, atol=2e-2)


def test_public_shape_helpers_and_validation():
    assert normalize_moe_activation("GELU_PYTORCH_TANH", None) == (
        "gelu_tanh",
        True,
    )
    assert normalize_moe_activation("relu2", None) == ("relu2", False)
    assert moe_activation_output_size(64, True) == 32
    assert moe_activation_output_size(63, False) == 63

    with pytest.raises(ValueError, match="Unsupported activation"):
        normalize_moe_activation("unknown", None)
    with pytest.raises(ValueError, match="requires gated"):
        normalize_moe_activation("situ", False)
    with pytest.raises(ValueError, match="must be even"):
        moe_activation_output_size(63, True)
    with pytest.raises(ValueError, match="only supports CUDA"):
        moe_activation(torch.empty(2, 4), torch.empty(2, 8), "silu")


@GPU_REQUIRED
def test_preallocated_output_validation():
    input = torch.empty((3, 20), device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="activated_out shape"):
        moe_activation(
            torch.empty((3, 20), device=DEVICE, dtype=torch.bfloat16),
            input,
            "silu",
        )


def test_fused_activation_quant_allowlist_and_fallbacks():
    assert FUSED_ACTIVATION_QUANT_ACTIVATIONS == frozenset(
        {
            ("silu", True),
            ("gelu", True),
            ("gelu_tanh", True),
            ("situ", True),
            ("swiglustep", True),
            ("silu", False),
            ("silu_no_mul", False),
            ("gelu", False),
            ("gelu_no_mul", False),
            ("gelu_tanh", False),
            ("gelu_tanh_no_mul", False),
            ("relu2", False),
            ("relu2_no_mul", False),
        }
    )
    assert not is_fused_activation_quant_supported(
        "swigluoai", True, torch.int8, 128
    )
    assert not is_fused_activation_quant_supported(
        "silu",
        True,
        torch.int8,
        128,
        gemm1_alpha=1.702,
        gemm1_limit=7.0,
    )
    assert not is_fused_activation_quant_supported(
        "future_activation", True, torch.int8, 128
    )
    assert not is_fused_activation_quant_supported(
        "silu", True, torch.float16, 128
    )
    assert not is_fused_activation_quant_supported(
        "silu", True, torch.int8, 8193
    )
    assert not is_fused_activation_quant_supported(
        "situ", True, torch.int8, 128
    )


def test_fused_activation_quant_d128_configs():
    # All gated/non-gated formulas use the tuned BM8/W1 D=128 configuration.
    assert _get_fused_activation_quant_config(128) == {
        "BLOCK_M": 8,
        "BLOCK_D": 128,
        "num_warps": 1,
        "num_stages": 1,
    }


@GPU_REQUIRED
def test_fused_activation_quant_row_multiplier():
    input = torch.randn((7, 256), device=DEVICE, dtype=torch.bfloat16)
    multiplier = torch.rand((7,), device=DEVICE, dtype=torch.float32)
    q, scale = triton_activation_quant(
        input, "silu", True, torch.int8, row_multiplier=multiplier
    )
    expected_q, expected_scale = triton_activation_quant(
        input, "silu", True, torch.int8
    )
    assert torch.equal(q, expected_q)
    assert torch.equal(scale, expected_scale * multiplier[:, None])


def _assert_fused_activation_quant_matches_unfused(
    activation: str,
    is_gated: bool,
    quant_dtype: torch.dtype,
    output_dim: int,
    **activation_kwargs,
):
    torch.manual_seed(123)
    input_dim = output_dim * 2 if is_gated else output_dim
    input = torch.empty(
        (7, input_dim), device=DEVICE, dtype=torch.bfloat16
    ).uniform_(-5.0, 5.0)
    q, scale = triton_activation_quant(
        input,
        activation,
        is_gated,
        quant_dtype,
        **activation_kwargs,
    )

    activated = torch.empty(
        (input.shape[0], output_dim), device=DEVICE, dtype=input.dtype
    )
    moe_activation(
        activated,
        input,
        activation,
        is_gated,
        **activation_kwargs,
    )
    expected_q, expected_scale = per_token_quant_triton(
        activated, quant_dtype=quant_dtype
    )

    assert torch.equal(q, expected_q)
    assert torch.equal(scale, expected_scale)


@GPU_REQUIRED
@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("output_dim", [33, 128])
@pytest.mark.parametrize(
    ("activation", "is_gated", "activation_kwargs"),
    [
        ("silu", True, {}),
        ("silu", True, {"gemm1_limit": 4.0}),
        ("gelu", True, {}),
        ("gelu_tanh", True, {}),
        ("situ", True, {"activation_situ_beta": 6.0}),
        (
            "situ",
            True,
            {
                "activation_situ_beta": 6.0,
                "activation_situ_linear_beta": 5.0,
            },
        ),
        ("swiglustep", True, {}),
        ("swiglustep", True, {"gemm1_limit": 4.0}),
        ("silu", False, {}),
        ("silu_no_mul", False, {}),
        ("gelu", False, {}),
        ("gelu_no_mul", False, {}),
        ("gelu_tanh", False, {}),
        ("gelu_tanh_no_mul", False, {}),
        ("relu2", False, {}),
        ("relu2_no_mul", False, {}),
    ],
)
def test_fused_activation_quant_all_allowlisted_acts(
    activation,
    is_gated,
    activation_kwargs,
    quant_dtype,
    output_dim,
):
    _assert_fused_activation_quant_matches_unfused(
        activation,
        is_gated,
        quant_dtype,
        output_dim=output_dim,
        **activation_kwargs,
    )


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("output_dim", "activation", "is_gated", "quant_dtype", "kwargs"),
    [
        (64, "silu", True, torch.int8, {}),
        (128, "gelu_tanh_no_mul", False, torch.float8_e4m3fn, {}),
        (256, "gelu", True, torch.float8_e4m3fn, {}),
        (384, "relu2", False, torch.int8, {}),
        (1536, "situ", True, torch.int8, {"activation_situ_beta": 6.0}),
        (7168, "silu", False, torch.float8_e4m3fn, {}),
    ],
)
def test_fused_activation_quant_widths(
    output_dim,
    activation,
    is_gated,
    quant_dtype,
    kwargs,
):
    _assert_fused_activation_quant_matches_unfused(
        activation,
        is_gated,
        quant_dtype,
        output_dim,
        **kwargs,
    )


@pytest.mark.bench
@GPU_REQUIRED
def test_moe_activation_benchmark(time_ms, bench_warmup, bench_repeat):
    input = torch.randn((3394, 3072), device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty((3394, 1536), device=DEVICE, dtype=torch.bfloat16)
    latency = time_ms(
        lambda: moe_activation(out, input, "silu", True),
        bench_warmup,
        bench_repeat,
    )
    print(f"moe_activation silu [3394,3072]: {latency:.4f} ms")
