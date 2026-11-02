# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and CUDA-graph benchmarks for the public fused MoE operator."""

from __future__ import annotations

import importlib
import inspect

from statistics import median
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

triton = pytest.importorskip("triton")
import triton.testing  # noqa: E402

from boltops.fused_moe import fused_experts  # noqa: E402
import boltops.fused_moe.triton.fused_moe as fused_moe_module  # noqa: E402
import boltops.fused_moe.triton.moe_op as moe_op_module  # noqa: E402
from boltops.fused_moe.triton.moe_activation import (  # noqa: E402
    FUSED_ACTIVATION_QUANT_ACTIVATIONS,
)
from boltops.fused_moe.triton.moe_compat import pertoken_quant  # noqa: E402
from boltops.tools.fused_moe_triton_tune.routing import fused_topk  # noqa: E402
from boltops.tools.fused_moe_triton_tune.performance_csv import (  # noqa: E402
    spread_fraction,
)
from boltops.utility import arch_info  # noqa: E402
from boltops.fused_moe.triton.quant import (  # noqa: E402
    dynamic_per_token_quant_fp8_i8,
    static_per_tensor_quant_fp8_i8,
)
from boltops.tools.fused_moe_triton_tune.common import (  # noqa: E402
    quantize_weights,
    scalar_types,
)
from boltops.utility.triton_capability import (  # noqa: E402
    get_triton_capabilities,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
MXFP_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None
    or arch_info.get_arch() != "gfx946",
    reason="native MXFP tests require gfx946",
)
BLOCK_PINGPONG_REQUIRED = pytest.mark.skipif(
    not get_triton_capabilities().supports("use_block_pingpong"),
    reason="Triton backend does not support use_block_pingpong",
)

# Keep the matrix dimensions aligned with the tuning defaults while limiting
# the expert count to a representative local EP shard.  M=64 and topk=4 also
# make the second W8A8 quantization process enough 128-wide rows to exercise
# the tiled dispatch rather than only the tiny-input path.
NUM_TOKENS = 64
HIDDEN_SIZE = 384
INTERMEDIATE_SIZE = 7168
NUM_EXPERTS = 8
TOPK = 4

# Shapes and M samples from docs/ops/fused_moe/performance_report.md.
# N is the intermediate width: gated Gemm1 produces 2N columns.
BENCH_TOKENS = [
    1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768,
]
BENCH_SHAPES = [
    pytest.param(("bf16", 128, 384, 4096, torch.bfloat16), id="bf16-E128-N384-K4096"),
    pytest.param(("int8_channel", 256, 256, 7168, torch.float16),
                 id="int8-channel-E256-N256-K7168"),
    pytest.param(("int8_block", 256, 256, 7168, torch.float16),
                 id="int8-block128-E256-N256-K7168"),
    pytest.param(("fp8_channel", 256, 320, 7168, torch.float16),
                 id="fp8-channel-E256-N320-K7168"),
]


def _default_activation_kwargs(activation: str) -> dict:
    return {"activation_situ_beta": 6.0} if activation == "situ" else {}


ROUTER_FOLD_ACT_CASES = [
    (activation, is_gated, _default_activation_kwargs(activation))
    for activation, is_gated in sorted(FUSED_ACTIVATION_QUANT_ACTIVATIONS)
] + [
    ("silu", True, {"gemm1_limit": 4.0}),
    (
        "situ",
        True,
        {
            "activation_situ_beta": 6.0,
            "activation_situ_linear_beta": 5.0,
        },
    ),
    ("swiglustep", True, {"gemm1_limit": 4.0}),
]


def _routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(num_tokens, device=DEVICE)[:, None]
    expert_offsets = torch.arange(topk, device=DEVICE)[None, :]
    topk_ids = ((token_ids + expert_offsets) % num_experts).to(torch.int32)

    route_weights = torch.arange(topk, 0, -1, device=DEVICE, dtype=torch.float32)
    route_weights /= route_weights.sum()
    topk_weights = route_weights.expand(num_tokens, topk).contiguous()
    return topk_weights, topk_ids


def _torch_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Two-stage, gated SiLU MoE reference with kernel-like stage casts."""
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    repeated = (
        hidden_states[:, None, :]
        .expand(num_tokens, topk, hidden_size)
        .reshape(-1, hidden_size)
    )
    flat_ids = topk_ids.reshape(-1)
    routed_out = torch.zeros_like(repeated)

    for expert_id in range(w1.shape[0]):
        mask = flat_ids == expert_id
        if not mask.any():
            continue
        intermediate = (
            repeated[mask].float() @ w1[expert_id].float().T
        ).to(hidden_states.dtype)
        gate, up = intermediate.float().chunk(2, dim=-1)
        activated = (F.silu(gate) * up).to(hidden_states.dtype)
        routed_out[mask] = (
            activated.float() @ w2[expert_id].float().T
        ).to(hidden_states.dtype)

    return (
        routed_out.view(num_tokens, topk, hidden_size).float()
        * topk_weights[:, :, None].float()
    ).sum(dim=1).to(hidden_states.dtype)


@GPU_REQUIRED
@torch.inference_mode()
def test_fused_experts_topk1_matches_torch():
    torch.manual_seed(4)
    num_tokens = 32
    hidden_size = 336
    intermediate_size = 512
    num_experts = 4

    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    topk_weights, topk_ids = _routing(num_tokens, num_experts, 1)
    actual = fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        global_num_experts=num_experts,
    )
    expected = _torch_moe(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)


def _pack_w4a16_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an [out, in] matrix and pack adjacent uint4 values."""
    reference, quantized, scales, zero_points = quantize_weights(
        weight.T,
        scalar_types.uint4,
        group_size,
        zero_points=True,
        ref_zero_points_after_scales=False,
    )
    reference = reference.T.contiguous()
    quantized = quantized.T.contiguous().to(torch.uint8)
    scales = scales.T.contiguous()
    zero_points = zero_points.T.contiguous().to(torch.uint8)

    packed_weight = quantized[:, 1::2] * 16 + quantized[:, ::2]
    packed_zero_points = zero_points[1::2, :] * 16 + zero_points[::2, :]
    return reference, packed_weight, scales, packed_zero_points


def _dynamic_group_quant_int8(
    tensor: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = tensor.reshape(-1, group_size)
    absmax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10).float()
    scales = absmax / torch.iinfo(torch.int8).max
    quantized = (
        (groups.float() / scales).round().clamp(-128, 127).to(torch.int8)
    )
    return (
        quantized.reshape_as(tensor),
        scales.reshape(*tensor.shape[:-1], tensor.shape[-1] // group_size),
    )


def _block_scaled_int8_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    block_shape: tuple[int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference for A-scale/B-scale block-wise INT8 matmul."""
    block_n, block_k = block_shape
    output = torch.zeros(
        (lhs.shape[0], rhs.shape[0]),
        dtype=torch.float32,
        device=lhs.device,
    )
    for n_block, n_start in enumerate(range(0, rhs.shape[0], block_n)):
        n_end = min(n_start + block_n, rhs.shape[0])
        for k_block, k_start in enumerate(range(0, rhs.shape[1], block_k)):
            k_end = min(k_start + block_k, rhs.shape[1])
            product = (
                lhs[:, k_start:k_end].float()
                @ rhs[n_start:n_end, k_start:k_end].float().T
            )
            scale = lhs_scales[:, k_block, None] * rhs_scales[
                n_block, k_block
            ]
            output[:, n_start:n_end] += product * scale
    return output.to(output_dtype)


def _int8_rounding_input(rows: int, cols: int) -> torch.Tensor:
    values = torch.zeros((rows, cols), device=DEVICE, dtype=torch.float16)
    values[:, :5] = torch.tensor(
        [0.5, -0.5, 1.5, -1.5, 127.0],
        device=DEVICE,
        dtype=torch.float16,
    )
    return values


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("rows,cols", [(1, 8), (256, 128)])
def test_dynamic_per_token_int8_rounds_to_nearest(rows: int, cols: int):
    values = _int8_rounding_input(rows, cols)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(rows, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    expected = values.float().round().to(torch.int8)
    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, expected)


@GPU_REQUIRED
@torch.inference_mode()
def test_dynamic_per_token_int8_tiled_path_rounds_to_nearest():
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    rows = max(1025, 16 * num_cus)
    values = _int8_rounding_input(rows, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(rows, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    expected = values.float().round().to(torch.int8)
    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, expected)


@GPU_REQUIRED
@torch.inference_mode()
def test_dynamic_per_token_fp8_keeps_fractional_values():
    values = torch.tensor(
        [[0.5, 1.5, 2.5, 448.0]],
        device=DEVICE,
        dtype=torch.float16,
    )
    quantized = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized.float(), values.float())


@GPU_REQUIRED
@torch.inference_mode()
def test_static_per_tensor_int8_rounds_to_nearest():
    values = _int8_rounding_input(2, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)

    static_per_tensor_quant_fp8_i8(quantized, values, scale)

    expected = values.float().round().to(torch.int8)
    torch.testing.assert_close(quantized, expected)


@GPU_REQUIRED
@torch.inference_mode()
def test_torch_int8_quant_helper_rounds_to_nearest():
    values = _int8_rounding_input(2, 8)
    scale = torch.ones((2, 1), device=DEVICE, dtype=torch.float32)

    quantized, actual_scale = pertoken_quant(
        values, scale=scale, quant_dtype=torch.int8
    )

    expected = values.float().round().to(torch.int8)
    torch.testing.assert_close(actual_scale, scale)
    torch.testing.assert_close(quantized, expected)


def _torch_w8a8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    repeated = (
        hidden_states[:, None, :]
        .expand(num_tokens, topk, hidden_size)
        .reshape(-1, hidden_size)
        .contiguous()
    )
    input_q, input_scales = _dynamic_group_quant_int8(
        repeated, block_shape[1]
    )
    flat_ids = topk_ids.reshape(-1)
    routed_out = torch.zeros_like(repeated)

    for expert_id in range(w1.shape[0]):
        mask = flat_ids == expert_id
        if not mask.any():
            continue
        intermediate = _block_scaled_int8_matmul(
            input_q[mask],
            w1[expert_id],
            input_scales[mask],
            w1_scales[expert_id],
            block_shape,
            hidden_states.dtype,
        )
        gate, up = intermediate.float().chunk(2, dim=-1)
        activated = (F.silu(gate) * up).to(hidden_states.dtype).contiguous()
        activated_q, activated_scales = _dynamic_group_quant_int8(
            activated, block_shape[1]
        )
        routed_out[mask] = _block_scaled_int8_matmul(
            activated_q,
            w2[expert_id],
            activated_scales,
            w2_scales[expert_id],
            block_shape,
            hidden_states.dtype,
        )

    return (
        routed_out.view(num_tokens, topk, hidden_size).float()
        * topk_weights[:, :, None].float()
    ).sum(dim=1).to(hidden_states.dtype)


@GPU_REQUIRED
@torch.inference_mode()
def test_fused_experts_bf16_matches_torch():
    torch.manual_seed(1)
    hidden_states = torch.randn(
        (NUM_TOKENS, HIDDEN_SIZE), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    w1 = torch.randn(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    w2 = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    topk_weights, topk_ids = _routing(NUM_TOKENS, NUM_EXPERTS, TOPK)

    actual = fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        global_num_experts=NUM_EXPERTS,
    )
    expected = _torch_moe(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("persistent", [False, True])
def test_sorted_gemm_bridge_uses_buffer_load_without_mls(
    monkeypatch, persistent
):
    """The expert-sorted GEMM1 output is not coupled to matrix_load support."""
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_sorted_layout",
        lambda _config, _use_int4: True,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_mls",
        lambda _config, _use_int4: False,
    )
    monkeypatch.setattr(
        moe_op_module, "_USE_MOE_PERSISTENT_KERNEL", persistent
    )

    torch.manual_seed(19)
    num_tokens, hidden_size = 32, 336
    intermediate_size, num_experts, topk = 512, 4, 2
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    topk_weights, topk_ids = _routing(num_tokens, num_experts, topk)

    actual = fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        global_num_experts=num_experts,
    )
    expected = _torch_moe(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)


@GPU_REQUIRED
@torch.inference_mode()
def test_fused_experts_w4a16_matches_dequantized_reference():
    torch.manual_seed(2)
    group_size = 64
    hidden_states = torch.randn(
        (NUM_TOKENS, HIDDEN_SIZE), device=DEVICE, dtype=torch.float16
    ) / 10

    w1_parts = [
        _pack_w4a16_weight(
            torch.randn(
                (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                device=DEVICE,
                dtype=torch.float16,
            )
            / 10,
            group_size,
        )
        for _ in range(NUM_EXPERTS)
    ]
    w2_parts = [
        _pack_w4a16_weight(
            torch.randn(
                (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                device=DEVICE,
                dtype=torch.float16,
            )
            / 10,
            group_size,
        )
        for _ in range(NUM_EXPERTS)
    ]
    w1_ref, w1, w1_scales, w1_zero_points = map(torch.stack, zip(*w1_parts))
    w2_ref, w2, w2_scales, w2_zero_points = map(torch.stack, zip(*w2_parts))
    topk_weights, topk_ids = _routing(NUM_TOKENS, NUM_EXPERTS, TOPK)

    actual = fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        use_int4_w4a16=True,
        global_num_experts=NUM_EXPERTS,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_zero_points,
        w2_zp=w2_zero_points,
        block_shape=[0, group_size],
        output_dtype=hidden_states.dtype,
    )
    expected = _torch_moe(
        hidden_states, w1_ref, w2_ref, topk_weights, topk_ids
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)


@GPU_REQUIRED
@torch.inference_mode()
def test_fused_experts_w8a8_matches_block_quantized_reference():
    torch.manual_seed(3)
    block_shape = (128, 128)
    hidden_states = torch.randn(
        (NUM_TOKENS, HIDDEN_SIZE), device=DEVICE, dtype=torch.float16
    ) / 10
    w1 = torch.randint(
        -127,
        128,
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=DEVICE,
        dtype=torch.int8,
    )
    w2 = torch.randint(
        -127,
        128,
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=DEVICE,
        dtype=torch.int8,
    )
    w1_scales = torch.rand(
        (
            NUM_EXPERTS,
            2 * INTERMEDIATE_SIZE // block_shape[0],
            HIDDEN_SIZE // block_shape[1],
        ),
        device=DEVICE,
        dtype=torch.float32,
    ) * 1e-2
    w2_scales = torch.rand(
        (
            NUM_EXPERTS,
            HIDDEN_SIZE // block_shape[0],
            INTERMEDIATE_SIZE // block_shape[1],
        ),
        device=DEVICE,
        dtype=torch.float32,
    ) * 1e-2
    topk_weights, topk_ids = _routing(NUM_TOKENS, NUM_EXPERTS, TOPK)

    actual = fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        use_int8_w8a8=True,
        global_num_experts=NUM_EXPERTS,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        block_shape=list(block_shape),
        output_dtype=hidden_states.dtype,
    )
    expected = _torch_w8a8_moe(
        hidden_states,
        w1,
        w2,
        w1_scales,
        w2_scales,
        topk_weights,
        topk_ids,
        block_shape,
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    assert torch.isfinite(actual).all()
    mean_relative_error = (
        (actual.float() - expected.float()).abs().mean()
        / expected.float().abs().mean()
    )
    assert mean_relative_error < 0.02, (
        f"W8A8 mean relative error is {mean_relative_error.item():.6f}"
    )


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize(
    ("quant_kind", "activation", "is_gated"),
    [
        ("int8", "gelu", True),
        ("fp8", "gelu_tanh_no_mul", False),
    ],
)
def test_per_channel_moe_fused_activation_quant_matches_fallback(
    monkeypatch,
    quant_kind,
    activation,
    is_gated,
):
    # Keep this test scoped to activation+quant fusion. Router-scale folding
    # has its own numerical-equivalence coverage below.
    monkeypatch.setattr(
        fused_moe_module,
        "_should_fold_router_weight",
        lambda **kwargs: False,
    )
    torch.manual_seed(77)
    num_tokens, hidden_size = 8, 128
    intermediate_size, num_experts, topk = 64, 4, 2
    w1_out = 2 * intermediate_size if is_gated else intermediate_size
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.float16
    ) / 10
    if quant_kind == "int8":
        weight_dtype = torch.int8
        w1 = torch.randint(
            -127,
            128,
            (num_experts, w1_out, hidden_size),
            device=DEVICE,
            dtype=weight_dtype,
        )
        w2 = torch.randint(
            -127,
            128,
            (num_experts, hidden_size, intermediate_size),
            device=DEVICE,
            dtype=weight_dtype,
        )
    else:
        weight_dtype = torch.float8_e4m3fn
        w1 = (
            torch.randn(
                (num_experts, w1_out, hidden_size), device=DEVICE
            )
            / 3
        ).to(weight_dtype)
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size), device=DEVICE
            )
            / 3
        ).to(weight_dtype)

    w1_scale = torch.rand(
        (num_experts, w1_out), device=DEVICE, dtype=torch.float32
    ) * 1e-2
    w2_scale = torch.rand(
        (num_experts, hidden_size), device=DEVICE, dtype=torch.float32
    ) * 1e-2
    topk_weights, topk_ids = _routing(num_tokens, num_experts, topk)
    kwargs = {
        "output_dtype": hidden_states.dtype,
        "use_int8_w8a8": quant_kind == "int8",
        "use_fp8_w8a8": quant_kind == "fp8",
        "per_channel_quant": True,
        "global_num_experts": num_experts,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "activation": activation,
        "is_gated": is_gated,
    }

    real_fused_activation_quant = fused_moe_module.triton_activation_quant
    fused_activation_quant_calls = 0

    def tracked_fused_activation_quant(*args, **quant_kwargs):
        nonlocal fused_activation_quant_calls
        fused_activation_quant_calls += 1
        return real_fused_activation_quant(*args, **quant_kwargs)

    monkeypatch.setattr(
        fused_moe_module,
        "triton_activation_quant",
        tracked_fused_activation_quant,
    )
    fused_output = fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        **kwargs,
    )
    assert fused_activation_quant_calls == 1

    monkeypatch.setattr(
        fused_moe_module,
        "is_fused_activation_quant_supported",
        lambda *args, **kwargs: False,
    )
    fused_activation_quant_calls = 0
    fallback_output = fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        **kwargs,
    )
    assert fused_activation_quant_calls == 0
    assert torch.equal(fused_output, fallback_output)


def test_fused_activation_quant_weight_mode_policy():
    common = {
        "use_int8_w8a8": True,
        "use_fp8_w8a8": False,
        "use_int4_w4a8": False,
        "per_channel_quant": True,
        "has_expert_map": False,
        "activation": "silu",
        "is_gated": True,
        "output_dim": 128,
        "gemm1_alpha": None,
        "gemm1_limit": None,
        "activation_situ_beta": None,
        "activation_situ_linear_beta": None,
    }
    assert fused_moe_module._should_use_fused_activation_quant(**common)
    assert fused_moe_module._should_use_fused_activation_quant(
        **dict(common, use_int8_w8a8=False, use_fp8_w8a8=True)
    )
    assert not fused_moe_module._should_use_fused_activation_quant(
        **dict(common, use_int8_w8a8=False)
    )
    assert not fused_moe_module._should_use_fused_activation_quant(
        **dict(common, use_fp8_w8a8=True)
    )
    assert not fused_moe_module._should_use_fused_activation_quant(
        **dict(common, use_int4_w4a8=True)
    )


def test_should_fold_router_weight_policy():
    common = {
        "fuse_activation_quant": True,
        "needs_router_weight": True,
    }
    assert fused_moe_module._should_fold_router_weight(**common)
    for key in common:
        case = dict(common)
        case[key] = False
        assert not fused_moe_module._should_fold_router_weight(**case)


def _assert_router_fold_close(
    folded: torch.Tensor,
    baseline: torch.Tensor,
) -> None:
    reference = baseline.float().abs()
    error = (folded.float() - baseline.float()).abs()
    reference_max = reference.max().item()
    reference_mean = reference.mean().item()
    assert reference_max > 1e-2
    assert reference_mean > 1e-4
    assert error.max().item() <= reference_max * 2e-2 + 1e-6
    assert error.mean().item() <= reference_mean * 1e-2 + 1e-7


def _run_router_fold_pair(
    monkeypatch,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    kwargs,
):
    monkeypatch.setattr(
        fused_moe_module,
        "_should_fold_router_weight",
        lambda **predicate_kwargs: True,
    )
    folded = fused_moe_module.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_should_fold_router_weight",
        lambda **predicate_kwargs: False,
    )
    baseline = fused_moe_module.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
    )
    _assert_router_fold_close(folded, baseline)


def _make_router_fold_case(
    *,
    seed: int,
    quant_kind: str,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    is_gated: bool,
    scale_max: Optional[float] = None,
    with_bias: bool = False,
):
    torch.manual_seed(seed)
    w1_out = intermediate_size * (2 if is_gated else 1)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    if quant_kind == "int8":
        w1 = torch.randint(
            -127,
            128,
            (num_experts, w1_out, hidden_size),
            device=DEVICE,
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            128,
            (num_experts, hidden_size, intermediate_size),
            device=DEVICE,
            dtype=torch.int8,
        )
        default_scale_max = 1e-2
    else:
        if quant_kind != "fp8":
            raise ValueError(f"unsupported quant_kind: {quant_kind}")
        w1 = (
            torch.randn(
                (num_experts, w1_out, hidden_size), device=DEVICE
            ) / 3
        ).to(torch.float8_e4m3fn)
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size), device=DEVICE
            ) / 3
        ).to(torch.float8_e4m3fn)
        default_scale_max = 0.5

    scale_max = default_scale_max if scale_max is None else scale_max
    w1_scale = torch.rand(
        (num_experts, w1_out), device=DEVICE, dtype=torch.float32
    ) * scale_max
    w2_scale = torch.rand(
        (num_experts, hidden_size), device=DEVICE, dtype=torch.float32
    ) * scale_max
    topk_weights, topk_ids = _routing(num_tokens, num_experts, topk)
    kwargs = {
        "output_dtype": hidden_states.dtype,
        "use_int8_w8a8": quant_kind == "int8",
        "use_fp8_w8a8": quant_kind == "fp8",
        "per_channel_quant": True,
        "global_num_experts": num_experts,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "activation": activation,
        "is_gated": is_gated,
    }
    if with_bias:
        kwargs["b2"] = torch.randn(
            (num_experts, hidden_size),
            device=DEVICE,
            dtype=torch.bfloat16,
        ) / 100
    return hidden_states, w1, w2, topk_weights, topk_ids, kwargs


def test_naive_block_assignment_policy():
    common = {
        "num_routes": 64,
        "global_num_experts": 256,
        "has_expert_map": False,
        "bottom_a_sorted": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "use_int4_w4a8": False,
        "use_block_quant": False,
    }
    assert fused_moe_module._should_use_naive_block_assignment(**common)
    assert not fused_moe_module._should_use_naive_block_assignment(
        **dict(common, num_routes=65)
    )
    assert not fused_moe_module._should_use_naive_block_assignment(
        **dict(common, use_block_quant=True)
    )
    assert fused_moe_module._should_use_naive_block_assignment(
        **dict(common, use_block_quant=True, num_routes=32)
    )
    for key in (
        "has_expert_map",
        "bottom_a_sorted",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "use_int4_w4a8",
    ):
        assert not fused_moe_module._should_use_naive_block_assignment(
            **dict(common, **{key: True})
        )


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("quant_kind", ["int8", "fp8"])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_naive_block_assignment_matches_sorted_metadata(
    monkeypatch, quant_kind, id_dtype
):
    case = _make_router_fold_case(
        seed=99,
        quant_kind=quant_kind,
        num_tokens=2,
        hidden_size=128,
        intermediate_size=128,
        num_experts=64,
        topk=8,
        activation="silu",
        is_gated=True,
    )
    hidden_states, w1, w2, topk_weights, topk_ids, kwargs = case
    topk_ids = topk_ids.to(id_dtype)
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_sorted_layout",
        lambda _config, _use_int4: False,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_mls",
        lambda _config, _use_int4: False,
    )

    direct = fused_moe_module.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_should_use_naive_block_assignment",
        lambda **predicate_kwargs: False,
    )
    baseline = fused_moe_module.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
    )

    torch.testing.assert_close(direct, baseline, rtol=2e-2, atol=1e-6)


@GPU_REQUIRED
@torch.inference_mode()
def test_naive_block_assignment_bf16_matches_sorted_metadata(monkeypatch):
    torch.manual_seed(100)
    num_tokens, hidden_size = 2, 128
    intermediate_size, num_experts, topk = 128, 64, 8
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=DEVICE,
        dtype=torch.bfloat16,
    ) / 10
    topk_weights, topk_ids = _routing(num_tokens, num_experts, topk)
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_sorted_layout",
        lambda _config, _use_int4: False,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_bottom_a_uses_mls",
        lambda _config, _use_int4: False,
    )

    direct = fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        global_num_experts=num_experts,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_should_use_naive_block_assignment",
        lambda **predicate_kwargs: False,
    )
    baseline = fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        global_num_experts=num_experts,
    )

    torch.testing.assert_close(direct, baseline, rtol=3e-2, atol=2e-2)


@GPU_REQUIRED
@BLOCK_PINGPONG_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("quant_kind", ["int8", "fp8"])
def test_router_weight_fold_all_activations(monkeypatch, quant_kind):
    torch.manual_seed(93)
    num_tokens, hidden_size = 1024, 256
    intermediate_size = 128 if quant_kind == "int8" else 320
    num_experts, topk = 256, 8
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=DEVICE, dtype=torch.bfloat16
    ) / 10
    weight_dtype = (
        torch.int8 if quant_kind == "int8" else torch.float8_e4m3fn
    )
    if quant_kind == "int8":
        gated_w1 = torch.randint(
            -127,
            128,
            (num_experts, 2 * intermediate_size, hidden_size),
            device=DEVICE,
            dtype=weight_dtype,
        )
        w2 = torch.randint(
            -127,
            128,
            (num_experts, hidden_size, intermediate_size),
            device=DEVICE,
            dtype=weight_dtype,
        )
    else:
        gated_w1 = (
            torch.randn(
                (num_experts, 2 * intermediate_size, hidden_size),
                device=DEVICE,
            ) / 3
        ).to(weight_dtype)
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size), device=DEVICE
            ) / 3
        ).to(weight_dtype)
    nongated_w1 = gated_w1[:, :intermediate_size].contiguous()
    scale_max = 1e-2 if quant_kind == "int8" else 0.5
    gated_w1_scale = torch.rand(
        (num_experts, 2 * intermediate_size),
        device=DEVICE,
        dtype=torch.float32,
    ) * scale_max
    nongated_w1_scale = gated_w1_scale[:, :intermediate_size].contiguous()
    w2_scale = torch.rand(
        (num_experts, hidden_size), device=DEVICE, dtype=torch.float32
    ) * scale_max
    topk_weights, topk_ids = _routing(num_tokens, num_experts, topk)

    for activation, is_gated, activation_kwargs in ROUTER_FOLD_ACT_CASES:
        kwargs = {
            "output_dtype": hidden_states.dtype,
            "use_int8_w8a8": quant_kind == "int8",
            "use_fp8_w8a8": quant_kind == "fp8",
            "per_channel_quant": True,
            "global_num_experts": num_experts,
            "w1_scale": (
                gated_w1_scale if is_gated else nongated_w1_scale
            ),
            "w2_scale": w2_scale,
            "activation": activation,
            "is_gated": is_gated,
            **activation_kwargs,
        }
        w1 = gated_w1 if is_gated else nongated_w1
        _run_router_fold_pair(
            monkeypatch,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            kwargs,
        )


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("quant_kind", ["int8", "fp8"])
@pytest.mark.parametrize(
    ("activation", "is_gated", "num_tokens", "topk"),
    [
        ("gelu", True, 64, 4),
        ("relu2_no_mul", False, 64, 4),
        ("silu", True, 4, 8),
    ],
)
def test_router_weight_fold_unsorted_activation_with_bias(
    monkeypatch, activation, is_gated, num_tokens, topk, quant_kind
):
    case = _make_router_fold_case(
        seed=94,
        quant_kind=quant_kind,
        num_tokens=num_tokens,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        topk=topk,
        activation=activation,
        is_gated=is_gated,
        with_bias=True,
    )
    monkeypatch.setattr(fused_moe_module, "_bottom_moe_use_mls", lambda: False)
    _run_router_fold_pair(monkeypatch, *case)


@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("quant_kind", ["int8", "fp8"])
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize(
    ("activation", "is_gated"),
    [("silu", True), ("gelu_tanh_no_mul", False)],
)
def test_router_weight_fold_small_m(
    monkeypatch, quant_kind, num_tokens, activation, is_gated
):
    case = _make_router_fold_case(
        seed=96,
        quant_kind=quant_kind,
        num_tokens=num_tokens,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        topk=8,
        activation=activation,
        is_gated=is_gated,
        # Keep tiny FP8 cases safely above the nontrivial-output threshold.
        scale_max=1.0 if quant_kind == "fp8" else None,
    )
    _run_router_fold_pair(monkeypatch, *case)


@GPU_REQUIRED
@torch.inference_mode()
def test_router_weight_fold_multiple_chunks(monkeypatch):
    case = _make_router_fold_case(
        seed=97,
        quant_kind="int8",
        num_tokens=16385,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        topk=2,
        activation="silu",
        is_gated=True,
    )
    _run_router_fold_pair(monkeypatch, *case)


@GPU_REQUIRED
@BLOCK_PINGPONG_REQUIRED
@torch.inference_mode()
def test_router_weight_fold_persistent_mls(monkeypatch):
    """The persistent bottom kernel must accept pre-folded router weights."""
    case = _make_router_fold_case(
        seed=98,
        quant_kind="int8",
        num_tokens=1024,
        hidden_size=256,
        intermediate_size=128,
        num_experts=256,
        topk=8,
        activation="silu",
        is_gated=True,
    )
    monkeypatch.setattr(moe_op_module, "_USE_MOE_PERSISTENT_KERNEL", True)
    _run_router_fold_pair(monkeypatch, *case)


@pytest.fixture(scope="module", params=BENCH_SHAPES)
def fused_moe_bench_weights(request):
    """Reuse one shape's weights across M; all weight preparation is untimed."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device is required")
    mode, experts, n, k, dtype = request.param
    if mode == "fp8_channel" and not arch_info.is_fp8_avail():
        pytest.skip(f"{arch_info.get_arch()} does not support native FP8")

    torch.manual_seed(0)
    shapes = [(experts, 2 * n, k), (experts, k, n)]
    weights, scales = [], []
    for shape in shapes:
        if mode == "bf16":
            weight = torch.randn(shape, device=DEVICE, dtype=dtype).div_(10)
            scale = None
        elif mode == "fp8_channel":
            weight = torch.empty(shape, device=DEVICE, dtype=arch_info.get_fp8_e4m3_dtype())
            scale = torch.empty(shape[:2], device=DEVICE, dtype=torch.float32)
            # Quantize one expert at a time to avoid full-sized FP32 temporaries.
            for expert in range(experts):
                values = torch.randn(shape[1:], device=DEVICE, dtype=dtype).div_(10)
                quantized, expert_scale = pertoken_quant(values, quant_dtype=weight.dtype)
                weight[expert].copy_(quantized)
                scale[expert].copy_(expert_scale.squeeze(-1))
        else:
            weight = torch.randint(-128, 128, shape, device=DEVICE, dtype=torch.int8)
            scale_shape = (
                (experts, (shape[1] + 127) // 128, (shape[2] + 127) // 128)
                if mode == "int8_block" else shape[:2]
            )
            scale = torch.rand(scale_shape, device=DEVICE, dtype=torch.float32).mul_(1e-2)
        weights.append(weight)
        scales.append(scale)

    kwargs = dict(
        activation="silu", is_gated=True, global_num_experts=experts,
        output_dtype=dtype, use_int8_w8a8=mode.startswith("int8"),
        use_fp8_w8a8=mode == "fp8_channel", per_channel_quant=mode.endswith("channel"),
        w1_scale=scales[0], w2_scale=scales[1],
        block_shape=[128, 128] if mode == "int8_block" else None,
    )
    return mode, experts, n, k, dtype, weights, kwargs


@pytest.mark.bench
@GPU_REQUIRED
@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", BENCH_TOKENS, ids=lambda m: f"M{m}")
def test_fused_moe_benchmark(
    fused_moe_bench_weights, num_tokens, bench_warmup, bench_repeat,
):
    """Full MoE graph replay; --bench-repeat is each sample's duration in ms."""
    if bench_warmup < 0 or bench_repeat <= 0:
        raise ValueError("bench warmup must be >= 0 and repeat must be > 0")
    mode, experts, n, k, dtype, weights, kwargs = fused_moe_bench_weights
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, k), device=DEVICE, dtype=dtype).div_(10)
    scores = torch.randn((num_tokens, experts), device=DEVICE, dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden_states, scores, 8, renormalize=False)
    del scores

    def run():
        return fused_experts(hidden_states, *weights, topk_weights, topk_ids, **kwargs)

    output = run()  # Compile and initialize sorting before graph capture.
    assert output.shape == hidden_states.shape and output.dtype == dtype
    assert torch.isfinite(output).all().item()
    del output
    for _ in range(bench_warmup):
        run()
    torch.cuda.synchronize()
    # Same graph timer and five-sample median as the tune CLI's CSV benchmark.
    # do_bench_cudagraph interprets rep as a target measurement duration in ms.
    for attempt in range(3):
        samples_us = [
            float(triton.testing.do_bench_cudagraph(run, rep=bench_repeat)) * 1000
            for _ in range(5)
        ]
        assert all(0 < value < float("inf") for value in samples_us)
        spread = spread_fraction(samples_us)
        if spread <= 0.03:
            break
        print(f"[bench] retry={attempt + 1}/3 spread={spread:.2%}")
    else:
        pytest.fail("MoE graph timings remained unstable")
    print(
        f"[bench] fused_moe arch={arch_info.get_arch()} mode={mode} "
        f"M={num_tokens} E={experts} N={n} K={k} topk=8 dtype={dtype} "
        f"block_shape={kwargs['block_shape']} CUDA-graph median={median(samples_us):.3f} us "
        f"samples_us={[round(value, 3) for value in samples_us]}"
    )


def test_fused_moe_benchmark_does_not_probe_kfd(monkeypatch):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("Operator benchmarks must not inspect KFD queues")

    monkeypatch.setitem(globals(), "DeviceGuard", unexpected_probe)
    monkeypatch.setitem(globals(), "DEVICE", "cpu")
    monkeypatch.setitem(globals(), "fused_topk", lambda *args, **kwargs: (None, None))
    monkeypatch.setitem(globals(), "fused_experts", lambda x, *args, **kwargs: x.clone())
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx938")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", lambda fn, **kwargs: 0.001)
    shape = ("bf16", 1, 2, 4, torch.bfloat16, (None, None), {"block_shape": None})
    test_fused_moe_benchmark(shape, num_tokens=1, bench_warmup=0, bench_repeat=50)


@GPU_REQUIRED
@pytest.mark.parametrize("entry", ["legacy_positional", "legacy_keyword", "wrapper", "weighted"])
@pytest.mark.parametrize("width", [128, 137])
def test_moe_sum_preserves_legacy_launch_contract(entry, width):
    x = (torch.arange(3 * 2 * width).reshape(3, 2, width).float() / 256).bfloat16()
    weights = torch.tensor([[.2, .8], [.7, .3], [.4, .6]])
    expected = torch.zeros(3, width)
    expected_fused = torch.zeros(3, width)
    for rank in range(2):
        value = x[:, rank].float()
        if entry == "weighted":
            # Emulate FP32 FMA on CPU: FP64 holds the product and sum
            # exactly here, followed by one FP32 rounding.
            expected_fused = (expected_fused.double()
                              + value.double() * weights[:, rank, None].double()).float()
            value = value * weights[:, rank, None]
        expected += value
    expected = (expected * 1.25).bfloat16()
    expected_fused = (expected_fused * 1.25).bfloat16()
    device_x = x.cuda()
    result = torch.empty((3, width), dtype=torch.bfloat16, device="cuda")
    core = fused_moe_module
    if entry in ("wrapper", "weighted"):
        if entry == "weighted":
            # The optional route input also supports noncontiguous weights.
            device_weights = torch.stack((weights, weights), -1).cuda()[..., 0]
            core.triton_moe_sum(device_x, result, 1.25, router_weights=device_weights)
        else:
            core.triton_moe_sum(device_x, result, 1.25)
    else:
        strides = (*result.stride(), *device_x.stride())
        if entry == "legacy_positional":
            core.moe_sum_kernel[(3 * triton.cdiv(width, 128),)](
                result, device_x, 3, width, 2, 1.25, 128, *strides,
                triton.language.bfloat16, num_warps=4,
            )
        else:
            core.moe_sum_kernel[(3 * triton.cdiv(width, 128),)](
                result, device_x, M=3, N=width, topk=2, routed_scaling_factor=1.25,
                BLOCK_SIZE=128, stride_output_m=strides[0], stride_output_n=strides[1],
                stride_input_m=strides[2], stride_input_k=strides[3], stride_input_n=strides[4],
                compute_type=triton.language.bfloat16, num_warps=4,
            )
    actual = result.cpu()
    if entry == "weighted":
        # Compilers may fuse each weighted accumulation. At BF16 halfway
        # points, fused and unfused FP32 arithmetic can round differently.
        assert torch.all((actual == expected) | (actual == expected_fused))
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _quantize_mxfp4_reference(x, *, return_scales=False, group_size=128):
    groups = x.float().reshape(x.shape[0], -1, group_size)
    if group_size == 32:
        maximum = groups.abs().amax(-1)
        rounded = ((maximum.view(torch.int32) + 0x200000) & -0x800000).view(torch.float32)
        exponent = (torch.floor(torch.log2(rounded)) - 8).clamp(-127, 127)
        scale = torch.exp2(exponent)
        q = (groups * torch.exp2(-exponent)[..., None]).to(torch.float8_e4m3fn)
        if return_scales:
            return q.reshape(x.shape), (exponent + 127).to(torch.uint8)
        return (q.float() * scale[..., None]).reshape(x.shape)
    maximum = groups.abs().amax(-1)
    # Both native g128 input quantization and packed activation quantization
    # floor the scale after division. The generic Triton fallback differs.
    raw_scale = (maximum.clamp_min(1.e-10) / 448).clamp_min(1.e-10)
    bits = raw_scale.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    q = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    if return_scales:
        return q.reshape(x.shape), scale
    return (q.float() * scale[..., None]).reshape(x.shape)


def _decode_mxfp4_reference(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1., 1.)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _mxfp4_reference_gemm(a, w, dtype=torch.bfloat16):
    # Same scaled K32 accumulation and output conversion as the raw GEMM.
    result = torch.zeros(a.shape[0], w.shape[0])
    for k in range(0, a.shape[-1], 32):
        result += a[:, k:k+32] @ w[:, k:k+32].T
    return result.to(dtype)


@MXFP_REQUIRED
@pytest.mark.parametrize('case', [
    'dense', 'sparse_routing', 'clamp_no_combine', 'chunked', 'prequantized_chunked',
    'static', 'static_bias', 'static_clamp', 'static_input_router',
    'static_input_router_zero', 'static_prequantized_chunked',
    'static_expert_map_chunked', 'static_no_combine', 'static_input_router_no_combine',
    'prequantized_g128_swiglustep', 'prequantized_g128_situ',
    'expert_map_chunked', 'swiglustep', 'situ',
    'input_router', 'input_router_chunked', 'input_router_inplace',
    'input_router_no_combine', 'input_router_fp16',
    'input_router_zero', 'input_router_inplace_chunked',
    'input_router_expert_map_chunked',
])
def test_public_mxfp4_cpu_reference(monkeypatch, case):
    static = case.startswith('static')
    input_router = 'input_router' in case
    no_combine = 'no_combine' in case
    group_size = 32 if static else 128
    clamp = .25 if case in ('clamp_no_combine', 'swiglustep', 'static_clamp') else None
    activation = 'swigluoai' if static else ('swiglustep' if 'swiglustep' in case else ('situ' if 'situ' in case else 'silu'))
    if 'chunked' in case:
        monkeypatch.setenv('TRITON_FUSED_MOE_CHUNK_SIZE', '2')
    torch.manual_seed(429)
    m, experts, hidden, intermediate, topk = 3, 3, 128, 128, 2
    if case == 'sparse_routing':
        experts = 64
    if static:
        intermediate = 96  # Exercise packed K tails and paired gate/up N tails.
    x = (torch.randn(m, hidden) * .2).bfloat16()
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    weights = torch.tensor([[.2, .8], [.7, .3], [.4, .6]])
    if input_router and not static:
        ids, weights = ids[:, :1].contiguous(), weights[:, :1].contiguous()
        topk = 1
    if case in ('input_router_zero', 'static_input_router_zero'):
        weights[1].zero_()
    if case == 'input_router_fp16':
        x = x.half()
    w1 = torch.randint(0, 256, (experts, intermediate*2, hidden//2), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate//2), dtype=torch.uint8)
    s1 = torch.randint(121, 125, (experts, intermediate*2, hidden//32), dtype=torch.uint8)
    s2 = torch.randint(121, 125, (experts, hidden, intermediate//32), dtype=torch.uint8)
    if case == 'swiglustep':
        # Exercise values above 7, so overriding the upstream fixed limit
        # with the deliberately different gemm1_limit=.25 is observable.
        # Exact GEMM1 result: 128 * .125 * 1 == 16 in both halves.
        x.fill_(.125)
        w1.fill_(0x22)
        s1.fill_(127)
    expert_map = None
    if 'expert_map' in case:
        # Global experts 0/2 are local in reversed order; expert 1 is absent.
        expert_map = torch.tensor([1, -1, 0], dtype=torch.int32)
        ids[1, -1] = 1
        w1, w2, s1, s2 = (t[[2, 0]].contiguous() for t in (w1, w2, s1, s2))
    b1, b2 = _decode_mxfp4_reference(w1, s1), _decode_mxfp4_reference(w2, s2)
    static_s1, static_s2 = torch.tensor([.013]), torch.tensor([.017])
    bias1 = torch.randn(w1.shape[:2]) * .15 if static and case != 'static' else None
    bias2 = torch.randn(w2.shape[:2]) * .15 if static and case != 'static' else None
    alpha = 1.3 if case == 'static_clamp' else 1.702
    prepared_x = x * weights.to(x.dtype) if input_router and not static else x
    a = ((x.float() / static_s1).to(torch.float8_e4m3fn).float() if static else
         _quantize_mxfp4_reference(prepared_x, group_size=group_size))
    routes = []
    for token in range(m):
        for rank, global_expert in enumerate(ids[token].tolist()):
            expert = global_expert if expert_map is None else expert_map[global_expert].item()
            if expert == -1:
                routes.append(torch.zeros(hidden, dtype=x.dtype))
                continue
            g1 = _mxfp4_reference_gemm(
                a[token:token+1], b1[expert], torch.float32 if group_size == 32 else x.dtype)
            if static:
                g1 = g1 * static_s1
                if bias1 is not None:
                    g1 += bias1[expert]
            gate, up = g1.float().chunk(2, -1)
            if static:
                limit = 7. if clamp is None else clamp
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                act = gate * torch.sigmoid(alpha * gate) * (up + 1.)
                if input_router:
                    act *= weights[token, rank]
            elif activation == 'situ':
                act = (1.5 * torch.tanh(gate / 1.5) * torch.sigmoid(gate)
                       * (2.0 * torch.tanh(up / 2.0)))
            elif activation == 'swiglustep':
                act = (gate / (1 + torch.exp(-gate))).clamp(max=7.) * up.clamp(-7., 7.)
            else:
                if clamp is not None:
                    gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
                act = up * gate / (1 + torch.exp(-gate))
            # The copied vLLM packed SiLU quantizer rounds through BF16.
            if group_size == 128:
                act = act.bfloat16() if activation == 'silu' else act.to(x.dtype)
            quantized_act = ((act / static_s2).to(torch.float8_e4m3fn).float() if static else
                             _quantize_mxfp4_reference(act, group_size=group_size))
            g2 = _mxfp4_reference_gemm(
                quantized_act, b2[expert],
                torch.float32 if group_size == 32 else x.dtype)[0]
            if static:
                g2 = g2 * static_s2
                if bias2 is not None:
                    g2 += bias2[expert]
            if group_size == 32 and not no_combine and not input_router:
                g2 *= weights[token, rank]
            routes.append(g2.to(x.dtype))
    route_output = torch.stack(routes).reshape(m, topk, hidden)
    if no_combine:
        expected = route_output
    else:
        combined = torch.zeros(m, hidden)
        for k in range(topk):
            combined += route_output[:, k].float() * (1. if input_router or group_size == 32 else weights[:, k, None])
        expected = (combined * 1.25).to(x.dtype)

    core = importlib.import_module('boltops.fused_moe.triton.fused_moe')

    def make_cpu_routes(topk_ids, block_size, num_experts, expert_map=None):
        # Same flattened-route ABI as vLLM's align helper; GPU sorting remains
        # outside this test's scope. The actual global->local mapping matters.
        flat = topk_ids.cpu().flatten().tolist()
        local_map = None if expert_map is None else expert_map.cpu().tolist()
        sorted_routes, expert_blocks = [], []
        for e in range(num_experts):
            selected = [i for i, expert in enumerate(flat) if expert == e]
            padded = ((len(selected) + block_size - 1) // block_size) * block_size
            sorted_routes += selected + [len(flat)] * (padded - len(selected))
            local_expert = e if local_map is None else local_map[e]
            expert_blocks += [local_expert] * (padded // block_size)
        return (torch.tensor(sorted_routes, dtype=torch.int32),
                torch.tensor(expert_blocks, dtype=torch.int32),
                torch.tensor([len(sorted_routes)], dtype=torch.int32))

    def cpu_align(*args):
        return tuple(t.cuda() for t in make_cpu_routes(*args))

    def cpu_ck(topk_ids, topk_weights, num_experts, model_dim, dtype,
               block_size, expert_mask=None):
        assert expert_mask is None
        routes, blocks, padded = make_cpu_routes(topk_ids, block_size, num_experts)
        encoded = torch.where(
            routes < topk_ids.numel(),
            (routes % topk) * (1 << 24) + routes // topk,
            torch.full_like(routes, 0x00ffffff),
        )
        return encoded.cuda(), None, blocks.cuda(), padded.cuda(), None, None

    monkeypatch.setattr(core, 'moe_align_block_size', cpu_align)
    monkeypatch.setattr(core, 'moe_sorting_ck', cpu_ck)
    input_q, input_scale = _quantize_mxfp4_reference(
        x, return_scales=True, group_size=group_size)
    prequantized = 'prequantized' in case
    if static:
        input_q = (x.float() / static_s1).to(torch.float8_e4m3fn)
        input_scale = static_s1
    if case == 'prequantized_g32_fp32':
        input_scale = torch.exp2(input_scale.float() - 127)
    device_x = (input_q if prequantized else x).cuda()
    actual = fused_experts(
        device_x, w1.cuda(), w2.cuda(),
        weights.cuda(), ids.cuda(),
        a1_scale=input_scale.cuda() if prequantized else (static_s1.cuda() if static else None),
        a2_scale=static_s2.cuda() if static else None,
        b1=None if bias1 is None else bias1.cuda(), b2=None if bias2 is None else bias2.cuda(),
        gemm1_alpha=alpha if static else None,
        output_dtype=x.dtype,
        inplace='inplace' in case,
        apply_router_weight_on_input=input_router,
        activation=activation,
        activation_situ_beta=1.5 if activation == 'situ' else None,
        activation_situ_linear_beta=2.0 if activation == 'situ' else None,
        expert_map=None if expert_map is None else expert_map.cuda(),
        use_mxfp4_w4a8=True,
        use_ue8m0=True, w1_scale=s1.cuda(), w2_scale=s2.cuda(),
        block_shape=[1, 32], gemm1_limit=clamp, no_combine=no_combine,
        routed_scaling_factor=1.25,
    )
    if 'inplace' in case:
        assert actual.data_ptr() == device_x.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, rtol=.03, atol=.003)


@pytest.mark.parametrize('prequantized', [False, True])
def test_mxfp4_input_router_rejects_unsupported_preparation(prequantized):
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    if prequantized:
        x = x.to(torch.float8_e4m3fn)
    topk = 1 if prequantized else 2
    match = 'before FP8 quantization' if prequantized else 'require topk=1'
    with pytest.raises(ValueError, match=match):
        fused_experts(
            x, torch.zeros(1, 256, 64, dtype=torch.uint8),
            torch.zeros(1, 128, 64, dtype=torch.uint8),
            torch.ones(1, topk), torch.zeros(1, topk, dtype=torch.int32),
            use_mxfp4_w4a8=True, apply_router_weight_on_input=True,
            w1_scale=torch.full((1, 256, 4), 127, dtype=torch.uint8),
            w2_scale=torch.full((1, 128, 4), 127, dtype=torch.uint8))


def test_public_mxfp4_rejects_incompatible_weight_block():
    # Validation happens before GPU work and must propagate through the public
    # wrapper, rather than silently selecting another quantization algorithm.
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    w1 = torch.zeros(1, 256, 64, dtype=torch.uint8)
    w2 = torch.zeros(1, 128, 64, dtype=torch.uint8)
    with pytest.raises(ValueError, match='weight block_shape must be'):
        fused_experts(
            x, w1, w2, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_mxfp4_w4a8=True, block_shape=[1, 128],
            w1_scale=torch.full((1, 256, 4), 127, dtype=torch.uint8),
            w2_scale=torch.full((1, 128, 4), 127, dtype=torch.uint8),
        )


@pytest.mark.parametrize('quant_flag', ['use_int4_w4a16', 'use_int4_w4a8'])
@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('hidden', [128, 64, 129])
def test_packed_int4_hidden_dimension_is_independent_of_gating(
        monkeypatch, quant_flag, gated, hidden):
    core = importlib.import_module('boltops.fused_moe.triton.fused_moe')

    class DimensionAccepted(Exception):
        pass

    def stop_before_allocation(*args, **kwargs):
        raise DimensionAccepted

    monkeypatch.setattr(core, 'get_optimal_moe_config_func', stop_before_allocation)
    expected = DimensionAccepted if hidden == 128 else AssertionError
    match = None if hidden == 128 else 'Hidden size mismatch'
    # Stop at configuration selection: only the CPU-side public shape
    # contract is under test, without invoking a GPU kernel or quantizer.
    with pytest.raises(expected, match=match):
        fused_experts(
            torch.zeros(1, hidden, dtype=torch.bfloat16),
            torch.zeros(1, 256 if gated else 128, 64, dtype=torch.uint8),
            torch.zeros(1, hidden, 64, dtype=torch.uint8),
            torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            is_gated=gated, **{quant_flag: True})


@pytest.mark.parametrize('case,match', [
    ('missing_scale', 'require a1_scale and output_dtype'),
    ('missing_dtype', 'require a1_scale and output_dtype'),
    ('a2_scale', 'MXFP4 requires gated'),
    ('inplace', 'inplace MXFP4 requires output_dtype'),
])
def test_public_mxfp4_prequantized_contract(case, match):
    x = torch.zeros(1, 128).to(torch.float8_e4m3fn)
    w1 = torch.zeros(1, 256, 64, dtype=torch.uint8)
    w2 = torch.zeros(1, 128, 64, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        fused_experts(
            x, w1, w2, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_mxfp4_w4a8=True,
            a1_scale=None if case == 'missing_scale' else torch.ones(1, 1),
            output_dtype=None if case == 'missing_dtype' else torch.bfloat16,
            a2_scale=torch.ones(1, 1) if case == 'a2_scale' else None,
            inplace=case == 'inplace',
            w1_scale=torch.full((1, 256, 4), 127, dtype=torch.uint8),
            w2_scale=torch.full((1, 128, 4), 127, dtype=torch.uint8),
        )


@pytest.mark.parametrize('prequantized', [False, True])
def test_public_fake_output_contract(prequantized):

    with FakeTensorMode():
        dtype = torch.float8_e4m3fn if prequantized else torch.float32
        x = torch.empty(2, 128, dtype=dtype)
        ids = torch.empty(2, 2, dtype=torch.int32)
        w1 = torch.empty(3, 256, 64 if prequantized else 128,
                         dtype=torch.uint8 if prequantized else dtype)
        w2 = torch.empty(3, 128, 64 if prequantized else 128,
                         dtype=torch.uint8 if prequantized else dtype)
        out = fused_experts(
            x, w1, w2, torch.empty(2, 2), ids, no_combine=True,
            use_mxfp4_w4a8=prequantized,
            a1_scale=torch.empty(2, 1) if prequantized else None,
            output_dtype=torch.bfloat16 if prequantized else None,
            w1_scale=torch.empty(3, 256, 4, dtype=torch.uint8) if prequantized else None,
            w2_scale=torch.empty(3, 128, 4, dtype=torch.uint8) if prequantized else None,
        )
        assert out.shape == (2, 2, 128)
        assert out.dtype == (torch.bfloat16 if prequantized else torch.float32)


def test_moe_fake_preserves_legacy_fn_key_position():
    signature = inspect.signature(fused_moe_module.fused_moe_fake)
    names = list(signature.parameters)
    # fn_key was the last positional parameter before adding MX flags.
    assert names.index("fn_key") == 33
    x = torch.empty(2, 128)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    bound = signature.bind_partial(x, torch.empty(3, 256, 128),
                                   torch.empty(3, 128, 128), torch.ones(2, 2), ids)
    bound.apply_defaults()
    legacy_args = list(bound.args[:34])
    legacy_args[-1] = "legacy-key"
    assert signature.bind(*legacy_args).arguments["fn_key"] == "legacy-key"
    result = fused_moe_module.fused_moe_fake(*legacy_args)
    assert result.shape == x.shape and result.dtype == x.dtype


@pytest.mark.parametrize("flag", ["use_mxfp4_w4a8", "use_mxfp8"])
def test_public_fake_accepts_positional_mx_flags(flag):
    parameters = [p for p in inspect.signature(fused_moe_module.fused_moe_fake).parameters.values()
                  if p.name != "fn_key"]
    signature = inspect.Signature(parameters)
    with FakeTensorMode():
        x = torch.empty(2, 128, dtype=torch.bfloat16)
        bound = signature.bind_partial(
            x, torch.empty(3, 256, 128), torch.empty(3, 128, 128),
            torch.empty(2, 2), torch.empty(2, 2, dtype=torch.int32),
            no_combine=True, gemm1_beta=.3, **{flag: True},
        )
        bound.apply_defaults()
        result = torch.ops.boltops.fused_experts_impl(*bound.args)
        assert result.shape == (2, 2, 128) and result.dtype == x.dtype


def _quantize_w8a8_reference(x, dtype):
    limit = 127 if dtype == torch.int8 else 448
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1.e-10) / limit
    q = x.float() / scale
    if dtype == torch.int8:
        q = q.round()
    return q.to(dtype).float(), scale


@GPU_REQUIRED
@pytest.mark.parametrize('dtype,prequantized,persistent,fold', [
    (torch.int8, False, False, False), (torch.float8_e4m3fn, False, False, False),
    (torch.float8_e4m3fn, True, False, False), (torch.float8_e4m3fn, False, True, False),
    (torch.float8_e4m3fn, False, False, True),
], ids=['int8', 'fp8', 'fp8-prequantized-chunked', 'fp8-persistent-bias', 'fp8-router-fold'])
def test_public_w8a8_dequant_bias_then_router(monkeypatch, dtype, prequantized, persistent, fold):
    if persistent:
        op = importlib.import_module('boltops.fused_moe.triton.moe_op')
        monkeypatch.setattr(op, '_USE_MOE_PERSISTENT_KERNEL', True)
    if prequantized:
        monkeypatch.setenv('TRITON_FUSED_MOE_CHUNK_SIZE', '2')
    torch.manual_seed(617)
    m, experts, width, topk = 3, 3, 128, 2
    x = (torch.randn(m, width) * .1).bfloat16()
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    router = torch.tensor([[.003, .997], [.15, .85], [.9, .1]])
    if dtype == torch.int8:
        w1 = torch.randint(-64, 64, (experts, 2*width, width), dtype=dtype)
        w2 = torch.randint(-64, 64, (experts, width, width), dtype=dtype)
        scale_max = .02
    else:
        w1 = (torch.randn(experts, 2*width, width) * 3).to(dtype)
        w2 = (torch.randn(experts, width, width) * 3).to(dtype)
        scale_max = .1
    s1 = torch.rand(experts, 2*width) * scale_max + .001
    s2 = torch.rand(experts, width) * scale_max + .001
    b1 = (torch.randn(experts, 2*width) * .03).bfloat16()
    b2 = (torch.randn(experts, width) * .03).bfloat16()
    aq, sa = _quantize_w8a8_reference(x, dtype)
    routes = []
    for token in range(m):
        for rank, expert in enumerate(ids[token].tolist()):
            acc1 = aq[token] @ w1[expert].float().T
            g1 = (acc1 * sa[token] * s1[expert] + b1[expert].float()).bfloat16()
            gate, up = g1.float().chunk(2)
            act = (gate / (1 + torch.exp(-gate)) * up).bfloat16()
            q2, sa2 = _quantize_w8a8_reference(act[None, :], dtype)
            acc2 = q2[0] @ w2[expert].float().T
            # vLLM: dequantize -> add bias -> multiply router -> cast.
            if fold:
                out = (acc2 * (sa2[0] * router[token, rank]) * s2[expert]
                       + b2[expert].float() * router[token, rank])
            else:
                out = (acc2 * sa2[0] * s2[expert] + b2[expert].float()) * router[token, rank]
            routes.append(out.bfloat16())
    routes = torch.stack(routes).reshape(m, topk, width)
    expected = (routes[:, 0].float() + routes[:, 1].float()).bfloat16()

    core = importlib.import_module('boltops.fused_moe.triton.fused_moe')
    if not fold:
        # Upstream-order oracle cases isolate dequant/bias/router ordering.
        monkeypatch.setattr(core, '_should_fold_router_weight', lambda **kwargs: False)

    def config_query(*args, is_bottom=False, **kwargs):
        def query(_m):
            cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=32,
                       GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                       num_warps=4, num_stages=1)
            return (cfg, 16) if is_bottom else cfg
        return query

    def cpu_sort(*args):
        routes, blocks, sorted_weights = [], [], []
        chunk_ids = args[0].cpu().flatten().tolist()
        chunk_weights = args[1].cpu().flatten().tolist()
        for expert in range(experts):
            selected = [i for i, e in enumerate(chunk_ids) if e == expert]
            if not selected:
                continue
            routes.extend([(i % topk) * (1 << 24) + i // topk for i in selected])
            routes.extend([0x00ffffff] * (16 - len(selected)))
            sorted_weights.extend([chunk_weights[i] for i in selected])
            sorted_weights.extend([0.] * (16 - len(selected)))
            blocks.append(expert)
        return (torch.tensor(routes, dtype=torch.int32).cuda(),
                torch.tensor(sorted_weights).cuda(),
                torch.tensor(blocks, dtype=torch.int32).cuda(),
                torch.tensor([len(routes)], dtype=torch.int32).cuda(), None, None)

    monkeypatch.setattr(core, 'get_optimal_moe_config_func', config_query)
    monkeypatch.setattr(core, 'moe_sorting_ck', cpu_sort)
    actual = fused_experts(
        (aq.to(dtype) if prequantized else x).cuda(),
        w1.cuda(), w2.cuda(), router.cuda(), ids.cuda(),
        a1_scale=sa.cuda() if prequantized else None, output_dtype=torch.bfloat16,
        b1=b1.cuda(), b2=b2.cuda(), w1_scale=s1.cuda(), w2_scale=s2.cuda(),
        use_int8_w8a8=dtype == torch.int8, use_fp8_w8a8=dtype != torch.int8,
        per_channel_quant=True,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=.03, atol=.003)


@MXFP_REQUIRED
@pytest.mark.parametrize('case', [
    'dynamic', 'sparse_routing', 'dynamic_fp16', 'strided_routing', 'bf16_router',
    'chunked_ep', 'chunked_ep_invalid', 'chunked_ep_strided_router',
    'no_combine', 'swigluoai', 'silu_alpha', 'shared',
])
def test_public_mxfp8_cpu_reference(monkeypatch, case):
    torch.manual_seed(673)
    m, e, k, n, topk = 3, 3, 128, 128, 2
    if case == 'sparse_routing':
        e = 64
    group, bo = 32, 1
    x = (torch.randn(m, k) * .2).bfloat16()
    if case == 'dynamic_fp16':
        x = x.half()
    w1 = (torch.randn(e, 2*n, k) * 2).to(torch.float8_e4m3fn)
    w2 = (torch.randn(e, k, n) * 2).to(torch.float8_e4m3fn)
    s1 = torch.randint(121, 125, (e, 2*n//bo, k//group), dtype=torch.uint8)
    s2 = torch.randint(121, 125, (e, k//bo, n//group), dtype=torch.uint8)
    b1 = w1.float() * torch.exp2(s1.float()-127).repeat_interleave(bo, 1).repeat_interleave(group, 2)
    b2 = w2.float() * torch.exp2(s2.float()-127).repeat_interleave(bo, 1).repeat_interleave(group, 2)
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    weights = torch.tensor([[.2, .8], [.7, .3], [.4, .6]])
    if case == 'bf16_router':
        weights = weights.bfloat16().float()
    if case == 'chunked_ep_invalid':
        ids[0, 0], ids[1, 1], ids[2, 0] = -1, e, 1
    expert_map = torch.tensor([0, -1, 1], dtype=torch.int32) if case.startswith('chunked_ep') else None
    if expert_map is not None:
        monkeypatch.setenv('TRITON_FUSED_MOE_CHUNK_SIZE', '2')
    def native_quant(value):
        blocked = value.float().reshape(value.shape[0], -1, 32)
        amax = blocked.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
        exponent = (torch.ceil(torch.log2(amax / 448.)) + 127).clamp(0, 254)
        q = (blocked * torch.exp2(127 - exponent)[..., None]).to(torch.float8_e4m3fn)
        return (q.float() * torch.exp2(exponent - 127)[..., None]).reshape(value.shape)
    av = native_quant(x)
    expected = torch.zeros(m, topk, k, dtype=torch.float32)
    for token in range(m):
        for rank, ge in enumerate(ids[token].tolist()):
            if ge < 0 or ge >= e:
                continue
            expert = ge if expert_map is None else expert_map[ge].item()
            if expert < 0:
                continue
            g1 = av[token:token+1] @ b1[expert].T
            g1 = g1.to(x.dtype).float()
            gate, up = g1.chunk(2, -1)
            if case == 'swigluoai':
                gate, up = gate.clamp(max=.15), up.clamp(-.15, .15)
                act = gate * torch.sigmoid(1.2 * gate) * (up + .3)
            else:
                act = gate * torch.sigmoid((1.2 if case == 'silu_alpha' else 1.) * gate) * up
            a2 = native_quant(act)
            g2 = (a2 @ b2[expert].T)[0]
            if case != 'no_combine':
                g2 *= weights[token, rank]
            expected[token, rank] = g2
    expected = (expected if case == 'no_combine' else expected.sum(1)).to(x.dtype)

    core = importlib.import_module('boltops.fused_moe.triton.fused_moe')
    def align(topk_ids, block_size, num_experts, local_map=None):
        flat = topk_ids.cpu().flatten().tolist()
        mapping = None if local_map is None else local_map.cpu().tolist()
        routes, blocks = [], []
        for expert in range(num_experts):
            selected = [i for i, value in enumerate(flat) if value == expert]
            padded = triton.cdiv(len(selected), block_size) * block_size
            routes += selected + [len(flat)] * (padded - len(selected))
            blocks += [expert if mapping is None else mapping[expert]] * (padded // block_size)
        return tuple(t.cuda() for t in (torch.tensor(routes, dtype=torch.int32),
                     torch.tensor(blocks, dtype=torch.int32), torch.tensor([len(routes)], dtype=torch.int32)))
    def ck(topk_ids, topk_weights, num_experts, model_dim, dtype, block_size, expert_mask=None):
        assert topk_ids.is_contiguous()
        assert topk_weights.is_contiguous() and topk_weights.dtype == torch.float32
        routes, blocks, padded = align(topk_ids, block_size, num_experts)
        cpu = routes.cpu()
        safe = cpu.clamp_max(topk_ids.numel()-1)
        sw = topk_weights.cpu().flatten()[safe]
        encoded = torch.where(cpu < topk_ids.numel(), (cpu % topk) * (1 << 24) + cpu // topk,
                              torch.full_like(cpu, 0x00ffffff))
        return encoded.cuda(), sw.cuda(), blocks, padded, None, None
    if case != 'chunked_ep_invalid':
        monkeypatch.setattr(core, 'moe_align_block_size', align)
    monkeypatch.setattr(core, 'moe_sorting_ck', ck)
    device_weights = (torch.stack((weights, weights), -1).cuda()[..., 0]
                      if case in ('chunked_ep_strided_router', 'strided_routing') else weights.cuda())
    if case == 'bf16_router':
        device_weights = device_weights.bfloat16()
    device_ids = (torch.stack((ids, ids), -1).cuda()[..., 0]
                  if case == 'strided_routing' else ids.cuda())
    actual = fused_experts(
        x.cuda(), w1.cuda(), w2.cuda(), device_weights, device_ids,
        activation='swigluoai' if case == 'swigluoai' else 'silu',
        gemm1_alpha=1.2 if case in ('swigluoai', 'silu_alpha') else None,
        gemm1_beta=.3 if case == 'swigluoai' else None,
        gemm1_limit=.15 if case == 'swigluoai' else None,
        use_mxfp8=True, w1_scale=s1.cuda(), w2_scale=s2.cuda(), block_shape=[bo, group],
        output_dtype=x.dtype,
        expert_map=expert_map.cuda() if expert_map is not None else None,
        global_num_experts=e-1 if case == 'shared' else e, no_combine=case == 'no_combine',
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=.025, atol=.003)


@pytest.mark.parametrize('case', ['block128', 'fp32_scale', 'packed_scale'])
def test_public_mxfp8_rejects_non_native_recipe(case):
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    w1 = torch.zeros(1, 256, 128, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn)
    scale_dtype = {'block128': torch.uint8, 'fp32_scale': torch.float32,
                   'packed_scale': torch.int32}[case]
    with pytest.raises(ValueError, match='native MXFP8 requires'):
        fused_experts(
            x, w1, w2, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_mxfp8=True, block_shape=[128, 128] if case == 'block128' else [1, 32],
            w1_scale=torch.ones(1, 256, 4, dtype=scale_dtype),
            w2_scale=torch.ones(1, 128, 4, dtype=scale_dtype))


@pytest.mark.parametrize('case', ['bias', 'input_router', 'prequantized', 'output_dtype'])
def test_public_mxfp8_rejects_unsupported_native_options(case):
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    if case == 'prequantized':
        x = x.to(torch.float8_e4m3fn)
    w1 = torch.zeros(1, 256, 128, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='native MXFP8'):
        fused_experts(
            x, w1, w2, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_mxfp8=True, b1=torch.zeros(1, 256) if case == 'bias' else None,
            apply_router_weight_on_input=case == 'input_router',
            output_dtype=torch.float32 if case == 'output_dtype' else None,
            w1_scale=torch.ones(1, 256, 4, dtype=torch.uint8),
            w2_scale=torch.ones(1, 128, 4, dtype=torch.uint8))


def test_public_mxfp4_rejects_unverified_dynamic_g32():
    with pytest.raises(ValueError, match='dynamic MXFP4 fused_experts requires G128'):
        fused_experts(
            torch.zeros(1, 128, dtype=torch.float8_e4m3fn),
            torch.zeros(1, 256, 64, dtype=torch.uint8),
            torch.zeros(1, 128, 64, dtype=torch.uint8),
            torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_mxfp4_w4a8=True, output_dtype=torch.bfloat16,
            a1_scale=torch.full((1, 4), 127, dtype=torch.uint8),
            w1_scale=torch.full((1, 256, 4), 127, dtype=torch.uint8),
            w2_scale=torch.full((1, 128, 4), 127, dtype=torch.uint8))
