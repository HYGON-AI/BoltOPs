# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and CUDA-graph benchmarks for the public fused MoE operator."""

from __future__ import annotations

from statistics import median
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

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
    DeviceGuard,
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
    guard = DeviceGuard(torch.cuda.current_device())
    for attempt in range(3):
        guard.wait_until_exclusive()
        with guard:
            samples_us = [
                float(triton.testing.do_bench_cudagraph(run, rep=bench_repeat)) * 1000
                for _ in range(5)
            ]
        assert all(0 < value < float("inf") for value in samples_us)
        spread = spread_fraction(samples_us)
        if not guard.contaminated and spread <= 0.03:
            break
        print(f"[bench] retry={attempt + 1}/3 spread={spread:.2%} "
              f"foreign_queue={guard.contaminated}")
    else:
        pytest.fail("MoE graph timings remained unstable or the GPU was shared")
    print(
        f"[bench] fused_moe arch={arch_info.get_arch()} mode={mode} "
        f"M={num_tokens} E={experts} N={n} K={k} topk=8 dtype={dtype} "
        f"block_shape={kwargs['block_shape']} CUDA-graph median={median(samples_us):.3f} us "
        f"samples_us={[round(value, 3) for value in samples_us]}"
    )
