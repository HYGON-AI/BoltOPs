"""Correctness tests for the public fused MoE operator."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from boltops.fused_moe import fused_experts  # noqa: E402
from boltops.tools.fused_moe_triton_tune.common import (  # noqa: E402
    quantize_weights,
    scalar_types,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
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
    quantized = (groups.float() / scales).clamp(-128, 127).to(torch.int8)
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
