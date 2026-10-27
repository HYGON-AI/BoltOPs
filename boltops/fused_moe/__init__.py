# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Fused MoE operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.
"""

from __future__ import annotations

from typing import List, Optional

import torch

def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: Optional[str] = None,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    is_gated: Optional[bool] = None,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int4_w4a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    output_dtype: Optional[torch.dtype] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = 1.0,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Run fused MoE experts."""
    from boltops.fused_moe.triton import fused_experts as impl

    return impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        b1=b1,
        b2=b2,
        is_gated=is_gated,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        output_dtype=output_dtype,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


def normalize_moe_activation(
    activation: str,
    is_gated: Optional[bool] = None,
) -> tuple[str, bool]:
    """Normalize an activation name and resolve gated mode."""
    from boltops.fused_moe.triton.moe_activation import (
        normalize_moe_activation as impl,
    )

    return impl(activation, is_gated)


def moe_activation_output_size(n: int, is_gated: bool) -> int:
    """Return the output width for an input width and gated mode."""
    from boltops.fused_moe.triton.moe_activation import (
        moe_activation_output_size as impl,
    )

    return impl(n, is_gated)

def moe_activation(
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    activation: str = "silu",
    is_gated: Optional[bool] = None,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> None:
    """Apply an MoE activation into caller-provided 2D output storage."""
    from boltops.fused_moe.triton.moe_activation import (
        moe_activation as impl,
    )

    impl(
        activation=activation,
        is_gated=is_gated,
        activated_out=activated_out,
        ffn1_out_2d=ffn1_out_2d,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


__all__ = [
    "fused_experts",
    "normalize_moe_activation",
    "moe_activation_output_size",
    "moe_activation",
]
