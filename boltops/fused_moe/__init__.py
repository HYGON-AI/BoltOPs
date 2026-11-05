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
    use_mxfp4_w4a8: bool = False,
    use_ue8m0: bool = False,
    use_mxfp8: bool = False,
    gemm1_beta: Optional[float] = None,  # Up offset; None selects SiLU=0 or SwiGLU-OAI=1.
) -> torch.Tensor:
    """Run fused MoE experts with caller-provided top-k routing.

    ``use_mxfp4_w4a8`` uses packed E2M1 weights with K32 UE8M0 scales.
    The default dynamic recipe follows vLLM DeepGemmFP4Experts: A groups
    are K128, GEMM1/2 use the floating output dtype, and router weights are
    applied after GEMM2 conversion. Prequantized E4M3 input requires G128
    a1_scale and output_dtype. G32 dynamic activations are not exposed by
    this entry; their complete upstream execution flow is not aligned.

    Providing both a1_scale and a2_scale as scalar FP32 multipliers selects
    vLLM's AITER static-FP8 W4A8 recipe with activation="swigluoai": unit A
    microscale, scalar A scale after accumulation, FP32 bias, fused SwiGLU-OAI
    and static output quantization. gemm1_alpha/limit default to 1.702/7,
    with +1 on up. apply_router_weight_on_input places gamma after GEMM1
    activation, before the FP8 cast; otherwise gamma follows GEMM2 bias
    before output conversion. Gate/up weights use BoltOPs split layout;
    prepare upstream interleaved or swizzled checkpoint layouts outside
    experts. Static scales must be positive finite values supplied by the
    caller; no per-forward host readback is performed.

    ``use_mxfp8`` uses native E4M3 x E4M3 scaled dot in the regular,
    non-COMBINE_SCALE_LOAD expert kernel. Default A/B groups are K32 with
    uint8 E8M0 scales and vLLM 2**ceil(log2(amax/448)) scale selection. GEMM1
    stores the floating output dtype; GEMM2 keeps weighted routes in FP32
    until topk reduction and the final output cast. Native G32 supports
    split SwiGLU-OAI with gemm1_alpha/gemm1_beta/gemm1_limit (default 1.702/1/no
    clamp); SiLU defaults to alpha=1,beta=0. A/B scales are uint8 E8M0 per
    K32 group, with weight block_shape=[1,32]. Inputs must be BF16/FP16 and
    output_dtype must match the input dtype; activations are quantized inside
    experts. Prequantized input, a1_scale/a2_scale, bias and input routing
    weights are unsupported by this native path. G128/block broadcasting
    and FP32/packed-int32 MXFP8 scales are unsupported. Prepare checkpoint
    layouts outside experts.
    FNUZ is unsupported. MXFP8 and ordinary FP8 flags are mutually exclusive.

    ``use_ue8m0`` follows vLLM's FP8 quantizer naming. Ordinary block FP8
    uses ``block_shape`` for grouping and rounds scales up to powers of two.
    Native MXFP4 selects its scale algorithm from the activation recipe;
    ``block_shape`` describes its fixed [1,32] weight groups only.
    Ordinary FP8 scale arguments are decoded floating-point multipliers.
    Convert checkpoint UE8M0 scales once during weight preparation using
    ``moe_compat._upcast_e8m0_to_fp32``; experts does not decode static scales
    on every invocation. Native MXFP4 keeps its encoded scale containers.
    BoltOPs retains its flat quantization arguments; current vLLM places
    these settings in quant_config/backend descriptors.
    """
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
        use_mxfp4_w4a8=use_mxfp4_w4a8,
        use_ue8m0=use_ue8m0,
        use_mxfp8=use_mxfp8,
        gemm1_beta=gemm1_beta,
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
