"""Local replacements for non-Triton helpers used by the migrated MoE code."""

from __future__ import annotations

from typing import Optional

import torch


def pertoken_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.int8,
):
    values = x.float()
    dtype_max = (
        torch.finfo(quant_dtype).max
        if quant_dtype.is_floating_point
        else torch.iinfo(quant_dtype).max
    )
    if scale is None:
        scale = values.abs().amax(dim=-1, keepdim=True) / dtype_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / scale).to(quant_dtype), scale.float()


def per_token_quant_hip(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.int8,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
):
    if scale is not None:
        raise ValueError("unsupported: static per token quant")
    if num_rows is not None or num_rows_factor != 1:
        raise ValueError("num_rows quantization is not used by fused_moe")

    from bolt_ops.fused_moe.triton.quant import dynamic_per_token_quant_fp8_i8

    shape = x.shape
    output = torch.empty(shape, dtype=quant_dtype, device=x.device)
    scale = torch.empty((*shape[:-1], 1), dtype=torch.float32, device=x.device)
    dynamic_per_token_quant_fp8_i8(
        output.view(-1, shape[-1]), x.view(-1, shape[-1]), scale.view(-1)
    )
    return output, scale


def per_block_quant_wrapper(block_shape=(1, 128)):
    def decorator(per_token_quant_func):
        def wrapper(x, scale=None, quant_dtype=torch.int8):
            block_m, block_n = block_shape
            assert x.shape[-1] % block_n == 0, (
                f"block size {block_n} not match {x.shape[-1]}"
            )
            assert block_m == 1, "only support 1xN block, TODO: support MxN"
            m, n = x.shape
            values, scales = per_token_quant_func(
                x.view(-1, block_n), scale=scale, quant_dtype=quant_dtype
            )
            return values.view(m, n), scales.view(m, n // block_n)

        return wrapper

    return decorator


def _align_with_local_triton(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
):
    # Imported lazily to avoid a cycle while fused_moe.py is being initialized.
    from bolt_ops.fused_moe.triton import moe_align_block_size_triton

    return moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


sgl_moe_align_block_size = _align_with_local_triton
moe_align_block_size = _align_with_local_triton


__all__ = [
    "moe_align_block_size",
    "per_block_quant_wrapper",
    "per_token_quant_hip",
    "pertoken_quant",
    "sgl_moe_align_block_size",
]
