"""Attention-residual fusion operator package."""

from __future__ import annotations

import torch


def attn_res(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    """Normalize, score, and mix residual sources with one Triton kernel."""
    from bolt_ops.attn_res.triton.attn_res import attn_res as impl

    return impl(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)


__all__ = ["attn_res"]
