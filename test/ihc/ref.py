# ruff: noqa
"""Eager formulas from Hy4-p vllm/models/hy_v4/nvidia/hc.py."""

from __future__ import annotations

import torch


def ihc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hc, hidden = residual.shape
    x_flat = residual.reshape(num_tokens, hc * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, fn.t()) * rsqrt
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = magnitude * torch.sigmoid(mixes[..., hc:] * scale[1] + base[hc:]) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return y.to(residual.dtype), post


def ihc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    y = post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
    return y.to(residual.dtype)


def ihc_head_ref(
    residual: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, hc, hidden = residual.shape
    x_flat = residual.reshape(num_tokens, hc * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, head_fn.t()) * rsqrt
    pre = torch.sigmoid(mixes * head_scale.reshape(()).float() + head_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return y.to(residual.dtype)
