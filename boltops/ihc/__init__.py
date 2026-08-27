"""Public interfaces for Independent Hyper-Connections (IHC)."""

from __future__ import annotations

import torch


def ihc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    magnitude: float,
    layer_in: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
):
    from .tilelang._dispatch import resolve

    return resolve("ihc_pre", residual.device)(
        residual, fn, scale, base, rms_eps, hc_eps, magnitude, layer_in, post
    )


def ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from .tilelang._dispatch import resolve

    return resolve("ihc_post", residual.device)(x, residual, post, out)


def ihc_head(
    residual: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    hidden: torch.Tensor | None = None,
) -> torch.Tensor:
    from .tilelang._dispatch import resolve

    return resolve("ihc_head", residual.device)(
        residual, head_fn, head_scale, head_base, rms_eps, hc_eps, hidden
    )


__all__ = ["ihc_head", "ihc_post", "ihc_pre"]
