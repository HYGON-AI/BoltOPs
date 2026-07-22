"""Mamba state scatter operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.
"""

from __future__ import annotations

import torch


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    """Scatter selected speculative Mamba states into ``dst`` in place."""
    from bolt_ops.mamba_state_scatter.triton.mamba_state_scatter import (
        fused_mamba_state_scatter_with_mask as impl,
    )

    return impl(dst, src, dst_indices_raw, step_indices_raw)


__all__ = ["fused_mamba_state_scatter_with_mask"]
