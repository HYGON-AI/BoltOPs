"""Gated-delta WY recomputation operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute the gated-delta WY tensors ``w`` and ``u``."""
    from bolt_ops.recompute_w_u.triton.recompute_w_u import recompute_w_u_fwd as impl

    return impl(k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices)


__all__ = ["recompute_w_u_fwd"]
