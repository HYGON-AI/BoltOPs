"""Public interfaces for the Generic operator family.

Operators without a dedicated subpackage declare their public signatures here
and lazily import the Triton implementation (matching the GEMM / MoE family
convention). The causal_conv1d family is re-exported lazily from
``boltops.generic.triton.causal_conv1d`` via :func:`__getattr__`: the
signatures live in one place (``api.py`` / ``sglang.py`` / ``vllm.py``), so
they cannot drift from the implementation, and ``import boltops.generic``
still works without Triton installed (it is an optional dependency).
"""

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
    from boltops.generic.triton.attn_res import attn_res as impl

    return impl(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    """Scatter selected speculative Mamba states into ``dst`` in place."""
    from boltops.generic.triton.mamba_state_scatter import (
        fused_mamba_state_scatter_with_mask as impl,
    )

    return impl(dst, src, dst_indices_raw, step_indices_raw)


_CAUSAL_CONV1D_EXPORTS = frozenset(
    {
        "causal_conv1d_fn",
        "causal_conv1d_update",
    }
)


def __getattr__(name: str):
    """Lazily re-export the causal_conv1d entry points.

    ``boltops.generic.triton.causal_conv1d`` already exposes these names with
    their real signatures; this keeps ``boltops.generic`` importable without
    Triton while avoiding duplicated wrapper signatures in this module.
    """
    if name in _CAUSAL_CONV1D_EXPORTS:
        import importlib

        mod = importlib.import_module("boltops.generic.triton.causal_conv1d")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "attn_res",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_mamba_state_scatter_with_mask",
]
