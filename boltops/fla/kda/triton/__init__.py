"""Triton backend for Kernelized Delta Attention.

Standalone vendor of the directly callable Kimi-K3 KDA Triton kernels from
vLLM's AMD implementation. vLLM imports are replaced by local compatibility
helpers; orchestration that depends on vLLM's shared FLA modules is excluded.
"""

from __future__ import annotations

import torch


from .chunk import (
    chunk_kda,
    chunk_kda_fwd,
    chunk_kda_with_fused_gate,
    chunk_gla_fwd_o_gk,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
)
from .chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from .chunk_state import (
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_fwd_state_output_fused,
)
from .fused_recurrent import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)
def recompute_w_u_fwd(k, v, beta, A, q=None, gk=None, cu_seqlens=None, chunk_indices=None):
    """KDA/vLLM WY-recomputation facade returning ``(w, u, qg, kg)``."""
    from .chunk import recompute_w_u_fwd as recompute_w_u_fwd_kda

    return recompute_w_u_fwd_kda(k, v, beta, A, q, gk, cu_seqlens, chunk_indices)


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size=64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the intra public contract without implementation-only options."""
    from .chunk_intra import chunk_kda_fwd_intra as impl

    return impl(
        q,
        k,
        gk=gk,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Expose the fused-gate prefill contract without implementation options."""
    from .chunk import chunk_kda_with_fused_gate_fwd as impl

    return impl(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        g_bias,
        scale,
        initial_state,
        output_final_state,
        lower_bound,
        cu_seqlens,
    )

__all__ = [
    "chunk_gla_fwd_o_gk",
    "chunk_kda",
    "chunk_kda_fwd",
    "chunk_kda_with_fused_gate",
    "chunk_kda_with_fused_gate_fwd",
    "chunk_kda_fwd_intra",
    "chunk_kda_fwd_intra_token_parallel",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_kda_fwd_state_output_fused",
    "fused_kda_gate",
    "fused_kda_gate_chunk_cumsum",
    "fused_recurrent_kda",
    "fused_recurrent_kda_fwd",
    "fused_recurrent_kda_packed_decode",
    "recompute_w_u_fwd",
]
