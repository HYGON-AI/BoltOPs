"""Triton backend for Kernelized Delta Attention.

Standalone vendor of the directly callable Kimi-K3 KDA Triton kernels from
vLLM's AMD implementation. vLLM imports are replaced by local compatibility
helpers; orchestration that depends on vLLM's shared FLA modules is excluded.
"""

from .chunk import (
    chunk_kda,
    chunk_kda_fwd,
    chunk_kda_with_fused_gate,
    chunk_kda_with_fused_gate_fwd,
    chunk_gla_fwd_o_gk,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
)
from .chunk_intra import chunk_kda_fwd_intra
from .chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from .fused_recurrent import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)


def recompute_w_u_fwd(k, v, beta, A, q=None, gk=None, cu_seqlens=None, chunk_indices=None):
    """KDA/vLLM WY-recomputation facade returning ``(w, u, qg, kg)``."""
    from .chunk import recompute_w_u_fwd as recompute_w_u_fwd_kda

    return recompute_w_u_fwd_kda(k, v, beta, A, q, gk, cu_seqlens, chunk_indices)

__all__ = [
    "chunk_gla_fwd_o_gk",
    "chunk_kda",
    "chunk_kda_fwd",
    "chunk_kda_with_fused_gate",
    "chunk_kda_with_fused_gate_fwd",
    "chunk_kda_fwd_intra",
    "chunk_kda_fwd_intra_token_parallel",
    "fused_kda_gate",
    "fused_kda_gate_chunk_cumsum",
    "fused_recurrent_kda",
    "fused_recurrent_kda_fwd",
    "fused_recurrent_kda_packed_decode",
    "recompute_w_u_fwd",
]
