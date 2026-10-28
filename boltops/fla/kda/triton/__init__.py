# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# AMD/ROCm vendored copy of the Kimi-K3 KDA triton kernels.
#
# Provenance: mirror of vllm/models/kimi_k3/nvidia/ops/third_party/kda (tracker
# mke-tracker @ 7adebfcf9; FLA vendored per PRs #39/#86). Split per-vendor so
# AMD can carry gfx950-specific kernel changes without touching the NVIDIA copy.
#
# fla-org/flash-linear-attention#869 (the unmerged ROCm fixes our earlier
# amd_fla shim carried against FLA 0.5.0) is covered here by the *newer* vendored
# FLA rather than the literal patch:
#   - transpose-state-layout workaround: N/A (kernels rewritten; no
#     transpose_state_layout path remains),
#   - AMD autotune configs: present (is_amd num_warps/num_stages branches),
#   - OOB-mask correctness fix: present (all tl.load use mask=..., other=0).
# Validated on gfx950: no core-dump, gsm8k 94.1%.
#
# AMD-specific deltas vs the NVIDIA copy: NONE yet (byte-identical). Keep in sync
# with the NVIDIA copy on FLA updates; any divergence should be an intentional,
# documented gfx950-specific change (a #869-style AMD-only fix).
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/vllm/models/kimi_k3/amd/ops/third_party/kda/__init__.py
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
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    null_state_index: int = -1,
):
    """Expose the fused-gate prefill contract without implementation options.

    The state-pool protocol parameters (``initial_state_indices``,
    ``inplace_final_state``, ``null_state_index``) are part of the public
    contract and are forwarded; implementation options such as
    ``normalize_qk`` are not.
    """
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
        initial_state_indices,
        output_final_state,
        inplace_final_state,
        lower_bound,
        cu_seqlens,
        null_state_index=null_state_index,
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
