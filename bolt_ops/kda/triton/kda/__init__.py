# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Standalone vendor of the directly callable Kimi-K3 KDA Triton kernels from
# vLLM's AMD implementation. vLLM imports are replaced by local compatibility
# helpers; orchestration that depends on vLLM's shared FLA modules is excluded.

from .chunk import (
    chunk_gla_fwd_o_gk,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    recompute_w_u_fwd,
)
from .chunk_intra import chunk_kda_fwd_intra
from .chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from .fused_recurrent import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)

__all__ = [
    "chunk_gla_fwd_o_gk",
    "chunk_kda_fwd_intra",
    "chunk_kda_fwd_intra_token_parallel",
    "fused_kda_gate",
    "fused_kda_gate_chunk_cumsum",
    "fused_recurrent_kda",
    "fused_recurrent_kda_fwd",
    "fused_recurrent_kda_packed_decode",
    "recompute_w_u_fwd",
]
