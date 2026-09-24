"""Triton backend for Gated DeltaNet operators."""

from boltops.fla.common.triton.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_kernel,
)
from .chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from .chunk_o import chunk_fwd_kernel_o, chunk_fwd_o
from .fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)

# NOTE: the KDA+GDN shared decode+verify kernel keeps its SGLang name
# ``fused_sigmoid_gating_delta_rule_update`` and is NOT re-exported here,
# because that name is already taken by the vLLM decode operator above.
# Import it at module level instead:
#   from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import (
#       fused_sigmoid_gating_delta_rule_update,
#   )


def recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None, chunk_indices=None):
    """GDN/SGLang WY-recomputation facade returning ``(w, u)``."""
    from .recompute_w_u import recompute_w_u_fwd_gdn

    return recompute_w_u_fwd_gdn(k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices)

__all__ = [
    "chunk_fwd_kernel_o",
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_recurrent_gated_delta_rule_packed_decode_kernel",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update_kernel",
    "recompute_w_u_fwd",
]
