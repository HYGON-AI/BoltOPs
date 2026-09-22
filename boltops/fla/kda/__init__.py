"""Kernelized Delta Attention (KDA) operators -- public signature boundary.

Operator names align 1:1 with the vLLM / SGLang public names (see
vLLM / SGLang public APIs). Implementations live in
:mod:`boltops.fla.kda.triton`; shared FLA helpers and kernels live in
:mod:`boltops.fla.common.triton`. Importing this package never imports
Triton until an operator is first used.

Entry kinds (deliberately minimal):

- **Lazy re-exports** (``_LAZY_EXPORTS``): each public name resolves through
  the same declarative table; the backend owns direct exports and the two
  narrowing wrappers (which hide implementation-only ``normalize_qk``).
SGLang-style ``target_verify`` adapters are intentionally not exported here.
"""

from __future__ import annotations

import importlib

__all__ = [
    "chunk_kda",
    "chunk_kda_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_kda_fwd_state_output_fused",
    "chunk_kda_with_fused_gate",
    "chunk_kda_with_fused_gate_fwd",
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

# Public name -> (implementation module, attribute). Every entry is re-exported
# 1:1 (same function object): the signature has exactly one home, so the
# facade and the implementation cannot drift. Do NOT add an entry whose public
# contract differs from the implementation -- write a wrapper instead.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "chunk_kda": (".triton", "chunk_kda"),
    "chunk_kda_fwd": (".triton", "chunk_kda_fwd"),
    "chunk_gated_delta_rule_fwd_h": (".triton", "chunk_gated_delta_rule_fwd_h"),
    "chunk_kda_fwd_state_output_fused": (".triton", "chunk_kda_fwd_state_output_fused"),
    "chunk_kda_with_fused_gate": (".triton", "chunk_kda_with_fused_gate"),
    "chunk_kda_with_fused_gate_fwd": (".triton", "chunk_kda_with_fused_gate_fwd"),
    "chunk_gla_fwd_o_gk": (".triton", "chunk_gla_fwd_o_gk"),
    "chunk_kda_fwd_intra": (".triton", "chunk_kda_fwd_intra"),
    "chunk_kda_fwd_intra_token_parallel": (".triton", "chunk_kda_fwd_intra_token_parallel"),
    "fused_kda_gate": (".triton", "fused_kda_gate"),
    "fused_kda_gate_chunk_cumsum": (".triton", "fused_kda_gate_chunk_cumsum"),
    "fused_recurrent_kda": (".triton", "fused_recurrent_kda"),
    "fused_recurrent_kda_fwd": (".triton", "fused_recurrent_kda_fwd"),
    "fused_recurrent_kda_packed_decode": (".triton", "fused_recurrent_kda_packed_decode"),
    "recompute_w_u_fwd": (".triton", "recompute_w_u_fwd"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value  # cache: later accesses skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
