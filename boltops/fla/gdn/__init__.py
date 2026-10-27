# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Public Gated DeltaNet (GDN) operators -- public signature boundary.

Operator names align 1:1 with the vLLM / SGLang public names.
Implementations live in
:mod:`boltops.fla.gdn.triton`; the KDA+GDN shared decode/verify kernels live
in :mod:`boltops.fla.common.triton`. Importing this package never imports
Triton until an operator is first used.

Entry kinds (deliberately minimal):

- **Lazy re-exports** (``_LAZY_EXPORTS``): the public name resolves to the
  implementation object itself -- one source of truth for the signature.
  The KDA+GDN shared decode+verify kernel keeps the SGLang
  ``fused_sigmoid_gating_delta_rule_update`` name in
  :mod:`boltops.fla.common.triton.fused_sigmoid_gating_recurrent` and is
  imported at module level by framework adapters; it is not re-exported here
  because that name is this package's vLLM decode operator.
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch

__all__ = [
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "recompute_w_u_fwd",
]

# Public name -> (implementation module, attribute). Every entry is re-exported
# 1:1 (same function object): the signature has exactly one home. ``.triton``
# re-exports the KDA+GDN shared kernels from ``fla.common.triton``. Do NOT add
# an entry whose public contract
# differs from the implementation -- write a wrapper instead.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "chunk_fwd_o": (".triton", "chunk_fwd_o"),
    "chunk_gated_delta_rule_fwd_h": (".triton", "chunk_gated_delta_rule_fwd_h"),
    "fused_recurrent_gated_delta_rule_packed_decode": (".triton", "fused_recurrent_gated_delta_rule_packed_decode"),
    "fused_sigmoid_gating_delta_rule_update": (".triton", "fused_sigmoid_gating_delta_rule_update"),
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
