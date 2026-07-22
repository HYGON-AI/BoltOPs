"""HIPC MoE sorting kernel used by the Triton fused MoE path.

Private to ``bolt_ops.fused_moe.triton`` — not part of the public API.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from bolt_ops.fused_moe.triton.moe_sorting_hipc.compiler import load_hipc_extension

_CSRC = Path(__file__).with_name("csrc")


@lru_cache(maxsize=1)
def _extension():
    return load_hipc_extension(
        "moe_sorting",
        ("moe_sorting_api.cu", "moe_sorting_binding.cu"),
        csrc_dir=_CSRC,
    )


def moe_sorting_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens_positions_per_expert: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: Optional[torch.Tensor],
    num_experts: int,
    unit_size: int,
    local_expert_mask: Optional[torch.Tensor] = None,
) -> None:
    _extension().moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_positions_per_expert,
        num_valid_ids,
        moe_buf,
        num_experts,
        unit_size,
        local_expert_mask,
    )


__all__ = ["moe_sorting_fwd"]
