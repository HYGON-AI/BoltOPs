# SPDX-License-Identifier: Apache-2.0
"""No-sort MoE metadata specialization for tiny decode batches.

This module produces the flattened-route metadata accepted by the Triton MoE
GEMMs. It does not group routes by expert: each flattened route owns one
independently padded ``BLOCK_SIZE_M`` tile. That trades route packing
efficiency for much lower fixed overhead when decode has only a few tokens.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_sorting_decode_kernel(
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    num_routes: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Build one independently padded tile per flattened route."""
    route = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE_M)
    sorted_token_id = tl.where(
        lanes == 0,
        route,
        num_routes,
    )
    route_offset = route * BLOCK_SIZE_M
    tl.store(
        sorted_token_ids_ptr + route_offset + lanes,
        sorted_token_id,
    )
    if route == 0:
        tl.store(num_tokens_post_padded_ptr, num_routes * BLOCK_SIZE_M)


def _allocate_sorting_metadata(
    topk_ids: torch.Tensor,
    block_size_m: int,
):
    """Allocate metadata outputs, reusing int32 expert ids without a copy."""
    flat_ids = topk_ids.view(-1)
    sorted_token_ids = torch.empty(
        flat_ids.numel() * block_size_m,
        dtype=torch.int32,
        device=flat_ids.device,
    )
    expert_ids = (
        flat_ids
        if flat_ids.dtype == torch.int32
        else flat_ids.to(torch.int32)
    )
    num_tokens_post_padded = torch.empty(
        1,
        dtype=torch.int32,
        device=flat_ids.device,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def moe_sorting_decode_fwd(
    topk_ids: torch.Tensor,      # [M, topk]
    block_size_m: int,           # scalar: padded route-tile width
):
    """Build flattened-route metadata without sorting tiny decode routes.

    The general sorter groups routes that select the same expert and then pads
    each expert group to ``block_size_m``.  For decode-sized inputs that setup
    costs more than the grouping saves.  This specialization instead flattens
    ``[M, topk]`` into ``M * topk`` routes and gives every route its own padded
    tile and launches one single-warp program per route. Lane 0 of each token-id
    tile stores the flattened route id, while the remaining lanes store the
    first invalid route as a sentinel. For int32 routing, ``expert_ids`` is a
    zero-copy flattened view of ``topk_ids``. The GEMMs read routed weights
    directly from the original contiguous ``topk_weights`` tensor, so no
    padded weight array is needed either.
    """
    flat_ids = topk_ids.view(-1)
    num_routes = flat_ids.numel()
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        _allocate_sorting_metadata(topk_ids, block_size_m,)
    )
    _moe_sorting_decode_kernel[(num_routes,)](
        sorted_token_ids,
        num_tokens_post_padded,
        num_routes=num_routes,
        BLOCK_SIZE_M=block_size_m,
        num_warps=1,
        num_stages=1,
    )
    return (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


__all__ = ["moe_sorting_decode_fwd"]
