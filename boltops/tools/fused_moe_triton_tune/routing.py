"""Torch routing helpers used by the migrated MoE correctness/tuning programs."""

from __future__ import annotations

import torch


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    is_softmax: bool = True,
):
    assert hidden_states.shape[0] == gating_output.shape[0]
    scores = (
        torch.softmax(gating_output.float(), dim=-1)
        if is_softmax
        else gating_output.float().sigmoid()
    )
    weights, ids = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)
    if topk_weights is not None:
        topk_weights.copy_(weights)
        weights = topk_weights
    if topk_ids is not None:
        topk_ids.copy_(ids)
        ids = topk_ids
    return weights, ids


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float = 1.0,
):
    scores = input.float().sigmoid()
    choice = scores + bias.float().unsqueeze(0)
    tokens, experts = scores.shape
    group_scores = (
        choice.view(tokens, num_expert_group, -1).topk(2, dim=-1).values.sum(-1)
    )
    groups = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(tokens, num_expert_group, experts // num_expert_group)
        .reshape(tokens, experts)
    )
    selected = choice.masked_fill(~expert_mask, float("-inf")).topk(
        topk, dim=-1, sorted=num_fused_shared_experts > 0
    ).indices
    weights = scores.gather(1, selected)
    if num_fused_shared_experts:
        assert num_fused_shared_experts == 1
        selected[:, -1] = experts
        weights[:, -1] = weights[:, :-1].sum(-1) / routed_scaling_factor
    denom = (
        weights[:, :-1].sum(-1, keepdim=True)
        if num_fused_shared_experts
        else weights.sum(-1, keepdim=True)
    )
    weights = weights / denom
    topk_weights.copy_(weights)
    topk_ids.copy_(selected.to(torch.int32))
    return [topk_weights, topk_ids]


__all__ = ["fused_topk", "moe_fused_gate"]

