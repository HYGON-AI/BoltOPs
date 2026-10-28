# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Independent PyTorch references for Sparse-GQA correctness checks."""

from __future__ import annotations

import torch


def reference_compact(
    data: dict,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the asymmetric HCU-FA paged K/V layouts with PyTorch."""
    out_k = torch.empty_like(data["out_k"])
    out_v = torch.empty_like(data["out_v"])
    page_size = data["page_size"]
    for batch in range(data["batch"]):
        start = int(data["cu_k"][batch])
        end = int(data["cu_k"][batch + 1])
        request = int(data["req_indices"][batch])
        logical = data["indices"][batch, : end - start].long()
        slots = data["req_to_token"][request, logical].long()
        pages, offsets = slots // page_size, slots % page_size
        out_k[start:end] = (
            data["k"][pages, :, offsets, :].float() * k_scale
        ).to(out_k.dtype)
        out_v[start:end] = (
            data["v"][pages, :, :, offsets].float() * v_scale
        ).to(out_v.dtype)
    return out_k, out_v


def reference_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Compute the Sparse-GQA attention oracle with an FP32 softmax."""
    out = torch.empty_like(q)
    group_size = q.shape[1] // k.shape[1]
    for batch in range(kv_lens.shape[0]):
        q_start, q_end = int(cu_q[batch]), int(cu_q[batch + 1])
        k_start = int(cu_k[batch])
        for query in range(q_start, q_end):
            for q_head in range(q.shape[1]):
                kv_head = q_head // group_size
                selected = (
                    indices[query, kv_head]
                    if indices.ndim == 3
                    else indices[query]
                )
                selected = selected[
                    (selected >= 0) & (selected < int(kv_lens[batch]))
                ].long()
                keys = k[k_start + selected, kv_head].float()
                values = v[k_start + selected, kv_head].float()
                scores = torch.mv(keys, q[query, q_head].float()) * scale * k_scale
                out[query, q_head] = (
                    torch.softmax(scores, 0) @ values * v_scale
                ).to(out.dtype)
    return out
