# Copyright (c) 2026 The Qwen team, Alibaba Group.
# SPDX-License-Identifier: MIT
"""Prepared-input wrapper for the first HCU fused GDN forward kernel."""

from __future__ import annotations

import functools

import torch

from .common import require_contiguous, tilelang_dtype
from .fused_chunk_fwd_kernel import _build_fused_chunk_delta_rule_fwd


@functools.lru_cache(maxsize=64)
def _get_fused_kernel(*args):
    return _build_fused_chunk_delta_rule_fwd(*args)


def fused_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fused GDN state recurrence/output from prepared ``a`` and ``g``.

    ``g`` must already be a chunk-local cumulative gate and ``a`` must be the
    triangular KKT solve output.  Gate cumsum and KKT solve intentionally stay
    outside this first-phase kernel.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must use BTHD layout")
    batch, seq_len, q_heads, dim_k = q.shape
    value_heads, dim_v = v.shape[2:]
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError("k must have the same shape as q")
    if tuple(v.shape[:2]) != (batch, seq_len):
        raise ValueError("v must share q's batch and sequence dimensions")
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by q/k heads")
    if chunk_size != 64 or dim_k != 128 or dim_v != 128:
        raise NotImplementedError(
            "the first fused TileLang kernel requires BT=64 and K=V=128"
        )
    if seq_len == 0 or seq_len % chunk_size:
        raise NotImplementedError(
            "the first fused TileLang kernel requires a non-empty sequence "
            "whose length is divisible by 64"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q/k/v/a must use fp16 or bf16")
    if k.dtype != q.dtype or v.dtype != q.dtype or a.dtype != q.dtype:
        raise TypeError("q, k, v and a must have the same dtype")
    if tuple(a.shape) != (batch, seq_len, value_heads, chunk_size):
        raise ValueError("a must have shape [B, T, H, 64]")
    if tuple(g.shape) != (batch, seq_len, value_heads):
        raise ValueError("g must have shape [B, T, H]")
    if tuple(beta.shape) != tuple(g.shape):
        raise ValueError("beta must have the same shape as g")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("the first fused TileLang kernel requires fp32 g/beta")
    tensors = {"q": q, "k": k, "v": v, "a": a, "g": g, "beta": beta}
    if not all(tensor.is_cuda for tensor in tensors.values()):
        raise ValueError("all inputs must be CUDA/HIP tensors")
    require_contiguous(**tensors)

    if initial_state is None:
        state_buffer = torch.empty(
            batch,
            value_heads,
            dim_v,
            dim_k,
            dtype=torch.float32,
            device=q.device,
        )
        use_initial_state = False
    else:
        if tuple(initial_state.shape) != (
            batch,
            value_heads,
            dim_v,
            dim_k,
        ):
            raise ValueError("initial_state must have shape [B, H, V, K]")
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be float32")
        if initial_state.device != q.device or not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous and on q's device")
        state_buffer = initial_state
        use_initial_state = True

    output = torch.empty_like(v)
    final_state_buffer = torch.empty(
        batch,
        value_heads,
        dim_v,
        dim_k,
        dtype=torch.float32,
        device=q.device,
    )
    kernel = _get_fused_kernel(
        batch,
        seq_len,
        q_heads,
        value_heads,
        dim_k,
        dim_v,
        tilelang_dtype(q.dtype),
        tilelang_dtype(g.dtype),
        tilelang_dtype(beta.dtype),
        chunk_size,
        float(dim_k**-0.5 if scale is None else scale),
        use_initial_state,
        output_final_state,
        32,
    )
    kernel(
        q.view(batch, seq_len, q_heads * dim_k),
        k.view(batch, seq_len, q_heads * dim_k),
        v,
        a,
        g,
        beta,
        state_buffer,
        output,
        final_state_buffer,
    )
    return output, final_state_buffer if output_final_state else None


__all__ = ["fused_chunk_gated_delta_rule_fwd"]
