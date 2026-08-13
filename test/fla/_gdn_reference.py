# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# flash-linear-attention repository. Adapted from commit
# 2e38c1fab332174d056928feaf29f8c5fd5ac550.

"""PyTorch references adapted from flash-linear-attention.

The recurrent and gate equations come from ``fla/ops/gated_delta_rule/naive.py``
and ``gate.py``. The component-level chunk helpers expose the intermediate
states of the same equations used by ``fla/ops/common/chunk_delta_h.py`` and
``chunk_o.py`` so Bolt's two public chunk stages can be checked independently.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_gdn_gate(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    *,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """FLA's scalar GDN gate with Bolt's configurable softplus."""
    x = a.float()
    if dt_bias is not None:
        x = x + dt_bias.float()
    scaled = softplus_beta * x
    softplus = torch.where(
        scaled <= softplus_threshold,
        torch.log1p(torch.exp(scaled)) / softplus_beta,
        x,
    )
    return -A_log.float().exp() * softplus


def naive_kda_gate(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    *,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """FLA's per-key-dimension KDA gate with Bolt's softplus options."""
    H, K = a.shape[-2:]
    x = a.float()
    if dt_bias is not None:
        x = x + dt_bias.float().reshape(H, K)
    scaled = softplus_beta * x
    softplus = torch.where(
        scaled <= softplus_threshold,
        torch.log1p(torch.exp(scaled)) / softplus_beta,
        x,
    )
    return -A_log.float().reshape(H, 1).exp() * softplus


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # Match the fused kernels, which use rsqrt(sum(x*x) + 1e-6).
    return x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA recurrent reference, extended mechanically for grouped V heads.

    ``initial_state`` and the returned state use FLA's ``[B, HV, K, V]``
    layout. Bolt fused kernels use the transposed physical layout at their
    public boundary; tests transpose only in their protocol adapters.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    if HV % H:
        raise ValueError("HV must be divisible by H")
    groups = HV // H
    qf = q.float().repeat_interleave(groups, dim=2)
    kf = k.float().repeat_interleave(groups, dim=2)
    if use_qk_l2norm:
        qf = _normalize(qf)
        kf = _normalize(kf)
    vf, bf, gf = v.float(), beta.float(), g.float()
    if scale is None:
        scale = K**-0.5

    state = torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        state = initial_state.float().clone()
    out = torch.empty(B, T, HV, V, device=q.device, dtype=torch.float32)

    for step in range(T):
        q_i = qf[:, step] * scale
        k_i = kf[:, step]
        state = state * gf[:, step, :, :, None].exp() if gf.ndim == 4 else state * gf[:, step, :, None, None].exp()
        residual = vf[:, step] - torch.einsum("bhkv,bhk->bhv", state, k_i)
        residual = residual * bf[:, step, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k_i, residual)
        out[:, step] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    return out.to(dtype), state if output_final_state else None


def _sequence_ranges(
    batch: int,
    total_t: int,
    cu_seqlens: torch.Tensor | None,
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(i, i * total_t, (i + 1) * total_t) for i in range(batch)]
    boundaries = cu_seqlens.detach().cpu().tolist()
    return [(i, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def naive_chunk_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Expose FLA chunk state recurrence using PyTorch operations."""
    B, T, Hg, K = k.shape
    H, V = u.shape[2:]
    groups = H // Hg
    ranges = _sequence_ranges(B, T, cu_seqlens)
    total_chunks = sum(math.ceil((eos - bos) / chunk_size) for _, bos, eos in ranges)
    h_shape = (B, total_chunks if cu_seqlens is not None else math.ceil(T / chunk_size), H)
    trailing = (V, K) if transpose_state_layout else (K, V)
    h = torch.empty(*h_shape, *trailing, device=k.device, dtype=k.dtype)
    v_new = torch.empty_like(u) if save_new_value else None
    compact_final = torch.empty(len(ranges), H, *trailing, device=k.device, dtype=torch.float32)
    indices = None if initial_state_indices is None else initial_state_indices.detach().cpu().tolist()
    exp_fn = torch.exp2 if use_exp2 else torch.exp

    global_chunk = 0
    for seq, bos, eos in ranges:
        state_index = seq if indices is None else int(indices[seq])
        valid = initial_state is not None and 0 <= state_index < initial_state.shape[0]
        if valid:
            physical = initial_state[state_index].float()
            state = physical.transpose(-1, -2).contiguous() if transpose_state_layout else physical.clone()
        else:
            state = torch.zeros(H, K, V, device=k.device, dtype=torch.float32)

        for local_chunk, start in enumerate(range(bos, eos, chunk_size)):
            end = min(start + chunk_size, eos)
            h_index = global_chunk if cu_seqlens is not None else local_chunk
            physical_h = state.transpose(-1, -2) if transpose_state_layout else state
            h[0 if cu_seqlens is not None else seq, h_index] = physical_h.to(k.dtype)

            k_chunk = k.reshape(-1, Hg, K)[start:end].float().repeat_interleave(groups, dim=1)
            w_chunk = w.reshape(-1, H, K)[start:end].float()
            u_chunk = u.reshape(-1, H, V)[start:end].float()
            residual = u_chunk - torch.einsum("thk,hkv->thv", w_chunk, state)
            if v_new is not None:
                v_new.reshape(-1, H, V)[start:end] = residual.to(u.dtype)

            last = end - 1
            weighted = residual
            if g is not None:
                flat_g = g.reshape(-1, H).float()
                relative = exp_fn(flat_g[last] - flat_g[start:end])
                weighted = weighted * relative[..., None]
                state = state * exp_fn(flat_g[last])[:, None, None]
            if gk is not None:
                flat_gk = gk.reshape(-1, H, K).float()
                state = state * exp_fn(flat_gk[last])[:, :, None]
            # The Triton kernel rounds v_new to the input dtype before this dot.
            weighted = weighted.to(k.dtype).float()
            state = state + torch.einsum("thk,thv->hkv", k_chunk, weighted)
            global_chunk += 1

        physical_final = state.transpose(-1, -2) if transpose_state_layout else state
        compact_final[seq] = physical_final

    if not output_final_state:
        final_state = None
    elif inplace_final_state:
        final_state = initial_state
        assert final_state is not None
        for seq, _, _ in ranges:
            state_index = seq if indices is None else int(indices[seq])
            if 0 <= state_index < final_state.shape[0]:
                final_state[state_index].copy_(compact_final[seq])
    else:
        final_state = compact_final
    return h, v_new, final_state


def naive_chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> torch.Tensor:
    """FLA chunk output equation with component inputs exposed."""
    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    groups = H // Hg
    if scale is None:
        scale = K**-0.5
    ranges = _sequence_ranges(B, T, cu_seqlens)
    out = torch.empty_like(v)
    exp_fn = torch.exp2 if use_exp2 else torch.exp
    global_chunk = 0

    for seq, bos, eos in ranges:
        for local_chunk, start in enumerate(range(bos, eos, chunk_size)):
            end = min(start + chunk_size, eos)
            h_index = global_chunk if cu_seqlens is not None else local_chunk
            physical = h[0 if cu_seqlens is not None else seq, h_index].float()
            state = physical.transpose(-1, -2) if transpose_state_layout else physical
            q_chunk = q.reshape(-1, Hg, K)[start:end].float().repeat_interleave(groups, dim=1)
            k_chunk = k.reshape(-1, Hg, K)[start:end].float().repeat_interleave(groups, dim=1)
            v_chunk = v.reshape(-1, H, V)[start:end].float()
            state_term = torch.einsum("thk,hkv->thv", q_chunk, state)
            scores = torch.einsum("thk,shk->hts", q_chunk, k_chunk)

            if g is not None:
                gate = g.reshape(-1, H)[start:end].float()
                state_term = state_term * exp_fn(gate)[..., None]
                scores = scores * exp_fn(gate.transpose(0, 1)[:, :, None] - gate.transpose(0, 1)[:, None, :])
            if g_gamma is not None:
                positions = torch.arange(1, end - start + 1, device=q.device, dtype=torch.float32)
                gamma_gate = positions[:, None] * g_gamma.float()[None, :]
                state_term = state_term * exp_fn(gamma_gate)[..., None]
                scores = scores * exp_fn(gamma_gate.transpose(0, 1)[:, :, None] - gamma_gate.transpose(0, 1)[:, None, :])

            scores = torch.tril(scores)
            local = torch.einsum("hts,shv->thv", scores, v_chunk)
            out.reshape(-1, H, V)[start:end] = ((state_term + local) * scale).to(v.dtype)
            global_chunk += 1
    return out
