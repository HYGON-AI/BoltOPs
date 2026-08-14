"""Small Torch references for the public KDA protocols.

The references deliberately use explicit loops and FP32 state.  They are test
oracles, not implementations intended for performance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_kda_gate(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    *,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Reference for the per-key KDA gate."""
    H, K = raw_g.shape[-2:]
    x = raw_g.float()
    if g_bias is not None:
        x = x + g_bias.float().reshape(H, K)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A_log.float().exp().reshape(H, 1) * x)
    return -A_log.float().exp().reshape(H, 1) * F.softplus(x)


def naive_gate_chunk_cumsum(
    gate: torch.Tensor,
    raw_beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for beta sigmoid and sequence-local chunk cumsum."""
    result = torch.empty_like(gate, dtype=torch.float32)
    B, T = gate.shape[:2]
    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        bounds = cu_seqlens.detach().cpu().tolist()
        ranges = list(zip(bounds[:-1], bounds[1:]))
    for start, end in ranges:
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            result[:, chunk_start:chunk_end] = (
                gate[:, chunk_start:chunk_end].float().cumsum(1)
                * math.log2(math.e)
            )
    return result, raw_beta.float().sigmoid()


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent KDA with physical state layout ``[H,V,K]``."""
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    if HV != H:
        raise ValueError("the KDA reference currently requires H == HV")
    if scale is None:
        scale = K**-0.5
    qf, kf = q.float(), k.float()
    if use_qk_l2norm:
        qf, kf = _normalize(qf), _normalize(kf)
    state = (
        torch.zeros(B, H, V, K, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    output = torch.empty(B, T, H, V, device=q.device, dtype=v.dtype)
    for token in range(T):
        decay = torch.exp2(gate[:, token].float())
        state = state * decay[:, :, None, :]
        residual = v[:, token].float() - torch.einsum(
            "bhvk,bhk->bhv", state, kf[:, token]
        )
        state = state + torch.einsum(
            "bhv,bhk->bhvk",
            residual * beta[:, token, :, None].float(),
            kf[:, token],
        )
        output[:, token] = torch.einsum(
            "bhk,bhvk->bhv", qf[:, token] * scale, state
        ).to(v.dtype)
    return output, state


def naive_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the WY ``w/u/kg`` reconstruction."""
    A_f = A.permute(0, 2, 1, 3).float()
    v_beta = v.float() * beta[..., None].float()
    k_beta_gate = k.float() * beta[..., None].float() * torch.exp2(gk.float())
    u = torch.matmul(A_f, v_beta.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    w = torch.matmul(A_f, k_beta_gate.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    kg = k.float() * torch.exp2(gk[:, -1:] - gk)
    return w.to(k.dtype), u.to(v.dtype), kg.to(k.dtype)


def naive_chunk_state(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the fixed-length KDA chunk-state kernel."""
    B, T, H, K = k.shape
    V = u.shape[-1]
    n_chunks = math.ceil(T / chunk_size)
    state = initial_state.float().transpose(-1, -2).contiguous()
    h = torch.empty(B, n_chunks, H, V, K, device=k.device, dtype=k.dtype)
    v_new = torch.empty_like(u)
    for chunk in range(n_chunks):
        start, end = chunk * chunk_size, min((chunk + 1) * chunk_size, T)
        h[:, chunk] = state.transpose(-1, -2).to(k.dtype)
        residual = u[:, start:end].float() - torch.einsum(
            "bthk,bhkv->bthv", w[:, start:end].float(), state
        )
        v_new[:, start:end] = residual.to(u.dtype)
        state = state * torch.exp2(gk[:, end - 1].float())[:, :, :, None]
        state = state + torch.einsum(
            "bthk,bthv->bhkv", k[:, start:end].float(), residual
        )
    return h, v_new, state.transpose(-1, -2).contiguous()
