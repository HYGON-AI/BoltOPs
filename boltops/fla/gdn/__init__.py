"""Public Gated DeltaNet operators with lazy Triton imports."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute GDN WY intermediates with the GDN Triton kernel."""
    from .triton import recompute_w_u_fwd as impl

    return impl(k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    kernel_cfg: dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .triton.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as impl,
    )

    return impl(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        kernel_cfg=kernel_cfg,
    )


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    kernel_cfg: dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .triton.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as impl,
    )

    return impl(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        kernel_cfg=kernel_cfg,
    )


def fused_sigmoid_gating_delta_rule_update_recurrent(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    kernel_cfg: dict | None = None,
) -> torch.Tensor:
    from .triton.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_recurrent as impl,
    )

    return impl(
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        kernel_cfg=kernel_cfg,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = True,
    kernel_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute chunk states, optionally updating ``initial_state`` in place.

    In-place updates require unique valid ``initial_state_indices`` within a
    launch. Out-of-place final states are compact and indexed by sequence.
    """
    from .triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h as impl

    return impl(
        k,
        w,
        u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        kernel_cfg=kernel_cfg,
    )


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    kernel_cfg: dict | None = None,
) -> torch.Tensor:
    from .triton.chunk_o import chunk_fwd_o as impl

    return impl(
        q,
        k,
        v,
        h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        kernel_cfg=kernel_cfg,
    )


__all__ = [
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update_recurrent",
    "recompute_w_u_fwd",
]
