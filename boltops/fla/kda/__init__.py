"""Kernelized Delta Attention operators.

Public signatures live at the package boundary. Implementations are imported
inside each wrapper so importing :mod:`boltops` does not require Triton.
"""

from __future__ import annotations

import torch


def chunk_kda(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, scale: float | None = None,
    initial_state: torch.Tensor | None = None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None, **kwargs,
):
    from boltops.fla.kda.triton import chunk_kda as impl
    return impl(q, k, v, g, beta, scale, initial_state, output_final_state,
                use_qk_l2norm_in_kernel, cu_seqlens, **kwargs)


def chunk_kda_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, scale: float, initial_state: torch.Tensor | None,
    output_final_state: bool, cu_seqlens: torch.Tensor | None = None,
):
    from boltops.fla.kda.triton import chunk_kda_fwd as impl
    return impl(q, k, v, g, beta, scale, initial_state, output_final_state,
                cu_seqlens)


def chunk_kda_with_fused_gate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, raw_g: torch.Tensor,
    raw_beta: torch.Tensor, A_log: torch.Tensor, g_bias: torch.Tensor | None,
    scale: float | None = None, initial_state: torch.Tensor | None = None,
    output_final_state: bool = False, lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None, **kwargs,
):
    from boltops.fla.kda.triton import chunk_kda_with_fused_gate as impl
    return impl(q, k, v, raw_g, raw_beta, A_log, g_bias, scale,
                initial_state, output_final_state, lower_bound,
                use_qk_l2norm_in_kernel, cu_seqlens, **kwargs)


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, raw_g: torch.Tensor,
    raw_beta: torch.Tensor, A_log: torch.Tensor, g_bias: torch.Tensor | None,
    scale: float, initial_state: torch.Tensor | None, output_final_state: bool,
    lower_bound: float | None = None, cu_seqlens: torch.Tensor | None = None,
):
    from boltops.fla.kda.triton import chunk_kda_with_fused_gate_fwd as impl
    return impl(q, k, v, raw_g, raw_beta, A_log, g_bias, scale,
                initial_state, output_final_state, lower_bound, cu_seqlens)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
    """Recompute the KDA ``w`` and ``u`` chunk intermediates."""
    from boltops.fla.kda.triton import recompute_w_u_fwd as impl

    return impl(
        k,
        v,
        beta,
        A,
        q=q,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute the chunk KDA output from cumulative gates and state."""
    from boltops.fla.kda.triton import chunk_gla_fwd_o_gk as impl

    return impl(
        q,
        v,
        g,
        A,
        h,
        o,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the KDA gate and produce its chunk-wise cumulative sum."""
    from boltops.fla.kda.triton import fused_kda_gate_chunk_cumsum as impl

    return impl(
        raw_g,
        raw_beta,
        A_log,
        g_bias=g_bias,
        beta=beta,
        threshold=threshold,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Apply the KDA gate transformation."""
    from boltops.fla.kda.triton import fused_kda_gate as impl

    return impl(
        g,
        A,
        head_k_dim,
        g_bias=g_bias,
        beta=beta,
        threshold=threshold,
        lower_bound=lower_bound,
    )


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the intra-chunk KDA coefficient matrices."""
    from boltops.fla.kda.triton import chunk_kda_fwd_intra as impl

    return impl(
        q,
        k,
        gk=gk,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute intra-chunk coefficients with token parallelism."""
    from boltops.fla.kda.triton import (
        chunk_kda_fwd_intra_token_parallel as impl,
    )

    return impl(
        q,
        k,
        gk=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        sub_chunk_size=sub_chunk_size,
    )


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA with preprocessed gate and beta tensors."""
    from boltops.fla.kda.triton import fused_recurrent_kda_fwd as impl

    return impl(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        out=out,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA from raw gate and beta tensors."""
    from boltops.fla.kda.triton import fused_recurrent_kda as impl

    return impl(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        fuse_gate=fuse_gate,
    )


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-token KDA decode from packed post-convolution QKV."""
    from boltops.fla.kda.triton import fused_recurrent_kda_packed_decode as impl

    return impl(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        state_indices,
        scale=scale,
    )


__all__ = [
    "chunk_kda",
    "chunk_kda_fwd",
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
