"""SGLang/GDN-specific WY recomputation kernels shared by FLA families."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...common.triton.utils import prepare_chunk_indices


# Offline-tuned on Hygon BW1000_H.  Generic retains its own configurations
# because its grouped-query layout differs from the KDA-specialized kernel.
_GDN_RECOMPUTE_W_U_CONFIGS = (
    {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 2},
    {"BK": 128, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 256, "BV": 256, "num_warps": 4, "num_stages": 1},
)


def _select_recompute_w_u_fwd_gdn_config(K: int, V: int) -> dict[str, int]:
    """Select the fixed GDN-kernel configuration for a head shape."""
    if K <= 64 and V <= 64:
        return _GDN_RECOMPUTE_W_U_CONFIGS[0]
    if K <= 128 or V <= 128:
        return _GDN_RECOMPUTE_W_U_CONFIGS[1]
    return _GDN_RECOMPUTE_W_U_CONFIGS[2]


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_gdn_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K, (T, K), (Hg * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_gdn_kernel_head_first(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """Varlen bf16 fast path with head-first scheduling and beta factoring."""
    i_h, i_tc = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_tc * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_tc * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    Tloc = eos - bos
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (Tloc,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h), (Tloc,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (Tloc, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    b_Ab = (tl.load(p_A, boundary_check=(0, 1)) * b_beta[None, :]).to(tl.bfloat16)

    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V, (Tloc, V), (H * V, 1),
        (i_t * BT, 0), (BT, BV), (1, 0),
    )
    p_u = tl.make_block_ptr(
        u + (bos * H + i_h) * V, (Tloc, V), (H * V, 1),
        (i_t * BT, 0), (BT, BV), (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_u = tl.dot(b_Ab, b_v, allow_tf32=False)
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    p_k = tl.make_block_ptr(
        k + (bos * Hg + i_h // (H // Hg)) * K, (Tloc, K), (Hg * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0),
    )
    p_w = tl.make_block_ptr(
        w + (bos * H + i_h) * K, (Tloc, K), (H * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kg = (b_k * b_g[:, None]).to(b_k.dtype)
    b_w = tl.dot(b_Ab, b_kg)
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd_gdn(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute GDN WY tensors with the GDN-specialized kernel."""
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]
    if H % Hg:
        raise ValueError("v heads must be divisible by k heads")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    config = _select_recompute_w_u_fwd_gdn_config(K, V)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    use_head_first = (
        B == 1 and cu_seqlens is not None and chunk_indices is not None
        and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        and A.dtype == torch.bfloat16 and K <= config["BK"] and V <= config["BV"]
    )
    if use_head_first:
        recompute_w_u_fwd_gdn_kernel_head_first[(H, NT)](
            k=k, v=v, beta=beta, w=w, u=u, A=A, g=g_cumsum,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
            Hg=Hg, K=K, V=V, BT=BT, BK=config["BK"], BV=config["BV"],
            num_warps=4, num_stages=1,
        )
    else:
        recompute_w_u_fwd_gdn_kernel[(NT, B * H)](
            k=k, v=v, beta=beta, w=w, u=u, A=A, g=g_cumsum,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
            Hg=Hg, K=K, V=V, BT=BT, BK=config["BK"], BV=config["BV"],
            IS_VARLEN=cu_seqlens is not None, num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
    return w, u


__all__ = ["recompute_w_u_fwd_gdn"]
