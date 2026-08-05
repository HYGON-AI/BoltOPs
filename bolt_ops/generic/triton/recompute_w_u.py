"""Standalone Triton implementation of gated-delta WY recomputation.

The public entry and kernel body intentionally stay close to SGLang's vendored
FLA implementation.  The only SGLang-specific dependency, chunk-index
preparation, is implemented locally in this module.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Offline-tuned on Hygon BW1000_H.  Keep the table small so every supported
# shape selects a fixed config without paying runtime autotune overhead.
_RECOMPUTE_W_U_CONFIGS = (
    {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 2},
    {"BK": 128, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 256, "BV": 256, "num_warps": 4, "num_stages": 1},
)


def _select_recompute_w_u_config(K: int, V: int) -> dict[str, int]:
    """Select one of three fixed configs from the key/value head dimensions."""
    if K <= 64 and V <= 64:
        return _RECOMPUTE_W_U_CONFIGS[0]
    if K <= 128 or V <= 128:
        return _RECOMPUTE_W_U_CONFIGS[1]
    return _RECOMPUTE_W_U_CONFIGS[2]


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
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
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_head_first(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
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
        A + (bos * H + i_h) * BT, (Tloc, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    b_Ab = (tl.load(p_A, boundary_check=(0, 1)) * b_beta[None, :]).to(tl.bfloat16)

    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (Tloc, V),
        (H * V, 1),
        (i_t * BT, 0),
        (BT, BV),
        (1, 0),
    )
    p_u = tl.make_block_ptr(
        u + (bos * H + i_h) * V,
        (Tloc, V),
        (H * V, 1),
        (i_t * BT, 0),
        (BT, BV),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_u = tl.dot(b_Ab, b_v, allow_tf32=False)
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    p_k = tl.make_block_ptr(
        k + (bos * Hg + i_h // (H // Hg)) * K,
        (Tloc, K),
        (Hg * K, 1),
        (i_t * BT, 0),
        (BT, BK),
        (1, 0),
    )
    p_w = tl.make_block_ptr(
        w + (bos * H + i_h) * K,
        (Tloc, K),
        (H * K, 1),
        (i_t * BT, 0),
        (BT, BK),
        (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kg = (b_k * b_g[:, None]).to(b_k.dtype)
    b_w = tl.dot(b_Ab, b_kg)
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    config = _select_recompute_w_u_config(K, V)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    use_head_first = (
        B == 1
        and cu_seqlens is not None
        and chunk_indices is not None
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and A.dtype == torch.bfloat16
        and H % Hg == 0
        and K <= config["BK"]
        and V <= config["BV"]
    )
    if use_head_first:
        recompute_w_u_fwd_kernel_head_first[(H, NT)](
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=config["BK"],
            BV=config["BV"],
            num_warps=4,
            num_stages=1,
        )
    else:
        recompute_w_u_fwd_kernel[(NT, B * H)](
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=config["BK"],
            BV=config["BV"],
            IS_VARLEN=cu_seqlens is not None,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
    return w, u


fwd_recompute_w_u = recompute_w_u_fwd

__all__ = ["recompute_w_u_fwd"]
