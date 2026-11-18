# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Chunk-parallel GDN output, following TileOPs' BTHD three-GEMM design."""

import tilelang
import tilelang.language as T
import torch

from boltops.fla.common.triton.utils import prepare_chunk_indices

from .common import require_contiguous, tilelang_dtype


@tilelang.jit(
    out_idx=[],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _build_chunk_o(
    batch, tokens, heads, q_heads, chunks, sequences, dtype,
    gate_dtype, gamma_dtype, index_dtype, chunk_index_dtype,
    varlen, use_g, use_gamma, use_exp2, transpose_state,
    scale, block_v, threads,
):
    K, V, BT = 128, 128, 64
    hs0, hs1 = (V, K) if transpose_state else (K, V)
    exponent_scale = 1.0 if use_exp2 else 1.4426950408889634

    @T.prim_func
    def kernel(
        q: T.Tensor((batch, tokens, q_heads, K), dtype),
        k: T.Tensor((batch, tokens, q_heads, K), dtype),
        v: T.Tensor((batch, tokens, heads, V), dtype),
        h: T.Tensor((batch * chunks, heads, hs0, hs1), dtype),
        g: T.Tensor((batch, tokens, heads) if use_g else (0, 0, 0), gate_dtype),
        gamma: T.Tensor((heads if use_gamma else 0,), gamma_dtype),
        cu: T.Tensor((sequences + 1 if varlen else 0,), index_dtype),
        indices: T.Tensor((chunks if varlen else 0, 2), chunk_index_dtype),
        o: T.Tensor((batch, tokens, heads, V), dtype),
    ):
        with T.Kernel(
            V // block_v, chunks, batch * heads, threads=threads
        ) as (vid, cid, bh):
            bid = bh // heads
            hid = bh % heads
            qid = hid // (heads // q_heads)
            if varlen:
                sid = indices[cid, 0]
                start = cu[sid] + indices[cid, 1] * BT
                end = cu[sid + 1]
            else:
                start = cid * BT
                end = tokens
            vs = vid * block_v
            qs = T.alloc_shared((BT, K), dtype)
            ks = T.alloc_shared((BT, K), dtype)
            vs_shared = T.alloc_shared((BT, block_v), dtype)
            hs = T.alloc_fragment(
                (block_v, K) if transpose_state else (K, block_v), dtype
            )
            attention = T.alloc_shared((BT, BT), dtype)
            gates = T.alloc_shared((BT,), "float32")
            acc = T.alloc_fragment((BT, block_v), "float32")
            scores = T.alloc_fragment((BT, BT), "float32")

            if start + BT <= end:
                T.copy(q[bid, start:start + BT, qid, :], qs)
            else:
                # A packed tail must not read tokens from the next sequence.
                for i, j in T.Parallel(BT, K):
                    qs[i, j] = T.if_then_else(
                        start + i < end, q[bid, start + i, qid, j], 0
                    )
            if transpose_state:
                T.copy(h[bid * chunks + cid, hid, vs:vs + block_v, :], hs)
            else:
                T.copy(h[bid * chunks + cid, hid, :, vs:vs + block_v], hs)
            for i in T.Parallel(BT):
                gates[i] = 0
                if use_g:
                    gates[i] = T.if_then_else(
                        start + i < end,
                        T.cast(g[bid, start + i, hid], "float32"), 0,
                    )
            T.gemm(qs, hs, acc, transpose_B=transpose_state, clear_accum=True, k_pack=2)
            for i, j in T.Parallel(BT, block_v):
                if use_g:
                    acc[i, j] *= T.exp2(gates[i] * exponent_scale)
                if use_gamma:
                    acc[i, j] *= T.exp2(
                        T.cast(gamma[hid], "float32") * (i + 1) * exponent_scale
                    )
            # State is dead: its LDS storage can now hold K.
            if start + BT <= end:
                T.copy(k[bid, start:start + BT, qid, :], ks)
            else:
                for i, j in T.Parallel(BT, K):
                    ks[i, j] = T.if_then_else(
                        start + i < end, k[bid, start + i, qid, j], 0
                    )
            T.gemm(qs, ks, scores, transpose_B=True, clear_accum=True, k_pack=2)
            for i, j in T.Parallel(BT, BT):
                if use_g:
                    scores[i, j] *= T.exp2(T.if_then_else(
                        gates[i] <= gates[j],
                        (gates[i] - gates[j]) * exponent_scale,
                        -T.infinity("float32"),
                    ))
                if use_gamma:
                    scores[i, j] *= T.exp2(T.if_then_else(
                        gamma[hid] * (i - j) <= 0,
                        T.cast(gamma[hid], "float32") * (i - j) * exponent_scale,
                        -T.infinity("float32"),
                    ))
                attention[i, j] = T.if_then_else(
                    (i >= j) & (start + i < end) & (start + j < end),
                    scores[i, j], 0,
                )
            # Q is dead: its LDS storage can now hold V.
            if start + BT <= end:
                T.copy(v[bid, start:start + BT, hid, vs:vs + block_v], vs_shared)
            else:
                for i, j in T.Parallel(BT, block_v):
                    vs_shared[i, j] = T.if_then_else(
                        start + i < end, v[bid, start + i, hid, vs + j], 0
                    )
            T.gemm(attention, vs_shared, acc)
            for i, j in T.Parallel(BT, block_v):
                acc[i, j] *= scale
            if start + BT <= end:
                T.copy(acc, o[bid, start:start + BT, hid, vs:vs + block_v])
            else:
                for i, j in T.Parallel(BT, block_v):
                    if start + i < end:
                        o[bid, start + i, hid, vs + j] = acc[i, j]
    return kernel


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
    full_chunks: bool | None = None,
) -> torch.Tensor:
    """Compute GDN output with the Triton API and BTHD inputs.

    Initial specialization: K=V=128, BT=64, FP16/BF16; fixed and packed
    varlen, grouped heads and both state layouts. ``full_chunks`` is accepted
    as the Triton caller promise; this baseline retains boundary checks.
    ``kernel_cfg`` accepts ``block_v`` (128, default) and ``threads`` (512, default).
    """
    if q.ndim != 4 or v.ndim != 4 or k.shape != q.shape:
        raise ValueError("q/k/v must use BTHD layout and q/k must share shape")
    batch, tokens, q_heads, dim_k = q.shape
    heads, dim_v = v.shape[2:]
    if (dim_k, dim_v, chunk_size) != (128, 128, 64):
        raise NotImplementedError("TileLang chunk-o currently requires K=V=128 and BT=64")
    if q_heads <= 0 or heads <= 0 or heads % q_heads or v.shape[:2] != q.shape[:2]:
        raise ValueError("invalid batch/token dimensions or grouped head counts")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (k, v, h)):
        raise TypeError("q/k/v/h must share FP16 or BF16 dtype")
    tensors = dict(q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
                   cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    require_contiguous(**tensors)
    if not q.is_cuda or any(x is not None and x.device != q.device for x in tensors.values()):
        raise ValueError("all inputs must share a CUDA/HIP device")
    if g is not None and (g.shape != (batch, tokens, heads) or g.dtype not in (q.dtype, torch.float32)):
        raise ValueError("g must have shape [B,T,H] and input or FP32 dtype")
    if g_gamma is not None and (g_gamma.shape != (heads,) or g_gamma.dtype not in (q.dtype, torch.float32)):
        raise ValueError("g_gamma must have shape [H] and input or FP32 dtype")
    config = dict(kernel_cfg or {})
    if config.keys() - {"block_v", "threads"}:
        raise ValueError("TileLang chunk-o kernel_cfg accepts only block_v and threads")
    block_v, threads = config.get("block_v", 128), config.get("threads", 512)
    if block_v not in (32, 64, 128) or threads not in (128, 256, 512):
        raise NotImplementedError("block_v must be 32/64/128 and threads must be 128/256/512")
    varlen = cu_seqlens is not None
    if varlen:
        if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("packed varlen requires B=1 and int32/int64 cu_seqlens")
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2 or chunk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("chunk_indices must be an int32/int64 [NT,2] tensor")
        chunks = len(chunk_indices)
        sequences = len(cu_seqlens) - 1
    else:
        chunks = (tokens + chunk_size - 1) // chunk_size
        sequences = batch
    trailing = (dim_v, dim_k) if transpose_state_layout else (dim_k, dim_v)
    if h.shape != (batch, chunks, heads, *trailing):
        raise ValueError("h must have shape [B,NT,H,K,V] or [B,NT,H,V,K]")
    output = torch.empty_like(v)
    if not chunks or not batch:
        return output
    dummy = torch.empty(0, device=q.device, dtype=torch.float32)
    index_dummy = torch.empty(0, device=q.device, dtype=torch.int32)
    kernel = _build_chunk_o(
        batch, tokens, heads, q_heads, chunks, sequences, tilelang_dtype(q.dtype),
        tilelang_dtype(g.dtype) if g is not None else "float32",
        tilelang_dtype(g_gamma.dtype) if g_gamma is not None else "float32",
        str(cu_seqlens.dtype).split(".")[-1] if varlen else "int32",
        str(chunk_indices.dtype).split(".")[-1] if varlen else "int32",
        varlen, g is not None, g_gamma is not None, use_exp2,
        transpose_state_layout, float(dim_k ** -0.5 if scale is None else scale), block_v, threads,
    )
    kernel(q, k, v, h.view(batch * chunks, heads, *trailing),
           g if g is not None else dummy.view(0, 0, 0), g_gamma if g_gamma is not None else dummy,
           cu_seqlens if varlen else index_dummy, chunk_indices if varlen else index_dummy.view(0, 2),
           output)
    return output
