# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/yht183/sglang-das/blob/4cc62ab1d0c7a63c770e2a275d42b67ae20c345c/python/sglang/srt/layers/attention/qsa/mqa.py
"""Weighted (non-QSA) TileLang MQA logits prefill, bf16.

Same tiling as the generic QSA prefill in ``mqa.py``; the head reduction is
weighted per (query, head)::

    logits[m, n] = sum_h relu(Q[m, h] . K[n]) * W[m, h]

``W`` is ``[rows, heads]`` float32 (may be negative, so it cannot be folded
into the ReLU).  It is constant across the key loop, so each CTA loads its
``block_q x heads`` slice into registers once before the loop.  Window and
``-inf`` masking reuse the QSA mask kernel.  Interface follows
``triton_mqa_logits`` (half path, ``kv_scales=None``); bf16 only for now.
"""

from typing import Optional

import torch

from .mqa import HAS_TILELANG, _is_hip

if HAS_TILELANG:
    import tilelang
    from tilelang import language as T

    from .mqa import _tilelang_qsa_mqa_mask_kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_weighted_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def weighted_mqa_prefill_generic(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Weights: T.Tensor([rows, heads], T.float32),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                weights_f = T.alloc_fragment([block_q, heads], T.float32)
                starts_f = T.alloc_fragment([block_q], T.int32)
                ends_f = T.alloc_fragment([block_q], T.int32)
                start_min_f = T.alloc_fragment([1], T.int32)
                end_max_f = T.alloc_fragment([1], T.int32)
                row_base = bx * block_q
                for qi in T.Parallel(block_q):
                    starts_f[qi] = Starts[row_base + qi]
                    ends_f[qi] = Ends[row_base + qi]
                T.reduce_min(starts_f, start_min_f, dim=0, clear=True)
                T.reduce_max(ends_f, end_max_f, dim=0, clear=True)
                start_min = T.min(start_min_f[0], keys)
                end_max = T.min(end_max_f[0], keys)

                T.copy(Q[row_base * heads, 0], q_shared)
                # Key-invariant: load once, reuse for every key tile.
                T.copy(Weights[row_base, 0], weights_f)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = (
                            T.max(scores_3d[n, qi, head], 0.0) * weights_f[qi, head]
                        )
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = (
                            reduced[n, qi]
                        )

        return weighted_mqa_prefill_generic


def weighted_mqa_prefill(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: Optional[torch.Tensor],
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MQA logits prefill (bf16) through the generic TileLang kernel."""
    if kv_scales is not None:
        raise NotImplementedError("TileLang weighted FP8 MQA prefill requires gfx938 WASP")
    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    if Q.dtype != torch.bfloat16 or KV.dtype != torch.bfloat16:
        raise ValueError(f"expects bf16 Q/KV, got {Q.dtype}/{KV.dtype}")
    if Q.ndim != 3 or KV.ndim != 2 or Q.shape[-1] != KV.shape[-1]:
        raise ValueError(
            f"expects Q [rows, H, D] and KV [keys, D], got {tuple(Q.shape)} "
            f"and {tuple(KV.shape)}"
        )
    rows, heads, head_dim = Q.shape
    keys = KV.shape[0]
    if weights.shape != (rows, heads):
        raise ValueError(
            f"weights must be [rows, H]={rows, heads}, got {tuple(weights.shape)}"
        )

    if D_out is not None:
        if D_out.shape != (rows, keys) or D_out.dtype != torch.float32:
            raise ValueError("D_out must be float32 [rows, keys]")
        logits = D_out
    else:
        logits = torch.empty((rows, keys), dtype=torch.float32, device=Q.device)
    if not rows or not keys:
        return logits.fill_(-float("inf"))

    starts = cu_starts.to(device=Q.device, dtype=torch.int32).contiguous()
    ends = cu_ends.to(device=Q.device, dtype=torch.int32).contiguous()
    # Same launch geometry as the generic QSA prefill: logical-M=128.
    threads = 256
    block_q = max(1, 128 // heads)
    num_stages = 1 if _is_hip() else 3

    _tilelang_weighted_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
        num_stages=num_stages,
        threads=threads,
    )(
        Q.contiguous().reshape(-1, head_dim),
        KV.contiguous(),
        weights.to(device=Q.device, dtype=torch.float32).contiguous(),
        logits,
        starts,
        ends,
    )
    _tilelang_qsa_mqa_mask_kernel()(logits, starts, ends)
    return logits


def torch_mqa_logits(
    Q: torch.Tensor,
    KV: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """float32 reference for the weighted head reduction.

    ``logits[m, n] = sum_h relu(Q[m, h] . KV[n]) * weights[m, h]``, with
    columns outside ``[start, end)`` set to ``-inf``.
    """
    scores = torch.relu(torch.einsum("mhd,nd->mnh", Q.float(), KV.float()))
    logits = (scores * weights.float().unsqueeze(1)).sum(dim=-1)
    columns = torch.arange(KV.shape[0], device=Q.device).unsqueeze(0)
    valid = (columns >= cu_starts.to(Q.device).reshape(-1, 1)) & (
        columns < cu_ends.to(Q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


__all__ = ["weighted_mqa_prefill", "torch_mqa_logits"]
