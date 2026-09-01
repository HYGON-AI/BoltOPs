"""Weighted-ReLU indexer logits over e4m3-rounded proxy activations.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``dsa_kernels.indexer_logits``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _indexer_logits_kernel(
    index_q_ptr,
    index_k_ptr,
    weights_ptr,
    out_ptr,
    seq_q,
    seq_k,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_k_token,
    stride_w_token,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Weighted-ReLU indexer scores for one (query tile, key tile, provider group).

        score[q, r] = sum_h relu(dot(q_qh, k_r)) * w_qh

    The ReLU is per head and *inside* the sum, so the heads cannot be folded into a
    single matmul -- hence the loop. The per-head weights are signed, so applying
    ReLU to each dot does **not** make the final weighted sum non-negative. Exact
    zeros are still common because a score is zero whenever every head dot is
    negative; for Step-4's 4 heads per group that event is about 6.25%. Those zeros
    are what the top-k tie-break then has to resolve, so they are load-bearing,
    not noise.
    """
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    query_ok = queries < seq_q
    key_ok = keys < seq_k

    key_tile = tl.load(
        index_k_ptr + dims[:, None] + keys[None, :] * stride_k_token,
        mask=key_ok[None, :],
        other=0.0,
    )

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + queries[:, None] * stride_q_token
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + queries * stride_w_token + group * stride_w_group + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    rows = group * seq_q + queries
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def round_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the indexer's e4m3 activation rounding, then widen back to bfloat16.

    The deployed kernel quantizes indexer activations to e4m3 before the score matmul,
    even though the checkpoint stores these weights in bf16. Skipping the rounding
    changes which regions get selected, so it is part of the semantics and not an
    optimization we can decline.

    Widening back to bf16 instead of feeding e4m3 into the matmul is deliberate. It is
    lossless -- e4m3 keeps 3 mantissa bits and bf16 keeps 8 -- and it keeps the kernel
    off Triton's fp8 MMA path, which is where portability across Triton versions and
    non-NVIDIA backends breaks down. The cost is peak throughput on Hopper, not
    accuracy.
    """
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_q: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores over proxy keys.

    Args:
        index_q: ``[seq_q, groups, heads_per_group, proxy_dim]``.
        weights: ``[seq_q, groups, heads_per_group]``, already carrying the
            ``heads_per_group ** -0.5`` prescale.
        index_k: ``[seq_k, 1, proxy_dim]`` -- one shared key head (the indexer is MQA
            even though the main attention is GQA).

    Returns:
        ``[groups * seq_q, seq_k]`` float32, group-major then query. That row order is
        what the region selector consumes, so it is a contract rather than a convenience.
    """
    if index_k.shape[1] != 1:
        raise ValueError(f"indexer keys are MQA; expected one head, got {index_k.shape[1]}")
    seq_q, groups, heads_per_group, proxy_dim = index_q.shape
    seq_k = index_k.shape[0]
    if index_k.shape[2] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} k={index_k.shape[2]}")

    quantized_q = round_activations_e4m3(index_q)
    quantized_k = round_activations_e4m3(index_k)
    out = torch.empty((groups * seq_q, seq_k), device=index_q.device, dtype=torch.float32)

    _indexer_logits_kernel[
        (triton.cdiv(seq_q, block_q), triton.cdiv(seq_k, block_k), groups)
    ](
        quantized_q,
        quantized_k,
        weights,
        out,
        seq_q,
        seq_k,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        quantized_k.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )
    return out
