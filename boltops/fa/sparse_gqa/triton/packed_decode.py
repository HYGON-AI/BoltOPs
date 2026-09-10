"""Packed ragged Sparse-GQA decode migrated from the Qwen QSA path."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .compact import is_fp8_kv_dtype, _unit_scale
from .config import load_config


def _validate_sparse_gqa_dtypes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> bool:
    if k.dtype != v.dtype:
        raise ValueError(f"QSA K/V dtypes must match, got {k.dtype} and {v.dtype}")
    kv_is_fp8 = is_fp8_kv_dtype(k.dtype)
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"QSA expects BF16/FP16 queries, got {q.dtype}")
    if not kv_is_fp8 and k.dtype != q.dtype:
        raise ValueError(f"QSA K/V must match query dtype {q.dtype} or use FP8")
    return kv_is_fp8


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    k_scale,
    v_scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return

    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        if KV_IS_FP8:
            keys = keys.to(q_values.dtype)
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        if KV_IS_FP8:
            values = values.to(q_values.dtype)
        scores = tl.dot(q_values, keys)
        if KV_IS_FP8:
            scores *= k_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        if KV_IS_FP8:
            accumulator = (
                accumulator * alpha[:, None]
                + tl.dot(probabilities.to(values.dtype), values) * v_scale
            )
        else:
            accumulator = tl.dot(
                probabilities.to(values.dtype), values, accumulator * alpha[:, None]
            )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max

    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def _validate_decode(q, k, v, indices, cu_q, cu_k, kv_lens) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be packed rank-3 [N,H,D] tensors")
    if k.shape != v.shape or q.shape[2] != k.shape[2]:
        raise ValueError("packed Q/K/V dimensions must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if indices.ndim not in (2, 3):
        raise ValueError("indices must be [T,topk] or [T,Hkv,topk]")
    if indices.shape[0] < q.shape[0]:
        raise ValueError("indices must provide a row for every query")
    if indices.ndim == 3 and indices.shape[1] != k.shape[1]:
        raise ValueError("per-KV-head indices must match the KV head count")
    if cu_q.ndim != 1 or cu_k.ndim != 1 or kv_lens.ndim != 1:
        raise ValueError("cu_q, cu_k and kv_lens must be rank 1")
    if cu_q.shape != cu_k.shape or kv_lens.shape[0] != cu_q.shape[0] - 1:
        raise ValueError("ragged metadata batch dimensions must match")
    tensors = (q, k, v, indices, cu_q, cu_k, kv_lens)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must use the same device")
    for name, tensor in (
        ("indices", indices),
        ("cu_q", cu_q),
        ("cu_k", cu_k),
        ("kv_lens", kv_lens),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64")
    _validate_sparse_gqa_dtypes(q, k, v)


def _launch_sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    grid_m: int,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    config: Optional[dict] = None,
):
    _validate_decode(q, k, v, indices, cu_q, cu_k, kv_lens)
    kv_is_fp8 = is_fp8_kv_dtype(k.dtype)
    _, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    tuned = (
        load_config(
            "decode",
            cu_q.shape[0] - 1,
            indices.shape[-1],
            num_q_heads,
            num_kv_heads,
            head_dim,
            q.dtype,
        )
        if config is None
        else config
    )
    config = {"BLOCK_N": 32, "num_warps": 8, "num_stages": 2} | tuned
    block_n = int(config["BLOCK_N"])
    launch = {key: value for key, value in config.items() if key != "BLOCK_N"}
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(grid_m, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        _unit_scale(k_scale),
        _unit_scale(v_scale),
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        KV_IS_FP8=kv_is_fp8,
        **launch,
    )
    return out


def sparse_gqa_packed_decode_triton(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    """Run one packed sparse-attention row per request without a host sync.

    Decode and speculative-verify already compact the selected K/V rows into
    contiguous request segments. This wrapper reuses the chunk-prefill kernel
    with a fixed query length of one, avoiding a device-to-host sync during
    graph capture.
    """
    k, v = k.contiguous(), v.contiguous()
    return _launch_sparse_gqa_chunk_prefill(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale, 1, k_scale, v_scale
    )
