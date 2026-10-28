# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py
"""FP8 paged MQA logits for sparse-attention indexer (decode / paged KV).

Public entries: ``triton_fp8_paged_mqa_logits`` and
``triton_fp8_paged_mqa_logits_stage1`` (re-exported from ``boltops.mqa_logits``).

Vendored from hygon aiter ``aiter.ops.triton.attention.pa_mqa_logits``
(JIT ``_deepgemm_fp8_paged_mqa_logits`` in ``_triton_kernels``). Formula matches
DeepGEMM / vLLM paged decode path:

    logits[b*next_n+n, t] = sum_h ReLU(Q[b,n,h] · K[t] * scale[t]) * W[b*next_n+n,h]
    causal: t <= context_lens[b] - next_n + n; outside valid KV -> -inf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.utility.arch_info import get_fp8_e4m3_dtype
from boltops.utility.triton_capability import get_triton_capabilities

# ChunkK=64 beat 128 and 256 at every SplitKV tried, on both short (avg_kv
# ~4k) and long (~72k) contexts; the margin grows with context length (728us vs
# 858us at avg_kv=72000, SplitKV=24). The narrower tile keeps the loop body
# small enough to stay off the LDS/barrier ceiling that the M-axis reduction
# used to dominate. See test/mqa_logits/bench_chunkk_sweep.py.
DEFAULT_CHUNK_K = 64
DEFAULT_CHUNK_Q_STAGE1 = 64
DEFAULT_WAVE_PER_EU = 2
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1
DEFAULT_KV_BLOCK_SIZE = 1
DEFAULT_KV_BLOCK_SIZE_PAGED = 64
# The batch-derived SplitKV term sizes the grid for occupancy but ignores how
# much KV each program will walk: at SplitKV=10 a 79k-context row leaves every
# program ~1230 tiles of work while the machine idles. Capping per-program work
# at roughly this many KV tokens recovers ~1.19x at long context and is a no-op
# below ~16k (see test/mqa_logits/bench_splitkv_heuristic.py).
SPLIT_KV_TOKENS_PER_PROGRAM = 1536
# Floor on the SplitKV grid, in waves. The occupancy term in the heuristic
# targets exactly *one* wave, which is only right when the whole problem is about
# one wave's worth of work. It is badly wrong when there are few q tiles: at
# batch=1 there is a single q tile, so the grid *is* ``split_kv``, and a
# 72000-token row got 75 CTAs for ~1125 KV tiles -- 15 serial tiles each, measured
# 1.66x slower than the optimum of 576 CTAs (8 waves). Measured optima track a
# floor near here: batch 1/2/4 want 576/288/144 programs, while batch >= 16 is
# already above it and is untouched. The curve turns back up past ~12 waves
# because every program re-loads the loop-invariant q (8KB), which at 576
# programs is already 44% of that row's KV stream. See
# test/mqa_logits/bench_paged_splitkv_sweep.py.
MIN_SPLIT_KV_WAVES = 8
# Short rows are the mirror image of the case above. The tokens term bounds
# per-program work from *above* (1536 tokens = 24 tiles); nothing bounds it from
# below, so at ~4k context the grid is set purely by the occupancy term, which
# sizes about one wave of q tiles. Measured: batch=64/heads=64 gets split_kv=10
# (640 CTAs for a 71-tile row) and is 4.1% slower than ~24 waves; heads=32 is
# 7.0%. Letting a short row walk ~3 tiles per program recovers that. The same
# "3 tiles per program" applied unconditionally costs up to 31.6% on long rows
# (batch=128/avg_kv=72000 wants split_kv=52, i.e. ~24 tiles per program), so the
# term is gated to rows short enough that the tokens term is not already doing
# its job, and capped in total programs so a wide batch cannot inflate the grid.
# See test/mqa_logits/bench_paged_splitkv_regime.py and
# test/mqa_logits/bench_paged_splitkv_rule_ab.py.
SPLIT_KV_TILES_PER_PROGRAM = 3
SPLIT_KV_SHORT_ROW_TILES = 96
SPLIT_KV_SHORT_ROW_PROGRAMS = 2048
# Bucket width for the persistent bucket scheduler, in KV tokens (= 2 x
# DEFAULT_CHUNK_K). The earlier host-schedule sweep landed on 1024, but that was
# measured against a scheduler whose ~500us build dominated the comparison; with
# the schedule on the device, narrow buckets win everywhere. A bucket is a
# program's serial work, so width is exposed memory latency per program -- at
# B=1 that alone was the difference between 27us and 8us. See
# test/mqa_logits/bench_bucket_tokens_sweep.py.
DEFAULT_BUCKET_TOKENS = 128
# CTAs per CU for the persistent bucket grid. Only used when ``max_ctx`` is
# unavailable -- see ``_default_bucket_num_programs`` for the adaptive rule.
DEFAULT_BUCKET_WAVES_PER_EU = 4
# Bounds on the adaptive grid, in CTAs per CU. The grid is a host constant, so
# it cannot track the real bucket count; over-provisioning is cheap (an empty
# program only runs the binary search before exiting) but not free. Scored on the
# full batch x distribution matrix: 32 beat 8/16/64 on the sum, and the wide
# batches are the ones that need the headroom.
DEFAULT_BUCKET_MIN_PROGRAMS_PER_CU = 2
DEFAULT_BUCKET_MAX_PROGRAMS_PER_CU = 32


@dataclass(frozen=True)
class PagedBatchInfo:
    """Host-side batch facts for one decode launch.

    Produced by :func:`triton_fp8_paged_mqa_logits_prepare`. The launch path
    only consumes ``max_ctx`` (for the SplitKV heuristic); the shape fields are
    informational. Shape asserts stay on the launch path -- ``prepare`` does
    not re-validate them.

    The launch path never reads ``context_lens`` values on the host (same trust
    model as hipc / aiter). ``prepare`` resolves ``max_ctx`` once per batch
    so the SplitKV heuristic stays context-aware without a per-step sync, and
    so the op stays graph-capturable.
    """

    batch_size: int
    next_n: int
    heads: int
    hidden_dim: int
    block_size: int
    index_dim: int
    max_block_len: int
    max_ctx: Optional[int] = None


@dataclass(frozen=True)
class PagedBucketSchedule:
    """Device-resident work partitioner for the bucket scheduler.

    Produced by :func:`triton_fp8_paged_mqa_logits_build_bucket_schedule` and
    handed to ``triton_fp8_paged_mqa_logits`` as ``bucket_schedule=``.

    ``base`` is the exclusive prefix sum of per-row bucket counts (``base[r]`` =
    flat index of row ``r``'s first bucket, ``base[num_rows]`` = total), written
    by ``_paged_bucket_prefix_kernel``. It is the counterpart of hipc's
    ``schedule_metadata``: a small int32 buffer computed on the device by a
    kernel, read by the compute kernel, and never read back by the host.

    The buffer is passed in (rather than allocated here) so that its address is
    stable across calls, which is what lets the whole sequence -- prefix kernel
    included -- sit inside a CUDA/HIP graph capture. Allocate it once for the
    largest batch the graph bucket will see and keep overwriting it in place.
    """

    base: torch.Tensor
    num_rows: int
    bucket_tokens: int
    num_programs: int

# Shared-memory model for the paged kernels (measured on HCU, HiddenDim=128,
# num_stages has no effect, 64KB hardware limit):
#
#   ChunkK    ChunkQ < 16        ChunkQ >= 16
#   -------   --------------     -------------
#      64        32768               8192
#     128        65536              16384
#     256       131072 (over)       32768
#
# When the dot M dimension (== ChunkQ) drops below 16 Triton picks a layout that
# no longer shares the K tile, quadrupling smem. ChunkK=128 then sits exactly on
# the 64KB limit with zero headroom, so small-M shapes must use ChunkK=64.
_CHUNK_Q_TINY = 16
_CHUNK_K_TINY_Q = 64
_CHUNK_K_MAX = 256


def _resolve_chunk_k(chunk_k: int, chunk_q: int) -> int:
    """Clamp ``chunk_k`` for shapes where the M dimension is too small.

    With ``ChunkQ < 16`` the K tile is no longer shared, so ``ChunkK=256``
    overflows the 64KB shared-memory budget and ``ChunkK=128`` lands exactly on
    the limit. ``ChunkK=64`` is the largest safe tile for those shapes.
    """
    if chunk_q < _CHUNK_Q_TINY:
        return min(chunk_k, _CHUNK_K_TINY_Q)
    return chunk_k


def _resolve_stage1_chunk_q(chunk_q: Optional[int], heads: int) -> int:
    """Pick a ``chunk_q`` that divides ``heads`` when the caller left it unset.

    The stage-1 kernel tiles the head dimension and requires
    ``heads % chunk_q == 0``. Defaulting to 64 breaks small head counts
    (``heads=1`` raised ``heads must divide chunk_q``), so fall back to the
    largest power-of-two divisor of ``heads`` that does not exceed the default.
    """
    if chunk_q is not None:
        return chunk_q
    if heads % DEFAULT_CHUNK_Q_STAGE1 == 0:
        return DEFAULT_CHUNK_Q_STAGE1
    chunk = 1
    while chunk * 2 <= heads and (heads % (chunk * 2) == 0):
        chunk *= 2
    return min(chunk, DEFAULT_CHUNK_Q_STAGE1)


def _check_paged_inputs(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    *,
    out_len_dim: int,
    stage1: bool,
) -> PagedBatchInfo:
    """Validate host-visible shapes on the launch path.

    Never asserts on ``context_lens`` numeric bounds -- the caller owns that
    (same as hipc / aiter). Does not resolve ``max_ctx``; use
    :func:`triton_fp8_paged_mqa_logits_prepare` for that.
    """
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    assert heads >= 1 and (heads & (heads - 1)) == 0, (
        f"num q. heads should be positive power of 2; got {heads}."
    )
    assert hidden_dim & (hidden_dim - 1) == 0, "head size should be power of 2."
    assert hidden_dim <= index_dim, (
        f"head_dim {hidden_dim} exceeds kv_cache last dim {index_dim}."
    )
    assert context_lens.numel() == batch_size, (
        f"context_lens must have batch size {batch_size}; got {context_lens.numel()}."
    )
    assert kv_indices.size(0) == batch_size, (
        f"kv_indices batch dim {kv_indices.size(0)} != batch_size {batch_size}."
    )
    if not stage1:
        # out_logits: [batch * next_n, max_model_len]
        assert out_len_dim >= max_model_len, (
            f"out_logits width {out_len_dim} < max_model_len {max_model_len}."
        )

    return PagedBatchInfo(
        batch_size=batch_size,
        next_n=next_n,
        heads=heads,
        hidden_dim=hidden_dim,
        block_size=block_size,
        index_dim=index_dim,
        max_block_len=max_block_len,
        max_ctx=None,
    )


def triton_fp8_paged_mqa_logits_prepare(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
) -> PagedBatchInfo:
    """Resolve ``max_ctx`` once on the host, outside the hot path.

    Returns the :class:`PagedBatchInfo` to pass as
    ``triton_fp8_paged_mqa_logits(..., batch_info=...)``. Reads
    ``max(context_lens)`` here (one sync per batch) so the SplitKV heuristic
    stays context-aware and the launch path stays graph-capturable.

    Does not re-run shape asserts (launch still does) and does not validate
    numeric ``context_lens`` bounds. ``out_logits`` / ``max_model_len`` are
    launch-only and are not accepted here.
    """
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()
    max_ctx = (
        int(context_lens.max().item()) if context_lens.numel() > 0 else None
    )
    return PagedBatchInfo(
        batch_size=batch_size,
        next_n=next_n,
        heads=heads,
        hidden_dim=hidden_dim,
        block_size=block_size,
        index_dim=index_dim,
        max_block_len=max_block_len,
        max_ctx=max_ctx,
    )


def _filter_hip_launch_kwargs(kwargs: dict) -> dict:
    """Drop unsupported HIPOptions instead of passing False/unsupported values."""
    caps = get_triton_capabilities()
    out = {}
    for key, value in kwargs.items():
        if key in (
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
            "schedule_hint",
            "use_block_pingpong",
            "sched_latency",
        ):
            if not caps.supports(key):
                continue
        out[key] = value
    return out


_TOTAL_CU_COUNT: Optional[int] = None


def _default_total_cu_count() -> int:
    """CU count for the SplitKV heuristic, cached after the first call.

    ``get_device_properties`` costs ~2.6us/call and the value never changes for
    a device, so re-querying it every decode step is pure overhead.
    """
    global _TOTAL_CU_COUNT
    if _TOTAL_CU_COUNT is None:
        try:
            _TOTAL_CU_COUNT = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        except Exception:
            _TOTAL_CU_COUNT = 256
    return _TOTAL_CU_COUNT


def _resolve_split_kv(
    split_kv: Optional[int],
    *,
    tile_q_count: int,
    total_cu_count: int,
    waves_per_eu: int,
    max_ctx: Optional[int],
    chunk_k: int,
) -> int:
    """Pick ``SplitKV`` for one launch. ``split_kv`` wins if the caller set it.

    Four terms, each a lower bound; the result is their max, then capped at one
    KV tile per program:

    * occupancy: about one wave of q tiles, times ``waves_per_eu``.
    * tokens: at most ``SPLIT_KV_TOKENS_PER_PROGRAM`` KV tokens per program
      (long rows).
    * waves: at least ``MIN_SPLIT_KV_WAVES`` waves of programs (few q tiles).
    * short-row tiles: at most ``SPLIT_KV_TILES_PER_PROGRAM`` tiles per program
      on rows of ``<= SPLIT_KV_SHORT_ROW_TILES`` tiles, capped at
      ``SPLIT_KV_SHORT_ROW_PROGRAMS`` programs total.

    Context-aware terms need ``max_ctx``. Without it (no ``batch_info``) only
    the occupancy term applies.
    """
    if split_kv is not None:
        return split_kv
    split_kv = (
        (max(1, total_cu_count // tile_q_count) + 4) // 5 * 5 * waves_per_eu
    )
    if max_ctx is None:
        return split_kv
    tiles = -(-max_ctx // chunk_k)
    split_kv = max(split_kv, -(-max_ctx // SPLIT_KV_TOKENS_PER_PROGRAM))
    floor = -(-(MIN_SPLIT_KV_WAVES * total_cu_count) // tile_q_count)
    split_kv = max(split_kv, min(floor, tiles))
    if tiles <= SPLIT_KV_SHORT_ROW_TILES:
        short = min(
            -(-tiles // SPLIT_KV_TILES_PER_PROGRAM),
            max(1, SPLIT_KV_SHORT_ROW_PROGRAMS // tile_q_count),
        )
        split_kv = max(split_kv, min(short, tiles))
    return split_kv


@triton.jit
def _sum_combine(a, b):
    return a + b


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    # Same operand swap as the block kernel: q is fed transposed so the head
    # reduction runs along N, the accumulator's per-lane-contiguous axis.
    q_t = tl.trans(q)

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        pos = context_idx + tl.arange(0, ChunkK)
        mask_kv = pos < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + pos,
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        # [ChunkK, ChunkQ]: KV positions on M, heads on N. The per-token scale
        # now broadcasts along M, so it no longer needs a transpose.
        o = tl.dot(k, q_t)
        o = o * k_scale_f
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :]

        mask = pos <= context_length - next_n + pid_next_n
        o = tl.where(mask[:, None], o, float("-inf"))

        logits = tl.sum(o, axis=1)
        tl.store(
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + pos,
            logits,
            mask=pos < max_model_len,
        )


@triton.jit
def _fp8_paged_mqa_logits_block_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    SplitKV: tl.constexpr = 1,
    FillTail: tl.constexpr = 0,
    TailBlock: tl.constexpr = 4096,
):
    """Paged decode with ``block_table`` (logical block -> physical page), row-major KV."""
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    # The head reduction runs along the MMA's N axis, so q is fed transposed as
    # the B operand. In the MFMA accumulator N is contiguous per lane while M is
    # spread across lanes, so reducing M costs shuffles plus an LDS round-trip on
    # every KV tile -- measured at ~2.1x the rest of the loop body combined.
    # Loop-invariant, so transpose once.
    q_t = tl.trans(q)

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        pos = context_idx + tl.arange(0, ChunkK)
        mask_kv = pos < context_length
        block_rk = pos // KVBlockSize
        tok_in_block = pos % KVBlockSize
        phys_block = tl.load(
            block_table + pid_batch * max_block_len + block_rk,
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + phys_block[:, None] * stride_k_page
            + tok_in_block[:, None] * stride_k_token
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer
            + phys_block * stride_scale_page
            + tok_in_block * stride_scale_token,
            mask=mask_kv,
            other=0.0,
        )

        # [ChunkK, ChunkQ]: KV positions on M, heads on N.
        o = tl.dot(k, q_t)
        o = o * k_scale_f[:, None]
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :]

        mask = pos <= context_length - next_n + pid_next_n
        o = tl.where(mask[:, None], o, float("-inf"))

        logits = tl.sum(o, axis=1)
        tl.store(
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + pos,
            logits,
            mask=pos < max_model_len,
        )

    if FillTail:
        # See the note in _fp8_paged_mqa_logits_block64_kernel: the row tail
        # [context_length, max_model_len) is -inf by contract, and filling it
        # here is what lets the caller skip its own full().
        out_row = (OutLogits_buffer
                   + (pid_batch * next_n + pid_next_n) * stride_out_batch)
        for tail in range(context_length + pid_split_kv * TailBlock,
                          max_model_len,
                          SplitKV * TailBlock):
            tail_pos = tail + tl.arange(0, TailBlock)
            tl.store(
                out_row + tail_pos,
                float("-inf"),
                mask=tail_pos < max_model_len,
            )


@triton.jit
def _fp8_paged_mqa_logits_block64_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    SplitKV: tl.constexpr = 1,
    FillTail: tl.constexpr = 0,
    TailBlock: tl.constexpr = 4096,
):
    """Paged decode, specialized for ``ChunkK == KVBlockSize`` (one page per tile).

    Identical math to :func:`_fp8_paged_mqa_logits_block_kernel` -- bitwise so,
    see ``test/mqa_logits/bench_paged_elem_ab.py`` -- but with the page index
    scalarized. The specialized form is a separate kernel rather than a constexpr
    branch in the generic one because Triton scopes variables defined inside a
    constexpr ``if`` to that branch, so a branch cannot share the loop-carried
    page index with the code after it.

    With ``ChunkK == KVBlockSize`` and a loop start that is a multiple of
    ``ChunkK``, every tile lies inside exactly one page::

        block_rk     = pos // KVBlockSize = context_idx // KVBlockSize  # scalar
        tok_in_block = pos %  KVBlockSize = arange(0, ChunkK)          # invariant

    The generic kernel instead derives both as ``[ChunkK]`` vectors every
    iteration and gathers one address 64 times. That costs ~25 instructions of
    address ALU per tile and, more importantly, makes the K address depend on a
    *loaded* value, so the pipeliner can neither hoist it nor meaningfully
    pipeline the loop (``num_stages=2`` nearly triples the emitted code instead
    of helping). With a scalar page index the K address is affine in the loop
    variable, which also makes the block-table load hand-prefetchable one tile
    ahead -- worth a further ~4.5% over scalarizing alone.

    Measured on an idle HCU at ``B=64, heads=64, avg_kv=72000, chunk_k=64``:
    -8.7% for the scalar index, -13.3% with the prefetch.
    """
    tl.static_assert(ChunkK == KVBlockSize,
                     "one page per tile is required; use the generic kernel")
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )
    q_t = tl.trans(q)

    # loop-invariant: the per-token offset inside a page, and the full K element
    # grid. Hoisting these out is what removes the per-tile address ALU.
    tok_in_block = tl.arange(0, ChunkK)
    k_off = (tok_in_block[:, None] * stride_k_token
             + tl.arange(0, HiddenDim)[None, :])
    s_off = tok_in_block * stride_scale_token
    bt_row = block_table + pid_batch * max_block_len

    # The lookahead must target the *last* index the loop visits, not
    # context_idx + ChunkK: on the final tile the latter is already past the end
    # and would hand that iteration the wrong page. ``cdiv`` goes <= 0 for an
    # empty split (split_context_length can be <= 0), so the prologue index is
    # clamped into [0, context_length - 1] -- always inside the block table --
    # rather than relying on the first tile being valid, since the prologue load
    # executes even when the loop body never runs.
    _iters = tl.cdiv(split_context_length, ChunkK)
    _last = split_context_start + (_iters - 1) * ChunkK
    _p0 = tl.minimum(tl.maximum(tl.minimum(split_context_start, _last), 0),
                     tl.maximum(context_length - 1, 0))
    phys_block = tl.load(bt_row + _p0 // KVBlockSize)

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        pos = context_idx + tl.arange(0, ChunkK)
        mask_kv = pos < context_length

        # one tile of lookahead: without it the chain
        # phys_block -> k address -> k load -> dot is fully serialized, and the
        # automatic pipeliner will not hoist this load (num_stages=2 makes the
        # kernel ~1.7x slower, not faster).
        nxt = tl.minimum(context_idx + ChunkK, _last)
        next_phys = tl.load(bt_row + nxt // KVBlockSize)

        k = tl.load(
            KV_buffer + phys_block * stride_k_page + k_off,
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer + phys_block * stride_scale_page + s_off,
            mask=mask_kv,
            other=0.0,
        )

        # [ChunkK, ChunkQ]: KV positions on M, heads on N.
        o = tl.dot(k, q_t)
        o = o * k_scale_f[:, None]
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :]

        mask = pos <= context_length - next_n + pid_next_n
        o = tl.where(mask[:, None], o, float("-inf"))

        logits = tl.sum(o, axis=1)
        tl.store(
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + pos,
            logits,
            mask=pos < max_model_len,
        )
        phys_block = next_phys

    if FillTail:
        # Row tail [context_length, max_model_len) is -inf by contract: the
        # consumer topk's over the *declared* width and reads -inf as "no token
        # here", so a row missing its tail can surface stale indices as real
        # ones. Filling it here removes the caller's separate full(), which in
        # the framework is a whole extra kernel plus its host alloc + dispatch.
        #
        # The split owns offsets pid_split_kv * TailBlock, then strides by
        # SplitKV * TailBlock, so the tail -- which is most of the row -- is
        # spread across the splits instead of landing on one program.
        #
        # The main loop's final tile can also cover [context_length,
        # align_up(context_length)) and writes -inf there; the overlap is the
        # same value on both sides, so the duplicate store is benign.
        out_row = (OutLogits_buffer
                   + (pid_batch * next_n + pid_next_n) * stride_out_batch)
        for tail in range(context_length + pid_split_kv * TailBlock,
                          max_model_len,
                          SplitKV * TailBlock):
            tail_pos = tail + tl.arange(0, TailBlock)
            tl.store(
                out_row + tail_pos,
                float("-inf"),
                mask=tail_pos < max_model_len,
            )


@triton.jit
def _fp8_paged_mqa_logits_bucket_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    schedule_ptr,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    BucketTokens: tl.constexpr,
):
    """Paged decode driven by an explicit, load-balanced work schedule.

    The sibling ``_fp8_paged_mqa_logits_block_kernel`` gives every row the same
    ``SplitKV``, so every row gets the same number of programs. A production
    batch mixes context lengths (4k rows next to 72k rows), so the short rows'
    programs finish immediately while the long rows' programs serialise -- the
    parallelism is spent where there is no work.

    This variant instead reads a flat ``(row, k_start)`` list built on the host:
    row ``r`` contributes ``ceil(ctx_r / BucketTokens)`` entries, so its program
    count tracks its own KV length. ``row`` is ``batch * next_n + n_idx``, which
    keeps the ``next_n`` handling identical to the block kernel.
    """
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    row = tl.load(schedule_ptr + remain_pid * 2 + 0)
    k_start = tl.load(schedule_ptr + remain_pid * 2 + 1)

    pid_next_n = row % next_n
    pid_batch = row // next_n

    context_length = tl.load(context_len_ptr + pid_batch)
    # This bucket owns [k_start, k_start + BucketTokens), clamped to the row.
    k_end = tl.minimum(k_start + BucketTokens, context_length)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    # Same operand swap as the block kernel: feed q transposed so the head
    # reduction runs along N, the accumulator's contiguous axis.
    q_t = tl.trans(q)

    for context_idx in range(k_start, k_end, ChunkK):
        pos = context_idx + tl.arange(0, ChunkK)
        mask_kv = pos < context_length
        block_rk = pos // KVBlockSize
        tok_in_block = pos % KVBlockSize
        phys_block = tl.load(
            block_table + pid_batch * max_block_len + block_rk,
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + phys_block[:, None] * stride_k_page
            + tok_in_block[:, None] * stride_k_token
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer
            + phys_block * stride_scale_page
            + tok_in_block * stride_scale_token,
            mask=mask_kv,
            other=0.0,
        )

        # [ChunkK, ChunkQ]: KV positions on M, heads on N.
        o = tl.dot(k, q_t)
        o = o * k_scale_f[:, None]
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :]

        mask = pos <= context_length - next_n + pid_next_n
        o = tl.where(mask[:, None], o, float("-inf"))

        logits = tl.sum(o, axis=1)
        tl.store(
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + pos,
            logits,
            mask=pos < max_model_len,
        )


@triton.jit
def _paged_bucket_prefix_kernel(
    context_len_ptr,
    base_ptr,
    num_rows,
    NextN: tl.constexpr,
    BucketTokens: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Device-side work partitioner: exclusive prefix sum of per-row bucket counts.

    Writes ``base[r]`` = flat index of row ``r``'s first bucket, with
    ``base[num_rows]`` holding the total bucket count. That single monotone array
    is all the compute kernel needs to map a global bucket index back to
    ``(row, k_start)`` with one binary search, and it is what makes the grid a
    host constant.

    This replaces the host-side ``build_paged_bucket_schedule``. That version did
    ~10 small tensor ops plus one ``.item()`` to size the grid -- ~500us of host
    time on the critical path, which made the whole path both slow and unusable
    under graph capture. Here the work is B elements of arithmetic on the device,
    the output buffer is fixed and written in place, and nothing syncs.

    ``context_lens`` is indexed by batch while buckets are indexed by query row
    (``batch * next_n + n_idx``), so the load broadcasts through ``offs // NextN``.
    """
    offs = tl.arange(0, BLOCK_ROWS)
    row_in_bounds = offs < num_rows
    ctx = tl.load(context_len_ptr + (offs // NextN), mask=row_in_bounds, other=0)
    counts = tl.where(row_in_bounds, tl.cdiv(ctx, BucketTokens), 0)
    tl.store(base_ptr + offs, tl.cumsum(counts, axis=0) - counts, mask=row_in_bounds)
    tl.store(base_ptr + num_rows, tl.sum(counts, axis=0))


@triton.jit
def _fp8_paged_mqa_logits_bucket_persistent_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    base_ptr,
    num_rows,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    BucketTokens: tl.constexpr,
    LOG2_ROWS: tl.constexpr,
):
    """Bucket scheduler with a host-constant grid (``num_programs``).

    ``_fp8_paged_mqa_logits_bucket_kernel`` needs one program per ``(row,
    bucket)`` pair, so the grid depends on ``context_lens`` and the host has to
    read it back before launching -- a sync per step, and a hard blocker for
    CUDA graph capture.

    Here the grid is a fixed ``num_programs`` (a plain host constant, tuned to
    the device) and each program claims a *contiguous* slice of the global
    bucket list. Because every bucket is ``BucketTokens`` wide, splitting the
    list evenly by count is also splitting it evenly by work, so the balance the
    flat scheduler bought is preserved. The slice boundaries come from ``base``
    (device-resident, see ``_paged_bucket_prefix_kernel``), so no sync is needed
    to learn how many buckets there are.

    One binary search per program locates the owning row, after which the walk
    is sequential: ``k_start`` advances by ``BucketTokens`` and ``row`` advances
    once the row is exhausted. ``q``/``weights`` are reloaded only on a row
    change, since every bucket of a row shares them.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total = tl.load(base_ptr + num_rows)

    # Even split of the global bucket list; the last program absorbs the
    # remainder. ``flat_end > flat_start`` for every pid < num_programs as long
    # as total >= num_programs, and empty slices cost only the search below.
    flat_start = total * pid // num_programs
    flat_end = total * (pid + 1) // num_programs

    # Largest row whose first bucket is <= flat_start (``base`` is non-decreasing,
    # and strictly increasing whenever every row has at least one bucket -- which
    # the caller validates for the real entry point, but is not assumed here).
    lo = 0
    hi = num_rows - 1
    for _ in range(LOG2_ROWS):
        mid = (lo + hi) >> 1
        if tl.load(base_ptr + mid + 1) <= flat_start:
            lo = mid + 1
        else:
            hi = mid
    row = lo

    # Walk the slice row by row. ``take`` is how many of this row's buckets fall
    # inside [flat, flat_end); it is <= 0 for a row that is exhausted or empty
    # (a zero-length context is reachable -- the op trusts the caller for
    # numeric bounds), and then only ``row`` advances. Both loop variables
    # are bounded: ``flat`` by ``flat_end``, ``row`` by ``num_rows`` because the
    # last row ends at ``total >= flat_end``. So this always terminates.
    #
    # Deriving ``take`` from ``base`` rather than from ``k_start < context_length``
    # is what makes that guarantee hold. A zero-count row makes ``base`` flat, a
    # flat ``base`` can hand the binary search a row that owns no bucket, and
    # gating the body on the KV range alone would then spin without ever moving
    # ``flat``.
    flat = flat_start
    while flat < flat_end:
        row_end = tl.load(base_ptr + row + 1)
        take = tl.minimum(row_end, flat_end) - flat
        if take > 0:
            pid_batch = row // next_n
            pid_next_n = row % next_n
            context_length = tl.load(context_len_ptr + pid_batch)
            # This row's first bucket inside the slice. ``flat >= base[row]`` and
            # ``flat < base[row+1]``, so the offset is a real bucket of this row.
            k_start = (flat - tl.load(base_ptr + row)) * BucketTokens

            # Row-invariant: one reload per row, not per bucket.
            q = tl.load(
                Q_buffer
                + pid_batch * stride_q_batch
                + pid_next_n * stride_q_next_n
                + (tl.arange(0, ChunkQ) * stride_q_heads)[:, None]
                + tl.arange(0, HiddenDim)[None, :],
            )
            scale_weight = tl.load(
                weights
                + (pid_batch * next_n + pid_next_n) * stride_w_batch
                + tl.arange(0, ChunkQ)
            )
            # Same operand swap as the block kernel: feed q transposed so the head
            # reduction runs along N, the accumulator's contiguous axis.
            q_t = tl.trans(q)

            for bucket in range(0, take):
                bucket_start = k_start + bucket * BucketTokens
                k_end = tl.minimum(bucket_start + BucketTokens, context_length)
                for context_idx in range(bucket_start, k_end, ChunkK):
                    pos = context_idx + tl.arange(0, ChunkK)
                    mask_kv = pos < context_length
                    block_rk = pos // KVBlockSize
                    tok_in_block = pos % KVBlockSize
                    phys_block = tl.load(
                        block_table + pid_batch * max_block_len + block_rk,
                        mask=mask_kv,
                        other=0,
                    )

                    k = tl.load(
                        KV_buffer
                        + phys_block[:, None] * stride_k_page
                        + tok_in_block[:, None] * stride_k_token
                        + tl.arange(0, HiddenDim)[None, :],
                        mask=mask_kv[:, None],
                        other=0.0,
                    )
                    k_scale_f = tl.load(
                        scale_buffer
                        + phys_block * stride_scale_page
                        + tok_in_block * stride_scale_token,
                        mask=mask_kv,
                        other=0.0,
                    )

                    # [ChunkK, ChunkQ]: KV positions on M, heads on N.
                    o = tl.dot(k, q_t)
                    o = o * k_scale_f[:, None]
                    o = tl.maximum(o, 0.0)
                    o = o * scale_weight[None, :]

                    mask = pos <= context_length - next_n + pid_next_n
                    o = tl.where(mask[:, None], o, float("-inf"))

                    logits = tl.sum(o, axis=1)
                    tl.store(
                        OutLogits_buffer
                        + (pid_batch * next_n + pid_next_n) * stride_out_batch
                        + pos,
                        logits,
                        mask=pos < max_model_len,
                    )
            flat += take
        row += 1


@triton.jit
def _fp8_paged_mqa_logits_bucket_persistent64_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    base_ptr,
    num_rows,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    BucketTokens: tl.constexpr,
    LOG2_ROWS: tl.constexpr,
):
    """``_fp8_paged_mqa_logits_bucket_persistent_kernel`` with a scalarized page
    index (``ChunkK == KVBlockSize``). Bitwise identical to it.

    ``k_start`` is a multiple of ``BucketTokens`` and ``BucketTokens`` is a
    multiple of ``ChunkK``, so every tile starts on a page boundary and the whole
    tile lies in one page::

        block_rk     = context_idx // KVBlockSize   # scalar
        tok_in_block = arange(0, ChunkK)            # invariant

    The generic kernel derives both as ``[ChunkK]`` vectors per tile and gathers a
    single address 64 times, which costs address ALU in the loop and makes the K
    address depend on a loaded value.

    No block-table prefetch here, unlike the block kernel: a bucket is
    ``BucketTokens / ChunkK`` = 2 tiles, so there is no loop depth to amortize the
    lookahead over, and the next tile may live in the next bucket (a different
    ``k_start``) which the lookahead would have to special-case.
    """
    tl.static_assert(ChunkK == KVBlockSize,
                     "one page per tile is required; use the generic kernel")
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total = tl.load(base_ptr + num_rows)

    flat_start = total * pid // num_programs
    flat_end = total * (pid + 1) // num_programs

    lo = 0
    hi = num_rows - 1
    for _ in range(LOG2_ROWS):
        mid = (lo + hi) >> 1
        if tl.load(base_ptr + mid + 1) <= flat_start:
            lo = mid + 1
        else:
            hi = mid
    row = lo

    # Row-independent, so hoisted out of the row walk entirely
    tok_in_block = tl.arange(0, ChunkK)
    k_off = (tok_in_block[:, None] * stride_k_token
             + tl.arange(0, HiddenDim)[None, :])
    s_off = tok_in_block * stride_scale_token

    flat = flat_start
    while flat < flat_end:
        row_end = tl.load(base_ptr + row + 1)
        take = tl.minimum(row_end, flat_end) - flat
        if take > 0:
            pid_batch = row // next_n
            pid_next_n = row % next_n
            context_length = tl.load(context_len_ptr + pid_batch)
            k_start = (flat - tl.load(base_ptr + row)) * BucketTokens

            q = tl.load(
                Q_buffer
                + pid_batch * stride_q_batch
                + pid_next_n * stride_q_next_n
                + (tl.arange(0, ChunkQ) * stride_q_heads)[:, None]
                + tl.arange(0, HiddenDim)[None, :],
            )
            scale_weight = tl.load(
                weights
                + (pid_batch * next_n + pid_next_n) * stride_w_batch
                + tl.arange(0, ChunkQ)
            )
            q_t = tl.trans(q)
            bt_row = block_table + pid_batch * max_block_len

            for bucket in range(0, take):
                bucket_start = k_start + bucket * BucketTokens
                k_end = tl.minimum(bucket_start + BucketTokens, context_length)
                for context_idx in range(bucket_start, k_end, ChunkK):
                    pos = context_idx + tl.arange(0, ChunkK)
                    mask_kv = pos < context_length

                    # one page per tile (see docstring)
                    phys_block = tl.load(bt_row + context_idx // KVBlockSize)
                    k = tl.load(
                        KV_buffer + phys_block * stride_k_page + k_off,
                        mask=mask_kv[:, None],
                        other=0.0,
                    )
                    k_scale_f = tl.load(
                        scale_buffer + phys_block * stride_scale_page + s_off,
                        mask=mask_kv,
                        other=0.0,
                    )

                    # [ChunkK, ChunkQ]: KV positions on M, heads on N.
                    o = tl.dot(k, q_t)
                    o = o * k_scale_f[:, None]
                    o = tl.maximum(o, 0.0)
                    o = o * scale_weight[None, :]

                    mask = pos <= context_length - next_n + pid_next_n
                    o = tl.where(mask[:, None], o, float("-inf"))

                    logits = tl.sum(o, axis=1)
                    tl.store(
                        OutLogits_buffer
                        + (pid_batch * next_n + pid_next_n) * stride_out_batch
                        + pos,
                        logits,
                        mask=pos < max_model_len,
                    )
            flat += take
        row += 1


@triton.jit
def _fp8_paged_mqa_logits_stage1_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = (
            context_idx + tl.arange(0, ChunkK) <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        # aiter-style 3D store. Add ``pos < max_model_len`` so ChunkK>max_model_len
        # does not OOB-alias into later head rows (aiter omits this mask).
        pos = context_idx + tl.arange(0, ChunkK)
        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
            + pos[None, None, :],
            o[:, None, :],
            mask=(mask & (pos < max_model_len))[None, None, :],
        )


@triton.jit
def _fp8_paged_mqa_logits_stage1_block_kernel(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_page,
    stride_k_token,
    scale_buffer,
    stride_scale_page,
    stride_scale_token,
    context_len_ptr,
    block_table,
    weights,
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_block_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    """Paged decode stage-1 with ``block_table`` (logical block -> physical page).

    Per-head scores before summing over ``H`` (same contract as the
    ``KVBlockSize=1`` stage-1 kernel). Row-major in-page KV layout.
    """
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        pos = context_idx + tl.arange(0, ChunkK)
        mask_kv = pos < context_length
        block_rk = pos // KVBlockSize
        tok_in_block = pos % KVBlockSize
        phys_block = tl.load(
            block_table + pid_batch * max_block_len + block_rk,
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + phys_block[:, None] * stride_k_page
            + tok_in_block[:, None] * stride_k_token
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer
            + phys_block * stride_scale_page
            + tok_in_block * stride_scale_token,
            mask=mask_kv,
            other=0.0,
        )

        o = tl.dot(q, k.T)
        o = o * k_scale_f[None, :]
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = pos <= context_length - next_n + pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
            + pos[None, None, :],
            o[:, None, :],
            mask=(mask & (pos < max_model_len))[None, None, :],
        )


def _launch_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    weights: torch.Tensor,
    out_qk: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    max_block_len: int,
    *,
    chunk_q: int,
    chunk_k: int,
    split_kv: int,
    waves_per_eu: int,
    num_warps: int,
    num_stages: int,
) -> None:
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    grid = (batch_size * next_n * (heads // chunk_q * split_kv), 1, 1)
    hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
    _fp8_paged_mqa_logits_stage1_kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_block_len,
        num_warps=num_warps,
        num_stages=num_stages,
        ChunkQ=chunk_q,
        ChunkK=chunk_k,
        SplitKV=split_kv,
        HiddenDim=hidden_dim,
        **hip_kwargs,
    )


def _launch_fp8_paged_mqa_logits_stage1_block(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    weights: torch.Tensor,
    out_qk: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    max_block_len: int,
    *,
    kv_block_size: int,
    chunk_q: int,
    chunk_k: int,
    split_kv: int,
    waves_per_eu: int,
    num_warps: int,
    num_stages: int,
) -> None:
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    grid = (batch_size * next_n * (heads // chunk_q * split_kv), 1, 1)
    hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
    _fp8_paged_mqa_logits_stage1_block_kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_fp8.stride(1),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
        context_lens,
        block_table,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_block_len,
        num_warps=num_warps,
        num_stages=num_stages,
        ChunkQ=chunk_q,
        ChunkK=chunk_k,
        SplitKV=split_kv,
        HiddenDim=hidden_dim,
        KVBlockSize=kv_block_size,
        **hip_kwargs,
    )


def triton_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_qk: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    *,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    chunk_q: Optional[int] = None,
    chunk_k: Optional[int] = None,
    total_cu_count: Optional[int] = None,
    split_kv: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """Paged decode stage-1: per-head scores before summing over ``H``.

    Supports ``kv_block_size=1`` (per-token ``kv_indices``, aiter JIT path) and
    ``kv_block_size=64`` (``block_table`` logical-block -> physical-page mapping).
    Any other page size is rejected up front (``AssertionError``): upstream only
    ever uses 1 or 64 (``kv_block_size`` tracks the allocator's ``page_size``), and
    the ``page_size=1`` per-token path cannot be expressed with a block table.
    Only the row-major in-page KV layout is implemented; ``kv_block_size`` is the
    page size and is orthogonal to that layout. Mathematically equivalent to
    :func:`triton_fp8_paged_mqa_logits`, only the reduction split point differs.
    vLLM non-gfx942 path: ``out_qk.sum(dim=0)`` for final logits.

    Args:
        out_qk: ``[H, B*next_n, max_model_len]`` float32; prefilled with ``-inf``.
        kv_indices: for ``kv_block_size=1``, per-token physical rows; for
            ``kv_block_size=64``, pass ``block_table`` here.
        chunk_q: head tile size (default 64; vLLM often passes ``heads``).
        Other args: same as ``triton_fp8_paged_mqa_logits``.

    Returns:
        ``out_qk`` (in-place).
    """
    fp8_dtype = get_fp8_e4m3_dtype()
    if q_fp8.dtype != fp8_dtype:
        q_fp8 = q_fp8.to(fp8_dtype)

    chunk_q = _resolve_stage1_chunk_q(chunk_q, q_fp8.size(2))
    chunk_k = DEFAULT_CHUNK_K if chunk_k is None else chunk_k
    waves_per_eu = DEFAULT_WAVE_PER_EU if waves_per_eu is None else waves_per_eu
    num_warps = DEFAULT_NUM_WARPS if num_warps is None else num_warps
    num_stages = DEFAULT_NUM_STAGES if num_stages is None else num_stages

    if total_cu_count is None:
        total_cu_count = _default_total_cu_count()

    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    assert kv_block_size in (DEFAULT_KV_BLOCK_SIZE, DEFAULT_KV_BLOCK_SIZE_PAGED), (
        f"BoltOPs supports KVBlockSize {DEFAULT_KV_BLOCK_SIZE} and "
        f"{DEFAULT_KV_BLOCK_SIZE_PAGED}; got {kv_block_size}."
    )
    assert block_size == kv_block_size
    assert heads % chunk_q == 0, f"heads must divide chunk_q; got heads={heads}, chunk_q={chunk_q}."

    # Reject shapes whose ``context_lens`` would walk past the allocations.
    _check_paged_inputs(
        q_fp8, kv_cache, context_lens, kv_indices, max_model_len,
        out_len_dim=out_qk.size(-1), stage1=True,
    )

    # ChunkQ == chunk_q on this path.
    chunk_k = _resolve_chunk_k(chunk_k, chunk_q)

    tile_q_count = batch_size * next_n * (heads // chunk_q)
    if split_kv is None:
        split_kv = (max(1, total_cu_count // tile_q_count) + 4) // 5 * 5 * waves_per_eu

    kv_cache_fp8, kv_cache_scale = _prepare_paged_kv_views(
        kv_cache, kv_block_size, hidden_dim, fp8_dtype
    )

    if kv_block_size == DEFAULT_KV_BLOCK_SIZE_PAGED:
        _launch_fp8_paged_mqa_logits_stage1_block(
            q_fp8,
            kv_cache_fp8,
            kv_cache_scale,
            weights,
            out_qk,
            context_lens,
            kv_indices,
            max_model_len,
            max_block_len,
            kv_block_size=kv_block_size,
            chunk_q=chunk_q,
            chunk_k=chunk_k,
            split_kv=split_kv,
            waves_per_eu=waves_per_eu,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out_qk

    _launch_fp8_paged_mqa_logits_stage1(
        q_fp8,
        kv_cache_fp8,
        kv_cache_scale,
        weights,
        out_qk,
        context_lens,
        kv_indices,
        max_model_len,
        max_block_len,
        chunk_q=chunk_q,
        chunk_k=chunk_k,
        split_kv=split_kv,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_qk


def _build_paged_kv_views(
    kv_cache: torch.Tensor,
    kv_block_size: int,
    hidden_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the packed KV cache into a value view and a scale view.

    The cache stores ``[block_size * (D + 4)]`` bytes per page: ``block_size * D``
    bytes of fp8 values followed by ``block_size`` fp32 scales. Both outputs are
    *views* (no copy), built via ``as_strided`` on a byte view so the per-call
    ATen dispatch stays minimal -- this runs on every decode step.
    """
    num_blocks, block_size, _, index_dim = kv_cache.size()
    assert block_size == kv_block_size
    assert index_dim == hidden_dim + 4, (
        f"packed KV row must be D+4 bytes; got index_dim={index_dim}, "
        f"hidden_dim={hidden_dim}."
    )
    assert kv_cache.is_contiguous(), "packed KV cache must be contiguous to view."
    assert kv_cache.dtype in (torch.uint8, fp8_dtype) or kv_cache.element_size() == 1, (
        f"packed KV cache must be byte-addressed (uint8 or float8), got "
        f"{kv_cache.dtype} (element_size={kv_cache.element_size()})."
    )

    # Work in bytes end to end: the scale region is only addressable as raw bytes
    # (4 bytes per fp32), so the page pitch must be expressed in bytes rather than
    # in the input dtype's elements -- otherwise a non-uint8 cache would get the
    # wrong stride.
    kv_bytes = kv_cache.reshape(-1).view(torch.uint8)
    row_bytes = kv_block_size * index_dim

    # Values: the first ``block_size * D`` bytes of each row.
    kv_cache_fp8 = torch.as_strided(
        kv_bytes,
        size=(num_blocks, kv_block_size, hidden_dim),
        stride=(row_bytes, hidden_dim, 1),
    ).view(fp8_dtype)

    # Scales: the trailing ``block_size`` fp32 values of each row. They form a
    # contiguous run, so the in-page stride is 1 float and the page stride is the
    # row pitch expressed in floats.
    kv_cache_scale = torch.as_strided(
        kv_bytes,
        size=(num_blocks, kv_block_size * 4),
        stride=(row_bytes, 1),
        storage_offset=kv_block_size * hidden_dim,
    ).view(torch.float32)

    # Shape contract consumed by the two dispatch paths:
    #   kv_block_size == 1 -> per-token indices; kernels use stride(0) on both
    #                         tensors, so they stay 2D [num_blocks, block_size].
    #   kv_block_size >  1 -> block table; kernels use stride(0) and stride(1),
    #                         so the page dim is kept 3D / 2D.
    if kv_block_size == 1:
        kv_cache_fp8 = kv_cache_fp8.view(num_blocks, hidden_dim)
        kv_cache_scale = kv_cache_scale.view(num_blocks, 1)

    return kv_cache_fp8, kv_cache_scale


# ``_build_paged_kv_views`` is pure in (storage, page size, hidden dim, dtype):
# a production decode loop hands us the *same* persistent cache on every step, for
# every indexer layer, and would otherwise rebuild two identical views each time
# (~14us measured, on a path whose whole host budget is ~115us).
#
# Entries hold the views, and a view keeps its storage alive, so a cached
# ``data_ptr`` can never be recycled for a different tensor -- the address stays
# valid for as long as the entry lives. Shape/stride/dtype are in the key because
# one storage can back several differently-shaped caches.
_VIEW_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_VIEW_CACHE_MAX = 8


def _prepare_paged_kv_views(
    kv_cache: torch.Tensor,
    kv_block_size: int,
    hidden_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached :func:`_build_paged_kv_views`.

    Bounded FIFO: a caller that hands us a fresh cache every step degrades to the
    uncached behaviour rather than growing without limit.
    """
    key = (
        kv_cache.data_ptr(),
        tuple(kv_cache.size()),
        tuple(kv_cache.stride()),
        kv_cache.dtype,
        kv_block_size,
        hidden_dim,
        fp8_dtype,
    )
    hit = _VIEW_CACHE.get(key)
    if hit is not None:
        return hit

    views = _build_paged_kv_views(kv_cache, kv_block_size, hidden_dim, fp8_dtype)
    if len(_VIEW_CACHE) >= _VIEW_CACHE_MAX:
        _VIEW_CACHE.pop(next(iter(_VIEW_CACHE)))
    _VIEW_CACHE[key] = views
    return views


def _launch_fp8_paged_mqa_logits_block(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    max_block_len: int,
    *,
    kv_block_size: int,
    chunk_k: int,
    split_kv: int,
    waves_per_eu: int,
    num_warps: int,
    num_stages: int,
    fuse_output_init: bool = False,
) -> None:
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    grid = (batch_size * next_n * split_kv, 1, 1)
    hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
    # One page per tile lets the page index be a scalar (see the kernel doc:
    # -15.9% at avg_kv=72000, bitwise identical). Only applies when the tile
    # equals the page, i.e. chunk_k == 64 with the paged layout.
    kernel = (_fp8_paged_mqa_logits_block64_kernel
              if chunk_k == kv_block_size
              else _fp8_paged_mqa_logits_block_kernel)
    kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        hidden_dim,
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
        context_lens,
        block_table,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        num_warps=num_warps,
        num_stages=num_stages,
        ChunkQ=heads,
        ChunkK=chunk_k,
        HiddenDim=hidden_dim,
        KVBlockSize=kv_block_size,
        SplitKV=split_kv,
        FillTail=1 if fuse_output_init else 0,
        **hip_kwargs,
    )


def build_paged_bucket_schedule(
    context_lens: torch.Tensor,
    bucket_tokens: int = 2048,
    max_model_len: Optional[int] = None,
    next_n: int = 1,
) -> torch.Tensor:
    """Flat, load-balanced work list for ``_fp8_paged_mqa_logits_bucket_kernel``.

    Returns ``int32[total, 2]`` where each row is ``(row, k_start)``; ``row`` is
    ``batch * next_n + n_idx`` and the bucket owns KV positions
    ``[k_start, k_start + bucket_tokens)`` of that row.

    Each query row emits ``ceil(ctx / bucket_tokens)`` entries, so its program
    count is proportional to its own KV length. That is the whole point: the
    ``SplitKV``-based kernel gives every row the same program count, which
    wastes parallelism on short rows in a mixed batch.

    ``context_lens`` is indexed by batch, so with ``next_n > 1`` it is expanded
    to one entry per query row (all ``n_idx`` of a batch share the same KV
    length). The kernel reads ``row`` as a query-row index, not a batch index.

    Pure host-side tensor arithmetic -- no sync beyond the single ``.item()``
    needed to size the grid.
    """
    if bucket_tokens <= 0:
        raise ValueError(f"bucket_tokens must be positive; got {bucket_tokens}.")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive; got {next_n}.")

    ctx = context_lens.to(torch.int64)
    if max_model_len is not None:
        ctx = torch.clamp(ctx, max=int(max_model_len))
    if next_n > 1:
        ctx = ctx.repeat_interleave(next_n)
    if ctx.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.int32, device=context_lens.device)

    n_buckets = torch.clamp(
        torch.div(ctx + bucket_tokens - 1, bucket_tokens, rounding_mode="floor"),
        min=1,
    )
    total = int(n_buckets.sum().item())
    starts = torch.cumsum(n_buckets, dim=0) - n_buckets
    rows = torch.repeat_interleave(
        torch.arange(ctx.numel(), device=ctx.device, dtype=torch.int64), n_buckets
    )
    # Trick: build the within-row index by subtracting each row's running start.
    arange = torch.arange(total, device=ctx.device, dtype=torch.int64)
    starts_per_entry = torch.repeat_interleave(starts, n_buckets)
    idx = arange - starts_per_entry
    k_start = idx * bucket_tokens
    return torch.stack([rows, k_start], dim=1).to(torch.int32)


def _default_bucket_num_programs(
    batch_size: int,
    next_n: int,
    max_ctx: Optional[int],
    bucket_tokens: int,
) -> int:
    """Grid size for the persistent bucket kernel, resolved on the host.

    The kernel cannot size its own grid and the host cannot read the real bucket
    count without a sync, so this estimates. ``batch * next_n *
    ceil(max_ctx / bucket_tokens)`` is a host-side upper bound on
    ``sum(ceil(ctx_i / bucket_tokens))`` -- what the flat scheduler used as its
    grid, and the sweet spot, since each program then does about one bucket.

    It is a loose bound (it uses the longest row for every row), so it is clamped
    to ``[2, 32]`` CTAs per CU. Below the floor a tiny batch underfills the
    machine; above the ceiling the empty programs start to cost more than the
    balance they buy.
    """
    cu = _default_total_cu_count()
    if max_ctx is None:
        return cu * DEFAULT_BUCKET_WAVES_PER_EU
    estimate = batch_size * next_n * (-(-int(max_ctx) // bucket_tokens))
    return max(
        cu * DEFAULT_BUCKET_MIN_PROGRAMS_PER_CU,
        min(cu * DEFAULT_BUCKET_MAX_PROGRAMS_PER_CU, estimate),
    )


def triton_fp8_paged_mqa_logits_build_bucket_schedule(
    context_lens: torch.Tensor,
    *,
    batch_size: Optional[int] = None,
    next_n: int = 1,
    bucket_tokens: int = DEFAULT_BUCKET_TOKENS,
    num_programs: Optional[int] = None,
    max_ctx: Optional[int] = None,
    waves_per_eu: int = DEFAULT_BUCKET_WAVES_PER_EU,
    base: Optional[torch.Tensor] = None,
) -> PagedBucketSchedule:
    """Build the bucket schedule on the device, with no host sync.

    Drop-in replacement for :func:`build_paged_bucket_schedule`, which pays for
    its schedule on the host: ~10 small tensor ops plus a ``.item()`` to size the
    grid, measured at ~480-540us per call on a 64-row batch. Since the schedule
    only changes when ``context_lens`` changes, that is a per-step cost the
    serving loop should never pay -- but as written it was paid on *every* call,
    which is why ``use_bucket`` looked 3x slower than ``SplitKV`` end to end
    while its kernel is actually ~2x faster on long-tailed batches.

    Here the device computes ``base`` (see ``_paged_bucket_prefix_kernel``) and
    the host never reads it back, so the call is safe inside a graph capture.
    Pass a persistent ``base`` to keep the address stable; the kernel overwrites
    it in place.

    Args:
        context_lens: ``int32[batch_size]`` effective context length per batch
            element, as consumed by ``triton_fp8_paged_mqa_logits``.
        batch_size: rows in ``context_lens``. Defaults to ``context_lens.numel()``.
        next_n: query rows per batch element; buckets are indexed per query row.
        bucket_tokens: KV tokens per bucket. Must match what the compute call
            uses -- it comes from this object, so there is nothing to keep in sync.
        num_programs: grid size for the compute kernel. Defaults to
            ``_default_bucket_num_programs``: one program per estimated bucket,
            clamped to a couple of dozen waves either side.
        max_ctx: ``max(context_lens)``, used only to size the default grid. Pass
            ``PagedBatchInfo.max_ctx`` to get the adaptive default; without it the
            grid falls back to a flat ``cus * waves_per_eu``. Does not sync.
        waves_per_eu: CTAs per CU, used only when ``max_ctx`` is unavailable.
        base: persistent ``int32`` scratch of at least ``num_rows + 1`` elements.
            Allocate once per graph bucket and reuse; a fresh allocation every
            step would defeat graph capture.

    Returns:
        :class:`PagedBucketSchedule` to pass as
        ``triton_fp8_paged_mqa_logits(..., bucket_schedule=...)``.
    """
    if bucket_tokens <= 0:
        raise ValueError(f"bucket_tokens must be positive; got {bucket_tokens}.")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive; got {next_n}.")
    if batch_size is None:
        batch_size = int(context_lens.numel())
    num_rows = batch_size * next_n
    if num_rows < 1:
        raise ValueError(
            f"a bucket schedule needs at least one query row; got "
            f"batch_size={batch_size}, next_n={next_n}."
        )

    expected = num_rows + 1
    if base is None:
        base = torch.empty(expected, dtype=torch.int32, device=context_lens.device)
    else:
        if base.dtype != torch.int32:
            raise ValueError(f"base must be int32; got {base.dtype}.")
        if base.numel() < expected:
            raise ValueError(
                f"base needs at least {expected} elements for batch_size="
                f"{batch_size}, next_n={next_n}; got {base.numel()}."
            )

    if num_programs is None:
        num_programs = _default_bucket_num_programs(
            batch_size, next_n, max_ctx, bucket_tokens
        )
    if num_programs < 1:
        raise ValueError(f"num_programs must be >= 1; got {num_programs}.")

    _paged_bucket_prefix_kernel[(1,)](
        context_lens,
        base,
        num_rows,
        NextN=next_n,
        BucketTokens=bucket_tokens,
        BLOCK_ROWS=triton.next_power_of_2(num_rows),
    )
    return PagedBucketSchedule(
        base=base,
        num_rows=num_rows,
        bucket_tokens=bucket_tokens,
        num_programs=num_programs,
    )


def _launch_fp8_paged_mqa_logits_bucket_persistent(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule: PagedBucketSchedule,
    max_model_len: int,
    max_block_len: int,
    *,
    kv_block_size: int,
    chunk_k: int,
    waves_per_eu: int,
    num_warps: int,
    num_stages: int,
) -> None:
    """Launch the bucket kernel with a host-constant grid."""
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    # ChunkQ == heads so one program owns every head of a row; the kernel has no
    # head-chunk decomposition, matching what the block/bucket kernels do here
    # (num_block_q_head is always 1 because ChunkQ is pinned to heads_num).
    grid = (schedule.num_programs, 1, 1)
    hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
    # One page per tile (chunk_k == 64) makes the page index scalar; see the
    # kernel doc. Bitwise identical, keeps this path in step with the block one.
    kernel = (_fp8_paged_mqa_logits_bucket_persistent64_kernel
              if chunk_k == kv_block_size
              else _fp8_paged_mqa_logits_bucket_persistent_kernel)
    kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_fp8.stride(1),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
        context_lens,
        block_table,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        schedule.base,
        schedule.num_rows,
        num_warps=num_warps,
        num_stages=num_stages,
        ChunkQ=heads,
        ChunkK=chunk_k,
        HiddenDim=hidden_dim,
        KVBlockSize=kv_block_size,
        BucketTokens=schedule.bucket_tokens,
        LOG2_ROWS=max(1, (schedule.num_rows - 1).bit_length()),
        **hip_kwargs,
    )


def _launch_fp8_paged_mqa_logits_bucket(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule: torch.Tensor,
    max_model_len: int,
    max_block_len: int,
    *,
    kv_block_size: int,
    chunk_k: int,
    bucket_tokens: int,
    waves_per_eu: int,
    num_warps: int,
    num_stages: int,
) -> None:
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    grid = (schedule.size(0), 1, 1)
    hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
    _fp8_paged_mqa_logits_bucket_kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_fp8.stride(1),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
        context_lens,
        block_table,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        schedule,
        num_warps=num_warps,
        num_stages=num_stages,
        ChunkQ=heads,
        ChunkK=chunk_k,
        HiddenDim=hidden_dim,
        KVBlockSize=kv_block_size,
        BucketTokens=bucket_tokens,
        **hip_kwargs,
    )


def triton_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    out_logits: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    *,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    preshuffle: bool = False,
    chunk_k: Optional[int] = None,
    total_cu_count: Optional[int] = None,
    split_kv: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    use_bucket: bool = False,
    bucket_tokens: int = 2048,
    bucket_schedule: Optional[PagedBucketSchedule] = None,
    batch_info: Optional[PagedBatchInfo] = None,
    fuse_output_init: bool = False,
) -> torch.Tensor:
    """Compute FP8 paged MQA logits for decode.

    Supports:

    * ``kv_block_size=1``: per-token ``kv_indices`` (aiter JIT path).
    * ``kv_block_size=64``: ``block_table`` logical-block -> physical-page mapping.

    The two page sizes differ in one way that matters for correctness: **who
    writes the ``-inf`` row tail** ``[context_lens, max_model_len)``.

    * ``kv_block_size=1`` writes only ``[0, context_lens)``; the caller must
      pre-fill the whole row with ``-inf``.
    * ``kv_block_size=64`` behaves the same *by default*. Pass
      ``fuse_output_init=True`` and the kernel writes the whole row instead, so
      the caller's ``torch.full`` can be dropped -- but that is opt-in, and only
      the SplitKV block path implements it (bucket / ``kv_block_size=1`` raise).
      See ``fuse_output_init`` below.

    Either way the tail has to end up ``-inf``: consumers (hipc's
    ``top_k_per_row_decode``, or ``torch.topk`` + ``-inf -> -1``) treat a
    non-``-inf`` value as a real index, so a stale tail silently corrupts the
    top-k -- and the op does not verify the pre-fill, it simply leaves the tail
    as it found it. Full matrix and measurements: ``docs/ops/mqa_logits_triton.md``.

    Two independent knobs describe the KV cache, and they are orthogonal:

    * **page size** (``kv_block_size``) — how many tokens one physical page holds,
      i.e. the unit the ``block_table`` is indexed in. ``1`` = per-token indices,
      ``64`` = one page per 64 tokens.
    * **in-page layout** (``preshuffle``) — how the ``[tokens, head_dim]`` rows are
      ordered *inside* a page. Row-major (``preshuffle=False``) vs the 16x16 tiled
      layout the aiter tiled path consumes (``preshuffle=True``).

    ``kv_block_size=64`` does **not** imply preshuffle: a 64-token page can still
    be row-major, which is what BoltOPs implements. BoltOPs only lacks the tiled
    in-page layout; that is a separate axis from ``kv_block_size``.

    Only ``kv_block_size`` 1 and 64 are accepted; any other value raises
    ``AssertionError`` before the KV cache is touched. Upstream tracks the
    allocator's ``page_size``, which is only ever 1 or 64.

    Args:
        q_fp8: ``[batch, next_n, H, D]`` float8 queries for the trailing
            ``next_n`` tokens per sequence.
        kv_cache: ``[num_blocks, block_size, 1, D+4]`` packed FP8 keys + fp32
            scales (uint8 or float8 view).
        weights: ``[batch * next_n, H]`` float32.
        out_logits: ``[batch * next_n, max_model_len]`` float32. Must be prefilled
            with ``-inf`` outside the valid causal window -- unless
            ``fuse_output_init=True``, in which case an uninitialised buffer is
            correct because the kernel writes the row itself.
        context_lens: ``[batch]`` int32, current sequence lengths (including
            this step's queries).
        kv_indices: ``[batch, max_blk_len]`` int32. For ``block_size=1``,
            ``kv_indices[b, t]`` is the physical cache row for logical token ``t``.
            For ``block_size=64``, pass ``block_table`` here:
            ``kv_indices[b, t // 64]`` is the physical page for logical block ``t // 64``.
        max_model_len: width of ``out_logits``.
        kv_block_size: ``1`` (per-token indices) or ``64`` (block table). This is
            the page size and is independent of ``preshuffle``. It also decides
            who writes the ``-inf`` row tail -- see the note above and
            ``fuse_output_init``.
        preshuffle: in-page KV layout flag. Must be ``False`` on BoltOPs — only
            the row-major page layout is implemented (the tiled layout used by the
            aiter tiled path is not). Note this is orthogonal to ``kv_block_size``:
            ``kv_block_size=64`` with ``preshuffle=False`` (row-major 64-token
            pages) is supported.
        chunk_k: KV tile size (default 64).
        total_cu_count: device CU count used to derive ``SplitKV`` when unset.
        split_kv: override KV split factor; ``None`` uses aiter-style heuristic.
            Ignored when ``use_bucket=True`` (the schedule size replaces it).
        use_bucket: use the load-balanced bucket scheduler instead of the
            ``SplitKV`` grid. Each batch row gets ``ceil(ctx / bucket_tokens)``
            programs, so a batch mixing short and long contexts no longer spends
            its parallelism on rows that have almost no KV to walk. Costs one
            host sync per call to size the grid.
        bucket_tokens: KV positions owned by one bucketed program (default 2048;
            must be a multiple of ``chunk_k``).
        batch_info: result of :func:`triton_fp8_paged_mqa_logits_prepare` for
            this batch. Supplies ``max_ctx`` so the SplitKV heuristic stays
            context-aware without a host sync on the launch path. Shape
            asserts still run here. Recommended in any decode loop: prepare
            once per batch (graph-outside), then pass it on every step.
        fuse_output_init: fill the row tail ``[context_lens, max_model_len)``
            with ``-inf`` from inside the kernel, so ``out_logits`` does not need
            to be pre-filled. Default ``False`` -- note that ``kv_block_size=64``
            does **not** imply this, so by default both page sizes expect the
            caller to pre-fill. The tail is most of the row (96% at
            ``max_model_len=111000``, ``avg_kv=4096``), so this is the difference
            between owning the row and paying for it: measured 15-31us saved
            versus a separate ``torch.full``. Only implemented on the SplitKV
            block path (``kv_block_size=64``, row-major, no bucket scheduler);
            every other path raises ``NotImplementedError``
            rather than quietly leaving the tail unwritten.
        waves_per_eu / num_warps / num_stages: Triton launch options.

    Note:
        Numeric ``context_lens`` bounds are the caller's responsibility (same
        as hipc / aiter). The launch path never reads those values back for
        scheduling. Pass ``batch_info`` from
        :func:`triton_fp8_paged_mqa_logits_prepare` to keep context-aware
        SplitKV without a per-step sync; shape asserts still run here.

    Returns:
        ``out_logits`` (same tensor, in-place fill).
    """
    fp8_dtype = get_fp8_e4m3_dtype()
    if q_fp8.dtype != fp8_dtype:
        q_fp8 = q_fp8.to(fp8_dtype)

    if bucket_schedule is not None:
        if use_bucket:
            raise ValueError(
                "bucket_schedule supersedes use_bucket; it is the same "
                "scheduler computed on the device. Pass only bucket_schedule."
            )
        expected_rows = q_fp8.size(0) * q_fp8.size(1)
        if bucket_schedule.num_rows != expected_rows:
            raise ValueError(
                f"bucket_schedule was built for {bucket_schedule.num_rows} query "
                f"rows but this batch has {expected_rows} "
                f"(batch={q_fp8.size(0)} * next_n={q_fp8.size(1)}). Rebuild it "
                f"when the batch shape changes."
            )

    if preshuffle:
        raise NotImplementedError(
            "BoltOPs triton_fp8_paged_mqa_logits implements only the row-major "
            "in-page KV layout; the tiled (preshuffle) layout used by the aiter "
            "tiled path is not implemented. This is independent of kv_block_size: "
            "use kv_block_size=64 with preshuffle=False for row-major 64-token pages."
        )

    # Refuse to silently skip the fill: if a path cannot do it, the caller would
    # get a row with a stale tail and no error, and the consumer reads a stale
    # index as a real one. Only the two block kernels implement it.
    if fuse_output_init and (
        use_bucket
        or bucket_schedule is not None
        or kv_block_size != DEFAULT_KV_BLOCK_SIZE_PAGED
    ):
        raise NotImplementedError(
            "fuse_output_init is implemented only on the SplitKV block path "
            "(kv_block_size=64, row-major, no bucket scheduler); "
            "the bucket and kv_block_size=1 kernels do not fill the row tail."
        )

    chunk_k = DEFAULT_CHUNK_K if chunk_k is None else chunk_k
    waves_per_eu = DEFAULT_WAVE_PER_EU if waves_per_eu is None else waves_per_eu
    num_warps = DEFAULT_NUM_WARPS if num_warps is None else num_warps
    num_stages = DEFAULT_NUM_STAGES if num_stages is None else num_stages

    if total_cu_count is None:
        total_cu_count = _default_total_cu_count()

    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    # Fail fast on unsupported page sizes, before touching the KV views: the
    # dispatch below only implements 1 (per-token) and 64 (block table), and
    # ``_prepare_paged_kv_views`` would happily view any >1 value.
    assert kv_block_size in (DEFAULT_KV_BLOCK_SIZE, DEFAULT_KV_BLOCK_SIZE_PAGED), (
        f"BoltOPs supports KVBlockSize {DEFAULT_KV_BLOCK_SIZE} and "
        f"{DEFAULT_KV_BLOCK_SIZE_PAGED}; got {kv_block_size}."
    )
    assert block_size == kv_block_size
    assert chunk_k % kv_block_size == 0 or kv_block_size % chunk_k == 0
    assert heads & (heads - 1) == 0, "num q. heads should be power of 2."
    assert hidden_dim & (hidden_dim - 1) == 0, "head size should be power of 2."

    # Shape asserts only -- never read ``context_lens`` values here.
    # ``max_ctx`` for SplitKV comes from ``batch_info`` when the caller prepared.
    _check_paged_inputs(
        q_fp8, kv_cache, context_lens, kv_indices, max_model_len,
        out_len_dim=out_logits.size(-1), stage1=False,
    )
    max_ctx = None if batch_info is None else batch_info.max_ctx

    # ChunkQ == heads on this path.
    chunk_k = _resolve_chunk_k(chunk_k, heads)

    tile_q_count = batch_size * next_n
    # Without ``batch_info`` keep the occupancy-only SplitKV rather than syncing.
    split_kv = _resolve_split_kv(
        split_kv,
        tile_q_count=tile_q_count,
        total_cu_count=total_cu_count,
        waves_per_eu=waves_per_eu,
        max_ctx=max_ctx,
        chunk_k=chunk_k,
    )

    kv_cache_fp8, kv_cache_scale = _prepare_paged_kv_views(
        kv_cache, kv_block_size, hidden_dim, fp8_dtype
    )

    if kv_block_size == 1:
        grid = (batch_size * next_n * split_kv, 1, 1)
        hip_kwargs = _filter_hip_launch_kwargs({"waves_per_eu": waves_per_eu})
        _fp8_paged_mqa_logits_kernel[grid](
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            num_warps=num_warps,
            num_stages=num_stages,
            ChunkQ=heads,
            ChunkK=chunk_k,
            SplitKV=split_kv,
            HiddenDim=hidden_dim,
            **hip_kwargs,
        )
        return out_logits

    if kv_block_size == DEFAULT_KV_BLOCK_SIZE_PAGED:
        if bucket_schedule is not None:
            # Preferred bucket path: the schedule already lives on the device, so
            # there is nothing to build and nothing to sync here. The grid is
            # ``bucket_schedule.num_programs``, a host constant, which keeps this
            # branch graph-capturable.
            _launch_fp8_paged_mqa_logits_bucket_persistent(
                q_fp8,
                kv_cache_fp8,
                kv_cache_scale,
                weights,
                out_logits,
                context_lens,
                kv_indices,
                bucket_schedule,
                max_model_len,
                max_block_len,
                kv_block_size=kv_block_size,
                chunk_k=chunk_k,
                waves_per_eu=waves_per_eu,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return out_logits

        if use_bucket:
            # Load-balanced schedule: each row emits ceil(ctx / bucket_tokens)
            # programs, so parallelism tracks actual KV length instead of being
            # spread evenly over rows the way SplitKV does. Building the schedule
            # costs one host sync (the grid size must be known on the host).
            #
            # Superseded by ``bucket_schedule``, which computes the same
            # partition on the device and so pays neither the ~500us of host
            # tensor ops nor the sync. Kept for the eager path and as the
            # reference the device-side version is checked against.
            schedule = build_paged_bucket_schedule(
                context_lens,
                bucket_tokens=bucket_tokens,
                max_model_len=max_model_len,
                next_n=next_n,
            )
            _launch_fp8_paged_mqa_logits_bucket(
                q_fp8,
                kv_cache_fp8,
                kv_cache_scale,
                weights,
                out_logits,
                context_lens,
                kv_indices,
                schedule,
                max_model_len,
                max_block_len,
                kv_block_size=kv_block_size,
                chunk_k=chunk_k,
                bucket_tokens=bucket_tokens,
                waves_per_eu=waves_per_eu,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return out_logits

        _launch_fp8_paged_mqa_logits_block(
            q_fp8,
            kv_cache_fp8,
            kv_cache_scale,
            weights,
            out_logits,
            context_lens,
            kv_indices,
            max_model_len,
            max_block_len,
            kv_block_size=kv_block_size,
            chunk_k=chunk_k,
            split_kv=split_kv,
            waves_per_eu=waves_per_eu,
            num_warps=num_warps,
            num_stages=num_stages,
            fuse_output_init=fuse_output_init,
        )
        return out_logits

    raise ValueError(
        f"Unsupported kv_block_size={kv_block_size}; BoltOPs supports "
        f"{DEFAULT_KV_BLOCK_SIZE} and {DEFAULT_KV_BLOCK_SIZE_PAGED}."
    )
