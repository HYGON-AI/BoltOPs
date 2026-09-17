"""Half-precision (fp16/bf16) MQA logits — Triton per-row path.

Port of FP8 ``_fp8_mqa_logits_kernel`` without ``kv_scales``. Intended for short
KV (e.g. N≈66) where BN tiles waste work relative to lightop / row scan.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.mqa_logits.triton.fp8_mqa_logits import _filter_hip_launch_kwargs

DEFAULT_BLOCK_KV = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2
DEFAULT_WAVES_PER_EU = 0
DEFAULT_CLEAR_BLOCK_KV = 512
DEFAULT_CLEAR_NUM_WARPS = 4
DEFAULT_CLEAR_NUM_STAGES = 1


def _half_row_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    """Lightop-like split for tiny KV; keep divisible into H."""
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


@triton.jit
def _mqa_logits_half_row_kernel(
    Q_ptr,
    KV_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    BLOCK_KV: tl.constexpr,
    LOGITS_MASKING: tl.constexpr,
    ASSUME_KS_ZERO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_CHUNK: tl.constexpr,
):
    """One program per query row (+ optional head split): scan KV in BLOCK_KV."""
    row_id = tl.program_id(0)
    # Larger rows first to reduce tail effect when windows grow with q index.
    row_id = tl.num_programs(0) - row_id - 1
    split_id = tl.program_id(1)
    tl.assume(row_id >= 0)

    start_h = split_id * HEAD_CHUNK
    if start_h >= NUM_HEADS:
        return

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_local = tl.arange(0, HEAD_CHUNK)
    h_inds = (start_h + h_local)[:, None]
    h_mask = (start_h + h_local) < NUM_HEADS
    d_inds = tl.arange(0, HEAD_SIZE)

    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(
        w_ptrs, mask=h_mask[:, None], other=0.0, cache_modifier=".cg"
    ).to(tl.float32)

    if ASSUME_KS_ZERO:
        start_ind = 0
    else:
        start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    if LOGITS_MASKING:
        start_ind = tl.maximum(start_ind, 0)
        end_ind = tl.minimum(end_ind, seq_len_kv)
    else:
        start_ind = tl.maximum(start_ind, 0) // BLOCK_KV * BLOCK_KV
        end_ind = tl.cdiv(tl.minimum(end_ind, seq_len_kv), BLOCK_KV) * BLOCK_KV
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )
    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        if NUM_SPLITS == 1:
            tl.store(logits_ptrs, scores)
        else:
            tl.atomic_add(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    if LOGITS_MASKING:
        kv_col_mask = kv_col_offsets < end_ind
        kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        if ASSUME_KS_ZERO:
            in_window = kv_col_offsets < end_ind
        else:
            in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
        if NUM_SPLITS == 1:
            tl.store(logits_ptrs, scores, mask=in_window)
        else:
            tl.atomic_add(logits_ptrs, scores, mask=in_window)


@triton.jit
def _mqa_logits_half_clear_logits_kernel(
    logits_ptr,
    cu_start_ptr,
    cu_end_ptr,
    seq_len_kv,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    BLOCK_KV: tl.constexpr,
    ASSUME_KS_ZERO: tl.constexpr,
):
    """Set logits outside each row's [cu_start, cu_end) window to -inf."""
    row_id = tl.program_id(0)
    cu_end = tl.load(cu_end_ptr + row_id)
    if not ASSUME_KS_ZERO:
        cu_start = tl.load(cu_start_ptr + row_id)

    kv_offsets = tl.arange(0, BLOCK_KV)
    for _ in tl.range(0, tl.cdiv(seq_len_kv, BLOCK_KV)):
        if ASSUME_KS_ZERO:
            invalid = kv_offsets >= cu_end
        else:
            invalid = (kv_offsets < cu_start) | (kv_offsets >= cu_end)
        in_bounds = kv_offsets < seq_len_kv
        store_mask = invalid & in_bounds
        logits_ptrs = (
            logits_ptr + row_id * stride_logits_s + kv_offsets * stride_logits_k
        )
        tl.store(
            logits_ptrs,
            tl.full([BLOCK_KV], float("-inf"), dtype=tl.float32),
            mask=store_mask,
        )
        kv_offsets += BLOCK_KV


def mqa_logits_half_row(
    Q: torch.Tensor,
    KV: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
    assume_ks_zero: bool = False,
    block_kv: Optional[int] = None,
    head_split_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    clear_block_kv: Optional[int] = None,
    clear_num_warps: Optional[int] = None,
    clear_num_stages: Optional[int] = None,
) -> torch.Tensor:
    """Prefill MQA logits for fp16/bf16 via per-row KV scan (no scales)."""
    if Q.dtype != KV.dtype or Q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"mqa_logits_half_row expects fp16/bf16 Q/KV of same dtype, got "
            f"{Q.dtype}/{KV.dtype}"
        )
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = int(KV.shape[0])
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    assert head_size == 128, "half path currently expects head_dim=128"

    from boltops.mqa_logits.triton.mqa_config_utils import (
        get_mqa_logits_half_row_launch_config,
    )

    cfg = get_mqa_logits_half_row_launch_config(seq_len, seq_len_kv, num_heads)
    if block_kv is None and "block_kv" in cfg:
        block_kv = int(cfg["block_kv"])
    if num_warps is None and "num_warps" in cfg:
        num_warps = int(cfg["num_warps"])
    if num_stages is None and "num_stages" in cfg:
        num_stages = int(cfg["num_stages"])
    if waves_per_eu is None and "waves_per_eu" in cfg:
        waves_per_eu = int(cfg["waves_per_eu"])
    if head_split_k is None and "head_split_k" in cfg:
        head_split_k = int(cfg["head_split_k"])
    if clear_block_kv is None and "clear_block_kv" in cfg:
        clear_block_kv = int(cfg["clear_block_kv"])
    if clear_num_warps is None and "clear_num_warps" in cfg:
        clear_num_warps = int(cfg["clear_num_warps"])
    if clear_num_stages is None and "clear_num_stages" in cfg:
        clear_num_stages = int(cfg["clear_num_stages"])

    block_kv = int(DEFAULT_BLOCK_KV if block_kv is None else block_kv)
    n_warps = int(DEFAULT_NUM_WARPS if num_warps is None else num_warps)
    n_stages = int(DEFAULT_NUM_STAGES if num_stages is None else num_stages)
    waves = int(DEFAULT_WAVES_PER_EU if waves_per_eu is None else waves_per_eu)
    clear_block_kv = int(
        DEFAULT_CLEAR_BLOCK_KV if clear_block_kv is None else clear_block_kv
    )
    clear_n_warps = int(
        DEFAULT_CLEAR_NUM_WARPS if clear_num_warps is None else clear_num_warps
    )
    clear_n_stages = int(
        DEFAULT_CLEAR_NUM_STAGES if clear_num_stages is None else clear_num_stages
    )

    if head_split_k is None:
        num_splits = _half_row_split_k(seq_len, seq_len_kv, num_heads)
    else:
        num_splits = int(head_split_k)
    while num_splits > 1 and num_heads % num_splits != 0:
        num_splits //= 2
    head_chunk = num_heads // num_splits
    use_split_k = num_splits > 1

    logits_masking = seq_len_kv % block_kv != 0

    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv)
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if use_split_k:
            logits.zero_()
        elif logits_masking:
            logits.fill_(float("-inf"))
    elif use_split_k:
        logits = torch.zeros(
            (seq_len, seq_len_kv), dtype=torch.float32, device=Q.device
        )
    elif logits_masking:
        logits = torch.full(
            (seq_len, seq_len_kv),
            float("-inf"),
            dtype=torch.float32,
            device=Q.device,
        )
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv), dtype=torch.float32, device=Q.device
        )

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    hip = _filter_hip_launch_kwargs({"waves_per_eu": waves} if waves else {})
    kernel_logits_masking = True if use_split_k else logits_masking

    _mqa_logits_half_row_kernel[(seq_len, num_splits)](
        Q_ptr=Q,
        KV_ptr=KV,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        num_warps=n_warps,
        num_stages=n_stages,
        LOGITS_MASKING=kernel_logits_masking,
        ASSUME_KS_ZERO=assume_ks_zero,
        NUM_SPLITS=num_splits,
        HEAD_CHUNK=head_chunk,
        **hip,
    )

    if use_split_k or not logits_masking:
        _mqa_logits_half_clear_logits_kernel[(seq_len,)](
            logits_ptr=logits,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            seq_len_kv=seq_len_kv,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=clear_block_kv,
            num_warps=clear_n_warps,
            num_stages=clear_n_stages,
            ASSUME_KS_ZERO=assume_ks_zero,
        )
    return logits


__all__ = ["mqa_logits_half_row", "_half_row_split_k"]
