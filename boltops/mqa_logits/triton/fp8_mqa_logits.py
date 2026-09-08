"""FP8 MQA logits for sparse-attention indexer (prefill / non-paged).

Vendored from hygon aiter ``aiter.ops.triton.attention.fp8_mqa_logits``
(and the matching ``_triton_kernels`` file). Formula matches DeepGEMM /
vLLM ``fp8_mqa_logits``:

    logits[m, n] = sum_h ReLU(Q[m,h] · KV[n] * scale[n]) * W[m,h]
    outside [cu_starts[m], cu_ends[m]) -> -inf
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.mqa_logits.triton.mqa_config_utils import (
    get_bn_tile_dispatch,
    get_fp8_mqa_launch_config,
)
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities

# Code fallbacks when no shipped JSON is found / for explicit overrides.
DEFAULT_BLOCK_KV = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1
DEFAULT_WAVES_PER_EU = 0
DEFAULT_MATRIX_INSTR_NONKDIM = 16
DEFAULT_KPACK = 2
DEFAULT_SCHEDULE_HINT = "none"
DEFAULT_USE_BLOCK_PINGPONG = False
DEFAULT_CLEAR_BLOCK_KV = 512
DEFAULT_CLEAR_NUM_WARPS = 8
DEFAULT_CLEAR_NUM_STAGES = 1
DEFAULT_CLEAR_WAVES_PER_EU = 2
DEFAULT_CLEAR_KPACK = 2
DEFAULT_CLEAR_SCHEDULE_HINT = "none"


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


@triton.jit
def _fp8_mqa_logits_bn_tile_kernel(
    Q_ptr,
    KV_ptr,
    kv_scales_ptr,
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
    # constexpr strides: help AxisInfo contiguity (esp. logits store along N).
    stride_kv_s: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.constexpr,
    stride_logits_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_FP16: tl.constexpr,
    ASSUME_KS_ZERO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_CHUNK: tl.constexpr,
    # Weight group width for register-tile W loads (must divide HEAD_CHUNK, even).
    W_GROUP: tl.constexpr,
):
    """BN tile with Q register prefetch + grouped W tiles (ship opt path)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M
    start_h = split_id * HEAD_CHUNK
    if start_h >= NUM_HEADS:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    m_mask = offs_m < seq_len
    n_mask = offs_n < seq_len_kv

    if ASSUME_KS_ZERO:
        cu_start = tl.zeros([BLOCK_M], dtype=tl.int32)
    else:
        cu_start = tl.load(cu_start_ptr + offs_m, mask=m_mask, other=0)
        cu_start = tl.maximum(cu_start, 0)
    cu_end = tl.load(cu_end_ptr + offs_m, mask=m_mask, other=0)
    cu_end = tl.minimum(cu_end, seq_len_kv)

    # Entire N-tile past / before every row window → only -inf (split 0) or skip.
    tile_past = n_start >= tl.max(cu_end, axis=0)
    tile_before = False
    if not ASSUME_KS_ZERO:
        tile_before = (n_start + BLOCK_N) <= tl.min(cu_start, axis=0)
    if tile_past or tile_before:
        if NUM_SPLITS == 1 or split_id == 0:
            logits_ptrs = (
                logits_ptr
                + offs_m[:, None] * stride_logits_s
                + offs_n[None, :] * stride_logits_k
            )
            tl.store(
                logits_ptrs,
                tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=m_mask[:, None] & n_mask[None, :],
            )
        return

    kv_ptrs = (
        KV_ptr + offs_d[:, None] * stride_kv_d + offs_n[None, :] * stride_kv_s
    )
    kv = tl.load(kv_ptrs, mask=n_mask[None, :], other=0.0)
    scales = tl.load(kv_scales_ptr + offs_n, mask=n_mask, other=0.0)
    if DOT_FP16:
        kv_dot = kv.to(tl.float16)
    else:
        kv_dot = kv

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Optimal BN path: Q register prefetch + W_GROUP tiled weight loads.
    tl.static_assert(HEAD_CHUNK % W_GROUP == 0)
    tl.static_assert(W_GROUP % 2 == 0)
    n_groups: tl.constexpr = HEAD_CHUNK // W_GROUP
    offs_wg = tl.arange(0, W_GROUP)
    for g in tl.static_range(n_groups):
        offs_h_g = start_h + g * W_GROUP + offs_wg
        w_g = tl.load(
            weights_ptr
            + offs_m[:, None] * stride_w_s
            + offs_h_g[None, :] * stride_w_h,
            mask=m_mask[:, None] & (offs_h_g[None, :] < NUM_HEADS),
            other=0.0,
        ).to(tl.float32)
        h0 = start_h + g * W_GROUP
        q_cur = tl.load(
            Q_ptr
            + offs_m[:, None] * stride_q_s
            + h0 * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=m_mask[:, None] & (h0 < NUM_HEADS),
            other=0.0,
        )
        for hi in tl.static_range(W_GROUP):
            h = start_h + g * W_GROUP + hi
            if hi + 1 < W_GROUP:
                q_next = tl.load(
                    Q_ptr
                    + offs_m[:, None] * stride_q_s
                    + (h + 1) * stride_q_h
                    + offs_d[None, :] * stride_q_d,
                    mask=m_mask[:, None] & ((h + 1) < NUM_HEADS),
                    other=0.0,
                )
            w = tl.sum(
                tl.where(tl.arange(0, W_GROUP)[None, :] == hi, w_g, 0.0),
                axis=1,
            )
            if DOT_FP16:
                q_dot = q_cur.to(tl.float16)
            else:
                q_dot = q_cur
            scores = tl.dot(q_dot, kv_dot, input_precision="ieee")
            scores = tl.maximum(scores, 0.0)
            acc += scores * w[:, None]
            if hi + 1 < W_GROUP:
                q_cur = q_next

    acc = acc * scales[None, :]
    if ASSUME_KS_ZERO:
        in_window = offs_n[None, :] < cu_end[:, None]
    else:
        in_window = (offs_n[None, :] >= cu_start[:, None]) & (
            offs_n[None, :] < cu_end[:, None]
        )
    store_ok = m_mask[:, None] & n_mask[None, :]
    logits_ptrs = (
        logits_ptr
        + offs_m[:, None] * stride_logits_s
        + offs_n[None, :] * stride_logits_k
    )
    if NUM_SPLITS == 1:
        out = tl.where(in_window & store_ok, acc, float("-inf"))
        tl.store(logits_ptrs, out, mask=store_ok)
    else:
        tl.atomic_add(logits_ptrs, acc, mask=store_ok & in_window)
        if split_id == 0:
            tl.store(
                logits_ptrs,
                tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=store_ok & (~in_window),
            )


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8 [seq_len, H, D]
    KV_ptr,  # fp8 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
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
    DOT_FP16: tl.constexpr,
    ASSUME_KS_ZERO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_CHUNK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Larger rows first to reduce tail effect when windows grow with q index.
    row_id = tl.num_programs(0) - row_id - 1
    split_id = tl.program_id(1)
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

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
    q_block = tl.load(
        q_ptrs, mask=h_mask[:, None], other=0.0, cache_modifier=".cg"
    )
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(
        w_ptrs, mask=h_mask[:, None], other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    if DOT_FP16:
        q_dot = q_block.to(tl.float16)
    else:
        q_dot = q_block

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
    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets
    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)
        if DOT_FP16:
            kv_dot = kv_block.to(tl.float16)
        else:
            kv_dot = kv_block

        # [HEAD_CHUNK, BLOCK_KV] = [HEAD_CHUNK, D] x [D, BLOCK_KV]
        scores = tl.dot(q_dot, kv_dot, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        scores = scores * kv_scales
        if NUM_SPLITS == 1:
            tl.store(logits_ptrs, scores)
        else:
            tl.atomic_add(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    if LOGITS_MASKING:
        kv_col_mask = kv_col_offsets < end_ind
        kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
        kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)
        if DOT_FP16:
            kv_dot = kv_block.to(tl.float16)
        else:
            kv_dot = kv_block

        scores = tl.dot(q_dot, kv_dot, input_precision="ieee")
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        scores = scores * kv_scales
        if ASSUME_KS_ZERO:
            in_window = kv_col_offsets < end_ind
        else:
            in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
        if NUM_SPLITS == 1:
            tl.store(logits_ptrs, scores, mask=in_window)
        else:
            tl.atomic_add(logits_ptrs, scores, mask=in_window)


@triton.jit
def _fp8_mqa_clear_logits_kernel(
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
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


def fp8_mqa_logits(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    block_kv: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    schedule_hint: Optional[str] = None,
    use_block_pingpong: Optional[bool] = None,
    clear_block_kv: Optional[int] = None,
    clear_num_warps: Optional[int] = None,
    clear_num_stages: Optional[int] = None,
    clear_waves_per_eu: Optional[int] = None,
    clear_kpack: Optional[int] = None,
    clear_schedule_hint: Optional[str] = None,
    assume_ks_zero: bool = False,
    D_out: Optional[torch.Tensor] = None,
    head_split_k: Optional[int] = None,
    bn_tile: Optional[int] = None,
    block_m_tile: Optional[int] = None,
    auto_bn_tile: bool = True,
    bn_w_group: Optional[int] = None,
) -> torch.Tensor:
    """Compute FP8 MQA logits for sparse-attention top-k (prefill / dense KV).

    Args:
        Q: ``[seq_len, NUM_HEADS, HEAD_SIZE]``, float8
        KV: ``[seq_len_kv, HEAD_SIZE]``, float8
        kv_scales: ``[seq_len_kv]``, float32 per-token dequant scales
        weights: ``[seq_len, NUM_HEADS]``, float32
        cu_starts: ``[seq_len]``, int32, inclusive KV start per query row
        cu_ends: ``[seq_len]``, int32, exclusive KV end per query row
        block_kv / num_warps / num_stages / HIP opts:
            main kernel launch config. ``None`` loads shipped tuned config
            under ``configs/triton_<capability>/fp8_mqa_logits,arch=*.json``,
            or BN-tile dispatch when ``auto_bn_tile`` selects that path.
        clear_*: clear-kernel launch config used on the fast path only
            (``seq_len_kv % block_kv == 0``).
        D_out: optional preallocated ``[seq_len, seq_len_kv]`` float32 buffer
            When set, written in-place; otherwise allocated here.
        head_split_k: split heads across grid dim.
            ``None``/``1`` keeps a single pass; ``>1`` uses ``atomic_add``.
        bn_tile: if set (e.g. ``64``/``128``), use 2D tile grid
            ``(ceil(N/bn_tile), ceil(M/block_m_tile), head_split_k)`` instead of
            the row-walk kernel.
        block_m_tile: M tile for ``bn_tile`` path (default from dispatch / 16).
        auto_bn_tile: when ``bn_tile`` is unset, pick BN geometry by shape
            (replaces the old dedicated Triton tiled kernel). Set ``False``
            to force the classic row-walk path.
        bn_w_group: BN weight-tile group width (even, divides ``HEAD_CHUNK``).
            ``None`` picks 8 (clamped to divide ``HEAD_CHUNK``).

    Returns:
        ``logits`` of shape ``[seq_len, seq_len_kv]``, float32.
        Positions outside ``[cu_starts, cu_ends)`` are ``-inf``.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]

    disp = None
    if bn_tile is None and auto_bn_tile:
        disp = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
        bn_tile = int(disp["bn_tile"])
        if block_m_tile is None:
            block_m_tile = int(disp["block_m_tile"])
        if head_split_k is None:
            head_split_k = int(disp["head_split_k"])

    cfg = get_fp8_mqa_launch_config(seq_len_kv)

    def _pick(name: str, override, default):
        if override is not None:
            return override
        if disp is not None and name in disp:
            return disp[name]
        if name in cfg:
            return cfg[name]
        return default

    block_kv = _pick("block_kv", block_kv, DEFAULT_BLOCK_KV)
    num_warps = _pick("num_warps", num_warps, DEFAULT_NUM_WARPS)
    num_stages = _pick("num_stages", num_stages, DEFAULT_NUM_STAGES)
    waves_per_eu = _pick("waves_per_eu", waves_per_eu, DEFAULT_WAVES_PER_EU)
    matrix_instr_nonkdim = _pick(
        "matrix_instr_nonkdim", matrix_instr_nonkdim, DEFAULT_MATRIX_INSTR_NONKDIM
    )
    kpack = _pick("kpack", kpack, DEFAULT_KPACK)
    schedule_hint = _pick("schedule_hint", schedule_hint, DEFAULT_SCHEDULE_HINT)
    use_block_pingpong = _pick(
        "use_block_pingpong", use_block_pingpong, DEFAULT_USE_BLOCK_PINGPONG
    )
    clear_block_kv = _pick("clear_block_kv", clear_block_kv, DEFAULT_CLEAR_BLOCK_KV)
    clear_num_warps = _pick("clear_num_warps", clear_num_warps, DEFAULT_CLEAR_NUM_WARPS)
    clear_num_stages = _pick(
        "clear_num_stages", clear_num_stages, DEFAULT_CLEAR_NUM_STAGES
    )
    clear_waves_per_eu = _pick(
        "clear_waves_per_eu", clear_waves_per_eu, DEFAULT_CLEAR_WAVES_PER_EU
    )
    clear_kpack = _pick("clear_kpack", clear_kpack, DEFAULT_CLEAR_KPACK)
    clear_schedule_hint = _pick(
        "clear_schedule_hint", clear_schedule_hint, DEFAULT_CLEAR_SCHEDULE_HINT
    )
    # gfx936 and older lack native fp8 MMAC; emulate f16 dot like Triton 3.6.
    dot_fp16 = not arch_info.is_fp8_avail()
    # Host must recompute masking whenever BLOCK_KV changes.
    logits_masking = seq_len_kv % block_kv != 0
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    num_splits = 1 if head_split_k is None else int(head_split_k)
    assert num_splits >= 1, f"head_split_k must be >= 1, got {num_splits}"
    while num_splits > 1 and num_heads % num_splits != 0:
        num_splits //= 2
    head_chunk = num_heads // num_splits
    use_split_k = num_splits > 1
    use_bn_tile = bn_tile is not None and int(bn_tile) > 0
    tile_bn = int(bn_tile) if use_bn_tile else 0
    tile_bm = int(block_m_tile) if block_m_tile is not None else 16
    if use_bn_tile:
        assert tile_bm in (16, 32), f"block_m_tile must be 16 or 32, got {tile_bm}"
        assert tile_bn in (64, 128), f"bn_tile must be 64 or 128, got {tile_bn}"
        # Q prefetch wants stages>=2 for compiler LDS pipeline around early loads.
        num_stages = max(int(num_stages), 2)
        preferred = 8 if bn_w_group is None else int(bn_w_group)
        w_grp = preferred
        if w_grp < 2 or w_grp % 2 != 0 or head_chunk % w_grp != 0:
            w_grp = 2
            for g in (16, 8, 4, 2):
                if head_chunk % g == 0:
                    w_grp = g
                    break
        assert head_chunk % w_grp == 0 and w_grp % 2 == 0 and w_grp >= 2
    else:
        w_grp = 8
    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv), (
            f"D_out shape {tuple(D_out.shape)} != {(seq_len, seq_len_kv)}"
        )
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if use_split_k:
            # atomic_add accumulates into in-window cells; start from 0.
            logits.zero_()
        elif logits_masking and not use_bn_tile:
            # Masked row path only stores in-window; match fresh ``full(-inf)`` alloc.
            logits.fill_(-float("inf"))
    elif use_split_k:
        logits = torch.zeros(
            (seq_len, seq_len_kv),
            dtype=torch.float32,
            device=Q.device,
        )
    elif use_bn_tile:
        # Full N-grid writes every column (score or -inf).
        logits = torch.empty(
            (seq_len, seq_len_kv),
            dtype=torch.float32,
            device=Q.device,
        )
    elif logits_masking:
        logits = torch.full(
            (seq_len, seq_len_kv),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv),
            dtype=torch.float32,
            device=Q.device,
        )

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    main_hip = _filter_hip_launch_kwargs(
        {
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "kpack": kpack,
            "schedule_hint": schedule_hint,
            "use_block_pingpong": use_block_pingpong,
        }
    )

    if use_bn_tile:
        assert stride_kv_d == 1, f"BN tile expects contiguous KV dim, stride_d={stride_kv_d}"
        assert stride_logits_k == 1, (
            f"BN tile expects contiguous logits N, stride_k={stride_logits_k}"
        )
        grid_n = triton.cdiv(seq_len_kv, tile_bn)
        grid_m = triton.cdiv(seq_len, tile_bm)
        _fp8_mqa_logits_bn_tile_kernel[(grid_n, grid_m, num_splits)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
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
            BLOCK_M=tile_bm,
            BLOCK_N=tile_bn,
            num_warps=num_warps,
            num_stages=num_stages,
            DOT_FP16=dot_fp16,
            ASSUME_KS_ZERO=assume_ks_zero,
            NUM_SPLITS=num_splits,
            HEAD_CHUNK=head_chunk,
            W_GROUP=w_grp,
            **main_hip,
        )
        # split_k>1: split0 writes OOB -inf inside each tile; still safe to skip clear.
        return logits

    # Split-k path only writes in-window via atomic_add; always mask the KV tail.
    kernel_logits_masking = True if use_split_k else logits_masking
    _fp8_mqa_logits_kernel[(seq_len, num_splits)](
        Q_ptr=Q,
        KV_ptr=KV,
        kv_scales_ptr=kv_scales,
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
        num_warps=num_warps,
        num_stages=num_stages,
        LOGITS_MASKING=kernel_logits_masking,
        DOT_FP16=dot_fp16,
        ASSUME_KS_ZERO=assume_ks_zero,
        NUM_SPLITS=num_splits,
        HEAD_CHUNK=head_chunk,
        **main_hip,
    )

    # Non-split fast path: clear only when BLOCK_KV divides N.
    # Split-k: buffer starts at 0, so always paint out-of-window -inf.
    if use_split_k or not logits_masking:
        clear_hip = _filter_hip_launch_kwargs(
            {
                "waves_per_eu": clear_waves_per_eu,
                "kpack": clear_kpack,
                "schedule_hint": clear_schedule_hint,
            }
        )
        _fp8_mqa_clear_logits_kernel[(seq_len,)](
            logits_ptr=logits,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            seq_len_kv=seq_len_kv,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=clear_block_kv,
            num_warps=clear_num_warps,
            num_stages=clear_num_stages,
            ASSUME_KS_ZERO=assume_ks_zero,
            **clear_hip,
        )

    return logits
