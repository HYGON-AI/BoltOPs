"""gfx946 WASP kernels for MQA logits (float8 + half).

* **Float8** — head-serial N-loop (``fp8_mqa_logits_head_serial``): prologue MLS /
  ``force_grouped`` / ``chunk8_mls`` / ``Q_GROUP``; masked BN for edges.
* **Half** — BN-WASP (``half_mqa_logits_bn_wasp``): one CTA owns all heads; default
  ``Q_GROUP=1`` + MLS Q; window on store.

Shared BN tile helpers and HIP launch filtering live in this module.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from boltops.mqa_logits.triton.fp8_mqa_logits import _filter_hip_launch_kwargs


# ---------------------------------------------------------------------------
# Float8 (head-serial)
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_bn_masked_kernel(
    Q_ptr,
    KV_ptr,
    W_ptr,
    Scales_ptr,
    Logits_ptr,
    cu_start_ptr,
    cu_end_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv_s,
    stride_kv_d,
    stride_w_s,
    stride_w_h,
    stride_logits_s,
    stride_logits_k,
    seq_len,
    seq_len_kv,
    pid_m_offset,
    pid_n_offset,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Masked BN edge / full-grid fallback; no WASP."""
    pid_m = tl.program_id(0) + pid_m_offset
    pid_n = tl.program_id(1) + pid_n_offset

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_h = tl.arange(0, NUM_HEADS)
    m_mask = offs_m < seq_len
    n_mask = offs_n < seq_len_kv

    cu_start = tl.load(cu_start_ptr + offs_m, mask=m_mask, other=0)
    cu_start = tl.maximum(cu_start, 0)
    cu_end = tl.load(cu_end_ptr + offs_m, mask=m_mask, other=0)
    cu_end = tl.minimum(cu_end, seq_len_kv)

    logits_ptrs = (
        Logits_ptr
        + offs_m[:, None] * stride_logits_s
        + offs_n[None, :] * stride_logits_k
    )
    store_ok = m_mask[:, None] & n_mask[None, :]
    tile_past = n_start >= tl.max(cu_end, axis=0)
    tile_before = (n_start + BLOCK_N) <= tl.min(cu_start, axis=0)
    if tile_past or tile_before:
        tl.store(
            logits_ptrs,
            tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
            mask=store_ok,
        )
        return

    kv = tl.load(
        KV_ptr + offs_d[:, None] * stride_kv_d + offs_n[None, :] * stride_kv_s,
        mask=n_mask[None, :],
        other=0.0,
    )
    scales = tl.load(Scales_ptr + offs_n, mask=n_mask, other=0.0)
    w_all = tl.load(
        W_ptr + offs_m[:, None] * stride_w_s + offs_h[None, :] * stride_w_h,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    for hi in tl.range(0, NUM_HEADS):
        q = tl.load(q_ptrs + hi * stride_q_h, mask=m_mask[:, None], other=0.0)
        w_idx = tl.broadcast_to(hi + tl.arange(0, 1)[None, :], (BLOCK_M, 1))
        w = tl.reshape(tl.gather(w_all, w_idx, axis=1), (BLOCK_M,))
        scores = tl.dot(q, kv)
        acc += tl.maximum(scores, 0.0) * w[:, None]
    acc = acc * scales[None, :]

    in_window = (offs_n[None, :] >= cu_start[:, None]) & (
        offs_n[None, :] < cu_end[:, None]
    )
    out = tl.where(in_window & store_ok, acc, float("-inf"))
    tl.store(logits_ptrs, out, mask=store_ok)


@triton.jit
def _fp8_nloop_head_q(
    Q_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    offs_m,
    offs_d,
    mls_m0,
    hi,
    seq_len,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_MLS_Q: tl.constexpr,
):
    """One-head ``[BM, D]`` Q: MLS (mloop-style) or plain load."""
    if USE_MLS_Q:
        return tl.matrix_load(
            Q_ptr + hi * stride_q_h,
            shape=[seq_len, HEAD_SIZE],
            strides=[stride_q_s, stride_q_d],
            block_shape=[BLOCK_M, HEAD_SIZE],
            offsets=[mls_m0, 0],
        )
    return tl.load(
        Q_ptr
        + offs_m[:, None] * stride_q_s
        + hi * stride_q_h
        + offs_d[None, :] * stride_q_d
    )


@triton.jit
def _fp8_nloop_head_w(w_all, hi, offs_g, BLOCK_M: tl.constexpr):
    w_idx = tl.broadcast_to(hi + offs_g[None, :], (BLOCK_M, 1))
    return tl.reshape(tl.gather(w_all, w_idx, axis=1), (BLOCK_M,))


@triton.jit
def _fp8_nloop_head_serial_kernel(
    Q_ptr,
    KV_ptr,
    W_ptr,
    Scales_ptr,
    Logits_ptr,
    cu_start_ptr,
    cu_end_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv_s,
    stride_kv_d,
    stride_w_s,
    stride_w_h,
    stride_logits_s,
    stride_logits_k,
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
    Q_GROUP: tl.constexpr,
    USE_MLS_Q: tl.constexpr,
    CHUNK8_MLS: tl.constexpr,
    FORCE_GROUPED: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """N-loop head-serial: per-head ``[BM,D]`` Q (MLS/load), stream K along N.

    Avoids fat ``[H*BM,D]`` pack so ``matrix_load → dot`` stays mloop-legal.

    * ``NUM_HEADS==8`` (and not ``FORCE_GROUPED``): Q prologue + WASP on N.
    * ``CHUNK8_MLS``: K once per N tile; loop ``H/8`` chunks, each chunk
      unrolls 8× MLS Q like the H=8 path (regs hold one chunk only).
    * else / ``FORCE_GROUPED``: each N tile loads K once, then heads or
      ``Q_GROUP`` packs. Head-WASP only when ``N_TILES==1``.
    """
    pid_m = tl.program_id(0)
    pid_n_blk = tl.program_id(1)
    ACC_N: tl.constexpr = N_TILES * BLOCK_N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_h = tl.arange(0, NUM_HEADS)
    offs_g = tl.arange(0, 1)
    offs_qg = tl.arange(0, Q_GROUP)
    n_base = pid_n_blk * ACC_N
    offs_n_all = n_base + tl.arange(0, ACC_N)
    mls_m0 = pid_m * BLOCK_M
    # Nested head-WASP under N_TILES>1 fails HCU warp-specialize lowering
    # for both MLS and grouped packs; only single-N-tile is safe.
    HEAD_WASP: tl.constexpr = WARP_SPECIALIZE and (N_TILES == 1)
    n_groups: tl.constexpr = NUM_HEADS // Q_GROUP
    n_chunks8: tl.constexpr = NUM_HEADS // 8

    w_all = tl.load(
        W_ptr + offs_m[:, None] * stride_w_s + offs_h[None, :] * stride_w_h
    ).to(tl.float32)
    scales_all = tl.load(Scales_ptr + offs_n_all)
    cu_start = tl.maximum(tl.load(cu_start_ptr + offs_m), 0)
    cu_end = tl.minimum(tl.load(cu_end_ptr + offs_m), seq_len_kv)

    if NUM_HEADS == 8 and not FORCE_GROUPED:
        # Fast path: hold all Q in regs; WASP streams K.
        q0 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 0,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q1 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 1,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q2 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 2,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q3 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 3,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q4 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 4,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q5 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 5,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q6 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 6,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        q7 = _fp8_nloop_head_q(
            Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d, mls_m0, 7,
            seq_len, HEAD_SIZE, BLOCK_M, USE_MLS_Q,
        )
        w0 = _fp8_nloop_head_w(w_all, 0, offs_g, BLOCK_M)
        w1 = _fp8_nloop_head_w(w_all, 1, offs_g, BLOCK_M)
        w2 = _fp8_nloop_head_w(w_all, 2, offs_g, BLOCK_M)
        w3 = _fp8_nloop_head_w(w_all, 3, offs_g, BLOCK_M)
        w4 = _fp8_nloop_head_w(w_all, 4, offs_g, BLOCK_M)
        w5 = _fp8_nloop_head_w(w_all, 5, offs_g, BLOCK_M)
        w6 = _fp8_nloop_head_w(w_all, 6, offs_g, BLOCK_M)
        w7 = _fp8_nloop_head_w(w_all, 7, offs_g, BLOCK_M)

        for t in tl.range(0, N_TILES, warp_specialize=WARP_SPECIALIZE):
            local0 = t * BLOCK_N
            n0 = n_base + local0
            offs_n = n0 + tl.arange(0, BLOCK_N)
            kv = tl.load(
                KV_ptr
                + offs_d[:, None] * stride_kv_d
                + offs_n[None, :] * stride_kv_s
            )
            acc = tl.maximum(tl.dot(q0, kv), 0.0) * w0[:, None]
            acc += tl.maximum(tl.dot(q1, kv), 0.0) * w1[:, None]
            acc += tl.maximum(tl.dot(q2, kv), 0.0) * w2[:, None]
            acc += tl.maximum(tl.dot(q3, kv), 0.0) * w3[:, None]
            acc += tl.maximum(tl.dot(q4, kv), 0.0) * w4[:, None]
            acc += tl.maximum(tl.dot(q5, kv), 0.0) * w5[:, None]
            acc += tl.maximum(tl.dot(q6, kv), 0.0) * w6[:, None]
            acc += tl.maximum(tl.dot(q7, kv), 0.0) * w7[:, None]
            sc = tl.gather(scales_all, local0 + tl.arange(0, BLOCK_N), axis=0)
            acc = acc * sc[None, :]
            in_window = (offs_n[None, :] >= cu_start[:, None]) & (
                offs_n[None, :] < cu_end[:, None]
            )
            out = tl.where(in_window, acc, float("-inf"))
            tl.store(
                Logits_ptr
                + offs_m[:, None] * stride_logits_s
                + offs_n[None, :] * stride_logits_k,
                out,
            )
    elif CHUNK8_MLS:
        # H=64 etc.: K once per N tile; 8-head MLS chunks (H=8 unrolled × H/8).
        # No head-WASP: 8× MLS per step needs >16 abarrier IDs.
        for t in tl.range(0, N_TILES):
            local0 = t * BLOCK_N
            n0 = n_base + local0
            offs_n = n0 + tl.arange(0, BLOCK_N)
            kv = tl.load(
                KV_ptr
                + offs_d[:, None] * stride_kv_d
                + offs_n[None, :] * stride_kv_s
            )
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for c in tl.range(0, n_chunks8):
                h0 = c * 8
                q0 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 0, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q1 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 1, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q2 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 2, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q3 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 3, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q4 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 4, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q5 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 5, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q6 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 6, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                q7 = _fp8_nloop_head_q(
                    Q_ptr, stride_q_s, stride_q_h, stride_q_d, offs_m, offs_d,
                    mls_m0, h0 + 7, seq_len, HEAD_SIZE, BLOCK_M, True,
                )
                w0 = _fp8_nloop_head_w(w_all, h0 + 0, offs_g, BLOCK_M)
                w1 = _fp8_nloop_head_w(w_all, h0 + 1, offs_g, BLOCK_M)
                w2 = _fp8_nloop_head_w(w_all, h0 + 2, offs_g, BLOCK_M)
                w3 = _fp8_nloop_head_w(w_all, h0 + 3, offs_g, BLOCK_M)
                w4 = _fp8_nloop_head_w(w_all, h0 + 4, offs_g, BLOCK_M)
                w5 = _fp8_nloop_head_w(w_all, h0 + 5, offs_g, BLOCK_M)
                w6 = _fp8_nloop_head_w(w_all, h0 + 6, offs_g, BLOCK_M)
                w7 = _fp8_nloop_head_w(w_all, h0 + 7, offs_g, BLOCK_M)
                acc += tl.maximum(tl.dot(q0, kv), 0.0) * w0[:, None]
                acc += tl.maximum(tl.dot(q1, kv), 0.0) * w1[:, None]
                acc += tl.maximum(tl.dot(q2, kv), 0.0) * w2[:, None]
                acc += tl.maximum(tl.dot(q3, kv), 0.0) * w3[:, None]
                acc += tl.maximum(tl.dot(q4, kv), 0.0) * w4[:, None]
                acc += tl.maximum(tl.dot(q5, kv), 0.0) * w5[:, None]
                acc += tl.maximum(tl.dot(q6, kv), 0.0) * w6[:, None]
                acc += tl.maximum(tl.dot(q7, kv), 0.0) * w7[:, None]
            sc = tl.gather(scales_all, local0 + tl.arange(0, BLOCK_N), axis=0)
            acc = acc * sc[None, :]
            in_window = (offs_n[None, :] >= cu_start[:, None]) & (
                offs_n[None, :] < cu_end[:, None]
            )
            out = tl.where(in_window, acc, float("-inf"))
            tl.store(
                Logits_ptr
                + offs_m[:, None] * stride_logits_s
                + offs_n[None, :] * stride_logits_k,
                out,
            )
    else:
        # General H: stream K along N; reload Q per head or Q_GROUP pack.
        # HEAD_WASP only when N_TILES==1 (matches BN-WASP producer shape).
        offs_r = tl.arange(0, Q_GROUP * BLOCK_M)
        local_m = offs_r % BLOCK_M
        h_in_g = offs_r // BLOCK_M
        q_m_base = pid_m * BLOCK_M + local_m
        for t in tl.range(0, N_TILES):
            local0 = t * BLOCK_N
            n0 = n_base + local0
            offs_n = n0 + tl.arange(0, BLOCK_N)
            kv = tl.load(
                KV_ptr
                + offs_d[:, None] * stride_kv_d
                + offs_n[None, :] * stride_kv_s
            )
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if Q_GROUP == 1:
                for hi in tl.range(0, NUM_HEADS, warp_specialize=HEAD_WASP):
                    q = _fp8_nloop_head_q(
                        Q_ptr,
                        stride_q_s,
                        stride_q_h,
                        stride_q_d,
                        offs_m,
                        offs_d,
                        mls_m0,
                        hi,
                        seq_len,
                        HEAD_SIZE,
                        BLOCK_M,
                        USE_MLS_Q,
                    )
                    w = _fp8_nloop_head_w(w_all, hi, offs_g, BLOCK_M)
                    acc += tl.maximum(tl.dot(q, kv), 0.0) * w[:, None]
            else:
                # BN-WASP-style grouped pack (tl.load; MLS stays Q_GROUP==1).
                for g in tl.range(0, n_groups, warp_specialize=HEAD_WASP):
                    local_h = g * Q_GROUP + h_in_g
                    q_mat = tl.load(
                        Q_ptr
                        + q_m_base[:, None] * stride_q_s
                        + local_h[:, None] * stride_q_h
                        + offs_d[None, :] * stride_q_d
                    )
                    scores = tl.maximum(tl.dot(q_mat, kv), 0.0)
                    scores = tl.reshape(scores, (Q_GROUP, BLOCK_M, BLOCK_N))
                    w_idx = tl.broadcast_to(
                        g * Q_GROUP + offs_qg[None, :], (BLOCK_M, Q_GROUP)
                    )
                    w_group = tl.gather(w_all, w_idx, axis=1)
                    w_t = tl.trans(w_group)
                    acc += tl.sum(scores * w_t[:, :, None], axis=0)
            sc = tl.gather(scales_all, local0 + tl.arange(0, BLOCK_N), axis=0)
            acc = acc * sc[None, :]
            in_window = (offs_n[None, :] >= cu_start[:, None]) & (
                offs_n[None, :] < cu_end[:, None]
            )
            out = tl.where(in_window, acc, float("-inf"))
            tl.store(
                Logits_ptr
                + offs_m[:, None] * stride_logits_s
                + offs_n[None, :] * stride_logits_k,
                out,
            )


def _fp8_mqa_logits_nloop_head_serial(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    w: torch.Tensor,
    cu_starts: Optional[torch.Tensor] = None,
    cu_ends: Optional[torch.Tensor] = None,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    n_tiles: Optional[int] = None,
    warp_specialize: bool = True,
    num_stages: int = 2,
    num_warps: int = 4,
    wasp_partition_warps: Sequence[int] = (4, 4),
    wasp_wdra: Optional[bool] = None,
    use_mls_q: bool = True,
    q_group: Optional[int] = None,
    chunk8_mls: bool = False,
    force_grouped: bool = False,
    waves_per_eu: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """N-loop with head-serial Q: per-head ``[BM,D]`` MLS/load, stream K.

    Unlike packed nloop, Q never becomes ``[H*BM,D]``, so MLS keeps the
    mloop-legal ``matrix_load → dot`` path. ``NUM_HEADS==8`` uses a Q
    prologue + WASP on N; other head counts use ``Q_GROUP`` packs (BN-WASP
    style) and head-WASP only when ``N_TILES==1`` (small KV).

    ``chunk8_mls``: for ``H>8`` and ``H%8==0``, K once per N tile then
    ``H/8`` chunks of 8× MLS Q (H=8 unrolled). Opt-in vs default grouped load.
    ``force_grouped``: even for H=8, skip prologue and use ``Q_GROUP`` packs
    (tuning / short-KV).
    """
    seq, heads, dim = q.shape
    seq_kv, dim_kv = kv.shape
    if dim != dim_kv or w.shape != (seq, heads):
        raise ValueError("q, kv, w shapes do not match")
    if kv_scales.shape != (seq_kv,):
        raise ValueError("kv_scales must be [seq_len_kv]")
    if heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {heads}")
    if use_mls_q and int(q.stride(2)) != 1:
        raise ValueError(
            "use_mls_q requires contiguous HEAD_SIZE (stride_d==1), "
            f"got stride_d={q.stride(2)}"
        )

    rec = recommend_fp8_head_serial_launch(seq, seq_kv, heads)
    user_block_m = block_m is not None
    block_m = int(rec["block_m"] if block_m is None else block_m)
    block_n = int(rec["block_n"] if block_n is None else block_n)
    # Match BN-WASP: H>=32 + M>=32 prefers BM=32 (then Q_GROUP=4).
    if not user_block_m and heads >= 32 and seq >= 32 and block_m == 16:
        block_m = 32
    n_tiles = int(rec["n_tiles"] if n_tiles is None else n_tiles)
    if n_tiles not in (1, 2, 4, 8):
        raise ValueError(f"n_tiles must be one of 1/2/4/8, got {n_tiles}")
    force_grouped_v = bool(force_grouped)
    # H=8 prologue ignores Q_GROUP unless force_grouped (BN-WASP-like packs).
    if heads == 8 and not force_grouped_v:
        q_group_v = 1
    elif q_group is None:
        if heads >= 32 and block_m >= 32 and heads % 4 == 0:
            q_group_v = 4
        elif heads % 8 == 0:
            q_group_v = 8
        elif heads % 4 == 0:
            q_group_v = 4
        elif heads % 2 == 0:
            q_group_v = 2
        else:
            q_group_v = 1
    else:
        q_group_v = int(q_group)
        if q_group_v < 1 or heads % q_group_v != 0:
            raise ValueError(
                f"q_group={q_group_v} must divide heads={heads}"
            )
    if force_grouped_v and heads == 8 and q_group is None:
        q_group_v = 8
    # MLS: Q_GROUP==1 head loop, or opt-in 8-head chunk path for H>8.
    chunk8_mls_v = bool(chunk8_mls)
    if chunk8_mls_v:
        if heads <= 8 or heads % 8 != 0:
            raise ValueError(
                "chunk8_mls requires num_heads>8 and num_heads%8==0, "
                f"got {heads}"
            )
        if int(q.stride(2)) != 1:
            raise ValueError(
                "chunk8_mls requires contiguous HEAD_SIZE (stride_d==1), "
                f"got stride_d={q.stride(2)}"
            )
        q_group_v = 8
    use_mls_q_v = (bool(use_mls_q) and q_group_v == 1) or chunk8_mls_v

    if cu_starts is None:
        cu_starts = torch.zeros(seq, device=q.device, dtype=torch.int32)
    if cu_ends is None:
        cu_ends = torch.full((seq,), seq_kv, device=q.device, dtype=torch.int32)

    if out is None:
        logits = torch.empty(seq, seq_kv, device=q.device, dtype=torch.float32)
    else:
        if out.shape != (seq, seq_kv) or out.dtype != torch.float32:
            raise ValueError("out must be float32 [M, N]")
        logits = out

    waves = int(rec["waves_per_eu"] if waves_per_eu is None else waves_per_eu)
    hip_kwargs = {
        "assume_no_nans": int(rec["assume_no_nans"]),
        "misched_regpressure": bool(rec["misched_regpressure"]),
        "enable_v_mmac_cluster": int(rec["enable_v_mmac_cluster"]),
        "matrix_instr_nonkdim": int(rec["matrix_instr_nonkdim"]),
        "kpack": int(rec["kpack"]),
    }
    if waves:
        hip_kwargs["waves_per_eu"] = waves
    hip = _filter_hip_launch_kwargs(hip_kwargs)

    grid_m = (seq + block_m - 1) // block_m
    floor_m = seq // block_m
    n_tiles_total = seq_kv // block_n
    floor_n_blk = n_tiles_total // n_tiles
    covered_n = floor_n_blk * n_tiles

    def _launch_masked(gm, gn, m0, n0, stages):
        if gm <= 0 or gn <= 0:
            return
        _fp8_bn_masked_kernel[(gm, gn)](
            q,
            kv,
            w,
            kv_scales,
            logits,
            cu_starts,
            cu_ends,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            w.stride(0),
            w.stride(1),
            logits.stride(0),
            logits.stride(1),
            seq,
            seq_kv,
            m0,
            n0,
            NUM_HEADS=heads,
            HEAD_SIZE=dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=stages,
            **hip,
        )

    if floor_m > 0 and floor_n_blk > 0:
        # Small-KV / high-H: WDRA helps grouped head-WASP (same as BN-WASP).
        wdra_v = (
            bool(rec["wasp_wdra"])
            if wasp_wdra is None
            else bool(wasp_wdra)
        )
        if wasp_wdra is None and heads > 8 and n_tiles == 1:
            wdra_v = True
        common = dict(
            NUM_HEADS=heads,
            HEAD_SIZE=dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            N_TILES=n_tiles,
            Q_GROUP=q_group_v,
            USE_MLS_Q=use_mls_q_v,
            CHUNK8_MLS=chunk8_mls_v,
            FORCE_GROUPED=force_grouped_v,
            WARP_SPECIALIZE=bool(warp_specialize),
            num_warps=num_warps,
            num_stages=num_stages,
            **hip,
        )
        if warp_specialize:
            common["wasp_partition_warps"] = tuple(
                int(x) for x in wasp_partition_warps
            )
            common["wasp_wdra"] = wdra_v
        args = (
            q,
            kv,
            w,
            kv_scales,
            logits,
            cu_starts,
            cu_ends,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            w.stride(0),
            w.stride(1),
            logits.stride(0),
            logits.stride(1),
            seq,
            seq_kv,
        )
        _fp8_nloop_head_serial_kernel[(floor_m, floor_n_blk)](*args, **common)
        if covered_n < n_tiles_total:
            _launch_masked(grid_m, n_tiles_total - covered_n, 0, covered_n, 1)
        if floor_m < grid_m:
            _launch_masked(grid_m - floor_m, n_tiles_total, floor_m, 0, 1)
        grid_n = (seq_kv + block_n - 1) // block_n
        if n_tiles_total < grid_n:
            _launch_masked(grid_m, grid_n - n_tiles_total, 0, n_tiles_total, 1)
    else:
        grid_n = (seq_kv + block_n - 1) // block_n
        _launch_masked(grid_m, grid_n, 0, 0, num_stages)

    return logits


def recommend_fp8_head_serial_launch(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Production knobs for gfx946 FP8 head-serial auto.

    Includes BN tile geometry, N-loop ``n_tiles``, and shape-tuned Q path:

    * H=8 mid/large KV: prologue MLS + N-WASP.
    * H=8 short KV + larger M (``N_TILES==1``, ``M>=64``): ``force_grouped``
      with BM32/qg8.
    * H>8 and ``N_TILES>=2``: ``chunk8_mls`` when ``H%8==0``.
    * else: grouped ``Q_GROUP`` / qg1 (small KV, high H).
    """
    block_m, block_n = (16, 64)
    try:
        from boltops.mqa_logits.triton.mqa_config_utils import get_bn_tile_dispatch

        disp = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
        block_m = int(disp["block_m_tile"])
        block_n = int(disp["bn_tile"])
    except Exception:
        pass
    n_tiles_total = max(int(seq_len_kv) // int(block_n), 1)
    # Gem5: N_TILES=8 (in-loop store) beats 4 when the KV span covers it.
    if n_tiles_total >= 8:
        n_tiles = 8
    elif n_tiles_total >= 4:
        n_tiles = 4
    elif n_tiles_total >= 2:
        n_tiles = 2
    else:
        n_tiles = 1
    if int(num_heads) >= 32 and int(seq_len) >= 32 and block_m == 16:
        block_m = 32
    chunk8_mls = (
        int(num_heads) > 8
        and int(num_heads) % 8 == 0
        and n_tiles >= 2
    )
    short_kv_h8_large_m = (
        int(num_heads) == 8 and n_tiles == 1 and int(seq_len) >= 64
    )
    force_grouped = short_kv_h8_large_m
    q_group = 8 if short_kv_h8_large_m else None
    if short_kv_h8_large_m:
        block_m = 32
    return {
        "block_m": block_m,
        "block_n": block_n,
        "n_tiles": n_tiles,
        "num_stages": 1 if short_kv_h8_large_m else 2,
        "wasp_partition_warps": (4, 4),
        "wasp_wdra": False,
        "assume_no_nans": 1,
        "misched_regpressure": 0 if num_heads <= 8 else 1,
        "enable_v_mmac_cluster": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "waves_per_eu": 0,
        "chunk8_mls": chunk8_mls,
        "force_grouped": force_grouped,
        "q_group": q_group,
        "use_mls_q": not force_grouped,
        "warp_specialize": True,
    }


def fp8_mqa_logits_head_serial(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    w: torch.Tensor,
    cu_starts: Optional[torch.Tensor] = None,
    cu_ends: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """gfx946 FP8 production entry: head-serial with shape-tuned knobs."""
    seq, heads, _ = q.shape
    seq_kv = kv.shape[0]
    rec = recommend_fp8_head_serial_launch(seq, seq_kv, heads)
    q_group = rec.get("q_group")
    return _fp8_mqa_logits_nloop_head_serial(
        q,
        kv,
        kv_scales,
        w,
        cu_starts,
        cu_ends,
        block_m=int(rec["block_m"]),
        block_n=int(rec["block_n"]),
        n_tiles=int(rec["n_tiles"]),
        warp_specialize=bool(rec["warp_specialize"]),
        num_stages=int(rec.get("num_stages", 2)),
        use_mls_q=bool(rec["use_mls_q"]),
        q_group=None if q_group is None else int(q_group),
        chunk8_mls=bool(rec["chunk8_mls"]),
        force_grouped=bool(rec.get("force_grouped", False)),
        wasp_wdra=bool(rec.get("wasp_wdra", False)),
        out=out,
    )


def try_fp8_mqa_logits_head_serial(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    w: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """gfx946-only FP8 head-serial launch; ``None`` on other arches."""
    try:
        from boltops.utility.arch_info import get_arch

        if get_arch() != "gfx946":
            return None
    except Exception:
        return None
    return fp8_mqa_logits_head_serial(
        q, kv, kv_scales, w, cu_starts, cu_ends, out=D_out
    )


# ---------------------------------------------------------------------------
# Half (BN-WASP)
# ---------------------------------------------------------------------------


def recommend_half_bn_wasp_launch(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Launch knobs for ``half_mqa_logits_bn_wasp`` on gfx946 half prefill.

    Default Q path is ``q_group=1`` + ``use_mls_q=True`` (gem5: beats max
    ``Q_GROUP`` + plain load). Caller may still override.
    """
    block_m, block_n = (64, 64) if num_heads <= 8 else (16, 64)
    try:
        from boltops.mqa_logits.triton.mqa_logits_half import (
            get_half_bn_dispatch,
        )

        disp = get_half_bn_dispatch(seq_len, seq_len_kv, num_heads)
        block_m = int(disp["block_m"])
        block_n = int(disp["block_n"])
    except Exception:
        pass
    return {
        "block_m": block_m,
        "block_n": block_n,
        "num_stages": 2,
        "wasp_partition_warps": (4, 4),
        # qg1 + MLS: WDRA stays useful for H>8 (single-head producer).
        "wasp_wdra": num_heads > 8,
        "q_group": 1,
        "use_mls_q": True,
        "assume_no_nans": 1,
        "misched_regpressure": 0 if num_heads <= 8 else 1,
        "enable_v_mmac_cluster": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "waves_per_eu": 0,
    }


def try_mqa_logits_half_bn_wasp(
    q: torch.Tensor,
    kv: torch.Tensor,
    w: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Run BN-WASP (or masked BN fallback) on gfx946; else return ``None``."""
    try:
        from boltops.utility.arch_info import get_arch

        if get_arch() != "gfx946":
            return None
    except Exception:
        return None
    seq, heads, _dim = q.shape
    seq_kv = int(kv.shape[0])
    rec = recommend_half_bn_wasp_launch(seq, seq_kv, heads)
    return half_mqa_logits_bn_wasp(
        q,
        kv,
        w,
        cu_starts=cu_starts,
        cu_ends=cu_ends,
        block_m=int(rec["block_m"]),
        block_n=int(rec["block_n"]),
        warp_specialize=True,
        q_group=int(rec["q_group"]),
        use_mls_q=bool(rec["use_mls_q"]),
        wasp_wdra=bool(rec["wasp_wdra"]),
        out=D_out,
    )


@triton.jit
def _half_bn_wasp_kernel(
    Q_ptr,
    KV_ptr,
    W_ptr,
    Logits_ptr,
    cu_start_ptr,
    cu_end_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv_s,
    stride_kv_d,
    stride_w_s,
    stride_w_h,
    stride_logits_s,
    stride_logits_k,
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_GROUP: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_MLS_Q: tl.constexpr,
):
    """WASP step loads ``Q_GROUP`` heads; window applied on the store.

    ``USE_MLS_Q`` (``Q_GROUP==1`` only): per-head ``[BM,D]`` ``tl.matrix_load``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_g = tl.arange(0, Q_GROUP)
    offs_h = tl.arange(0, NUM_HEADS)
    n_groups: tl.constexpr = NUM_HEADS // Q_GROUP
    mls_m0 = pid_m * BLOCK_M

    kv = tl.load(
        KV_ptr + offs_d[:, None] * stride_kv_d + offs_n[None, :] * stride_kv_s
    )
    # Weights stay outside the WASP loop: a second in-loop load is another
    # producer and HCU WASP rejects it.
    w_all = tl.load(
        W_ptr + offs_m[:, None] * stride_w_s + offs_h[None, :] * stride_w_h
    ).to(tl.float32)
    # Defer window loads until after WASP to cut SGPR live range across the
    # specialized loop (grouped tiles otherwise fail WDRA on SGPR spill).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if Q_GROUP == 1:
        q_ptrs = (
            Q_ptr + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
        )
        for hi in tl.range(0, NUM_HEADS, warp_specialize=WARP_SPECIALIZE):
            if USE_MLS_Q:
                q = tl.matrix_load(
                    Q_ptr + hi * stride_q_h,
                    shape=[seq_len, HEAD_SIZE],
                    strides=[stride_q_s, stride_q_d],
                    block_shape=[BLOCK_M, HEAD_SIZE],
                    offsets=[mls_m0, 0],
                )
            else:
                q = tl.load(q_ptrs + hi * stride_q_h)
            w_idx = tl.broadcast_to(hi + offs_g[None, :], (BLOCK_M, 1))
            w = tl.reshape(tl.gather(w_all, w_idx, axis=1), (BLOCK_M,))
            scores = tl.dot(q, kv)
            acc += tl.maximum(scores, 0.0) * w[:, None]
    else:
        offs_r = tl.arange(0, Q_GROUP * BLOCK_M)
        local_m = offs_r % BLOCK_M
        h_in_g = offs_r // BLOCK_M
        q_m_base = pid_m * BLOCK_M + local_m
        for g in tl.range(0, n_groups, warp_specialize=WARP_SPECIALIZE):
            local_h = g * Q_GROUP + h_in_g
            q_mat = tl.load(
                Q_ptr
                + q_m_base[:, None] * stride_q_s
                + local_h[:, None] * stride_q_h
                + offs_d[None, :] * stride_q_d
            )
            scores = tl.maximum(tl.dot(q_mat, kv), 0.0)
            scores = tl.reshape(scores, (Q_GROUP, BLOCK_M, BLOCK_N))
            w_idx = tl.broadcast_to(
                g * Q_GROUP + offs_g[None, :], (BLOCK_M, Q_GROUP)
            )
            w_group = tl.gather(w_all, w_idx, axis=1)
            w_t = tl.trans(w_group)
            acc += tl.sum(scores * w_t[:, :, None], axis=0)

    cu_start = tl.maximum(tl.load(cu_start_ptr + offs_m), 0)
    cu_end = tl.minimum(tl.load(cu_end_ptr + offs_m), seq_len_kv)
    in_window = (offs_n[None, :] >= cu_start[:, None]) & (
        offs_n[None, :] < cu_end[:, None]
    )
    out = tl.where(in_window, acc, float("-inf"))
    tl.store(
        Logits_ptr
        + offs_m[:, None] * stride_logits_s
        + offs_n[None, :] * stride_logits_k,
        out,
    )


@triton.jit
def _half_bn_masked_kernel(
    Q_ptr,
    KV_ptr,
    W_ptr,
    Logits_ptr,
    cu_start_ptr,
    cu_end_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv_s,
    stride_kv_d,
    stride_w_s,
    stride_w_h,
    stride_logits_s,
    stride_logits_k,
    seq_len,
    seq_len_kv,
    pid_m_offset,
    pid_n_offset,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Masked BN tile for ragged windows / non-aligned edges; no WASP."""
    pid_m = tl.program_id(0) + pid_m_offset
    pid_n = tl.program_id(1) + pid_n_offset

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_h = tl.arange(0, NUM_HEADS)
    m_mask = offs_m < seq_len
    n_mask = offs_n < seq_len_kv

    cu_start = tl.load(cu_start_ptr + offs_m, mask=m_mask, other=0)
    cu_start = tl.maximum(cu_start, 0)
    cu_end = tl.load(cu_end_ptr + offs_m, mask=m_mask, other=0)
    cu_end = tl.minimum(cu_end, seq_len_kv)

    tile_past = n_start >= tl.max(cu_end, axis=0)
    tile_before = (n_start + BLOCK_N) <= tl.min(cu_start, axis=0)
    logits_ptrs = (
        Logits_ptr
        + offs_m[:, None] * stride_logits_s
        + offs_n[None, :] * stride_logits_k
    )
    store_ok = m_mask[:, None] & n_mask[None, :]
    if tile_past or tile_before:
        tl.store(
            logits_ptrs,
            tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
            mask=store_ok,
        )
        return

    kv = tl.load(
        KV_ptr + offs_d[:, None] * stride_kv_d + offs_n[None, :] * stride_kv_s,
        mask=n_mask[None, :],
        other=0.0,
    )
    w_all = tl.load(
        W_ptr + offs_m[:, None] * stride_w_s + offs_h[None, :] * stride_w_h,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    for hi in tl.range(0, NUM_HEADS):
        q = tl.load(q_ptrs + hi * stride_q_h, mask=m_mask[:, None], other=0.0)
        w_idx = tl.broadcast_to(hi + tl.arange(0, 1)[None, :], (BLOCK_M, 1))
        w = tl.reshape(tl.gather(w_all, w_idx, axis=1), (BLOCK_M,))
        scores = tl.dot(q, kv)
        acc += tl.maximum(scores, 0.0) * w[:, None]

    in_window = (offs_n[None, :] >= cu_start[:, None]) & (
        offs_n[None, :] < cu_end[:, None]
    )
    out = tl.where(in_window & store_ok, acc, float("-inf"))
    tl.store(logits_ptrs, out, mask=store_ok)


def half_mqa_logits_bn_wasp(
    q: torch.Tensor,
    kv: torch.Tensor,
    w: torch.Tensor,
    cu_starts: Optional[torch.Tensor] = None,
    cu_ends: Optional[torch.Tensor] = None,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    warp_specialize: bool = False,
    num_stages: int = 2,
    num_warps: int = 4,
    wasp_partition_warps: Sequence[int] = (4, 4),
    wasp_wdra: Optional[bool] = None,
    q_group: Optional[int] = None,
    use_mls_q: bool = False,
    assume_no_nans: Optional[int] = None,
    misched_regpressure: Optional[int] = None,
    enable_v_mmac_cluster: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BN logits. ``q`` is [M, H, D], ``kv`` is [N, D], ``w`` is [M, H].

    Floor tiles use WASP when enabled; remainder uses masked edges. Window is
    applied on the store (no separate clear).

    ``use_mls_q``: ``Q_GROUP==1`` loads Q via ``tl.matrix_load`` (needs
    contiguous D). Production recommend defaults to ``q_group=1`` + MLS;
    non-contiguous Q falls back to max legal ``Q_GROUP`` + ``tl.load``.
    """
    seq, heads, dim = q.shape
    seq_kv, dim_kv = kv.shape
    if dim != dim_kv or w.shape != (seq, heads):
        raise ValueError("q, kv, w shapes do not match")
    rec = recommend_half_bn_wasp_launch(seq, seq_kv, heads)
    block_m = int(rec["block_m"] if block_m is None else block_m)
    block_n = int(rec["block_n"] if block_n is None else block_n)

    auto_q = q_group is None
    if auto_q:
        q_group = int(rec.get("q_group", 1))
        use_mls_q_v = bool(rec.get("use_mls_q", True))
    else:
        q_group = int(q_group)
        if q_group < 1 or heads % q_group != 0:
            raise ValueError(f"q_group={q_group} must divide heads={heads}")
        use_mls_q_v = bool(use_mls_q)

    if use_mls_q_v and q_group != 1:
        raise ValueError(
            f"use_mls_q requires q_group==1, got q_group={q_group}"
        )
    if use_mls_q_v and int(q.stride(2)) != 1:
        if not auto_q:
            raise ValueError(
                "use_mls_q requires contiguous HEAD_SIZE (stride_d==1), "
                f"got stride_d={q.stride(2)}"
            )
        # Auto path: fall back to legacy max Q_GROUP + plain load.
        use_mls_q_v = False
        if heads % 8 == 0:
            q_group = 8
        elif heads % 4 == 0:
            q_group = 4
        elif heads % 2 == 0:
            q_group = 2
        else:
            q_group = 1

    if cu_starts is None:
        cu_starts = torch.zeros(seq, device=q.device, dtype=torch.int32)
    if cu_ends is None:
        cu_ends = torch.full((seq,), seq_kv, device=q.device, dtype=torch.int32)

    if out is None:
        logits = torch.empty(seq, seq_kv, device=q.device, dtype=torch.float32)
    else:
        if out.shape != (seq, seq_kv) or out.dtype != torch.float32:
            raise ValueError("out must be float32 [M, N]")
        logits = out

    hip_kwargs = {
        "assume_no_nans": int(
            rec["assume_no_nans"] if assume_no_nans is None else assume_no_nans
        ),
        "misched_regpressure": bool(
            int(
                rec["misched_regpressure"]
                if misched_regpressure is None
                else misched_regpressure
            )
        ),
        "enable_v_mmac_cluster": int(
            rec["enable_v_mmac_cluster"]
            if enable_v_mmac_cluster is None
            else enable_v_mmac_cluster
        ),
        "matrix_instr_nonkdim": int(
            rec["matrix_instr_nonkdim"]
            if matrix_instr_nonkdim is None
            else matrix_instr_nonkdim
        ),
        "kpack": int(rec["kpack"] if kpack is None else kpack),
    }
    waves = int(rec["waves_per_eu"] if waves_per_eu is None else waves_per_eu)
    if waves:
        hip_kwargs["waves_per_eu"] = waves
    hip = _filter_hip_launch_kwargs(hip_kwargs)

    grid_m = (seq + block_m - 1) // block_m
    grid_n = (seq_kv + block_n - 1) // block_n
    floor_m = seq // block_m
    floor_n = seq_kv // block_n

    def _launch_masked(gm, gn, m0, n0, stages):
        if gm <= 0 or gn <= 0:
            return
        _half_bn_masked_kernel[(gm, gn)](
            q,
            kv,
            w,
            logits,
            cu_starts,
            cu_ends,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            w.stride(0),
            w.stride(1),
            logits.stride(0),
            logits.stride(1),
            seq,
            seq_kv,
            m0,
            n0,
            NUM_HEADS=heads,
            HEAD_SIZE=dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=stages,
            **hip,
        )

    if warp_specialize and floor_m > 0 and floor_n > 0:
        launch = dict(
            NUM_HEADS=heads,
            HEAD_SIZE=dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            Q_GROUP=q_group,
            WARP_SPECIALIZE=True,
            USE_MLS_Q=use_mls_q_v,
            num_stages=num_stages,
            num_warps=num_warps,
            **hip,
        )
        parts: Tuple[int, ...] = tuple(int(x) for x in wasp_partition_warps)
        launch["wasp_partition_warps"] = parts
        wdra = rec["wasp_wdra"] if wasp_wdra is None else bool(wasp_wdra)
        # Grouped Q can SGPR-spill under WDRA (same as FP8).
        if q_group > 1 and wasp_wdra is None:
            wdra = False
        launch["wasp_wdra"] = wdra
        _half_bn_wasp_kernel[(floor_m, floor_n)](
            q,
            kv,
            w,
            logits,
            cu_starts,
            cu_ends,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            w.stride(0),
            w.stride(1),
            logits.stride(0),
            logits.stride(1),
            seq,
            seq_kv,
            **launch,
        )
        if floor_n < grid_n:
            _launch_masked(grid_m, grid_n - floor_n, 0, floor_n, 1)
        if floor_m < grid_m:
            _launch_masked(grid_m - floor_m, grid_n, floor_m, 0, 1)
    else:
        _launch_masked(grid_m, grid_n, 0, 0, num_stages)

    return logits
