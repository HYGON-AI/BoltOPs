"""Experimental Gluon half (fp16/bf16) MQA logits: BN-aligned dataflow.

Port of half BN without scales. Defaults match FP8 bn_align and half benches:
``w_lds_pp=False``, ``q_double_buf=False`` (register W + BN Q software pipe).
Half W is tiny; W LDS ping-pong is usually a net loss vs BN / plain Gluon.

MFMA: ``instr_shape=[16,16,16]``, ``k_width=4``, shared ``vec=4/maxPhase=16``
(matching half BN TTGIR), ``AMDMFMALayout(transposed=True)`` (mmacLayout=2).

Launch: ``mqa_logits_half_gluon_bn_align(...)`` or half ``backend="gluon_bn_align"``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.mqa_logits.triton.fp8_mqa_logits import _filter_hip_launch_kwargs
from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
    _mfma_version_for_arch,
    gluon_prefill_available,
)
from boltops.mqa_logits.triton.mqa_config_utils import (
    clamp_gluon_w_group,
    tiled_grid_n,
)
from boltops.mqa_logits.triton.mqa_logits_half import get_half_bn_dispatch

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _GLUON_AVAILABLE = True
except ImportError:
    gluon = None  # type: ignore[assignment]
    gl = None  # type: ignore[assignment]
    _GLUON_AVAILABLE = False


if _GLUON_AVAILABLE:
    from triton.experimental.gluon.language._layouts import constexpr_function

    @constexpr_function
    def _lift_row_layout_to_1m(ll, block_m):
        """Embed a 1-D DistributedLinearLayout on [M] as [1, M] (broadcast dim 0)."""
        return gl.DistributedLinearLayout(
            reg_bases=[[0, b[0]] for b in ll.reg_bases],
            lane_bases=[[0, b[0]] for b in ll.lane_bases],
            warp_bases=[[0, b[0]] for b in ll.warp_bases],
            block_bases=[[0, b[0]] for b in ll.block_bases] if ll.block_bases else [],
            shape=[1, block_m],
        )

    @gluon.jit
    def _mqa_logits_half_gluon_bn_align_kernel(
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
        stride_kv_s: tl.constexpr,
        stride_kv_d: tl.constexpr,
        stride_w_s: tl.int64,
        stride_w_h: tl.constexpr,
        stride_logits_s: tl.constexpr,
        stride_logits_k: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ASSUME_KS_ZERO: tl.constexpr,
        MFMA_VERSION: tl.constexpr,
        KV_TILES_N: tl.constexpr,
        NUM_WARPS: tl.constexpr,
        W_GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HEAD_CHUNK: tl.constexpr,
        # If true: buffer_load KV already in DotOperand B (skip LDS cast).
        KV_DIRECT_DOT: tl.constexpr,
        # If true: continuous ship-style Q even/odd LDS ping-pong across HEAD_CHUNK.
        Q_DOUBLE_BUF: tl.constexpr,
        # If true: ship-style W LDS ping-pong; else BN #mma group buffer_load + where/sum.
        W_LDS_PP: tl.constexpr,
        # If true: BN-style epilogue (where on #mma, then convert out) — default.
        # If false: store-layout masks + convert only acc.
        BN_STYLE_EPILOGUE: tl.constexpr,
    ):
        """BN-aligned Gluon: register W; Q/KV via init-from-value shared alloc."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        split_id = tl.program_id(2)
        start_h = split_id * HEAD_CHUNK
        if start_h >= NUM_HEADS:
            return
        head_end = start_h + HEAD_CHUNK
        if head_end > NUM_HEADS:
            head_end = NUM_HEADS

        n_start = pid_n * BLOCK_N
        m_start = pid_m * BLOCK_M
        NumWarps: gl.constexpr = NUM_WARPS

        layout_q: gl.constexpr = gl.BlockedLayout(
            # Half: size_per_thread=8 packs into shared vec=8 (b128).
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[NumWarps, 1],
            order=[1, 0],
        )
        # Match half BN `#blocked` for KV loads.
        layout_kv: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[16, 4],
            warps_per_cta=[1, NumWarps],
            order=[0, 1],
        )
        layout_store: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[2, 32],
            warps_per_cta=[NumWarps, 1],
            order=[1, 0],
        )
        # Match half BN `#shared` / `#shared1` (vec=4, maxPhase=16).
        shared_q: gl.constexpr = gl.SwizzledSharedLayout(
            vec=4, per_phase=1, max_phase=16, order=[1, 0]
        )
        shared_kv: gl.constexpr = gl.SwizzledSharedLayout(
            vec=4, per_phase=1, max_phase=16, order=[0, 1]
        )
        # M tiles scale with BLOCK_M so we can match lightop-like BM=128
        # (``mqa_logits_128x128x32``): shape_m = 16 * (BLOCK_M/16) * warps_m.
        M_TILES: gl.constexpr = BLOCK_M // 16
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=MFMA_VERSION,
            instr_shape=[16, 16, 16],
            # True → HCU mmacLayout=INTERLEAVE(2), matching half BN accelerate.
            transposed=True,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[M_TILES, KV_TILES_N],
        )
        # Half `tl.dot` typically uses kWidth=4 for f16/bf16 on CDNA3.
        mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mfma_layout, k_width=4
        )
        mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mfma_layout, k_width=4
        )
        w_row_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)

        offs_n_col = n_start + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        offs_m_row = m_start + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
        )
        # KV offsets: blocked (coalesced) or DotOperand B (direct, no LDS).
        if KV_DIRECT_DOT:
            offs_n_kv = n_start + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout_b)
            )
            offs_d_kv = gl.arange(
                0, HEAD_SIZE, layout=gl.SliceLayout(1, mfma_layout_b)
            )
        else:
            offs_n_kv = n_start + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, layout_kv)
            )
            offs_d_kv = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, layout_kv))
        offs_m_q = m_start + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, layout_q)
        )
        offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))
        offs_m_w = m_start + gl.arange(0, BLOCK_M, layout=w_row_layout)
        offs_m_st = m_start + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, layout_store)
        )
        offs_n_st = n_start + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, layout_store)
        )

        n_mask = offs_n_col < seq_len_kv
        n_mask_kv = offs_n_kv < seq_len_kv
        m_mask = offs_m_row < seq_len
        m_mask_q = offs_m_q < seq_len
        m_mask_w = offs_m_w < seq_len
        store_off = (
            offs_m_st[:, None] * stride_logits_s + offs_n_st[None, :] * stride_logits_k
        )
        store_mask = (offs_m_st[:, None] < seq_len) & (offs_n_st[None, :] < seq_len_kv)

        cu_end = gl.amd.cdna3.buffer_load(
            ptr=cu_end_ptr, offsets=offs_m_row.to(tl.int32), mask=m_mask
        )
        cu_end = tl.minimum(cu_end, seq_len_kv)
        if ASSUME_KS_ZERO:
            cu_start = tl.zeros([BLOCK_M], dtype=tl.int32)
        else:
            cu_start = gl.amd.cdna3.buffer_load(
                ptr=cu_start_ptr, offsets=offs_m_row.to(tl.int32), mask=m_mask
            )
            cu_start = tl.maximum(cu_start, 0)

        n_end = n_start + BLOCK_N
        tile_past = n_start >= tl.max(cu_end, axis=0)
        tile_before = False
        if not ASSUME_KS_ZERO:
            tile_before = n_end <= tl.min(cu_start, axis=0)
        if tile_past or tile_before:
            if NUM_SPLITS == 1 or split_id == 0:
                # Store-layout -inf: avoid extra #mma→#blocked convert on early-exit.
                neg_inf = gl.full(
                    [BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=layout_store,
                )
                gl.amd.cdna3.buffer_store(
                    neg_inf,
                    ptr=logits_ptr,
                    offsets=store_off,
                    mask=store_mask,
                )
            return

        # KV: blocked→LDS cast (BN), or direct DotOperand B load (experiment).
        kv = gl.amd.cdna3.buffer_load(
            ptr=KV_ptr,
            offsets=(
                offs_d_kv[:, None] * stride_kv_d + offs_n_kv[None, :] * stride_kv_s
            ).to(tl.int32),
            mask=n_mask_kv[None, :],
        )
        if KV_DIRECT_DOT:
            mfma_k = kv
        else:
            smem_kv = gl.allocate_shared_memory(
                KV_ptr.type.element_ty, [HEAD_SIZE, BLOCK_N], shared_kv, value=kv
            )
            mfma_k = smem_kv.load(layout=mfma_layout_b)

        mfma_zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32, layout=mfma_layout)
        acc = mfma_zero

        tl.static_assert(HEAD_CHUNK % W_GROUP == 0)
        tl.static_assert(W_GROUP >= 1)
        # W LDS ping-pong needs even W_GROUP; plain BN path allows W_GROUP=1.
        if W_LDS_PP:
            tl.static_assert(W_GROUP % 2 == 0)
        n_groups: gl.constexpr = HEAD_CHUNK // W_GROUP

        # ---- Optional continuous Q LDS double-buf (ship-style across HEAD_CHUNK) ----
        if Q_DOUBLE_BUF:
            smem_q_even = gl.allocate_shared_memory(
                Q_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE], shared_q
            )
            smem_q_odd = gl.allocate_shared_memory(
                Q_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE], shared_q
            )
            q0 = gl.amd.cdna3.buffer_load(
                ptr=Q_ptr,
                offsets=(
                    offs_m_q[:, None] * stride_q_s
                    + start_h * stride_q_h
                    + offs_d_q[None, :] * stride_q_d
                ).to(tl.int32),
                mask=m_mask_q[:, None],
            )
            smem_q_even.store(q0)
            q_pref = q0
            if HEAD_CHUNK > 1:
                q_pref = gl.amd.cdna3.buffer_load(
                    ptr=Q_ptr,
                    offsets=(
                        offs_m_q[:, None] * stride_q_s
                        + (start_h + 1) * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
                    mask=m_mask_q[:, None],
                )

        # ---- Optional W LDS ping-pong (ship-style) ----
        if W_LDS_PP:
            # Warp has 64 threads; cover BLOCK_M rows (BM<=64 historically;
            # for BM=128 use size_per_thread on M when needed).
            tl.static_assert(BLOCK_M <= 128)
            if W_GROUP < 8:
                WG_SPT: gl.constexpr = 1
                WG_TPW: gl.constexpr = W_GROUP
                m_per_warp: gl.constexpr = 64 // W_GROUP
            else:
                WG_SPT: gl.constexpr = W_GROUP // 8
                WG_TPW: gl.constexpr = 8
                m_per_warp: gl.constexpr = 8
            M_SPT: gl.constexpr = (BLOCK_M + NumWarps * m_per_warp - 1) // (
                NumWarps * m_per_warp
            )
            M_TPW: gl.constexpr = m_per_warp
            layout_w_hm: gl.constexpr = gl.BlockedLayout(
                size_per_thread=[WG_SPT, M_SPT],
                threads_per_warp=[WG_TPW, M_TPW],
                warps_per_cta=[1, NumWarps],
                order=[0, 1],
            )
            shared_w: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[1, 0]
            )
            layout_w_1m: gl.constexpr = _lift_row_layout_to_1m(
                gl.to_linear_layout(w_row_layout, [BLOCK_M]), BLOCK_M
            )
            smem_w = gl.allocate_shared_memory(
                gl.float32, [2, W_GROUP, BLOCK_M], shared_w
            )
            offs_h_wg_lds = gl.arange(
                0, W_GROUP, layout=gl.SliceLayout(1, layout_w_hm)
            )
            offs_m_wg = m_start + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(0, layout_w_hm)
            )
            w_g0 = gl.amd.cdna3.buffer_load(
                ptr=weights_ptr,
                offsets=(
                    (start_h + offs_h_wg_lds)[:, None] * stride_w_h
                    + offs_m_wg[None, :] * stride_w_s
                ).to(tl.int32),
            ).to(gl.float32)
            smem_w.index(0).store(w_g0)
        else:
            # BN: load ``[BLOCK_M, W_GROUP]`` already on `#mma`, then select+reduce.
            offs_h_wg = gl.arange(0, W_GROUP, layout=gl.SliceLayout(0, mfma_layout))

        # ---- Head loops: 4 combinations (default = BN Q + BN W) ----
        if Q_DOUBLE_BUF and W_LDS_PP:
            # Ship continuous Q + W LDS ping-pong.
            for g in tl.range(0, n_groups - 1):
                buf = g & 1
                offs_h_n = start_h + (g + 1) * W_GROUP + offs_h_wg_lds
                w_gn = gl.amd.cdna3.buffer_load(
                    ptr=weights_ptr,
                    offsets=(
                        offs_h_n[:, None] * stride_w_h
                        + offs_m_wg[None, :] * stride_w_s
                    ).to(tl.int32),
                ).to(gl.float32)
                smem_w.index(1 - buf).store(w_gn)
                w_plane = smem_w.index(buf)
                for hi in tl.static_range(W_GROUP):
                    h = start_h + g * W_GROUP + hi
                    q_cur = smem_q_odd if (hi & 1) else smem_q_even
                    q_nxt = smem_q_even if (hi & 1) else smem_q_odd
                    q_tile = q_cur.load(layout=mfma_layout_a)
                    scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                    scores = gl.maximum(scores, 0.0)
                    q_nxt.store(q_pref)
                    h_pref = tl.minimum(h + 2, head_end - 1)
                    q_pref = gl.amd.cdna3.buffer_load(
                        ptr=Q_ptr,
                        offsets=(
                            offs_m_q[:, None] * stride_q_s
                            + h_pref * stride_q_h
                            + offs_d_q[None, :] * stride_q_d
                        ).to(tl.int32),
                        mask=m_mask_q[:, None],
                    )
                    w_h = gl.convert_layout(
                        w_plane.slice(hi, 1, dim=0)
                        .load(layout=layout_w_1m)
                        .reshape([BLOCK_M]),
                        w_row_layout,
                    )
                    acc = acc + scores * w_h[:, None]
            buf = (n_groups - 1) & 1
            w_plane = smem_w.index(buf)
            for hi in tl.static_range(W_GROUP - 1):
                h = start_h + (n_groups - 1) * W_GROUP + hi
                q_cur = smem_q_odd if (hi & 1) else smem_q_even
                q_nxt = smem_q_even if (hi & 1) else smem_q_odd
                q_tile = q_cur.load(layout=mfma_layout_a)
                scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                scores = gl.maximum(scores, 0.0)
                q_nxt.store(q_pref)
                h_pref = tl.minimum(h + 2, head_end - 1)
                q_pref = gl.amd.cdna3.buffer_load(
                    ptr=Q_ptr,
                    offsets=(
                        offs_m_q[:, None] * stride_q_s
                        + h_pref * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
                    mask=m_mask_q[:, None],
                )
                w_h = gl.convert_layout(
                    w_plane.slice(hi, 1, dim=0)
                    .load(layout=layout_w_1m)
                    .reshape([BLOCK_M]),
                    w_row_layout,
                )
                acc = acc + scores * w_h[:, None]
            q_tile = smem_q_odd.load(layout=mfma_layout_a)
            scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
            scores = gl.maximum(scores, 0.0)
            w_h = gl.convert_layout(
                w_plane.slice(W_GROUP - 1, 1, dim=0)
                .load(layout=layout_w_1m)
                .reshape([BLOCK_M]),
                w_row_layout,
            )
            acc = acc + scores * w_h[:, None]

        elif Q_DOUBLE_BUF:
            # Continuous Q LDS + BN W where/sum.
            for g in tl.range(0, n_groups - 1):
                h0 = start_h + g * W_GROUP
                offs_h_g = h0 + offs_h_wg
                w_g = gl.amd.cdna3.buffer_load(
                    ptr=weights_ptr,
                    offsets=(
                        offs_m_w[:, None] * stride_w_s
                        + offs_h_g[None, :] * stride_w_h
                    ).to(tl.int32),
                    mask=m_mask_w[:, None] & (offs_h_g[None, :] < NUM_HEADS),
                ).to(gl.float32)
                for hi in tl.static_range(W_GROUP):
                    h = h0 + hi
                    q_cur = smem_q_odd if (hi & 1) else smem_q_even
                    q_nxt = smem_q_even if (hi & 1) else smem_q_odd
                    w_h = tl.sum(
                        tl.where(offs_h_wg[None, :] == hi, w_g, 0.0),
                        axis=1,
                    )
                    q_tile = q_cur.load(layout=mfma_layout_a)
                    scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                    scores = gl.maximum(scores, 0.0)
                    q_nxt.store(q_pref)
                    h_pref = tl.minimum(h + 2, head_end - 1)
                    q_pref = gl.amd.cdna3.buffer_load(
                        ptr=Q_ptr,
                        offsets=(
                            offs_m_q[:, None] * stride_q_s
                            + h_pref * stride_q_h
                            + offs_d_q[None, :] * stride_q_d
                        ).to(tl.int32),
                        mask=m_mask_q[:, None],
                    )
                    acc = acc + scores * w_h[:, None]
            h0 = start_h + (n_groups - 1) * W_GROUP
            offs_h_g = h0 + offs_h_wg
            w_g = gl.amd.cdna3.buffer_load(
                ptr=weights_ptr,
                offsets=(
                    offs_m_w[:, None] * stride_w_s
                    + offs_h_g[None, :] * stride_w_h
                ).to(tl.int32),
                mask=m_mask_w[:, None] & (offs_h_g[None, :] < NUM_HEADS),
            ).to(gl.float32)
            for hi in tl.static_range(W_GROUP - 1):
                h = h0 + hi
                q_cur = smem_q_odd if (hi & 1) else smem_q_even
                q_nxt = smem_q_even if (hi & 1) else smem_q_odd
                w_h = tl.sum(
                    tl.where(offs_h_wg[None, :] == hi, w_g, 0.0),
                    axis=1,
                )
                q_tile = q_cur.load(layout=mfma_layout_a)
                scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                scores = gl.maximum(scores, 0.0)
                q_nxt.store(q_pref)
                h_pref = tl.minimum(h + 2, head_end - 1)
                q_pref = gl.amd.cdna3.buffer_load(
                    ptr=Q_ptr,
                    offsets=(
                        offs_m_q[:, None] * stride_q_s
                        + h_pref * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
                    mask=m_mask_q[:, None],
                )
                acc = acc + scores * w_h[:, None]
            q_tile = smem_q_odd.load(layout=mfma_layout_a)
            scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
            scores = gl.maximum(scores, 0.0)
            w_h = tl.sum(
                tl.where(offs_h_wg[None, :] == (W_GROUP - 1), w_g, 0.0),
                axis=1,
            )
            acc = acc + scores * w_h[:, None]

        elif W_LDS_PP:
            # BN Q single-alloc + W LDS ping-pong.
            for g in tl.range(0, n_groups - 1):
                buf = g & 1
                offs_h_n = start_h + (g + 1) * W_GROUP + offs_h_wg_lds
                w_gn = gl.amd.cdna3.buffer_load(
                    ptr=weights_ptr,
                    offsets=(
                        offs_h_n[:, None] * stride_w_h
                        + offs_m_wg[None, :] * stride_w_s
                    ).to(tl.int32),
                ).to(gl.float32)
                smem_w.index(1 - buf).store(w_gn)
                w_plane = smem_w.index(buf)
                h0 = start_h + g * W_GROUP
                q_cur = gl.amd.cdna3.buffer_load(
                    ptr=Q_ptr,
                    offsets=(
                        offs_m_q[:, None] * stride_q_s
                        + h0 * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
                    mask=m_mask_q[:, None],
                )
                for hi in tl.static_range(W_GROUP):
                    h = h0 + hi
                    if hi + 1 < W_GROUP:
                        q_next = gl.amd.cdna3.buffer_load(
                            ptr=Q_ptr,
                            offsets=(
                                offs_m_q[:, None] * stride_q_s
                                + (h + 1) * stride_q_h
                                + offs_d_q[None, :] * stride_q_d
                            ).to(tl.int32),
                            mask=m_mask_q[:, None],
                        )
                    w_h = gl.convert_layout(
                        w_plane.slice(hi, 1, dim=0)
                        .load(layout=layout_w_1m)
                        .reshape([BLOCK_M]),
                        w_row_layout,
                    )
                    smem_q = gl.allocate_shared_memory(
                        Q_ptr.type.element_ty,
                        [BLOCK_M, HEAD_SIZE],
                        shared_q,
                        value=q_cur,
                    )
                    q_tile = smem_q.load(layout=mfma_layout_a)
                    scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                    scores = gl.maximum(scores, 0.0)
                    acc = acc + scores * w_h[:, None]
                    if hi + 1 < W_GROUP:
                        q_cur = q_next
            buf = (n_groups - 1) & 1
            w_plane = smem_w.index(buf)
            h0 = start_h + (n_groups - 1) * W_GROUP
            q_cur = gl.amd.cdna3.buffer_load(
                ptr=Q_ptr,
                offsets=(
                    offs_m_q[:, None] * stride_q_s
                    + h0 * stride_q_h
                    + offs_d_q[None, :] * stride_q_d
                ).to(tl.int32),
                mask=m_mask_q[:, None],
            )
            for hi in tl.static_range(W_GROUP):
                h = h0 + hi
                if hi + 1 < W_GROUP:
                    q_next = gl.amd.cdna3.buffer_load(
                        ptr=Q_ptr,
                        offsets=(
                            offs_m_q[:, None] * stride_q_s
                            + (h + 1) * stride_q_h
                            + offs_d_q[None, :] * stride_q_d
                        ).to(tl.int32),
                        mask=m_mask_q[:, None],
                    )
                w_h = gl.convert_layout(
                    w_plane.slice(hi, 1, dim=0)
                    .load(layout=layout_w_1m)
                    .reshape([BLOCK_M]),
                    w_row_layout,
                )
                smem_q = gl.allocate_shared_memory(
                    Q_ptr.type.element_ty,
                    [BLOCK_M, HEAD_SIZE],
                    shared_q,
                    value=q_cur,
                )
                q_tile = smem_q.load(layout=mfma_layout_a)
                scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                scores = gl.maximum(scores, 0.0)
                acc = acc + scores * w_h[:, None]
                if hi + 1 < W_GROUP:
                    q_cur = q_next

        else:
            # Default: BN-style. W_GROUP==1 uses register Q/W prefetch (half BN).
            if W_GROUP == 1:
                h0 = start_h
                q_cur = gl.amd.cdna3.buffer_load(
                    ptr=Q_ptr,
                    offsets=(
                        offs_m_q[:, None] * stride_q_s
                        + h0 * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
                    mask=m_mask_q[:, None] & (h0 < NUM_HEADS),
                )
                w_cur = gl.amd.cdna3.buffer_load(
                    ptr=weights_ptr,
                    offsets=(offs_m_w * stride_w_s + h0 * stride_w_h).to(tl.int32),
                    mask=m_mask_w & (h0 < NUM_HEADS),
                ).to(gl.float32)
                for hi in tl.static_range(HEAD_CHUNK):
                    h = start_h + hi
                    if hi + 1 < HEAD_CHUNK:
                        q_next = gl.amd.cdna3.buffer_load(
                            ptr=Q_ptr,
                            offsets=(
                                offs_m_q[:, None] * stride_q_s
                                + (h + 1) * stride_q_h
                                + offs_d_q[None, :] * stride_q_d
                            ).to(tl.int32),
                            mask=m_mask_q[:, None] & ((h + 1) < NUM_HEADS),
                        )
                        w_next = gl.amd.cdna3.buffer_load(
                            ptr=weights_ptr,
                            offsets=(
                                offs_m_w * stride_w_s + (h + 1) * stride_w_h
                            ).to(tl.int32),
                            mask=m_mask_w & ((h + 1) < NUM_HEADS),
                        ).to(gl.float32)
                    smem_q = gl.allocate_shared_memory(
                        Q_ptr.type.element_ty,
                        [BLOCK_M, HEAD_SIZE],
                        shared_q,
                        value=q_cur,
                    )
                    q_tile = smem_q.load(layout=mfma_layout_a)
                    scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                    scores = gl.maximum(scores, 0.0)
                    acc = acc + scores * w_cur[:, None]
                    if hi + 1 < HEAD_CHUNK:
                        q_cur = q_next
                        w_cur = w_next
            else:
                # W_GROUP>=2: BN batched weight load + where/sum + Q software pipe.
                for g in tl.static_range(n_groups):
                    h0 = start_h + g * W_GROUP
                    offs_h_g = h0 + offs_h_wg
                    w_g = gl.amd.cdna3.buffer_load(
                        ptr=weights_ptr,
                        offsets=(
                            offs_m_w[:, None] * stride_w_s
                            + offs_h_g[None, :] * stride_w_h
                        ).to(tl.int32),
                        mask=m_mask_w[:, None] & (offs_h_g[None, :] < NUM_HEADS),
                    ).to(gl.float32)
                    q_cur = gl.amd.cdna3.buffer_load(
                        ptr=Q_ptr,
                        offsets=(
                            offs_m_q[:, None] * stride_q_s
                            + h0 * stride_q_h
                            + offs_d_q[None, :] * stride_q_d
                        ).to(tl.int32),
                        mask=m_mask_q[:, None],
                    )
                    for hi in tl.static_range(W_GROUP):
                        h = h0 + hi
                        if hi + 1 < W_GROUP:
                            q_next = gl.amd.cdna3.buffer_load(
                                ptr=Q_ptr,
                                offsets=(
                                    offs_m_q[:, None] * stride_q_s
                                    + (h + 1) * stride_q_h
                                    + offs_d_q[None, :] * stride_q_d
                                ).to(tl.int32),
                                mask=m_mask_q[:, None],
                            )
                        w_h = tl.sum(
                            tl.where(offs_h_wg[None, :] == hi, w_g, 0.0),
                            axis=1,
                        )
                        smem_q = gl.allocate_shared_memory(
                            Q_ptr.type.element_ty,
                            [BLOCK_M, HEAD_SIZE],
                            shared_q,
                            value=q_cur,
                        )
                        q_tile = smem_q.load(layout=mfma_layout_a)
                        scores = gl.amd.cdna3.mfma(q_tile, mfma_k, mfma_zero)
                        scores = gl.maximum(scores, 0.0)
                        acc = acc + scores * w_h[:, None]
                        if hi + 1 < W_GROUP:
                            q_cur = q_next

        # Epilogue: store-layout vs BN-style (#mma where + convert).
        if BN_STYLE_EPILOGUE:
            if ASSUME_KS_ZERO:
                in_window = offs_n_col[None, :] < cu_end[:, None]
            else:
                in_window = (offs_n_col[None, :] >= cu_start[:, None]) & (
                    offs_n_col[None, :] < cu_end[:, None]
                )
            store_ok = m_mask[:, None] & n_mask[None, :]
            if NUM_SPLITS == 1:
                neg_inf = gl.full(
                    [BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=mfma_layout,
                )
                out = tl.where(in_window & store_ok, acc, neg_inf)
                out_st = gl.convert_layout(out, layout_store)
                gl.amd.cdna3.buffer_store(
                    out_st, ptr=logits_ptr, offsets=store_off, mask=store_mask
                )
            else:
                acc_st = gl.convert_layout(acc, layout_store)
                in_window_st = gl.convert_layout(in_window, layout_store)
                store_ok_st = gl.convert_layout(store_ok, layout_store)
                gl.amd.cdna3.buffer_atomic_add(
                    logits_ptr, store_off, acc_st, mask=store_ok_st & in_window_st
                )
                if split_id == 0:
                    neg_inf = gl.full(
                        [BLOCK_M, BLOCK_N],
                        float("-inf"),
                        dtype=tl.float32,
                        layout=layout_store,
                    )
                    gl.amd.cdna3.buffer_store(
                        neg_inf,
                        ptr=logits_ptr,
                        offsets=store_off,
                        mask=store_ok_st & (~in_window_st),
                    )
        else:
            # Masks on store layout: only convert acc (#mma -> #blocked).
            cu_end_st = gl.amd.cdna3.buffer_load(
                ptr=cu_end_ptr,
                offsets=offs_m_st.to(tl.int32),
                mask=offs_m_st < seq_len,
            )
            cu_end_st = tl.minimum(cu_end_st, seq_len_kv)
            if ASSUME_KS_ZERO:
                in_window_st = offs_n_st[None, :] < cu_end_st[:, None]
            else:
                cu_start_st = gl.amd.cdna3.buffer_load(
                    ptr=cu_start_ptr,
                    offsets=offs_m_st.to(tl.int32),
                    mask=offs_m_st < seq_len,
                )
                cu_start_st = tl.maximum(cu_start_st, 0)
                in_window_st = (offs_n_st[None, :] >= cu_start_st[:, None]) & (
                    offs_n_st[None, :] < cu_end_st[:, None]
                )

            acc_st = gl.convert_layout(acc, layout_store)
            if NUM_SPLITS == 1:
                neg_inf = gl.full(
                    [BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=layout_store,
                )
                out = tl.where(in_window_st & store_mask, acc_st, neg_inf)
                gl.amd.cdna3.buffer_store(
                    out, ptr=logits_ptr, offsets=store_off, mask=store_mask
                )
            else:
                gl.amd.cdna3.buffer_atomic_add(
                    logits_ptr, store_off, acc_st, mask=store_mask & in_window_st
                )
                if split_id == 0:
                    neg_inf = gl.full(
                        [BLOCK_M, BLOCK_N],
                        float("-inf"),
                        dtype=tl.float32,
                        layout=layout_store,
                    )
                    gl.amd.cdna3.buffer_store(
                        neg_inf,
                        ptr=logits_ptr,
                        offsets=store_off,
                        mask=store_mask & (~in_window_st),
                    )


def mqa_logits_half_gluon_bn_align(
    Q: torch.Tensor,
    KV: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    schedule_hint: Optional[str] = None,
    use_block_pingpong: Optional[bool] = None,
    assume_ks_zero: bool = False,
    kv_tiles_n: Optional[int] = None,
    w_group: Optional[int] = None,
    head_split_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    kv_direct_dot: bool = False,
    q_double_buf: bool = False,
    w_lds_pp: bool = False,
    bn_style_epilogue: bool = True,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Experimental Gluon half path whose dataflow mirrors half BN-tile.

    No scales. Defaults: ``w_lds_pp=False``, ``q_double_buf=False``,
    ``bn_style_epilogue=True`` (#mma where + convert; beats store-layout
    epilogue on the all-sizes grid vs lightop).
    Tile / warps / split from ``get_half_bn_dispatch`` (+ half JSON).
    """
    if not gluon_prefill_available() or not _GLUON_AVAILABLE:
        raise RuntimeError("Gluon bn_align requires triton.experimental.gluon")

    if Q.dtype != KV.dtype or Q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"mqa_logits_half_gluon_bn_align expects fp16/bf16 Q/KV, got "
            f"{Q.dtype}/{KV.dtype}"
        )

    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]

    bn = get_half_bn_dispatch(seq_len, seq_len_kv, num_heads)

    block_m = int(block_m if block_m is not None else bn["block_m"])
    block_n = int(block_n if block_n is not None else bn["block_n"])
    num_stages = int(num_stages if num_stages is not None else bn["num_stages"])
    waves_per_eu = int(
        waves_per_eu if waves_per_eu is not None else bn.get("waves_per_eu", 0)
    )
    matrix_instr_nonkdim = int(
        matrix_instr_nonkdim if matrix_instr_nonkdim is not None else 16
    )
    kpack = int(kpack if kpack is not None else 2)
    schedule_hint = schedule_hint if schedule_hint is not None else "none"
    use_block_pingpong = (
        bool(use_block_pingpong) if use_block_pingpong is not None else False
    )

    if head_split_k is not None:
        num_splits = int(head_split_k)
    else:
        num_splits = int(bn["head_split_k"])
    while num_splits > 1 and num_heads % num_splits != 0:
        num_splits //= 2
    head_chunk = num_heads // num_splits

    preferred_wg = int(bn.get("w_group", 8) if w_group is None else w_group)
    if w_lds_pp:
        # WPP path requires even w_group.
        w_grp = clamp_gluon_w_group(preferred_wg, head_chunk)
    else:
        # Match half BN: allow w_group=1 (and any divisor of head_chunk).
        w_grp = preferred_wg
        while w_grp > 1 and head_chunk % w_grp != 0:
            w_grp //= 2
        if w_grp < 1 or head_chunk % w_grp != 0:
            w_grp = 1

    n_warps_cfg = int(num_warps if num_warps is not None else bn["num_warps"])
    if kv_tiles_n is not None:
        kv_tiles_n = int(kv_tiles_n)
        num_warps = block_n // (16 * kv_tiles_n)
    else:
        if n_warps_cfg > 0 and block_n % (16 * n_warps_cfg) == 0:
            kv_tiles_n = block_n // (16 * n_warps_cfg)
            if kv_tiles_n not in (1, 2, 4):
                kv_tiles_n = 1 if block_n <= 64 else 2
                num_warps = block_n // (16 * kv_tiles_n)
            else:
                num_warps = n_warps_cfg
        else:
            kv_tiles_n = 1 if block_n <= 64 else 2
            num_warps = block_n // (16 * kv_tiles_n)

    assert head_size & (head_size - 1) == 0
    assert head_size == 128, "half path currently expects head_dim=128"
    # Allow lightop-like BM=128 (``mqa_logits_128x128x32`` geometry on M/N).
    assert block_m in (16, 32, 64, 128) and block_m % 16 == 0
    assert kv_tiles_n in (1, 2, 4)
    assert block_n % (16 * kv_tiles_n) == 0
    assert num_warps == block_n // (16 * int(kv_tiles_n))

    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv)
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if num_splits > 1:
            logits.zero_()
    elif num_splits > 1:
        logits = torch.zeros(
            (seq_len, seq_len_kv), dtype=torch.float32, device=Q.device
        )
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv), dtype=torch.float32, device=Q.device
        )

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()
    assert stride_kv_d == 1

    grid_m = triton.cdiv(seq_len, block_m)
    grid_n = tiled_grid_n(seq_len_kv, block_n)
    hip = _filter_hip_launch_kwargs(
        {
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "kpack": kpack,
            "schedule_hint": schedule_hint,
            "use_block_pingpong": use_block_pingpong,
        }
    )
    _mqa_logits_half_gluon_bn_align_kernel[(grid_n, grid_m, num_splits)](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        ASSUME_KS_ZERO=assume_ks_zero,
        MFMA_VERSION=_mfma_version_for_arch(),
        KV_TILES_N=int(kv_tiles_n),
        NUM_WARPS=int(num_warps),
        W_GROUP=w_grp,
        NUM_SPLITS=num_splits,
        HEAD_CHUNK=head_chunk,
        KV_DIRECT_DOT=bool(kv_direct_dot),
        Q_DOUBLE_BUF=bool(q_double_buf),
        W_LDS_PP=bool(w_lds_pp),
        BN_STYLE_EPILOGUE=bool(bn_style_epilogue),
        **hip,
    )
    return logits
