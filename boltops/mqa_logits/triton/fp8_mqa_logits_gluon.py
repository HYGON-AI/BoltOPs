"""Gluon tile-parallel FP8 MQA logits prefill.

* Grid ``(ceil(N/128), ceil(M/BLOCK_M))`` — KV tile loaded once to MFMA B operand.
* Head loop: Q tile ``[BLOCK_M, HEAD_SIZE]`` double-buffered in LDS, ``mfma`` →
  ``[BLOCK_M, BLOCK_N]``, ReLU, multiply weight, accumulate.
* Weights prefetched in groups (``w_group``), with Q double-buffer in LDS.
* ``kv_scale`` and causal ``cu_ends`` mask applied at store.

Requires ``triton.experimental.gluon`` and native FP8 MMAC
(``arch_info.is_fp8_avail()``; e.g. gfx938). gfx936 can import Gluon but
cannot lower ``V_MMAC_F32_16X16X32_FP8_FP8``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.mqa_logits.triton.fp8_mqa_logits import (
    _filter_hip_launch_kwargs,
    fp8_mqa_logits,
)
from boltops.mqa_logits.triton.mqa_config_utils import (
    get_bn_tile_dispatch,
    get_fp8_mqa_gluon_launch_config,
    get_fp8_mqa_shape_dispatch,
    get_fp8_mqa_tiled_launch_config,
    tiled_block_m,
    tiled_block_n,
    tiled_grid_n,
)
from boltops.utility import arch_info

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _GLUON_AVAILABLE = True
except ImportError:
    gluon = None  # type: ignore[assignment]
    gl = None  # type: ignore[assignment]
    _GLUON_AVAILABLE = False


def gluon_language_available() -> bool:
    """True when ``triton.experimental.gluon`` imports (half Gluon paths)."""
    return bool(_GLUON_AVAILABLE)


def gluon_prefill_available() -> bool:
    """True when the FP8 Gluon prefill path can run on this device.

    Needs both the Gluon language package and native FP8 MMAC. Arches without
    ``is_fp8_avail()`` (e.g. gfx936) must use the float8 ``row`` / ``bn``
    emulation paths instead.
    """
    return bool(_GLUON_AVAILABLE) and bool(arch_info.is_fp8_avail())


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
    def _fp8_mqa_logits_gluon_tiled_kernel(
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
        # constexpr avoids i64 extsi/muli/trunci on KV offsets (AxisInfo contiguity).
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
        # MFMA N tiling: tiles_n=2 → 4 warps (default); tiles_n=1 → 8 warps.
        KV_TILES_N: tl.constexpr,
        NUM_WARPS: tl.constexpr,
        # Heads per weight LDS ping-pong group. Must divide HEAD_CHUNK and be even.
        W_GROUP: tl.constexpr,
        # Head split across grid-z. HEAD_CHUNK = NUM_HEADS // NUM_SPLITS.
        NUM_SPLITS: tl.constexpr,
        HEAD_CHUNK: tl.constexpr,
        # If true (ship default): BN-style epilogue (#mma where + convert out).
        # If false: store-layout masks + convert only acc.
        BN_STYLE_EPILOGUE: tl.constexpr,
    ):
        """KV once via LDS; Q double-buf + W_GROUP ping-pong (ship defaults)."""
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
            size_per_thread=[1, 16],
            threads_per_warp=[8, 8],
            warps_per_cta=[NumWarps, 1],
            order=[1, 0],
        )
        layout_kv: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[16, 1],
            threads_per_warp=[8, 8],
            warps_per_cta=[1, NumWarps],
            order=[0, 1],
        )
        shared_q: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=16, order=[1, 0]
        )
        shared_kv: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=16, order=[0, 1]
        )
        layout_store: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[2, 32],
            warps_per_cta=[NumWarps, 1],
            order=[1, 0],
        )

        # Gluon AMDMFMALayout requires (M, N, K) instr_shape.
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=MFMA_VERSION,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, KV_TILES_N],
        )

        mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mfma_layout, k_width=8
        )
        mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mfma_layout, k_width=8
        )
        offs_n_col = n_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        offs_m_row = m_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
        offs_n_kv = n_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout_kv))
        offs_d_kv = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, layout_kv))
        offs_m_q_row = m_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout_q))
        offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))
        n_mask = offs_n_col < seq_len_kv
        m_mask = offs_m_row < seq_len
        offs_m_st = m_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout_store))
        offs_n_st = n_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout_store))
        store_off = (
            offs_m_st[:, None] * stride_logits_s + offs_n_st[None, :] * stride_logits_k
        )
        store_mask = (offs_m_st[:, None] < seq_len) & (offs_n_st[None, :] < seq_len_kv)

        cu_end = gl.amd.cdna3.buffer_load(
            ptr=cu_end_ptr,
            offsets=offs_m_row.to(tl.int32),
            mask=m_mask,
        )
        cu_end = tl.minimum(cu_end, seq_len_kv)
        if ASSUME_KS_ZERO:
            cu_start = tl.zeros([BLOCK_M], dtype=tl.int32)
        else:
            cu_start = gl.amd.cdna3.buffer_load(
                ptr=cu_start_ptr,
                offsets=offs_m_row.to(tl.int32),
                mask=m_mask,
            )
            cu_start = tl.maximum(cu_start, 0)

        n_end = n_start + BLOCK_N
        tile_past_all = n_start >= tl.max(cu_end, axis=0)
        tile_before_all = False
        if not ASSUME_KS_ZERO:
            tile_before_all = n_end <= tl.min(cu_start, axis=0)
        if tile_past_all or tile_before_all:
            if NUM_SPLITS == 1 or split_id == 0:
                neg_inf = gl.full(
                    [BLOCK_M, BLOCK_N],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=mfma_layout,
                )
                gl.amd.cdna3.buffer_store(
                    gl.convert_layout(neg_inf, layout_store),
                    ptr=logits_ptr,
                    offsets=store_off,
                    mask=store_mask,
                )
            return

        kv_scales = gl.amd.cdna3.buffer_load(
            ptr=kv_scales_ptr,
            offsets=offs_n_col.to(tl.int32),
            mask=n_mask,
        )

        # KV: global -> LDS (swizzled) -> DotOperand B.
        kv = gl.amd.cdna3.buffer_load(
            ptr=KV_ptr,
            offsets=(
                offs_d_kv[:, None] * stride_kv_d + offs_n_kv[None, :] * stride_kv_s
            ).to(tl.int32),
        )
        smem_kv = gl.allocate_shared_memory(
            KV_ptr.type.element_ty, [HEAD_SIZE, BLOCK_N], shared_kv
        )
        smem_kv.store(kv)
        mfma_k = smem_kv.load(layout=mfma_layout_b)

        mfma_zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32, layout=mfma_layout)
        acc = mfma_zero

        q0 = gl.amd.cdna3.buffer_load(
            ptr=Q_ptr,
            offsets=(
                offs_m_q_row[:, None] * stride_q_s
                + start_h * stride_q_h
                + offs_d_q[None, :] * stride_q_d
            ).to(tl.int32),
        )

        # Q double-buf with separate odd allocation (avoids false membar aliasing).
        # W_GROUP ping-pong with derived [1,M] LDS layout (former w_cache_mode=3).
        w_row_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
        smem_q_even = gl.allocate_shared_memory(
            Q_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE], shared_q
        )
        smem_q_odd = gl.allocate_shared_memory(
            Q_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE], shared_q
        )
        smem_q_even.store(q0)
        q_pref = q0
        if HEAD_CHUNK > 1:
            q_pref = gl.amd.cdna3.buffer_load(
                ptr=Q_ptr,
                offsets=(
                    offs_m_q_row[:, None] * stride_q_s
                    + (start_h + 1) * stride_q_h
                    + offs_d_q[None, :] * stride_q_d
                ).to(tl.int32),
            )

        if W_GROUP < 8:
            WG_SPT: gl.constexpr = 1
            WG_TPW: gl.constexpr = W_GROUP
            M_TPW: gl.constexpr = 64 // W_GROUP
        else:
            WG_SPT: gl.constexpr = W_GROUP // 8
            WG_TPW: gl.constexpr = 8
            M_TPW: gl.constexpr = 8
        layout_w_hm: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[WG_SPT, 1],
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
        offs_h_wg = gl.arange(0, W_GROUP, layout=gl.SliceLayout(1, layout_w_hm))
        offs_m_wg = m_start + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(0, layout_w_hm)
        )
        w_g0 = gl.amd.cdna3.buffer_load(
            ptr=weights_ptr,
            offsets=(
                (start_h + offs_h_wg)[:, None] * stride_w_h
                + offs_m_wg[None, :] * stride_w_s
            ).to(tl.int32),
        ).to(gl.float32)
        smem_w.index(0).store(w_g0)

        n_groups: gl.constexpr = HEAD_CHUNK // W_GROUP
        tl.static_assert(W_GROUP % 2 == 0)
        tl.static_assert(HEAD_CHUNK % W_GROUP == 0)
        for g in tl.range(0, n_groups - 1):
            buf = g & 1
            offs_h_n = start_h + (g + 1) * W_GROUP + offs_h_wg
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
                        offs_m_q_row[:, None] * stride_q_s
                        + h_pref * stride_q_h
                        + offs_d_q[None, :] * stride_q_d
                    ).to(tl.int32),
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
                    offs_m_q_row[:, None] * stride_q_s
                    + h_pref * stride_q_h
                    + offs_d_q[None, :] * stride_q_d
                ).to(tl.int32),
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

        acc = acc * kv_scales[None, :]

        if BN_STYLE_EPILOGUE:
            # Reuse prologue cu_start/cu_end on #mma; convert out once.
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
            # Store-layout: reload KS/KE, convert only acc.
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
                add_mask = store_mask & in_window_st
                gl.amd.cdna3.buffer_atomic_add(
                    logits_ptr, store_off, acc_st, mask=add_mask
                )
                if split_id == 0:
                    neg_inf = gl.full(
                        [BLOCK_M, BLOCK_N],
                        float("-inf"),
                        dtype=tl.float32,
                        layout=layout_store,
                    )
                    oob = store_mask & (~in_window_st)
                    gl.amd.cdna3.buffer_store(
                        neg_inf, ptr=logits_ptr, offsets=store_off, mask=oob
                    )



def _mfma_version_for_arch() -> int:
    arch = arch_info.get_arch()
    if arch in ("gfx950", "gfx946"):
        return 4
    return 3


def fp8_mqa_logits_gluon(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    kpack: Optional[int] = None,
    schedule_hint: Optional[str] = None,
    use_block_pingpong: Optional[bool] = None,
    force_row_kernel: bool = False,
    force_gluon_kernel: bool = False,
    assume_ks_zero: bool = False,
    kv_tiles_n: Optional[int] = None,
    w_group: Optional[int] = None,
    head_split_k: Optional[int] = None,
    bn_style_epilogue: bool = True,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gluon tiled FP8 MQA logits prefill.

    Ship path: KV via LDS, Q double-buffer, weight ``W_GROUP`` ping-pong with
    derived LDS layout (former ``q_double_buf=2`` / ``w_cache_mode=3``).

    ``force_gluon_kernel=True``: skip shape dispatch (public ``backend="gluon"``).
    ``w_group`` / ``kv_tiles_n``: ``None`` loads shipped Gluon JSON; ``w_group``
    is clamped to divide ``HEAD_CHUNK``.
    ``bn_style_epilogue=True`` (default): reuse prologue KS/KE on ``#mma`` then
    convert out; set False to reload masks on store layout and convert only ``acc``.
    ``D_out``: optional preallocated ``[seq_len, seq_len_kv]`` float32 buffer.
    """
    if not _GLUON_AVAILABLE:
        if force_row_kernel:
            return fp8_mqa_logits(
                Q,
                KV,
                kv_scales,
                weights,
                cu_starts,
                cu_ends,
                assume_ks_zero=assume_ks_zero,
                D_out=D_out,
                auto_bn_tile=False,
            )
        seq_len, num_heads, _ = Q.shape
        seq_len_kv = KV.shape[0]
        disp = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
        return fp8_mqa_logits(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            num_warps=num_warps if num_warps is not None else int(disp["num_warps"]),
            num_stages=num_stages if num_stages is not None else int(disp["num_stages"]),
            waves_per_eu=(
                waves_per_eu if waves_per_eu is not None else int(disp["waves_per_eu"])
            ),
            matrix_instr_nonkdim=(
                matrix_instr_nonkdim
                if matrix_instr_nonkdim is not None
                else int(disp["matrix_instr_nonkdim"])
            ),
            kpack=kpack if kpack is not None else int(disp["kpack"]),
            schedule_hint=(
                schedule_hint
                if schedule_hint is not None
                else str(disp["schedule_hint"])
            ),
            use_block_pingpong=(
                use_block_pingpong
                if use_block_pingpong is not None
                else bool(disp["use_block_pingpong"])
            ),
            assume_ks_zero=assume_ks_zero,
            D_out=D_out,
            head_split_k=int(disp["head_split_k"]),
            bn_tile=int(block_n) if block_n is not None else int(disp["bn_tile"]),
            block_m_tile=int(block_m) if block_m is not None else int(disp["block_m_tile"]),
            auto_bn_tile=False,
        )

    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]

    shape_disp = get_fp8_mqa_shape_dispatch(
        seq_len, seq_len_kv, num_heads, gluon_available=True
    )
    if force_gluon_kernel and force_row_kernel:
        raise ValueError("force_gluon_kernel and force_row_kernel are mutually exclusive")
    if force_gluon_kernel:
        path = "gluon"
    elif force_row_kernel:
        path = "row"
    else:
        path = str(shape_disp["path"])
    if path not in ("row", "gluon"):
        path = "gluon"

    if path == "row":
        row_kw: dict = {
            "assume_ks_zero": assume_ks_zero,
            "D_out": D_out,
            "auto_bn_tile": False,
            "head_split_k": int(shape_disp.get("head_split_k", 1)),
        }
        for key in ("block_kv", "num_warps", "num_stages"):
            if key in shape_disp:
                row_kw[key] = int(shape_disp[key])
        return fp8_mqa_logits(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            **row_kw,
        )

    cfg = get_fp8_mqa_tiled_launch_config(seq_len, seq_len_kv)
    gcfg = get_fp8_mqa_gluon_launch_config(seq_len, seq_len_kv, num_heads)
    block_m = (
        block_m
        if block_m is not None
        else int(gcfg["block_m"]) if "block_m" in gcfg else tiled_block_m(seq_len)
    )
    block_n = (
        block_n
        if block_n is not None
        else int(gcfg["block_n"]) if "block_n" in gcfg else tiled_block_n()
    )
    num_stages = (
        num_stages
        if num_stages is not None
        else int(gcfg["num_stages"]) if "num_stages" in gcfg else int(cfg["num_stages"])
    )
    waves_per_eu = (
        waves_per_eu
        if waves_per_eu is not None
        else (
            int(gcfg["waves_per_eu"])
            if "waves_per_eu" in gcfg
            else int(cfg["waves_per_eu"])
        )
    )
    matrix_instr_nonkdim = (
        matrix_instr_nonkdim
        if matrix_instr_nonkdim is not None
        else int(cfg["matrix_instr_nonkdim"])
    )
    kpack = (
        kpack
        if kpack is not None
        else int(gcfg["kpack"]) if "kpack" in gcfg else int(cfg["kpack"])
    )
    schedule_hint = (
        schedule_hint if schedule_hint is not None else str(cfg["schedule_hint"])
    )
    use_block_pingpong = (
        use_block_pingpong
        if use_block_pingpong is not None
        else bool(cfg["use_block_pingpong"])
    )
    kv_tiles_n = int(gcfg["kv_tiles_n"] if kv_tiles_n is None else kv_tiles_n)
    w_group = int(gcfg["w_group"] if w_group is None else w_group)
    if head_split_k is not None:
        num_splits = int(head_split_k)
    elif "head_split_k" in gcfg:
        num_splits = int(gcfg["head_split_k"])
    else:
        num_splits = int(shape_disp.get("head_split_k", 1))
    assert num_splits >= 1, f"head_split_k must be >= 1, got {num_splits}"
    while num_splits > 1 and num_heads % num_splits != 0:
        num_splits //= 2
    head_chunk = num_heads // num_splits

    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    assert block_m in (16, 32), f"gluon tiled BLOCK_M must be 16 or 32, got {block_m}"
    assert kv_tiles_n in (1, 2), f"kv_tiles_n must be 1 or 2, got {kv_tiles_n}"

    from boltops.mqa_logits.triton.mqa_config_utils import clamp_gluon_w_group

    w_grp = clamp_gluon_w_group(int(w_group), head_chunk)
    assert w_grp >= 2 and w_grp % 2 == 0, f"w_group must be even and >=2, got {w_grp}"
    assert head_chunk % w_grp == 0, (
        f"HEAD_CHUNK % w_group == 0 required, got HEAD_CHUNK={head_chunk} w_group={w_grp}"
    )
    assert num_heads >= 1, "num_heads must be >= 1"
    num_warps = block_n // (16 * int(kv_tiles_n))

    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv), (
            f"D_out shape {tuple(D_out.shape)} != {(seq_len, seq_len_kv)}"
        )
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if num_splits > 1:
            logits.zero_()
    elif num_splits > 1:
        logits = torch.zeros(
            (seq_len, seq_len_kv),
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
    assert stride_kv_d == 1, f"expected contiguous KV head_dim, stride_d={stride_kv_d}"

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
    _fp8_mqa_logits_gluon_tiled_kernel[(grid_n, grid_m, num_splits)](
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
        BN_STYLE_EPILOGUE=bool(bn_style_epilogue),
        **hip,
    )
    return logits
