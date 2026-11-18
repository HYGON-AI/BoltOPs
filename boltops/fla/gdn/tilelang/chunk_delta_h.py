# Copyright (c) Tile-AI.
# SPDX-License-Identifier: MIT
# Adapted from TileLang examples/gdn/example_chunk_delta_h.py.
"""TileLang kernels for the GDN chunk state recurrence."""

import functools
import os
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from boltops.fla.common.triton.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

from .common import require_contiguous, require_first_version, tilelang_dtype

_LOG2E = 1.4426950408889634
_PRODUCER_THREADS = 256
_TOTAL_THREADS = 512
_TOTAL_WAVES = _TOTAL_THREADS // 64
_WASP_CONSUMER_WAVES = (_TOTAL_THREADS - _PRODUCER_THREADS) // 64
_WASP_PRODUCER_WAVES = _PRODUCER_THREADS // 64
_WASP_U_FREE_PING = 0
_WASP_U_READY_PING = 1
_WASP_K_FREE_PING = 2
_WASP_K_READY_PING = 3
_WASP_W_FREE_PING = 4
_WASP_W_READY_PING = 5
_WASP_STATE_FREE_PING = 6
_WASP_STATE_READY_PING = 7
_WASP_V_NEW_FREE_PING = 8
_WASP_V_NEW_READY_PING = 9
_WASP_INIT_EBAR_ID = 0
_WASP_CONSUMER_EBAR_ID = 1
# The two four-wave WDRA branches must have a four-VGPR-aligned average:
# Fixed: (48 + 192) / 2 = 120; varlen: (48 + 208) / 2 = 128.
# Reserve enough producer registers for BV64 address setup and output saves.
_WASP_PRODUCER_MAX_NREG = 48
_WASP_CONSUMER_MAX_NREG = 192
_WASP_VARLEN_PRODUCER_MAX_NREG = 48
_WASP_VARLEN_CONSUMER_MAX_NREG = 208

@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def _build_chunk_h_copy_pipeline(
    batch: int,
    heads: int,
    kv_heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    block_v: int = 32,
    threads: int = 256,
):
    """Build the explicit one-stage copy-pipeline kernel.

    The recurrent state is stored transposed as ``S_i = H_i.T`` in VK order.
    For each chunk ``i`` this computes

        V_new_i = U_i - W_i @ S_i.T
        V_gated_i[t] = V_new_i[t] * exp(g_i[last] - g_i[t])
        S_{i+1} = exp(g_i[last]) * S_i + V_gated_i.T @ K_i

    Each program owns one [block_v, dim_k] row slice of ``S``. The 256-thread
    block contains four 64-thread waves. Gemm1 uses the native LIT-only
    accumulator layout, whose lane-local outputs are contiguous in K. They are
    published as the transposed H[K, V] through a K-major LDS layout, so Gemm0
    can consume H directly while the public h output remains in VK order.
    """
    seq_len = T.dynamic("seq_len")
    full_chunks = seq_len // chunk_size
    tail_size = seq_len % chunk_size
    num_chunks = T.ceildiv(seq_len, chunk_size)
    state_shape = (
        (heads, dim_v, dim_k)
        if transpose_state_layout
        else (heads, dim_k, dim_v)
    )
    chunk_state_shape = (
        (batch, num_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, num_chunks, heads, dim_k, dim_v)
    )
    # No-padding K-major cyclic wrap. Each row remains exactly 128 BF16 values.
    # A K8 block is 16B. Eight row phases spread the LIT producer's sixteen
    # row writes over all eight K8 bank classes while preserving contiguous,
    # aligned K8 blocks for Gemm0's ds_read_b128 loads:
    #   physical(iv, ik) =
    #       (iv, ((ik // 8) xor (iv % 8)) * 8 + ik % 8).
    state_shared_h_k_major_layout = T.Layout(
        (block_v, dim_k),
        lambda iv, ik: (
            iv,
            ((ik // 8) ^ (iv % 8)) * 8 + ik % 8,
        ),
    )
    # Gemm1 A is logically [K, V]. Keep every V8 segment physically contiguous
    # for ds_read_m32x16_b16, while rotating the four V8 blocks across K-row
    # phases to spread the ordinary fragment-to-LDS producer stores. Unlike
    # k_shared, this buffer needs no async-copy transaction or idxen wrap.
    v_new_shared_v8_phase_layout = T.Layout(
        (chunk_size, block_v),
        lambda it, iv: (
            it,
            ((iv // 8) ^ (it % 4)) * 8 + iv % 8,
        ),
    )

    @T.macro
    def run_steady_chunk(
        cid,
        w_current,
        state_operand,
        launch_following_w,
        w,
        u,
        k,
        g,
        h,
        v_new,
        bid,
        hid,
        v_offset,
        v_new_fragment,
        u_shared,
        u_fragment,
        v_new_shared,
        v_new_operand,
        g_fragment,
        state_fragment,
        k_shared,
        k_fragment,
        state_shared_h,
        g_last,
    ):
        base = cid * chunk_size
        next_base = base + chunk_size

        # U_i ready. K_i remains outstanding.
        T.ptx_wait_group(1)
        T.sync_warp()

        T.copy(u_shared, u_fragment)
        T.s_waitcnt(2, "lgkmcnt")

        T.sched_barrier()
        T.gemm(
            w_current,
            state_operand,
            v_new_fragment,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
            clear_accum=True,
            k_pack=2,
        )
        T.sched_barrier()

        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()
        T.async_copy(
            u[
                bid,
                next_base : next_base + chunk_size,
                hid,
                v_offset : v_offset + block_v,
            ],
            u_shared,
        )

        # Gemm0 has released w_current. Reuse that register slot to carry
        # W_{i+2} directly from global memory across the rest of this chunk.
        if launch_following_w:
            T.copy(
                w[
                    bid,
                    base + 2 * chunk_size : base + 3 * chunk_size,
                    hid,
                    0:dim_k,
                ],
                w_current,
            )
            T.sched_barrier()

        for it, iv in T.Parallel(chunk_size, block_v):
            v_new_fragment[it, iv] = (
                u_fragment[it, iv] - v_new_fragment[it, iv]
            )

        if save_new_value:
            T.copy(
                v_new_fragment,
                v_new[
                    bid,
                    base : base + chunk_size,
                    hid,
                    v_offset : v_offset + block_v,
                ],
            )
        T.sched_barrier()

        # K_i is ready, U_{i+1} remains outstanding
        T.ptx_wait_group(1)
        T.sync_warp()

        # k_fragment outstanding
        T.copy(k_shared, k_fragment)
        T.s_waitcnt(4, "lgkmcnt")
        T.sched_barrier()

        g_last = g[bid, base + chunk_size - 1, hid]
        for it, iv in T.Parallel(chunk_size, block_v):
            diff = g_last - g_fragment[it, iv]
            v_new_fragment[it, iv] = T.if_then_else(
                diff <= 0,
                v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                T.float32(0.0),
            )
        g_last = T.exp2(g_last * _LOG2E)
        for iv, ik in T.Parallel(block_v, dim_k):
            state_fragment[iv, ik] *= g_last

        T.copy(v_new_fragment, v_new_shared) # ds_write -> LDS

        for it, iv in T.Parallel(chunk_size, block_v):
            g_fragment[it, iv] = g[bid, next_base + it, hid]
        T.sched_barrier()

        # k_fragment, v_new_shared ready
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        T.copy(v_new_shared, v_new_operand)

        # The preceding wait/barrier completed all K fragment reads before
        # reusing k_shared. Keep v_new_operand's LDS reads outstanding here.
        T.async_copy(
            k[
                bid,
                next_base : next_base + chunk_size,
                hid // (heads // kv_heads),
                0:dim_k,
            ],
            k_shared,
        )
        T.sched_barrier()
        T.gemm(
            v_new_operand,
            k_fragment,
            state_fragment,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=False,
            k_pack=1,
        )
        T.sched_barrier()
        T.copy(state_fragment, state_shared_h)

        # state_{i+1} ready
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        T.copy(state_shared_h, state_operand)
        T.copy(
            state_shared_h,
            h[
                bid,
                cid + 1,
                hid,
                v_offset : v_offset + block_v,
                0:dim_k,
            ],
        )
        T.sched_barrier()

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_copy_pipeline_kernel(
        k: T.Tensor([batch, seq_len, kv_heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([batch], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor(
            [state_rows if inplace_final_state else batch, *state_shape],
            "float32",
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), batch * heads, threads=threads
        ) as (vid, bbh):
            bid, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            # The FP32 accumulator and its BF16 LDS publication are both
            # logical S[V, K]; the annotated physical layout keeps K8 packed.
            state_shared_h = T.alloc_shared((block_v, dim_k), qkv_dtype)
            state_fragment = T.alloc_fragment((block_v, dim_k), "float32")
            u_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            u_fragment = T.alloc_fragment((chunk_size, block_v), qkv_dtype)
            w_operand_0 = T.alloc_fragment((chunk_size, dim_k), qkv_dtype)
            w_operand_1 = T.alloc_fragment((chunk_size, dim_k), qkv_dtype)
            state_operand = T.alloc_fragment((block_v, dim_k), qkv_dtype)
            v_new_fragment = T.alloc_fragment((chunk_size, block_v), "float32")
            v_new_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            v_new_operand = T.alloc_fragment((chunk_size, block_v), qkv_dtype)
            # K_i is copied completely into k_fragment before K_{i+1}
            # overwrites this single LDS tile.
            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            k_fragment = T.alloc_fragment((chunk_size, dim_k), qkv_dtype)
            # G uses one register fragment. After G_i is consumed, the same
            # registers are overwritten by G_{i+1} and kept across Gemm1.
            g_fragment = T.alloc_fragment((chunk_size, block_v), gate_dtype)
            g_last = T.alloc_var("float32")
            state_idx = T.alloc_var("int32")

            T.annotate_layout(
                {
                    state_shared_h: state_shared_h_k_major_layout,
                    v_new_shared: v_new_shared_v8_phase_layout,
                }
            )
            # T.use_swizzle(10)

            state_idx = initial_state_indices[bid]
            for iv, ik in T.Parallel(block_v, dim_k):
                if (
                    use_initial_state
                    and state_idx >= 0
                    and state_idx < state_rows
                ):
                    if transpose_state_layout:
                        state_fragment[iv, ik] = initial_state[
                            state_idx, hid, v_offset + iv, ik
                        ]
                    else:
                        state_fragment[iv, ik] = initial_state[
                            state_idx, hid, ik, v_offset + iv
                        ]
                else:
                    state_fragment[iv, ik] = T.float32(0.0)
            T.copy(state_fragment, state_shared_h)

            # W is carried directly in two register fragments. Gemm0 uses a
            # FullRow wave partition so every W element is fetched only once.
            T.copy(
                w[bid, 0:chunk_size, hid, 0:dim_k],
                w_operand_0,
            )
            T.sched_barrier()

            if full_chunks > 1:
                T.copy(
                    w[bid, chunk_size : 2 * chunk_size, hid, 0:dim_k],
                    w_operand_1,
                )
                T.sched_barrier()

            # prologue U0
            T.async_copy(
                u[
                    bid,
                    0:chunk_size,
                    hid,
                    v_offset : v_offset + block_v,
                ],
                u_shared,
            )
            for it, iv in T.Parallel(chunk_size, block_v):
                g_fragment[it, iv] = g[bid, it, hid]
            T.s_waitcnt(0, "lgkmcnt")
            T.sync_warp()
            T.copy(
                state_shared_h,
                h[
                    bid,
                    0,
                    hid,
                    v_offset : v_offset + block_v,
                    0:dim_k,
                ],
            )
            T.copy(state_shared_h, state_operand)
            T.sched_barrier()

            # Prologue K0
            T.async_copy(
                k[bid, 0:chunk_size, hid // (heads // kv_heads), 0:dim_k],
                k_shared,
            )

            # Each hot-loop iteration expands two chunks with statically
            # swapped operand slots; no cid-parity branch is generated.
            pipeline_chunks = T.max(full_chunks - 2, 0)
            pair_count = pipeline_chunks // 2
            for pair in T.Pipelined(pair_count, num_stages=0):
                cid = pair * 2
                run_steady_chunk(
                    cid,
                    w_operand_0,
                    state_operand,
                    True,
                    w, u, k, g, h, v_new, bid, hid, v_offset,
                    v_new_fragment, u_shared, u_fragment,
                    v_new_shared, v_new_operand, g_fragment, state_fragment,
                    k_shared, k_fragment, state_shared_h, g_last,
                )
                run_steady_chunk(
                    cid + 1,
                    w_operand_1,
                    state_operand,
                    True,
                    w, u, k, g, h, v_new, bid, hid, v_offset,
                    v_new_fragment, u_shared, u_fragment,
                    v_new_shared, v_new_operand, g_fragment, state_fragment,
                    k_shared, k_fragment, state_shared_h, g_last,
                )

            # Handle at most one unpaired pipelined chunk and the final steady
            # chunk outside the hot pair loop. The selected epilogue slot is
            # therefore static within each branch.
            if full_chunks > 1:
                if pipeline_chunks % 2 != 0:
                    cid = pair_count * 2
                    run_steady_chunk(
                        cid,
                        w_operand_0,
                        state_operand,
                        True,
                        w, u, k, g, h, v_new, bid, hid, v_offset,
                        v_new_fragment, u_shared, u_fragment,
                        v_new_shared, v_new_operand, g_fragment, state_fragment,
                        k_shared, k_fragment, state_shared_h, g_last,
                    )
                    run_steady_chunk(
                        cid + 1,
                        w_operand_1,
                        state_operand,
                        False,
                        w, u, k, g, h, v_new, bid, hid, v_offset,
                        v_new_fragment, u_shared, u_fragment,
                        v_new_shared, v_new_operand, g_fragment, state_fragment,
                        k_shared, k_fragment, state_shared_h, g_last,
                    )
                else:
                    cid = pair_count * 2
                    run_steady_chunk(
                        cid,
                        w_operand_0,
                        state_operand,
                        False,
                        w, u, k, g, h, v_new, bid, hid, v_offset,
                        v_new_fragment, u_shared, u_fragment,
                        v_new_shared, v_new_operand, g_fragment, state_fragment,
                        k_shared, k_fragment, state_shared_h, g_last,
                    )

            # Epilogue: U_last and K_last are the retained async groups. h_last
            # and both complete Gemm0 operands were published by the prologue
            # or the final steady chunk.
            last_cid = full_chunks - 1
            last_base = last_cid * chunk_size
            if full_chunks == 1:
                T.gemm(
                    w_operand_0,
                    state_operand,
                    v_new_fragment,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                    k_pack=2,
                )
            else:
                if pipeline_chunks % 2 != 0:
                    T.gemm(
                        w_operand_0,
                        state_operand,
                        v_new_fragment,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                        k_pack=2,
                    )
                else:
                    T.gemm(
                        w_operand_1,
                        state_operand,
                        v_new_fragment,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                        k_pack=2,
                    )
            T.ptx_wait_group(1)
            T.sync_warp()
            T.copy(u_shared, u_fragment)
            for it, iv in T.Parallel(chunk_size, block_v):
                v_new_fragment[it, iv] = (
                    u_fragment[it, iv] - v_new_fragment[it, iv]
                )
            if save_new_value:
                T.copy(
                    v_new_fragment,
                    v_new[
                        bid,
                        last_base : last_base + chunk_size,
                        hid,
                        v_offset : v_offset + block_v,
                    ],
                )

            g_last = g[bid, last_base + chunk_size - 1, hid]
            for it, iv in T.Parallel(chunk_size, block_v):
                diff = g_last - g_fragment[it, iv]
                v_new_fragment[it, iv] = T.if_then_else(
                    diff <= 0,
                    v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                    T.float32(0.0),
                )

            g_last = T.exp2(g_last * _LOG2E)
            for iv, ik in T.Parallel(block_v, dim_k):
                state_fragment[iv, ik] *= g_last
            T.copy(v_new_fragment, v_new_shared)
            T.s_waitcnt(0, "lgkmcnt")
            T.sync_warp()
            T.copy(v_new_shared, v_new_operand)

            T.ptx_wait_group(0)
            T.sync_warp()
            T.copy(k_shared, k_fragment)
            T.gemm(
                v_new_operand,
                k_fragment,
                state_fragment,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=False,
                k_pack=1,
            )
            T.copy(state_fragment, state_shared_h)

            # A non-aligned final chunk uses predicated ordinary copies. Full
            # chunks stay on the unchanged async-copy fast path above.
            if tail_size > 0:
                tail_cid = full_chunks
                tail_base = full_chunks * chunk_size

                T.s_waitcnt(0, "lgkmcnt")
                T.sync_warp()
                T.copy(
                    state_shared_h,
                    h[
                        bid,
                        tail_cid,
                        hid,
                        v_offset : v_offset + block_v,
                        0:dim_k,
                    ],
                )
                for it, ik in T.Parallel(chunk_size, dim_k):
                    if it < tail_size:
                        w_operand_0[it, ik] = w[
                            bid, tail_base + it, hid, ik
                        ]
                        k_shared[it, ik] = k[
                            bid, tail_base + it, hid // (heads // kv_heads), ik
                        ]
                    else:
                        w_operand_0[it, ik] = T.cast(0, qkv_dtype)
                        k_shared[it, ik] = T.cast(0, qkv_dtype)
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        u_shared[it, iv] = u[
                            bid, tail_base + it, hid, v_offset + iv
                        ]
                        g_fragment[it, iv] = g[
                            bid, tail_base + it, hid
                        ]
                    else:
                        u_shared[it, iv] = T.cast(0, qkv_dtype)
                        g_fragment[it, iv] = T.float32(0.0)

                T.s_waitcnt(0, "lgkmcnt")
                T.sync_warp()
                T.gemm(
                    w_operand_0,
                    state_shared_h,
                    v_new_fragment,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                    k_pack=2,
                )
                T.copy(u_shared, u_fragment)
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        v_new_fragment[it, iv] = (
                            u_fragment[it, iv] - v_new_fragment[it, iv]
                        )
                    else:
                        v_new_fragment[it, iv] = T.float32(0.0)

                if save_new_value:
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            v_new[
                                bid, tail_base + it, hid, v_offset + iv
                            ] = T.cast(v_new_fragment[it, iv], qkv_dtype)

                g_last = g[bid, tail_base + tail_size - 1, hid]
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        diff = g_last - g_fragment[it, iv]
                        v_new_fragment[it, iv] = T.if_then_else(
                            diff <= 0,
                            v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                            T.float32(0.0),
                        )

                g_last = T.exp2(g_last * _LOG2E)
                for iv, ik in T.Parallel(block_v, dim_k):
                    state_fragment[iv, ik] *= g_last
                T.copy(v_new_fragment, v_new_shared)
                T.s_waitcnt(0, "lgkmcnt")
                T.sync_warp()
                T.copy(v_new_shared, v_new_operand)
                T.copy(k_shared, k_fragment)
                T.gemm(
                    v_new_operand,
                    k_fragment,
                    state_fragment,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=False,
                    k_pack=1,
                )
                T.copy(state_fragment, state_shared_h)

            if store_final_state:
                if inplace_final_state:
                    if state_idx >= 0 and state_idx < state_rows:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            if transpose_state_layout:
                                final_state[
                                    state_idx, hid, v_offset + iv, ik
                                ] = state_fragment[iv, ik]
                            else:
                                final_state[
                                    state_idx, hid, ik, v_offset + iv
                                ] = state_fragment[iv, ik]
                else:
                    for iv, ik in T.Parallel(block_v, dim_k):
                        if transpose_state_layout:
                            final_state[bid, hid, v_offset + iv, ik] = (
                                state_fragment[iv, ik]
                            )
                        else:
                            final_state[bid, hid, ik, v_offset + iv] = (
                                state_fragment[iv, ik]
                            )

    return chunk_gated_delta_rule_fwd_h_copy_pipeline_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _build_chunk_h_copy_pipeline_varlen(
    batch: int,
    heads: int,
    kv_heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    block_v: int = 32,
    threads: int = 256,
):
    """Build the fixed copy pipeline specialized for packed varlen.

    The recurrent state is stored transposed as ``S_i = H_i.T`` in VK order.
    For each chunk ``i`` this computes

        V_new_i = U_i - W_i @ S_i.T
        V_gated_i[t] = V_new_i[t] * exp(g_i[last] - g_i[t])
        S_{i+1} = exp(g_i[last]) * S_i + V_gated_i.T @ K_i

    Each program owns one [block_v, dim_k] row slice of ``S``. The 256-thread
    block contains four 64-thread waves. Gemm1 uses the native LIT-only
    accumulator layout, whose lane-local outputs are contiguous in K. They are
    published as the transposed H[K, V] through a K-major LDS layout, so Gemm0
    can consume H directly while the public h output remains in VK order.
    """
    seq_len = T.dynamic("seq_len")
    num_sequences = T.dynamic("num_sequences")
    num_chunks = T.dynamic("num_chunks")
    cu_size = T.dynamic("cu_size")
    chunk_offsets_size = T.dynamic("chunk_offsets_size")
    state_shape = (
        (heads, dim_v, dim_k)
        if transpose_state_layout
        else (heads, dim_k, dim_v)
    )
    chunk_state_shape = (
        (batch, num_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, num_chunks, heads, dim_k, dim_v)
    )
    # No-padding K-major cyclic wrap. Each row remains exactly 128 BF16 values.
    # A K8 block is 16B. Eight row phases spread the LIT producer's sixteen
    # row writes over all eight K8 bank classes while preserving contiguous,
    # aligned K8 blocks for Gemm0's ds_read_b128 loads:
    #   physical(iv, ik) =
    #       (iv, ((ik // 8) xor (iv % 8)) * 8 + ik % 8).
    state_shared_h_k_major_layout = T.Layout(
        (block_v, dim_k),
        lambda iv, ik: (
            iv,
            ((ik // 8) ^ (iv % 8)) * 8 + ik % 8,
        ),
    )
    # Gemm1 A is logically [K, V]. Keep every V8 segment physically contiguous
    # for ds_read_m32x16_b16, while rotating the four V8 blocks across K-row
    # phases to spread the ordinary fragment-to-LDS producer stores. Unlike
    # k_shared, this buffer needs no async-copy transaction or idxen wrap.
    v_new_shared_v8_phase_layout = T.Layout(
        (chunk_size, block_v),
        lambda it, iv: (
            it,
            ((iv // 8) ^ (it % 4)) * 8 + iv % 8,
        ),
    )

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_copy_pipeline_varlen_kernel(
        k: T.Tensor([batch, seq_len, kv_heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([num_sequences], "int32"),
        cu_seqlens: T.Tensor([cu_size], "int32"),
        chunk_offsets: T.Tensor([chunk_offsets_size], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor(
            [
                state_rows if inplace_final_state else num_sequences,
                *state_shape,
            ],
            "float32",
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), num_sequences * heads, threads=threads
        ) as (vid, bbh):
            # Apply the block remap before deriving sequence/head coordinates.
            # T.use_swizzle(10)
            sequence_id, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            data_bid = 0
            token_begin = cu_seqlens[sequence_id]
            actual_seq_len = cu_seqlens[sequence_id + 1] - token_begin
            chunk_begin = chunk_offsets[sequence_id]
            full_chunks = actual_seq_len // chunk_size
            tail_size = actual_seq_len % chunk_size
            # The FP32 accumulator and its BF16 LDS publication are both
            # logical S[V, K]; the annotated physical layout keeps K8 packed.
            state_shared_h = T.alloc_shared((block_v, dim_k), qkv_dtype)
            state_fragment = T.alloc_fragment((block_v, dim_k), "float32")
            u_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            u_fragment = T.alloc_fragment((chunk_size, block_v), "float32")
            w_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            v_new_fragment = T.alloc_fragment((chunk_size, block_v), "float32")
            v_new_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            v_new_operand = T.alloc_fragment((chunk_size, block_v), qkv_dtype)
            # Ping-pong K in LDS. Gemm1 consumes the single static fragment,
            # while K_{i+1} is written to the alternate LDS tile immediately
            # after K_i has been copied out of its tile.
            k_shared = T.alloc_shared((2, chunk_size, dim_k), qkv_dtype)
            k_fragment = T.alloc_fragment((chunk_size, dim_k), qkv_dtype)
            # G uses one register fragment. After G_i is consumed, the same
            # registers are overwritten by G_{i+1} and kept across Gemm1.
            g_fragment = T.alloc_fragment((chunk_size, block_v), gate_dtype)
            g_last = T.alloc_var("float32")
            state_idx = T.alloc_var("int32")

            T.annotate_layout(
                {
                    state_shared_h: state_shared_h_k_major_layout,
                    v_new_shared: v_new_shared_v8_phase_layout,
                }
            )
            state_idx = initial_state_indices[sequence_id]
            for iv, ik in T.Parallel(block_v, dim_k):
                if (
                    use_initial_state
                    and state_idx >= 0
                    and state_idx < state_rows
                ):
                    if transpose_state_layout:
                        state_fragment[iv, ik] = initial_state[
                            state_idx, hid, v_offset + iv, ik
                        ]
                    else:
                        state_fragment[iv, ik] = initial_state[
                            state_idx, hid, ik, v_offset + iv
                        ]
                else:
                    state_fragment[iv, ik] = T.float32(0.0)
            T.copy(state_fragment, state_shared_h)

            if full_chunks > 0:
                # Reproduce the automatic one-stage schedule explicitly. W/K and
                # steady-state U use ordered async groups; G is prefetched into a
                # single long-lived register fragment after its prior use.
                T.async_copy(
                    w[data_bid, token_begin : token_begin + chunk_size, hid, 0:dim_k],
                    w_shared,
                )
                T.async_copy(
                    u[
                        data_bid,
                        token_begin : token_begin + chunk_size,
                        hid,
                        v_offset : v_offset + block_v,
                    ],
                    u_shared,
                )
                for it, iv in T.Parallel(chunk_size, block_v):
                    g_fragment[it, iv] = g[data_bid, token_begin + it, hid]
                T.async_copy(
                    k[data_bid, token_begin : token_begin + chunk_size, hid // (heads // kv_heads), 0:dim_k],
                    k_shared[0, :, :],
                )

                for cid in T.Pipelined(full_chunks - 1, num_stages=0):
                    base = cid * chunk_size
                    next_base = base + chunk_size

                    T.sync_threads()
                    T.copy(
                        state_shared_h,
                        h[
                            data_bid,
                            chunk_begin + cid,
                            hid,
                            v_offset : v_offset + block_v,
                            0:dim_k,
                        ],
                    )
                    # Publish W_i while retaining U_i and K_i as the younger
                    # async groups. U0 follows the same async path as steady U.
                    T.ptx_wait_group(2)
                    T.sync_threads()
                    T.gemm(
                        w_shared,           # (chunk_size, dim_k)    At
                        state_shared_h,     # (block_v, dim_k)       Bn
                        v_new_fragment,     # (chunk_size, block_v)
                        transpose_B=True,
                        clear_accum=True,
                        k_pack=2,
                    )
                    T.sched_barrier()

                    # Gemm0 has consumed W_i. Capture U_i before overwriting its
                    # single LDS buffer, and launch W/U for the following chunk.
                    T.sync_threads()
                    T.async_copy(
                        w[
                            data_bid,
                            token_begin + next_base : token_begin + next_base + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        w_shared,
                    )
                    # U_i is the oldest outstanding group here. Keep K_i and
                    # W_{i+1} in flight, then publish U_i to its fragment.
                    T.ptx_wait_group(2)
                    T.sync_threads()
                    T.copy(u_shared, u_fragment)
                    T.sync_threads()
                    T.async_copy(
                        u[
                            data_bid,
                            token_begin + next_base : token_begin + next_base + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    for it, iv in T.Parallel(chunk_size, block_v):
                        v_new_fragment[it, iv] = (
                            u_fragment[it, iv] - v_new_fragment[it, iv]
                        )

                    if save_new_value:
                        T.copy(
                            v_new_fragment,
                            v_new[
                                data_bid,
                                token_begin + base : token_begin + base + chunk_size,
                                hid,
                                v_offset : v_offset + block_v,
                            ],
                        )
                    T.sched_barrier()

                    # G_i's last use. Its registers can be reused after Gemm1,
                    # when the older Global-memory traffic has had time to drain.
                    g_last = g[data_bid, token_begin + base + chunk_size - 1, hid]
                    for it, iv in T.Parallel(chunk_size, block_v):
                        diff = g_last - g_fragment[it, iv]
                        v_new_fragment[it, iv] = T.if_then_else(
                            diff <= 0,
                            v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                            T.float32(0.0),
                        )
                    g_last = T.exp2(g_last * _LOG2E)
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= g_last

                    # v_new_fragment(FP32) -> v_new_operand(FP16)
                    T.copy(v_new_fragment, v_new_shared)

                    for it, iv in T.Parallel(chunk_size, block_v):
                        g_fragment[it, iv] = g[data_bid, token_begin + next_base + it, hid]

                    T.sched_barrier()

                    T.copy(v_new_shared, v_new_operand)

                    # wait prefetch k_shared ready
                    T.ptx_wait_group(2)
                    T.sync_threads()
                    T.copy(k_shared[cid % 2, :, :], k_fragment)
                    T.sched_barrier()
                    T.async_copy(
                        k[
                            data_bid,
                            token_begin + next_base : token_begin + next_base + chunk_size,
                            hid // (heads // kv_heads),
                            0:dim_k,
                        ],
                        k_shared[(cid + 1) % 2, :, :],
                    )
                    T.gemm(
                        v_new_operand,   # (chunk_size, block_v)  An
                        k_fragment,      # (chunk_size, dim_k)    Bt
                        state_fragment,  # (block_v, dim_k)
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=False,
                        k_pack=1,
                    )
                    T.sched_barrier()
                    T.copy(state_fragment, state_shared_h)

                # Epilogue: W_last and K_last are retained async groups. U_last is
                # in LDS, while G_last is already resident in g_fragment.
                last_cid = full_chunks - 1
                last_base = last_cid * chunk_size
                T.sync_threads()
                T.copy(
                    state_shared_h,
                    h[
                        data_bid,
                        chunk_begin + last_cid,
                        hid,
                        v_offset : v_offset + block_v,
                        0:dim_k,
                    ],
                )
                T.ptx_wait_group(1)
                T.sync_threads()
                T.gemm(
                    w_shared,
                    state_shared_h,
                    v_new_fragment,
                    transpose_B=True,
                    clear_accum=True,
                    k_pack=2,
                )

                T.copy(u_shared, u_fragment)
                for it, iv in T.Parallel(chunk_size, block_v):
                    v_new_fragment[it, iv] = (
                        u_fragment[it, iv] - v_new_fragment[it, iv]
                    )
                if save_new_value:
                    T.copy(
                        v_new_fragment,
                        v_new[
                            data_bid,
                            token_begin + last_base : token_begin + last_base + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                    )

                g_last = g[
                    data_bid, token_begin + last_base + chunk_size - 1, hid
                ]
                for it, iv in T.Parallel(chunk_size, block_v):
                    diff = g_last - g_fragment[it, iv]
                    v_new_fragment[it, iv] = T.if_then_else(
                        diff <= 0,
                        v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                        T.float32(0.0),
                    )

                g_last = T.exp2(g_last * _LOG2E)
                for iv, ik in T.Parallel(block_v, dim_k):
                    state_fragment[iv, ik] *= g_last
                T.copy(v_new_fragment, v_new_shared)
                T.copy(v_new_shared, v_new_operand)

                T.ptx_wait_group(0)
                T.sync_threads()
                T.copy(k_shared[last_cid % 2, :, :], k_fragment)
                T.gemm(
                    v_new_operand,
                    k_fragment,
                    state_fragment,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=False,
                    k_pack=1,
                )
                T.copy(state_fragment, state_shared_h)

            # A non-aligned final chunk uses predicated ordinary copies. Full
            # chunks stay on the unchanged async-copy fast path above.
            if tail_size > 0:
                tail_cid = full_chunks
                tail_base = full_chunks * chunk_size

                T.sync_threads()
                T.copy(
                    state_shared_h,
                    h[
                        data_bid,
                        chunk_begin + tail_cid,
                        hid,
                        v_offset : v_offset + block_v,
                        0:dim_k,
                    ],
                )
                for it, ik in T.Parallel(chunk_size, dim_k):
                    if it < tail_size:
                        w_shared[it, ik] = w[
                            data_bid, token_begin + tail_base + it, hid, ik
                        ]
                        k_shared[0, it, ik] = k[
                            data_bid, token_begin + tail_base + it, hid // (heads // kv_heads), ik
                        ]
                    else:
                        w_shared[it, ik] = T.cast(0, qkv_dtype)
                        k_shared[0, it, ik] = T.cast(0, qkv_dtype)
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        u_shared[it, iv] = u[
                            data_bid, token_begin + tail_base + it, hid, v_offset + iv
                        ]
                        g_fragment[it, iv] = g[
                            data_bid, token_begin + tail_base + it, hid
                        ]
                    else:
                        u_shared[it, iv] = T.cast(0, qkv_dtype)
                        g_fragment[it, iv] = T.float32(0.0)

                T.sync_threads()
                T.gemm(
                    w_shared,
                    state_shared_h,
                    v_new_fragment,
                    transpose_B=True,
                    clear_accum=True,
                    k_pack=2,
                )
                T.copy(u_shared, u_fragment)
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        v_new_fragment[it, iv] = (
                            u_fragment[it, iv] - v_new_fragment[it, iv]
                        )
                    else:
                        v_new_fragment[it, iv] = T.float32(0.0)

                if save_new_value:
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            v_new[
                                data_bid, token_begin + tail_base + it, hid, v_offset + iv
                            ] = T.cast(v_new_fragment[it, iv], qkv_dtype)

                g_last = g[data_bid, token_begin + tail_base + tail_size - 1, hid]
                for it, iv in T.Parallel(chunk_size, block_v):
                    if it < tail_size:
                        diff = g_last - g_fragment[it, iv]
                        v_new_fragment[it, iv] = T.if_then_else(
                            diff <= 0,
                            v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                            T.float32(0.0),
                        )

                g_last = T.exp2(g_last * _LOG2E)
                for iv, ik in T.Parallel(block_v, dim_k):
                    state_fragment[iv, ik] *= g_last
                T.copy(v_new_fragment, v_new_shared)
                T.copy(v_new_shared, v_new_operand)
                T.copy(k_shared[0, :, :], k_fragment)
                T.gemm(
                    v_new_operand,
                    k_fragment,
                    state_fragment,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=False,
                    k_pack=1,
                )
                T.copy(state_fragment, state_shared_h)

            if store_final_state:
                if inplace_final_state:
                    if state_idx >= 0 and state_idx < state_rows:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            if transpose_state_layout:
                                final_state[
                                    state_idx, hid, v_offset + iv, ik
                                ] = state_fragment[iv, ik]
                            else:
                                final_state[
                                    state_idx, hid, ik, v_offset + iv
                                ] = state_fragment[iv, ik]
                else:
                    for iv, ik in T.Parallel(block_v, dim_k):
                        if transpose_state_layout:
                            final_state[
                                sequence_id, hid, v_offset + iv, ik
                            ] = state_fragment[iv, ik]
                        else:
                            final_state[
                                sequence_id, hid, ik, v_offset + iv
                            ] = state_fragment[iv, ik]

    return chunk_gated_delta_rule_fwd_h_copy_pipeline_varlen_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def _build_chunk_h_wasp_4p4c(
    batch: int,
    heads: int,
    kv_heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    block_v: int = 64,
    fixed_has_tail: bool = False,
):
    """Build fixed WASP with register-resident state and independent LDS tiles."""
    assert block_v in (32, 64) and transpose_state_layout
    seq_len = T.dynamic("seq_len")
    num_chunks = seq_len // chunk_size
    h_chunks = T.ceildiv(seq_len, chunk_size) if fixed_has_tail else num_chunks
    state_shape = (
        (heads, dim_v, dim_k)
        if transpose_state_layout
        else (heads, dim_k, dim_v)
    )
    chunk_state_shape = (
        (batch, h_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, h_chunks, heads, dim_k, dim_v)
    )
    w_register_layout = T.Fragment(
        (chunk_size, dim_k),
        forward_thread_fn=lambda t, k, rep=0: rep * 64 + (k % 32) // 8 * 16 + t % 16,
        forward_index_fn=lambda t, k: (k // 32) * 32 + (t // 16) * 8 + k % 8,
        replicate=block_v // 32,
    )

    @T.macro
    def run_consumer_chunk(
        cid,
        w_current,
        w_next,
        prefetch_following_w,
        has_following_chunk,
        w,
        g,
        h,
        v_new,
        bid,
        hid,
        v_offset,
        hf,
        wo,
        state_operand,
        w_shared,
        u_shared,
        u_fragment,
        v_new_fragment,
        g_fragment,
        k_shared,
        k_fragment,
        v_new_shared,
        v_new_operand,
        state_fragment,
    ):
        base = cid * chunk_size
        next_base = base + chunk_size

        T.copy(u_shared, u_fragment)
        # 2 u_fragment reads (outstanding)
        T.s_waitcnt(2, "lgkmcnt")

        T.sched_barrier()
        for kb in T.unroll(dim_k // 32):
            T.copy(state_fragment[:, kb * 32:(kb + 1) * 32], hf)
            T.copy(hf, h[bid, cid, hid, v_offset:v_offset + block_v, kb * 32:(kb + 1) * 32])
            for iv, ik in T.Parallel(block_v, 32):
                state_operand[iv, ik] = hf[
                    (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16, ik
                ]
            T.copy(w_current[:, kb * 32:(kb + 1) * 32], wo)
            T.gemm(
                wo,
                state_operand,
                v_new_fragment,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=(kb == 0),
                k_pack=2,
            )
        T.sched_barrier()

        # Barrier 1: u_shared -> u_fragment for chunk cid is complete, so
        # producers may overwrite u_shared.
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        for it, iv in T.Parallel(chunk_size, block_v):
            v_new_fragment[it, iv] = (
                T.cast(u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16], "float32")
                - v_new_fragment[it, iv]
            )

        # Save the ungated v_new_fragment for chunk cid directly from consumer
        # VGPRs.
        if save_new_value:
            for it, iv in T.Parallel(chunk_size, block_v):
                u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16] = T.cast(v_new_fragment[it, iv], qkv_dtype)
            T.copy(
                u_fragment,
                v_new[
                    bid,
                    base : base + chunk_size,
                    hid,
                    v_offset : v_offset + block_v,
                ],
            )

        # Barrier 2: wait for producer publication of k_shared for chunk cid.
        T.sync_warp()

        T.ds_read_vector(k_shared, k_fragment, alt=2)
        T.s_waitcnt(4, "lgkmcnt")

        g_last_value = g[bid, base + chunk_size - 1, hid]
        T.s_waitcnt(6, "vmcnt")
        for it, iv in T.Parallel(chunk_size, block_v):
            diff = g_last_value - g_fragment[it, iv]
            v_new_fragment[it, iv] = T.if_then_else(
                diff <= 0,
                v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                T.float32(0.0),
            )
        g_last_scale = T.exp2(g_last_value * _LOG2E)
        for iv, ik in T.Parallel(block_v, dim_k):
            state_fragment[iv, ik] *= g_last_scale

        for it, iv in T.Parallel(chunk_size, block_v):
            u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16] = T.cast(v_new_fragment[it, iv], qkv_dtype)
        T.copy(u_fragment, v_new_shared)
        T.sched_barrier()

        # Barrier 3: complete k_shared -> k_fragment and publish v_new_shared
        # for chunk cid. Producers may then overwrite k_shared with data for
        # chunk cid+1.
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()
        T.ds_read_vector(v_new_shared, v_new_operand, alt=2)
        if prefetch_following_w:
            T.copy(w_shared, w_next)

        T.sched_barrier()
        T.gemm(
            v_new_operand,
            k_fragment,
            state_fragment,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullRow,
            clear_accum=False,
            k_pack=1,
        )
        T.sched_barrier()

        # Prefetch the gate for the next chunk after GEMM1.
        if has_following_chunk:
            for it, iv in T.Parallel(chunk_size, block_v):
                g_fragment[it, iv] = g[bid, next_base + it, hid]
        T.sched_barrier()

        # Barrier 4
        # Producer U for the next chunk is ready.
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        if has_following_chunk:
            T.sched_barrier()
            T.sched_barrier()

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_kernel(
        k: T.Tensor([batch, seq_len, kv_heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([batch], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor(
            [state_rows if inplace_final_state else batch, *state_shape],
            "float32",
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), batch * heads, threads=256 + (block_v // 32) * 64
        ) as (vid, bbh):
            bid, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            tx = T.get_thread_binding()

            # Consumers reuse one W fragment. Producer waves
            # own the single-buffered U/K/W global-to-LDS path.
            w_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            u_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            v_new_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)

            # T.use_swizzle(10)

            if tx < _PRODUCER_THREADS:
                if not fixed_has_tail or num_chunks > 0:
                    # Prologue: U_0 is the oldest group, K_0 remains outstanding.
                    T.async_copy(
                        u[
                            bid,
                            0:chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    T.async_copy(
                        k[bid, 0:chunk_size, hid // (heads // kv_heads), 0:dim_k],
                        k_shared,
                    )

                    # Barrier 0: u_shared ready; k_shared outstanding.
                    T.ptx_wait_group(1)
                    T.sync_warp()

                    for cid in T.serial(num_chunks - 1):
                        base = cid * chunk_size
                        next_base = base + chunk_size

                        # Produce W_i+1 while consumers execute Gemm0 for chunk i.
                        T.async_copy(
                            w[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                                0:dim_k,
                            ],
                            w_shared,
                        )

                        # Barrier 1: consumers have captured u_shared for chunk
                        # cid, so producers may overwrite it.
                        T.sync_warp()

                        T.async_copy(
                            u[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                                v_offset : v_offset + block_v,
                            ],
                            u_shared,
                        )
                        # Complete k_shared for chunk cid; retain W/U copies.
                        T.ptx_wait_group(2)

                        # Barrier 2: k_shared for chunk cid is ready; u_shared for
                        # chunk cid+1 may remain outstanding.
                        T.sync_warp()

                        # Complete w_shared for chunk cid+1; retain its U copy.
                        T.ptx_wait_group(1)

                        # Barrier 3: consumers have captured k_shared for chunk
                        # cid; v_new_shared and W for chunk cid+1 are ready.
                        T.sync_warp()

                        T.async_copy(
                            k[
                                bid,
                                next_base : next_base + chunk_size,
                                hid // (heads // kv_heads),
                                0:dim_k,
                            ],
                            k_shared,
                        )
                        # Complete u_shared for chunk cid+1; retain its K copy.
                        T.ptx_wait_group(1)

                        # Barrier 4: next u_shared ready.
                        T.sync_warp()

                    # Barrier 1: final U captured.
                    T.sync_warp()
                    T.ptx_wait_group(0)

                    # Barrier 2: final k_shared ready.
                    T.sync_warp()

                    # Barrier 3: final K captured and v_new_shared published.
                    T.sync_warp()

                    # Barrier 4: close the producer/consumer protocol.
                    T.sync_warp()

                if fixed_has_tail:
                    T.sync_warp()
                    T.sync_warp()

            else:
                state_fragment = T.alloc_fragment(
                    (block_v, dim_k), "float32"
                )
                u_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                v_new_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                v_new_operand = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                k_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                g_fragment = T.alloc_fragment(
                    (chunk_size, block_v), gate_dtype
                )
                w_operand_0 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_operand = T.alloc_fragment((block_v, 32), qkv_dtype)
                hf = T.alloc_fragment((block_v, 32), qkv_dtype)
                wo = T.alloc_fragment((chunk_size, 32), qkv_dtype)
                T.annotate_layout({w_operand_0: w_register_layout})
                state_io_tile = T.alloc_fragment((block_v, 32), "float32")
                state_idx = T.alloc_var("int32")

                state_idx = initial_state_indices[bid]
                if use_initial_state and state_idx >= 0 and state_idx < state_rows:
                    for bk in T.unroll(dim_k // 32):
                        T.copy(initial_state[state_idx, hid, v_offset:v_offset + block_v, bk * 32:(bk + 1) * 32], state_io_tile)
                        for iv, ik in T.Parallel(block_v, 32):
                            state_fragment[iv, bk * 32 + ik] = state_io_tile[iv, ik]
                else:
                    T.fill(state_fragment, 0)

                if not fixed_has_tail or num_chunks > 0:
                    # Keep W in consumer VGPRs; reuse the slot after GEMM0.
                    T.copy(
                        w[bid, 0:chunk_size, hid, 0:dim_k],
                        w_operand_0,
                    )
                    for it, iv in T.Parallel(chunk_size, block_v):
                        g_fragment[it, iv] = g[bid, it, hid]

                    # Barrier 0: wait for producer publication of the first U tile.
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()
                    T.sched_barrier()
                    T.sched_barrier()

                    # A single W fragment and a separate final chunk avoid
                    # paired-chunk register pressure for every BV.
                    # Keep the W/G prefetch overlap in the steady loop.
                    for cid in T.Pipelined(num_chunks - 1, num_stages=0):
                        run_consumer_chunk(
                            cid, w_operand_0, w_operand_0,
                            True, True,
                            w, g, h, v_new, bid, hid, v_offset,
                            hf, wo, state_operand, w_shared,
                            u_shared, u_fragment,
                            v_new_fragment, g_fragment, k_shared, k_fragment,
                            v_new_shared, v_new_operand, state_fragment,
                        )

                    run_consumer_chunk(
                        num_chunks - 1, w_operand_0, w_operand_0,
                        False, False,
                        w, g, h, v_new, bid, hid, v_offset,
                        hf, wo, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )


                if fixed_has_tail:
                    T.sync_warp()
                    tail_size = seq_len % chunk_size
                    tail_base = num_chunks * chunk_size
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        w_operand_0[it, ik] = T.if_then_else(
                            it < tail_size, w[bid, tail_base + it, hid, ik], T.cast(0, qkv_dtype)
                        )
                    for it, iv in T.Parallel(chunk_size, block_v):
                        u_fragment[it, iv] = T.if_then_else(
                            it < tail_size, u[bid, tail_base + it, hid, v_offset + iv], T.cast(0, qkv_dtype)
                        )
                    for it, iv in T.Parallel(chunk_size, block_v):
                        g_fragment[it, iv] = T.if_then_else(
                            it < tail_size, g[bid, tail_base + it, hid], T.float32(0)
                        )
                    T.s_waitcnt(0, "vmcnt")
                    for kb in T.unroll(dim_k // 32):
                        T.copy(state_fragment[:, kb * 32:(kb + 1) * 32], hf)
                        T.copy(hf, h[bid, num_chunks, hid, v_offset:v_offset + block_v, kb * 32:(kb + 1) * 32])
                        for iv, ik in T.Parallel(block_v, 32):
                            state_operand[iv, ik] = hf[(iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16, ik]
                        T.copy(w_operand_0[:, kb * 32:(kb + 1) * 32], wo)
                        T.gemm(wo, state_operand, v_new_fragment, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullCol, clear_accum=(kb == 0), k_pack=2)
                    T.sched_barrier()
                    for it, iv in T.Parallel(chunk_size, block_v):
                        v_new_fragment[it, iv] = T.if_then_else(
                            it < tail_size,
                            T.cast(u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16], "float32") - v_new_fragment[it, iv],
                            T.float32(0),
                        )
                    if save_new_value:
                        for it, iv in T.Parallel(chunk_size, block_v):
                            u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16] = T.cast(v_new_fragment[it, iv], qkv_dtype)
                        for it, iv in T.Parallel(chunk_size, block_v):
                            if it < tail_size:
                                v_new[bid, tail_base + it, hid, v_offset + iv] = u_fragment[it, iv]
                    tail_g_last = g[bid, tail_base + tail_size - 1, hid]
                    T.s_waitcnt(0, "vmcnt")
                    for it, iv in T.Parallel(chunk_size, block_v):
                        diff = tail_g_last - g_fragment[it, iv]
                        v_new_fragment[it, iv] = T.if_then_else(
                            it < tail_size and diff <= 0,
                            v_new_fragment[it, iv] * T.exp2(diff * _LOG2E), T.float32(0),
                        )
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= T.exp2(tail_g_last * _LOG2E)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        u_fragment[it, (iv // 32) * 32 + 2 * (iv % 16) + (iv % 32) // 16] = T.cast(v_new_fragment[it, iv], qkv_dtype)
                    T.copy(u_fragment, v_new_shared)
                    T.sched_barrier()
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        k_fragment[it, ik] = T.if_then_else(
                            it < tail_size, k[bid, tail_base + it, hid // (heads // kv_heads), ik], T.cast(0, qkv_dtype)
                        )
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()
                    T.ds_read_vector(v_new_shared, v_new_operand, alt=2)
                    T.s_waitcnt(0, "vmcnt")
                    T.gemm(v_new_operand, k_fragment, state_fragment, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow, clear_accum=False, k_pack=1)

                if store_final_state:
                    if inplace_final_state:
                        if state_idx >= 0 and state_idx < state_rows:
                            for bk in T.unroll(dim_k // 32):
                                T.copy(state_fragment[:, bk * 32:(bk + 1) * 32], state_io_tile)
                                T.copy(state_io_tile, final_state[state_idx, hid, v_offset:v_offset + block_v, bk * 32:(bk + 1) * 32])
                    else:
                        for bk in T.unroll(dim_k // 32):
                            T.copy(state_fragment[:, bk * 32:(bk + 1) * 32], state_io_tile)
                            T.copy(state_io_tile, final_state[bid, hid, v_offset:v_offset + block_v, bk * 32:(bk + 1) * 32])

    return chunk_gated_delta_rule_fwd_h_wasp_4p4c_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_HCU_WDRA: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def _build_chunk_h_wasp_4p4c_gfx946(
    batch: int,
    heads: int,
    kv_heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    is_varlen: bool = False,
    fixed_has_tail: bool = False,
    block_v: int = 32,
):
    """Build gfx946 fixed/varlen kernels with grouped K heads.

    Only K uses kv_heads: consecutive groups of heads // kv_heads value
    heads share one K head. W/U/G and all states remain indexed by heads.
    """
    assert block_v in (32, 64)
    seq_len = T.dynamic("seq_len")
    full_chunks = seq_len // chunk_size
    num_chunks = (
        T.ceildiv(seq_len, chunk_size) if fixed_has_tail else full_chunks
    )
    state_shape = (
        (heads, dim_v, dim_k)
        if transpose_state_layout
        else (heads, dim_k, dim_v)
    )
    chunk_state_shape = (
        (batch, num_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, num_chunks, heads, dim_k, dim_v)
    )
    state_shared_h_k_major_layout = T.Layout(
        (block_v, dim_k),
        lambda iv, ik: (
            iv,
            ((ik // 8) ^ (iv % 8)) * 8 + ik % 8,
        ),
    )
    v_new_shared_v8_phase_layout = T.Layout(
        (chunk_size, block_v),
        lambda it, iv: (
            it,
            ((iv // 8) ^ (it % 4)) * 8 + iv % 8,
        ),
    )

    @T.macro
    def run_consumer_chunk(
        cid,
        w_current,
        w_next,
        prefetch_next,
        has_following_chunk,
        w,
        g,
        h,
        v_new,
        bid,
        hid,
        v_offset,
        state_shared_h,
        state_operand,
        w_shared,
        u_shared,
        u_fragment,
        v_new_fragment,
        g_fragment,
        k_shared,
        k_fragment,
        v_new_shared,
        v_new_operand,
        state_fragment,
        token_begin=0,
    ):
        # Prefetch gates directly into consumer registers, overlapping Gemm0.
        # Avoid a separate 256-byte LDS allocation so BV64 fits in 64 KiB.
        for it, iv in T.Parallel(chunk_size, block_v):
            g_fragment[it, iv] = g[bid, token_begin + cid * chunk_size + it, hid]
        g_last_value = g[bid, token_begin + (cid + 1) * chunk_size - 1, hid]

        # U uses its own single-buffer handoff.  The producer-side READY
        # arrival completes only after the bound global-to-LDS copy.
        T.abarrier_try_wait(_WASP_U_READY_PING, cid & 1)
        T.copy(u_shared, u_fragment)
        # 2 u_fragment reads (outstanding)
        T.s_waitcnt(2, "lgkmcnt")

        T.sched_barrier()
        T.gemm(
            w_current,
            state_operand,
            v_new_fragment,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
            clear_accum=True,
            k_pack=2,
        )
        T.sched_barrier()

        # Barrier 1: u_shared -> u_fragment for chunk cid is complete, so
        # producers may overwrite u_shared.
        T.s_waitcnt(0, "lgkmcnt")
        T.abarrier_arrive(_WASP_U_FREE_PING)

        for it, iv in T.Parallel(chunk_size, block_v):
            v_new_fragment[it, iv] = (
                T.cast(u_fragment[it, iv], "float32")
                - v_new_fragment[it, iv]
            )

        # Publish the ungated value for producer-side global storage.  The
        # producer releases v_new_shared after capturing it in VGPRs, while
        # consumers continue with the elementwise gating work.
        if save_new_value:
            T.abarrier_try_wait(_WASP_V_NEW_FREE_PING, cid & 1)
            T.abarrier_seq(_WASP_V_NEW_READY_PING)
            T.copy(v_new_fragment, v_new_shared)
            T.s_waitcnt(0, "lgkmcnt")
            T.abarrier_arrive(_WASP_V_NEW_READY_PING)

        # Barrier 2: wait for producer publication of k_shared for chunk cid.
        T.abarrier_try_wait(_WASP_K_READY_PING, cid & 1)

        T.copy(k_shared, k_fragment)
        T.s_waitcnt(4, "lgkmcnt")

        for it, iv in T.Parallel(chunk_size, block_v):
            diff = g_last_value - g_fragment[it, iv]
            v_new_fragment[it, iv] = T.if_then_else(
                diff <= 0,
                v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                T.float32(0.0),
            )
        g_last_scale = T.exp2(g_last_value * _LOG2E)
        for iv, ik in T.Parallel(block_v, dim_k):
            state_fragment[iv, ik] *= g_last_scale

        if save_new_value:
            T.abarrier_try_wait(_WASP_V_NEW_FREE_PING, (cid + 1) & 1)
        T.copy(v_new_fragment, v_new_shared)
        T.sched_barrier()

        # Barrier 3: complete k_shared -> k_fragment and publish v_new_shared
        # for chunk cid. Producers may then overwrite k_shared with data for
        # chunk cid+1.
        T.s_waitcnt(0, "lgkmcnt")
        T.abarrier_arrive(_WASP_K_FREE_PING)
        T.ebarrier_sync_cnt(
            _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
        )
        T.copy(v_new_shared, v_new_operand)
        if prefetch_next:
            # Both fixed and varlen publish W_0 through the producer, so the
            # following chunk always uses the next READY phase.
            T.abarrier_try_wait(_WASP_W_READY_PING, (cid + 1) & 1)
            T.copy(w_shared, w_next)

        T.sched_barrier()
        T.gemm(
            v_new_operand,
            k_fragment,
            state_fragment,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=False,
            k_pack=1,
        )
        T.sched_barrier()
        T.abarrier_try_wait(_WASP_STATE_FREE_PING, (cid + 1) & 1)
        T.abarrier_seq(_WASP_STATE_READY_PING)
        T.copy(state_fragment, state_shared_h)

        T.sched_barrier()

        # Barrier 4
        # state_shared_h ready
        T.s_waitcnt(0, "lgkmcnt")
        T.abarrier_arrive(_WASP_STATE_READY_PING)
        if prefetch_next:
            T.abarrier_arrive(_WASP_W_FREE_PING)
        T.ebarrier_sync_cnt(
            _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
        )

        if has_following_chunk:
            T.copy(state_shared_h, state_operand)
            T.sched_barrier()

    @T.macro
    def drain_producer_v_new(
        cid,
        v_new,
        bid,
        hid,
        v_offset,
        v_new_shared,
        v_new_save_fragment,
    ):
        base = cid * chunk_size
        if save_new_value:
            T.abarrier_try_wait(_WASP_V_NEW_READY_PING, cid & 1)
            T.copy(v_new_shared, v_new_save_fragment)
            T.s_waitcnt(0, "lgkmcnt")
            T.abarrier_arrive(_WASP_V_NEW_FREE_PING)
            T.copy(
                v_new_save_fragment,
                v_new[
                    bid,
                    base : base + chunk_size,
                    hid,
                    v_offset : v_offset + block_v,
                ],
            )

    @T.macro
    def drain_producer_state(
        cid,
        h,
        bid,
        hid,
        v_offset,
        state_shared_h,
        state_save_fragment,
    ):
        T.abarrier_try_wait(_WASP_STATE_READY_PING, cid & 1)
        T.copy(state_shared_h, state_save_fragment)
        T.s_waitcnt(0, "lgkmcnt")
        T.abarrier_arrive(_WASP_STATE_FREE_PING)
        if transpose_state_layout:
            T.copy(
                state_save_fragment,
                h[
                    bid,
                    cid,
                    hid,
                    v_offset : v_offset + block_v,
                    0:dim_k,
                ],
            )
        else:
            for iv, ik in T.Parallel(block_v, dim_k):
                h[bid, cid, hid, ik, v_offset + iv] = state_save_fragment[iv, ik]

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_gfx946_kernel(
        k: T.Tensor([batch, seq_len, kv_heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([batch], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor(
            [state_rows if inplace_final_state else batch, *state_shape],
            "float32",
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), batch * heads, threads=_TOTAL_THREADS
        ) as (vid, bbh):
            bid, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            tx = T.get_thread_binding()

            # Consumer waves retain W in two VGPR fragments. Producer waves
            # own the single-buffered U/K/W global-to-LDS path.
            state_shared_h = T.alloc_shared((block_v, dim_k), qkv_dtype)
            w_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            u_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            v_new_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)

            T.annotate_layout(
                {
                    state_shared_h: state_shared_h_k_major_layout,
                    v_new_shared: v_new_shared_v8_phase_layout,
                }
            )
            # T.use_swizzle(10)

            T.abarrier_init(_WASP_U_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_U_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(_WASP_K_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_K_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(_WASP_W_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_W_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(
                _WASP_STATE_FREE_PING, _WASP_PRODUCER_WAVES
            )
            T.abarrier_init(
                _WASP_STATE_READY_PING, _WASP_CONSUMER_WAVES
            )
            if save_new_value:
                T.abarrier_init(
                    _WASP_V_NEW_FREE_PING, _WASP_PRODUCER_WAVES
                )
                T.abarrier_init(
                    _WASP_V_NEW_READY_PING, _WASP_CONSUMER_WAVES
                )
            T.ebarrier_sync_cnt(_WASP_INIT_EBAR_ID, _TOTAL_WAVES)

            if tx < _PRODUCER_THREADS:
                T.set_max_nreg(_WASP_PRODUCER_MAX_NREG, 0)
                state_save_fragment = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                v_new_save_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                # The initial state_shared_h slot is free for consumers.
                T.abarrier_arrive(_WASP_STATE_FREE_PING)
                if save_new_value:
                    T.abarrier_arrive(_WASP_V_NEW_FREE_PING)

                # Use one uniform producer iteration for every full chunk.
                # Each iteration publishes the current W/U/K inputs and then
                # drains the current consumer outputs.
                for cid in T.serial(full_chunks):
                    base = cid * chunk_size

                    T.abarrier_try_wait(_WASP_W_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_W_READY_PING)
                    T.async_copy(
                        w[
                            bid,
                            base : base + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        w_shared,
                    )
                    T.abarrier_arrive(_WASP_W_READY_PING)

                    T.abarrier_try_wait(_WASP_U_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_U_READY_PING)
                    T.async_copy(
                        u[
                            bid,
                            base : base + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    T.abarrier_arrive(_WASP_U_READY_PING)

                    T.abarrier_try_wait(_WASP_K_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_K_READY_PING)
                    T.async_copy(
                        k[
                            bid,
                            base : base + chunk_size,
                            hid // (heads // kv_heads),
                            0:dim_k,
                        ],
                        k_shared,
                    )
                    T.abarrier_arrive(_WASP_K_READY_PING)

                    drain_producer_v_new(
                        cid,
                        v_new,
                        bid,
                        hid,
                        v_offset,
                        v_new_shared,
                        v_new_save_fragment,
                    )
                    drain_producer_state(
                        cid,
                        h,
                        bid,
                        hid,
                        v_offset,
                        state_shared_h,
                        state_save_fragment,
                    )

            else:
                T.set_max_nreg(
                    _WASP_VARLEN_CONSUMER_MAX_NREG
                    if fixed_has_tail
                    else _WASP_CONSUMER_MAX_NREG,
                    0,
                )
                # Initially all three single-buffered LDS regions are free.
                T.abarrier_arrive(_WASP_U_FREE_PING)
                T.abarrier_arrive(_WASP_K_FREE_PING)
                T.abarrier_arrive(_WASP_W_FREE_PING)

                state_fragment = T.alloc_fragment(
                    (block_v, dim_k), "float32"
                )
                u_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                v_new_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                v_new_operand = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                k_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                g_fragment = T.alloc_fragment(
                    (chunk_size, block_v), gate_dtype
                )
                w_operand_0 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                w_operand_1 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_operand = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                state_idx = T.alloc_var("int32")
                if fixed_has_tail:
                    tail_g_last = T.alloc_var("float32")

                state_idx = initial_state_indices[bid]
                for iv, ik in T.Parallel(block_v, dim_k):
                    if (
                        use_initial_state
                        and state_idx >= 0
                        and state_idx < state_rows
                    ):
                        if transpose_state_layout:
                            state_fragment[iv, ik] = initial_state[
                                state_idx, hid, v_offset + iv, ik
                            ]
                        else:
                            state_fragment[iv, ik] = initial_state[
                                state_idx, hid, ik, v_offset + iv
                            ]
                    else:
                        state_fragment[iv, ik] = T.float32(0.0)
                T.abarrier_try_wait(_WASP_STATE_FREE_PING, 0)
                T.abarrier_seq(_WASP_STATE_READY_PING)
                T.copy(state_fragment, state_shared_h)

                # Publish the initial state among the four consumer waves.
                # U_0 readiness is handled independently at the beginning of
                # run_consumer_chunk.
                T.s_waitcnt(0, "lgkmcnt")
                T.abarrier_arrive(_WASP_STATE_READY_PING)
                T.ebarrier_sync_cnt(
                    _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
                )
                T.copy(state_shared_h, state_operand)
                T.sched_barrier()

                # Load each pair's even W at pair entry. The even chunk then
                # prefetches the odd W while Gemm1 executes. This keeps every
                # full pair uniform and leaves only an odd tail outside.
                full_chunk_pairs = full_chunks // 2
                for pair in T.serial(full_chunk_pairs):
                    cid = pair * 2
                    T.abarrier_try_wait(_WASP_W_READY_PING, cid & 1)
                    T.copy(w_shared, w_operand_0)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.abarrier_arrive(_WASP_W_FREE_PING)
                    run_consumer_chunk(
                        cid,
                        w_operand_0,
                        w_operand_1,
                        True,
                        True,
                        w,
                        g,
                        h,
                        v_new,
                        bid,
                        hid,
                        v_offset,
                        state_shared_h,
                        state_operand,
                        w_shared,
                        u_shared,
                        u_fragment,
                        v_new_fragment,
                        g_fragment,
                        k_shared,
                        k_fragment,
                        v_new_shared,
                        v_new_operand,
                        state_fragment,
                    )
                    run_consumer_chunk(
                        cid + 1,
                        w_operand_1,
                        w_operand_0,
                        False,
                        True,
                        w,
                        g,
                        h,
                        v_new,
                        bid,
                        hid,
                        v_offset,
                        state_shared_h,
                        state_operand,
                        w_shared,
                        u_shared,
                        u_fragment,
                        v_new_fragment,
                        g_fragment,
                        k_shared,
                        k_fragment,
                        v_new_shared,
                        v_new_operand,
                        state_fragment,
                    )

                if full_chunks % 2 != 0:
                    tail_cid = full_chunk_pairs * 2
                    T.abarrier_try_wait(
                        _WASP_W_READY_PING, tail_cid & 1
                    )
                    T.copy(w_shared, w_operand_0)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.abarrier_arrive(_WASP_W_FREE_PING)
                    run_consumer_chunk(
                        tail_cid,
                        w_operand_0,
                        w_operand_1,
                        False,
                        True,
                        w,
                        g,
                        h,
                        v_new,
                        bid,
                        hid,
                        v_offset,
                        state_shared_h,
                        state_operand,
                        w_shared,
                        u_shared,
                        u_fragment,
                        v_new_fragment,
                        g_fragment,
                        k_shared,
                        k_fragment,
                        v_new_shared,
                        v_new_operand,
                        state_fragment,
                    )

                if fixed_has_tail:
                    tail_size = seq_len % chunk_size
                    tail_base = full_chunks * chunk_size
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        if it < tail_size:
                            w_operand_0[it, ik] = w[
                                bid, tail_base + it, hid, ik
                            ]
                        else:
                            w_operand_0[it, ik] = T.cast(0, qkv_dtype)
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        if it < tail_size:
                            k_fragment[it, ik] = k[
                                bid, tail_base + it, hid // (heads // kv_heads), ik
                            ]
                        else:
                            k_fragment[it, ik] = T.cast(0, qkv_dtype)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            u_fragment[it, iv] = u[
                                bid,
                                tail_base + it,
                                hid,
                                v_offset + iv,
                            ]
                        else:
                            u_fragment[it, iv] = T.cast(0, qkv_dtype)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            g_fragment[it, iv] = g[
                                bid, tail_base + it, hid
                            ]
                        else:
                            g_fragment[it, iv] = T.float32(0.0)
                    T.s_waitcnt(0, "vmcnt")

                    if transpose_state_layout:
                        T.copy(
                            state_fragment,
                            h[
                                bid,
                                full_chunks,
                                hid,
                                v_offset : v_offset + block_v,
                                0:dim_k,
                            ],
                        )
                    else:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            h[
                                bid,
                                full_chunks,
                                hid,
                                ik,
                                v_offset + iv,
                            ] = state_fragment[iv, ik]

                    T.sched_barrier()
                    T.gemm(
                        w_operand_0,
                        state_operand,
                        v_new_fragment,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                        k_pack=2,
                    )
                    T.sched_barrier()
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            v_new_fragment[it, iv] = (
                                T.cast(u_fragment[it, iv], "float32")
                                - v_new_fragment[it, iv]
                            )
                        else:
                            v_new_fragment[it, iv] = T.float32(0.0)

                    if save_new_value:
                        for it, iv in T.Parallel(chunk_size, block_v):
                            if it < tail_size:
                                v_new[
                                    bid,
                                    tail_base + it,
                                    hid,
                                    v_offset + iv,
                                ] = T.cast(v_new_fragment[it, iv], qkv_dtype)

                    tail_g_last = g[bid, tail_base + tail_size - 1, hid]
                    T.s_waitcnt(0, "vmcnt")
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            diff = tail_g_last - g_fragment[it, iv]
                            v_new_fragment[it, iv] = T.if_then_else(
                                diff <= 0,
                                v_new_fragment[it, iv]
                                * T.exp2(diff * _LOG2E),
                                T.float32(0.0),
                            )
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= T.exp2(
                            tail_g_last * _LOG2E
                        )

                    T.copy(v_new_fragment, v_new_shared)
                    T.sched_barrier()
                    T.s_waitcnt(0, "lgkmcnt")
                    T.ebarrier_sync_cnt(
                        _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
                    )
                    T.copy(v_new_shared, v_new_operand)
                    T.sched_barrier()
                    T.gemm(
                        v_new_operand,
                        k_fragment,
                        state_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=False,
                        k_pack=1,
                    )

                if store_final_state:
                    if inplace_final_state:
                        if state_idx >= 0 and state_idx < state_rows:
                            for iv, ik in T.Parallel(block_v, dim_k):
                                if transpose_state_layout:
                                    final_state[
                                        state_idx, hid, v_offset + iv, ik
                                    ] = state_fragment[iv, ik]
                                else:
                                    final_state[
                                        state_idx, hid, ik, v_offset + iv
                                    ] = state_fragment[iv, ik]
                    else:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            if transpose_state_layout:
                                final_state[
                                    bid, hid, v_offset + iv, ik
                                ] = state_fragment[iv, ik]
                            else:
                                final_state[
                                    bid, hid, ik, v_offset + iv
                                ] = state_fragment[iv, ik]

    if not is_varlen:
        return chunk_gated_delta_rule_fwd_h_wasp_4p4c_gfx946_kernel

    num_sequences = T.dynamic("num_sequences")
    total_num_chunks = T.dynamic("num_chunks")
    cu_size = T.dynamic("cu_size")
    chunk_offsets_size = T.dynamic("chunk_offsets_size")
    varlen_chunk_state_shape = (
        (batch, total_num_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, total_num_chunks, heads, dim_k, dim_v)
    )

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_gfx946_varlen_kernel(
        k: T.Tensor([batch, seq_len, kv_heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([num_sequences], "int32"),
        cu_seqlens: T.Tensor([cu_size], "int32"),
        chunk_offsets: T.Tensor([chunk_offsets_size], "int32"),
        h: T.Tensor(varlen_chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor(
            [
                state_rows if inplace_final_state else num_sequences,
                *state_shape,
            ],
            "float32",
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v),
            num_sequences * heads,
            threads=_TOTAL_THREADS,
        ) as (vid, bbh):
            sequence_id, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            tx = T.get_thread_binding()

            state_shared_h = T.alloc_shared((block_v, dim_k), qkv_dtype)
            w_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            u_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            v_new_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)

            T.annotate_layout(
                {
                    state_shared_h: state_shared_h_k_major_layout,
                    v_new_shared: v_new_shared_v8_phase_layout,
                }
            )

            T.abarrier_init(_WASP_U_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_U_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(_WASP_K_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_K_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(_WASP_W_FREE_PING, _WASP_CONSUMER_WAVES)
            T.abarrier_init(_WASP_W_READY_PING, _WASP_PRODUCER_WAVES)
            T.abarrier_init(
                _WASP_STATE_FREE_PING, _WASP_PRODUCER_WAVES
            )
            T.abarrier_init(
                _WASP_STATE_READY_PING, _WASP_CONSUMER_WAVES
            )
            if save_new_value:
                T.abarrier_init(
                    _WASP_V_NEW_FREE_PING, _WASP_PRODUCER_WAVES
                )
                T.abarrier_init(
                    _WASP_V_NEW_READY_PING, _WASP_CONSUMER_WAVES
                )
            T.ebarrier_sync_cnt(_WASP_INIT_EBAR_ID, _TOTAL_WAVES)

            if tx < _PRODUCER_THREADS:
                T.set_max_nreg(_WASP_VARLEN_PRODUCER_MAX_NREG, 0)
                token_begin = cu_seqlens[sequence_id]
                actual_seq_len = (
                    cu_seqlens[sequence_id + 1] - token_begin
                )
                chunk_begin = chunk_offsets[sequence_id]
                full_chunks = actual_seq_len // chunk_size
                state_save_fragment = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                v_new_save_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                T.abarrier_arrive(_WASP_STATE_FREE_PING)
                if save_new_value:
                    T.abarrier_arrive(_WASP_V_NEW_FREE_PING)

                # Match the fixed kernel: publish this chunk's W/U/K, then
                # capture its v_new and entering state before advancing.
                for cid in T.serial(full_chunks):
                    base = token_begin + cid * chunk_size

                    T.abarrier_try_wait(_WASP_W_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_W_READY_PING)
                    T.async_copy(
                        w[0, base : base + chunk_size, hid, 0:dim_k],
                        w_shared,
                    )
                    T.abarrier_arrive(_WASP_W_READY_PING)

                    T.abarrier_try_wait(_WASP_U_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_U_READY_PING)
                    T.async_copy(
                        u[
                            0,
                            base : base + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    T.abarrier_arrive(_WASP_U_READY_PING)

                    T.abarrier_try_wait(_WASP_K_FREE_PING, cid & 1)
                    T.abarrier_seq(_WASP_K_READY_PING)
                    T.async_copy(
                        k[
                            0,
                            base : base + chunk_size,
                            hid // (heads // kv_heads),
                            0:dim_k,
                        ],
                        k_shared,
                    )
                    T.abarrier_arrive(_WASP_K_READY_PING)

                    if save_new_value:
                        T.abarrier_try_wait(
                            _WASP_V_NEW_READY_PING, cid & 1
                        )
                        T.copy(v_new_shared, v_new_save_fragment)
                        T.s_waitcnt(0, "lgkmcnt")
                        T.abarrier_arrive(_WASP_V_NEW_FREE_PING)
                        T.copy(
                            v_new_save_fragment,
                            v_new[
                                0,
                                base : base + chunk_size,
                                hid,
                                v_offset : v_offset + block_v,
                            ],
                        )

                    T.abarrier_try_wait(
                        _WASP_STATE_READY_PING, cid & 1
                    )
                    T.copy(state_shared_h, state_save_fragment)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.abarrier_arrive(_WASP_STATE_FREE_PING)
                    if transpose_state_layout:
                        T.copy(
                            state_save_fragment,
                            h[
                                0,
                                chunk_begin + cid,
                                hid,
                                v_offset : v_offset + block_v,
                                0:dim_k,
                            ],
                        )
                    else:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            h[
                                0,
                                chunk_begin + cid,
                                hid,
                                ik,
                                v_offset + iv,
                            ] = state_save_fragment[iv, ik]

            else:
                T.set_max_nreg(_WASP_VARLEN_CONSUMER_MAX_NREG, 0)
                token_begin = cu_seqlens[sequence_id]
                actual_seq_len = (
                    cu_seqlens[sequence_id + 1] - token_begin
                )
                chunk_begin = chunk_offsets[sequence_id]
                full_chunks = actual_seq_len // chunk_size
                tail_size = actual_seq_len % chunk_size
                tail_base = token_begin + full_chunks * chunk_size
                T.abarrier_arrive(_WASP_U_FREE_PING)
                T.abarrier_arrive(_WASP_K_FREE_PING)
                T.abarrier_arrive(_WASP_W_FREE_PING)

                state_fragment = T.alloc_fragment(
                    (block_v, dim_k), "float32"
                )
                u_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                v_new_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                v_new_operand = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                k_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                g_fragment = T.alloc_fragment(
                    (chunk_size, block_v), gate_dtype
                )
                w_operand_0 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                w_operand_1 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_operand = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                state_idx = T.alloc_var("int32")
                tail_g_last = T.alloc_var("float32")

                state_idx = initial_state_indices[sequence_id]
                for iv, ik in T.Parallel(block_v, dim_k):
                    if (
                        use_initial_state
                        and state_idx >= 0
                        and state_idx < state_rows
                    ):
                        if transpose_state_layout:
                            state_fragment[iv, ik] = initial_state[
                                state_idx, hid, v_offset + iv, ik
                            ]
                        else:
                            state_fragment[iv, ik] = initial_state[
                                state_idx, hid, ik, v_offset + iv
                            ]
                    else:
                        state_fragment[iv, ik] = T.float32(0.0)

                T.abarrier_try_wait(_WASP_STATE_FREE_PING, 0)
                T.abarrier_seq(_WASP_STATE_READY_PING)
                T.copy(state_fragment, state_shared_h)
                T.s_waitcnt(0, "lgkmcnt")
                T.abarrier_arrive(_WASP_STATE_READY_PING)
                T.ebarrier_sync_cnt(
                    _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
                )
                T.copy(state_shared_h, state_operand)
                T.sched_barrier()

                # As in fixed, even chunks prefetch odd W during Gemm1;
                # each pair starts by acquiring its even W from the producer.
                full_chunk_pairs = full_chunks // 2
                for pair in T.serial(full_chunk_pairs):
                    cid = pair * 2
                    T.abarrier_try_wait(_WASP_W_READY_PING, cid & 1)
                    T.copy(w_shared, w_operand_0)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.abarrier_arrive(_WASP_W_FREE_PING)
                    run_consumer_chunk(
                        cid,
                        w_operand_0,
                        w_operand_1,
                        True,
                        True,
                        w, g, h, v_new, 0, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared,
                        k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                        token_begin,
                    )
                    run_consumer_chunk(
                        cid + 1,
                        w_operand_1,
                        w_operand_0,
                        False,
                        True,
                        w, g, h, v_new, 0, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared,
                        k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                        token_begin,
                    )

                if full_chunks % 2 != 0:
                    tail_cid = full_chunk_pairs * 2
                    T.abarrier_try_wait(
                        _WASP_W_READY_PING, tail_cid & 1
                    )
                    T.copy(w_shared, w_operand_0)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.abarrier_arrive(_WASP_W_FREE_PING)
                    run_consumer_chunk(
                        tail_cid,
                        w_operand_0,
                        w_operand_1,
                        False,
                        True,
                        w, g, h, v_new, 0, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared,
                        k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                        token_begin,
                    )

                if tail_size > 0:
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        if it < tail_size:
                            w_operand_0[it, ik] = w[
                                0, tail_base + it, hid, ik
                            ]
                        else:
                            w_operand_0[it, ik] = T.cast(0, qkv_dtype)
                    for it, ik in T.Parallel(chunk_size, dim_k):
                        if it < tail_size:
                            k_fragment[it, ik] = k[
                                0, tail_base + it, hid // (heads // kv_heads), ik
                            ]
                        else:
                            k_fragment[it, ik] = T.cast(0, qkv_dtype)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            u_fragment[it, iv] = u[
                                0, tail_base + it, hid, v_offset + iv
                            ]
                        else:
                            u_fragment[it, iv] = T.cast(0, qkv_dtype)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            g_fragment[it, iv] = g[
                                0, tail_base + it, hid
                            ]
                        else:
                            g_fragment[it, iv] = T.float32(0.0)
                    T.s_waitcnt(0, "vmcnt")

                    if transpose_state_layout:
                        T.copy(
                            state_fragment,
                            h[
                                0,
                                chunk_begin + full_chunks,
                                hid,
                                v_offset : v_offset + block_v,
                                0:dim_k,
                            ],
                        )
                    else:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            h[
                                0,
                                chunk_begin + full_chunks,
                                hid,
                                ik,
                                v_offset + iv,
                            ] = state_fragment[iv, ik]

                    T.sched_barrier()
                    T.gemm(
                        w_operand_0,
                        state_operand,
                        v_new_fragment,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                        k_pack=2,
                    )
                    T.sched_barrier()
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            v_new_fragment[it, iv] = (
                                T.cast(u_fragment[it, iv], "float32")
                                - v_new_fragment[it, iv]
                            )
                        else:
                            v_new_fragment[it, iv] = T.float32(0.0)

                    if save_new_value:
                        for it, iv in T.Parallel(chunk_size, block_v):
                            if it < tail_size:
                                v_new[
                                    0,
                                    tail_base + it,
                                    hid,
                                    v_offset + iv,
                                ] = T.cast(v_new_fragment[it, iv], qkv_dtype)

                    tail_g_last = g[0, tail_base + tail_size - 1, hid]
                    T.s_waitcnt(0, "vmcnt")
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            diff = tail_g_last - g_fragment[it, iv]
                            v_new_fragment[it, iv] = T.if_then_else(
                                diff <= 0,
                                v_new_fragment[it, iv]
                                * T.exp2(diff * _LOG2E),
                                T.float32(0.0),
                            )
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= T.exp2(
                            tail_g_last * _LOG2E
                        )

                    T.copy(v_new_fragment, v_new_shared)
                    T.sched_barrier()
                    T.s_waitcnt(0, "lgkmcnt")
                    T.ebarrier_sync_cnt(
                        _WASP_CONSUMER_EBAR_ID, _WASP_CONSUMER_WAVES
                    )
                    T.copy(v_new_shared, v_new_operand)
                    T.sched_barrier()
                    T.gemm(
                        v_new_operand,
                        k_fragment,
                        state_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=False,
                        k_pack=1,
                    )

                if store_final_state:
                    if inplace_final_state:
                        if state_idx >= 0 and state_idx < state_rows:
                            for iv, ik in T.Parallel(block_v, dim_k):
                                if transpose_state_layout:
                                    final_state[
                                        state_idx, hid, v_offset + iv, ik
                                    ] = state_fragment[iv, ik]
                                else:
                                    final_state[
                                        state_idx, hid, ik, v_offset + iv
                                    ] = state_fragment[iv, ik]
                    else:
                        for iv, ik in T.Parallel(block_v, dim_k):
                            if transpose_state_layout:
                                final_state[
                                    sequence_id, hid, v_offset + iv, ik
                                ] = state_fragment[iv, ik]
                            else:
                                final_state[
                                    sequence_id, hid, ik, v_offset + iv
                                ] = state_fragment[iv, ik]

    return chunk_gated_delta_rule_fwd_h_wasp_4p4c_gfx946_varlen_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def _build_chunk_h_wasp_4p4c_varlen(
    batch,
    heads,
    kv_heads,
    dim_k,
    dim_v,
    state_rows,
    qkv_dtype,
    gate_dtype,
    chunk_size,
    use_initial_state,
    store_final_state,
    inplace_final_state,
    save_new_value,
    transpose_state_layout,
    is_varlen=True,
    block_v=128,
):
    # Dedicated packed-varlen path; fixed BV64 remains in the original builder.
    assert (
        batch == 1
        and dim_k == 128
        and dim_v == 128
        and block_v in (32, 64, 128)
        and chunk_size == 64
    )
    assert is_varlen and transpose_state_layout
    # Each consumer wave owns V32; smaller BV uses fewer consumer waves.
    BT = chunk_size
    BK = dim_k
    BV = block_v
    UV_WIDTH = max(BK, BV)
    # Instruction/warp geometry, not arbitrary shape parameters.
    K_STEP = 32
    T_STEP = 32
    GATE_ROWS = 16
    PRODUCER_THREADS = 256
    NUM_THREADS = PRODUCER_THREADS + (BV // 32) * 64
    total = T.dynamic("seq_len")
    n = T.dynamic("num_sequences")
    nc = T.dynamic("num_chunks")
    cs = T.dynamic("cu_size")
    co = T.dynamic("chunk_offsets_size")

    def make_storage(use_xor):
        def address(t, c):
            group = (c // 8) ^ ((t // 2) % 4) if use_xor else c // 8
            return 8 * (
                256 * (t // 16)
                + 64 * (t % 4)
                + (16 * ((t % 16) // 4) + group + 4 * (t % 2)) % 64
            ) + c % 8

        def inverse(r, c):
            flat = r * UV_WIDTH + c
            point = flat // 8
            row_low = (point % 256) // 64
            packed = (point % 64 - 4 * (row_low % 2)) % 64
            row = 16 * (point // 256) + 4 * (packed // 16) + row_low
            group = (packed % 16) ^ ((row // 2) % 4) if use_xor else packed % 16
            return row, group * 8 + flat % 8

        x = T.Layout(
            (BT, UV_WIDTH),
            lambda t, c: (address(t, c) // UV_WIDTH, address(t, c) % UV_WIDTH),
        )
        return x.with_inverse(T.Layout((BT, UV_WIDTH), inverse))

    uv_layout = make_storage(True)

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_varlen_kernel(
        k: T.Tensor((batch, total, kv_heads, dim_k), qkv_dtype),
        w: T.Tensor((batch, total, heads, dim_k), qkv_dtype),
        u: T.Tensor((batch, total, heads, dim_v), qkv_dtype),
        g: T.Tensor((batch, total, heads), gate_dtype),
        initial: T.Tensor((state_rows, heads, dim_v, dim_k), "float32"),
        indices: T.Tensor((n,), 'int32'),
        cu: T.Tensor((cs,), 'int32'),
        offsets: T.Tensor((co,), 'int32'),
        h: T.Tensor((batch, nc, heads, dim_v, dim_k), qkv_dtype),
        vn: T.Tensor((batch, total, heads, dim_v), qkv_dtype),
        final: T.Tensor(
            (state_rows if inplace_final_state else n, heads, dim_v, dim_k),
            "float32",
        ),
    ):
        with T.Kernel(T.ceildiv(dim_v, BV), n * heads, threads=NUM_THREADS) as (vid, nh):
            v_start = vid * BV
            seq = nh // heads
            hid = nh % heads
            bos = cu[seq]
            length = cu[seq + 1] - bos
            start = offsets[seq]
            slot = indices[seq]
            # Contracts of well-formed packed sequence/chunk metadata.
            # Do not assume slot is valid: -1 retains the zero-initial behavior.
            T.assume(bos >= 0)
            T.assume(length >= 0)
            T.assume(bos + length <= total)
            T.assume(start >= 0)
            T.assume(start + T.ceildiv(length, BT) <= nc)
            uv = T.alloc_shared((BT, UV_WIDTH), qkv_dtype)
            ks = T.alloc_shared((BT, BK), qkv_dtype)
            T.annotate_layout({ uv: uv_layout})
            tx = T.get_thread_binding()
            if tx < PRODUCER_THREADS:
                up = T.alloc_fragment((BT, UV_WIDTH), qkv_dtype)
                if length > 0:
                    T.async_copy(w[0, bos: bos + BT, hid, :], uv, coalesced_width=8)
                for cid in T.serial(length // BT):
                    base = bos + cid * BT
                    T.async_copy(
                        k[0, base: base + BT, hid // (heads // kv_heads), :],
                        ks,
                        coalesced_width=8,
                    )
                    T.ptx_wait_group(1)  # W prefix complete; newest K group may remain
                    T.sync_threads()  # B0: W ready; K may still be in flight
                    T.copy(u[0, base: base + BT, hid, v_start: v_start + BV], up[:, 0:BV])
                    T.sync_threads()  # B1: W consumed
                    T.copy(up[:, 0:BV], uv[:, 0:BV])
                    T.ptx_wait_group(0)
                    T.sync_threads()  # B2: U/K ready
                    if cid + 1 < T.ceildiv(length, BT):
                        T.copy(w[0, base + BT: base + 2 * BT, hid, :], up)
                    T.sync_threads()  # B4: second operands read; UV free
                    if cid + 1 < T.ceildiv(length, BT):
                        T.copy(up, uv)
                if length % BT > 0:
                    base = bos + (length // BT) * BT
                    T.ptx_wait_group(0)
                    T.sync_threads()  # tail W ready
                    T.sync_threads()  # tail W masking complete
                    T.async_copy(
                        k[0, base: base + BT, hid // (heads // kv_heads), :],
                        ks,
                        coalesced_width=8,
                    )
                    T.sync_threads()  # tail W consumed
                    if BV == 128:
                        T.async_copy(u[0, base: base + BT, hid, :], uv, coalesced_width=8)
                    else:
                        # Swizzled LDS async copies require a full matrix.
                        # Reuse the existing producer registers for smaller V tiles.
                        T.copy(u[0, base: base + BT, hid, v_start: v_start + BV], up[:, 0:BV])
                        T.copy(up[:, 0:BV], uv[:, 0:BV])
                    T.ptx_wait_group(0)
                    T.sync_threads()  # tail U/K ready
                    T.sync_threads()  # tail U/K masking complete
                    T.sync_threads()  # tail gated V published
                    T.sync_threads()  # tail first operands read
                    T.sync_threads()  # tail second operands read
                    T.sync_threads()  # tail finished
            else:
                state = T.alloc_fragment((BV, BK), "float32")
                sop = T.alloc_fragment((BV, K_STEP), qkv_dtype)
                hf = T.alloc_fragment((BV, K_STEP), qkv_dtype)
                state_io_tile = T.alloc_fragment((BV, K_STEP), "float32")
                factor = T.alloc_var("float32")
                gate_prefetch = T.alloc_local((BT // GATE_ROWS,), gate_dtype)
                wo = T.alloc_fragment((BT, K_STEP), qkv_dtype)
                uf = T.alloc_fragment((BT, BV), qkv_dtype)
                val = T.alloc_fragment((BT, BV), "float32")
                vo = T.alloc_fragment((T_STEP, BV), qkv_dtype)
                ko = T.alloc_fragment((T_STEP, BK), qkv_dtype)
                gl = T.alloc_var("float32")
                if use_initial_state and slot >= 0 and slot < state_rows:
                    for bk in T.unroll(BK // K_STEP):
                        T.copy(
                            initial[slot, hid, v_start: v_start + BV, bk * K_STEP: (bk + 1) * K_STEP],
                            state_io_tile,
                        )
                        for v, kk in T.Parallel(BV, K_STEP):
                            state[v, bk * K_STEP + kk] = state_io_tile[v, kk]
                else:
                    for v, kk in T.Parallel(BV, BK):
                        state[v, kk] = T.float32(0)
                for cid in T.serial(length // BT):
                    base = bos + cid * BT
                    valid = BT
                    gl = g[0, base + valid - 1, hid]
                    T.sync_threads()
                    if valid < BT:
                        for t, kk in T.Parallel(BT, BK):
                            if t >= valid:
                                uv[t, kk] = T.cast(0, qkv_dtype)
                        T.sync_threads()
                    for kb in T.unroll(BK // K_STEP):
                        # Cast and coordinate conversion are elementwise; let
                        # the TileOp express ownership instead of scalar tx reads.
                        T.copy(state[:, kb * K_STEP: (kb + 1) * K_STEP], hf)
                        for v, kk in T.Parallel(BV, K_STEP):
                            sop[v, kk] = hf[(v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16, kk]
                        T.copy(
                            hf,
                            h[0, start + cid, hid, v_start: v_start + BV, kb * K_STEP: (kb + 1) * K_STEP],
                        )
                        T.copy(uv[0:BT, kb * K_STEP: (kb + 1) * K_STEP], wo)
                        T.gemm(
                            wo,
                            sop,
                            val,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=(kb == 0),
                            k_pack=2,
                        )
                    T.sync_threads()
                    T.sync_threads()
                    # Issue gate reads once U/K publication has completed.
                    for ts in T.unroll(BT // GATE_ROWS):
                        gate_prefetch[ts] = g[
                            0,
                            base + T.min(ts * GATE_ROWS + tx % GATE_ROWS, valid - 1),
                            hid,
                        ]
                    T.sched_barrier(0x64F)  # Allow ALU/MMAC, VMEM stores, DS writes, transcendentals.
                    if valid < BT:
                        for t, kk in T.Parallel(BT, BK):
                            if t >= valid:
                                uv[t, kk] = T.cast(0, qkv_dtype)
                                ks[t, kk] = T.cast(0, qkv_dtype)
                        T.sync_threads()
                    T.copy(uv[:, 0:BV], uf)
                    for t, v in T.Parallel(BT, BV):
                        val[t, v] = T.if_then_else(
                            t < valid,
                            T.cast(
                                uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16],
                                "float32",
                            ) - val[t, v],
                            T.float32(0),
                        )
                    if save_new_value:
                        for t, v in T.Parallel(BT, BV):
                            uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16] = T.cast(
                                val[t, v], qkv_dtype
                            )
                        T.copy(uf, vn[0, base: base + BT, hid, v_start: v_start + BV])
                    for ts in T.unroll(BT // GATE_ROWS):
                        factor = gl - gate_prefetch[ts]
                        factor = T.if_then_else(
                            factor <= 0,
                            T.exp2(factor * 1.4426950408889634),
                            T.float32(0),
                        )
                        for t, v in T.Parallel(GATE_ROWS, BV):
                            if ts * GATE_ROWS + t < valid:
                                val[ts * GATE_ROWS + t, v] *= factor
                    for v, kk in T.Parallel(BV, BK):
                        state[v, kk] *= T.exp2(gl * 1.4426950408889634)
                    for t, v in T.Parallel(BT, BV):
                        uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16] = T.cast(
                            val[t, v], qkv_dtype
                        )
                    T.copy(uf, uv[:, 0:BV])
                    T.s_waitcnt(0, "lgkmcnt")  # V is owned by the same consumer wave
                    for tp in T.unroll(BT // T_STEP):
                        T.ds_read_vector(uv[tp * T_STEP: (tp + 1) * T_STEP, 0:BV], vo, alt=2)
                        T.ds_read_vector(ks[tp * T_STEP: (tp + 1) * T_STEP, :], ko, alt=2)
                        T.s_waitcnt(0, "lgkmcnt")
                        if tp == BT // T_STEP - 1:
                            T.sync_threads()
                        T.gemm(
                            vo,
                            ko,
                            state,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=False,
                            k_pack=1,
                        )
                if length % BT > 0:
                    tailcid = length // BT
                    base = bos + tailcid * BT
                    valid = length % BT
                    gl = g[0, base + valid - 1, hid]
                    T.sync_threads()
                    if valid < BT:
                        for t, kk in T.Parallel(BT, BK):
                            if t >= valid:
                                uv[t, kk] = T.cast(0, qkv_dtype)
                        T.sync_threads()
                    for kb in T.unroll(BK // K_STEP):
                        # Cast and coordinate conversion are elementwise; let
                        # the TileOp express ownership instead of scalar tx reads.
                        T.copy(state[:, kb * K_STEP: (kb + 1) * K_STEP], hf)
                        for v, kk in T.Parallel(BV, K_STEP):
                            sop[v, kk] = hf[(v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16, kk]
                        T.copy(
                            hf,
                            h[0, start + tailcid, hid, v_start: v_start + BV, kb * K_STEP: (kb + 1) * K_STEP],
                        )
                        T.copy(uv[0:BT, kb * K_STEP: (kb + 1) * K_STEP], wo)
                        T.gemm(
                            wo,
                            sop,
                            val,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=(kb == 0),
                            k_pack=2,
                        )
                    T.sync_threads()
                    T.sync_threads()
                    # Issue gate reads once U/K publication has completed.
                    for ts in T.unroll(BT // GATE_ROWS):
                        gate_prefetch[ts] = g[
                            0,
                            base + T.min(ts * GATE_ROWS + tx % GATE_ROWS, valid - 1),
                            hid,
                        ]
                    T.sched_barrier(0x64F)  # Allow ALU/MMAC, VMEM stores, DS writes, transcendentals.
                    if valid < BT:
                        for t, kk in T.Parallel(BT, BK):
                            if t >= valid:
                                uv[t, kk] = T.cast(0, qkv_dtype)
                                ks[t, kk] = T.cast(0, qkv_dtype)
                        T.sync_threads()
                    T.copy(uv[:, 0:BV], uf)
                    for t, v in T.Parallel(BT, BV):
                        val[t, v] = T.if_then_else(
                            t < valid,
                            T.cast(
                                uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16],
                                "float32",
                            ) - val[t, v],
                            T.float32(0),
                        )
                    if save_new_value:
                        for t, v in T.Parallel(BT, BV):
                            uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16] = T.cast(
                                val[t, v], qkv_dtype
                            )
                        T.copy(
                            uf[0: valid, :],
                            vn[0, base: base + valid, hid, v_start: v_start + BV],
                        )
                    for ts in T.unroll(BT // GATE_ROWS):
                        factor = gl - gate_prefetch[ts]
                        factor = T.if_then_else(
                            factor <= 0,
                            T.exp2(factor * 1.4426950408889634),
                            T.float32(0),
                        )
                        for t, v in T.Parallel(GATE_ROWS, BV):
                            if ts * GATE_ROWS + t < valid:
                                val[ts * GATE_ROWS + t, v] *= factor
                    for v, kk in T.Parallel(BV, BK):
                        state[v, kk] *= T.exp2(gl * 1.4426950408889634)
                    for t, v in T.Parallel(BT, BV):
                        uf[t, (v // 32) * 32 + 2 * (v % 16) + (v % 32) // 16] = T.cast(
                            val[t, v], qkv_dtype
                        )
                    T.copy(uf, uv[:, 0:BV])
                    T.sync_threads()
                    for tp in T.unroll(BT // T_STEP):
                        T.ds_read_vector(uv[tp * T_STEP: (tp + 1) * T_STEP, 0:BV], vo, alt=2)
                        T.ds_read_vector(ks[tp * T_STEP: (tp + 1) * T_STEP, :], ko, alt=2)
                        T.s_waitcnt(0, "lgkmcnt")
                        T.sync_threads()
                        T.gemm(
                            vo,
                            ko,
                            state,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=False,
                            k_pack=1,
                        )
                    T.sync_threads()
                if store_final_state:
                    if inplace_final_state:
                        if slot >= 0 and slot < state_rows:
                            for bk in T.unroll(BK // K_STEP):
                                for v, kk in T.Parallel(BV, K_STEP):
                                    state_io_tile[v, kk] = state[v, bk * K_STEP + kk]
                                T.copy(
                                    state_io_tile,
                                    final[slot, hid, v_start: v_start + BV, bk * K_STEP: (bk + 1) * K_STEP],
                                )
                    else:
                        for bk in T.unroll(BK // K_STEP):
                            for v, kk in T.Parallel(BV, K_STEP):
                                state_io_tile[v, kk] = state[v, bk * K_STEP + kk]
                            T.copy(
                                state_io_tile,
                                final[seq, hid, v_start: v_start + BV, bk * K_STEP: (bk + 1) * K_STEP],
                            )
    return chunk_gated_delta_rule_fwd_h_wasp_4p4c_varlen_kernel

@functools.lru_cache(None)
def _get_wasp_4p4c_varlen_kernel(*args):
    return _build_chunk_h_wasp_4p4c_varlen(*args)


@functools.lru_cache(maxsize=128)
def _get_copy_pipeline_kernel(*args):
    return _build_chunk_h_copy_pipeline(*args)


@functools.lru_cache(maxsize=128)
def _get_copy_pipeline_varlen_kernel(*args):
    return _build_chunk_h_copy_pipeline_varlen(*args)


@functools.lru_cache(maxsize=128)
def _get_wasp_4p4c_kernel(*args):
    return _build_chunk_h_wasp_4p4c(*args)


@functools.lru_cache(maxsize=128)
def _get_wasp_4p4c_gfx946_kernel(*args):
    return _build_chunk_h_wasp_4p4c_gfx946(*args)


def _get_gfx946_block_v() -> int:
    value = os.environ.get("TILELANG_FLA_FORCE_BV", "32")
    if value not in ("32", "64"):
        raise ValueError("TILELANG_FLA_FORCE_BV must be 32 or 64")
    return int(value)



def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = True,
    null_state_index: int = -1,
    kernel_cfg: Optional[dict] = None,
):
    implementation = (
        "wasp_4p4c"
        if kernel_cfg is None
        else kernel_cfg.get("implementation", "wasp_4p4c")
    )
    if kernel_cfg is not None:
        unknown = set(kernel_cfg) - {"implementation", "block_v"}
        if unknown:
            raise ValueError(f"unsupported kernel_cfg keys: {sorted(unknown)}")
    if implementation not in (
        "wasp_4p4c",
        "wasp_4p4c_gfx946",
    ):
        raise NotImplementedError(
            f"unsupported TileLang implementation: {kernel_cfg}"
        )
    block_v = _get_gfx946_block_v() if implementation == "wasp_4p4c_gfx946" else 32
    if null_state_index not in (-1, 0):
        raise ValueError(
            "null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI)."
        )
    require_first_version(
        chunk_size=chunk_size,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        cu_seqlens=None,
        chunk_indices=None,
        kernel_cfg=None,
    )
    if g is None or gk is not None:
        raise NotImplementedError(
            "the first TileLang version requires g and does not support gk."
        )
    if inplace_final_state and initial_state is None:
        raise ValueError(
            "initial_state is required when inplace_final_state=True."
        )
    if inplace_final_state and not output_final_state:
        raise ValueError(
            "output_final_state must be True when inplace_final_state=True."
        )
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("k, w, u and g must use BTHD/BTH layouts.")
    batch, seq_len, kv_heads, dim_k = k.shape
    heads, dim_v = w.shape[2], u.shape[-1]
    is_varlen = cu_seqlens is not None
    if chunk_indices is not None and not is_varlen:
        raise ValueError("chunk_indices requires cu_seqlens.")
    if is_varlen:
        if batch != 1:
            raise ValueError("varlen inputs must use packed batch size B=1.")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D tensor with at least two entries.")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have int32 or int64 dtype.")
        if cu_seqlens.device != k.device:
            raise ValueError("cu_seqlens and k must be on the same device.")
        if chunk_indices is not None and (
            chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_indices.dtype not in (torch.int32, torch.int64)
            or chunk_indices.device != k.device
        ):
            raise ValueError(
                "chunk_indices must have shape [NT, 2], int32/int64 dtype, "
                "and reside on the input device."
            )
        if implementation == "wasp_4p4c" and cu_seqlens.numel() == 2:
            # Validate before discarding packed metadata. This host read is
            # confined to the single-sequence normalization path.
            if cu_seqlens.tolist() != [0, seq_len]:
                raise ValueError("single-sequence cu_seqlens must be [0, T].")
            if chunk_indices is not None and chunk_indices.shape[0] != (
                seq_len + chunk_size - 1
            ) // chunk_size:
                raise ValueError("single-sequence chunk_indices has the wrong chunk count.")
            is_varlen = False

    if is_varlen:
        cu_seqlens_kernel = cu_seqlens.to(dtype=torch.int32)
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_offsets_kernel = prepare_chunk_offsets(
            cu_seqlens, chunk_size
        )[:-1].to(dtype=torch.int32)
        num_sequences = cu_seqlens.numel() - 1
        num_chunks = chunk_indices.shape[0]
    else:
        cu_seqlens_kernel = None
        chunk_offsets_kernel = None
        num_sequences = batch
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads:
        raise ValueError("H and Hg must be positive and H must be divisible by Hg.")
    if tuple(w.shape) != (batch, seq_len, heads, dim_k):
        raise ValueError("w must have shape [B, T, H, K].")
    if tuple(u.shape) != (batch, seq_len, heads, dim_v):
        raise ValueError("u must have shape [B, T, H, V].")
    if tuple(g.shape) != (batch, seq_len, heads):
        raise ValueError("g must have shape [B, T, H].")
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the first TileLang version supports fp16 and bf16 inputs.")
    if w.dtype != k.dtype or u.dtype != k.dtype or g.dtype != torch.float32:
        raise TypeError("k/w/u must share fp16 or bf16 dtype and g must be float32.")
    if dim_k % 16 or dim_v % 16:
        raise ValueError("K and V must be multiples of 16.")
    if batch == 0 or seq_len == 0 or heads == 0:
        raise ValueError("B, T and H must be positive.")
    if not all(tensor.is_cuda for tensor in (k, w, u, g)):
        raise ValueError("all inputs must be CUDA/HIP tensors.")

    state_trailing = (
        (heads, dim_v, dim_k)
        if transpose_state_layout
        else (heads, dim_k, dim_v)
    )
    if initial_state is None:
        if initial_state_indices is not None:
            raise ValueError("initial_state_indices requires initial_state.")
        state_rows = num_sequences
        state_buffer = torch.empty(
            num_sequences, *state_trailing, dtype=torch.float32, device=k.device
        )
        state_indices = torch.arange(
            num_sequences, dtype=torch.int32, device=k.device
        )
    else:
        if initial_state.dtype != torch.float32:
            raise TypeError(
                "the first TileLang version requires float32 initial_state."
            )
        if initial_state.device != k.device:
            raise ValueError("initial_state and k must be on the same device.")
        if (
            initial_state.ndim != 4
            or tuple(initial_state.shape[1:]) != state_trailing
        ):
            layout = (
                "[N, H, V, K]"
                if transpose_state_layout
                else "[N, H, K, V]"
            )
            raise ValueError(f"initial_state must have shape {layout}.")
        state_rows = initial_state.shape[0]
        state_buffer = initial_state
        if initial_state_indices is None:
            if state_rows < num_sequences:
                raise ValueError(
                    "initial_state has fewer rows than the number of sequences."
                )
            state_indices = torch.arange(
                num_sequences, dtype=torch.int32, device=k.device
            )
        else:
            if initial_state_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("initial_state_indices must have int32 or int64 dtype.")
            if tuple(initial_state_indices.shape) != (num_sequences,):
                raise ValueError(
                    "initial_state_indices must have shape "
                    f"({num_sequences},)."
                )
            if initial_state_indices.device != k.device:
                raise ValueError(
                    "initial_state_indices and k must be on the same device."
                )
            state_indices = initial_state_indices.to(dtype=torch.int32)

    if null_state_index == 0:
        state_indices = torch.where(
            state_indices > 0,
            state_indices,
            state_indices.new_full((), -1),
        )

    require_contiguous(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=state_buffer,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens_kernel,
        chunk_offsets=chunk_offsets_kernel,
    )
    h = torch.empty(
        batch, num_chunks, *state_trailing, dtype=k.dtype, device=k.device
    )
    v_new_buffer = torch.empty_like(u)
    if inplace_final_state:
        final_state_buffer = state_buffer
    else:
        final_state_buffer = torch.empty(
            num_sequences,
            *state_trailing,
            dtype=torch.float32,
            device=k.device,
        )
    supported = (
        chunk_size == 64
        and dim_k == 128
        and dim_v % block_v == 0
        and k.dtype == torch.bfloat16
        and output_final_state
        and save_new_value
    )
    if not supported or not transpose_state_layout or num_chunks < 1:
        raise NotImplementedError(
            "TileLang kernels require bf16 BT64/K128/V32 shapes, VK "
            "state layout, final_state and v_new enabled."
        )
    if implementation == "wasp_4p4c" and is_varlen and dim_v != 128:
        raise NotImplementedError("packed-varlen wasp_4p4c requires V=128.")

    default_bv = 128 if implementation == "wasp_4p4c" and is_varlen else 64
    block_v = (
        _get_gfx946_block_v()
        if implementation == "wasp_4p4c_gfx946"
        else default_bv if kernel_cfg is None else kernel_cfg.get("block_v", default_bv)
    )
    if implementation == "wasp_4p4c_gfx946":
        if dim_v % block_v:
            raise ValueError("gfx946 requires V divisible by the selected BV.")
    elif is_varlen:
        if block_v not in (32, 64, 128):
            raise ValueError("packed-varlen block_v must be 32, 64 or 128.")
    elif block_v not in (32, 64) or dim_v % block_v:
        raise ValueError("fixed block_v must be 32 or 64 and divide V.")

    kernel_args = (
        batch,
        heads,
        kv_heads,
        dim_k,
        dim_v,
        state_rows,
        tilelang_dtype(k.dtype),
        tilelang_dtype(g.dtype),
        chunk_size,
        initial_state is not None,
        output_final_state,
        inplace_final_state,
        save_new_value,
        transpose_state_layout,
    )
    if implementation == "wasp_4p4c_gfx946":
        kernel = _get_wasp_4p4c_gfx946_kernel(
            *kernel_args, is_varlen,
            False if is_varlen else seq_len % chunk_size != 0, block_v,
        )
    elif is_varlen:
        kernel = _get_wasp_4p4c_varlen_kernel(*kernel_args, True, block_v)
    else:
        kernel = _get_wasp_4p4c_kernel(
            *kernel_args, block_v, seq_len % chunk_size != 0,
        )
    if is_varlen:
        kernel(
            k,
            w,
            u,
            g,
            state_buffer,
            state_indices,
            cu_seqlens_kernel,
            chunk_offsets_kernel,
            h,
            v_new_buffer,
            final_state_buffer,
        )
    else:
        kernel(
            k,
            w,
            u,
            g,
            state_buffer,
            state_indices,
            h,
            v_new_buffer,
            final_state_buffer,
        )
    return (
        h,
        v_new_buffer if save_new_value else None,
        final_state_buffer if output_final_state else None,
    )


__all__ = ["chunk_gated_delta_rule_fwd_h"]
