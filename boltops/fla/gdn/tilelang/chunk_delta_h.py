# SPDX-License-Identifier: MIT
# Adapted from TileLang examples/gdn/example_chunk_delta_h.py.
"""TileLang kernels for the GDN chunk state recurrence."""

import functools
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
                hid,
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
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
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
            T.use_swizzle(10)

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
                k[bid, 0:chunk_size, hid, 0:dim_k],
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
                            bid, tail_base + it, hid, ik
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
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
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
            T.use_swizzle(10)
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
                    k[data_bid, token_begin : token_begin + chunk_size, hid, 0:dim_k],
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
                            hid,
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
                            data_bid, token_begin + tail_base + it, hid, ik
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
    block_v: int = 32,
):
    """Build the four-producer/four-consumer WASP kernel."""
    seq_len = T.dynamic("seq_len")
    num_chunks = seq_len // chunk_size
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
        prefetch_following_w,
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
    ):
        base = cid * chunk_size
        next_base = base + chunk_size

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
        T.sync_warp()

        # h update (store)
        if transpose_state_layout:
            T.copy(
                state_fragment,
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
                h[bid, cid, hid, ik, v_offset + iv] = state_fragment[iv, ik]

        for it, iv in T.Parallel(chunk_size, block_v):
            v_new_fragment[it, iv] = (
                T.cast(u_fragment[it, iv], "float32")
                - v_new_fragment[it, iv]
            )

        # Save the ungated v_new_fragment for chunk cid directly from consumer
        # VGPRs.
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

        # Barrier 2: wait for producer publication of k_shared for chunk cid.
        T.sync_warp()

        T.copy(k_shared, k_fragment)
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

        T.copy(v_new_fragment, v_new_shared)
        T.sched_barrier()

        # Barrier 3: complete k_shared -> k_fragment and publish v_new_shared
        # for chunk cid. Producers may then overwrite k_shared with data for
        # chunk cid+1.
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()
        T.copy(v_new_shared, v_new_operand)
        if prefetch_following_w:
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
        T.copy(state_fragment, state_shared_h)

        # Prefetch g_fragment for chunk cid+1 after the state LDS writes,
        # matching the current copy-pipeline experiment.
        if has_following_chunk:
            for it, iv in T.Parallel(chunk_size, block_v):
                g_fragment[it, iv] = g[bid, next_base + it, hid]
        T.sched_barrier()

        # Barrier 4
        # state_shared_h ready
        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        if has_following_chunk:
            T.copy(state_shared_h, state_operand)
            T.sched_barrier()

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_kernel(
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
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
            T.use_swizzle(10)

            if tx < _PRODUCER_THREADS:
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
                    k[bid, 0:chunk_size, hid, 0:dim_k],
                    k_shared,
                )

                # Barrier 0: state_shared_h/u_shared ready; k_shared outstanding.
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
                            hid,
                            0:dim_k,
                        ],
                        k_shared,
                    )
                    # Complete u_shared for chunk cid+1; retain its K copy.
                    T.ptx_wait_group(1)

                    # Barrier 4: next state_shared_h/u_shared ready.
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
                w_operand_1 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_operand = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                state_idx = T.alloc_var("int32")

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

                # Keep W on consumer VGPRs. Each chunk prefetches W_i+1 into
                # the alternate slot before consuming W_i.
                T.copy(
                    w[bid, 0:chunk_size, hid, 0:dim_k],
                    w_operand_0,
                )
                for it, iv in T.Parallel(chunk_size, block_v):
                    g_fragment[it, iv] = g[bid, it, hid]

                # Barrier 0 (prologue): publish state_shared_h for chunk 0 and
                # wait for producer publication of u_shared for chunk 0.
                T.s_waitcnt(0, "lgkmcnt")
                T.sync_warp()
                T.copy(state_shared_h, state_operand)
                T.sched_barrier()

                # Expand two chunks per hot-loop iteration with statically
                # selected W operands. This avoids a cid-parity branch in the
                # consumer path.
                steady_pair_count = T.max((num_chunks - 2) // 2, 0)
                for pair in T.Pipelined(steady_pair_count, num_stages=0):
                    cid = pair * 2
                    run_consumer_chunk(
                        cid,
                        w_operand_0,
                        w_operand_1,
                        True,
                        True,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                    run_consumer_chunk(
                        cid + 1,
                        w_operand_1,
                        w_operand_0,
                        True,
                        True,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )

                # Handle the final one, two, or three chunks outside the hot
                # pair loop. Any odd/even selection is therefore paid once,
                # rather than once per chunk.
                tail_cid = steady_pair_count * 2
                if num_chunks == 1:
                    run_consumer_chunk(
                        tail_cid,
                        w_operand_0,
                        w_operand_1,
                        False,
                        False,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                elif num_chunks % 2 != 0:
                    run_consumer_chunk(
                        tail_cid,
                        w_operand_0,
                        w_operand_1,
                        True,
                        True,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                    run_consumer_chunk(
                        tail_cid + 1,
                        w_operand_1,
                        w_operand_0,
                        True,
                        True,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                    run_consumer_chunk(
                        tail_cid + 2,
                        w_operand_0,
                        w_operand_1,
                        False,
                        False,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                else:
                    run_consumer_chunk(
                        tail_cid,
                        w_operand_0,
                        w_operand_1,
                        True,
                        True,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
                    )
                    run_consumer_chunk(
                        tail_cid + 1,
                        w_operand_1,
                        w_operand_0,
                        False,
                        False,
                        w, g, h, v_new, bid, hid, v_offset,
                        state_shared_h, state_operand, w_shared,
                        u_shared, u_fragment,
                        v_new_fragment, g_fragment, k_shared, k_fragment,
                        v_new_shared, v_new_operand, state_fragment,
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
        return chunk_gated_delta_rule_fwd_h_wasp_4p4c_kernel

    num_sequences = T.dynamic("num_sequences")
    total_num_chunks = T.dynamic("num_chunks")
    cu_size = T.dynamic("cu_size")
    chunk_offsets_size = T.dynamic("chunk_offsets_size")
    varlen_chunk_state_shape = (
        (batch, total_num_chunks, heads, dim_v, dim_k)
        if transpose_state_layout
        else (batch, total_num_chunks, heads, dim_k, dim_v)
    )

    @T.macro
    def run_consumer_chunk_varlen(
        cid,
        w_current,
        w_next,
        prefetch_following_w,
        has_following_chunk,
        w,
        g,
        h,
        v_new,
        token_begin,
        chunk_begin,
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
    ):
        base = token_begin + cid * chunk_size
        next_base = base + chunk_size

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

        if transpose_state_layout:
            T.copy(
                state_fragment,
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
                h[0, chunk_begin + cid, hid, ik, v_offset + iv] = (
                    state_fragment[iv, ik]
                )

        for it, iv in T.Parallel(chunk_size, block_v):
            v_new_fragment[it, iv] = (
                T.cast(u_fragment[it, iv], "float32")
                - v_new_fragment[it, iv]
            )

        if save_new_value:
            for it, iv in T.Parallel(chunk_size, block_v):
                v_new[0, base + it, hid, v_offset + iv] = T.cast(
                    v_new_fragment[it, iv], qkv_dtype
                )

        T.sync_warp()

        T.copy(k_shared, k_fragment)
        T.s_waitcnt(4, "lgkmcnt")

        g_last_value = g[0, base + chunk_size - 1, hid]
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

        T.copy(v_new_fragment, v_new_shared)
        T.sched_barrier()

        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()
        T.copy(v_new_shared, v_new_operand)
        if prefetch_following_w:
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
        T.copy(state_fragment, state_shared_h)

        if has_following_chunk:
            for it, iv in T.Parallel(chunk_size, block_v):
                g_fragment[it, iv] = g[0, next_base + it, hid]
        T.sched_barrier()

        T.s_waitcnt(0, "lgkmcnt")
        T.sync_warp()

        if has_following_chunk:
            T.copy(state_shared_h, state_operand)
            T.sched_barrier()

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_wasp_4p4c_varlen_kernel(
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
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
            T.use_swizzle(10)
            sequence_id, hid = bbh // heads, bbh % heads
            v_offset = vid * block_v
            tx = T.get_thread_binding()
            token_begin = cu_seqlens[sequence_id]
            actual_seq_len = cu_seqlens[sequence_id + 1] - token_begin
            chunk_begin = chunk_offsets[sequence_id]
            full_chunks = actual_seq_len // chunk_size
            tail_size = actual_seq_len % chunk_size
            tail_base = token_begin + full_chunks * chunk_size

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

            if tx < _PRODUCER_THREADS:
                if full_chunks > 0:
                    T.async_copy(
                        u[
                            0,
                            token_begin : token_begin + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    T.async_copy(
                        k[
                            0,
                            token_begin : token_begin + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        k_shared,
                    )

                    T.ptx_wait_group(1)
                    T.sync_warp()

                    for cid in T.serial(full_chunks - 1):
                        next_base = token_begin + (cid + 1) * chunk_size

                        T.async_copy(
                            w[
                                0,
                                next_base : next_base + chunk_size,
                                hid,
                                0:dim_k,
                            ],
                            w_shared,
                        )
                        T.sync_warp()

                        T.async_copy(
                            u[
                                0,
                                next_base : next_base + chunk_size,
                                hid,
                                v_offset : v_offset + block_v,
                            ],
                            u_shared,
                        )
                        T.ptx_wait_group(2)
                        T.sync_warp()

                        T.ptx_wait_group(1)
                        T.sync_warp()

                        T.async_copy(
                            k[
                                0,
                                next_base : next_base + chunk_size,
                                hid,
                                0:dim_k,
                            ],
                            k_shared,
                        )
                        T.ptx_wait_group(1)
                        T.sync_warp()

                    T.sync_warp()
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.sync_warp()
                    T.sync_warp()

                if tail_size > 0:
                    T.async_copy(
                        w[
                            0,
                            tail_base : tail_base + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        w_shared,
                    )
                    T.async_copy(
                        k[
                            0,
                            tail_base : tail_base + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        k_shared,
                    )
                    T.async_copy(
                        u[
                            0,
                            tail_base : tail_base + chunk_size,
                            hid,
                            v_offset : v_offset + block_v,
                        ],
                        u_shared,
                    )
                    T.ptx_wait_group(0)
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
                w_operand_1 = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_operand = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                state_idx = T.alloc_var("int32")
                tail_g_last = T.alloc_var("float32")
                tail_g_last_scale = T.alloc_var("float32")

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
                    T.copy(
                        w[
                            0,
                            token_begin : token_begin + chunk_size,
                            hid,
                            0:dim_k,
                        ],
                        w_operand_0,
                    )
                    for it, iv in T.Parallel(chunk_size, block_v):
                        g_fragment[it, iv] = g[0, token_begin + it, hid]

                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()
                    T.copy(state_shared_h, state_operand)
                    T.sched_barrier()

                    steady_pair_count = T.max((full_chunks - 2) // 2, 0)
                    for pair in T.Pipelined(steady_pair_count, num_stages=0):
                        cid = pair * 2
                        run_consumer_chunk_varlen(
                            cid, w_operand_0, w_operand_1, True, True,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                        run_consumer_chunk_varlen(
                            cid + 1, w_operand_1, w_operand_0, True, True,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )

                    tail_cid = steady_pair_count * 2
                    if full_chunks == 1:
                        run_consumer_chunk_varlen(
                            tail_cid, w_operand_0, w_operand_1, False, False,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                    elif full_chunks % 2 != 0:
                        run_consumer_chunk_varlen(
                            tail_cid, w_operand_0, w_operand_1, True, True,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                        run_consumer_chunk_varlen(
                            tail_cid + 1, w_operand_1, w_operand_0, True, True,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                        run_consumer_chunk_varlen(
                            tail_cid + 2, w_operand_0, w_operand_1, False, False,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                    else:
                        run_consumer_chunk_varlen(
                            tail_cid, w_operand_0, w_operand_1, True, True,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )
                        run_consumer_chunk_varlen(
                            tail_cid + 1, w_operand_1, w_operand_0, False, False,
                            w, g, h, v_new, token_begin, chunk_begin,
                            hid, v_offset, state_shared_h, state_operand,
                            w_shared, u_shared, u_fragment, v_new_fragment,
                            g_fragment, k_shared, k_fragment, v_new_shared,
                            v_new_operand, state_fragment,
                        )

                if tail_size > 0:
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()

                    T.copy(state_shared_h, state_operand)
                    T.copy(w_shared, w_operand_0)
                    T.copy(u_shared, u_fragment)
                    T.copy(k_shared, k_fragment)
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            g_fragment[it, iv] = g[0, tail_base + it, hid]
                        else:
                            g_fragment[it, iv] = T.float32(0.0)
                    T.s_waitcnt(0, "lgkmcnt")

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
                                v_new[0, tail_base + it, hid, v_offset + iv] = (
                                    T.cast(v_new_fragment[it, iv], qkv_dtype)
                                )

                    tail_g_last = g[0, tail_base + tail_size - 1, hid]
                    T.s_waitcnt(6, "vmcnt")
                    for it, iv in T.Parallel(chunk_size, block_v):
                        if it < tail_size:
                            diff = tail_g_last - g_fragment[it, iv]
                            v_new_fragment[it, iv] = T.if_then_else(
                                diff <= 0,
                                v_new_fragment[it, iv]
                                * T.exp2(diff * _LOG2E),
                                T.float32(0.0),
                            )
                    tail_g_last_scale = T.exp2(tail_g_last * _LOG2E)
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= tail_g_last_scale

                    T.copy(v_new_fragment, v_new_shared)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()

                    T.copy(v_new_shared, v_new_operand)
                    T.s_waitcnt(0, "lgkmcnt")
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

    return chunk_gated_delta_rule_fwd_h_wasp_4p4c_varlen_kernel




@functools.lru_cache(maxsize=128)
def _get_copy_pipeline_kernel(*args):
    return _build_chunk_h_copy_pipeline(*args)


@functools.lru_cache(maxsize=128)
def _get_copy_pipeline_varlen_kernel(*args):
    return _build_chunk_h_copy_pipeline_varlen(*args)


@functools.lru_cache(maxsize=128)
def _get_wasp_4p4c_kernel(*args):
    return _build_chunk_h_wasp_4p4c(*args)



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
        "copy-pipeline"
        if kernel_cfg is None
        else kernel_cfg.get("implementation")
    )
    if implementation not in ("copy-pipeline", "wasp_4p4c"):
        raise NotImplementedError(
            f"unsupported TileLang implementation: {kernel_cfg}"
        )
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
        cu_seqlens_kernel = cu_seqlens.to(dtype=torch.int32)
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        elif (
            chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_indices.dtype not in (torch.int32, torch.int64)
            or chunk_indices.device != k.device
        ):
            raise ValueError(
                "chunk_indices must have shape [NT, 2], int32/int64 dtype, "
                "and reside on the input device."
            )
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
    if kv_heads != heads:
        raise NotImplementedError("the first TileLang version requires Hg == H.")
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
        and dim_v % 32 == 0
        and k.dtype == torch.bfloat16
        and output_final_state
        and save_new_value
    )
    if (
        not supported
        or not transpose_state_layout
        or num_chunks < 1
        or (
            implementation == "copy-pipeline"
            and not is_varlen
            and seq_len < chunk_size
        )
        or (
            implementation == "wasp_4p4c"
            and not is_varlen
            and seq_len % chunk_size != 0
        )
    ):
        raise NotImplementedError(
            "TileLang kernels require bf16 BT64/K128/V32 shapes, VK "
            "state layout, final_state and v_new enabled; fixed copy-pipeline "
            "needs T >= 64, and fixed WASP requires T divisible by 64."
        )
    if implementation == "wasp_4p4c":
        get_kernel = _get_wasp_4p4c_kernel
    elif is_varlen:
        get_kernel = _get_copy_pipeline_varlen_kernel
    else:
        get_kernel = _get_copy_pipeline_kernel
    kernel_args = (
        batch,
        heads,
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
    if is_varlen and implementation == "copy-pipeline":
        # Compile the fixed BV32 pipeline as a separate packed-varlen kernel.
        # The fixed builder and its generated code remain unchanged.
        kernel = get_kernel(*kernel_args, 32, 256)
    elif is_varlen:
        kernel = get_kernel(*kernel_args, True)
    else:
        kernel = get_kernel(*kernel_args)
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
