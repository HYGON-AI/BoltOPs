# Copyright (c) Tile-AI.
# SPDX-License-Identifier: MIT
# Adapted from TileLang examples/gdn/example_chunk_delta_h.py.
"""Experimental ALT-formatted-read TileLang GDN chunk-state kernel."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from .common import require_contiguous, require_first_version, tilelang_dtype

_LOG2E = 1.4426950408889634
_PRODUCER_THREADS = 256
_TOTAL_THREADS = 512

_GDN_K_ALT_SOURCE = r"""
#include <tl_templates/hcu/core.hpp>

using gdn_i32x4 = int32_t __attribute__((ext_vector_type(4)));

TL_DEVICE void gdn_prefetch_k_alt_bf16(
    const __bf16* src, TL_LDS_ADDR __bf16* dst, int row_stride) {
  const int warp = static_cast<int>(threadIdx.x) / 64;
  const int lane = static_cast<int>(threadIdx.x) & 63;
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  gdn_i32x4 resource = {
      static_cast<int32_t>(__builtin_amdgcn_readfirstlane(
          static_cast<uint32_t>(src_addr))),
      static_cast<int32_t>(__builtin_amdgcn_readfirstlane(
          static_cast<uint32_t>(src_addr >> 32))),
      static_cast<int32_t>(0xfffffffeu),
      static_cast<int32_t>(0x00020000u)};
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    const int row = warp + (lane / 16) * 4 + stage * 16;
    const int col = (lane & 15) * 8;
    const int byte_offset = (row * row_stride + col) * sizeof(__bf16);
    const int lds_offset = warp * 1024 + stage * 4096;
    const int wrap = (warp & 1) ? (4 << 16) : 0;
    const int target = __builtin_amdgcn_readfirstlane(
        static_cast<int>(reinterpret_cast<uintptr_t>(dst))
        + lds_offset + wrap);
    auto* lds_addr = reinterpret_cast<TL_LDS_ADDR int*>(
        static_cast<uintptr_t>(target));
    __builtin_hcu_raw_buffer_load_lds(
        resource, lds_addr, 16, byte_offset, 0, 0, 0);
  }
}

TL_DEVICE void gdn_read_k_m32x16_alt_bf16(
    TL_LDS_ADDR __bf16* src, __bf16* dst) {
  const int consumer_warp = static_cast<int>(threadIdx.x) / 64 - 4;
  const int lane = static_cast<int>(threadIdx.x) & 63;
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    const int logical_row = stage * 16 + lane / 4;
    const int logical_col = consumer_warp * 4 + (lane & 3);
    const int partition = (logical_row / 16) * 4 + logical_row % 4;
    const int partition_row = (logical_row % 16) / 4;
    const int wrap = (logical_row & 1) * 4;
    const int segment =
        partition * 64 + ((partition_row * 16 + logical_col + wrap) & 63);
    auto value = __builtin_hcu_ds_read_m32x16_bf16_alt(
        reinterpret_cast<TL_LDS_ADDR short*>(src + segment * 8));
    *reinterpret_cast<decltype(value)*>(dst + stage * 8) = value;
  }
}

TL_DEVICE void gdn_store_state_k_pair_bf16(
    const float* src, TL_LDS_ADDR __bf16* dst, bool transpose_layout) {
  using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
  const int consumer_warp = static_cast<int>(threadIdx.x) / 64 - 4;
  const int lane = static_cast<int>(threadIdx.x) & 63;
  const int lane_k = lane & 15;
  const int lane_v = lane / 16;
  const int k_even = consumer_warp * 32 + lane_k * 2;
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int group = slot / 4;
    const int in_group = slot & 3;
    const int even_local = group * 8 + in_group;
    const int v = lane_v + slot * 4;
    const int swizzle = transpose_layout
                            ? (v % 4) * 4 + (v % 16) / 4
                            : v % 16;
    const int physical_k =
        ((k_even / 8) ^ swizzle) * 8 + k_even % 8;
    const bf16x2 pair = {
        static_cast<__bf16>(src[even_local]),
        static_cast<__bf16>(src[even_local + 4])};
    *reinterpret_cast<TL_LDS_ADDR uint32_t*>(
        dst + v * 128 + physical_k) = __builtin_bit_cast(uint32_t, pair);
  }
}
"""


@tilelang.jit(out_idx=[])
def _build_chunk_h(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    block_v: int = 32,
    threads: int = 128,
):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
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

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_kernel(
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([batch], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor([batch, *state_shape], "float32"),
    ):
        with T.Kernel(T.ceildiv(dim_v, block_v), batch, heads, threads=threads) as (
            vid,
            bid,
            hid,
        ):
            v_offset = vid * block_v
            h_shared = T.alloc_shared([dim_k, block_v], qkv_dtype)
            k_shared = T.alloc_shared([chunk_size, dim_k], qkv_dtype)
            w_shared = T.alloc_shared([chunk_size, dim_k], qkv_dtype)
            u_shared = T.alloc_shared([chunk_size, block_v], qkv_dtype)
            v_new_shared = T.alloc_shared([chunk_size, block_v], qkv_dtype)
            g_shared = T.alloc_shared([chunk_size], gate_dtype)
            h_fragment = T.alloc_fragment([dim_k, block_v], "float32")
            next_h_fragment = T.alloc_fragment([dim_k, block_v], "float32")
            v_new_fragment = T.alloc_fragment([chunk_size, block_v], "float32")
            g_last = T.alloc_var("float32")
            state_idx = T.alloc_var("int32")

            state_idx = initial_state_indices[bid]
            for ik, iv in T.Parallel(dim_k, block_v):
                if (
                    use_initial_state
                    and state_idx >= 0
                    and state_idx < state_rows
                    and v_offset + iv < dim_v
                ):
                    if transpose_state_layout:
                        h_fragment[ik, iv] = initial_state[
                            state_idx, hid, v_offset + iv, ik
                        ]
                    else:
                        h_fragment[ik, iv] = initial_state[
                            state_idx, hid, ik, v_offset + iv
                        ]
                else:
                    h_fragment[ik, iv] = T.float32(0.0)

            for cid in T.Pipelined(num_chunks, num_stages=1):
                base = cid * chunk_size
                last = T.min(base + chunk_size, seq_len) - 1
                for ik, iv in T.Parallel(dim_k, block_v):
                    if v_offset + iv < dim_v:
                        if transpose_state_layout:
                            h[bid, cid, hid, v_offset + iv, ik] = h_fragment[
                                ik, iv
                            ]
                        else:
                            h[bid, cid, hid, ik, v_offset + iv] = h_fragment[
                                ik, iv
                            ]
                        h_shared[ik, iv] = h_fragment[ik, iv]
                    else:
                        h_shared[ik, iv] = T.cast(0.0, qkv_dtype)
                for it, ik in T.Parallel(chunk_size, dim_k):
                    if base + it < seq_len:
                        k_shared[it, ik] = k[bid, base + it, hid, ik]
                        w_shared[it, ik] = w[bid, base + it, hid, ik]
                    else:
                        k_shared[it, ik] = T.cast(0.0, qkv_dtype)
                        w_shared[it, ik] = T.cast(0.0, qkv_dtype)
                for it, iv in T.Parallel(chunk_size, block_v):
                    if base + it < seq_len and v_offset + iv < dim_v:
                        u_shared[it, iv] = u[bid, base + it, hid, v_offset + iv]
                    else:
                        u_shared[it, iv] = T.cast(0.0, qkv_dtype)
                for it in T.Parallel(chunk_size):
                    if base + it < seq_len:
                        g_shared[it] = g[bid, base + it, hid]
                    else:
                        g_shared[it] = T.cast(0.0, gate_dtype)

                T.clear(v_new_fragment)
                T.gemm(w_shared, h_shared, v_new_fragment)
                for it, iv in T.Parallel(chunk_size, block_v):
                    v_new_fragment[it, iv] = (
                        T.cast(u_shared[it, iv], "float32") - v_new_fragment[it, iv]
                    )
                    if (
                        save_new_value
                        and base + it < seq_len
                        and v_offset + iv < dim_v
                    ):
                        v_new[bid, base + it, hid, v_offset + iv] = v_new_fragment[
                            it, iv
                        ]

                g_last = T.cast(g[bid, last, hid], "float32")
                for it, iv in T.Parallel(chunk_size, block_v):
                    if base + it < seq_len and v_offset + iv < dim_v:
                        diff = g_last - T.cast(g_shared[it], "float32")
                        v_new_shared[it, iv] = T.if_then_else(
                            diff <= 0,
                            v_new_fragment[it, iv] * T.exp2(diff * _LOG2E),
                            T.cast(0.0, qkv_dtype),
                        )
                    else:
                        v_new_shared[it, iv] = T.cast(0.0, qkv_dtype)

                decay = T.exp2(g_last * _LOG2E)
                for ik, iv in T.Parallel(dim_k, block_v):
                    next_h_fragment[ik, iv] = h_fragment[ik, iv] * decay
                T.gemm(
                    k_shared,
                    v_new_shared,
                    next_h_fragment,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(next_h_fragment, h_fragment)

            if store_final_state:
                for ik, iv in T.Parallel(dim_k, block_v):
                    if v_offset + iv < dim_v:
                        if transpose_state_layout:
                            final_state[bid, hid, v_offset + iv, ik] = (
                                h_fragment[ik, iv]
                            )
                        else:
                            final_state[bid, hid, ik, v_offset + iv] = (
                                h_fragment[ik, iv]
                            )

    return chunk_gated_delta_rule_fwd_h_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
)
def _build_chunk_h_producer_consumer_alt(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    state_rows: int,
    qkv_dtype: str,
    gate_dtype: str,
    chunk_size: int,
    use_initial_state: bool,
    store_final_state: bool,
    save_new_value: bool,
    transpose_state_layout: bool,
    block_v: int = 32,
):
    """Build the ALT/K-pair two-role producer/consumer kernel."""
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
    state_shared_shape = (block_v, dim_k)
    k_shared_shape = (chunk_size, dim_k)
    k_fragment_layout = T.Fragment(
        k_shared_shape,
        forward_thread_fn=lambda it, ik: (
            (ik // 32) * 64
            + (ik % 32) // 2
            + 16 * ((it % 16) // 4)
        ),
        forward_index_fn=lambda it, ik: (
            (it // 16) * 8 + (ik % 2) * 4 + it % 4
        ),
    )
    state_fragment_layout = T.Fragment(
        state_shared_shape,
        forward_thread_fn=lambda iv, ik: (
            (ik // 32) * 64 + (ik % 32) // 2 + 16 * (iv % 4)
        ),
        forward_index_fn=lambda iv, ik: (
            (iv // 16) * 8 + (ik % 2) * 4 + (iv % 16) // 4
        ),
    )
    if transpose_state_layout:
        # Transpose the two 2-bit halves of V before applying the K-vector
        # XOR. This preserves eight contiguous BF16 values for GEMM-B loads
        # while spreading scalar state publishes across all LDS banks.
        state_shared_layout = T.Layout(
            state_shared_shape,
            lambda iv, ik: (
                iv,
                ((ik // 8) ^ ((iv % 4) * 4 + (iv % 16) // 4))
                * 8
                + ik % 8,
            ),
        )
    else:
        state_shared_layout = T.Layout(
            state_shared_shape,
            lambda iv, ik: (
                iv,
                ((ik // 8) ^ (iv % 16)) * 8 + ik % 8,
            ),
        )

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_producer_consumer_alt_kernel(
        k: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        w: T.Tensor([batch, seq_len, heads, dim_k], qkv_dtype),
        u: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        g: T.Tensor([batch, seq_len, heads], gate_dtype),
        initial_state: T.Tensor([state_rows, *state_shape], "float32"),
        initial_state_indices: T.Tensor([batch], "int32"),
        h: T.Tensor(chunk_state_shape, qkv_dtype),
        v_new: T.Tensor([batch, seq_len, heads, dim_v], qkv_dtype),
        final_state: T.Tensor([batch, *state_shape], "float32"),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), batch, heads,
            threads=_TOTAL_THREADS,
        ) as (vid, bid, hid):
            v_offset = vid * block_v
            tx = T.get_thread_binding()
            T.import_source(_GDN_K_ALT_SOURCE)

            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            w_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            vn_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            state_shared = T.alloc_shared(state_shared_shape, qkv_dtype)
            g_shared = T.alloc_shared((chunk_size,), gate_dtype)
            T.annotate_layout(
                {
                    state_shared: state_shared_layout,
                }
            )

            if tx < _PRODUCER_THREADS:
                T.call_extern(
                    "handle",
                    "gdn_prefetch_k_alt_bf16",
                    T.access_ptr(k[bid, 0, hid, 0], "r"),
                    T.access_ptr(k_shared, "w"),
                    heads * dim_k,
                )
                T.async_copy(
                    w[bid, 0:chunk_size, hid, 0:dim_k],
                    w_shared,
                )
                T.async_copy(g[bid, 0:chunk_size, hid], g_shared)
                T.ptx_wait_group(0)
                # B0: publish chunk 0 inputs.
                T.sync_warp()

                for cid in T.serial(num_chunks):
                    # B1: current inputs have been captured in compute VGPRs.
                    T.sync_warp()
                    if cid + 1 < num_chunks:
                        next_base = (cid + 1) * chunk_size
                        T.call_extern(
                            "handle",
                            "gdn_prefetch_k_alt_bf16",
                            T.access_ptr(k[bid, next_base, hid, 0], "r"),
                            T.access_ptr(k_shared, "w"),
                            heads * dim_k,
                        )
                        T.async_copy(
                            w[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                                0:dim_k,
                            ],
                            w_shared,
                        )
                        T.async_copy(
                            g[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                            ],
                            g_shared,
                        )
                        T.ptx_wait_group(0)
                    # B2: publish next inputs while compute publishes H[n+1].
                    T.sync_warp()

            else:
                state_fragment = T.alloc_fragment(
                    (block_v, dim_k), "float32"
                )
                T.annotate_layout({state_fragment: state_fragment_layout})
                v_new_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                v_shuffle0 = T.alloc_local((8,), "float32")
                v_shuffle1 = T.alloc_local((8,), "float32")
                u_local = T.alloc_local((8,), qkv_dtype)
                k_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                T.annotate_layout({k_fragment: k_fragment_layout})
                vn_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )

                state_idx = initial_state_indices[bid]
                for ik, iv in T.Parallel(dim_k, block_v):
                    if (
                        use_initial_state
                        and state_idx >= 0
                        and state_idx < state_rows
                        and v_offset + iv < dim_v
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
                T.call_extern(
                    "handle",
                    "gdn_store_state_k_pair_bf16",
                    T.access_ptr(state_fragment, "r"),
                    T.access_ptr(state_shared, "w"),
                    transpose_state_layout,
                )
                # B0: publish H[0] and wait for chunk 0 inputs.
                T.sync_warp()

                for cid in T.serial(num_chunks):
                    base = cid * chunk_size
                    for ik, iv in T.Parallel(dim_k, block_v):
                        if v_offset + iv < dim_v:
                            if transpose_state_layout:
                                h[bid, cid, hid, v_offset + iv, ik] = T.cast(
                                    state_fragment[iv, ik], qkv_dtype
                                )
                            else:
                                h[bid, cid, hid, ik, v_offset + iv] = T.cast(
                                    state_fragment[iv, ik], qkv_dtype
                                )

                    T.sched_barrier()
                    T.gemm(
                        w_shared,
                        state_shared,
                        v_new_fragment,
                        transpose_B=True,
                        clear_accum=True,
                        k_pack=2,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.sched_barrier()

                    g_last = T.cast(
                        g_shared[chunk_size - 1], "float32"
                    )
                    consumer_wave = tx // 64 - _PRODUCER_THREADS // 64
                    lane = tx % 64
                    lane_group = lane // 16
                    # FullRow gives each lane one T row and eight V values,
                    # but in native MMAC order: V = 4 * i + lane_group.
                    # Transpose the two lane-group bits with local-index bits
                    # 1 and 2 so each lane instead owns a contiguous V8.
                    for i in T.unroll(8):
                        v_shuffle0[i] = v_new_fragment[
                            consumer_wave * 16 + lane % 16,
                            i * 4 + lane_group,
                        ]
                    for i in T.unroll(8):
                        v_shuffle1[i] = T.if_then_else(
                            lane_group % 2 == (i // 2) % 2,
                            v_shuffle0[i],
                            T.shfl_xor(v_shuffle0[i ^ 2], 16, width=64),
                        )
                    for i in T.unroll(8):
                        v_shuffle0[i] = T.if_then_else(
                            lane_group // 2 == i // 4,
                            v_shuffle1[i],
                            T.shfl_xor(v_shuffle1[i ^ 4], 32, width=64),
                        )
                    it = consumer_wave * 16 + lane % 16
                    v_base = lane_group * 8
                    for i in T.vectorized(8):
                        u_local[i] = u[
                            bid, base + it, hid, v_offset + v_base + i
                        ]
                    for i in T.unroll(8):
                        v_shuffle0[i] = (
                            T.cast(u_local[i], "float32") - v_shuffle0[i]
                        )
                    if save_new_value:
                        for i in T.vectorized(8):
                            u_local[i] = T.cast(v_shuffle0[i], qkv_dtype)
                        for i in T.vectorized(8):
                            v_new[
                                bid,
                                base + it,
                                hid,
                                v_offset + v_base + i,
                            ] = u_local[i]
                    diff = g_last - T.cast(g_shared[it], "float32")
                    safe_diff = T.min(diff, T.float32(0.0))
                    gate_scale = (
                        T.exp2(safe_diff * _LOG2E)
                        * T.cast(diff <= 0, "float32")
                    )
                    for i in T.vectorized(8):
                        u_local[i] = T.cast(
                            v_shuffle0[i] * gate_scale, qkv_dtype
                        )
                    for i in T.vectorized(8):
                        vn_shared[it, v_base + i] = u_local[i]

                    # Capture K before B1 because the producer reuses k_shared
                    # for chunk n + 1 immediately after the barrier.
                    T.call_extern(
                        "handle",
                        "gdn_read_k_m32x16_alt_bf16",
                        T.access_ptr(k_shared, "r"),
                        T.access_ptr(k_fragment, "w"),
                    )
                    T.s_waitcnt(0, "lgkmcnt")
                    decay = T.exp2(g_last * _LOG2E)
                    for ik, iv in T.Parallel(dim_k, block_v):
                        state_fragment[iv, ik] *= decay
                    # B1 publishes Vn while the producer starts filling the
                    # next K/W/g tiles.
                    T.sync_warp()

                    T.sched_barrier()
                    T.copy(vn_shared, vn_fragment)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.gemm(
                        vn_fragment,
                        k_fragment,
                        state_fragment,
                        transpose_A=True,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol,
                        annotations={
                            "trans_c": True,
                            "gdn_k_pair": 1,
                        },
                    )
                    T.sched_barrier()
                    T.call_extern(
                        "handle",
                        "gdn_store_state_k_pair_bf16",
                        T.access_ptr(state_fragment, "r"),
                        T.access_ptr(state_shared, "w"),
                        transpose_state_layout,
                    )
                    # B2: publish H[n+1] and wait for next inputs.
                    T.sync_warp()

                if store_final_state:
                    for ik, iv in T.Parallel(dim_k, block_v):
                        if v_offset + iv < dim_v:
                            if transpose_state_layout:
                                final_state[bid, hid, v_offset + iv, ik] = (
                                    state_fragment[iv, ik]
                                )
                            else:
                                final_state[bid, hid, ik, v_offset + iv] = (
                                    state_fragment[iv, ik]
                                )

    return chunk_gated_delta_rule_fwd_h_producer_consumer_alt_kernel


@functools.lru_cache(maxsize=128)
def _get_kernel(*args):
    return _build_chunk_h(*args)


@functools.lru_cache(maxsize=128)
def _get_producer_consumer_alt_kernel(*args):
    return _build_chunk_h_producer_consumer_alt(*args)


def chunk_gated_delta_rule_fwd_h_alt(
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
    kernel_cfg: Optional[dict] = None,
):
    require_first_version(
        chunk_size=chunk_size,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        kernel_cfg=kernel_cfg,
    )
    if g is None or gk is not None:
        raise NotImplementedError(
            "the first TileLang version requires g and does not support gk."
        )
    if inplace_final_state:
        raise NotImplementedError(
            "the first TileLang version does not support in-place final state."
        )
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("k, w, u and g must use BTHD/BTH layouts.")
    batch, seq_len, kv_heads, dim_k = k.shape
    heads, dim_v = w.shape[2], u.shape[-1]
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
        state_rows = batch
        state_buffer = torch.empty(
            batch, *state_trailing, dtype=torch.float32, device=k.device
        )
        state_indices = torch.arange(batch, dtype=torch.int32, device=k.device)
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
            if state_rows < batch:
                raise ValueError("initial_state has fewer rows than the batch.")
            state_indices = torch.arange(batch, dtype=torch.int32, device=k.device)
        else:
            if initial_state_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("initial_state_indices must have int32 or int64 dtype.")
            if tuple(initial_state_indices.shape) != (batch,):
                raise ValueError(f"initial_state_indices must have shape ({batch},).")
            if initial_state_indices.device != k.device:
                raise ValueError(
                    "initial_state_indices and k must be on the same device."
                )
            state_indices = initial_state_indices.to(dtype=torch.int32)

    require_contiguous(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=state_buffer,
        initial_state_indices=state_indices,
    )
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    h = torch.empty(
        batch, num_chunks, *state_trailing, dtype=k.dtype, device=k.device
    )
    v_new_buffer = torch.empty_like(u)
    final_state_buffer = torch.empty(
        batch, *state_trailing, dtype=torch.float32, device=k.device
    )
    use_producer_consumer = (
        chunk_size == 64
        and seq_len % chunk_size == 0
        and dim_k == 128
        and dim_v % 32 == 0
        and k.dtype == torch.bfloat16
        and output_final_state
        and save_new_value
    )
    get_kernel = (
        _get_producer_consumer_alt_kernel if use_producer_consumer else _get_kernel
    )
    kernel = get_kernel(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        state_rows,
        tilelang_dtype(k.dtype),
        tilelang_dtype(g.dtype),
        chunk_size,
        initial_state is not None,
        output_final_state,
        save_new_value,
        transpose_state_layout,
    )
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


__all__ = ["chunk_gated_delta_rule_fwd_h_alt"]
