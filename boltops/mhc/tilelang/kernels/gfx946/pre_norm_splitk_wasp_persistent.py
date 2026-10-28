# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""gfx946 persistent warp-specialized MHC split-K pre-norm kernel."""

import functools

import tilelang
from tilelang import language as T

_TOKEN_BLOCK = 128
_HIDDEN_BLOCK = 64
_THREADS = 12 * 64
_PERSISTENT_CTAS = 48
_PRODUCER_THREADS = 4 * 64
_CONSUMER0_END = 8 * 64

_FREE_X_PING = 0
_READY_X_PING = 1
_FREE_FN_PING = 2
_READY_FN_PING = 3
_FREE_X_PONG = 4
_READY_X_PONG = 5
_FREE_FN_PONG = 6
_READY_FN_PONG = 7

_INIT_EBAR_ID = 3
_CONSUMER0_EBAR_ID = 4
_CONSUMER1_EBAR_ID = 5
_CONSUMER_PHASE_EBAR_ID = 6
_CONSUMER_ACK_EBAR_ID = 7
_CONSUMER0_PHASE_SCHED_MASK = 0
_CONSUMER1_PHASE_SCHED_MASK = 0x2  # Allow only VALU to cross.

_PRODUCER_MAX_NREG = 72
_CONSUMER_MAX_NREG = 216
_TF32_ROUND_BITS = 0x1000
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_ENABLE_HCU_WDRA: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


def _gfx946_mls_16x32_b16_swizzle(inner_m, inner_k, slot):
    """Swizzle one 16x32 b16 (16 rows of 64-byte requests) on 64 banks."""
    vector = inner_k % 8
    vector_group = inner_k // 8
    row_group = inner_m // 4
    swizzled_group = vector_group ^ row_group
    swizzled_m = inner_m ^ slot
    return swizzled_m, swizzled_group * 8 + vector


def _gfx946_mls_b16_layout(m, k, hidden_block):
    """Physical LDS layout written by gfx946 matrix_load_16x64_b16.

    A logical 16x64 tile is split into two 16x32 (64-byte-row) halves.
    Four M tiles occupy the four MLS slots, and every four tiles start a
    new tile group.  Keeping the decomposition explicit also makes the
    64-bank phase of each 16x32 half visible in the final physical offset.
    """
    tile_m = m // 16
    inner_m = m % 16
    tile_group = tile_m // 4
    slot = tile_m % 4

    k_half = k // 32
    inner_k = k % 32
    swizzled_m, swizzled_k = _gfx946_mls_16x32_b16_swizzle(inner_m, inner_k, slot)

    # PackedShape outer dimensions: (tile_group, k_half, slot).  Each entry
    # owns one contiguous 16x32 b16 physical block; apply the bank swizzle
    # only to coordinates inside that block.
    k_halves = hidden_block // 32
    block_index = (tile_group * k_halves + k_half) * 4 + slot
    block_base = block_index * 16 * 32
    inner_offset = swizzled_m * 32 + swizzled_k
    physical_offset = block_base + inner_offset
    return physical_offset // hidden_block, physical_offset % hidden_block


def _gfx946_mls_16x32_b16_layout(m, k):
    return _gfx946_mls_b16_layout(m, k, 32)


def _gfx946_mls_16x64_b16_layout(m, k):
    return _gfx946_mls_b16_layout(m, k, 64)


def _gfx946_mls_16x128_b16_layout(m, k):
    return _gfx946_mls_b16_layout(m, k, 128)


@functools.cache
def mhc_pre_gemm_sqrsum_splitk_persistent_kernel(
    mhc_mult3: int,
    mhc_hidden_size: int,
    split_k: int,
    token_block: int = _TOKEN_BLOCK,
    hidden_block: int = _HIDDEN_BLOCK,
    threads: int = _THREADS,
    round_fn_to_tf32: bool = True,
    separate_x_fn_smem: bool = False,
) -> tuple[tilelang.JITKernel, None]:
    if not (
        split_k > 0
        and token_block in (32, 64, 128, 256)
        and hidden_block in (32, 64, 128)
        and mhc_hidden_size % split_k == 0
        and (mhc_hidden_size // split_k) % (2 * hidden_block) == 0
    ):
        raise ValueError(
            "Unsupported gfx946 persistent split-K configuration: "
            f"mhc_hidden_size={mhc_hidden_size}, split_k={split_k}, "
            f"token_block={token_block}, hidden_block={hidden_block}"
        )

    split_size = mhc_hidden_size // split_k
    k_tiles = split_size // hidden_block
    k_pairs = k_tiles // 2
    half_m = token_block // 2
    num_tokens = T.dynamic("num_tokens")
    total_tiles = split_k * T.ceildiv(num_tokens, token_block)
    x_mls_layout = {
        32: _gfx946_mls_16x32_b16_layout,
        64: _gfx946_mls_16x64_b16_layout,
        128: _gfx946_mls_16x128_b16_layout,
    }[hidden_block]

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def stage_0(
        x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, mhc_mult3), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(_PERSISTENT_CTAS, threads=_THREADS) as block_id:
            tx = T.get_thread_binding()

            # The custom annotated layout below computes the four 1-KiB
            # hardware slot phases relative to a phase-zero (4-KiB-aligned)
            # buffer base.
            # Keep both x stages at 4-KiB-aligned allocation offsets by
            # declaring them first, keeping their sizes 4-KiB aligned, and not
            # placing a live, non-4-KiB-sized shared allocation before them.
            # Recheck the generated .cu offsets whenever allocations change.
            x_shared = T.alloc_shared((2, token_block, hidden_block), T.bfloat16)
            fn_shared = T.alloc_shared((2, 32, hidden_block), T.float32)
            out_shared = T.alloc_shared((token_block, 32), T.float32)

            x0_16 = T.alloc_fragment((half_m, hidden_block), T.bfloat16)
            x0 = T.alloc_fragment((half_m, hidden_block), T.float32)
            fn0 = T.alloc_fragment((32, hidden_block), T.float32)
            out0 = T.alloc_fragment((half_m, 32), T.float32)
            sq0 = T.alloc_fragment((half_m, 16), T.float32)

            x1_16 = T.alloc_fragment((half_m, hidden_block), T.bfloat16)
            x1 = T.alloc_fragment((half_m, hidden_block), T.float32)
            fn1 = T.alloc_fragment((32, hidden_block), T.float32)
            out1 = T.alloc_fragment((half_m, 32), T.float32)
            sq1 = T.alloc_fragment((half_m, 16), T.float32)

            T.annotate_layout(
                {
                    x_shared: T.Layout(
                        x_shared.shape,
                        lambda stage, row, col: (stage, *x_mls_layout(row, col)),
                    ),
                    out_shared: T.Layout(
                        out_shared.shape,
                        lambda row, col: (
                            row,
                            ((col // 4) ^ ((row // 2) % 8)) * 4 + col % 4,
                        ),
                    ),
                }
            )
            T.abarrier_init(_FREE_X_PING, 8)
            T.abarrier_init(_READY_X_PING, 4)
            T.abarrier_init(_FREE_FN_PING, 8)
            T.abarrier_init(_READY_FN_PING, 4)
            T.abarrier_init(_FREE_X_PONG, 8)
            T.abarrier_init(_READY_X_PONG, 4)
            T.abarrier_init(_FREE_FN_PONG, 8)
            T.abarrier_init(_READY_FN_PONG, 4)
            T.ebarrier_sync_cnt(_INIT_EBAR_ID, 12)

            if tx < _PRODUCER_THREADS:
                T.set_max_nreg(_PRODUCER_MAX_NREG, 0)
                producer_rounds = T.if_then_else(
                    block_id < total_tiles,
                    T.ceildiv(total_tiles - block_id, _PERSISTENT_CTAS),
                    0,
                )
                for persistent_round in T.Serial(producer_rounds):
                    tile_id = persistent_round * _PERSISTENT_CTAS + block_id
                    bz = tile_id % split_k
                    px = tile_id // split_k
                    k_base = bz * split_size
                    for pair in T.Serial(k_pairs):
                        phase = (persistent_round * k_pairs + pair) & 1
                        for stage in T.Unroll(2):
                            k_offset = k_base + (pair * 2 + stage) * hidden_block
                            barrier_base = stage * 4
                            free_x = barrier_base
                            ready_x = barrier_base + 1
                            free_fn = barrier_base + 2
                            ready_fn = barrier_base + 3
                            T.abarrier_try_wait(free_x, phase)
                            T.abarrier_seq(ready_x)
                            T.matrix_load(
                                x[px * token_block, k_offset],
                                x_shared[stage, :, :],
                            )
                            T.abarrier_arrive(ready_x)
                            T.abarrier_try_wait(free_fn, phase)
                            T.abarrier_seq(ready_fn)
                            T.matrix_load(fn[0, k_offset], fn_shared[stage, :, :])
                            T.abarrier_arrive(ready_fn)
            elif tx < _CONSUMER0_END:
                T.set_max_nreg(_CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(_FREE_X_PING)
                T.abarrier_arrive(_FREE_FN_PING)
                T.abarrier_arrive(_FREE_X_PONG)
                T.abarrier_arrive(_FREE_FN_PONG)
                consumer0_rounds = T.if_then_else(
                    block_id < total_tiles,
                    T.ceildiv(total_tiles - block_id, _PERSISTENT_CTAS),
                    0,
                )
                for persistent_round in T.Serial(consumer0_rounds):
                    tile_id = persistent_round * _PERSISTENT_CTAS + block_id
                    bz = tile_id % split_k
                    px = tile_id // split_k
                    k_base = bz * split_size
                    T.clear(out0)
                    T.clear(sq0)
                    for pair in T.Serial(k_pairs):
                        phase = (persistent_round * k_pairs + pair) & 1
                        for stage in T.Unroll(2):
                            barrier_base = stage * 4
                            free_x = barrier_base
                            ready_x = barrier_base + 1
                            free_fn = barrier_base + 2
                            ready_fn = barrier_base + 3
                            T.abarrier_try_wait(ready_x, phase)
                            T.copy(x_shared[stage, 0:half_m, :], x0_16)
                            T.copy(x0_16, x0)
                            T.sched_barrier(0)
                            T.abarrier_arrive(free_x)
                            T.sched_barrier(_CONSUMER0_PHASE_SCHED_MASK)
                            T.ebarrier_arrive(_CONSUMER_PHASE_EBAR_ID, 8)
                            T.call_extern("tl::promote_prio", dtype="void")
                            for jj in T.serial(hidden_block // 16):
                                for i, j in T.Parallel(half_m, 16):
                                    v = x0[i, jj * 16 + j]
                                    sq0[i, j] += v * v
                            T.call_extern("tl::restore_prio", dtype="void")
                            T.abarrier_try_wait(ready_fn, phase)
                            T.ds_read_format(fn_shared[stage, :, :], fn0)
                            T.sched_barrier(0)
                            T.abarrier_arrive(free_fn)
                            T.sched_barrier(0)
                            T.ebarrier_sync_cnt(_CONSUMER_ACK_EBAR_ID, 8)
                            T.sched_barrier(0)
                            if round_fn_to_tf32:
                                fn0_i = T.view(fn0, (32, hidden_block), T.int32)
                                for i, j in T.Parallel(32, hidden_block):
                                    fn0_i[i, j] += T.int32(_TF32_ROUND_BITS)
                                    fn0[i, j] = T.reinterpret(fn0_i[i, j], T.float32)
                            T.call_extern("tl::promote_prio", dtype="void")
                            T.gemm(
                                x0,
                                fn0,
                                out0,
                                transpose_B=True,
                                k_pack=1,
                                policy=T.GemmWarpPolicy.FullRow,
                                use_tf32=True,
                            )
                            T.call_extern("tl::restore_prio", dtype="void")

                    sq0_l = T.alloc_fragment((half_m,), T.float32)
                    T.reduce_sum(sq0, sq0_l)
                    T.copy(out0, out_shared[0:half_m, :])
                    for i in T.Parallel(half_m):
                        t = px * token_block + i
                        if t < num_tokens:
                            sqrsum_partial[bz, t] = sq0_l[i]
                    T.ebarrier_sync_cnt(_CONSUMER0_EBAR_ID, 4)
                    for i, j in T.Parallel(half_m, 32):
                        t = px * token_block + i
                        if t < num_tokens and j < mhc_mult3:
                            out_partial[bz, t, j] = out_shared[i, j]
            else:
                T.set_max_nreg(_CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(_FREE_X_PING)
                T.abarrier_arrive(_FREE_FN_PING)
                T.abarrier_arrive(_FREE_X_PONG)
                T.abarrier_arrive(_FREE_FN_PONG)
                consumer1_rounds = T.if_then_else(
                    block_id < total_tiles,
                    T.ceildiv(total_tiles - block_id, _PERSISTENT_CTAS),
                    0,
                )
                for persistent_round in T.Serial(consumer1_rounds):
                    tile_id = persistent_round * _PERSISTENT_CTAS + block_id
                    bz = tile_id % split_k
                    px = tile_id // split_k
                    k_base = bz * split_size
                    T.clear(out1)
                    T.clear(sq1)
                    for pair in T.Serial(k_pairs):
                        phase = (persistent_round * k_pairs + pair) & 1
                        for stage in T.Unroll(2):
                            barrier_base = stage * 4
                            free_x = barrier_base
                            ready_x = barrier_base + 1
                            free_fn = barrier_base + 2
                            ready_fn = barrier_base + 3
                            T.ebarrier_sync_cnt(_CONSUMER_PHASE_EBAR_ID, 8)
                            T.sched_barrier(_CONSUMER1_PHASE_SCHED_MASK)
                            T.abarrier_try_wait(ready_x, phase)
                            T.copy(x_shared[stage, half_m:token_block, :], x1_16)
                            T.copy(x1_16, x1)
                            T.sched_barrier(0)
                            T.abarrier_arrive(free_x)
                            for jj in T.serial(hidden_block // 16):
                                for i, j in T.Parallel(half_m, 16):
                                    v = x1[i, jj * 16 + j]
                                    sq1[i, j] += v * v
                            T.sched_barrier(0)
                            T.ebarrier_arrive(_CONSUMER_ACK_EBAR_ID, 8)
                            T.sched_barrier(_CONSUMER1_PHASE_SCHED_MASK)
                            T.abarrier_try_wait(ready_fn, phase)
                            T.ds_read_format(fn_shared[stage, :, :], fn1)
                            T.abarrier_arrive(free_fn)
                            if round_fn_to_tf32:
                                fn1_i = T.view(fn1, (32, hidden_block), T.int32)
                                for i, j in T.Parallel(32, hidden_block):
                                    fn1_i[i, j] += T.int32(_TF32_ROUND_BITS)
                                    fn1[i, j] = T.reinterpret(fn1_i[i, j], T.float32)
                                T.sched_barrier(0)
                            T.gemm(
                                x1,
                                fn1,
                                out1,
                                transpose_B=True,
                                k_pack=1,
                                policy=T.GemmWarpPolicy.FullRow,
                                use_tf32=True,
                            )
                            T.sched_barrier(0)

                    sq1_l = T.alloc_fragment((half_m,), T.float32)
                    T.reduce_sum(sq1, sq1_l)
                    T.copy(out1, out_shared[half_m:token_block, :])
                    for i in T.Parallel(half_m):
                        t = px * token_block + half_m + i
                        if t < num_tokens:
                            sqrsum_partial[bz, t] = sq1_l[i]
                    T.ebarrier_sync_cnt(_CONSUMER1_EBAR_ID, 4)
                    for i, j in T.Parallel(half_m, 32):
                        t = px * token_block + half_m + i
                        if t < num_tokens and j < mhc_mult3:
                            out_partial[bz, t, j] = out_shared[half_m + i, j]

    # The fused path consumes partial buffers directly; no stage-1 reducer is
    # required by this specialization.
    return stage_0, None
