# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Gluon block-scaled GEMM kernels, selected by recipe and tile name.

The current recipe uses E4M3FN inputs, FP32 A(1,128)/B(128,128) scales,
BF16 output and tile M/N/K=(128,256,128). Each entry owns its layouts,
partition resources, pipeline and compile-time legality checks.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout,
    MlsSharedLayout,
    matrix_load,
    matrix_store,
    MmacInterleave2Layout,
    get_wave_id,
    set_priority,
    wait_lgkmcnt,
    abarrier_init,
    abarrier_try_wait,
    abarrier_seq,
    abarrier_arrive,
    ebarrier_arrive,
    ebarrier_sync,
    sched_barrier,
    packed_scale_fma,
)


@g.jit
def _initialize_pipeline_barriers(wave):
    if wave == 0:
        abarrier_init(0, 4)
        abarrier_init(1, 4)
        abarrier_init(2, 4)
        abarrier_init(3, 4)
        abarrier_init(6, 4)
        abarrier_init(7, 4)
        abarrier_init(8, 8)
        abarrier_init(9, 4)
        abarrier_init(10, 4)
        abarrier_init(11, 4)
        abarrier_init(12, 8)
        abarrier_init(13, 4)
        abarrier_init(14, 8)
    if wave < 4:
        ebarrier_arrive(8, 8)
    ebarrier_sync(14, 16)


@g.jit
def _wait_all_lds_reads():
    wait_lgkmcnt(0)


@g.jit
def _synchronize_mmac_turn(GROUP: gl.constexpr, WAIT: gl.constexpr):
    if WAIT:
        ebarrier_sync(8 + GROUP, 8)
    else:
        ebarrier_arrive(9 - GROUP, 8)
        set_priority(0)


@g.jit
def _idle_default_partition():
    pass


@g.jit
def _produce_ab_and_scales(
    A,
    B,
    AS,
    BS,
    smAS0,
    smAS1,
    smBS,
    smA0,
    smA1,
    smB0,
    smB1,
    K: gl.constexpr,
    GROUP: gl.constexpr,
):
    wave = get_wave_id()
    _initialize_pipeline_barriers(wave)
    SL: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    for i in range(K // 128):
        if i % 2 == 0:
            smA = smA0
            smB = smB0
        else:
            smA = smA1
            smB = smB1
        slot = i % 2
        if GROUP == 0:
            if i >= 2:
                abarrier_try_wait(slot * 2, ((i // 2) - 1) % 2)
            abarrier_seq(slot * 2 + 1)
            matrix_load(smA, A, (64, K), (K, 1), (0, i * 128), pred=wave % 4 == 0)
            abarrier_arrive(slot * 2 + 1)
            if i >= 2:
                abarrier_try_wait(8 + slot * 4, ((i // 2) - 1) % 2)
            abarrier_seq(9 + slot * 4)
            matrix_load(smB, B, (K, 256), (1, K), (i * 128, 0))
            abarrier_arrive(9 + slot * 4)
        else:
            if i >= 2:
                abarrier_try_wait(6 + slot * 4, ((i // 2) - 1) % 2)
            abarrier_seq(7 + slot * 4)
            matrix_load(
                smA, A + 64 * K, (64, K), (K, 1), (0, i * 128), pred=wave % 4 == 0
            )
            ar = gl.arange(0, 128, layout=SL)
            async_copy.buffer_load_to_shared(smAS0.index(i), AS, ar * (K // 128) + i)
            bi = gl.arange(0, 2, layout=SL)
            async_copy.buffer_load_to_shared(smBS.index(i), BS, bi * (K // 128) + i)
            abarrier_arrive(7 + slot * 4)
            abarrier_try_wait(14, i % 2)


@g.jit
def _wait_ab_reads_and_raise_priority():
    set_priority(3)
    wait_lgkmcnt(9)


@g.jit
def _wait_a_tile_ready(i, GROUP: gl.constexpr):
    if GROUP == 0:
        abarrier_try_wait((i % 2) * 2 + 1, (i // 2) % 2)
    else:
        abarrier_try_wait(7 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def _wait_b_tile_ready(i):
    abarrier_try_wait(9 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def _ack_scale_tile_ready(i, GROUP: gl.constexpr):
    if GROUP == 0:
        abarrier_try_wait(7 + (i % 2) * 4, (i // 2) % 2)
    # ACK means readiness observed, not completion of pending LDS reads.
    # Scale slices are immutable; the producer writes the following distinct slice.
    # TODO: use abarrier_arrive(14, 1) once LLVM supports this readiness-only
    # ACK without inserting lgkmcnt(0), which drains A/B prefetch and breaks
    # overlap. This ACK does not publish or release LDS storage.
    gl.inline_asm_elementwise(
        "s_abarrier_arrive $0, 14, 1",
        constraints="=s",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _release_ab_lds_slots(i, GROUP: gl.constexpr):
    _wait_ab_reads_and_raise_priority()
    if GROUP == 0:
        abarrier_arrive((i % 2) * 2)
    else:
        abarrier_arrive(6 + (i % 2) * 4)
    abarrier_arrive(8 + (i % 2) * 4)


@g.jit
def _prefetch_scale_tile(
    smAS0,
    smAS1,
    smBS,
    i,
    m,
    lane_m_byte,
    bs_group,
    GROUP: gl.constexpr,
    F: gl.constexpr,
):
    # release/3.6.x exposes descriptor loads; the reference's raw LDS
    # slice pointer API is not present in this release branch.
    sa = smAS0.index(i).slice(GROUP * 64, 64).load(gl.SliceLayout(1, F))
    bits = smBS.index(i)._reinterpret(gl.uint64, (1,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
    sb = bits.load(gl.BlockedLayout([1], [64], [4], [0])).reshape(())
    return sa, sb


@g.jit
def _select_b_scale_for_wave(sb, bs_group):
    lo = sb.to(gl.uint32)
    hi = (sb >> 32).to(gl.uint32)
    wait_lgkmcnt(0)
    selected = gl.where(bs_group != 0, hi, lo)
    return selected.to(gl.float32, bitcast=True)


@g.jit
def _consume_ab_and_accumulate(
    AS,
    BS,
    C,
    smAS0,
    smAS1,
    smBS,
    smA0,
    smA1,
    smB0,
    smB1,
    K: gl.constexpr,
    N: gl.constexpr,
    GROUP: gl.constexpr,
    F: gl.constexpr,
):
    wave = get_wave_id()
    _initialize_pipeline_barriers(wave)
    RF: gl.constexpr = MmacInterleave2Layout(F)
    acc = gl.full((64, 256), 0, gl.float32, RF)
    # TODO: replace with gl.full once LLVM keeps the zero VGPR tuple outside
    # the loop instead of rematerializing it and repacking registers per MMAC.
    zero_mmac = gl.inline_asm_elementwise(
        "v_mov_b32 $0, 0\n\tv_mov_b32 $1, 0\n\tv_mov_b32 $2, 0\n\tv_mov_b32 $3, 0",
        constraints="=v,=v,=v,=v,v,v,v,v",
        args=[gl.full((64, 256), 0, gl.int32, RF)],
        dtype=gl.float32,
        is_pure=True,
        pack=4,
    )
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, RF))
    bs_group = (wave % 4) // 2
    # Preserve a common lane base so A-scale reads combine into ds_read2.
    lane_m_byte = (m % 4) * 4
    _wait_a_tile_ready(0, GROUP)
    a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
    _wait_b_tile_ready(0)
    b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
    sched_barrier(0)
    _ack_scale_tile_ready(0, GROUP)
    sa, sb = _prefetch_scale_tile(smAS0, smAS1, smBS, 0, m, lane_m_byte, bs_group, GROUP, RF)
    _release_ab_lds_slots(0, GROUP)
    for pair in range(K // 256):
        i = pair * 2
        sched_barrier(0)
        _synchronize_mmac_turn(GROUP, True)
        dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
        _synchronize_mmac_turn(GROUP, False)
        selected_bs = _select_b_scale_for_wave(sb, bs_group)
        j = i + 1
        _wait_a_tile_ready(j, GROUP)
        if j % 2 == 0:
            a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
        else:
            a = async_copy.load_shared_relaxed(smA1, gl.DotOperandLayout(0, F, 8))
        _wait_b_tile_ready(j)
        if j % 2 == 0:
            b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        else:
            b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        sched_barrier(0)
        _ack_scale_tile_ready(j, GROUP)
        next_sa, next_sb = _prefetch_scale_tile(
            smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
        )
        # Keep next-scale reads ahead of the current scale arithmetic.
        sched_barrier(0)
        acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
        _release_ab_lds_slots(j, GROUP)
        sa = next_sa
        sb = next_sb
        sched_barrier(0)
        if pair + 1 < K // 256:
            i = pair * 2 + 1
            sched_barrier(0)
            _synchronize_mmac_turn(GROUP, True)
            dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
            _synchronize_mmac_turn(GROUP, False)
            selected_bs = _select_b_scale_for_wave(sb, bs_group)
            j = i + 1
            _wait_a_tile_ready(j, GROUP)
            if j % 2 == 0:
                a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
            else:
                a = async_copy.load_shared_relaxed(smA1, gl.DotOperandLayout(0, F, 8))
            _wait_b_tile_ready(j)
            if j % 2 == 0:
                b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            else:
                b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            sched_barrier(0)
            _ack_scale_tile_ready(j, GROUP)
            next_sa, next_sb = _prefetch_scale_tile(
                smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
            )
            # Keep next-scale reads ahead of the current scale arithmetic.
            sched_barrier(0)
            acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
            _release_ab_lds_slots(j, GROUP)
            sa = next_sa
            sb = next_sb
            sched_barrier(0)
    _synchronize_mmac_turn(GROUP, True)
    dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
    _synchronize_mmac_turn(GROUP, False)
    selected_bs = _select_b_scale_for_wave(sb, bs_group)
    # Keep scale reads ahead of the final scale arithmetic.
    sched_barrier(0)
    acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
    packed = acc.to(gl.bfloat16).to(gl.float16, bitcast=True)
    matrix_store(
        C.to(gl.pointer_type(gl.float16)),
        packed,
        (128, N),
        (N, 1),
        (GROUP * 64, 0),
        tile=(32, 32),
        interleave=1,
        version=3,
        order=(1, 0),
    )


@g.jit
def _gemm_a8w8_blockscale_as_fp32_bs_fp32_recipe_a_1x128_recipe_b_128x128_bm128_bn256_bk128_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    GROUP_N: gl.constexpr,
    GROUP_K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
):
    gl.static_assert((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) == (128, 256, 128),
                     "this pipeline requires tile (128,256,128)")
    gl.static_assert(M % 128 == 0 and N % 256 == 0 and K % 256 == 0 and K >= 256 and K <= 6144,
                     "this pipeline requires complete tiles and K=256..6144 divisible by 256")
    gl.static_assert(GROUP_N == 128 and GROUP_K == 128,
                     "this pipeline requires A(1,128)/B(128,128) scales")
    gl.static_assert(A.dtype.element_ty == gl.float8e4nv and B.dtype.element_ty == gl.float8e4nv,
                     "this pipeline requires E4M3FN inputs")
    gl.static_assert(AS.dtype.element_ty == gl.float32 and BS.dtype.element_ty == gl.float32,
                     "this pipeline requires FP32 scales")
    gl.static_assert(C.dtype.element_ty == gl.bfloat16, "this pipeline requires BF16 output")
    SA: gl.constexpr = MlsSharedLayout(0, (64, 128))
    SB: gl.constexpr = MlsSharedLayout(1, (128, 64), alt=1, order=(0, 1))
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4], tiles_per_warp=[4, 4],
    )
    tile_m = gl.program_id(0) * 128
    tile_n = gl.program_id(1) * 256
    A += tile_m * K
    B += tile_n * K
    AS += tile_m * (K // 128)
    BS += (tile_n // 128) * (K // 128)
    C += tile_m * N + tile_n
    scaleL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    sas0 = gl.allocate_shared_memory(gl.float32, (K // 128, 128), scaleL)
    sas1 = sas0
    sbs = gl.allocate_shared_memory(gl.float32, (K // 128, 2), scaleL)
    a0 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a0_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    b_next = gl.allocate_shared_memory(gl.float8e4nv, (128, 256), SB)
    b = gl.allocate_shared_memory(gl.float8e4nv, (128, 256), SB)
    gl.warp_specialize(
        [
            (_idle_default_partition, ()),
            (_produce_ab_and_scales, (A, B, AS, BS, sas0, sas1, sbs, a0, a0_next, b, b_next, K, 0)),
            (_produce_ab_and_scales, (A, B, AS, BS, sas0, sas1, sbs, a1, a1_next, b, b_next, K, 1)),
            (
                _consume_ab_and_accumulate,
                (AS, BS, C, sas0, sas1, sbs, a0, a0_next, b, b_next, K, N, 0, F),
            ),
            (
                _consume_ab_and_accumulate,
                (AS, BS, C, sas0, sas1, sbs, a1, a1_next, b, b_next, K, N, 1, F),
            ),
        ],
        [4, 4, 4, 4],
        [8, 8, 240, 240],
    )

