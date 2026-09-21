# SPDX-License-Identifier: MIT
"""gfx938 MLS WASP QSA MQA prefill (FP8 + per-token kv_scales).

Producer issues ``T.matrix_load`` for Q/K and ``T.async_copy`` for the 512 B
``kv_scales`` tile.  Consumers keep the BF16 12-wave schedule.
Scale ping/pong LDS is two ``[block_n]`` buffers; producer issues one
``T.async_copy`` per tile.  ``s_waitcnt`` still counts outstanding
global ops; ``T.ptx_wait_group`` can replace that later (one Python
issue = one group).
"""

import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        # Dynamic cyclic N-tile mapping is bijective, but the symbolic
        # parallel-loop checker cannot prove that after the wrap select
        # and reports false Logits write races for impossible n_tiles
        # combinations.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
            "-ffast-math",
            "-mllvm=-disallow-uniform-vmed3-combine=true",
        ],
    }
)
def _tilelang_qsa_mqa_prefill_wasp_fp8_kernel(
    heads: int = 4,
    head_dim: int = 128,
    block_n: int = 64,
    block_q: int = 32,
    num_stages: int = 0,
    threads: int = 768,
    rows_divisible: bool = False,
    n_tile_phases: int = 1,
    split_n: int = 1,
    apply_scale: bool = True,
    gemm_policy_override: str | None = None,
):
    """gfx938 12-wave producer/consumer QSA MQA FP8 prefill."""
    assert heads in (4, 8, 32, 64), "unsupported WASP query-head count"
    assert head_dim == 128, "the WASP schedule fixes head_dim=128"
    assert block_q >= 2 and block_q % 2 == 0
    assert block_q * heads in (32, 64, 128, 256), (
        "WASP requires logical-M=block_q*heads in {32,64,128,256}"
    )
    assert block_n in (32, 64, 128), "the ping-pong design supports block_n=32/64/128"
    assert num_stages == 0, "the kernel owns the software pipeline"
    assert threads == 768, "the WASP schedule requires twelve waves"
    assert n_tile_phases in (1, 2, 4, 8, 16, 32)
    assert split_n in (1, 2, 4, 8, 16, 32, 64)
    # At block_q=2, consumer_q is one.  A 2-D/Square wave partition would
    # try to split that singleton Q axis and produces an invalid mapping.
    # FullRow keeps every consumer wave on the Q row and partitions block-N.
    # ``gemm_policy_override`` is for experiments (e.g. H64 FullRow vs Square).
    if gemm_policy_override == "FullRow":
        gemm_policy = T.GemmWarpPolicy.FullRow
    elif gemm_policy_override == "Square":
        gemm_policy = T.GemmWarpPolicy.Square
    elif gemm_policy_override is None:
        gemm_policy = (
            T.GemmWarpPolicy.FullRow if block_q == 2 else T.GemmWarpPolicy.Square
        )
    else:
        raise ValueError(
            "gemm_policy_override must be None, 'FullRow', or 'Square', got "
            f"{gemm_policy_override!r}"
        )
    score_rows = block_n
    consumer_split = 512
    consumer_q = block_q // 2
    consumer_cols = consumer_q * heads

    elem_bytes = 1
    q_lds_bytes = block_q * heads * head_dim * elem_bytes
    k_pingpong_lds_bytes = 2 * block_n * head_dim * elem_bytes
    scale_pingpong_lds_bytes = 2 * block_n * 4
    use_lds_overlay = (
        q_lds_bytes + k_pingpong_lds_bytes + scale_pingpong_lds_bytes > 65536
    )
    assert max(q_lds_bytes, k_pingpong_lds_bytes) + scale_pingpong_lds_bytes <= 65536, (
        "FP8 Q/K ping-pong plus kv_scales cannot fit in the 64-KiB LDS budget"
    )

    rows = T.dynamic("rows")
    keys = T.dynamic("keys")

    @T.macro
    def score_gemm(k_local, q_local, scores):
        T.gemm(
            k_local,
            q_local,
            scores,
            transpose_B=True,
            clear_accum=True,
            # Square uses a 2x2 split for block_q>2.  block_q=2 uses FullRow
            # as selected above so all four waves split block-N only.
            policy=gemm_policy,
        )

    @T.macro
    def priority_3():
        T.call_extern("tl::set_prio<3>", dtype="void")

    @T.macro
    def priority_2():
        T.call_extern("tl::set_prio<2>", dtype="void")

    @T.macro
    def priority_1():
        T.call_extern("tl::set_prio<1>", dtype="void")

    @T.macro
    def priority_0():
        T.call_extern("tl::set_prio<0>", dtype="void")

    @T.macro
    def score_valu(scores_3d, reduced):
        for n, qi, head in T.Parallel(score_rows, consumer_q, heads):
            scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
        T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)

    def score_for_store(value, inv_scale, kv_scale):
        scaled = value * kv_scale
        if not apply_scale:
            return scaled
        return scaled * inv_scale

    def tile_row(tile_start, tile_idx, wrap_end, wrap_span):
        row = tile_start + tile_idx * block_n
        if n_tile_phases > 1:
            # The phase is strictly smaller than n_tiles.  The traversal
            # can therefore cross the padded range end at most once, so a
            # branchless select replaces dynamic integer remainder.
            row = T.if_then_else(row >= wrap_end, row - wrap_span, row)
        return row

    @T.macro
    def score_store(
        reduced,
        inv_scale,
        kv_scale_frag,
        logits,
        row_base,
        query_offset,
        tile_context,
        tile_idx,
    ):
        if rows_divisible:
            for qi, n in T.Parallel(consumer_q, score_rows):
                column = (
                    tile_row(
                        tile_context[0], tile_idx, tile_context[1], tile_context[2]
                    )
                    + n
                )
                logits[row_base + query_offset + qi, column] = score_for_store(
                    reduced[n, qi], inv_scale, kv_scale_frag[n]
                )
        else:
            for qi, n in T.Parallel(consumer_q, score_rows):
                column = (
                    tile_row(
                        tile_context[0],
                        tile_idx,
                        tile_context[1],
                        tile_context[2],
                    )
                    + n
                )
                if row_base + query_offset + qi < rows:
                    logits[row_base + query_offset + qi, column] = score_for_store(
                        reduced[n, qi], inv_scale, kv_scale_frag[n]
                    )

    @T.macro
    def score_store_boundary(
        reduced,
        inv_scale,
        kv_scale_frag,
        logits,
        row_base,
        query_offset,
        tile_context,
        end_max,
        tile_idx,
    ):
        if rows_divisible:
            for qi, n in T.Parallel(consumer_q, score_rows):
                column = tile_context[3] + n
                if column < end_max:
                    logits[row_base + query_offset + qi, column] = score_for_store(
                        reduced[n, qi], inv_scale, kv_scale_frag[n]
                    )
        else:
            for qi, n in T.Parallel(consumer_q, score_rows):
                column = tile_context[3] + n
                if row_base + query_offset + qi < rows and column < end_max:
                    logits[row_base + query_offset + qi, column] = score_for_store(
                        reduced[n, qi], inv_scale, kv_scale_frag[n]
                    )

    @T.macro
    def issue_scale_async(KvScales, k_row, scale_shared):
        T.async_copy(KvScales[k_row : k_row + block_n], scale_shared)

    @T.macro
    def consumer0(
        q_shared,
        k_ping_shared,
        k_pong_shared,
        q_local,
        k_ping_local,
        k_pong_local,
        scale_ping_shared,
        scale_pong_shared,
        scores,
        scores_3d,
        reduced,
        logits,
        inv_scale,
        row_base,
        start_min,
        end_max,
        n_tiles,
        full_pairs,
    ):
        scale_ping_local = T.alloc_fragment([block_n], T.float32)
        scale_pong_local = T.alloc_fragment([block_n], T.float32)
        # Consumer A owns the first four waves of the logical
        # eight-wave FullCol layout.  Keep every load/read in this
        # explicit tx branch so TileLang sees the intended scope.
        if n_tiles > 0:
            T.sync_warp()
            priority_1()
            T.ds_read_format(q_shared[0:consumer_cols, :], q_local)
            T.s_waitcnt(0, flag="lgkmcnt")
            T.sync_warp()  # S_init
            if use_lds_overlay:
                T.sync_warp()  # wait until overlaid K ping is ready
            T.ds_read_format(k_ping_shared, k_ping_local)
            T.copy(scale_ping_shared, scale_ping_local)
            T.s_waitcnt(0, flag="lgkmcnt")

            if full_pairs > 0:
                T.sync_warp()  # S_ping(0)

                if full_pairs > 1:
                    for pair in T.Serial(full_pairs - 1):
                        priority_1()
                        T.ds_read_format(k_pong_shared, k_pong_local)
                        T.copy(scale_pong_shared, scale_pong_local)
                        T.sched_barrier()
                        score_gemm(k_ping_local, q_local, scores)
                        priority_0()
                        T.sched_barrier()
                        score_valu(scores_3d, reduced)
                        score_store(
                            reduced,
                            inv_scale,
                            scale_ping_local,
                            logits,
                            row_base,
                            0,
                            start_min,
                            2 * pair,
                        )
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sync_warp()  # S_pong(pair)

                        priority_1()
                        T.ds_read_format(k_ping_shared, k_ping_local)
                        T.copy(scale_ping_shared, scale_ping_local)
                        T.sched_barrier()
                        score_gemm(k_pong_local, q_local, scores)
                        priority_0()
                        T.sched_barrier()
                        score_valu(scores_3d, reduced)
                        score_store(
                            reduced,
                            inv_scale,
                            scale_pong_local,
                            logits,
                            row_base,
                            0,
                            start_min,
                            2 * pair + 1,
                        )
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sync_warp()  # S_ping(pair+1)

                final_ping = 2 * (full_pairs - 1)
                final_pong = final_ping + 1
                priority_1()
                T.ds_read_format(k_pong_shared, k_pong_local)
                T.copy(scale_pong_shared, scale_pong_local)
                T.sched_barrier()
                score_gemm(k_ping_local, q_local, scores)
                priority_0()
                T.sched_barrier()
                score_valu(scores_3d, reduced)
                score_store(
                    reduced,
                    inv_scale,
                    scale_ping_local,
                    logits,
                    row_base,
                    0,
                    start_min,
                    final_ping,
                )
                T.sched_barrier()
                T.s_waitcnt(0, flag="lgkmcnt")
                T.sync_warp()  # S_pong(final)

                if n_tiles % 2 != 0:
                    tail = 2 * full_pairs
                    priority_1()
                    T.ds_read_format(k_ping_shared, k_ping_local)
                    T.copy(scale_ping_shared, scale_ping_local)
                    T.sched_barrier()
                    score_gemm(k_pong_local, q_local, scores)
                    priority_0()
                    T.sched_barrier()
                    score_valu(scores_3d, reduced)
                    score_store(
                        reduced,
                        inv_scale,
                        scale_pong_local,
                        logits,
                        row_base,
                        0,
                        start_min,
                        final_pong,
                    )
                    T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()  # S_tail

                    priority_1()
                    score_gemm(k_ping_local, q_local, scores)
                    priority_0()
                    T.sched_barrier()
                    score_valu(scores_3d, reduced)
                    score_store_boundary(
                        reduced,
                        inv_scale,
                        scale_ping_local,
                        logits,
                        row_base,
                        0,
                        start_min,
                        end_max,
                        tail,
                    )
                else:
                    priority_1()
                    score_gemm(k_pong_local, q_local, scores)
                    priority_0()
                    T.sched_barrier()
                    score_valu(scores_3d, reduced)
                    score_store_boundary(
                        reduced,
                        inv_scale,
                        scale_pong_local,
                        logits,
                        row_base,
                        0,
                        start_min,
                        end_max,
                        final_pong,
                    )

            else:
                # n_tiles == 1
                T.sync_warp()  # S_tail
                priority_1()
                score_gemm(k_ping_local, q_local, scores)
                priority_0()
                T.sched_barrier()
                score_valu(scores_3d, reduced)
                score_store_boundary(
                    reduced,
                    inv_scale,
                    scale_ping_local,
                    logits,
                    row_base,
                    0,
                    start_min,
                    end_max,
                    0,
                )

    @T.macro
    def consumer1(
        q_shared,
        k_ping_shared,
        k_pong_shared,
        q_local,
        k_ping_local,
        k_pong_local,
        scale_ping_shared,
        scale_pong_shared,
        scores,
        scores_3d,
        reduced,
        logits,
        inv_scale,
        row_base,
        start_min,
        end_max,
        n_tiles,
        full_pairs,
    ):
        scale_ping_local = T.alloc_fragment([block_n], T.float32)
        scale_pong_local = T.alloc_fragment([block_n], T.float32)
        # Consumer B owns the second four-wave FullCol slice and
        # stays one main-compute phase ahead of consumer A.
        if n_tiles > 0:
            T.sync_warp()
            priority_2()
            T.ds_read_format(
                q_shared[consumer_cols : 2 * consumer_cols, :],
                q_local,
            )
            T.s_waitcnt(0, flag="lgkmcnt")
            T.sync_warp()  # S_init
            if use_lds_overlay:
                T.sync_warp()  # wait until overlaid K ping is ready
            T.ds_read_format(k_ping_shared, k_ping_local)
            T.copy(scale_ping_shared, scale_ping_local)
            T.s_waitcnt(0, flag="lgkmcnt")

            if full_pairs > 0:
                T.sync_warp()  # S_ping(0)

                if full_pairs > 1:
                    for pair in T.Serial(full_pairs - 1):
                        priority_2()
                        score_gemm(k_ping_local, q_local, scores)
                        T.sched_barrier()
                        T.ds_read_format(k_pong_shared, k_pong_local)
                        T.copy(scale_pong_shared, scale_pong_local)
                        T.sched_barrier()
                        priority_0()
                        score_valu(scores_3d, reduced)
                        score_store(
                            reduced,
                            inv_scale,
                            scale_ping_local,
                            logits,
                            row_base,
                            consumer_q,
                            start_min,
                            2 * pair,
                        )
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sync_warp()  # S_pong(pair)

                        priority_2()
                        score_gemm(k_pong_local, q_local, scores)
                        T.sched_barrier()
                        T.ds_read_format(k_ping_shared, k_ping_local)
                        T.copy(scale_ping_shared, scale_ping_local)
                        T.sched_barrier()
                        priority_0()
                        score_valu(scores_3d, reduced)
                        score_store(
                            reduced,
                            inv_scale,
                            scale_pong_local,
                            logits,
                            row_base,
                            consumer_q,
                            start_min,
                            2 * pair + 1,
                        )
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sync_warp()  # S_ping(pair+1)

                final_ping = 2 * (full_pairs - 1)
                final_pong = final_ping + 1
                priority_2()
                score_gemm(k_ping_local, q_local, scores)
                T.sched_barrier()
                T.ds_read_format(k_pong_shared, k_pong_local)
                T.copy(scale_pong_shared, scale_pong_local)
                T.sched_barrier()
                priority_0()
                score_valu(scores_3d, reduced)
                score_store(
                    reduced,
                    inv_scale,
                    scale_ping_local,
                    logits,
                    row_base,
                    consumer_q,
                    start_min,
                    final_ping,
                )
                T.sched_barrier()
                T.s_waitcnt(0, flag="lgkmcnt")
                T.sync_warp()  # S_pong(final)

                if n_tiles % 2 != 0:
                    tail = 2 * full_pairs
                    priority_2()
                    score_gemm(k_pong_local, q_local, scores)
                    T.sched_barrier()
                    T.ds_read_format(k_ping_shared, k_ping_local)
                    T.copy(scale_ping_shared, scale_ping_local)
                    T.sched_barrier()
                    priority_0()
                    score_valu(scores_3d, reduced)
                    score_store(
                        reduced,
                        inv_scale,
                        scale_pong_local,
                        logits,
                        row_base,
                        consumer_q,
                        start_min,
                        final_pong,
                    )
                    T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()  # S_tail

                    priority_2()
                    score_gemm(k_ping_local, q_local, scores)
                    priority_0()
                    T.sched_barrier()
                    score_valu(scores_3d, reduced)
                    score_store_boundary(
                        reduced,
                        inv_scale,
                        scale_ping_local,
                        logits,
                        row_base,
                        consumer_q,
                        start_min,
                        end_max,
                        tail,
                    )
                else:
                    priority_2()
                    score_gemm(k_pong_local, q_local, scores)
                    priority_0()
                    T.sched_barrier()
                    score_valu(scores_3d, reduced)
                    score_store_boundary(
                        reduced,
                        inv_scale,
                        scale_pong_local,
                        logits,
                        row_base,
                        consumer_q,
                        start_min,
                        end_max,
                        final_pong,
                    )

            else:
                # n_tiles == 1
                T.sync_warp()  # S_tail
                priority_2()
                score_gemm(k_ping_local, q_local, scores)
                priority_0()
                T.sched_barrier()
                score_valu(scores_3d, reduced)
                score_store_boundary(
                    reduced,
                    inv_scale,
                    scale_ping_local,
                    logits,
                    row_base,
                    consumer_q,
                    start_min,
                    end_max,
                    0,
                )

    consumer0_dispatch = consumer0
    consumer1_dispatch = consumer1

    @T.prim_func
    def qsa_mqa_prefill_wasp_fp8(
        Q: T.Tensor([rows * heads, head_dim], T.float8_e4m3fn),  # type: ignore
        K: T.Tensor([keys, head_dim], T.float8_e4m3fn),  # type: ignore
        KvScales: T.Tensor([keys], T.float32),  # type: ignore
        Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
        Starts: T.Tensor([rows], T.int32),  # type: ignore
        Ends: T.Tensor([rows], T.int32),  # type: ignore
        InvScale: T.float32,
    ):
        # Compile-time specialization selected from the actual row count.
        # The aligned path removes all tail-CTA row predicates; the
        # generic specialization keeps explicit protection for its tail.
        if rows_divisible:
            T.assume(rows % block_q == 0)
        q_blocks = T.ceildiv(rows, block_q)
        with T.Kernel(q_blocks * split_n, threads=threads) as bx:
            tx = T.get_thread_binding()
            # Linear order: (M0,N0), (M0,N1), ..., (M1,N0), ... .  Grouping
            # the N splits of one M block was more robust for ragged ranges.
            q_block = bx // split_n
            split_id = bx % split_n
            row_base = q_block * block_q
            # Allow HCU codegen to move blockIdx-dependent terms from
            # ordinary Logits buffer-store offsets into the 64-bit
            # resource base.  No pass-wide rebase switch is required.
            T.annotate_buffer_ops_rebase(Logits)

            # Keep Q and K ping/pong in independent allocations only when
            # their combined footprint fits in the 64-KiB LDS budget.
            # Otherwise use one arena sized for max(Q, K ping+pong): Q is
            # dead after both consumers read it into VGPRs, so K can then
            # safely overlay it.
            q_rows = block_q * heads
            k_pair_rows = 2 * block_n
            if not use_lds_overlay:
                q_shared = T.alloc_shared([q_rows, head_dim], T.float8_e4m3fn)
                k_ping_shared = T.alloc_shared([block_n, head_dim], T.float8_e4m3fn)
                k_pong_shared = T.alloc_shared([block_n, head_dim], T.float8_e4m3fn)
            else:
                shared_arena = T.alloc_shared(
                    [max(q_rows, k_pair_rows), head_dim], T.float8_e4m3fn
                )
                q_shared = T.decl_buffer(
                    [q_rows, head_dim],
                    T.float8_e4m3fn,
                    data=shared_arena.data,
                    elem_offset=0,
                    scope="shared",
                )
                k_ping_shared = T.decl_buffer(
                    [block_n, head_dim],
                    T.float8_e4m3fn,
                    data=shared_arena.data,
                    elem_offset=0,
                    scope="shared",
                )
                k_pong_shared = T.decl_buffer(
                    [block_n, head_dim],
                    T.float8_e4m3fn,
                    data=shared_arena.data,
                    elem_offset=block_n * head_dim,
                    scope="shared",
                )

            scale_ping_shared = T.alloc_shared([block_n], T.float32)
            scale_pong_shared = T.alloc_shared([block_n], T.float32)

            q_a_local = T.alloc_fragment([consumer_cols, head_dim], T.float8_e4m3fn)
            k_ping_a_local = T.alloc_fragment([block_n, head_dim], T.float8_e4m3fn)
            k_pong_a_local = T.alloc_fragment([block_n, head_dim], T.float8_e4m3fn)
            scores_a = T.alloc_fragment([score_rows, consumer_cols], T.float32)
            scores_a_3d = T.reshape(scores_a, (score_rows, consumer_q, heads))
            reduced_a = T.alloc_fragment([score_rows, consumer_q], T.float32)

            q_b_local = T.alloc_fragment([consumer_cols, head_dim], T.float8_e4m3fn)
            k_ping_b_local = T.alloc_fragment([block_n, head_dim], T.float8_e4m3fn)
            k_pong_b_local = T.alloc_fragment([block_n, head_dim], T.float8_e4m3fn)
            scores_b = T.alloc_fragment([score_rows, consumer_cols], T.float32)
            scores_b_3d = T.reshape(scores_b, (score_rows, consumer_q, heads))
            reduced_b = T.alloc_fragment([score_rows, consumer_q], T.float32)

            starts_f = T.alloc_fragment([block_q], T.int32)
            ends_f = T.alloc_fragment([block_q], T.int32)
            start_min_f = T.alloc_fragment([1], T.int32)
            end_max_f = T.alloc_fragment([1], T.int32)
            for qi in T.Parallel(block_q):
                if rows_divisible:
                    starts_f[qi] = Starts[row_base + qi]
                    ends_f[qi] = Ends[row_base + qi]
                else:
                    if row_base + qi < rows:
                        starts_f[qi] = Starts[row_base + qi]
                        ends_f[qi] = Ends[row_base + qi]
                    else:
                        # Neutral elements for the CTA min/max reduction.
                        starts_f[qi] = keys
                        ends_f[qi] = 0
            T.reduce_min(starts_f, start_min_f, dim=0, clear=True)
            T.reduce_max(ends_f, end_max_f, dim=0, clear=True)
            start_min = T.min(start_min_f[0], keys)
            end_max = T.min(end_max_f[0], keys)
            # Starts is an offsets tensor and its values are guaranteed
            # non-negative by the operator contract.  Expose that fact to
            # the bounds prover so steady Logits stores do not retain
            # redundant lower-bound predicates (including the pong
            # canonical form `-block_n <= base`).
            T.assume(start_min >= 0)
            T.assume(end_max <= keys)
            global_n_tiles = T.ceildiv(end_max - start_min, block_n)
            split_begin = global_n_tiles * split_id // split_n
            split_end = global_n_tiles * (split_id + 1) // split_n
            split_start_min = start_min + split_begin * block_n
            n_tiles = split_end - split_begin
            full_pairs = n_tiles // 2
            padded_range = n_tiles * block_n
            tile_phase = 0
            if n_tile_phases > 1:
                has_partial_tail = split_start_min + padded_range > end_max
                phased_n_tiles = T.if_then_else(
                    has_partial_tail, n_tiles - 1, n_tiles
                )
                # Preserve the phase spacing of the complete logical range
                # when peeling a partial boundary tile.  The boundary itself
                # remains outside the wrapped steady traversal; clamp only
                # the small-range case where a phase could land on it.
                tile_phase = T.min(
                    T.bitwise_and(q_block, n_tile_phases - 1)
                    * n_tiles
                    // n_tile_phases,
                    T.max(phased_n_tiles - 1, 0),
                )
            else:
                phased_n_tiles = n_tiles
            phased_start_min = split_start_min + tile_phase * block_n
            phased_range = phased_n_tiles * block_n
            phased_wrap_end = split_start_min + phased_range
            if n_tile_phases > 1:
                last_tile_idx = n_tiles - 1
                logical_boundary_start = split_start_min + last_tile_idx * block_n
                traversal_boundary_start = tile_row(
                    phased_start_min,
                    last_tile_idx,
                    phased_wrap_end,
                    phased_range,
                )
                boundary_start = T.if_then_else(
                    has_partial_tail,
                    logical_boundary_start,
                    traversal_boundary_start,
                )
            else:
                boundary_start = tile_row(
                    phased_start_min,
                    n_tiles - 1,
                    phased_wrap_end,
                    phased_range,
                )
            tile_context = T.alloc_local((4,), T.int32)
            tile_context[0] = phased_start_min
            tile_context[1] = phased_wrap_end
            tile_context[2] = phased_range
            tile_context[3] = boundary_start

            if tx < 256:
                # The range reduction is uniform, but its fragment result
                # arrives in a VGPR.  Scalarize it once for the producer so
                # loop/tail control and MLS address updates stay on SALU.
                producer_start_min = T.call_pure_extern(
                    "__builtin_amdgcn_readfirstlane",
                    phased_start_min,
                    dtype="int32",
                )
                producer_wrap_end = T.call_pure_extern(
                    "__builtin_amdgcn_readfirstlane",
                    tile_context[1],
                    dtype="int32",
                )
                producer_wrap_span = T.call_pure_extern(
                    "__builtin_amdgcn_readfirstlane",
                    tile_context[2],
                    dtype="int32",
                )
                producer_boundary_start = T.call_pure_extern(
                    "__builtin_amdgcn_readfirstlane",
                    tile_context[3],
                    dtype="int32",
                )
                producer_n_tiles = T.call_pure_extern(
                    "__builtin_amdgcn_readfirstlane", n_tiles, dtype="int32"
                )
                producer_full_pairs = producer_n_tiles // 2
                # Keep the conditional wrap out of matrix_load's address
                # expression: HCU MLS codegen requires a named scalar
                # origin and otherwise inlines the Select statement into
                # make_array(), producing invalid C++.
                k_row = T.alloc_var(T.int32)

                # Producer address generation and matrix_load issue are on
                # the critical path for both consumer groups.
                priority_3()
                # Producer owns the single Q global->LDS transfer.  The
                # first barrier publishes Q to both consumer groups.
                if producer_n_tiles > 0:
                    T.matrix_load(
                        Q[row_base * heads, 0],
                        q_shared,
                        boundary=(not rows_divisible, False),
                    )

                    if producer_full_pairs > 0:
                        if not use_lds_overlay:
                            k_row = tile_row(
                                producer_start_min,
                                0,
                                producer_wrap_end,
                                producer_wrap_span,
                            )
                            T.matrix_load(
                                K[k_row, 0],
                                k_ping_shared,
                                boundary=(False, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_ping_shared)
                            T.sched_barrier()
                            T.s_waitcnt(2)
                            T.sync_warp()  # Q ready
                            T.s_waitcnt(0)
                            T.sync_warp()  # S_init: K ping(0) ready
                        else:
                            T.s_waitcnt(0)
                            T.sync_warp()  # Q ready
                            T.sync_warp()  # Q consumers released LDS
                            k_row = tile_row(
                                producer_start_min,
                                0,
                                producer_wrap_end,
                                producer_wrap_span,
                            )
                            T.matrix_load(
                                K[k_row, 0],
                                k_ping_shared,
                                boundary=(False, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_ping_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()  # S_init: K ping(0) ready

                        final_ping = 2 * (producer_full_pairs - 1)
                        final_pong = final_ping + 1
                        if producer_full_pairs > 1:
                            k_row = tile_row(
                                producer_start_min,
                                1,
                                producer_wrap_end,
                                producer_wrap_span,
                            )
                            T.matrix_load(
                                K[k_row, 0],
                                k_pong_shared,
                                boundary=(False, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_pong_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()

                            if producer_full_pairs > 2:
                                for pair_offset in T.Serial(producer_full_pairs - 2):
                                    pair = pair_offset + 1
                                    k_row = tile_row(
                                        producer_start_min,
                                        2 * pair,
                                        producer_wrap_end,
                                        producer_wrap_span,
                                    )
                                    T.matrix_load(
                                        K[k_row, 0],
                                        k_ping_shared,
                                        boundary=(False, False),
                                    )
                                    issue_scale_async(KvScales, k_row, scale_ping_shared)
                                    T.sched_barrier()
                                    T.s_waitcnt(0)
                                    T.sync_warp()

                                    k_row = tile_row(
                                        producer_start_min,
                                        2 * pair + 1,
                                        producer_wrap_end,
                                        producer_wrap_span,
                                    )
                                    T.matrix_load(
                                        K[k_row, 0],
                                        k_pong_shared,
                                        boundary=(False, False),
                                    )
                                    issue_scale_async(KvScales, k_row, scale_pong_shared)
                                    T.sched_barrier()
                                    T.s_waitcnt(0)
                                    T.sync_warp()

                            k_row = tile_row(
                                producer_start_min,
                                final_ping,
                                producer_wrap_end,
                                producer_wrap_span,
                            )
                            T.matrix_load(
                                K[k_row, 0],
                                k_ping_shared,
                                boundary=(False, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_ping_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()

                        if producer_n_tiles % 2 != 0:
                            k_row = tile_row(
                                producer_start_min,
                                final_pong,
                                producer_wrap_end,
                                producer_wrap_span,
                            )
                            T.matrix_load(
                                K[k_row, 0],
                                k_pong_shared,
                                boundary=(True, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_pong_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()

                            k_row = producer_boundary_start
                            T.matrix_load(
                                K[k_row, 0],
                                k_ping_shared,
                                boundary=(True, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_ping_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.sync_warp()  # S_tail: consumers finished read
                        else:
                            k_row = producer_boundary_start
                            T.matrix_load(
                                K[k_row, 0],
                                k_pong_shared,
                                boundary=(True, False),
                            )
                            issue_scale_async(KvScales, k_row, scale_pong_shared)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.sync_warp()

                    else:
                        T.s_waitcnt(0)
                        T.sync_warp()  # Q ready
                        if use_lds_overlay:
                            T.sync_warp()  # Q consumers released LDS
                        k_row = producer_boundary_start
                        T.matrix_load(
                            K[k_row, 0],
                            k_ping_shared,
                            boundary=(True, False),
                        )
                        issue_scale_async(KvScales, k_row, scale_ping_shared)
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()  # S_init
                        T.sync_warp()  # S_tail

            elif tx < consumer_split:
                consumer0_dispatch(
                    q_shared,
                    k_ping_shared,
                    k_pong_shared,
                    q_a_local,
                    k_ping_a_local,
                    k_pong_a_local,
                    scale_ping_shared,
                    scale_pong_shared,
                    scores_a,
                    scores_a_3d,
                    reduced_a,
                    Logits,
                    InvScale,
                    row_base,
                    tile_context,
                    end_max,
                    n_tiles,
                    full_pairs,
                )
            else:
                consumer1_dispatch(
                    q_shared,
                    k_ping_shared,
                    k_pong_shared,
                    q_b_local,
                    k_ping_b_local,
                    k_pong_b_local,
                    scale_ping_shared,
                    scale_pong_shared,
                    scores_b,
                    scores_b_3d,
                    reduced_b,
                    Logits,
                    InvScale,
                    row_base,
                    tile_context,
                    end_max,
                    n_tiles,
                    full_pairs,
                )

    return qsa_mqa_prefill_wasp_fp8
