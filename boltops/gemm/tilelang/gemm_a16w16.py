# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/gemm/basic/gemm_a16w16.py
import tilelang
import tilelang.language as T

shape = T.dynamic("M, N, K")

@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def _gemm_a16_w16_kernel(
    M,
    N,
    K,
    dtype: str,
    out_dtype: str,
    activation,
    add_bias: bool,
    block_M: int = 256,
    block_N: int = 256,
    block_K: int = 32,
    threads: int = 512,
    num_stages: int = 0,
    enable_rasteration=False,
    group_size=1,
    wgs_per_cu=1,
    accum_dtype="float",
    use_mls=False,
):
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks  # noqa: F841

    split_n = 2
    sub_block_N = block_N // split_n
    k_loop_ = T.ceildiv(K, block_K)
    k_loop = T.ceildiv(k_loop_, 2)
    remain = 2 - (k_loop_ % 2)
    store_vecsize = (8 if N % 8 == 0 else 4 if N % 4 == 0 else 2 if N % 2 == 0 else 1) if isinstance(N, int) else 1

    @T.prim_func
    def _gemm_a16_w16_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        Bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            T.use_swizzle(
                panel_size=group_size,
                order="row",
                enable=group_size > 1 or enable_rasteration,
            )
            if K >= 8192:
                T.cache_warmup(
                    A[by * block_M : (by + 1) * block_M, :],
                    W[bx * block_N : (bx + 1) * block_N, :],
                )
            warp_idx = T.get_warp_idx()
            A_shared = T.alloc_shared((2, block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((2, sub_block_N, block_K), dtype)
            B_shared_1 = T.alloc_shared((2, sub_block_N, block_K), dtype)

            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K), dtype)

            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            local_out0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            local_out1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), out_dtype)
            T.annotate_layout(
                {
                    C_shared_0: tilelang.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                }
            )

            T.clear(C_local_0)
            T.clear(C_local_1)

            if add_bias:
                for i, j in T.Parallel(block_M, sub_block_N):
                    if bx * block_N + j < N:
                        C_local_0[i, j] = T.cast(Bias[bx * block_N + j], accum_dtype)
                    if bx * block_N + sub_block_N + j < N:
                        C_local_1[i, j] = T.cast(Bias[bx * block_N + sub_block_N + j], accum_dtype)

            if warp_idx < 4:
                T.matrix_load(A[by * block_M, 0], A_shared[0, :, :])
                T.matrix_load(W[bx * block_N, 0], B_shared_0[0, :, :])
                T.matrix_load(W[bx * block_N + sub_block_N, 0], B_shared_1[0, :, :])
                T.sched_barrier()
                T.ptx_wait_group(0)
                T.sync_warp()
                T.matrix_load(A[by * block_M, block_K], A_shared[1, :, :])
                T.matrix_load(W[bx * block_N, block_K], B_shared_0[1, :, :])
                T.matrix_load(W[bx * block_N + sub_block_N, block_K], B_shared_1[1, :, :])
                T.sched_barrier()
                T.ds_read_format(A_shared[0, :, :], A_local_0)
                T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                T.ds_read_format(B_shared_1[0, :, :], B_local_1)

                for k in T.Serial(k_loop - 1):
                    base = 2 * (k + 1)
                    # T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()
                    T.sched_barrier()
                    T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                    T.matrix_load(W[bx * block_N, base * block_K], B_shared_0[0, :, :])
                    T.matrix_load(W[bx * block_N + sub_block_N, base * block_K], B_shared_1[0, :, :])
                    T.ptx_wait_group(3)
                    T.sync_warp()
                    T.ds_read_format(A_shared[1, :, :], A_local_0_)
                    T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=True)
                    T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=True)
                    T.sched_barrier()
                    T.ds_read_format(B_shared_1[1, :, :], B_local_1_)

                    T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()
                    T.sched_barrier()
                    T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared[1, :, :])
                    T.matrix_load(W[bx * block_N, (base + 1) * block_K], B_shared_0[1, :, :])
                    T.matrix_load(W[bx * block_N + sub_block_N, (base + 1) * block_K], B_shared_1[1, :, :])
                    T.sched_barrier()
                    T.ptx_wait_group(3)
                    T.sync_warp()
                    T.ds_read_format(A_shared[0, :, :], A_local_0)
                    T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                    T.sched_barrier()
                    T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=True)
                    T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=True)
                    T.sched_barrier()
                    T.ds_read_format(B_shared_1[0, :, :], B_local_1)
            else:
                T.matrix_load(A[by * block_M, 0], A_shared[0, :, :])
                T.matrix_load(W[bx * block_N, 0], B_shared_0[0, :, :])
                T.matrix_load(W[bx * block_N + sub_block_N, 0], B_shared_1[0, :, :])
                T.sched_barrier()
                T.ptx_wait_group(0)
                T.sync_warp()
                T.ds_read_format(A_shared[0, :, :], A_local_0)
                T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                T.sched_barrier()
                T.matrix_load(A[by * block_M, block_K], A_shared[1, :, :])
                T.matrix_load(W[bx * block_N, block_K], B_shared_0[1, :, :])
                T.matrix_load(W[bx * block_N + sub_block_N, block_K], B_shared_1[1, :, :])

                for k in T.Serial(k_loop - 1):
                    base = 2 * (k + 1)
                    # T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()

                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=True)
                    T.sched_barrier()
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                    T.matrix_load(W[bx * block_N, base * block_K], B_shared_0[0, :, :])
                    T.matrix_load(W[bx * block_N + sub_block_N, base * block_K], B_shared_1[0, :, :])
                    T.s_waitcnt(15)  # just to avoid clang auto insert vmcnt 0
                    T.ds_read_format(A_shared[1, :, :], A_local_0_)
                    T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                    T.ds_read_format(B_shared_1[1, :, :], B_local_1_)
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=True)

                    T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()

                    T.sched_barrier()
                    T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=True)
                    T.sched_barrier()
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared[1, :, :])
                    T.matrix_load(W[bx * block_N, (base + 1) * block_K], B_shared_0[1, :, :])
                    T.matrix_load(W[bx * block_N + sub_block_N, (base + 1) * block_K], B_shared_1[1, :, :])
                    T.s_waitcnt(15)  # just to avoid clang auto insert vmcnt 0
                    T.ds_read_format(A_shared[0, :, :], A_local_0)
                    T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                    T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                    T.sched_barrier()
                    T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=True)

            if remain > 1:
                T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=True)
                T.ptx_wait_group(0)
                T.sync_warp()
                T.ds_read_format(A_shared[1, :, :], A_local_0_)
                T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                T.ds_read_format(B_shared_1[1, :, :], B_local_1_)
                T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=True)
                T.sched_barrier()
                T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=True)
                T.sync_warp()
                if activation is not None:
                    for i, j in T.Parallel(block_M, sub_block_N):
                        C_local_0[i, j] = activation(C_local_0[i, j])
                T.copy(C_local_0, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                T.copy(local_out0, C[by * block_M, bx * block_N])
                T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=True)
                T.sync_threads()
                if activation is not None:
                    for i, j in T.Parallel(block_M, sub_block_N):
                        C_local_1[i, j] = activation(C_local_1[i, j])
                T.copy(C_local_1, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])
            else:
                T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=True)
                T.sync_warp()
                if activation is not None:
                    for i, j in T.Parallel(block_M, sub_block_N):
                        C_local_0[i, j] = activation(C_local_0[i, j])
                T.copy(C_local_0, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                T.copy(local_out0, C[by * block_M, bx * block_N])
                T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=True)
                T.sync_threads()
                if activation is not None:
                    for i, j in T.Parallel(block_M, sub_block_N):
                        C_local_1[i, j] = activation(C_local_1[i, j])
                T.copy(C_local_1, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_a16_w16_kernel


def get_swizzle_group_size(
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
    *,
    threshold: int = 4096,
    large_group_size: int = 8,
    small_group_size: int = 1,
) -> int:
    """Pick threadblock swizzle panel size from GEMM shape.

    When any of M/N/K exceeds ``threshold``, use ``large_group_size`` (L2 tile
    swizzle); otherwise disable threadblock swizzle via ``small_group_size`` (1).
    """
    if M is None or N is None or K is None:
        return small_group_size
    if max(M, N, K) > threshold:
        return large_group_size
    return small_group_size

__all__ = ["_gemm_a16_w16_kernel", "get_swizzle_group_size"]
