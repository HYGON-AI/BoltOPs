# SPDX-License-Identifier: MIT

"""IHC GEMM + sqrsum (MHC pre_norm_splitk skeleton, block_n=16).

Logical N is n_out (8 for pre, 4 for head). Fragments pad to BLOCK_N=16
in-kernel; no host-side pad_fn_rows.
"""

import functools
import tilelang
from tilelang import language as T

from boltops.ihc.tilelang.configs.pre import BLOCK_N

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

# Round-to-nearest TF32 mantissa bias before use_tf32 GEMM (same as MHC).
_TF32_ROUND_BITS = 0x1000


@functools.cache
def ihc_gemm_sqrsum_splitk_kernel(
    n_out: int,
    k_total: int,
    split_k: int,
    token_block: int = 64,
    hidden_block: int = 128,
    threads: int = 256,
    separate_x_fn_smem: bool = True,
    block_n: int = BLOCK_N,
    use_tf32: bool = False,
) -> tilelang.JITKernel:
    """Stage-0 only: write partials shaped (split_k, T, n_out). Combine is in epilogue."""
    assert n_out <= block_n
    assert block_n == BLOCK_N
    assert k_total % hidden_block == 0
    assert k_total % split_k == 0
    split_size = k_total // split_k
    assert split_size % hidden_block == 0

    num_tokens = T.dynamic("num_tokens")

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def ihc_gemm_sqrsum_splitk_stage_0(
        x: T.Tensor[(num_tokens, k_total), T.bfloat16],
        fn: T.Tensor[(n_out, k_total), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, n_out), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(split_k, T.ceildiv(num_tokens, token_block), threads=threads) as (
            bz,
            px,
        ):
            out_frag = T.alloc_fragment((token_block, block_n), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 16), T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)

            k_base = bz * split_size

            for pz in T.Pipelined(split_size // hidden_block, num_stages=0):
                x_frag_pre = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                fn_frag_pre = T.alloc_fragment((block_n, hidden_block), T.float32)
                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                fn_frag = T.alloc_fragment((block_n, hidden_block), T.float32)

                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((block_n, hidden_block), T.float32)
                T.annotate_layout(
                    {
                        x_smem_16: tilelang.layout.make_hcu_swizzled_layout(
                            x_smem_16, major_pack=2
                        )
                    }
                )
                T.annotate_layout(
                    {
                        fn_smem: tilelang.layout.make_hcu_swizzled_layout(
                            fn_smem, major_pack=2
                        )
                    }
                )

                T.copy(x[px * token_block, k_base + pz * hidden_block], x_frag_pre)
                # Pad N in-kernel: clear block_n rows, then load only logical n_out.
                T.clear(fn_frag_pre)
                fn_valid = T.alloc_fragment((n_out, hidden_block), T.float32)
                T.copy(fn[0, k_base + pz * hidden_block], fn_valid)
                if use_tf32:
                    # Element-wise TF32 round on logical N (avoid T.view vs block_n pad conflict).
                    for i, j in T.Parallel(n_out, hidden_block):
                        vi = T.reinterpret(fn_valid[i, j], T.int32) + T.int32(_TF32_ROUND_BITS)
                        fn_frag_pre[i, j] = T.reinterpret(vi, T.float32)
                else:
                    for i, j in T.Parallel(n_out, hidden_block):
                        fn_frag_pre[i, j] = fn_valid[i, j]

                if separate_x_fn_smem:
                    T.copy(x_frag_pre, x_smem_16)
                    T.copy(fn_frag_pre, fn_smem)
                    T.copy(x_smem_16, x_frag_16)
                    T.copy(fn_smem, fn_frag)
                    T.copy(x_frag_16, x_frag)
                else:
                    T.copy(x_frag_pre, x_smem_16)
                    T.copy(x_smem_16, x_frag_16)
                    T.copy(x_frag_16, x_frag)
                    T.copy(fn_frag_pre, fn_smem)
                    T.copy(fn_smem, fn_frag)

                for jj in T.serial(hidden_block // 16):
                    for i, j in T.Parallel(token_block, 16):
                        v = x_frag[i, jj * 16 + j]
                        sq_part4[i, j] += v * v

                T.gemm(
                    x_frag,
                    fn_frag,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    k_pack=2,
                    policy=T.GemmWarpPolicy.FullRow,
                    use_tf32=use_tf32,
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)
            T.reduce_sum(sq_part4, sq_l)
            out_shared = T.alloc_shared((token_block, block_n), T.float32)
            T.copy(out_frag, out_shared)

            for i in T.Parallel(token_block):
                t = px * token_block + i
                if t < num_tokens:
                    sqrsum_partial[bz, t] = sq_l[i]

            for i, j in T.Parallel(token_block, block_n):
                t = px * token_block + i
                if t < num_tokens and j < n_out:
                    out_partial[bz, t, j] = out_shared[i, j]

    return ihc_gemm_sqrsum_splitk_stage_0
