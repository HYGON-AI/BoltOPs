# SPDX-License-Identifier: MIT

import math

import tilelang
import torch
from tilelang import language as T

from boltops.utility.arch_info import get_device_info


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def _mhc_post_fwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=1):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(b_shared, b_local)
                T.copy(d[pid_n, i0_h * h_blk], d_local, disable_tma=True)

                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                T.copy(x_local, x[pid_n, 0, i0_h * h_blk], disable_tma=True, coalesced_width=8)

    return _mhc_post_fwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def _mhc_post_fwd_split_h(
    mhc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_split_h_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, T.ceildiv(h, h_blk), threads=n_thr) as (pid_n, pid_h):
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            h_start = pid_h * h_blk
            T.copy(b[pid_n, 0, h_start], b_shared, disable_tma=True)
            T.copy(b_shared, b_local)
            T.copy(d[pid_n, h_start], d_local, disable_tma=True)

            for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                for i_mhci in T.serial(mhc):
                    x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
            T.copy(x_local, x[pid_n, 0, h_start], disable_tma=True, coalesced_width=8)

    return _mhc_post_fwd_split_h_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def _mhc_post_fwd_wo_shmem(
    mhc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_wo_shmem_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=0):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_local, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_local, disable_tma=True)

                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]

                T.copy(x_local, x[pid_n, 0, i0_h * h_blk], disable_tma=True, coalesced_width=8)

    return _mhc_post_fwd_wo_shmem_kernel


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    cu_count = get_device_info(residual.device).cu_count
    num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f"x.dtype={x.dtype}"
    assert residual.dtype == torch.bfloat16, f"residual.dtype={residual.dtype}"
    assert post_layer_mix.dtype == torch.float32, f"post_layer_mix.dtype={post_layer_mix.dtype}"
    assert comb_res_mix.dtype == torch.float32, f"comb_res_mix.dtype={comb_res_mix.dtype}"
    assert x.shape == (num_tokens, hidden), f"x.shape={x.shape}"
    assert post_layer_mix.shape == (num_tokens, mhc), f"post_layer_mix.shape={post_layer_mix.shape}"
    assert comb_res_mix.shape == (num_tokens, mhc, mhc), f"comb_res_mix.shape={comb_res_mix.shape}"


    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)
    n = num_tokens
    h_tiles = math.gcd(hidden, 1024)
    h_tiles = hidden // h_tiles
    n_thr = 128
    if n < cu_count * 2 and h_tiles > 1:
        # increase cu num usage by adding h_split
        kernel = _mhc_post_fwd_split_h(mhc, hidden, n_thr=n_thr)
    elif n < cu_count * 2:
        # use shared mem and stage pipeline
        kernel = _mhc_post_fwd(mhc, hidden, n_thr=n_thr)
    else:
        # only use registers and no pipeline
        kernel = _mhc_post_fwd_wo_shmem(mhc, hidden, n_thr=n_thr)
    kernel(
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        out,
    )
    return out
