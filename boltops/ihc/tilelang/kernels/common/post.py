# SPDX-License-Identifier: MIT

"""IHC post: ``out = post * x + residual``.

Launch policy mirrors MHC ``mhc_post_fwd`` (CU-aware path select + ``h_blk=1024``),
but the body stays IHC-only (no comb). Large-T with ``T >= 4096`` and effective
``h_blk >= 1024`` reuses residual-smem / stage=1; otherwise large-T stays register-only.
"""

import math

import tilelang
import torch
from tilelang import language as T

from boltops.utility.arch_info import get_device_info

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}

_DEFAULT_H_BLK = 1024
_DEFAULT_N_THR = 128
# Large-T smem+stage1 only helps once tokens are big enough (measured on gfx938).
_LARGE_TOKEN_S1 = 4096


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _ihc_post_split_h(hc: int, hidden: int, n_thr: int = _DEFAULT_N_THR, h_blk: int = _DEFAULT_H_BLK):
    """Small-T path: split H across the grid to raise CU occupancy."""
    n = T.dynamic("num_tokens")
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def ihc_post_split_h(
        x: T.Tensor[(n, hidden), T.bfloat16],
        residual: T.Tensor[(n, hc, hidden), T.bfloat16],
        post: T.Tensor[(n, hc), T.float32],
        out: T.Tensor[(n, hc, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(n, T.ceildiv(hidden, h_blk), threads=n_thr) as (pid_n, pid_h):
            residual_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            x_local = T.alloc_fragment(h_blk, T.float32)
            residual_local = T.alloc_fragment((hc, h_blk), T.float32)
            post_local = T.alloc_fragment(hc, T.float32)
            out_local = T.alloc_fragment((hc, h_blk), T.float32)

            h_start = pid_h * h_blk
            T.copy(post[pid_n, 0], post_local)
            T.copy(residual[pid_n, 0, h_start], residual_shared, disable_tma=True)
            T.copy(residual_shared, residual_local)
            T.copy(x[pid_n, h_start], x_local, disable_tma=True)
            for i_hc, i_h in T.Parallel(hc, h_blk):
                out_local[i_hc, i_h] = post_local[i_hc] * x_local[i_h] + residual_local[i_hc, i_h]
            T.copy(out_local, out[pid_n, 0, h_start], disable_tma=True, coalesced_width=8)

    return ihc_post_split_h


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _ihc_post_pipeline(hc: int, hidden: int, n_thr: int = _DEFAULT_N_THR, h_blk: int = _DEFAULT_H_BLK):
    """One CTA/token: residual tile via smem + Pipelined(stage=1), write each tile.

    Used for (1) small-T when H cannot be split, and (2) large-T when
    ``num_tokens >= _LARGE_TOKEN_S1`` and effective ``h_blk >= _DEFAULT_H_BLK``.
    """
    n = T.dynamic("num_tokens")
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def ihc_post_pipeline(
        x: T.Tensor[(n, hidden), T.bfloat16],
        residual: T.Tensor[(n, hc, hidden), T.bfloat16],
        post: T.Tensor[(n, hc), T.float32],
        out: T.Tensor[(n, hc, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            residual_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            x_local = T.alloc_fragment(h_blk, T.float32)
            residual_local = T.alloc_fragment((hc, h_blk), T.float32)
            post_local = T.alloc_fragment(hc, T.float32)
            out_local = T.alloc_fragment((hc, h_blk), T.float32)

            T.copy(post[pid_n, 0], post_local)
            for i0_h in T.Pipelined(T.ceildiv(hidden, h_blk), num_stages=1):
                h_start = i0_h * h_blk
                T.copy(residual[pid_n, 0, h_start], residual_shared, disable_tma=True)
                T.copy(residual_shared, residual_local)
                T.copy(x[pid_n, h_start], x_local, disable_tma=True)
                for i_hc, i_h in T.Parallel(hc, h_blk):
                    out_local[i_hc, i_h] = post_local[i_hc] * x_local[i_h] + residual_local[i_hc, i_h]
                T.copy(out_local, out[pid_n, 0, h_start], disable_tma=True, coalesced_width=8)

    return ihc_post_pipeline


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _ihc_post_wo_shmem(hc: int, hidden: int, n_thr: int = _DEFAULT_N_THR, h_blk: int = _DEFAULT_H_BLK):
    """Large-T fallback: one CTA/token, register tiles only (no smem / stage=0)."""
    n = T.dynamic("num_tokens")
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def ihc_post_wo_shmem(
        x: T.Tensor[(n, hidden), T.bfloat16],
        residual: T.Tensor[(n, hc, hidden), T.bfloat16],
        post: T.Tensor[(n, hc), T.float32],
        out: T.Tensor[(n, hc, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_local = T.alloc_fragment(h_blk, T.float32)
            residual_local = T.alloc_fragment((hc, h_blk), T.float32)
            post_local = T.alloc_fragment(hc, T.float32)
            out_local = T.alloc_fragment((hc, h_blk), T.float32)

            T.copy(post[pid_n, 0], post_local)
            for i0_h in T.Pipelined(T.ceildiv(hidden, h_blk), num_stages=0):
                h_start = i0_h * h_blk
                T.copy(x[pid_n, h_start], x_local, disable_tma=True)
                T.copy(residual[pid_n, 0, h_start], residual_local, disable_tma=True)
                for i_hc, i_h in T.Parallel(hc, h_blk):
                    out_local[i_hc, i_h] = post_local[i_hc] * x_local[i_h] + residual_local[i_hc, i_h]
                T.copy(out_local, out[pid_n, 0, h_start], disable_tma=True, coalesced_width=8)

    return ihc_post_wo_shmem


def ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    cu_count = get_device_info(residual.device).cu_count
    num_tokens, hc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f"x.dtype={x.dtype}"
    assert residual.dtype == torch.bfloat16, f"residual.dtype={residual.dtype}"
    assert post.dtype == torch.float32, f"post.dtype={post.dtype}"
    assert x.shape == (num_tokens, hidden), f"x.shape={x.shape}"
    assert post.shape == (num_tokens, hc), f"post.shape={post.shape}"

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)

    h_blk = math.gcd(hidden, _DEFAULT_H_BLK)
    h_tiles = hidden // h_blk
    n_thr = _DEFAULT_N_THR
    use_s1_large_t = num_tokens >= _LARGE_TOKEN_S1 and h_blk >= _DEFAULT_H_BLK
    if num_tokens < cu_count * 2 and h_tiles > 1:
        kernel = _ihc_post_split_h(hc, hidden, n_thr=n_thr)
    elif num_tokens < cu_count * 2 or use_s1_large_t:
        # Small-T no-split, or large-T (T>=4096) with large h_blk: smem + stage=1.
        kernel = _ihc_post_pipeline(hc, hidden, n_thr=n_thr)
    else:
        kernel = _ihc_post_wo_shmem(hc, hidden, n_thr=n_thr)
    kernel(x, residual, post, out)
    return out
