# SPDX-License-Identifier: MIT

"""ihc_head: split-k GEMM + bigfuse-style combine; head gates only (n_out=hc)."""

import math

import tilelang
import torch
from tilelang import language as T

from boltops.ihc.tilelang.configs.head import resolve_get_config
from boltops.ihc.tilelang.kernels.common.gemm import ihc_gemm_sqrsum_splitk_kernel
from boltops.utility.arch_info import get_device_info

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _ihc_head_epilogue(hc: int, hidden: int, n_splits: int, rms_eps: float, hc_eps: float):
    """No post work: combine on full CTA, apply-mix on 64-thread CTA."""
    n = T.dynamic("num_tokens")
    k_total = hc * hidden
    # Apply tile fixed to MHC policy: gcd(512, H).
    h_blk = math.gcd(hidden, 512)
    threads = 64
    n_splits_aligned = tilelang.math.next_power_of_2(n_splits)
    # head n_out=hc (4); max_groups = 64 // 4 = 16.
    max_groups = threads // hc
    # Pad split rows to pow2 so non-pow2 sk can use Parallel/grouped reduce.
    use_copy_reduce = n_splits >= 2 and n_splits_aligned <= max_groups
    use_grouped_reduce = (
        n_splits >= 2
        and n_splits_aligned > max_groups
        and n_splits_aligned % max_groups == 0
    )
    if use_grouped_reduce:
        split_groups = max_groups
        group_rows = n_splits_aligned // split_groups

    @T.prim_func
    def ihc_head_epilogue(
        residual: T.Tensor[(n, hc, hidden), T.bfloat16],
        gemm_out_mul: T.Tensor[(n_splits, n, hc), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, n), T.float32],
        head_scale: T.Tensor[(1,), T.float32],
        head_base: T.Tensor[(hc,), T.float32],
        hidden_out: T.Tensor[(n, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=threads) as pid:
            tx = T.get_thread_binding()
            mixes_shared = T.alloc_shared(hc, T.float32)
            rms = T.alloc_fragment(1, T.float32)

            if use_copy_reduce:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes_pre = T.alloc_fragment((n_splits_aligned, hc), T.float32)
                mixes_aligned = T.alloc_fragment(hc, T.float32)
                T.clear(mixes_pre)
                for i, j in T.Parallel(n_splits_aligned, hc):
                    if i < n_splits:
                        mixes_pre[i, j] = gemm_out_mul[i, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(hc):
                    mixes_shared[i] = mixes_aligned[i] * rms[0]
            elif use_grouped_reduce:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes_pre = T.alloc_fragment((split_groups, hc), T.float32)
                mixes_aligned = T.alloc_fragment(hc, T.float32)
                T.clear(mixes_pre)
                for r in T.serial(group_rows):
                    for i, j in T.Parallel(split_groups, hc):
                        si = i * group_rows + r
                        if si < n_splits:
                            mixes_pre[i, j] += gemm_out_mul[si, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(hc):
                    mixes_shared[i] = mixes_aligned[i] * rms[0]
            else:
                # n_splits == 1
                rms[0] = gemm_out_sqrsum[0, pid]
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes = T.alloc_fragment(hc, T.float32)
                for j in T.Parallel(hc):
                    mixes[j] = gemm_out_mul[0, pid, j] * rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            scale0 = head_scale[0]
            pre_mix = T.alloc_fragment(hc, T.float32)
            for j in T.serial(hc):
                pre_mix[j] = T.sigmoid(mixes_shared[j] * scale0 + head_base[j]) + hc_eps

            apply_threads = 64
            lane = tx
            vec_size = h_blk // apply_threads
            tile_elems = hc * vec_size
            h_iters = hidden // h_blk
            xl = T.alloc_local((2 * tile_elems,), T.bfloat16)
            ol = T.alloc_local((vec_size,), T.float32)

            for i_hc in T.serial(hc):
                off = i_hc * vec_size
                for v in T.vectorized(vec_size):
                    xl[off + v] = residual[pid, i_hc, lane * vec_size + v]

            for i0_h in T.serial(h_iters - 1):
                cur_slot = (i0_h % 2) * tile_elems
                nxt_slot = ((i0_h + 1) % 2) * tile_elems
                h_base = i0_h * h_blk + lane * vec_size
                next_base = (i0_h + 1) * h_blk + lane * vec_size

                for i_hc in T.serial(hc):
                    off = nxt_slot + i_hc * vec_size
                    for v in T.vectorized(vec_size):
                        xl[off + v] = residual[pid, i_hc, next_base + v]

                for v in T.vectorized(vec_size):
                    ol[v] = 0.0
                for i_hc in T.serial(hc):
                    pre = pre_mix[i_hc]
                    off = cur_slot + i_hc * vec_size
                    for v in T.vectorized(vec_size):
                        ol[v] += pre * T.Cast(T.float32, xl[off + v])
                for v in T.vectorized(vec_size):
                    hidden_out[pid, h_base + v] = ol[v]

            last_h = h_iters - 1
            last_slot = (last_h % 2) * tile_elems
            last_base = last_h * h_blk + lane * vec_size
            for v in T.vectorized(vec_size):
                ol[v] = 0.0
            for i_hc in T.serial(hc):
                pre = pre_mix[i_hc]
                off = last_slot + i_hc * vec_size
                for v in T.vectorized(vec_size):
                    ol[v] += pre * T.Cast(T.float32, xl[off + v])
            for v in T.vectorized(vec_size):
                hidden_out[pid, last_base + v] = ol[v]

    return ihc_head_epilogue


def ihc_head(
    residual: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    hidden: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, hc, hidden_size = residual.shape
    k_total = hc * hidden_size
    assert residual.dtype == torch.bfloat16
    assert head_fn.dtype == torch.float32
    assert head_fn.shape == (hc, k_total)
    assert head_base.dtype == torch.float32 and head_base.shape == (hc,)

    head_scale = head_scale.reshape(1).to(dtype=torch.float32)
    x_flat = residual.view(num_tokens, k_total)
    get_config = resolve_get_config(get_device_info(residual.device).arch)
    cfg = get_config(num_tokens, k_total)

    gemm_out_mul = torch.empty(
        (cfg.split_k, num_tokens, hc), dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        (cfg.split_k, num_tokens), dtype=torch.float32, device=residual.device
    )
    ihc_gemm_sqrsum_splitk_kernel(
        hc,
        k_total,
        cfg.split_k,
        token_block=cfg.token_block,
        hidden_block=cfg.hidden_block,
        separate_x_fn_smem=cfg.separate_x_fn_smem,
        use_tf32=True,
    )(x_flat, head_fn, gemm_out_mul, gemm_out_sqrsum)

    if hidden is None:
        hidden = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=residual.device
        )
    _ihc_head_epilogue(hc, hidden_size, cfg.split_k, float(rms_eps), float(hc_eps))(
        residual, gemm_out_mul, gemm_out_sqrsum, head_scale, head_base, hidden
    )
    return hidden
