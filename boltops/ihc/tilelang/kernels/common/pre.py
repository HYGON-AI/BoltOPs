# SPDX-License-Identifier: MIT

"""ihc_pre: split-k GEMM + bigfuse-style combine/epilogue (no comb / Sinkhorn)."""

import math

import tilelang
import torch
from tilelang import language as T

from boltops.ihc.tilelang.configs.pre import resolve_get_config
from boltops.ihc.tilelang.kernels.common.gemm import ihc_gemm_sqrsum_splitk_kernel
from boltops.utility.arch_info import get_device_info

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _ihc_pre_epilogue(
    hc: int,
    hidden: int,
    n_splits: int,
    rms_eps: float,
    hc_eps: float,
    magnitude: float,
):
    """Combine split-k partials, then CTA-split like MHC bigfuse:

    - all: combine -> mixes_shared (incl. n_splits>=4 parallel reduce)
    - tx < 64: write post gates
    - tx >= 64: compute pre + apply-mix
    """
    n = T.dynamic("num_tokens")
    n_out = hc * 2
    k_total = hc * hidden
    # Apply tile fixed to MHC policy: gcd(512, H) → 512 when H%512==0.
    h_blk = math.gcd(hidden, 512)
    threads = 128
    n_splits_aligned = tilelang.math.next_power_of_2(n_splits)
    # IHC n_out is already a power of 2 (8); no pad-to-32.
    max_groups = threads // n_out  # 16
    # Pad split rows to pow2 so non-pow2 sk can use Parallel/grouped reduce
    # (OOB pad rows stay 0). max_groups is pow2, so aligned % max_groups == 0
    # whenever aligned > max_groups.
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
    def ihc_pre_epilogue(
        residual: T.Tensor[(n, hc, hidden), T.bfloat16],
        gemm_out_mul: T.Tensor[(n_splits, n, n_out), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, n), T.float32],
        scale: T.Tensor[(2,), T.float32],
        base: T.Tensor[(n_out,), T.float32],
        layer_in: T.Tensor[(n, hidden), T.bfloat16],
        post: T.Tensor[(n, hc), T.float32],
    ) -> None:
        with T.Kernel(n, threads=threads) as pid:
            tx = T.get_thread_binding()
            mixes_shared = T.alloc_shared(n_out, T.float32)
            rms = T.alloc_fragment(1, T.float32)

            if use_copy_reduce:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes_pre = T.alloc_fragment((n_splits_aligned, n_out), T.float32)
                mixes_aligned = T.alloc_fragment(n_out, T.float32)
                T.clear(mixes_pre)
                for i, j in T.Parallel(n_splits_aligned, n_out):
                    if i < n_splits:
                        mixes_pre[i, j] = gemm_out_mul[i, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(n_out):
                    mixes_shared[i] = mixes_aligned[i] * rms[0]
            elif use_grouped_reduce:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes_pre = T.alloc_fragment((split_groups, n_out), T.float32)
                mixes_aligned = T.alloc_fragment(n_out, T.float32)
                T.clear(mixes_pre)
                for r in T.serial(group_rows):
                    for i, j in T.Parallel(split_groups, n_out):
                        si = i * group_rows + r
                        if si < n_splits:
                            mixes_pre[i, j] += gemm_out_mul[si, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(n_out):
                    mixes_shared[i] = mixes_aligned[i] * rms[0]
            else:
                # n_splits == 1
                rms[0] = gemm_out_sqrsum[0, pid]
                rms[0] = T.rsqrt(rms[0] / T.float32(k_total) + rms_eps)
                mixes = T.alloc_fragment(n_out, T.float32)
                for j in T.Parallel(n_out):
                    mixes[j] = gemm_out_mul[0, pid, j] * rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if tx < 64:
                for j in T.Parallel(hc):
                    post[pid, j] = (
                        magnitude
                        * T.sigmoid(mixes_shared[j + hc] * scale[1] + base[j + hc])
                        + hc_eps
                    )
            else:
                pre_mix = T.alloc_fragment(hc, T.float32)
                for j in T.serial(hc):
                    pre_mix[j] = (
                        T.sigmoid(mixes_shared[j] * scale[0] + base[j]) + hc_eps
                    )

                apply_threads = 64
                lane = tx - 64
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
                        layer_in[pid, h_base + v] = ol[v]

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
                    layer_in[pid, last_base + v] = ol[v]

    return ihc_pre_epilogue


def ihc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    magnitude: float,
    layer_in: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
):
    num_tokens, hc, hidden = residual.shape
    n_out = 2 * hc
    k_total = hc * hidden
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert scale.dtype == torch.float32 and scale.numel() == 2
    assert base.dtype == torch.float32 and base.shape == (n_out,)
    assert fn.shape == (n_out, k_total)

    scale = scale.view(2)
    x_flat = residual.view(num_tokens, k_total)
    get_config = resolve_get_config(get_device_info(residual.device).arch)
    cfg = get_config(num_tokens, k_total)

    gemm_out_mul = torch.empty(
        (cfg.split_k, num_tokens, n_out), dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        (cfg.split_k, num_tokens), dtype=torch.float32, device=residual.device
    )
    ihc_gemm_sqrsum_splitk_kernel(
        n_out,
        k_total,
        cfg.split_k,
        token_block=cfg.token_block,
        hidden_block=cfg.hidden_block,
        separate_x_fn_smem=cfg.separate_x_fn_smem,
        use_tf32=True,
    )(x_flat, fn, gemm_out_mul, gemm_out_sqrsum)

    if layer_in is None:
        layer_in = torch.empty(
            (num_tokens, hidden), dtype=torch.bfloat16, device=residual.device
        )
    if post is None:
        post = torch.empty((num_tokens, hc), dtype=torch.float32, device=residual.device)

    _ihc_pre_epilogue(
        hc, hidden, cfg.split_k, float(rms_eps), float(hc_eps), float(magnitude)
    )(residual, gemm_out_mul, gemm_out_sqrsum, scale, base, layer_in, post)
    return layer_in, post
