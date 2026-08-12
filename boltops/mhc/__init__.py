"""Public interfaces for the Multi-Head Connection (MHC) operator family."""

from __future__ import annotations

import torch


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from .tilelang._dispatch import resolve

    return resolve("mhc_post_fwd", residual.device)(
        x, residual, post_layer_mix, comb_res_mix, out
    )


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 32,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .tilelang._dispatch import resolve

    return resolve("hc_split_sinkhorn", mixes.device)(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps,
        token_block_size,
        threads,
    )


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .tilelang._dispatch import resolve

    return resolve("mhc_pre_big_fuse", residual.device)(
        residual,
        fn,
        mhc_scale,
        mhc_base,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def pre_big_fuse_tilelang(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
) -> None:
    from .tilelang._dispatch import resolve

    return resolve("pre_big_fuse_tilelang", residual.device)(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        mhc_mult,
    )


def mhc_fused_tilelang(
    comb_mix: torch.Tensor,
    residual_in: torch.Tensor,
    post_mix: torch.Tensor,
    x_in: torch.Tensor,
    weight_t: torch.Tensor,
    yp_out: torch.Tensor,
    rp_out: torch.Tensor,
    residual_out: torch.Tensor,
    mhc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
):
    from .tilelang._dispatch import resolve

    return resolve("mhc_fused_tilelang", residual_in.device)(
        comb_mix,
        residual_in,
        post_mix,
        x_in,
        weight_t,
        yp_out,
        rp_out,
        residual_out,
        mhc,
        hidden,
        n_out,
        n_thr=n_thr,
        h_blk=h_blk,
        tile_n=tile_n,
        split_k=split_k,
    )


__all__ = [
    "hc_split_sinkhorn",
    "mhc_fused_tilelang",
    "mhc_post_fwd",
    "mhc_pre_big_fuse",
    "pre_big_fuse_tilelang",
]
