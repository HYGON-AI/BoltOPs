# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mhc import mhc_fused_tilelang


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.einsum("abmn,abmc->abnc", comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_tokens",
    # [1, 34, 65, 128, 133, 288, 577, 1010, 2722, 4572, 8192, 9217, 21111],
    [1, 128],
)
@pytest.mark.parametrize(
    "hidden_size",
    # [4096, 7168],
    [4096, 7168],
)
@pytest.mark.parametrize("mhc_mult", [4])
def test_mhc_fused_tilelang_accuracy(num_tokens: int, hidden_size: int, mhc_mult: int) -> None:
    device = "cuda"
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    residual = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((num_tokens, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((num_tokens, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float32, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    # Keep split strategy aligned with fused kernel call site.
    fma_token_threshold = 16
    if num_tokens <= fma_token_threshold:
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
        tile_n = 2 if num_tokens < 8 else 3
    else:
        # conservative fixed split for test stability
        n_splits = 2
        tile_n = 1

    gemm_out_mul = torch.empty((n_splits, num_tokens, mhc_mult3), dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.empty((n_splits, num_tokens), dtype=torch.float32, device=device)
    residual_out = torch.empty_like(residual)

    mhc_fused_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        fn.view(mhc_mult3, mhc_mult, hidden_size),
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_out,
        mhc_mult,
        hidden_size,
        mhc_mult3,
        tile_n=tile_n,
        split_k=n_splits,
    )

    residual_ref = mhc_post_ref(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post_layer_mix.unsqueeze(0),
        comb_res_mix.unsqueeze(0),
    ).squeeze(0)

    h_per_split = hidden_size // n_splits
    residual_fp32 = residual_out.float()
    fn_3d = fn.view(mhc_mult3, mhc_mult, hidden_size)
    gemm_ref = torch.zeros_like(gemm_out_mul)
    sqr_ref = torch.zeros_like(gemm_out_sqrsum)
    for ks in range(n_splits):
        hs = ks * h_per_split
        he = hs + h_per_split
        x_part = residual_fp32[:, :, hs:he]
        w_part = fn_3d[:, :, hs:he]
        gemm_ref[ks, :, :] = torch.einsum("tmh,omh->to", x_part, w_part)
        sqr_ref[ks, :] = x_part.square().sum(dim=(1, 2))

    torch.testing.assert_close(residual_out, residual_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gemm_out_mul, gemm_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gemm_out_sqrsum, sqr_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.bench
@pytest.mark.parametrize(
    "num_tokens,hidden_size,mhc_mult",
    [
        (1, 4096, 4), (4, 4096, 4), (8, 4096, 4), (32, 4096, 4),
        (32, 7168, 4), (64, 4096, 4), (64, 7168, 4),
        (128, 4096, 4), (128, 7168, 4),
        (1024, 4096, 4), (1024, 7168, 4),
    ],
)
def test_bench_mhc_fused_tilelang(
    num_tokens: int,
    hidden_size: int,
    mhc_mult: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.bfloat16, device="cuda")
    post_mix = torch.randn((num_tokens, mhc_mult), dtype=torch.float32, device="cuda")
    comb_mix = torch.randn((num_tokens, mhc_mult, mhc_mult), dtype=torch.float32, device="cuda")
    weight = torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float32, device="cuda") * 1e-4
    if num_tokens <= 16:
        split_k = 8 if num_tokens < 8 and hidden_size <= 4096 else 4
        tile_n = 2 if num_tokens < 8 else 3
    else:
        split_k, tile_n = 2, 1
    yp = torch.empty((split_k, num_tokens, mhc_mult3), dtype=torch.float32, device="cuda")
    rp = torch.empty((split_k, num_tokens), dtype=torch.float32, device="cuda")
    residual_out = torch.empty_like(residual)
    fn = lambda: mhc_fused_tilelang(
        comb_mix, residual, post_mix, x, weight, yp, rp, residual_out,
        mhc_mult, hidden_size, mhc_mult3, tile_n=tile_n, split_k=split_k,
    )
    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    print(
        f"[bench] mhc_fused_tilelang tokens={num_tokens} H={hidden_size} "
        f"MHC={mhc_mult}: {ms:.4f} ms"
    )
