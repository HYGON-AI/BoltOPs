# ruff: noqa
import pytest
import torch

pytest.importorskip("tilelang")
from tilelang.profiler import do_bench_cudagraph

from boltops.mhc import mhc_pre_big_fuse

GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HCU device is required"
)


def _sinkhorn_matrix_torch(comb: torch.Tensor, sinkhorn_iters: int, eps: float) -> torch.Tensor:
    row_max = comb.max(dim=2, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return comb


def generate_big_fuse_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> dict[str, torch.Tensor | float]:
    n0 = 1
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    device = "cuda"

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    mhc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "mhc_scale": mhc_scale,
        "mhc_base": mhc_base,
        "rms_eps": rms_eps,
        "mhc_pre_eps": mhc_pre_eps,
        "mhc_sinkhorn_eps": mhc_sinkhorn_eps,
        "mhc_post_mult_value": mhc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
        "n_splits": n_splits,
    }


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del n_splits

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x_flat = residual_flat.view(num_tokens, mhc_mult * hidden_size).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.matmul(x_flat, fn.t()) * rsqrt

    pre_mix = torch.sigmoid(mixes[:, :mhc_mult] * mhc_scale[0] + mhc_base[:mhc_mult]) + mhc_pre_eps
    post_mix = (
        torch.sigmoid(mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1] + mhc_base[mhc_mult : 2 * mhc_mult])
        * mhc_post_mult_value
    )
    comb_mix = mixes[:, 2 * mhc_mult : mhc_mult3].view(num_tokens, mhc_mult, mhc_mult) * mhc_scale[2]
    comb_mix = comb_mix + mhc_base[2 * mhc_mult : mhc_mult3].view(1, mhc_mult, mhc_mult)
    comb_mix = _sinkhorn_matrix_torch(comb_mix, sinkhorn_repeat, mhc_sinkhorn_eps)

    layer_input = torch.einsum("nh,nhd->nd", pre_mix, residual_flat.float()).to(torch.bfloat16)

    outer_shape = residual.shape[:-2]
    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


@GPU_REQUIRED
@pytest.mark.parametrize(
    "n1",
    # [1, 34, 65, 128, 133, 288, 577, 1010, 2722, 4572, 8192, 9217, 21111],
    [1, 128],
)
@pytest.mark.parametrize(
    "hidden_size",
    # [1280, 2560, 4096, 7168],
    [4096, 7168],
)
@pytest.mark.parametrize("mhc_mult", [4])
def test_correctness(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    test_data = generate_big_fuse_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
    )

    post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre_big_fuse(
        test_data["residual"],
        test_data["fn"],
        test_data["mhc_scale"],
        test_data["mhc_base"],
        rms_eps=test_data["rms_eps"],
        mhc_pre_eps=test_data["mhc_pre_eps"],
        mhc_sinkhorn_eps=test_data["mhc_sinkhorn_eps"],
        mhc_post_mult_value=test_data["mhc_post_mult_value"],
        sinkhorn_repeat=test_data["sinkhorn_repeat"],
        n_splits=test_data["n_splits"],
    )

    post_mix_ref, comb_mix_ref, layer_input_ref = big_fuse_reference(
        test_data["residual"],
        test_data["fn"],
        test_data["mhc_scale"],
        test_data["mhc_base"],
        test_data["rms_eps"],
        test_data["mhc_pre_eps"],
        test_data["mhc_sinkhorn_eps"],
        test_data["mhc_post_mult_value"],
        test_data["sinkhorn_repeat"],
        test_data["n_splits"],
    )

    assert torch.allclose(post_mix_fused, post_mix_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(comb_mix_fused, comb_mix_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(layer_input_fused, layer_input_ref, rtol=1e-2, atol=2e-3)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    "n1,hidden_size,mhc_mult",
    [
        (n1, hidden_size, 4)
        for n1 in (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6912, 8192, 16384)
        for hidden_size in (4096, 7168)
    ],
)
def test_bench_mhc_pre_big_fuse(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    td = generate_big_fuse_test_data(n1, mhc_mult, hidden_size)
    fn = lambda: mhc_pre_big_fuse(
        td["residual"], td["fn"], td["mhc_scale"], td["mhc_base"],
        rms_eps=td["rms_eps"], mhc_pre_eps=td["mhc_pre_eps"],
        mhc_sinkhorn_eps=td["mhc_sinkhorn_eps"],
        mhc_post_mult_value=td["mhc_post_mult_value"],
        sinkhorn_repeat=td["sinkhorn_repeat"], n_splits=td["n_splits"],
    )
    fn()
    # Match the original AITER benchmark. This allocation-owning API launches
    # multiple kernels; eager host-side timing is dominated by Python dispatch,
    # output allocation, and launch gaps rather than GPU kernel execution.
    ms = float(do_bench_cudagraph(fn))
    print(
        f"[bench] mhc_pre_big_fuse (CUDA graph replay) tokens={n1} "
        f"H={hidden_size} MHC={mhc_mult}: {ms * 1000:.2f} us"
    )
