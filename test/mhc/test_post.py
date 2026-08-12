# ruff: noqa
import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mhc import mhc_post_fwd


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.einsum("tmn,tmc->tnc", comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix.unsqueeze(-1) + term2).bfloat16()



def generate_mhc_post_test_data(
    num_tokens: int,
    h: int,
    mhc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    x = torch.randn((num_tokens, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((num_tokens, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((num_tokens, mhc_mult), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((num_tokens, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_tokens",
    # [1, 16, 32, 4096],
    [1, 32],
)
@pytest.mark.parametrize(
    "h",
    # [1280, 2560, 4096, 7168],
    [4096, 7168],
)
def test_mhc_post_fwd_correctness(num_tokens: int, h: int) -> None:
    td = generate_mhc_post_test_data(num_tokens=num_tokens, h=h, mhc_mult=4)

    out_tl = mhc_post_fwd(
        td["x"].contiguous(),
        td["residual"].contiguous(),
        td["post_layer_mix"].contiguous(),
        td["comb_res_mix"].contiguous(),
    )
    out_ref = mhc_post_ref(
        td["x"],
        td["residual"],
        td["post_layer_mix"],
        td["comb_res_mix"],
    )

    torch.testing.assert_close(out_tl, out_ref, atol=1e-2, rtol=1e-2)


def _estimate_io_bytes(num_tokens: int, h: int, mhc_mult: int) -> int:
    read_bytes = num_tokens * (
        h * 2 + mhc_mult * h * 2 + mhc_mult * 4 + mhc_mult * mhc_mult * 4
    )
    return read_bytes + num_tokens * mhc_mult * h * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.bench
@pytest.mark.parametrize(
    "num_tokens,h,mhc_mult",
    [
        *[(n, 7168, 4) for n in (1, 32, 64, 128, 256, 512, 1024, 2048, 8192, 65536)],
        *[(n, 4096, 4) for n in (1, 32, 64, 128, 256, 512, 1024, 2048, 8192, 65536)],
    ],
)
def test_bench_mhc_post_fwd(
    num_tokens: int,
    h: int,
    mhc_mult: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    td = generate_mhc_post_test_data(num_tokens, h, mhc_mult)
    fn = lambda: mhc_post_fwd(
        td["x"], td["residual"], td["post_layer_mix"], td["comb_res_mix"]
    )
    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    bandwidth = _estimate_io_bytes(num_tokens, h, mhc_mult) / (ms * 1e6)
    print(
        f"[bench] mhc_post_fwd tokens={num_tokens} H={h} MHC={mhc_mult}: "
        f"{ms:.4f} ms, {bandwidth:.2f} GB/s"
    )
