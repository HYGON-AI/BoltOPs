# ruff: noqa
import pytest
import torch

pytest.importorskip("tilelang")

from boltops.ihc import ihc_pre
from .ref import ihc_pre_ref

GPU_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU device is required")


def _inputs(num_tokens: int, hidden: int, hc: int):
    residual = torch.randn((num_tokens, hc, hidden), dtype=torch.bfloat16, device="cuda")
    fn = torch.randn((2 * hc, hc * hidden), dtype=torch.float32, device="cuda") * 1e-4
    scale = torch.tensor([0.01, 0.01], dtype=torch.float32, device="cuda")
    base = torch.zeros((2 * hc,), dtype=torch.float32, device="cuda")
    return residual, fn, scale, base


@GPU_REQUIRED
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("hc", [4])
def test_ihc_pre_correctness(num_tokens: int, hidden: int, hc: int) -> None:
    torch.manual_seed(0)
    residual, fn, scale, base = _inputs(num_tokens, hidden, hc)
    layer_in, post = ihc_pre(residual, fn, scale, base, 1e-5, 1e-6, 2.0)
    layer_ref, post_ref = ihc_pre_ref(residual, fn, scale, base, 1e-5, 1e-6, 2.0)
    # Match MHC pre_big_fuse tolerances: mix gates strict; bf16 reduce a bit looser.
    torch.testing.assert_close(post, post_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(layer_in, layer_ref, atol=2e-3, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("num_tokens,hidden", [(32, 4096), (128, 7168)])
def test_bench_ihc_pre(num_tokens: int, hidden: int, time_ms, bench_warmup, bench_repeat) -> None:
    residual, fn, scale, base = _inputs(num_tokens, hidden, 4)
    fn_call = lambda: ihc_pre(residual, fn, scale, base, 1e-5, 1e-6, 2.0)
    fn_call()
    ms = time_ms(fn_call, bench_warmup, bench_repeat)
    print(f"[bench] ihc_pre tokens={num_tokens} H={hidden}: {ms:.4f} ms")
