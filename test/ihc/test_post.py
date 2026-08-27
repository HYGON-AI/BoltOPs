# ruff: noqa
import pytest
import torch

pytest.importorskip("tilelang")

from boltops.ihc import ihc_post
from .ref import ihc_post_ref

GPU_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU device is required")


@GPU_REQUIRED
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("hc", [4])
def test_ihc_post_correctness(num_tokens: int, hidden: int, hc: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((num_tokens, hc, hidden), dtype=torch.bfloat16, device="cuda")
    post = torch.randn((num_tokens, hc), dtype=torch.float32, device="cuda")
    out = ihc_post(x, residual, post)
    ref = ihc_post_ref(x, residual, post)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("num_tokens,hidden", [(32, 4096), (128, 7168)])
def test_bench_ihc_post(num_tokens: int, hidden: int, time_ms, bench_warmup, bench_repeat) -> None:
    hc = 4
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((num_tokens, hc, hidden), dtype=torch.bfloat16, device="cuda")
    post = torch.randn((num_tokens, hc), dtype=torch.float32, device="cuda")
    fn = lambda: ihc_post(x, residual, post)
    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    print(f"[bench] ihc_post tokens={num_tokens} H={hidden}: {ms:.4f} ms")
