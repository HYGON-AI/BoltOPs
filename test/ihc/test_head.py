# ruff: noqa
import pytest
import torch

pytest.importorskip("tilelang")

from boltops.ihc import ihc_head
from .ref import ihc_head_ref

GPU_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU device is required")


@GPU_REQUIRED
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("hc", [4])
def test_ihc_head_correctness(num_tokens: int, hidden: int, hc: int) -> None:
    torch.manual_seed(0)
    residual = torch.randn((num_tokens, hc, hidden), dtype=torch.bfloat16, device="cuda")
    head_fn = torch.randn((hc, hc * hidden), dtype=torch.float32, device="cuda") * 1e-4
    head_scale = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    head_base = torch.zeros((hc,), dtype=torch.float32, device="cuda")
    out = ihc_head(residual, head_fn, head_scale, head_base, 1e-5, 1e-6)
    ref = ihc_head_ref(residual, head_fn, head_scale, head_base, 1e-5, 1e-6)
    # Same bf16 reduce budget as MHC layer_input / ihc_pre layer_in.
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-2)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("num_tokens,hidden", [(32, 4096), (128, 7168)])
def test_bench_ihc_head(num_tokens: int, hidden: int, time_ms, bench_warmup, bench_repeat) -> None:
    hc = 4
    residual = torch.randn((num_tokens, hc, hidden), dtype=torch.bfloat16, device="cuda")
    head_fn = torch.randn((hc, hc * hidden), dtype=torch.float32, device="cuda") * 1e-4
    head_scale = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    head_base = torch.zeros((hc,), dtype=torch.float32, device="cuda")
    fn = lambda: ihc_head(residual, head_fn, head_scale, head_base, 1e-5, 1e-6)
    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    print(f"[bench] ihc_head tokens={num_tokens} H={hidden}: {ms:.4f} ms")
