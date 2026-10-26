"""Correctness and optional performance tests for ``hc_combine_norm``.

Default ``pytest``: functional / edge-case correctness against an eager fp32
reference (the bf16 rounding points match the kernels: ``out`` is rounded to
bf16 before the norm).
``pytest --bench``: qwen3.8-flash token-count timing, baseline vs ship entry.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.generic import hc_combine_norm  # noqa: E402
from boltops.generic.triton.hc_combine_norm import (  # noqa: E402
    _MIN_ROW_TOKENS,
    _hc_combine_norm,
    _hc_combine_norm_row_split,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

# qwen3.8-flash: hidden_size=2560, hc_count=4 -> DIM = HC * HCDIM = 10240.
EPS = 1e-6
HC = 4


def _make_inputs(num_tokens: int, hidden: int, seed: int = 0):
    torch.manual_seed(seed)
    dim = HC * hidden
    res = torch.randn(num_tokens, dim, device=DEVICE, dtype=torch.bfloat16)
    block = torch.randn(num_tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    inj = torch.randn(num_tokens, HC, device=DEVICE, dtype=torch.bfloat16)
    return res, block, inj


def _reference(res, block, inj, w, eps):
    """Eager fp32 reference; rounds ``out`` to bf16 like the kernels do."""
    num_tokens, dim = res.shape
    hidden = dim // HC
    gate = 2.0 * torch.sigmoid(inj.float() / HC)
    of0 = res.float().view(-1, HC, hidden) + block.float().view(-1, 1, hidden) * gate.unsqueeze(-1)
    out_b = of0.to(torch.bfloat16)
    of = out_b.float().view(-1, HC, hidden)
    rrms = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    yf = of * rrms
    # Shared weights broadcast across streams; per-branch weights use the
    # same flattened HC layout as one residual row.
    w_shape = (1, 1, hidden) if w.numel() == hidden else (1, HC, hidden)
    yf = yf + yf * w.float().view(w_shape)
    return out_b.view(-1, dim), yf.to(torch.bfloat16).view(-1, dim)


def _assert_close_to_reference(actual, expected):
    # Reduction-order noise vs the fp32 reference scales with the output
    # magnitude: allow atol 0.05 plus 2% of the max |value| (covers a few
    # bf16 ulp). A real mismatch (wrong gate / wrong stream) produces O(1)
    # diffs, far above this.
    got = torch.cat([a.float().flatten() for a in actual])
    want = torch.cat([a.float().flatten() for a in expected])
    scale = want.abs().max().item()
    assert (got - want).abs().max().item() <= 0.05 + 0.02 * scale


@GPU_REQUIRED
@pytest.mark.parametrize(
    "num_tokens", [1, 4, 16, 64, 148, 255, 256, 512, 1024],
)
@pytest.mark.parametrize("weight_mode", ["per-branch", "shared"])
def test_matches_reference(num_tokens, weight_mode):
    """Both sides of the dispatch cut-over, production hidden=2560."""
    torch.manual_seed(42)
    res, block, inj = _make_inputs(num_tokens, hidden=2560)
    hidden = 2560
    dim = HC * hidden
    w = torch.randn(
        hidden if weight_mode == "shared" else dim,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    expected = _reference(res, block, inj, w, EPS)
    actual = hc_combine_norm(res, block, inj, w, EPS, HC)
    _assert_close_to_reference(actual, expected)


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("hidden", "expected_impl"),
    [
        pytest.param(512, "row_split", id="power-of-two"),
        pytest.param(1280, "row_split", id="split-1024-256"),
        pytest.param(3000, "baseline", id="not-splittable"),
        pytest.param(4352, "baseline", id="too-large"),
    ],
)
def test_other_hidden_sizes(hidden, expected_impl):
    """v4 handles any 2^k+2^m row; other rows fall back to the baseline."""
    num_tokens = _MIN_ROW_TOKENS
    res, block, inj = _make_inputs(num_tokens, hidden)
    w = torch.randn(HC * hidden, device=DEVICE, dtype=torch.bfloat16)
    expected = _reference(res, block, inj, w, EPS)
    actual = hc_combine_norm(res, block, inj, w, EPS, HC)
    _assert_close_to_reference(actual, expected)
    # Same input -> same kernel -> bitwise equal to the direct call.
    direct = (
        _hc_combine_norm_row_split if expected_impl == "row_split" else _hc_combine_norm
    )(res, block, inj, w, EPS, HC)
    for got, want in zip(actual, direct):
        assert torch.equal(got, want)


@GPU_REQUIRED
def test_fallback_is_bitwise_baseline():
    """Small batches and strided residuals take the baseline kernel verbatim."""
    res, block, inj = _make_inputs(_MIN_ROW_TOKENS - 1, hidden=2560)
    w = torch.randn(HC * 2560, device=DEVICE, dtype=torch.bfloat16)
    padded = res.new_empty(res.shape[0], res.shape[1] + 7)
    strided = padded[:, : res.shape[1]]
    strided.copy_(res)
    for candidate in (res, strided):
        got = hc_combine_norm(candidate, block, inj, w, EPS, HC)
        want = _hc_combine_norm(candidate, block, inj, w, EPS, HC)
        for a, b in zip(got, want):
            assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Performance (qwen3.8-flash token counts: decode draft / ngram-MTP verify /
# small decode / chunked prefill / full prefill). Different sizes than the
# functional cases above. Skipped unless ``pytest --bench``.
# ---------------------------------------------------------------------------

# bytes/token: read res (DIM) + block (HCDIM), write out + y (DIM each), bf16.
HIDDEN = 2560
DIM = HC * HIDDEN
BYTES_PER_TOKEN = (3 * DIM + HIDDEN) * 2

_BENCH_TOKENS = (1, 4, 16, 64, 148, 256, 512, 1024, 2048, 4096, 8192, 16384)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("num_tokens", _BENCH_TOKENS)
def test_bench_hc_combine_norm(num_tokens, time_ms, bench_warmup, bench_repeat):
    """Time the ship entry against the baseline kernel, both in bf16."""
    torch.manual_seed(0)
    res, block, inj = _make_inputs(num_tokens, HIDDEN)
    w = torch.randn(DIM, device=DEVICE, dtype=torch.bfloat16)
    args = (res, block, inj, w, EPS, HC)

    base_ms = time_ms(lambda: _hc_combine_norm(*args), bench_warmup, bench_repeat)
    ship_ms = time_ms(lambda: hc_combine_norm(*args), bench_warmup, bench_repeat)
    gbps = num_tokens * BYTES_PER_TOKEN / (ship_ms * 1e6) if ship_ms else float("inf")
    print(
        f"[bench] hc_combine_norm tokens={num_tokens}: "
        f"baseline {base_ms:.4f} ms, ship {ship_ms:.4f} ms "
        f"({base_ms / ship_ms:.2f}x), {gbps:.0f} GB/s"
    )
