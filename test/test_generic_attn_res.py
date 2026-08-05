"""Correctness and optional performance tests for ``attn_res``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from bolt_ops.generic import attn_res  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _reference(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "capacity", "hidden_size", "row_padding"),
    [
        pytest.param(0, 3, 5, 128, 0, id="empty"),
        pytest.param(1, 1, 2, 128, 0, id="decode-single"),
        pytest.param(17, 4, 6, 1024, 7, id="decode-padded"),
        pytest.param(320, 8, 10, 7168, 0, id="prefill-full"),
    ],
)
def test_attn_res_matches_torch(
    num_tokens: int,
    num_blocks: int,
    capacity: int,
    hidden_size: int,
    row_padding: int,
) -> None:
    torch.manual_seed(42)
    eps = 1e-5
    prefix = _randn_with_row_padding(
        num_tokens, hidden_size, padding=row_padding
    )
    blocks = _randn_with_row_padding(
        num_tokens, capacity, hidden_size, padding=row_padding
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device=DEVICE, dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    expected = _reference(
        prefix, blocks, norm_weight, qk_weight, num_blocks, eps
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    actual = attn_res(
        prefix, blocks, norm_weight, qk_weight, num_blocks, eps
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, original_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()


@dataclass(frozen=True)
class _BenchShape:
    tokens: int
    blocks: int
    hidden: int


_BENCH_SHAPES = {
    "decode": _BenchShape(tokens=1, blocks=8, hidden=7168),
    "small-batch": _BenchShape(tokens=17, blocks=8, hidden=7168),
    "prefill": _BenchShape(tokens=320, blocks=8, hidden=7168),
}


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape_name", list(_BENCH_SHAPES))
def test_bench_attn_res(
    shape_name: str,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    shape = _BENCH_SHAPES[shape_name]
    prefix = torch.randn(
        shape.tokens, shape.hidden, device=DEVICE, dtype=torch.bfloat16
    )
    blocks = torch.randn(
        shape.tokens,
        shape.blocks,
        shape.hidden,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    norm_weight = torch.ones(
        shape.hidden, device=DEVICE, dtype=torch.bfloat16
    )
    qk_weight = torch.randn_like(norm_weight) / shape.hidden**0.5

    elapsed = time_ms(
        lambda: attn_res(
            prefix,
            blocks,
            norm_weight,
            qk_weight,
            shape.blocks,
            1e-5,
        ),
        bench_warmup,
        bench_repeat,
    )
    print(
        f"attn_res {shape_name}: tokens={shape.tokens}, "
        f"blocks={shape.blocks}, hidden={shape.hidden}, {elapsed:.4f} ms"
    )
