"""Performance cases for KDA prefill (fused-gate mainline).

Times the end-to-end prefill entry ``chunk_kda_with_fused_gate_fwd`` on
representative shapes (a fixed-token grid, one varlen case, one B=4 case)
using the shipped config.  The shape grid mirrors the ``SHAPE_PRESETS`` of
``scripts/tune/triton/tune_kda_prefill.py`` -- keep them in sync.

FLOPS (printed for reference) approximates the linear-attention matmul
work of the mainline (chunk size ``CHUNK_SIZE=64``):

    4 * H * (T / CHUNK_SIZE) * K * V   # inter-chunk state propagation
  + 2 * H * T * (K + V)                 # intra-chunk qk / pv

The varlen case builds ``cu_seqlens`` once and reuses it across the timed
iterations: ``prepare_chunk_indices`` syncs to the host but is cached by
tensor identity, so the sync stays in warmup.  Run with ``pytest --bench``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fla.kda import chunk_kda_with_fused_gate_fwd  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

DTYPE = torch.bfloat16
CHUNK_SIZE = 64


@dataclass(frozen=True)
class _PrefillShape:
    name: str
    B: int
    total: int
    H: int = 24
    K: int = 128
    V: int = 128
    varlen: bool = False


_BENCH_SHAPES = (
    _PrefillShape("t512", 1, 512),
    _PrefillShape("t2048", 1, 2048),
    _PrefillShape("t4096", 1, 4096),
    _PrefillShape("t8192", 1, 8192),
    _PrefillShape("t16384", 1, 16384),
    _PrefillShape("varlen_t8192", 1, 8192, varlen=True),
    _PrefillShape("batch4_t2048", 4, 2048),
)


def _make_inputs(
    B: int, total: int, H: int, K: int, V: int, varlen: bool
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(221)
    if varlen:
        split = max(CHUNK_SIZE, (total // 2 // CHUNK_SIZE) * CHUNK_SIZE)
        lengths = [length for length in (split, total - split) if length > 0]
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=DEVICE, dtype=torch.int32,
        )
        n_sequences = len(lengths)
    else:
        lengths, cu, n_sequences = [total // B] * B, None, B

    q = torch.randn(B, total, H, K, device=DEVICE, dtype=DTYPE) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(B, total, H, V, device=DEVICE, dtype=DTYPE) * 0.2
    raw_g = -torch.rand(B, total, H, K, device=DEVICE, dtype=DTYPE) * 0.2
    raw_beta = torch.rand(B, total, H, device=DEVICE, dtype=DTYPE)
    a_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.2
    initial = (
        torch.randn(n_sequences, H, V, K, device=DEVICE, dtype=torch.float32)
        * 0.05
    )
    return q, k, v, raw_g, raw_beta, a_log, initial, cu


def _prefill_flops(total: int, H: int, K: int, V: int) -> float:
    """Approximate linear-attention matmul work over the total tokens."""
    return 4.0 * H * (total / CHUNK_SIZE) * K * V + 2.0 * H * total * (K + V)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape", _BENCH_SHAPES, ids=lambda s: s.name)
def test_bench_kda_prefill(
    shape: _PrefillShape,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Time the fused-gate prefill mainline end to end on model sizes."""
    q, k, v, raw_g, raw_beta, a_log, initial, cu = _make_inputs(
        shape.B, shape.total, shape.H, shape.K, shape.V, shape.varlen
    )

    def run():
        return chunk_kda_with_fused_gate_fwd(
            q=q, k=k, v=v, raw_g=raw_g,
            raw_beta=raw_beta, A_log=a_log, g_bias=None,
            scale=shape.K ** -0.5, initial_state=initial,
            output_final_state=True, lower_bound=None,
            cu_seqlens=cu,
        )

    output, final_state = run()
    assert torch.isfinite(output).all()
    if final_state is not None:
        assert torch.isfinite(final_state).all()

    ms = time_ms(run, bench_warmup, bench_repeat)
    flops = _prefill_flops(shape.total, shape.H, shape.K, shape.V)
    print(
        f"[bench] kda_prefill {shape.name} B={shape.B} "
        f"T={shape.total // shape.B} varlen={shape.varlen}: "
        f"{ms:.4f} ms, {flops / (ms * 1e9):.3f} TFLOPS"
    )
