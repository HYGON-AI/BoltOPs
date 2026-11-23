# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and performance cases for KDA prefill.

The unbounded mainline grid times ``chunk_kda_with_fused_gate_fwd`` and mirrors
``SHAPE_PRESETS`` in ``scripts/tune/triton/tune_kda_prefill.py``. Bounded cases
use the public ``chunk_kda_with_fused_gate`` entry to exercise its optimized
dispatch. Correctness tests compare that optimized path with an independent
natural-log-gate recurrence.

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

from boltops.fla.kda import (  # noqa: E402
    chunk_kda_with_fused_gate,
    chunk_kda_with_fused_gate_fwd,
)
from boltops.fla.kda.triton.prefill_optimized import prefill_optimized_fwd  # noqa: E402

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
    lower_bound: float | None = None


_BENCH_SHAPES = (
    _PrefillShape("t512", 1, 512),
    _PrefillShape("t2048", 1, 2048),
    _PrefillShape("t4096", 1, 4096),
    _PrefillShape("t8192", 1, 8192),
    _PrefillShape("t16384", 1, 16384),
    _PrefillShape("varlen_t8192", 1, 8192, varlen=True),
    _PrefillShape("batch4_t2048", 4, 2048),
    _PrefillShape("optimized_t16_h8", 1, 16, H=8, lower_bound=-0.1),
    _PrefillShape("optimized_t256_h8", 1, 256, H=8, lower_bound=-5.0),
    _PrefillShape("optimized_t1024_h8", 1, 1024, H=8, lower_bound=-0.0001),
    _PrefillShape("optimized_t1024_h24", 1, 1024, H=24, lower_bound=-0.1),
    _PrefillShape("optimized_t1024_h64", 1, 1024, H=64, lower_bound=-0.1),
    _PrefillShape("bounded_t4096_h24", 1, 4096, H=24, lower_bound=-0.1),
    _PrefillShape("bounded_t16384_h24", 1, 16384, H=24, lower_bound=-0.1),
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
        if shape.lower_bound is not None:
            return chunk_kda_with_fused_gate(
                q=q, k=k, v=v, raw_g=raw_g,
                raw_beta=raw_beta, A_log=a_log, g_bias=None,
                initial_state=initial, scale=shape.K**-0.5,
                output_final_state=True, lower_bound=shape.lower_bound,
                use_qk_l2norm_in_kernel=True,
            )
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


@pytest.fixture(scope="module")
def _reference_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _make_bounded_inputs(tokens, heads, lower_bound, initial, seed):
    generator = torch.Generator().manual_seed(seed)

    def random(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    return dict(
        q=random(1, tokens, heads, 128),
        k=random(1, tokens, heads, 128),
        v=random(1, tokens, heads, 128),
        g=random(1, tokens, heads, 128),
        beta=random(1, tokens, heads),
        A_log=random(heads, dtype=torch.float32, scale=0.5) - 2,
        g_bias=random(heads, 128, dtype=torch.float32, scale=0.1),
        initial_state=(random(1, heads, 128, 128, dtype=torch.float32, scale=0.05)
                       if initial else None),
        lower_bound=lower_bound, scale=128**-0.5,
        output_final_state=True, use_qk_l2norm=True, apply_beta_sigmoid=True,
    )


def _natural_log_reference(inputs):
    """Independent CPU FP64 recurrence; fused KDA gates describe exp(g)."""
    q, k, v = (inputs[key].double() for key in ("q", "k", "v"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = inputs["lower_bound"] * torch.sigmoid(
        inputs["A_log"].double()[None, None, :, None].exp()
        * (inputs["g"].double() + inputs["g_bias"].double()[None, None])
    )
    beta = inputs["beta"].double().sigmoid()
    initial = inputs["initial_state"]
    state = (torch.zeros(1, q.shape[2], 128, 128, dtype=torch.float64)
             if initial is None else initial.double().clone())
    output = torch.empty_like(v)
    for token in range(q.shape[1]):
        state = state * gate[:, token, :, None, :].exp()
        key = k[:, token]
        prediction = (state * key[:, :, None, :]).sum(-1)
        residual = (v[:, token] - prediction) * beta[:, token, :, None]
        state = state + residual[..., None] * key[:, :, None, :]
        output[:, token] = (
            (state * q[:, token, :, None, :]).sum(-1) * inputs["scale"]
        )
    return output, state


def _assert_relative_rms(actual, expected):
    for name, actual_tensor, expected_tensor in zip(
        ("output", "final state"), actual, expected,
    ):
        assert torch.isfinite(actual_tensor).all(), name
        error = (actual_tensor.cpu().double() - expected_tensor).norm()
        relative_rms = float(error / expected_tensor.norm())
        assert relative_rms < 0.005, f"{name} relative RMS: {relative_rms:.7f}"


_ACCURACY_CASES = [
    pytest.param(16, 1, -0.1, False, 1, 42, id="first-chunk-normalization"),
    pytest.param(256, 4, -5.0, True, 1, 42, id="strong-decay-with-state"),
    pytest.param(256, 4, -0.1, False, 1, 42, id="weak-decay-no-state"),
    pytest.param(1024, 4, -0.01, True, 1, 42, id="weak-decay-normalization"),
    pytest.param(1024, 4, -0.0001, True, 1, 42, id="near-unit-decay-seed42"),
    pytest.param(1024, 4, -0.0001, True, 1, 231, id="near-unit-decay-seed231"),
    pytest.param(64, 4, -0.1, True, 2, 42, id="segmented-recurrence"),
    pytest.param(4096, 4, -0.1, True, 1, 42, id="long-recurrence"),
    pytest.param(2048, 4, -0.0001, True, None, 42, id="auto-segmented-near-unit-decay"),
    pytest.param(16384, 4, -0.1, True, None, 42, id="max-window-auto-segmented"),
]


@GPU_REQUIRED
def test_repeated_key_satisfies_delta_rule():
    # With k=q=e_0, beta=1/2, v=1 and no decay, each value channel follows
    # s_t = s_(t-1)/2 + 1/2, hence o_t = 1 - 2^(-(t+1)).
    q = torch.zeros(1, 16, 1, 128, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    v = torch.ones_like(q)
    actual, state = prefill_optimized_fwd(
        q, q.clone(), v, torch.zeros_like(q),
        torch.full((1, 16, 1), 0.5, dtype=q.dtype, device=q.device),
        scale=1.0, output_final_state=True,
    )
    expected = (1 - 2.0 ** -torch.arange(1, 17, dtype=torch.float64))
    expected = expected.reshape(1, 16, 1, 1).expand_as(actual).to(q.dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    expected_state = torch.zeros(1, 1, 128, 128)
    expected_state[..., 0] = 1 - 2.0 ** -16
    torch.testing.assert_close(state.cpu(), expected_state, atol=0.004, rtol=0)


@GPU_REQUIRED
def test_residual_subtraction_preserves_cancellation():
    # k.S = 1 + 1/256 is halfway between BF16 values. Rounding that
    # prediction to BF16 before subtracting v=1 incorrectly gives zero.
    q = torch.zeros(1, 16, 1, 128, device=DEVICE, dtype=DTYPE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[:, 0, :, 2] = 1
    k[:, 0, :, 0] = 1
    k[:, 0, :, 1] = 0.5
    k[:, 0, :, 2] = 1
    v[:, 0] = 1
    beta = torch.zeros(1, 16, 1, device=DEVICE, dtype=DTYPE)
    beta[:, 0] = 0.5
    initial = torch.zeros(1, 1, 128, 128, device=DEVICE, dtype=torch.float32)
    initial[..., 0] = 1
    initial[..., 1] = 1 / 128
    actual, _ = prefill_optimized_fwd(
        q, k, v, torch.zeros_like(q), beta, scale=1.0,
        initial_state=initial,
    )
    expected = torch.zeros_like(actual)
    expected[:, 0] = -1 / 512
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@GPU_REQUIRED
@pytest.mark.usefixtures("_reference_threads")
@pytest.mark.parametrize("tokens,heads,lower_bound,initial,segments,seed", _ACCURACY_CASES)
def test_bounded_prefill_relative_rms(tokens, heads, lower_bound, initial, segments, seed):
    inputs = _make_bounded_inputs(tokens, heads, lower_bound, initial, seed)
    expected = _natural_log_reference(inputs)
    device_inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                     for key, value in inputs.items()}
    actual = prefill_optimized_fwd(**device_inputs, segments=segments)
    _assert_relative_rms(actual, expected)


@GPU_REQUIRED
@pytest.mark.usefixtures("_reference_threads")
def test_public_bounded_prefill_dispatch_matches_reference(monkeypatch):
    from boltops.fla.kda.triton import chunk as chunk_module

    inputs = _make_bounded_inputs(1024, 4, -0.0001, True, 42)
    expected = _natural_log_reference(inputs)
    device_inputs = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                     for key, value in inputs.items()}
    calls = []

    def optimized(*args, **kwargs):
        calls.append(True)
        return prefill_optimized_fwd(*args, **kwargs)

    monkeypatch.setattr(chunk_module, "prefill_optimized_fwd", optimized)
    actual = chunk_module.chunk_kda_with_fused_gate(
        device_inputs["q"], device_inputs["k"], device_inputs["v"],
        device_inputs["g"], device_inputs["beta"], device_inputs["A_log"],
        device_inputs["g_bias"], scale=inputs["scale"],
        initial_state=device_inputs["initial_state"], output_final_state=True,
        lower_bound=inputs["lower_bound"], use_qk_l2norm_in_kernel=True,
    )
    assert calls == [True], "public prefill must exercise the optimized kernel"
    _assert_relative_rms(actual, expected)
