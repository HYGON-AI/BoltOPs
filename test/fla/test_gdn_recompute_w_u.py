# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for ``recompute_w_u_fwd``.

Default ``pytest``: functional / edge-case correctness.
``pytest --bench``: real-model size timing (different shape set).
"""

from __future__ import annotations

from dataclasses import dataclass

import importlib
import json
from functools import partial

import pytest
import torch

pytest.importorskip("triton")
import triton  # noqa: E402

from boltops.fla.gdn.triton import recompute_w_u_fwd  # noqa: E402
from boltops.fla.gdn.triton.recompute_w_u import (  # noqa: E402
    _GDN_RECOMPUTE_W_U_CONFIGS as _RECOMPUTE_W_U_CONFIGS,
    _select_recompute_w_u_fwd_gdn_config as _select_recompute_w_u_config,
    prepare_chunk_indices,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _compute_chunk_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one sequence chunk using FP32 accumulation."""
    key_heads = k.shape[2]
    value_heads = v.shape[2]
    k_grouped = k.repeat_interleave(value_heads // key_heads, dim=2)

    A_f = A.permute(0, 2, 1, 3).float()
    beta_f = beta.float().unsqueeze(-1)
    # Match the kernel's operand rounding before tl.dot.
    v_beta = (v.float() * beta_f).to(v.dtype).float().permute(0, 2, 1, 3)
    k_beta_gate = (
        k_grouped.float() * beta_f * torch.exp(g_cumsum.float()).unsqueeze(-1)
    ).to(k.dtype).float().permute(0, 2, 1, 3)

    u = torch.matmul(A_f, v_beta).permute(0, 2, 1, 3).to(v.dtype)
    w = torch.matmul(A_f, k_beta_gate).permute(0, 2, 1, 3).to(k.dtype)
    return w, u


def recompute_w_u_fwd_torch(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for ``recompute_w_u_fwd``."""
    del chunk_indices
    B, T, key_heads, K = k.shape
    value_heads, V = v.shape[2:]
    BT = A.shape[-1]
    if value_heads % key_heads != 0:
        raise ValueError("v heads must be divisible by k heads")

    w = torch.empty((B, T, value_heads, K), dtype=k.dtype, device=k.device)
    u = torch.empty((B, T, value_heads, V), dtype=v.dtype, device=v.device)

    if cu_seqlens is None:
        sequences = [(b, 0, T) for b in range(B)]
    else:
        if B != 1:
            raise ValueError("variable-length layout requires k/v batch size B=1")
        offsets = cu_seqlens.tolist()
        sequences = [(0, offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]

    for batch, bos, eos in sequences:
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            length = end - start
            w_chunk, u_chunk = _compute_chunk_ref(
                k[batch : batch + 1, start:end],
                v[batch : batch + 1, start:end],
                beta[batch : batch + 1, start:end],
                g_cumsum[batch : batch + 1, start:end],
                A[batch : batch + 1, start:end, :, :length],
            )
            w[batch : batch + 1, start:end] = w_chunk
            u[batch : batch + 1, start:end] = u_chunk
    return w, u


def _assert_fla_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Use the upstream FLA RMS-relative error criterion."""
    assert not torch.isnan(actual).any()
    assert not torch.isnan(expected).any()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt()
    assert error / (baseline + 1e-8) < 5e-3


def _sequence_ranges(B: int, T: int, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return [(b, 0, T) for b in range(B)]
    offsets = cu_seqlens.cpu().tolist()
    return [(0, offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]


def _make_inputs(
    B: int,
    T: int,
    Hg: int,
    H: int,
    K: int,
    V: int,
    BT: int,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor | None = None,
    beta_dtype: torch.dtype | None = None,
):
    torch.manual_seed(42)
    # Build fixtures on CPU and transfer once. This avoids timing or depending
    # on first-use HIP kernels for random/tril/eye in the correctness harness.
    k = torch.randn(B, T, Hg, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    beta = torch.sigmoid(
        torch.randn(B, T, H, dtype=beta_dtype if beta_dtype is not None else dtype)
    )
    A = torch.zeros(B, T, H, BT, dtype=dtype)
    g = torch.empty(B, T, H, dtype=torch.float32)

    for batch, bos, eos in _sequence_ranges(B, T, cu_seqlens):
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            length = end - start
            lower = torch.randn(H, length, length) * 0.05
            lower = torch.tril(lower) + torch.eye(length).unsqueeze(0)
            A[batch, start:end, :, :length] = lower.permute(1, 0, 2).to(dtype)
            increments = -torch.rand(length, H) * 0.05
            g[batch, start:end] = increments.cumsum(0)
    return tuple(tensor.to(DEVICE) for tensor in (k, v, beta, g, A))


def _captured_cu_seqlens(
    T: int, num_sequences: int, NT: int, BT: int = 64
) -> torch.Tensor:
    """Synthesize sequence lengths matching captured T/sequence-count/NT metadata."""
    remainder_sum = T - BT * (NT - num_sequences)
    assert num_sequences <= remainder_sum <= num_sequences * BT
    chunks = [NT // num_sequences] * num_sequences
    for i in range(NT % num_sequences):
        chunks[i] += 1
    remainders = [remainder_sum // num_sequences] * num_sequences
    for i in range(remainder_sum % num_sequences):
        remainders[i] += 1
    lengths = [BT * (chunks[i] - 1) + remainders[i] for i in range(num_sequences)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device=DEVICE, dtype=torch.int32)


@pytest.mark.parametrize(
    ("B", "T", "Hg", "H", "K", "V", "BT", "dtype"),
    [
        pytest.param(2, 128, 2, 2, 64, 64, 64, torch.bfloat16, id="fla-bf16"),
        pytest.param(2, 128, 2, 4, 64, 64, 64, torch.bfloat16, id="fla-gqa"),
        pytest.param(1, 256, 4, 4, 32, 32, 64, torch.float16, id="fla-fp16"),
        pytest.param(2, 95, 1, 4, 48, 80, 32, torch.bfloat16, id="partial-chunk"),
        pytest.param(1, 65, 2, 4, 256, 256, 64, torch.bfloat16, id="d256"),
    ],
)
@GPU_REQUIRED
def test_recompute_w_u_matches_torch(B, T, Hg, H, K, V, BT, dtype):
    inputs = _make_inputs(B, T, Hg, H, K, V, BT, dtype)
    expected_w, expected_u = recompute_w_u_fwd_torch(*inputs)
    actual_w, actual_u = recompute_w_u_fwd(*inputs, cu_seqlens=None)
    _assert_fla_close(actual_w, expected_w)
    _assert_fla_close(actual_u, expected_u)


@GPU_REQUIRED
@pytest.mark.parametrize("pass_chunk_indices", [False, True])
def test_recompute_w_u_varlen_matches_torch(pass_chunk_indices):
    lengths = [17, 64, 79, 129]
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=DEVICE,
        dtype=torch.int64,
    )
    T = sum(lengths)
    inputs = _make_inputs(1, T, 2, 4, 64, 48, 64, torch.bfloat16, cu_seqlens)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 64) if pass_chunk_indices else None
    )
    expected_w, expected_u = recompute_w_u_fwd_torch(
        *inputs, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    actual_w, actual_u = recompute_w_u_fwd(
        *inputs, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    _assert_fla_close(actual_w, expected_w)
    _assert_fla_close(actual_u, expected_u)


REAL_NETWORK_SHAPES = [
    pytest.param(60, 1, 1, id="observed-short-t60-nt1-n1"),
    pytest.param(4156, 1, 65, id="observed-t4156-nt65-n1"),
    pytest.param(4543, 1, 71, id="observed-hot-t4543-nt71-n1"),
    pytest.param(8652, 2, 136, id="observed-t8652-nt136-n2"),
    pytest.param(12617, 3, 199, id="observed-t12617-nt199-n3"),
    pytest.param(13674, 3, 216, id="observed-max-t13674-nt216-n3"),
]


@GPU_REQUIRED
@pytest.mark.parametrize(("T", "num_sequences", "NT"), REAL_NETWORK_SHAPES)
def test_real_network_shapes(T, num_sequences, NT):
    """Cover representative metadata captured in /zhenxin/temp/unique_shapes.txt."""
    cu_seqlens = _captured_cu_seqlens(T, num_sequences, NT)
    inputs = _make_inputs(
        1,
        T,
        8,
        16,
        128,
        128,
        64,
        torch.bfloat16,
        cu_seqlens,
        beta_dtype=torch.float32,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    assert chunk_indices.shape == (NT, 2)
    expected_w, expected_u = recompute_w_u_fwd_torch(
        *inputs, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    actual_w, actual_u = recompute_w_u_fwd(
        *inputs, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    _assert_fla_close(actual_w, expected_w)
    _assert_fla_close(actual_u, expected_u)


def test_torch_reference_known_values_cpu():
    k = torch.tensor([[[[2.0]], [[3.0]]]])
    v = torch.tensor([[[[5.0]], [[7.0]]]])
    beta = torch.tensor([[[1.0], [0.5]]])
    g = torch.zeros(1, 2, 1)
    A = torch.tensor([[[[1.0, 0.0]], [[2.0, 1.0]]]])
    w, u = recompute_w_u_fwd_torch(k, v, beta, g, A)
    torch.testing.assert_close(w.flatten(), torch.tensor([2.0, 5.5]))
    torch.testing.assert_close(u.flatten(), torch.tensor([5.0, 13.5]))


def test_torch_reference_rejects_invalid_head_grouping():
    with pytest.raises(ValueError, match="v heads must be divisible"):
        recompute_w_u_fwd_torch(
            torch.empty(1, 2, 2, 4),
            torch.empty(1, 2, 3, 4),
            torch.empty(1, 2, 3),
            torch.empty(1, 2, 3),
            torch.empty(1, 2, 3, 2),
        )


def test_fixed_config_selection():
    assert len(_RECOMPUTE_W_U_CONFIGS) == 3
    assert _select_recompute_w_u_config(32, 32)["BK"] == 64
    assert _select_recompute_w_u_config(64, 64)["BV"] == 64
    assert _select_recompute_w_u_config(48, 80)["BK"] == 128
    assert _select_recompute_w_u_config(128, 128)["BV"] == 128
    assert _select_recompute_w_u_config(192, 64)["BK"] == 128
    assert _select_recompute_w_u_config(256, 256)["BV"] == 256


# ---------------------------------------------------------------------------
# Performance (real-model / observed sizes). Different shape set from the
# functional tests above. Skipped unless ``pytest --bench``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchShape:
    name: str
    B: int
    T: int
    Hg: int
    H: int
    K: int
    V: int
    BT: int = 64
    dtype: torch.dtype = torch.bfloat16
    lengths: tuple[int, ...] | None = None


def _bench_captured_lengths(
    T: int, num_sequences: int, NT: int, BT: int = 64
) -> tuple[int, ...]:
    remainder_sum = T - BT * (NT - num_sequences)
    if not num_sequences <= remainder_sum <= num_sequences * BT:
        raise ValueError("captured T/NT/sequence count are inconsistent")
    chunks = [NT // num_sequences] * num_sequences
    for i in range(NT % num_sequences):
        chunks[i] += 1
    remainders = [remainder_sum // num_sequences] * num_sequences
    for i in range(remainder_sum % num_sequences):
        remainders[i] += 1
    return tuple(
        BT * (chunks[i] - 1) + remainders[i] for i in range(num_sequences)
    )


_BENCH_SHAPES = {
    "observed-short": _BenchShape(
        "observed-short", 1, 60, 8, 16, 128, 128, lengths=(60,)
    ),
    "observed-common": _BenchShape(
        "observed-common", 1, 4156, 8, 16, 128, 128, lengths=(4156,)
    ),
    "observed-hot": _BenchShape(
        "observed-hot", 1, 4543, 8, 16, 128, 128, lengths=(4543,)
    ),
    "observed-common-n2": _BenchShape(
        "observed-common-n2",
        1,
        8652,
        8,
        16,
        128,
        128,
        lengths=_bench_captured_lengths(8652, 2, 136),
    ),
    "observed-hot-n3": _BenchShape(
        "observed-hot-n3",
        1,
        12617,
        8,
        16,
        128,
        128,
        lengths=_bench_captured_lengths(12617, 3, 199),
    ),
    "observed-max": _BenchShape(
        "observed-max",
        1,
        13674,
        8,
        16,
        128,
        128,
        lengths=_bench_captured_lengths(13674, 3, 216),
    ),
    "derived-d64-hot": _BenchShape(
        "derived-d64-hot", 1, 4543, 8, 16, 64, 64, lengths=(4543,)
    ),
    "derived-d256-hot": _BenchShape(
        "derived-d256-hot", 1, 4543, 8, 16, 256, 256, lengths=(4543,)
    ),
}


def _bench_sequence_ranges(case: _BenchShape):
    if case.lengths is None:
        return [(b, 0, case.T) for b in range(case.B)]
    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    return [(0, offsets[i], offsets[i + 1]) for i in range(len(case.lengths))]


def _bench_make_inputs(case: _BenchShape):
    torch.manual_seed(42)
    network_case = case.name.startswith(("observed-", "derived-"))
    k = torch.randn(case.B, case.T, case.Hg, case.K, dtype=case.dtype)
    v = torch.randn(case.B, case.T, case.H, case.V, dtype=case.dtype)
    beta_dtype = torch.float32 if network_case else case.dtype
    beta = torch.sigmoid(torch.randn(case.B, case.T, case.H, dtype=beta_dtype))
    g = torch.empty(case.B, case.T, case.H, dtype=torch.float32)
    A = torch.zeros(case.B, case.T, case.H, case.BT, dtype=case.dtype)
    for batch, bos, eos in _bench_sequence_ranges(case):
        for start in range(bos, eos, case.BT):
            end = min(start + case.BT, eos)
            length = end - start
            lower = torch.randn(case.H, length, length) * 0.05
            lower = torch.tril(lower) + torch.eye(length).unsqueeze(0)
            A[batch, start:end, :, :length] = lower.permute(1, 0, 2).to(case.dtype)
            g[batch, start:end] = (-torch.rand(length, case.H) * 0.05).cumsum(0)

    k, v, beta, g, A = (tensor.to(DEVICE) for tensor in (k, v, beta, g, A))
    cu_seqlens = None
    if case.lengths is not None:
        cu_dtype = torch.int32 if network_case else torch.int64
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(case.lengths).cumsum(0).tolist()],
            device=DEVICE,
            dtype=cu_dtype,
        )
    return k, v, beta, g, A, cu_seqlens


def _bench_useful_flops(case: _BenchShape) -> int:
    total = 0
    for _, bos, eos in _bench_sequence_ranges(case):
        for start in range(bos, eos, case.BT):
            length = min(case.BT, eos - start)
            total += 2 * case.H * length * length * (case.K + case.V)
    return total


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape_name", list(_BENCH_SHAPES))
def test_bench_recompute_w_u(shape_name, bench_warmup, bench_repeat, time_ms):
    """Time ship API on observed / derived network sizes."""
    case = _BENCH_SHAPES[shape_name]
    k, v, beta, g, A, cu_seqlens = _bench_make_inputs(case)
    triton_fn = lambda: recompute_w_u_fwd(k, v, beta, g, A, cu_seqlens)
    expected_w, expected_u = recompute_w_u_fwd_torch(k, v, beta, g, A, cu_seqlens)
    actual_w, actual_u = triton_fn()
    _assert_fla_close(actual_w, expected_w)
    _assert_fla_close(actual_u, expected_u)

    ms = time_ms(triton_fn, bench_warmup, bench_repeat)
    tflops = _bench_useful_flops(case) / (ms * 1e9)
    config = _select_recompute_w_u_config(case.K, case.V)
    print(
        f"[bench] recompute_w_u {shape_name}: "
        f"{ms:.4f} ms, {tflops:.3f} useful TFLOPS, "
        f"B{config['BK']}/W{config['num_warps']}/S{config['num_stages']}"
    )


_BENCH_1102_DTYPES = {'BF16': torch.bfloat16, 'FP16': torch.float16,
                      'FP32': torch.float32, 'FP8E4M3': torch.float8_e4m3fn}


_BENCH_1102_SHAPES = [
    {'id': 's003', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 16, 128], [6656, 16, 128], [6656, 32, 128], [6656, 32], [6656, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[6656, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 14}, {'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 13}]},
    {'id': 's009', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[2560, 16, 128], [2560, 16, 128], [2560, 32, 128], [2560, 32], [2560, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[2560, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 76}]},
    {'id': 's040', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[2048, 16, 128], [2048, 16, 128], [2048, 32, 128], [2048, 32], [2048, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[2048, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 88}]},
    {'id': 's101', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 16, 128], [6656, 16, 128], [6656, 64, 128], [6656, 64], [6656, 64], [1, 64, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[6656, 64, 128], [1, 64, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 14}]},
    {'id': 's126', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[3584, 16, 128], [3584, 16, 128], [3584, 64, 128], [3584, 64], [3584, 64], [1, 64, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[3584, 64, 128], [1, 64, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 101}]},
    {'id': 's132', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[53248, 2, 128], [53248, 2, 128], [53248, 8, 128], [53248, 8], [53248, 8], [1, 8, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[53248, 8, 128], [1, 8, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 15}]},
    {'id': 's139', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[28672, 2, 128], [28672, 2, 128], [28672, 8, 128], [28672, 8], [28672, 8], [1, 8, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[28672, 8, 128], [1, 8, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 88}]},
]


def _bench_1102_cases():
    cases = []
    for shape in _BENCH_1102_SHAPES:
        routes = ('recompute',)
        for route in routes:
            cases.append({
                'shape_id': shape['id'], 'source_op': shape['op'], 'route': route,
                'measurement': 'component' if shape['op'] == 'gdn_chunk_delta' else 'operator',
                'input_shapes': json.dumps(shape['inputs'], separators=(',', ':')),
                'input_dtypes': ','.join(shape['dtypes']),
                'output_shapes': json.dumps(shape['outputs'], separators=(',', ':')),
                'output_dtypes': ','.join(shape['output_dtypes']),
                'sources': json.dumps(shape['locations'], separators=(',', ':')),
                'shape': shape,
            })
    return cases


def _bench_1102_case_id(case):
    return f"{case['source_op']}-{case['route']}-{case['shape_id']}"


def _build_bench_1102_runner(case, device='cuda'):
    """Allocate inputs once; return an operator/component invocation for timing."""
    shape = case['shape']
    inputs = shape['inputs']
    dtypes = [_BENCH_1102_DTYPES[name] for name in shape['dtypes']]
    route = case['route']

    def tensor(*dims, dtype=torch.bfloat16, value=None):
        tensor = torch.empty(dims, dtype=dtype, device=device)
        if tensor.device.type != 'meta':
            if value is not None:
                tensor.fill_(value)
            elif dtype == torch.float8_e4m3fn:
                tensor.view(torch.uint8).random_(24, 40)
            else:
                tensor.uniform_(-0.02, 0.02)
        return tensor

    def indices(count):
        return torch.arange(count, dtype=torch.int32, device=device)
    (total, heads, width) = inputs[0]
    (value_heads, value_width) = inputs[2][1:]
    gates = tensor(1, total, value_heads, dtype=torch.float32, value=-0.01)
    keys = tensor(1, total, heads, width)
    values = tensor(1, total, value_heads, value_width)
    target = importlib.import_module('boltops.fla.gdn.triton.recompute_w_u')
    return partial(target.recompute_w_u_fwd_gdn, keys, values, tensor(1, total, value_heads, dtype=torch.float32, value=0.5), gates, tensor(1, total, value_heads, 64))


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize('case', _bench_1102_cases(), ids=_bench_1102_case_id)
def test_bench_1102_gdn_recompute_triton(case, bench_warmup, bench_repeat, time_ms):
    run = _build_bench_1102_runner(case)
    run()
    latency = time_ms(run, bench_warmup, bench_repeat, dtype=case['input_dtypes'])
    print(f'{_bench_1102_case_id(case)} ({case["measurement"]}): {latency * 1000:.3f} us')
