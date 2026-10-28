# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for dense and packed KDA decode."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fla.kda import (
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)
from ._kda_reference import naive_kda_gate, naive_recurrent_kda

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
@pytest.mark.parametrize("tokens", [1, 3], ids=["decode", "spec"])
def test_decode_protocols_match_torch_reference(
    lower_bound: float | None, tokens: int
) -> None:
    torch.manual_seed(210)
    # Dense/packed recurrent wrappers require equal token strides; use K=V
    # here. Non-square K/V is covered by the prefill chunk tests.
    B, H, K, V = 1, 2, 32, 32
    q = torch.randn(B, tokens, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, tokens, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, tokens, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, tokens, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    # KDA decode entries default to the vLLM ABI (null_state_index=0): slot 0
    # is NULL_BLOCK_ID, so live state sits on 1-based rows and row 0 must stay
    # untouched. The kernel contract forbids concurrent sequences from writing
    # one state slot, so each of the `tokens` one-token sequences owns its own
    # row (1..tokens); the multi-token chain through one row is covered by
    # test_dense_decode_single_sequence_state_chain.
    state = torch.zeros(
        tokens + 1, H, V, K, device=DEVICE, dtype=torch.float32
    )
    for slot in range(1, tokens + 1):
        state[slot] = torch.randn(H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    cu = torch.arange(tokens + 1, device=DEVICE, dtype=torch.int32)
    indices = torch.arange(1, tokens + 1, device=DEVICE, dtype=torch.int32)
    gate = fused_kda_gate(
        raw_g.reshape(tokens, H * K), A_log, K, g_bias=bias,
        lower_bound=lower_bound,
    ).reshape(B, tokens, H, K)
    beta = raw_beta.float().sigmoid()
    expected_out = torch.empty_like(v)
    expected_pool = state.clone()
    for seq in range(tokens):
        ref_out, ref_state = naive_recurrent_kda(
            q[:, seq:seq + 1], k[:, seq:seq + 1], v[:, seq:seq + 1],
            beta[:, seq:seq + 1], gate[:, seq:seq + 1], scale=K**-0.5,
            initial_state=state[seq + 1:seq + 2], use_qk_l2norm=True,
        )
        expected_out[:, seq] = ref_out[0]
        expected_pool[seq + 1] = ref_state[0]

    dense_state = state.clone()
    dense_out, _ = fused_recurrent_kda_fwd(
        q, k, v, gate, beta, initial_state=dense_state,
        cu_seqlens=cu, ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    fused_state = state.clone()
    fused_out, _ = fused_recurrent_kda(
        q, k, v, raw_g, raw_beta, A_log, bias, lower_bound,
        fused_state, cu, indices,
    )
    mixed_qkv = torch.cat(
        (q[0].reshape(tokens, H * K),
         k[0].reshape(tokens, H * K),
         v[0].reshape(tokens, H * V)),
        dim=-1,
    ).contiguous()
    packed_state = state.clone()
    packed_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv, raw_g, raw_beta, A_log, bias, lower_bound,
        packed_state, indices,
    )
    for actual in (dense_out, fused_out, packed_out):
        torch.testing.assert_close(actual, expected_out, atol=3e-2, rtol=2e-2)
    for name, actual_state in (
        ("dense", dense_state), ("fused", fused_state), ("packed", packed_state)
    ):
        torch.testing.assert_close(
            actual_state, expected_pool, atol=8e-2, rtol=4e-2,
            msg=f"{name} state mismatch",
        )


@GPU_REQUIRED
def test_dense_decode_single_sequence_state_chain() -> None:
    """Exercise the multi-token recurrent loop used by software pipelining."""
    torch.manual_seed(211)
    T, H, K, V = 3, 2, 32, 32
    q = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    gate = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.float32) * -0.01
    beta = torch.sigmoid(torch.randn(1, T, H, device=DEVICE, dtype=torch.float32))
    initial = torch.randn(1, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    expected_out, expected_state = naive_recurrent_kda(
        q, k, v, beta, gate, scale=K**-0.5,
        initial_state=initial, use_qk_l2norm=True,
    )

    state = torch.zeros(T + 1, H, V, K, device=DEVICE, dtype=torch.float32)
    state[1] = initial[0]
    out, _ = fused_recurrent_kda_fwd(
        q, k, v, gate, beta, initial_state=state,
        cu_seqlens=torch.tensor([0, T], device=DEVICE, dtype=torch.int32),
        ssm_state_indices=torch.arange(
            1, T + 1, device=DEVICE, dtype=torch.int32,
        ).view(1, T),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out, expected_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        state[T], expected_state[0], atol=8e-2, rtol=4e-2,
    )


ABI_CASES = [
    pytest.param(0, (0, 1), id="vllm-null-zero"),
    pytest.param(-1, (0, -1), id="sglang-null-negative"),
]


def _packed_kda_reference(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: tuple[int, ...],
    null_state_index: int,
    lower_bound: float | None,
):
    B = raw_g.shape[1]
    H, K = raw_g.shape[2], raw_g.shape[3]
    V = state.shape[2]
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, H * V], dim=-1)
    gate = naive_kda_gate(
        raw_g, A_log, g_bias=dt_bias, lower_bound=lower_bound
    ).reshape(B, H, K)
    beta = raw_beta.reshape(B, H).float().sigmoid()
    expected_out = torch.zeros(B, H, V, device=state.device, dtype=torch.bfloat16)
    expected_state = state.clone()
    for token, state_index in enumerate(indices):
        if state_index <= null_state_index:
            continue  # invalid slot: zero output, state untouched
        ref_out, ref_state = naive_recurrent_kda(
            q[token : token + 1].view(1, 1, H, K),
            k[token : token + 1].view(1, 1, H, K),
            v[token : token + 1].view(1, 1, H, V),
            beta[token : token + 1].view(1, 1, H),
            gate[token : token + 1].view(1, 1, H, K),
            scale=K**-0.5,
            initial_state=expected_state[state_index : state_index + 1],
            use_qk_l2norm=True,
        )
        expected_out[token] = ref_out[0]
        expected_state[state_index] = ref_state[0]
    return expected_out, expected_state


@GPU_REQUIRED
@pytest.mark.parametrize("null_state_index, indices", ABI_CASES)
def test_kda_packed_decode_state_index_abi(
    null_state_index: int, indices: tuple[int, ...]
) -> None:
    """Both frameworks can call packed KDA decode through ``null_state_index``."""
    torch.manual_seed(212)
    B, H, K, V = 2, 2, 32, 32
    q = torch.randn(B, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(1, B, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(1, B, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    state_before = torch.randn(
        2, H, V, K, device=DEVICE, dtype=torch.float32
    ) * 0.01
    mixed_qkv = torch.cat(
        (q.reshape(B, H * K), k.reshape(B, H * K), v.reshape(B, H * V)), dim=-1
    ).contiguous()

    expected_out, expected_state = _packed_kda_reference(
        mixed_qkv, raw_g, raw_beta, A_log, dt_bias,
        state_before, indices, null_state_index, None,
    )

    actual_state = state_before.clone()
    actual, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        None,
        actual_state,
        torch.tensor(indices, device=DEVICE, dtype=torch.int32),
        null_state_index=null_state_index,
    )
    torch.testing.assert_close(
        actual.reshape(B, H, V), expected_out, atol=3e-2, rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_state, expected_state, atol=8e-2, rtol=4e-2,
    )


@GPU_REQUIRED
@pytest.mark.parametrize("null_state_index, indices", ABI_CASES)
def test_kda_packed_decode_direct_3d_matches_reference(
    null_state_index: int, indices: tuple[int, ...]
) -> None:
    """Direct-3D kernel path: K=V=128, H>=24, FP32 state, bounded gate."""
    torch.manual_seed(213)
    B, H, K, V = 2, 24, 128, 128
    q = torch.randn(B, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(1, B, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(1, B, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    state_before = torch.randn(
        2, H, V, K, device=DEVICE, dtype=torch.float32
    ) * 0.01
    mixed_qkv = torch.cat(
        (q.reshape(B, H * K), k.reshape(B, H * K), v.reshape(B, H * V)), dim=-1
    ).contiguous()
    lower_bound = -3.0

    expected_out, expected_state = _packed_kda_reference(
        mixed_qkv, raw_g, raw_beta, A_log, dt_bias,
        state_before, indices, null_state_index, lower_bound,
    )

    actual_state = state_before.clone()
    actual, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        actual_state,
        torch.tensor(indices, device=DEVICE, dtype=torch.int32),
        null_state_index=null_state_index,
    )
    torch.testing.assert_close(
        actual.reshape(B, H, V), expected_out, atol=3e-2, rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_state, expected_state, atol=8e-2, rtol=4e-2,
    )


@GPU_REQUIRED
@pytest.mark.parametrize("entry", ["dense", "packed"], ids=["dense", "packed"])
def test_kda_rejects_invalid_null_state_index(entry: str) -> None:
    torch.manual_seed(214)
    B, H, K, V = 1, 2, 32, 32
    q = torch.randn(1, 1, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(1, 1, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(1, B, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(1, B, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    state = torch.zeros(1, H, V, K, device=DEVICE, dtype=torch.float32)
    for bad in (1, -2):
        with pytest.raises(ValueError, match="null_state_index"):
            if entry == "dense":
                fused_recurrent_kda_fwd(
                    q, k, v, torch.zeros_like(q),
                    torch.zeros(1, 1, H, device=DEVICE),
                    initial_state=state,
                    cu_seqlens=torch.tensor([0, 1], device=DEVICE, dtype=torch.int32),
                    ssm_state_indices=torch.zeros(
                        1, device=DEVICE, dtype=torch.int32
                    ),
                    null_state_index=bad,
                )
            else:
                fused_recurrent_kda_packed_decode(
                    torch.cat(
                        (
                            q.reshape(1, H * K),
                            k.reshape(1, H * K),
                            v.reshape(1, H * V),
                        ),
                        dim=-1,
                    ),
                    raw_g, raw_beta, A_log, dt_bias, None,
                    state,
                    torch.zeros(B, device=DEVICE, dtype=torch.int32),
                    null_state_index=bad,
                )


# ---------------------------------------------------------------------------
# Performance benchmarks (real-model decode sizes)
# ---------------------------------------------------------------------------

_BENCH_DTYPE = torch.bfloat16
_BENCH_HEADS = 24
_BENCH_HEAD_DIM = 128
# N = independent one-token sequences (dense) / tokens (packed)
_BENCH_N = (1, 6, 32, 64)


def _bench_common_inputs(N: int):
    H, K, V = _BENCH_HEADS, _BENCH_HEAD_DIM, _BENCH_HEAD_DIM
    q = torch.randn(1, N, H, K, device=DEVICE, dtype=_BENCH_DTYPE) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, N, H, V, device=DEVICE, dtype=_BENCH_DTYPE) * 0.1
    raw_g = torch.randn(1, N, H, K, device=DEVICE, dtype=_BENCH_DTYPE) * 0.1
    raw_beta = torch.randn(1, N, H, device=DEVICE, dtype=_BENCH_DTYPE) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    # Slot 0 of the pool is the vLLM null slot; live state sits on 1-based rows.
    state = torch.zeros(N + 1, H, V, K, device=DEVICE, dtype=torch.float32)
    state[1:] = torch.randn(N, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    cu = torch.arange(N + 1, device=DEVICE, dtype=torch.int32)
    indices = torch.arange(1, N + 1, device=DEVICE, dtype=torch.int32)
    return q, k, v, raw_g, raw_beta, A_log, dt_bias, state, cu, indices


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("entry", ["dense", "packed", "dense_pregated"])
@pytest.mark.parametrize("N", list(_BENCH_N), ids=lambda n: f"t{n}")
def test_bench_kda_decode(
    entry: str,
    N: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Time the three production decode entries on real-model sizes.

    H=24, K=V=128, BF16 inputs / FP32 state pool updated in place (the real
    decode-loop semantics).  ``dense`` is the gate-fused vLLM wrapper,
    ``packed`` the packed-QKV entry, and ``dense_pregated`` the SGLang path
    with the gate materialized once outside the timed call.
    """
    torch.manual_seed(222)
    H, K, V = _BENCH_HEADS, _BENCH_HEAD_DIM, _BENCH_HEAD_DIM
    q, k, v, raw_g, raw_beta, A_log, dt_bias, state, cu, indices = (
        _bench_common_inputs(N)
    )

    if entry == "dense":

        def run():
            return fused_recurrent_kda(
                q, k, v, raw_g, raw_beta, A_log, dt_bias, None,
                state, cu, indices,
            )

    elif entry == "packed":
        mixed_qkv = torch.cat(
            (q[0].reshape(N, H * K), k[0].reshape(N, H * K),
             v[0].reshape(N, H * V)),
            dim=-1,
        ).contiguous()

        def run():
            return fused_recurrent_kda_packed_decode(
                mixed_qkv, raw_g, raw_beta, A_log, dt_bias, None,
                state, indices,
            )

    else:
        gate = fused_kda_gate(
            raw_g.reshape(N, H * K), A_log, K, g_bias=dt_bias
        ).reshape(1, N, H, K)
        beta = raw_beta.float().sigmoid()

        def run():
            return fused_recurrent_kda_fwd(
                q, k, v, gate, beta, initial_state=state,
                cu_seqlens=cu, ssm_state_indices=indices,
            )

    output, returned_state = run()
    assert torch.isfinite(output).all() and torch.isfinite(state).all()

    ms = time_ms(run, bench_warmup, bench_repeat)
    flops = 4.0 * H * K * V * N
    print(
        f"[bench] kda_decode {entry} N={N}: "
        f"{ms:.4f} ms, {flops / (ms * 1e9):.3f} TFLOPS"
    )
