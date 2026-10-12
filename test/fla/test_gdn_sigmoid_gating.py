"""Correctness and performance cases for both sigmoid GDN update protocols."""

from __future__ import annotations

import inspect

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as fused_sigmoid_gating_delta_rule_update_verify,
)
from boltops.fla.gdn import fused_sigmoid_gating_delta_rule_update
from ._gdn_reference import (
    naive_gdn_gate,
    naive_kda_gate,
    naive_recurrent_gated_delta_rule,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_public_sigmoid_signatures_match_backends() -> None:
    from boltops.fla.gdn.triton.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as ordinary_backend,
    )

    def parameters(fn):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
        ]

    assert parameters(fused_sigmoid_gating_delta_rule_update) == parameters(ordinary_backend)


def _gate_for_tokens(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    is_kda: bool,
    softplus_beta: float,
    softplus_threshold: float,
) -> torch.Tensor:
    if is_kda:
        return naive_kda_gate(
            a,
            A_log,
            dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )
    return naive_gdn_gate(
        a,
        A_log,
        dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )


def _one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state_vk: torch.Tensor,
    *,
    scale: float,
    use_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, state_kv = naive_recurrent_gated_delta_rule(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
        beta.unsqueeze(0).unsqueeze(0),
        gate.unsqueeze(0).unsqueeze(0),
        scale=scale,
        initial_state=state_vk.unsqueeze(0).transpose(-1, -2),
        output_final_state=True,
        use_qk_l2norm=use_l2norm,
    )
    assert state_kv is not None
    return output[0, 0], state_kv[0].transpose(-1, -2)


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("tokens_per_seq", "inplace", "is_kda"),
    [
        pytest.param(1, True, False, id="decode-inplace-gdn"),
        pytest.param(3, True, False, id="spec-inplace-gdn"),
        pytest.param(3, False, False, id="spec-compact-gdn"),
        pytest.param(2, False, True, id="spec-compact-kda"),
    ],
)
def test_sigmoid_update_matches_fla_torch_reference(
    tokens_per_seq: int,
    inplace: bool,
    is_kda: bool,
) -> None:
    torch.manual_seed(110)
    N, H, HV, K, V, pool = 2, 2, 4, 32, 32, 10
    total = N * tokens_per_seq
    dtype = torch.bfloat16
    q = torch.randn(1, total, H, K, device=DEVICE, dtype=dtype) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(1, total, HV, V, device=DEVICE, dtype=dtype) * 0.2
    a_shape = (1, total, HV, K) if is_kda else (1, total, HV)
    a = torch.randn(*a_shape, device=DEVICE, dtype=dtype) * 0.2
    b = torch.randn(1, total, HV, device=DEVICE, dtype=dtype) * 0.2
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32) * 0.1
    dt_shape = (HV * K,) if is_kda else (HV,)
    dt_bias = torch.randn(*dt_shape, device=DEVICE, dtype=dtype) * 0.1
    state_before = torch.randn(pool, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    state_actual = state_before.clone()
    cu_seqlens = torch.arange(0, total + 1, tokens_per_seq, device=DEVICE, dtype=torch.int32)
    if tokens_per_seq == 1:
        indices = torch.tensor([3, 7], device=DEVICE, dtype=torch.int32)
        accepted = None
    else:
        indices = torch.tensor([[3, 4, 5], [7, 8, 9]], device=DEVICE, dtype=torch.int32)[:, :tokens_per_seq]
        accepted = torch.tensor([2, 1], device=DEVICE, dtype=torch.int32).clamp_max(tokens_per_seq)

    actual_output, actual_final = fused_sigmoid_gating_delta_rule_update(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        beta=1.0,
        threshold=20.0,
        scale=K**-0.5,
        initial_state=state_actual,
        inplace_final_state=inplace,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        is_kda=is_kda,
    )

    expected_output = torch.empty_like(v)
    expected_cache = state_before.clone()
    expected_compact = torch.empty(total, HV, V, K, device=DEVICE, dtype=torch.float32)
    flat_q, flat_k, flat_v = q[0], k[0], v[0]
    flat_a, flat_b = a.reshape(total, *a.shape[2:]), b[0]
    gate = _gate_for_tokens(
        flat_a,
        A_log,
        dt_bias,
        is_kda=is_kda,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    for seq in range(N):
        bos, eos = seq * tokens_per_seq, (seq + 1) * tokens_per_seq
        start_token = 0 if accepted is None else int(accepted[seq]) - 1
        index_row = indices if indices.ndim == 1 else indices[seq]
        state_index = int(index_row[seq] if indices.ndim == 1 else index_row[start_token])
        state = state_before[state_index].clone()
        for token in range(bos, eos):
            out, state = _one_step(
                flat_q[token],
                flat_k[token],
                flat_v[token],
                flat_b[token].float().sigmoid(),
                gate[token],
                state,
                scale=K**-0.5,
                use_l2norm=True,
            )
            expected_output[0, token] = out
            expected_compact[token] = state
            write_index = int(indices[seq] if indices.ndim == 1 else index_row[token - bos])
            expected_cache[write_index] = state

    torch.testing.assert_close(actual_output, expected_output, atol=3e-2, rtol=2e-2)
    if inplace:
        assert actual_final is state_actual
        torch.testing.assert_close(actual_final, expected_cache, atol=3e-2, rtol=2e-2)
    else:
        assert actual_final is not state_actual
        torch.testing.assert_close(state_actual, state_before)
        torch.testing.assert_close(actual_final, expected_compact, atol=3e-2, rtol=2e-2)



@GPU_REQUIRED
@pytest.mark.parametrize(
    ("null_state_index", "indices"),
    [
        pytest.param(0, [0, 1], id="vllm-null-zero"),
        pytest.param(-1, [0, -1], id="sglang-null-negative"),
    ],
)
def test_verify_update_state_index_abi(
    null_state_index: int, indices: list[int]
) -> None:
    """SGLang shared decode/verify entry honors both state-index ABIs."""
    torch.manual_seed(112)
    N, H, HV, K, V, pool = 2, 2, 4, 32, 32, 2
    dtype = torch.bfloat16
    q = torch.randn(1, N, H, K, device=DEVICE, dtype=dtype) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(1, N, HV, V, device=DEVICE, dtype=dtype) * 0.2
    a = torch.randn(1, N, HV, device=DEVICE, dtype=dtype) * 0.2
    b = torch.randn(1, N, HV, device=DEVICE, dtype=dtype) * 0.2
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=DEVICE, dtype=dtype) * 0.1
    state_before = torch.randn(
        pool, HV, V, K, device=DEVICE, dtype=torch.float32
    ) * 0.01
    state_actual = state_before.clone()
    cu_seqlens = torch.arange(0, N + 1, device=DEVICE, dtype=torch.int32)

    actual_output = fused_sigmoid_gating_delta_rule_update_verify(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        initial_state_source=state_actual,
        initial_state_indices=torch.tensor(
            indices, device=DEVICE, dtype=torch.int32
        ),
        scale=K**-0.5,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        null_state_index=null_state_index,
    )

    gate = naive_gdn_gate(
        a[0], A_log, dt_bias, softplus_beta=1.0, softplus_threshold=20.0
    )
    expected_output = torch.zeros_like(v)
    expected_cache = state_before.clone()
    for seq in range(N):
        state_index = int(indices[seq])
        if state_index <= null_state_index:
            continue  # invalid slot: zero output, state untouched
        state = expected_cache[state_index].clone()
        out, state = _one_step(
            q[0, seq],
            k[0, seq],
            v[0, seq],
            b[0, seq].float().sigmoid(),
            gate[seq],
            state,
            scale=K**-0.5,
            use_l2norm=True,
        )
        expected_output[0, seq] = out
        expected_cache[state_index] = state

    torch.testing.assert_close(actual_output, expected_output, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(state_actual, expected_cache, atol=3e-2, rtol=2e-2)


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("tree", "disable_state_update", "is_kda"),
    [
        pytest.param(False, False, False, id="linear-writeback-gdn"),
        pytest.param(False, True, False, id="linear-no-writeback-gdn"),
        pytest.param(True, True, False, id="tree-no-writeback-gdn"),
        pytest.param(True, True, True, id="tree-no-writeback-kda"),
    ],
)
def test_recurrent_sigmoid_update_matches_fla_torch_reference(
    tree: bool,
    disable_state_update: bool,
    is_kda: bool,
) -> None:
    torch.manual_seed(111)
    N, steps, H, HV, K, V, pool = 2, 3, 2, 4, 32, 32, 5
    total = N * steps
    dtype = torch.bfloat16
    q = torch.randn(1, total, H, K, device=DEVICE, dtype=dtype) * 0.15
    k = torch.randn_like(q) * 0.15
    v = torch.randn(1, total, HV, V, device=DEVICE, dtype=dtype) * 0.15
    a_shape = (1, total, HV, K) if is_kda else (1, total, HV)
    a = torch.randn(*a_shape, device=DEVICE, dtype=dtype) * 0.15
    b = torch.randn(1, total, HV, device=DEVICE, dtype=dtype) * 0.15
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32) * 0.1
    dt_shape = (HV * K,) if is_kda else (HV,)
    dt_bias = torch.randn(*dt_shape, device=DEVICE, dtype=dtype) * 0.1
    state_before = torch.randn(pool, HV, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    state_actual = state_before.clone()
    state_indices = torch.tensor([1, 4], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.arange(0, total + 1, steps, device=DEVICE, dtype=torch.int32)
    intermediate = torch.empty(N, steps, HV, V, K, device=DEVICE, dtype=torch.float32)
    intermediate_indices = torch.arange(N, device=DEVICE, dtype=torch.int32)
    parents = None
    if tree:
        parents = torch.tensor([[0, 0, 0], [0, 0, 1]], device=DEVICE, dtype=torch.int32)

    actual = fused_sigmoid_gating_delta_rule_update_verify(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        state_actual,
        state_indices,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=intermediate_indices,
        cache_steps=steps,
        retrieve_parent_token=parents,
    )

    expected_output = torch.empty_like(v)
    expected_state = state_before.clone()
    expected_intermediate = torch.empty_like(intermediate)
    gate = _gate_for_tokens(
        a.reshape(total, *a.shape[2:]),
        A_log,
        dt_bias,
        is_kda=is_kda,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    for seq in range(N):
        initial = state_before[int(state_indices[seq])].clone()
        state = initial
        for step in range(steps):
            token = seq * steps + step
            if tree and step > 0:
                state = expected_intermediate[seq, int(parents[seq, step])].clone()
            out, state = _one_step(
                q[0, token],
                k[0, token],
                v[0, token],
                b[0, token].float().sigmoid(),
                gate[token],
                state,
                scale=K**-0.5,
                use_l2norm=True,
            )
            expected_output[0, token] = out
            expected_intermediate[seq, step] = state
        if not disable_state_update:
            expected_state[int(state_indices[seq])] = state

    torch.testing.assert_close(actual, expected_output, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(intermediate, expected_intermediate, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(state_actual, expected_state, atol=3e-2, rtol=2e-2)


def _sigmoid_bytes(
    *,
    tokens: int,
    sequences: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    input_size: int,
    state_size: int,
    state_writes: int,
    intermediate_writes: int = 0,
) -> int:
    state_elements = HV * V * K
    return (
        tokens * (2 * H * K + HV * V) * input_size
        + 2 * tokens * HV * input_size
        + HV * (4 + input_size)
        + sequences * state_elements * state_size
        + state_writes * state_elements * state_size
        + intermediate_writes * state_elements * state_size
        + tokens * HV * V * input_size
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    ("N", "steps"),
    [pytest.param(4, 1, id="decode"), pytest.param(4, 4, id="spec-4")],
)
def test_bench_sigmoid_update_bandwidth(
    N: int,
    steps: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    torch.manual_seed(112)
    H, HV, K, V = 4, 16, 128, 128
    total, dtype = N * steps, torch.bfloat16
    q = torch.randn(1, total, H, K, device=DEVICE, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, total, HV, V, device=DEVICE, dtype=dtype) * 0.05
    a = torch.zeros(1, total, HV, device=DEVICE, dtype=dtype)
    b = torch.zeros_like(a)
    A_log = torch.zeros(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.zeros(HV, device=DEVICE, dtype=dtype)
    # vLLM's default mamba_ssm_cache_dtype="auto" follows the BF16 model dtype.
    state = torch.zeros(total, HV, V, K, device=DEVICE, dtype=dtype)
    indices = torch.arange(total, device=DEVICE, dtype=torch.int32).view(N, steps)
    accepted = torch.ones(N, device=DEVICE, dtype=torch.int32)
    cu = torch.arange(0, total + 1, steps, device=DEVICE, dtype=torch.int32)

    def run() -> None:
        fused_sigmoid_gating_delta_rule_update(
            A_log, a, b, dt_bias, q, k, v,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )

    run()
    assert torch.isfinite(state).all()
    elapsed = time_ms(run, bench_warmup, bench_repeat)
    logical_bytes = _sigmoid_bytes(
        tokens=total, sequences=N, H=H, HV=HV, K=K, V=V,
        input_size=q.element_size(), state_size=state.element_size(),
        state_writes=total,
    )
    bandwidth = logical_bytes / (elapsed * 1e-3) / 1e9
    print(f"GDN sigmoid update: N={N}, T={steps}, {elapsed * 1e3:.2f} us, {bandwidth:.2f} GB/s")


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("N", [1, 8, 32, 48], ids=lambda x: f"N{x}")
def test_bench_recurrent_speculative_verify_bandwidth(
    N: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    torch.manual_seed(113)
    steps, H, HV, K, V = 4, 4, 16, 128, 128
    total, dtype = N * steps, torch.bfloat16
    q = torch.randn(1, total, H, K, device=DEVICE, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, total, HV, V, device=DEVICE, dtype=dtype) * 0.05
    a = torch.zeros(1, total, HV, device=DEVICE, dtype=dtype)
    b = torch.zeros_like(a)
    A_log = torch.zeros(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.zeros(HV, device=DEVICE, dtype=dtype)
    state = torch.zeros(N, HV, V, K, device=DEVICE, dtype=torch.float32)
    state_indices = torch.arange(N, device=DEVICE, dtype=torch.int32)
    cu = torch.arange(0, total + 1, steps, device=DEVICE, dtype=torch.int32)
    intermediate = torch.empty(N, steps, HV, V, K, device=DEVICE, dtype=torch.float32)
    intermediate_indices = torch.arange(N, device=DEVICE, dtype=torch.int32)

    def run() -> None:
        fused_sigmoid_gating_delta_rule_update_verify(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
            state, state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            disable_state_update=True,
            intermediate_states_buffer=intermediate,
            intermediate_state_indices=intermediate_indices,
            cache_steps=steps,
        )

    run()
    assert torch.isfinite(intermediate).all()
    elapsed = time_ms(run, bench_warmup, bench_repeat)
    logical_bytes = _sigmoid_bytes(
        tokens=total, sequences=N, H=H, HV=HV, K=K, V=V,
        input_size=q.element_size(), state_size=state.element_size(),
        state_writes=0, intermediate_writes=total,
    )
    bandwidth = logical_bytes / (elapsed * 1e-3) / 1e9
    print(f"GDN target verify: N={N}, T={steps}, {elapsed * 1e3:.2f} us, {bandwidth:.2f} GB/s")
