"""Correctness and performance cases for packed GDN decode."""

from __future__ import annotations

import inspect

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fla.gdn import fused_recurrent_gated_delta_rule_packed_decode
from ._gdn_reference import (
    naive_gdn_gate,
    naive_recurrent_gated_delta_rule,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_public_packed_decode_signature_matches_backend() -> None:
    from boltops.fla.gdn.triton.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as backend,
    )

    assert inspect.signature(fused_recurrent_gated_delta_rule_packed_decode) == inspect.signature(backend)


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_l2norm", [False, True], ids=["raw-qk", "l2norm"])
def test_packed_decode_matches_fla_torch_reference(
    dtype: torch.dtype,
    use_l2norm: bool,
) -> None:
    torch.manual_seed(100)
    B, H, HV, K, V, state_rows = 4, 2, 4, 32, 24, 6
    qkv_dim = 2 * H * K + HV * V
    projection = torch.randn(B, qkv_dim + 17, device=DEVICE, dtype=dtype) * 0.2
    mixed_qkv = projection[:, :qkv_dim]  # Deliberately strided in the token dimension.
    a = torch.randn(B, HV, device=DEVICE, dtype=dtype) * 0.2
    b = torch.randn(B, HV, device=DEVICE, dtype=dtype) * 0.2
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=DEVICE, dtype=dtype) * 0.1
    indices = torch.tensor([3, 0, -1, 5], device=DEVICE, dtype=torch.int32)
    state_before = torch.randn(
        state_rows, HV, V, K, device=DEVICE, dtype=torch.float32
    ) * 0.02
    state_actual = state_before.clone()
    output = torch.empty(B, 1, HV, V, device=DEVICE, dtype=dtype)

    actual, returned_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        K**-0.5,
        state_actual,
        output,
        indices,
        use_qk_l2norm_in_kernel=use_l2norm,
    )

    expected_output = torch.zeros_like(output)
    expected_state = state_before.clone()
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    gate = naive_gdn_gate(a, A_log, dt_bias)
    beta = b.float().sigmoid()
    for token, state_index in enumerate(indices.tolist()):
        if state_index < 0:
            continue
        ref_out, ref_state_kv = naive_recurrent_gated_delta_rule(
            q[token : token + 1].view(1, 1, H, K),
            k[token : token + 1].view(1, 1, H, K),
            v[token : token + 1].view(1, 1, HV, V),
            beta[token : token + 1].view(1, 1, HV),
            gate[token : token + 1].view(1, 1, HV),
            scale=K**-0.5,
            initial_state=state_before[state_index : state_index + 1].transpose(-1, -2),
            output_final_state=True,
            use_qk_l2norm=use_l2norm,
        )
        expected_output[token] = ref_out[0]
        expected_state[state_index] = ref_state_kv[0].transpose(-1, -2)

    assert actual is output
    assert returned_state is state_actual
    torch.testing.assert_close(actual, expected_output, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(state_actual, expected_state, atol=3e-2, rtol=2e-2)


def _packed_decode_bytes(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    input_element_size: int,
) -> int:
    return (
        B * (2 * H * K + HV * V) * input_element_size
        + 2 * B * HV * input_element_size
        + HV * 4
        + HV * input_element_size
        + B * 4
        + 2 * B * HV * V * K * 4
        + B * HV * V * input_element_size
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("B", [1, 8, 32, 64], ids=lambda x: f"B{x}")
def test_bench_packed_decode_bandwidth(
    B: int,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    torch.manual_seed(101)
    H, HV, K, V = 4, 16, 128, 128
    dtype = torch.bfloat16
    mixed_qkv = torch.randn(
        B, 2 * H * K + HV * V, device=DEVICE, dtype=dtype
    ) * 0.05
    a = torch.randn(B, HV, device=DEVICE, dtype=dtype) * 0.05
    b = torch.randn(B, HV, device=DEVICE, dtype=dtype) * 0.05
    A_log = torch.zeros(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.zeros(HV, device=DEVICE, dtype=dtype)
    state = torch.zeros(B, HV, V, K, device=DEVICE, dtype=torch.float32)
    indices = torch.arange(B, device=DEVICE, dtype=torch.int32)
    output = torch.empty(B, 1, HV, V, device=DEVICE, dtype=dtype)

    def run() -> None:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            K**-0.5,
            state,
            output,
            indices,
            use_qk_l2norm_in_kernel=True,
        )

    run()
    assert torch.isfinite(output).all() and torch.isfinite(state).all()
    elapsed = triton.testing.do_bench(
        run, warmup=bench_warmup, rep=bench_repeat
    )
    logical_bytes = _packed_decode_bytes(B, H, HV, K, V, mixed_qkv.element_size())
    bandwidth = logical_bytes / (elapsed * 1e-3) / 1e9
    print(f"GDN packed decode: B={B}, {elapsed * 1e3:.2f} us, {bandwidth:.2f} GB/s")
