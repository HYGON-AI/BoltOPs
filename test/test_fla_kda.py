"""Correctness and optional performance tests for standalone KDA kernels."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from bolt_ops.fla.kda import (  # noqa: E402
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
    recompute_w_u_fwd,
)
from bolt_ops.fla.kda.triton import (  # noqa: E402
    chunk as kda_chunk_kernels,
    chunk_intra as kda_intra_kernels,
    chunk_intra_token_parallel as kda_token_kernels,
    fused_recurrent as kda_recurrent_kernels,
)
from bolt_ops.fla.kda.triton.fused_recurrent import (  # noqa: E402
    _fused_kda_gate_beta,
)

DEVICE = "cuda"
RCP_LN2 = 1.4426950408889634
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_public_apis_are_defined_at_package_boundary() -> None:
    import bolt_ops.fla.kda as kda
    import bolt_ops.fla.kda.triton as backend

    for name in kda.__all__:
        op = vars(kda).get(name)
        assert inspect.isfunction(op), f"{name} must be an explicit package API"
        assert op.__module__ == "bolt_ops.fla.kda"
        public_parameters = [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(op).parameters.values()
        ]
        backend_parameters = [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(
                getattr(backend, name)
            ).parameters.values()
        ]
        assert public_parameters == backend_parameters


def test_all_kda_kernels_use_bounded_bw1100_static_configs() -> None:
    kernels = {
        "recompute_w_u": (
            kda_chunk_kernels.recompute_w_u_fwd_kernel_beta_factored_head_first
        ),
        "chunk_output": kda_chunk_kernels.chunk_gla_fwd_kernel_o_local_first,
        "gate_cumsum": kda_chunk_kernels.kda_gate_chunk_cumsum_vector_kernel,
        "gate": kda_chunk_kernels.kda_gate_fwd_kernel,
        "inter_solve": kda_intra_kernels.chunk_kda_fwd_kernel_inter_solve_fused,
        "safe_intra": (
            kda_intra_kernels.chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized
        ),
        "token_intra": (
            kda_token_kernels.chunk_kda_fwd_kernel_intra_subchunk_raw_optimized
        ),
        "decode_gate_beta": kda_recurrent_kernels._kda_gate_beta_fwd_kernel,
        "recurrent": kda_recurrent_kernels.fused_recurrent_kda_fwd_kernel,
        "packed_decode": (
            kda_recurrent_kernels.fused_recurrent_kda_packed_decode_3d_kernel
        ),
    }
    assert len(kernels) == 10
    assert all(not hasattr(kernel, "configs") for kernel in kernels.values())

    config_sets = {
        "recompute_w_u": kda_chunk_kernels.RECOMPUTE_W_U_FWD_CONFIGS,
        "chunk_output": kda_chunk_kernels.CHUNK_GLA_FWD_O_CONFIGS,
        "gate_cumsum": (
            kda_chunk_kernels.KDA_GATE_CHUNK_CUMSUM_VECTOR_CONFIGS
        ),
        "gate": kda_chunk_kernels.KDA_GATE_FWD_CONFIGS,
        "inter_solve": kda_intra_kernels.CHUNK_KDA_INTER_SOLVE_CONFIGS,
        "safe_intra": (
            kda_intra_kernels.CHUNK_KDA_INTRA_SUB_CHUNK_CONFIGS
        ),
        "token_intra": (
            kda_token_kernels.CHUNK_KDA_INTRA_TOKEN_PARALLEL_CONFIGS
        ),
        "decode_gate_beta": kda_recurrent_kernels.KDA_GATE_BETA_CONFIGS,
        "recurrent": kda_recurrent_kernels.FUSED_RECURRENT_KDA_CONFIGS,
        "packed_decode": (
            kda_recurrent_kernels.FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS
        ),
    }
    assert config_sets.keys() == kernels.keys()
    for name, configs in config_sets.items():
        assert 2 <= len(configs) <= 3, (
            f"{name} must keep exactly 2-3 BW1100-tuned configs"
        )
        assert all("num_warps" in config for config in configs)
        signatures = {tuple(sorted(config.items())) for config in configs}
        assert len(signatures) == len(configs)

    selected_configs = {
        "recompute_w_u": kda_chunk_kernels.get_recompute_w_u_fwd_config(
            8192, 24, 128, 128, False
        ),
        "chunk_output": kda_chunk_kernels.get_chunk_gla_fwd_o_config(
            8192, 24, 128, 128, False
        ),
        "gate_cumsum": (
            kda_chunk_kernels.get_kda_gate_chunk_cumsum_vector_config(
                8192, 24, 128, False
            )
        ),
        "gate": kda_chunk_kernels.get_kda_gate_fwd_config(8192, 24, 128),
        "inter_solve": kda_intra_kernels.get_chunk_kda_inter_solve_config(
            8192, 24, 24, False, False
        ),
        "safe_intra": (
            kda_intra_kernels.get_chunk_kda_intra_sub_chunk_config(
                8192, 24, 24, False
            )
        ),
        "token_intra": (
            kda_token_kernels.get_chunk_kda_intra_token_parallel_config(
                8192, 24, 24, False
            )
        ),
        "decode_gate_beta": (
            kda_recurrent_kernels.get_kda_gate_beta_config(6, 24, 128)
        ),
        "recurrent": kda_recurrent_kernels.get_fused_recurrent_kda_config(
            6, 6, 24, True
        ),
        "packed_decode": (
            kda_recurrent_kernels.get_fused_recurrent_kda_packed_decode_config(
                6, 24, 128, 128
            )
        ),
    }
    for name, config in selected_configs.items():
        assert config in config_sets[name]

    # Exercise the shape branches used by small/common and large network cases.
    assert (
        kda_chunk_kernels.get_chunk_gla_fwd_o_config(128, 3, 128, 64, False)
        == kda_chunk_kernels.CHUNK_GLA_FWD_O_CONFIGS[2]
    )
    assert (
        kda_token_kernels.get_chunk_kda_intra_token_parallel_config(
            128, 3, 3, False
        )
        == kda_token_kernels.CHUNK_KDA_INTRA_TOKEN_PARALLEL_CONFIGS[0]
    )
    assert (
        kda_recurrent_kernels.get_fused_recurrent_kda_packed_decode_config(
            1, 3, 128, 128
        )
        == kda_recurrent_kernels.FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]
    )
    assert (
        kda_recurrent_kernels.get_fused_recurrent_kda_packed_decode_config(
            4, 3, 128, 128
        )
        == kda_recurrent_kernels.FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[1]
    )
    assert (
        kda_recurrent_kernels.get_fused_recurrent_kda_config(
            8, 8, 12, True
        )
        == kda_recurrent_kernels.FUSED_RECURRENT_KDA_CONFIGS[2]
    )
    assert (
        kda_recurrent_kernels.get_fused_recurrent_kda_packed_decode_config(
            8, 12, 128, 128
        )
        == kda_recurrent_kernels.FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[0]
    )
    assert (
        kda_recurrent_kernels.get_fused_recurrent_kda_packed_decode_config(
            16, 24, 128, 128
        )
        == kda_recurrent_kernels.FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]
    )


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt()
    return (error / (baseline + 1e-8)).item()


def _gate_reference(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    H, D = raw_g.shape[-2:]
    gate_input = raw_g.float().reshape(-1, H, D)
    if g_bias is not None:
        gate_input = gate_input + g_bias.float().reshape(H, D)
    a = A_log.float().exp().reshape(1, H, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * gate_input)
    return -a * F.softplus(gate_input)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
def test_fused_kda_gate_matches_torch(lower_bound: float | None) -> None:
    torch.manual_seed(10)
    T, H, D = 37, 4, 64
    raw_g = torch.randn(T, H * D, device=DEVICE, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    g_bias = torch.randn(H, D, device=DEVICE, dtype=torch.float32)

    actual = fused_kda_gate(
        raw_g, A_log, D, g_bias=g_bias, lower_bound=lower_bound
    )
    expected = _gate_reference(
        raw_g.view(T, H, D), A_log, g_bias, lower_bound
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
def test_fused_gate_chunk_cumsum_matches_torch(
    lower_bound: float | None,
) -> None:
    torch.manual_seed(11)
    B, T, H, D = 1, 70, 2, 32
    raw_g = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.bfloat16)
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    g_bias = torch.randn(H, D, device=DEVICE, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 19, 70], device=DEVICE, dtype=torch.int32)

    actual_g, actual_beta = fused_kda_gate_chunk_cumsum(
        raw_g,
        raw_beta,
        A_log,
        g_bias=g_bias,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    pointwise = _gate_reference(
        raw_g, A_log, g_bias, lower_bound
    ).reshape_as(raw_g)
    expected_g = torch.empty_like(pointwise)
    for start, end in ((0, 19), (19, 70)):
        for chunk_start in range(start, end, 64):
            chunk_end = min(chunk_start + 64, end)
            expected_g[:, chunk_start:chunk_end] = (
                pointwise[:, chunk_start:chunk_end].cumsum(1) * RCP_LN2
            )

    torch.testing.assert_close(actual_g, expected_g, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        actual_beta, raw_beta.float().sigmoid(), atol=2e-6, rtol=2e-6
    )


@GPU_REQUIRED
def test_recompute_and_chunk_output_match_torch() -> None:
    torch.manual_seed(12)
    B, T, H, K, V, BT = 1, 16, 2, 32, 24, 16
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    q = torch.randn_like(k)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)
    A = (
        torch.tril(
            torch.randn(B, H, T, BT, device=DEVICE, dtype=torch.float32)
            * 0.05
        )
        .permute(0, 2, 1, 3)
        .to(torch.bfloat16)
        .contiguous()
    )
    gk = (
        torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32)
        * 0.05
    )

    w, u, _, kg = recompute_w_u_fwd(k, v, beta, A, gk=gk)
    A_float = A.permute(0, 2, 1, 3).float()
    v_beta = (
        (v.float() * beta[..., None])
        .to(v.dtype)
        .float()
        .permute(0, 2, 1, 3)
    )
    k_beta_gate = (
        (k.float() * beta[..., None] * torch.exp2(gk))
        .to(k.dtype)
        .float()
        .permute(0, 2, 1, 3)
    )
    expected_u = torch.matmul(A_float, v_beta).permute(0, 2, 1, 3)
    expected_w = torch.matmul(A_float, k_beta_gate).permute(0, 2, 1, 3)
    expected_kg = k.float() * torch.exp2(gk[:, -1:] - gk)
    assert _relative_rmse(u, expected_u) < 5e-3
    assert _relative_rmse(w, expected_w) < 5e-3
    assert _relative_rmse(kg, expected_kg) < 5e-3

    h = torch.randn(1, H, V, K, device=DEVICE, dtype=torch.float32)
    output = torch.empty_like(v)
    scale = K**-0.5
    chunk_gla_fwd_o_gk(
        q, v, gk, A, h, output, scale=scale, chunk_size=BT
    )
    state_term = torch.matmul(
        (q.float() * scale * torch.exp2(gk)).permute(0, 2, 1, 3),
        h.transpose(-1, -2),
    ).permute(0, 2, 1, 3)
    local_term = torch.matmul(
        torch.tril(A_float), v.float().permute(0, 2, 1, 3)
    ).permute(0, 2, 1, 3)
    assert _relative_rmse(output, state_term + local_term) < 6e-3


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("T", "K", "boundaries"),
    [
        pytest.param(31, 32, None, id="fallback-tail"),
        pytest.param(70, 128, None, id="optimized-tail"),
        pytest.param(70, 128, (0, 19, 70), id="optimized-varlen"),
    ],
)
def test_chunk_intra_safe_and_token_parallel_agree(
    T: int,
    K: int,
    boundaries: tuple[int, ...] | None,
) -> None:
    torch.manual_seed(13)
    B, H, HV = 1, 2, 2
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    gk = (
        -torch.rand(B, T, HV, K, device=DEVICE, dtype=torch.float32)
        * 0.05
    ).cumsum(1) * RCP_LN2
    beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)

    cu_seqlens = None
    if boundaries is not None:
        cu_seqlens = torch.tensor(
            boundaries, device=DEVICE, dtype=torch.int32
        )
    token_aqk, token_akk = chunk_kda_fwd_intra(
        q, k, gk, beta, K**-0.5, cu_seqlens=cu_seqlens,
        safe_gate=False
    )
    safe_aqk, safe_akk = chunk_kda_fwd_intra(
        q, k, gk, beta, K**-0.5, cu_seqlens=cu_seqlens,
        safe_gate=True
    )
    rows = torch.empty(T, device=DEVICE, dtype=torch.long)
    spans = ((0, T),) if boundaries is None else zip(
        boundaries[:-1], boundaries[1:]
    )
    for start, end in spans:
        rows[start:end] = torch.arange(end - start, device=DEVICE) % 64
    columns = torch.arange(64, device=DEVICE)
    valid = (columns[None, :] <= rows[:, None])[None, :, None, :]
    token_aqk_valid = token_aqk.masked_select(valid)
    safe_aqk_valid = safe_aqk.masked_select(valid)
    token_akk_valid = token_akk.masked_select(valid)
    safe_akk_valid = safe_akk.masked_select(valid)
    assert not torch.isnan(token_aqk_valid).any()
    assert not torch.isnan(token_akk_valid).any()
    assert _relative_rmse(token_aqk_valid, safe_aqk_valid) < 5e-3
    assert _relative_rmse(token_akk_valid, safe_akk_valid) < 2e-2


def _make_decode_inputs(B: int, H: int, D: int):
    mixed_qkv = torch.randn(
        B, 3 * H * D, device=DEVICE, dtype=torch.bfloat16
    )
    raw_g = torch.randn(
        1, B, H, D, device=DEVICE, dtype=torch.bfloat16
    )
    raw_beta = torch.randn(
        1, B, H, device=DEVICE, dtype=torch.bfloat16
    )
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=DEVICE, dtype=torch.float32)
    state = (
        torch.randn(B + 1, H, D, D, device=DEVICE, dtype=torch.float32)
        * 0.01
    )
    state_indices = torch.arange(
        1, B + 1, device=DEVICE, dtype=torch.int32
    )
    return (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
    )


@GPU_REQUIRED
@pytest.mark.parametrize("lower_bound", [None, -3.0])
@pytest.mark.parametrize(
    ("B", "H"),
    [
        pytest.param(1, 3, id="tokens1-h3"),
        pytest.param(3, 2, id="tokens3-h2"),
    ],
)
def test_packed_decode_matches_dense_recurrent(
    B: int,
    H: int,
    lower_bound: float | None,
) -> None:
    torch.manual_seed(14)
    D = 128
    (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
    ) = _make_decode_inputs(B, H, D)
    q, k, v = (
        tensor.contiguous().view(1, B, H, D)
        for tensor in mixed_qkv.split(H * D, dim=-1)
    )
    gate = fused_kda_gate(
        raw_g.reshape(B, H * D),
        A_log,
        D,
        g_bias=dt_bias,
        lower_bound=lower_bound,
    ).unsqueeze(0)
    beta = raw_beta.float().sigmoid()
    cu_seqlens = torch.arange(B + 1, device=DEVICE, dtype=torch.int32)

    dense_state = state.clone()
    dense_out, _ = fused_recurrent_kda_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=dense_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    fused_state = state.clone()
    fused_out, _ = fused_recurrent_kda(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        fused_state,
        cu_seqlens,
        state_indices,
    )
    torch.testing.assert_close(fused_out, dense_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        fused_state, dense_state, atol=3e-3, rtol=3e-3
    )
    materialized_state = state.clone()
    materialized_out, _ = fused_recurrent_kda(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        materialized_state,
        cu_seqlens,
        state_indices,
        fuse_gate=False,
    )
    torch.testing.assert_close(
        materialized_out, dense_out, atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        materialized_state, dense_state, atol=3e-3, rtol=3e-3
    )
    packed_state = state.clone()
    packed_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        packed_state,
        state_indices,
    )
    torch.testing.assert_close(packed_out, dense_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        packed_state, dense_state, atol=3e-3, rtol=3e-3
    )


_PREFILL_B = 1
_PREFILL_T = 8192
_PREFILL_K = 128
_PREFILL_V = 128
_PREFILL_BT = 64
_PREFILL_BC = 16
_PREFILL_VARLEN_CU_SEQLENS = (0, 3000, 8192)

# Kimi-K3 has 96 total KDA heads.  SGLang DPA with TP32+CP8 uses
# attn_tp=TP/CP=4 (H=24); regular TP16/TP32 use H=6/H=3.  Keep H=96 for
# the full-head/unsharded-attention path.
_PARALLEL_HEAD_BENCH_CASES = [
    pytest.param(96, id="h96-full-heads"),
    pytest.param(24, id="h24-tp32-cp8-dpa"),
    pytest.param(6, id="h6-tp16"),
    pytest.param(3, id="h3-tp32"),
]

_DECODE_BENCH_SHAPES = [
    pytest.param(6, 96, 128, id="tokens6-h96-full-heads"),
    pytest.param(6, 24, 128, id="tokens6-h24-tp32-cp8-dpa"),
    pytest.param(6, 6, 128, id="tokens6-h6-tp16"),
    pytest.param(6, 3, 128, id="tokens6-h3-tp32"),
    pytest.param(1, 3, 128, id="tokens1-h3-tp32"),
    pytest.param(4, 3, 128, id="tokens4-h3"),
    pytest.param(8, 12, 128, id="tokens8-h12"),
    pytest.param(16, 24, 128, id="tokens16-h24"),
]


def _make_prefill_bench_inputs(H: int, use_varlen: bool = False):
    B, T, H, K, V, BT = (
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_K,
        _PREFILL_V,
        _PREFILL_BT,
    )
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    gate_steps = -torch.rand(
        B, T, H, K, device=DEVICE, dtype=torch.float32
    ) * 0.01
    gk = torch.empty_like(gate_steps)
    sequence_ranges = (
        zip(
            _PREFILL_VARLEN_CU_SEQLENS[:-1],
            _PREFILL_VARLEN_CU_SEQLENS[1:],
        )
        if use_varlen
        else ((0, T),)
    )
    for bos, eos in sequence_ranges:
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            gk[:, start:end] = gate_steps[:, start:end].cumsum(1)
    gk.mul_(RCP_LN2)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)
    return q, k, v, gk, beta


def _make_prefill_cu_seqlens(use_varlen: bool) -> torch.Tensor | None:
    if not use_varlen:
        return None
    return torch.tensor(
        _PREFILL_VARLEN_CU_SEQLENS, device=DEVICE, dtype=torch.int32
    )


def _prefill_num_chunks(use_varlen: bool) -> int:
    if not use_varlen:
        return (_PREFILL_T + _PREFILL_BT - 1) // _PREFILL_BT
    return sum(
        (eos - bos + _PREFILL_BT - 1) // _PREFILL_BT
        for bos, eos in zip(
            _PREFILL_VARLEN_CU_SEQLENS[:-1],
            _PREFILL_VARLEN_CU_SEQLENS[1:],
        )
    )


def test_prefill_bench_chunk_count_matches_varlen_layout() -> None:
    assert _prefill_num_chunks(False) == 128
    assert _prefill_num_chunks(True) == 129


def _print_prefill_bench(
    name: str,
    elapsed: float,
    H: int,
    use_varlen: bool | None = None,
) -> None:
    mode = ""
    if use_varlen is not None:
        mode = f", mode={'varlen' if use_varlen else 'fixed'}"
    print(
        f"KDA {name}: B={_PREFILL_B}, T={_PREFILL_T}, "
        f"H={H}, K={_PREFILL_K}, V={_PREFILL_V}, "
        f"{elapsed:.4f} ms{mode}"
    )


def _make_decode_bench_inputs(T: int, H: int, D: int):
    q = torch.randn(1, T, H, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn(1, T, H, D, device=DEVICE, dtype=torch.bfloat16)
    raw_beta = torch.randn(1, T, H, device=DEVICE, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=DEVICE, dtype=torch.float32)
    # Cache slot 0 is the invalid/sentinel entry in the recurrent kernels.
    # Keep one extra slot so every benchmark token exercises the recurrence.
    state = (
        torch.randn(T + 1, H, D, D, device=DEVICE, dtype=torch.float32)
        * 0.01
    )
    cu_seqlens = torch.arange(T + 1, device=DEVICE, dtype=torch.int32)
    state_indices = torch.arange(
        1, T + 1, device=DEVICE, dtype=torch.int32
    )
    mixed_qkv = torch.cat(
        tuple(x[0].reshape(T, H * D) for x in (q, k, v)), dim=-1
    ).contiguous()
    return (
        q,
        k,
        v,
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state,
        cu_seqlens,
        state_indices,
    )


def _print_decode_bench(
    name: str, elapsed: float, T: int, H: int, D: int
) -> None:
    print(
        f"KDA {name}: tokens={T}, H={H}, D={D}, {elapsed:.4f} ms"
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
def test_bench_kda_gate(
    H: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    raw_g = torch.randn(
        _PREFILL_B * _PREFILL_T,
        H * _PREFILL_K,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    g_bias = torch.randn(
        H, _PREFILL_K, device=DEVICE, dtype=torch.float32
    )
    elapsed = time_ms(
        lambda: fused_kda_gate(
            raw_g,
            A_log,
            _PREFILL_K,
            g_bias=g_bias,
        ),
        bench_warmup,
        bench_repeat,
    )
    _print_prefill_bench("gate", elapsed, H)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_kda_gate_chunk_cumsum(
    H: int,
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    raw_g = torch.randn(
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_K,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    raw_beta = torch.randn(
        _PREFILL_B,
        _PREFILL_T,
        H,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
    g_bias = torch.randn(
        H, _PREFILL_K, device=DEVICE, dtype=torch.float32
    )
    cu_seqlens = _make_prefill_cu_seqlens(use_varlen)
    elapsed = time_ms(
        lambda: fused_kda_gate_chunk_cumsum(
            raw_g,
            raw_beta,
            A_log,
            g_bias=g_bias,
            cu_seqlens=cu_seqlens,
            chunk_size=_PREFILL_BT,
        ),
        bench_warmup,
        bench_repeat,
    )
    _print_prefill_bench("gate chunk cumsum", elapsed, H, use_varlen)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_kda_intra_token_parallel(
    H: int,
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    q, k, _, gk, beta = _make_prefill_bench_inputs(H, use_varlen)
    Aqk = torch.empty(
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_BT,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    Akk = torch.empty(
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_BC,
        device=DEVICE,
        dtype=torch.float32,
    )
    cu_seqlens = _make_prefill_cu_seqlens(use_varlen)
    elapsed = time_ms(
        lambda: chunk_kda_fwd_intra_token_parallel(
            q,
            k,
            gk,
            beta,
            Aqk,
            Akk,
            _PREFILL_K**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_size=_PREFILL_BT,
            sub_chunk_size=_PREFILL_BC,
        ),
        bench_warmup,
        bench_repeat,
    )
    _print_prefill_bench("intra token parallel", elapsed, H, use_varlen)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
@pytest.mark.parametrize("safe_gate", [False, True], ids=["token", "safe"])
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_kda_prefill_intra(
    H: int,
    safe_gate: bool,
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    q, k, _, gk, beta = _make_prefill_bench_inputs(H, use_varlen)
    cu_seqlens = _make_prefill_cu_seqlens(use_varlen)
    elapsed = time_ms(
        lambda: chunk_kda_fwd_intra(
            q,
            k,
            gk,
            beta,
            _PREFILL_K**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_size=_PREFILL_BT,
            safe_gate=safe_gate,
        ),
        bench_warmup,
        bench_repeat,
    )
    path = "safe" if safe_gate else "token"
    _print_prefill_bench(f"prefill intra ({path})", elapsed, H, use_varlen)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_kda_recompute_w_u(
    H: int,
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    _, k, v, gk, beta = _make_prefill_bench_inputs(H, use_varlen)
    A = torch.randn(
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_BT,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    cu_seqlens = _make_prefill_cu_seqlens(use_varlen)
    elapsed = time_ms(
        lambda: recompute_w_u_fwd(
            k, v, beta, A, gk=gk, cu_seqlens=cu_seqlens
        ),
        bench_warmup,
        bench_repeat,
    )
    _print_prefill_bench("recompute W/U", elapsed, H, use_varlen)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("H", _PARALLEL_HEAD_BENCH_CASES)
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_kda_chunk_output(
    H: int,
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    q, _, v, gk, _ = _make_prefill_bench_inputs(H, use_varlen)
    A = torch.randn(
        _PREFILL_B,
        _PREFILL_T,
        H,
        _PREFILL_BT,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    h = torch.randn(
        _PREFILL_B,
        _prefill_num_chunks(use_varlen),
        H,
        _PREFILL_V,
        _PREFILL_K,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(v)
    cu_seqlens = _make_prefill_cu_seqlens(use_varlen)
    elapsed = time_ms(
        lambda: chunk_gla_fwd_o_gk(
            q,
            v,
            gk,
            A,
            h,
            output,
            _PREFILL_K**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_size=_PREFILL_BT,
        ),
        bench_warmup,
        bench_repeat,
    )
    _print_prefill_bench("chunk output", elapsed, H, use_varlen)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(("T", "H", "D"), _DECODE_BENCH_SHAPES)
def test_bench_kda_decode_gate_beta(
    T: int,
    H: int,
    D: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    (
        _,
        _,
        _,
        _,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        _,
        _,
        _,
    ) = _make_decode_bench_inputs(T, H, D)

    def run():
        return _fused_kda_gate_beta(
            raw_g, raw_beta, A_log, dt_bias, lower_bound=-3.0
        )

    gate, beta = run()
    assert gate.shape == (1, T, H, D)
    assert beta.shape == (1, T, H)
    assert gate.dtype == beta.dtype == torch.float32
    assert torch.isfinite(gate).all() and torch.isfinite(beta).all()
    elapsed = time_ms(run, bench_warmup, bench_repeat)
    _print_decode_bench("decode gate/beta", elapsed, T, H, D)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(("T", "H", "D"), _DECODE_BENCH_SHAPES)
def test_bench_kda_fused_recurrent(
    T: int,
    H: int,
    D: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    (
        q,
        k,
        v,
        _,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state,
        cu_seqlens,
        state_indices,
    ) = _make_decode_bench_inputs(T, H, D)
    output = torch.empty_like(v)

    def run():
        return fused_recurrent_kda(
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            -3.0,
            state,
            cu_seqlens,
            state_indices,
            out=output,
            fuse_gate=True,
        )

    actual, actual_state = run()
    assert actual is output
    assert actual.shape == (1, T, H, D)
    assert actual.dtype == torch.bfloat16
    assert actual_state.shape == (
        T + 1,
        H,
        D,
        D,
    )
    assert actual_state.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    elapsed = time_ms(run, bench_warmup, bench_repeat)
    _print_decode_bench("fused recurrent", elapsed, T, H, D)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(("T", "H", "D"), _DECODE_BENCH_SHAPES)
def test_bench_kda_packed_decode(
    T: int,
    H: int,
    D: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    (
        _,
        _,
        _,
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state,
        _,
        state_indices,
    ) = _make_decode_bench_inputs(T, H, D)

    def run():
        return fused_recurrent_kda_packed_decode(
            mixed_qkv,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            -3.0,
            state,
            state_indices,
        )

    actual, actual_state = run()
    assert actual.shape == (1, T, H, D)
    assert actual.dtype == torch.bfloat16
    assert actual_state.shape == (
        T + 1,
        H,
        D,
        D,
    )
    assert actual_state.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    elapsed = time_ms(run, bench_warmup, bench_repeat)
    _print_decode_bench("packed decode", elapsed, T, H, D)
