# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for KDA chunk stages and full prefill orchestration."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.fla.kda import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_state_output_fused,
    chunk_kda_fwd_intra,
    chunk_kda_with_fused_gate,
    recompute_w_u_fwd,
)
from boltops.fla.kda.triton.chunk import _use_kda_state_output_fused
from boltops.fla.kda.triton.chunk_intra import (
    get_chunk_kda_intra_sub_chunk_config,
)
from boltops.fla.kda.triton.chunk_state import (
    CHUNK_STATE_CONFIGS,
    get_chunk_state_config,
    get_fused_state_output_config,
)
from ._kda_reference import naive_kda_gate, naive_recompute_w_u, naive_recurrent_kda

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
def test_recompute_w_u_matches_torch_reference() -> None:
    torch.manual_seed(220)
    B, T, H, K, V, BT = 1, 16, 2, 32, 24, 16
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)
    gk = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.05
    A = torch.tril(torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.bfloat16))
    actual = recompute_w_u_fwd(k, v, beta, A, gk=gk)
    expected = naive_recompute_w_u(k, v, beta, A, gk)
    for got, want in zip((actual[0], actual[1], actual[3]), expected):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=1e-2)


@GPU_REQUIRED
def test_state_output_fused_wrapper_matches_existing_pipeline() -> None:
    torch.manual_seed(240)
    B, T, H, K, V, BT = 1, 70, 2, 32, 24, 64
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w = torch.randn_like(q) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.05
    kg = torch.randn_like(q) * 0.05
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    g[:, :BT] = g[:, :BT].cumsum(1)
    g[:, BT:] = g[:, BT:].cumsum(1)
    Aqk = torch.tril(torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.bfloat16))

    expected_h, expected_v_new, expected_final = chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=g, output_final_state=True, chunk_size=BT,
        use_exp2=True,
    )
    expected_o = torch.empty_like(u)
    chunk_gla_fwd_o_gk(
        q, expected_v_new, g, Aqk, expected_h, expected_o,
        K**-0.5, chunk_size=BT,
    )

    actual_o = torch.empty_like(u)
    actual_o, actual_final = chunk_kda_fwd_state_output_fused(
        q, w, u, g, kg, Aqk, actual_o, K**-0.5,
        output_final_state=False, chunk_size=BT,
    )
    torch.testing.assert_close(actual_o, expected_o, atol=6e-2, rtol=4e-2)
    assert actual_final is None


@GPU_REQUIRED
def test_state_output_fused_final_state_matches_per_head() -> None:
    """Every (batch, head) must write its own slice of ``ht``.

    Regression test for a missing base-pointer offset that made all programs
    race on ``ht[0, 0]``; it is only observable once B * H > 1.
    """
    torch.manual_seed(242)
    B, T, H, K, V, BT = 3, 192, 5, 32, 24, 64
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w = torch.randn_like(q) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.05
    kg = torch.randn_like(q) * 0.05
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    g = g.view(B, T // BT, BT, H, K).cumsum(2).view(B, T, H, K)
    Aqk = torch.tril(torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.bfloat16))
    initial = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01

    _, _, expected_final = chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=g, initial_state=initial.clone(),
        output_final_state=True, chunk_size=BT, use_exp2=True,
    )
    _, actual_final = chunk_kda_fwd_state_output_fused(
        q, w, u, g, kg, Aqk, torch.empty_like(u), K**-0.5,
        initial_state=initial.clone(), output_final_state=True, chunk_size=BT,
    )
    torch.testing.assert_close(actual_final, expected_final, atol=2e-3, rtol=2e-3)


def test_state_output_fused_selector_is_kda_specific() -> None:
    q = torch.empty(1, 8192, 96, 256, dtype=torch.bfloat16, device="meta")
    w = torch.empty_like(q)
    u = torch.empty(1, 8192, 96, 256, dtype=torch.bfloat16, device="meta")
    g = torch.empty(1, 8192, 96, 256, dtype=torch.float32, device="meta")
    Aqk = torch.empty(1, 8192, 96, 64, dtype=torch.bfloat16, device="meta")

    assert _use_kda_state_output_fused(q, w, u, g, w, Aqk, 64, None)
    assert not _use_kda_state_output_fused(q, w, u, g, w, Aqk, 32, None)
    assert not _use_kda_state_output_fused(
        q, w, u, g, torch.empty(1, 8192, 48, 256, dtype=torch.bfloat16, device="meta"), Aqk, 64, None
    )


def test_kda_prefill_config_selectors_cover_k128_fixed_and_varlen() -> None:
    assert get_chunk_kda_intra_sub_chunk_config(
        T=8192, H=96, HV=96, K=128, is_varlen=False,
        b=1, dtype=torch.bfloat16,
    )["BK"] == 128
    assert get_chunk_kda_intra_sub_chunk_config(
        T=8192, H=96, HV=96, K=128, is_varlen=True,
        b=1, dtype=torch.bfloat16,
    )["BK"] == 128
    assert get_chunk_state_config(
        N=1, H=96, V=128, K=128, is_varlen=False,
        b=1, dtype=torch.bfloat16,
    ) == CHUNK_STATE_CONFIGS[0]
    assert get_chunk_state_config(
        N=2, H=96, V=128, K=128, is_varlen=True,
        b=1, dtype=torch.bfloat16,
    ) == CHUNK_STATE_CONFIGS[1]

    q128 = torch.empty(1, 8192, 96, 128, dtype=torch.bfloat16, device="meta")
    assert _use_kda_state_output_fused(
        q128, torch.empty_like(q128),
        torch.empty(1, 8192, 96, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(1, 8192, 96, 128, dtype=torch.float32, device="meta"),
        torch.empty_like(q128),
        torch.empty(1, 8192, 96, 64, dtype=torch.bfloat16, device="meta"),
        64, None,
    )
    # output_final_state no longer disables the fusion: the fused kernel
    # writes a final state bit-identical to the standalone one.
    assert _use_kda_state_output_fused(
        q128, torch.empty_like(q128),
        torch.empty(1, 8192, 96, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(1, 8192, 96, 128, dtype=torch.float32, device="meta"),
        torch.empty_like(q128),
        torch.empty(1, 8192, 96, 64, dtype=torch.bfloat16, device="meta"),
        64, None, True,
    )

    # K/V=128 no longer has an N*H floor: the V-block is chosen from N*H so
    # the grid stays near one CU wave, and fusion wins at every head count.
    def _fused_at(b: int, h: int, t: int = 8192, cu: torch.Tensor | None = None) -> bool:
        q = torch.empty(b, t, h, 128, dtype=torch.bfloat16, device="meta")
        return _use_kda_state_output_fused(
            q, torch.empty_like(q),
            torch.empty(b, t, h, 128, dtype=torch.bfloat16, device="meta"),
            torch.empty(b, t, h, 128, dtype=torch.float32, device="meta"),
            torch.empty_like(q),
            torch.empty(b, t, h, 64, dtype=torch.bfloat16, device="meta"),
            64, cu,
        )

    assert _fused_at(1, 1)
    assert _fused_at(1, 6)
    assert _fused_at(1, 32)
    assert _fused_at(4, 8)
    assert _fused_at(1, 40)
    assert _fused_at(8, 8, t=4096)
    # varlen: N comes from cu_seqlens, not from the batch dim
    cu40 = torch.arange(41, dtype=torch.int32, device="meta")
    assert _fused_at(1, 1, t=40, cu=cu40)


def test_fused_state_output_config_tracks_the_cu_wave() -> None:
    """The V-block must follow the measured wave-quantization table.

    ``BV=V`` reads the K side once but carries a full ``[V,K]`` FP32
    accumulator, so it cannot co-schedule two workgroups per CU; past one wave
    the narrower tile wins despite duplicating its K-side reads.
    """
    measured = {16: 32, 24: 64, 32: 64, 40: 128, 48: 128, 64: 128, 72: 128, 96: 64}
    for n_programs, bv in measured.items():
        assert get_fused_state_output_config(n_programs, 128)["BV"] == bv, n_programs

    # BV never exceeds V.
    for n_programs in (8, 24, 64, 96):
        assert get_fused_state_output_config(n_programs, 64)["BV"] <= 64


@GPU_REQUIRED
def test_state_output_fused_wrapper_varlen_matches_existing_pipeline() -> None:
    torch.manual_seed(241)
    B, T, H, K, V, BT = 1, 70, 2, 32, 24, 64
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w = torch.randn_like(q) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.05
    kg = torch.randn_like(q) * 0.05
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    g[:, :33] = g[:, :33].cumsum(1)
    g[:, 33:] = g[:, 33:].cumsum(1)
    Aqk = torch.tril(torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.bfloat16))
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)
    initial = torch.randn(2, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01

    expected_h, expected_v_new, expected_final = chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=g, initial_state=initial.clone(),
        output_final_state=True, chunk_size=BT, use_exp2=True,
        cu_seqlens=cu,
    )
    expected_o = torch.empty_like(u)
    chunk_gla_fwd_o_gk(
        q, expected_v_new, g, Aqk, expected_h, expected_o,
        K**-0.5, cu_seqlens=cu, chunk_size=BT,
    )
    actual_o = torch.empty_like(u)
    actual_o, actual_final = chunk_kda_fwd_state_output_fused(
        q, w, u, g, kg, Aqk, actual_o, K**-0.5,
        initial_state=initial.clone(), output_final_state=False,
        cu_seqlens=cu, chunk_size=BT,
    )
    torch.testing.assert_close(actual_o, expected_o, atol=6e-2, rtol=4e-2)
    assert actual_final is None


@GPU_REQUIRED
def test_full_fused_gate_prefill_matches_recurrent_reference() -> None:
    torch.manual_seed(221)
    B, T, H, K, V = 1, 64, 2, 32, 24
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    initial = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    gate = naive_kda_gate(raw_g, A_log, bias, lower_bound=-3.0)
    beta = raw_beta.float().sigmoid()
    expected, expected_state = naive_recurrent_kda(
        q, k, v, beta, gate, scale=K**-0.5,
        initial_state=initial, use_qk_l2norm=True,
    )
    actual, actual_state = chunk_kda_with_fused_gate(
        q, k, v, raw_g, raw_beta, A_log, bias,
        scale=K**-0.5, initial_state=initial.clone(),
        output_final_state=True, lower_bound=-3.0,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=6e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=8e-2, rtol=6e-2)


@GPU_REQUIRED
def test_prefill_optimized_path_matches_recurrent_reference() -> None:
    """The K=128 fixed-schedule prefill (dispatched by ``chunk_kda_with_fused_gate``
    for T <= 1024) must match the naive recurrence reference, and T > 1024
    must fall back to the standard chunk pipeline at K=128."""
    from boltops.fla.kda.triton.prefill_optimized import can_run_prefill_optimized

    torch.manual_seed(231)
    B, H, K, V = 1, 24, 128, 128
    for T in (16, 1024, 1040):
        q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
        k = torch.randn_like(q)
        v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
        raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
        raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
        A_log = (torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.5) - 2.0
        bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
        initial = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
        # Inputs stay eligible for the fixed schedule at every T here; the
        # wrapper additionally gates T <= 1024 and H <= 64, so T=16/1024 run
        # the optimized path and T=1040 exercises the K=128 fallback.
        assert can_run_prefill_optimized(
            q, k, v, raw_g, raw_beta, lower_bound=-3.0
        ), T
        gate = naive_kda_gate(raw_g, A_log, bias, lower_bound=-3.0)
        beta = raw_beta.float().sigmoid()
        expected, expected_state = naive_recurrent_kda(
            q, k, v, beta, gate, scale=K**-0.5,
            initial_state=initial, use_qk_l2norm=True,
        )
        actual, actual_state = chunk_kda_with_fused_gate(
            q, k, v, raw_g, raw_beta, A_log, bias,
            scale=K**-0.5, initial_state=initial.clone(),
            output_final_state=True, lower_bound=-3.0,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=6e-2)
        torch.testing.assert_close(actual_state, expected_state, atol=8e-2, rtol=6e-2)


def test_intra_rejects_chunk_sizes_it_is_not_unrolled_for() -> None:
    """``chunk_size != 64`` must fail loudly rather than corrupt neighbours.

    ``chunk_kda_fwd_kernel_inter_solve_fused`` is hand-unrolled for four
    ``BC=16`` sub-chunks and guards its sub-chunk origins against ``T`` only,
    not against the chunk boundary.  At ``BT=16`` those origins land in the
    next three chunks and the stores race; the out-of-range column offsets are
    clipped by ``boundary_check``, so without this guard the failure is silent.
    """
    meta = dict(device="meta", dtype=torch.bfloat16)
    q = torch.empty(1, 64, 2, 128, **meta)
    gk = torch.empty(1, 64, 2, 128, device="meta", dtype=torch.float32)
    beta = torch.empty(1, 64, 2, device="meta", dtype=torch.float32)
    for bad in (16, 32, 128):
        with pytest.raises(ValueError, match="chunk_size == 64"):
            chunk_kda_fwd_intra(
                q=q, k=torch.empty_like(q), gk=gk, beta=beta,
                scale=128**-0.5, chunk_size=bad,
            )


@GPU_REQUIRED
@pytest.mark.parametrize("cu_list", [None, [0, 33, 70], [0, 1, 17, 200]])
def test_fused_diagonal_is_bit_identical_to_the_polynomial_pass(cu_list) -> None:
    """``FUSE_DIAG`` must reproduce the standalone polynomial kernel exactly.

    The fused path rebuilds the inverted diagonal blocks in registers instead
    of reading them back from ``Akkd``.  Both sides promote to FP32 before
    ``tl.dot(..., input_precision="ieee")``, so the result is bit-identical,
    not merely close -- assert that, since any drift means the gate reference
    row or the A2/A4/A8 chain diverged.
    """
    import boltops.fla.kda.triton.chunk_intra as chunk_intra
    from boltops.fla.common.triton.l2norm import l2norm_fwd
    from boltops.fla.kda.triton.chunk import fused_kda_gate_chunk_cumsum

    torch.manual_seed(303)
    BT, BC, K, H = 64, 16, 128, 4
    cu = None if cu_list is None else torch.tensor(cu_list, device=DEVICE, dtype=torch.int32)
    T = 320 if cu_list is None else cu_list[-1]
    q = l2norm_fwd(torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16))
    k = l2norm_fwd(torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16))
    g, beta = fused_kda_gate_chunk_cumsum(
        torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16).contiguous(),
        raw_beta=torch.randn(1, T, H, device=DEVICE, dtype=torch.bfloat16),
        A_log=torch.rand(H, device=DEVICE), g_bias=torch.rand(H, K, device=DEVICE),
        chunk_size=BT, lower_bound=-5.0, cu_seqlens=cu,
    )
    args = dict(q=q, k=k, gk=g, beta=beta, scale=K**-0.5, chunk_size=BT,
                safe_gate=True, cu_seqlens=cu)

    saved = chunk_intra.FUSE_INTRA_DIAGONAL
    try:
        chunk_intra.FUSE_INTRA_DIAGONAL = False
        Aqk_ref, Akk_ref = chunk_kda_fwd_intra(**args)
        chunk_intra.FUSE_INTRA_DIAGONAL = True
        Aqk_new, Akk_new = chunk_kda_fwd_intra(**args)
    finally:
        chunk_intra.FUSE_INTRA_DIAGONAL = saved

    # ``Aqk`` is allocated with ``torch.empty`` and only its causal
    # block-lower triangle is ever written; the rest is uninitialized on both
    # sides, so compare only the defined region.
    if cu_list is None:
        pos = torch.arange(T, device=DEVICE) % BT
    else:
        pos = torch.cat([
            torch.arange(cu_list[i + 1] - cu_list[i], device=DEVICE) % BT
            for i in range(len(cu_list) - 1)
        ])
    col = torch.arange(BT, device=DEVICE)
    defined = ((col[None, :] // BC) <= (pos[:, None] // BC))[None, :, None, :]
    torch.testing.assert_close(
        torch.where(defined, Aqk_new.float(), 0.0),
        torch.where(defined, Aqk_ref.float(), 0.0), atol=0, rtol=0,
    )
    torch.testing.assert_close(Akk_new.float(), Akk_ref.float(), atol=0, rtol=0)


POOL_CASES = [
    pytest.param(-1, (1, -1), id="sglang-null-negative"),
    pytest.param(0, (0, 2), id="vllm-null-zero"),
]


def _per_sequence_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    bias: torch.Tensor,
    pool: torch.Tensor,
    indices: tuple[int, ...],
    null_state_index: int,
    cu: torch.Tensor,
    lower_bound: float,
):
    bounds = cu.detach().cpu().tolist()
    gate = naive_kda_gate(raw_g[0], A_log, bias, lower_bound=lower_bound)
    beta = raw_beta[0].float().sigmoid()
    outs, finals = [], []
    for seq in range(len(bounds) - 1):
        bos, eos = bounds[seq], bounds[seq + 1]
        state_index = int(indices[seq])
        valid = state_index > null_state_index and state_index < pool.shape[0]
        initial = (
            pool[state_index : state_index + 1].clone() if valid
            else torch.zeros(1, *pool.shape[1:], device=pool.device, dtype=torch.float32)
        )
        ref_out, ref_state = naive_recurrent_kda(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            beta[bos:eos].unsqueeze(0),
            gate[bos:eos].unsqueeze(0),
            scale=k.shape[-1] ** -0.5,
            initial_state=initial,
            use_qk_l2norm=True,
        )
        outs.append((bos, eos, ref_out))
        finals.append((state_index, valid, ref_state))
    return outs, finals


@GPU_REQUIRED
@pytest.mark.parametrize("null_state_index, indices", POOL_CASES)
def test_full_prefill_pool_state_matches_recurrent_reference(
    null_state_index: int, indices: tuple[int, ...]
) -> None:
    """Pooled prefill (standard chunk-state path) writes the final state back
    into the indexed pool rows and leaves every other row untouched."""
    torch.manual_seed(250)
    B, T, H, K, V = 1, 70, 2, 32, 24
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    pool = torch.randn(5, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)

    outs, finals = _per_sequence_reference(
        q, k, v, raw_g, raw_beta, A_log, bias, pool, indices,
        null_state_index, cu, -3.0,
    )

    pool_before = pool.clone()
    actual, actual_final = chunk_kda_with_fused_gate(
        q, k, v, raw_g, raw_beta, A_log, bias,
        scale=K**-0.5,
        initial_state=pool,
        initial_state_indices=torch.tensor(
            indices, device=DEVICE, dtype=torch.int32
        ),
        output_final_state=True,
        inplace_final_state=True,
        lower_bound=-3.0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        null_state_index=null_state_index,
    )
    assert actual_final is pool
    for (bos, eos, ref_out), (state_index, valid, ref_state) in zip(outs, finals):
        torch.testing.assert_close(
            actual[:, bos:eos], ref_out, atol=8e-2, rtol=6e-2,
            msg=f"seq[{bos}:{eos}] output mismatch",
        )
        if valid:
            torch.testing.assert_close(
                pool[state_index : state_index + 1], ref_state,
                atol=8e-2, rtol=6e-2,
                msg=f"pool row {state_index} mismatch",
            )
    for row in range(pool.shape[0]):
        if not any(row == si and v_ for si, v_, _ in finals):
            assert torch.equal(pool_before[row], pool[row]), (
                f"pool row {row} must stay untouched"
            )


@GPU_REQUIRED
def test_full_prefill_pool_state_fused_path_matches_recurrent_reference() -> None:
    """Same pool contract on the fused state/output fast path (K=V=128)."""
    torch.manual_seed(251)
    B, T, H, K, V = 1, 70, 24, 128, 128
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = (torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.5) - 2.0
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    pool = torch.randn(5, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)
    indices = (1, -1)

    outs, finals = _per_sequence_reference(
        q, k, v, raw_g, raw_beta, A_log, bias, pool, indices,
        -1, cu, -3.0,
    )

    pool_before = pool.clone()
    actual, actual_final = chunk_kda_with_fused_gate(
        q, k, v, raw_g, raw_beta, A_log, bias,
        scale=K**-0.5,
        initial_state=pool,
        initial_state_indices=torch.tensor(
            indices, device=DEVICE, dtype=torch.int32
        ),
        output_final_state=True,
        inplace_final_state=True,
        lower_bound=-3.0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )
    assert actual_final is pool
    for (bos, eos, ref_out), (state_index, valid, ref_state) in zip(outs, finals):
        torch.testing.assert_close(
            actual[:, bos:eos], ref_out, atol=8e-2, rtol=6e-2,
            msg=f"seq[{bos}:{eos}] output mismatch",
        )
        if valid:
            torch.testing.assert_close(
                pool[state_index : state_index + 1], ref_state,
                atol=8e-2, rtol=6e-2,
                msg=f"pool row {state_index} mismatch",
            )
    for row in range(pool.shape[0]):
        if not any(row == si and v_ for si, v_, _ in finals):
            assert torch.equal(pool_before[row], pool[row]), (
                f"pool row {row} must stay untouched"
            )


@GPU_REQUIRED
def test_chunk_state_pool_non_inplace_final_state() -> None:
    """With inplace_final_state=False the final state lands in a dense
    per-sequence buffer while the pool itself is never written."""
    torch.manual_seed(252)
    B, T, H, K, V, BT = 1, 70, 2, 32, 24, 64
    kg = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w = torch.randn_like(kg) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.05
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    g[:, :33] = g[:, :33].cumsum(1)
    g[:, 33:] = g[:, 33:].cumsum(1)
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)
    pool = torch.randn(5, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    indices = torch.tensor([1, 3], device=DEVICE, dtype=torch.int32)
    pool_before = pool.clone()

    actual_h, actual_v_new, actual_final = chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=g, initial_state=pool, initial_state_indices=indices,
        output_final_state=True, cu_seqlens=cu, chunk_size=BT, use_exp2=True,
    )
    assert actual_final is not pool
    assert actual_final.shape == (2, H, V, K)
    assert torch.equal(pool_before, pool), "pool must never be written out-of-place"
    for seq, state_index in enumerate(indices.tolist()):
        bos, eos = (0, 33) if seq == 0 else (33, 70)
        ref_h, ref_v_new, ref_final = chunk_gated_delta_rule_fwd_h(
            kg[:, bos:eos], w[:, bos:eos], u[:, bos:eos], gk=g[:, bos:eos],
            initial_state=pool[state_index : state_index + 1].clone(),
            output_final_state=True, chunk_size=BT, use_exp2=True,
        )
        torch.testing.assert_close(
            actual_h[0, seq], ref_h[0, 0], atol=2e-3, rtol=2e-3,
            msg=f"seq {seq} chunk state mismatch",
        )
        torch.testing.assert_close(
            actual_v_new[:, bos:eos], ref_v_new, atol=2e-3, rtol=2e-3,
            msg=f"seq {seq} v_new mismatch",
        )
        torch.testing.assert_close(
            actual_final[seq], ref_final[0], atol=2e-3, rtol=2e-3,
            msg=f"seq {seq} final state mismatch",
        )


@GPU_REQUIRED
def test_chunk_state_pool_strided_rows() -> None:
    """Envelope-strided pools (per-row pitch > H*V*K) must work for both the
    initial read and the in-place final write."""
    torch.manual_seed(253)
    B, T, H, K, V, BT = 1, 70, 2, 32, 24, 64
    kg = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w = torch.randn_like(kg) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.05
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
    g[:, :33] = g[:, :33].cumsum(1)
    g[:, 33:] = g[:, 33:].cumsum(1)
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)
    envelope = torch.randn(5, 2 * H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    pool = envelope[:, :H]  # stride(0) == 2*H*V*K, rows stay contiguous
    assert pool.stride(0) == 2 * H * V * K
    indices = torch.tensor([1, 3], device=DEVICE, dtype=torch.int32)
    before = envelope.clone()
    pool_before = before[:, :H].contiguous().clone()

    actual_h, actual_v_new, actual_final = chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=g, initial_state=pool, initial_state_indices=indices,
        output_final_state=True, inplace_final_state=True,
        cu_seqlens=cu, chunk_size=BT, use_exp2=True,
    )
    assert actual_final is pool
    for seq, state_index in enumerate(indices.tolist()):
        bos, eos = (0, 33) if seq == 0 else (33, 70)
        _, _, ref_final = chunk_gated_delta_rule_fwd_h(
            kg[:, bos:eos], w[:, bos:eos], u[:, bos:eos], gk=g[:, bos:eos],
            initial_state=pool_before[state_index : state_index + 1].clone(),
            output_final_state=True, chunk_size=BT, use_exp2=True,
        )
        torch.testing.assert_close(
            actual_final[state_index], ref_final[0],
            atol=2e-3, rtol=2e-3,
            msg=f"pool row {state_index} mismatch",
        )
    for row in range(5):
        if row not in (1, 3):
            assert torch.equal(before[row], envelope[row]), (
                f"envelope row {row} must stay untouched"
            )


@GPU_REQUIRED
def test_chunk_state_pool_rejects_invalid_args() -> None:
    torch.manual_seed(254)
    B, T, H, K, V, BT = 1, 64, 2, 32, 24, 64
    kg = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn_like(kg)
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    pool = torch.randn(5, H, V, K, device=DEVICE, dtype=torch.float32)
    cu = torch.tensor([0, 33, 70], device=DEVICE, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires initial_state"):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, output_final_state=False,
            initial_state_indices=torch.tensor([1, 2], device=DEVICE, dtype=torch.int32),
            cu_seqlens=cu,
        )
    with pytest.raises(ValueError, match="null_state_index"):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, initial_state=pool,
            initial_state_indices=torch.tensor([1, 2], device=DEVICE, dtype=torch.int32),
            output_final_state=False, cu_seqlens=cu, null_state_index=1,
        )
    with pytest.raises(ValueError, match="initial_state is required"):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, inplace_final_state=True, output_final_state=True,
        )
    with pytest.raises(ValueError, match="output_final_state must be True"):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, initial_state=pool,
            initial_state_indices=torch.tensor([1, 2], device=DEVICE, dtype=torch.int32),
            inplace_final_state=True, output_final_state=False, cu_seqlens=cu,
        )
    with pytest.raises(ValueError, match="shape"):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, initial_state=pool,
            initial_state_indices=torch.tensor([1], device=DEVICE, dtype=torch.int32),
            output_final_state=False, cu_seqlens=cu,
        )
    with pytest.raises(TypeError):
        chunk_gated_delta_rule_fwd_h(
            kg, w, u, initial_state=pool,
            initial_state_indices=torch.tensor([1.0, 2.0], device=DEVICE),
            output_final_state=False, cu_seqlens=cu,
        )


@GPU_REQUIRED
def test_prefill_pool_state_full_chunk_intra_l2norm_matches_recurrent_reference() -> None:
    """Pooled prefill on full 64-token chunks (``T % 64 == 0``) at K=V=128.

    Pooling forces the standard pipeline and K=128 with ``T % 64 == 0``
    selects the in-kernel q/k l2norm branch.  Before the intra write-back
    fix, ``FUSE_DIAG`` consumed the raw (unnormalized) rows there and the
    output was corrupted by O(1e3) against the recurrent reference.
    """
    torch.manual_seed(255)
    B, T, H, K, V = 1, 256, 8, 128, 128
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16)
    raw_g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    A_log = (torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.5) - 2.0
    bias = torch.randn(H, K, device=DEVICE, dtype=torch.float32) * 0.1
    pool = torch.zeros(2, H, V, K, device=DEVICE, dtype=torch.float32)
    pool[1] = torch.randn(H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    pool_before = pool.clone()

    gate = naive_kda_gate(raw_g[0], A_log, bias, lower_bound=-3.0)
    beta = raw_beta[0].float().sigmoid()
    ref_out, ref_state = naive_recurrent_kda(
        q, k, v, beta.unsqueeze(0), gate.unsqueeze(0),
        scale=K ** -0.5, use_qk_l2norm=True,
    )

    actual, actual_final = chunk_kda_with_fused_gate(
        q, k, v, raw_g, raw_beta, A_log, bias,
        scale=K ** -0.5,
        initial_state=pool,
        initial_state_indices=torch.tensor([0], device=DEVICE, dtype=torch.int32),
        output_final_state=True,
        inplace_final_state=False,
        lower_bound=-3.0,
        use_qk_l2norm_in_kernel=True,
    )
    assert actual_final is not pool
    torch.testing.assert_close(
        actual, ref_out, atol=8e-2, rtol=6e-2,
        msg="full-chunk pooled prefill output mismatch",
    )
    torch.testing.assert_close(
        actual_final[0], ref_state[0], atol=8e-2, rtol=6e-2,
        msg="full-chunk pooled prefill final state mismatch",
    )
    assert torch.equal(pool_before, pool), "pool must never be written out-of-place"


@GPU_REQUIRED
@pytest.mark.parametrize("fuse_diag", [True, False])
def test_intra_normalize_qk_writeback_matches_l2norm(fuse_diag: bool) -> None:
    """``normalize_qk=True`` must leave l2-normalized rows in the caller's
    q/k buffers on every active diagonal path.

    With ``FUSE_INTRA_DIAGONAL=True`` the sub-chunk polynomial kernel (the
    only in-kernel writer) is not launched, so the wrapper normalizes the
    rows in place; with it ``False`` the sub-chunk kernel writes them back
    itself.  Either way the result must match the standalone ``l2norm_fwd``.
    """
    import boltops.fla.kda.triton.chunk_intra as chunk_intra
    from boltops.fla.common.triton.l2norm import l2norm_fwd
    from boltops.fla.kda.triton.chunk import fused_kda_gate_chunk_cumsum

    torch.manual_seed(256)
    B, T, H, K = 1, 128, 4, 128
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g, beta = fused_kda_gate_chunk_cumsum(
        torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16) * 0.1,
        raw_beta=torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16) * 0.1,
        A_log=(torch.randn(H, device=DEVICE) * 0.5) - 2.0,
        g_bias=torch.randn(H, K, device=DEVICE) * 0.1,
        chunk_size=64, lower_bound=-3.0,
    )
    q_ref = l2norm_fwd(q.clone())
    k_ref = l2norm_fwd(k.clone())

    saved = chunk_intra.FUSE_INTRA_DIAGONAL
    try:
        chunk_intra.FUSE_INTRA_DIAGONAL = fuse_diag
        chunk_intra.chunk_kda_fwd_intra(
            q=q, k=k, gk=g, beta=beta, scale=K ** -0.5,
            chunk_size=64, safe_gate=True, normalize_qk=True,
        )
    finally:
        chunk_intra.FUSE_INTRA_DIAGONAL = saved

    torch.testing.assert_close(q, q_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, k_ref, atol=1e-3, rtol=1e-3)
