"""Correctness and performance cases for the two public GDN chunk stages."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fla.gdn import chunk_fwd_o, chunk_gated_delta_rule_fwd_h
from ._gdn_reference import naive_chunk_h, naive_chunk_o

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_public_chunk_signatures_match_backends() -> None:
    from boltops.fla.gdn.triton.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as h_backend,
    )
    from boltops.fla.gdn.triton.chunk_o import chunk_fwd_o as o_backend

    def parameters(fn):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
        ]

    assert parameters(chunk_gated_delta_rule_fwd_h) == parameters(h_backend)
    assert parameters(chunk_fwd_o) == parameters(o_backend)


def _chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    boundaries = cu_seqlens.detach().cpu().tolist()
    for seq, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        rows.extend((seq, chunk) for chunk in range(math.ceil((eos - bos) / chunk_size)))
    return torch.tensor(rows, device=cu_seqlens.device, dtype=torch.int32)


def _make_chunk_cumsum(
    batch: int,
    total: int,
    H: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    *,
    use_exp2: bool,
) -> torch.Tensor:
    pointwise = -torch.rand(batch, total, H, device=DEVICE, dtype=torch.float32) * 0.04
    result = torch.empty_like(pointwise)
    boundaries = [0, total] if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()
    scale = math.log2(math.e) if use_exp2 else 1.0
    for bos, eos in zip(boundaries[:-1], boundaries[1:]):
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            result[:, start:end] = pointwise[:, start:end].cumsum(1) * scale
    return result


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("use_varlen", "transpose_state", "inplace", "gate_kind", "use_exp2"),
    [
        pytest.param(False, True, False, "g", False, id="fixed-vk-compact-g"),
        pytest.param(False, False, False, "gk", True, id="fixed-kv-compact-gk-exp2"),
        pytest.param(True, True, False, "g", False, id="varlen-vk-compact-invalid-index"),
        pytest.param(True, True, True, "g", True, id="varlen-vk-inplace-g-exp2"),
        pytest.param(True, False, True, "none", False, id="varlen-kv-inplace-no-gate"),
    ],
)
def test_chunk_h_and_o_match_fla_torch_reference(
    use_varlen: bool,
    transpose_state: bool,
    inplace: bool,
    gate_kind: str,
    use_exp2: bool,
) -> None:
    torch.manual_seed(120)
    chunk_size, H, Hg, K, V = 16, 4, 2, 32, 24
    if use_varlen:
        B, T = 1, 53
        cu = torch.tensor([0, 19, 53], device=DEVICE, dtype=torch.int32)
        N = 2
        indices = torch.tensor([2, -1], device=DEVICE, dtype=torch.int32)
        chunk_indices = _chunk_indices(cu, chunk_size)
    else:
        B, T, N = 2, 35, 2
        cu = None
        indices = torch.tensor([2, 0], device=DEVICE, dtype=torch.int32)
        chunk_indices = None
    dtype = torch.float16
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.15
    k = torch.randn_like(q) * 0.15
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.1
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=use_exp2) if gate_kind == "g" else None
    gk = None
    if gate_kind == "gk":
        gk = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32) * 0.03
        for start in range(0, T, chunk_size):
            gk[:, start : start + chunk_size] = gk[:, start : start + chunk_size].cumsum(1)
        if use_exp2:
            gk *= math.log2(math.e)
    trailing = (H, V, K) if transpose_state else (H, K, V)
    state_before = torch.randn(4, *trailing, device=DEVICE, dtype=torch.float32) * 0.01
    state_actual = state_before.clone()
    state_ref = state_before.clone()

    actual_h, actual_v, actual_final = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=gk,
        initial_state=state_actual,
        initial_state_indices=indices,
        output_final_state=True,
        inplace_final_state=inplace,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state,
    )
    expected_h, expected_v, expected_final = naive_chunk_h(
        k,
        w,
        u,
        g=g,
        gk=gk,
        initial_state=state_ref,
        initial_state_indices=indices,
        output_final_state=True,
        inplace_final_state=inplace,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state,
    )

    torch.testing.assert_close(actual_h, expected_h, atol=5e-2, rtol=3e-2)
    assert actual_v is not None and expected_v is not None
    torch.testing.assert_close(actual_v, expected_v, atol=5e-2, rtol=3e-2)
    assert actual_final is not None and expected_final is not None
    torch.testing.assert_close(actual_final, expected_final, atol=5e-2, rtol=3e-2)
    if inplace:
        assert actual_final is state_actual
    else:
        assert actual_final is not state_actual
        torch.testing.assert_close(state_actual, state_before)
        assert actual_final.shape[0] == N

    gamma = torch.full((H,), -0.003, device=DEVICE, dtype=torch.float32)
    actual_o = chunk_fwd_o(
        q,
        k,
        actual_v,
        actual_h,
        g=g,
        g_gamma=gamma,
        scale=K**-0.5,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state,
    )
    expected_o = naive_chunk_o(
        q,
        k,
        expected_v,
        expected_h,
        g=g,
        g_gamma=gamma,
        scale=K**-0.5,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state,
    )
    torch.testing.assert_close(actual_o, expected_o, atol=6e-2, rtol=4e-2)


@GPU_REQUIRED
def test_chunk_optional_outputs_do_not_modify_initial_state() -> None:
    torch.manual_seed(121)
    B, T, H, Hg, K, V = 1, 17, 2, 1, 32, 24
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.float16) * 0.1
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16) * 0.1
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16) * 0.1
    state = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32) * 0.01
    before = state.clone()
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        initial_state=state,
        output_final_state=False,
        inplace_final_state=False,
        chunk_size=16,
        save_new_value=False,
        transpose_state_layout=True,
    )
    assert h.shape == (B, 2, H, V, K)
    assert v_new is None and final is None
    torch.testing.assert_close(state, before)


def _chunk_h_flops(
    lengths: list[int], H: int, K: int, V: int, chunk_size: int
) -> int:
    return sum(
        4 * H * min(chunk_size, length - start) * K * V
        for length in lengths
        for start in range(0, length, chunk_size)
    )


def _chunk_o_flops(
    lengths: list[int], H: int, K: int, V: int, chunk_size: int
) -> int:
    total = 0
    for length in lengths:
        for start in range(0, length, chunk_size):
            size = min(chunk_size, length - start)
            total += 2 * H * size * K * V
            total += 2 * H * size * size * K
            total += 2 * H * size * size * V
    return total


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_chunk_h_o_tflops(
    use_varlen: bool,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    torch.manual_seed(122)
    B, T, H, Hg, K, V, chunk_size = 1, 13320, 16, 8, 128, 128, 64
    dtype = torch.float16
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.05
    if use_varlen:
        lengths = [4097, 4911, 4312]
        cu = torch.tensor([0, 4097, 9008, T], device=DEVICE, dtype=torch.int32)
        chunk_indices = _chunk_indices(cu, chunk_size)
        state_rows = len(lengths)
    else:
        lengths = [T]
        cu = None
        chunk_indices = None
        state_rows = B
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=False)
    state = torch.zeros(state_rows, H, V, K, device=DEVICE, dtype=torch.float32)
    state_indices = torch.arange(state_rows, device=DEVICE, dtype=torch.int32)

    def run_h():
        return chunk_gated_delta_rule_fwd_h(
            k, w, u, g=g, initial_state=state,
            initial_state_indices=state_indices,
            output_final_state=True, inplace_final_state=False,
            chunk_size=chunk_size, save_new_value=True,
            cu_seqlens=cu, chunk_indices=chunk_indices,
            transpose_state_layout=True,
        )

    h, v_new, final = run_h()
    assert v_new is not None and final is not None
    assert torch.isfinite(h).all() and torch.isfinite(v_new).all() and torch.isfinite(final).all()

    def run_o():
        return chunk_fwd_o(
            q, k, v_new, h, g=g, scale=K**-0.5,
            cu_seqlens=cu, chunk_size=chunk_size,
            chunk_indices=chunk_indices, transpose_state_layout=True,
        )

    output = run_o()
    assert torch.isfinite(output).all()

    def run_e2e():
        current_h, current_v, _ = run_h()
        assert current_v is not None
        return chunk_fwd_o(
            q, k, current_v, current_h, g=g, scale=K**-0.5,
            cu_seqlens=cu, chunk_size=chunk_size,
            chunk_indices=chunk_indices, transpose_state_layout=True,
        )

    h_ms = triton.testing.do_bench(
        run_h, warmup=bench_warmup, rep=bench_repeat
    )
    o_ms = triton.testing.do_bench(
        run_o, warmup=bench_warmup, rep=bench_repeat
    )
    e2e_ms = triton.testing.do_bench(
        run_e2e, warmup=bench_warmup, rep=bench_repeat
    )
    h_flops = _chunk_h_flops(lengths, H, K, V, chunk_size)
    o_flops = _chunk_o_flops(lengths, H, K, V, chunk_size)
    mode = "varlen" if use_varlen else "fixed"
    print(f"GDN chunk h ({mode}): {h_ms:.3f} ms, {h_flops / (h_ms * 1e-3) / 1e12:.2f} TFLOPS")
    print(f"GDN chunk o ({mode}): {o_ms:.3f} ms, {o_flops / (o_ms * 1e-3) / 1e12:.2f} TFLOPS")
    print(f"GDN chunk e2e ({mode}): {e2e_ms:.3f} ms, {(h_flops + o_flops) / (e2e_ms * 1e-3) / 1e12:.2f} TFLOPS")
