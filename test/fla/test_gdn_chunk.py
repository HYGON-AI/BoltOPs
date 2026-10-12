"""Correctness and performance cases for GDN chunk kernels."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable

import pytest
import torch

triton = pytest.importorskip("triton")
from triton.testing import do_bench as _do_bench_ref  # reference timing only, not collected

import boltops.fla.gdn as gdn
from boltops.utility.backends import discover_backends
from boltops.fla.gdn import chunk_gated_delta_rule_fwd_h
from ._gdn_reference import naive_chunk_h, naive_chunk_o

DEVICE = "cuda"
CORRECTNESS_SEQ_LEN = 320
BF16_ATOL = 8e-2
BF16_RTOL = 8e-2
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


CHUNK_H_BACKENDS = discover_backends(gdn, "chunk_gated_delta_rule_fwd_h")
CHUNK_O_BACKENDS = discover_backends(gdn, "chunk_fwd_o")


def _backend_params(backends: dict[str, Callable]):
    return [
        pytest.param((name, operator), id=name)
        for name, operator in backends.items()
    ]


def _chunk_h_correctness_params():
    params = []
    for backend_name in CHUNK_H_BACKENDS:
        implementations = (
            ("copy-pipeline", "wasp_4p4c")
            if backend_name == "tilelang"
            else (None,)
        )
        for implementation in implementations:
            case_name = backend_name
            if implementation is not None:
                case_name += f"-{implementation.replace('_', '-')}"
            params.append(
                pytest.param(
                    backend_name,
                    implementation,
                    id=case_name,
                )
            )
    return params


@pytest.fixture(params=_backend_params(CHUNK_H_BACKENDS))
def chunk_h_backend(
    request: pytest.FixtureRequest,
) -> tuple[str, Callable]:
    return request.param


@pytest.fixture(params=_backend_params(CHUNK_O_BACKENDS))
def chunk_o_backend(
    request: pytest.FixtureRequest,
) -> tuple[str, Callable]:
    return request.param


def test_tilelang_api_matches_public_api() -> None:
    if "tilelang" not in CHUNK_H_BACKENDS:
        pytest.skip("TileLang backend is unavailable")

    def parameters(fn: Callable):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(fn).parameters.values()
        ]

    assert parameters(CHUNK_H_BACKENDS["tilelang"]) == parameters(
        chunk_gated_delta_rule_fwd_h
    )


def _chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    boundaries = cu_seqlens.detach().cpu().tolist()
    for seq, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        num_chunks = math.ceil((eos - bos) / chunk_size)
        rows.extend((seq, chunk) for chunk in range(num_chunks))
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
    pointwise = (
        -torch.rand(batch, total, H, device=DEVICE, dtype=torch.float32)
        * 0.04
    )
    result = torch.empty_like(pointwise)
    boundaries = (
        [0, total]
        if cu_seqlens is None
        else cu_seqlens.detach().cpu().tolist()
    )
    scale = math.log2(math.e) if use_exp2 else 1.0
    for bos, eos in zip(boundaries[:-1], boundaries[1:]):
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            result[:, start:end] = pointwise[:, start:end].cumsum(1) * scale
    return result


def _kernel_cfg(
    backend_name: str,
    implementation: str | None = "wasp_4p4c",
) -> dict[str, str] | None:
    if backend_name == "tilelang":
        assert implementation is not None
        return {"implementation": implementation}
    return None


def _check_chunk_h_matches_reference(
    chunk_backend: tuple[str, Callable],
    use_varlen: bool,
    inplace_final_state: bool,
    implementation: str | None = "wasp_4p4c",
) -> None:
    backend_name, chunk_h = chunk_backend
    torch.manual_seed(120 if use_varlen else 123)
    B, H, Hg, K, V, chunk_size = 1, 16, 16, 128, 128, 64
    minimum_T = 5 * chunk_size
    T = CORRECTNESS_SEQ_LEN
    assert T >= minimum_T
    dtype = torch.bfloat16
    if use_varlen:
        first = T // 3
        second = (T - first) // 2
        lengths = [first, second, T - first - second]
        cu = torch.tensor(
            [0, lengths[0], lengths[0] + lengths[1], T],
            device=DEVICE,
            dtype=torch.int32,
        )
        chunk_indices = _chunk_indices(cu, chunk_size)
        num_sequences = len(lengths)
        state_indices = torch.tensor(
            [2, -1, 0], device=DEVICE, dtype=torch.int32
        )
    else:
        cu = None
        chunk_indices = None
        num_sequences = B
        state_indices = torch.tensor(
            [1], device=DEVICE, dtype=torch.int32
        )
    state_rows = num_sequences + 1

    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.05
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.05
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=False)
    state_before = (
        torch.randn(
            state_rows, H, V, K, device=DEVICE, dtype=torch.float32
        )
        * 0.01
    )
    state_actual = state_before.clone()
    # Keep the PyTorch reference on CPU.  HCU's rocBLAS installation does not
    # provide a Tensile library for every supported architecture, while the
    # reference's einsums do not need to run on the accelerator.
    reference_state = state_before.cpu()

    actual_h, actual_v, actual_final = chunk_h(
        k,
        w,
        u,
        g=g,
        initial_state=state_actual,
        initial_state_indices=state_indices,
        output_final_state=True,
        inplace_final_state=inplace_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=True,
        kernel_cfg=_kernel_cfg(backend_name, implementation),
    )
    expected_h, expected_v, expected_final = naive_chunk_h(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g=g.cpu(),
        initial_state=reference_state,
        initial_state_indices=state_indices.cpu(),
        output_final_state=True,
        inplace_final_state=inplace_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=None if cu is None else cu.cpu(),
        use_exp2=False,
        transpose_state_layout=True,
    )

    torch.testing.assert_close(
        actual_h.cpu(), expected_h, atol=BF16_ATOL, rtol=BF16_RTOL
    )
    assert actual_v is not None and expected_v is not None
    torch.testing.assert_close(
        actual_v.cpu(), expected_v, atol=BF16_ATOL, rtol=BF16_RTOL
    )
    assert actual_final is not None and expected_final is not None
    torch.testing.assert_close(
        actual_final.cpu(), expected_final, atol=BF16_ATOL, rtol=BF16_RTOL
    )
    if inplace_final_state:
        assert actual_final is state_actual
        assert expected_final is reference_state
        assert actual_final.shape[0] == state_rows
    else:
        assert actual_final is not state_actual
        torch.testing.assert_close(state_actual, state_before)
        assert actual_final.shape[0] == num_sequences


@GPU_REQUIRED
@pytest.mark.parametrize("use_varlen", (False, True), ids=("fixed", "varlen"))
@pytest.mark.parametrize(
    "backend_name,implementation",
    _chunk_h_correctness_params(),
)
@pytest.mark.parametrize(
    "inplace_final_state",
    (False, True),
    ids=("out-of-place", "in-place"),
)
def test_chunk_h_matches_reference(
    backend_name: str,
    implementation: str | None,
    use_varlen: bool,
    inplace_final_state: bool,
) -> None:
    _check_chunk_h_matches_reference(
        (backend_name, CHUNK_H_BACKENDS[backend_name]),
        use_varlen,
        inplace_final_state,
        implementation,
    )


def _check_chunk_o_matches_reference(
    chunk_o_backend: tuple[str, Callable],
    use_varlen: bool,
    transpose_state: bool,
) -> None:
    _, chunk_o = chunk_o_backend
    torch.manual_seed(121 if use_varlen else 124)
    B, H, Hg, K, V, chunk_size = 1, 16, 16, 128, 128, 64
    T = CORRECTNESS_SEQ_LEN
    dtype = torch.bfloat16
    if use_varlen:
        first = T // 3
        second = (T - first) // 2
        lengths = [first, second, T - first - second]
        cu = torch.tensor(
            [0, lengths[0], lengths[0] + lengths[1], T],
            device=DEVICE,
            dtype=torch.int32,
        )
        chunk_indices = _chunk_indices(cu, chunk_size)
        num_chunks = len(chunk_indices)
    else:
        cu = None
        chunk_indices = None
        num_chunks = math.ceil(T / chunk_size)

    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    value = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.05
    trailing = (H, V, K) if transpose_state else (H, K, V)
    h = torch.randn(
        B, num_chunks, *trailing, device=DEVICE, dtype=dtype
    ) * 0.05
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=False)

    actual = chunk_o(
        q,
        k,
        value,
        h,
        g=g,
        scale=K**-0.5,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=transpose_state,
    )
    expected = naive_chunk_o(
        q.cpu(),
        k.cpu(),
        value.cpu(),
        h=h.cpu(),
        g=g.cpu(),
        scale=K**-0.5,
        cu_seqlens=None if cu is None else cu.cpu(),
        chunk_size=chunk_size,
        use_exp2=False,
        transpose_state_layout=transpose_state,
    )
    torch.testing.assert_close(
        actual.cpu(), expected, atol=BF16_ATOL, rtol=BF16_RTOL
    )


@GPU_REQUIRED
@pytest.mark.parametrize("use_varlen", (False, True), ids=("fixed", "varlen"))
@pytest.mark.parametrize(
    "transpose_state", (True, False), ids=("vk", "kv")
)
def test_chunk_o_matches_reference(
    chunk_o_backend: tuple[str, Callable],
    use_varlen: bool,
    transpose_state: bool,
) -> None:
    _check_chunk_o_matches_reference(
        chunk_o_backend,
        use_varlen,
        transpose_state,
    )


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


def _benchmark_sequences(
    total: int,
    chunk_size: int,
    use_varlen: bool,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
    if not use_varlen:
        return [total], None, None
    first = total // 3
    second = (total - first) // 2
    lengths = [first, second, total - first - second]
    cu_seqlens = torch.tensor(
        [0, lengths[0], lengths[0] + lengths[1], total],
        device=DEVICE,
        dtype=torch.int32,
    )
    return lengths, cu_seqlens, _chunk_indices(cu_seqlens, chunk_size)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
@pytest.mark.parametrize(
    "transpose_state", [True, False], ids=["vk", "kv"]
)
def test_bench_chunk_h_tflops(
    chunk_h_backend: tuple[str, Callable],
    use_varlen: bool,
    transpose_state: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
    bench_seq_len: int | None,
) -> None:
    backend_name, chunk_h = chunk_h_backend
    if backend_name == "tilelang" and not transpose_state:
        pytest.skip("TileLang benchmark supports VK state layout only")
    torch.manual_seed(122)
    B, H, Hg, K, V, chunk_size = 1, 16, 16, 128, 128, 64
    default_T = 13312
    T = default_T if bench_seq_len is None else bench_seq_len
    if T <= 0:
        raise ValueError(f"--bench-t must be positive, got {T}")
    dtype = torch.bfloat16
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.05
    w = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype) * 0.05
    u = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.05
    lengths, cu, chunk_indices = _benchmark_sequences(
        T, chunk_size, use_varlen
    )
    state_rows = len(lengths) if use_varlen else B

    mode = "varlen" if use_varlen else "fixed"
    state_layout = "VK" if transpose_state else "KV"
    print(
        "GDN chunk h benchmark case: "
        f"backend={backend_name}, mode={mode}, state_layout={state_layout}, "
        f"B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, "
        f"chunk_size={chunk_size}, dtype={dtype}, device={DEVICE}"
    )
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=False)
    trailing = (H, V, K) if transpose_state else (H, K, V)
    state = torch.zeros(
        state_rows, *trailing, device=DEVICE, dtype=torch.float32
    )
    state_indices = torch.arange(state_rows, device=DEVICE, dtype=torch.int32)

    def run_h():
        return chunk_h(
            k, w, u, g=g, initial_state=state,
            initial_state_indices=state_indices,
            output_final_state=True, inplace_final_state=False,
            chunk_size=chunk_size, save_new_value=True,
            cu_seqlens=cu, chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state,
            kernel_cfg=_kernel_cfg(backend_name),
        )

    h, v_new, final = run_h()
    assert v_new is not None and final is not None
    assert torch.isfinite(h).all()
    assert torch.isfinite(v_new).all()
    assert torch.isfinite(final).all()
    actual_h = h
    actual_v_new = v_new

    h_ms = time_ms(run_h, bench_warmup, bench_repeat)
    h_flops = _chunk_h_flops(lengths, H, K, V, chunk_size)
    label = f"{backend_name}, {mode}, {state_layout}"
    h_tflops = h_flops / (h_ms * 1e-3) / 1e12
    print(f"GDN chunk h ({label}): {h_ms:.3f} ms, {h_tflops:.2f} TFLOPS")

    # AITER's kernel has the same VK state contract. Keep the reference in the
    # TileLang VK cases so both implementations use identical fixed/varlen
    # inputs and the same device selected above.
    if backend_name != "tilelang" or not transpose_state:
        return

    try:
        from aiter.ops.fla import (
            chunk_gated_delta_rule_fwd_sglang_hip_blockdim64 as aiter_chunk_h,
        )
    except ImportError as exc:
        print(f"AITER chunk-h reference is unavailable: {exc}")
        return

    def run_aiter_h():
        return aiter_chunk_h(
            k, w, u, g=g, gk=None,
            initial_state=state, initial_state_indices=state_indices,
            output_final_state=True, chunk_size=chunk_size,
            save_new_value=True, cu_seqlens=cu, chunk_indices=chunk_indices,
            chunk_offsets=None, use_exp2=False, transpose_state_layout=True,
        )

    reference_h, reference_v_new = run_aiter_h()
    assert reference_v_new is not None
    assert (
        torch.isfinite(reference_h).all()
        and torch.isfinite(reference_v_new).all()
        and torch.isfinite(state).all()
    )
    torch.testing.assert_close(
        actual_h, reference_h, atol=BF16_ATOL, rtol=BF16_RTOL
    )
    torch.testing.assert_close(
        actual_v_new,
        reference_v_new,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    aiter_h_ms = _do_bench_ref(
        run_aiter_h, warmup=bench_warmup, rep=bench_repeat
    )
    print(
        "GDN chunk h "
        "(AITER HIP chunk_gated_delta_rule_fwd_sglang_hip_blockdim64, "
        f"{mode}, VK): {aiter_h_ms:.3f} ms, "
        f"{h_flops / (aiter_h_ms * 1e-3) / 1e12:.2f} TFLOPS"
    )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
@pytest.mark.parametrize(
    "transpose_state", [True, False], ids=["vk", "kv"]
)
def test_bench_chunk_o_tflops(
    chunk_o_backend: tuple[str, Callable],
    use_varlen: bool,
    transpose_state: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
    bench_seq_len: int | None,
) -> None:
    backend_name, chunk_o = chunk_o_backend
    torch.manual_seed(122)
    B, H, Hg, K, V, chunk_size = 1, 16, 16, 128, 128, 64
    default_T = 13312
    T = default_T if bench_seq_len is None else bench_seq_len
    if T <= 0:
        raise ValueError(f"--bench-t must be positive, got {T}")
    dtype = torch.bfloat16
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    value = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.05
    lengths, cu, chunk_indices = _benchmark_sequences(
        T, chunk_size, use_varlen
    )
    num_chunks = (
        len(chunk_indices)
        if chunk_indices is not None
        else math.ceil(T / chunk_size)
    )
    trailing = (H, V, K) if transpose_state else (H, K, V)
    h = torch.randn(
        B, num_chunks, *trailing, device=DEVICE, dtype=dtype
    ) * 0.05
    g = _make_chunk_cumsum(B, T, H, chunk_size, cu, use_exp2=False)

    mode = "varlen" if use_varlen else "fixed"
    state_layout = "VK" if transpose_state else "KV"
    print(
        "GDN chunk o benchmark case: "
        f"backend={backend_name}, mode={mode}, state_layout={state_layout}, "
        f"B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, "
        f"chunk_size={chunk_size}, dtype={dtype}, device={DEVICE}"
    )

    def run_o():
        return chunk_o(
            q,
            k,
            value,
            h,
            g=g,
            scale=K**-0.5,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state,
        )

    output = run_o()
    assert torch.isfinite(output).all()

    o_ms = time_ms(run_o, bench_warmup, bench_repeat)
    o_flops = _chunk_o_flops(lengths, H, K, V, chunk_size)
    o_tflops = o_flops / (o_ms * 1e-3) / 1e12
    label = f"{backend_name}, {mode}, {state_layout}"
    print(f"GDN chunk o ({label}): {o_ms:.3f} ms, {o_tflops:.2f} TFLOPS")
