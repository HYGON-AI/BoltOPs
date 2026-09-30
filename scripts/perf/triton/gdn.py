"""Performance spec: triton GDN (chunk prefill e2e + packed decode).

GDN production entries, shapes mirrored from the bench cases and the
performance tables of ``docs/ops/fla/gdn.md`` and the ``test/fla``
bench tests -- keep them in sync:

- ``chunk_e2e``: the two public chunk stages chained the way the
  prefill pipeline runs them, ``chunk_gated_delta_rule_fwd_h`` followed
  by ``chunk_fwd_o`` (``inplace_final_state=False`` so the timed loop
  never mutates the initial state pool). FP16, ``transpose_state_layout``
  on, BT=64.  The caller-side w/u generation is the separate
  ``recompute_w_u`` entry below.
- ``recompute_w_u``: ``recompute_w_u_fwd`` (WY w/u generation, the
  stage ahead of the chunk pair), observed/derived real-model shapes
  from ``test_gdn_recompute_w_u.py`` (incl. the BT=256 variant).
- ``packed_decode``: ``fused_recurrent_gated_delta_rule_packed_decode``
  (SGLang ABI, ``null_state_index=-1`` so every pool row is live),
  BF16 input / FP32 state; the pool and the ``out`` buffer are updated
  in place, i.e. the real decode-loop semantics.
- ``sigmoid_update``: the vLLM decode operator
  ``fused_sigmoid_gating_delta_rule_update`` (decode T=1 and spec
  verify T=4, N=4, BF16 state per vLLM's ``mamba_ssm_cache_dtype
  =auto``), same in-place state semantics.

FLOPS: chunk stages use the same formulas as
``test/fla/test_gdn_chunk.py`` (``_chunk_h_flops`` / ``_chunk_o_flops``);
recompute_w_u uses the test's ``_bench_useful_flops``
(2 * H * L^2 * (K + V) per chunk); packed decode approximates the
per-token matmul work (4 * H * K * V * B).
"""

from __future__ import annotations

import math
from typing import Iterator

import torch

from ..bench import Case, KernelSpec, dtype_str

CHUNK_SIZE = 64
# (name, B, T, h_v, h_qk, K, V, dtype, lengths or None)
CHUNK_SHAPES = (
    ("fixed", 1, 13320, 16, 8, 128, 128, torch.float16, None),
    ("varlen", 1, 13320, 16, 8, 128, 128, torch.float16, (4097, 4911, 4312)),
)
DECODE_DTYPE = torch.bfloat16
DECODE_H, DECODE_HQK = 16, 4
DECODE_K = DECODE_V = 128
# (name, batch)
DECODE_SHAPES = (
    ("b1", 1),
    ("b8", 8),
    ("b32", 32),
    ("b64", 64),
)
SIGMOID_HQK, SIGMOID_HV = 4, 16
SIGMOID_K = SIGMOID_V = 128
# (name, N, steps) -- mirrors test_bench_sigmoid_update_bandwidth
SIGMOID_SHAPES = (
    ("n4_t1", 4, 1),
    ("n4_t4", 4, 4),
)

def _captured_lengths(
    T: int, num_sequences: int, NT: int, BT: int = 64,
) -> tuple[int, ...]:
    """Same helper as test_gdn_recompute_w_u._bench_captured_lengths."""
    remainder_sum = T - BT * (NT - num_sequences)
    chunks = [NT // num_sequences] * num_sequences
    for i in range(NT % num_sequences):
        chunks[i] += 1
    remainders = [remainder_sum // num_sequences] * num_sequences
    for i in range(remainder_sum % num_sequences):
        remainders[i] += 1
    return tuple(BT * (chunks[i] - 1) + remainders[i] for i in range(num_sequences))

# (name, B, T, h_qk, h_v, K, V, BT, lengths) -- mirrors the bench table of
# test/fla/test_gdn_recompute_w_u.py (real-model captured sizes)
WU_DTYPE = torch.bfloat16
WU_SHAPES = (
    ("observed-short", 1, 60, 8, 16, 128, 128, 64, (60,)),
    ("observed-common", 1, 4156, 8, 16, 128, 128, 64, (4156,)),
    ("observed-hot", 1, 4543, 8, 16, 128, 128, 64, (4543,)),
    ("observed-common-n2", 1, 8652, 8, 16, 128, 128, 64,
     _captured_lengths(8652, 2, 136)),
    ("observed-hot-n3", 1, 12617, 8, 16, 128, 128, 64,
     _captured_lengths(12617, 3, 199)),
    ("observed-max", 1, 13674, 8, 16, 128, 128, 64,
     _captured_lengths(13674, 3, 216)),
    ("derived-d64-hot", 1, 4543, 8, 16, 64, 64, 64, (4543,)),
    # derived-d256-hot (K=V=256, BT=256) is dropped: its selected tile
    # needs 128 KB shared memory, over the 64 KB gfx938 limit, and the
    # public entry exposes no config override.
)


def _chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    boundaries = cu_seqlens.detach().cpu().tolist()
    for seq, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        rows.extend((seq, chunk) for chunk in range(math.ceil((eos - bos) / chunk_size)))
    return torch.tensor(rows, device=cu_seqlens.device, dtype=torch.int32)


def _chunk_cumsum(
    batch: int, total: int, h_v: int, chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    pointwise = -torch.rand(batch, total, h_v, device="cuda", dtype=torch.float32) * 0.04
    result = torch.empty_like(pointwise)
    boundaries = [0, total] if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()
    for bos, eos in zip(boundaries[:-1], boundaries[1:]):
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            result[:, start:end] = pointwise[:, start:end].cumsum(1)
    return result


def _chunk_h_flops(lengths: list[int], h_v: int, K: int, V: int, cs: int) -> float:
    return float(sum(
        4 * h_v * min(cs, length - start) * K * V
        for length in lengths
        for start in range(0, length, cs)
    ))


def _chunk_o_flops(lengths: list[int], h_v: int, K: int, V: int, cs: int) -> float:
    total = 0
    for length in lengths:
        for start in range(0, length, cs):
            size = min(cs, length - start)
            total += 2 * h_v * size * K * V
            total += 2 * h_v * size * size * K
            total += 2 * h_v * size * size * V
    return float(total)


def _chunk_case(
    name: str, B: int, T: int, h_v: int, h_qk: int, K: int, V: int,
    dtype: torch.dtype, lengths: tuple[int, ...] | None, common: dict,
) -> Case:
    from boltops.fla.gdn import chunk_fwd_o, chunk_gated_delta_rule_fwd_h

    if lengths is None:
        lengths = (T,)
        cu: torch.Tensor | None = None
        chunk_indices: torch.Tensor | None = None
    else:
        assert sum(lengths) == T
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda", dtype=torch.int32,
        )
        chunk_indices = _chunk_indices(cu, CHUNK_SIZE)
    n_seqs = len(lengths)

    q = torch.randn(B, T, h_qk, K, device="cuda", dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    w = torch.randn(B, T, h_v, K, device="cuda", dtype=dtype) * 0.05
    u = torch.randn(B, T, h_v, V, device="cuda", dtype=dtype) * 0.05
    g = _chunk_cumsum(B, T, h_v, CHUNK_SIZE, cu)
    state = torch.zeros(n_seqs, h_v, V, K, device="cuda", dtype=torch.float32)
    state_indices = torch.arange(n_seqs, device="cuda", dtype=torch.int32)

    def run() -> None:
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k, w, u, g=g, initial_state=state,
            initial_state_indices=state_indices,
            output_final_state=True, inplace_final_state=False,
            chunk_size=CHUNK_SIZE, save_new_value=True,
            cu_seqlens=cu, chunk_indices=chunk_indices,
            transpose_state_layout=True,
        )
        chunk_fwd_o(
            q, k, v_new, h, g=g, scale=K ** -0.5,
            cu_seqlens=cu, chunk_size=CHUNK_SIZE,
            chunk_indices=chunk_indices, transpose_state_layout=True,
        )

    flops = _chunk_h_flops(list(lengths), h_v, K, V, CHUNK_SIZE) \
        + _chunk_o_flops(list(lengths), h_v, K, V, CHUNK_SIZE)
    return Case(
        label=f"chunk_e2e {name} T={T}",
        params={
            **common,
            "entry": "chunk_e2e",
            "shape": name,
            "tokens": T,
            "seqs": n_seqs,
            "h_qk": h_qk,
            "h_v": h_v,
            "K": K,
            "V": V,
            "dtype": dtype_str(dtype),
        },
        run=run,
        flops=flops,
    )


def _packed_case(name: str, B: int, common: dict) -> Case:
    from boltops.fla.gdn import fused_recurrent_gated_delta_rule_packed_decode

    h_v, h_qk, K, V = DECODE_H, DECODE_HQK, DECODE_K, DECODE_V
    mixed_qkv = torch.randn(
        B, 2 * h_qk * K + h_v * V, device="cuda", dtype=DECODE_DTYPE
    ) * 0.2
    a = torch.randn(B, h_v, device="cuda", dtype=DECODE_DTYPE) * 0.2
    b = torch.randn(B, h_v, device="cuda", dtype=DECODE_DTYPE) * 0.2
    A_log = torch.randn(h_v, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(h_v, device="cuda", dtype=DECODE_DTYPE) * 0.1
    state = torch.randn(B, h_v, V, K, device="cuda", dtype=torch.float32) * 0.02
    out = torch.empty(B, 1, h_v, V, device="cuda", dtype=DECODE_DTYPE)
    indices = torch.arange(B, device="cuda", dtype=torch.int32)

    def run() -> None:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv, a, b, A_log, dt_bias, K ** -0.5,
            state, out, indices,
        )

    flops = 4.0 * h_v * K * V * B
    return Case(
        label=f"packed_decode {name}",
        params={
            **common,
            "entry": "packed_decode",
            "shape": name,
            "tokens": B,
            "seqs": B,
            "h_qk": h_qk,
            "h_v": h_v,
            "K": K,
            "V": V,
            "dtype": dtype_str(DECODE_DTYPE),
        },
        run=run,
        flops=flops,
    )


def _sigmoid_update_case(name: str, N: int, steps: int, common: dict) -> Case:
    from boltops.fla.gdn import fused_sigmoid_gating_delta_rule_update

    H, HV, K, V = SIGMOID_HQK, SIGMOID_HV, SIGMOID_K, SIGMOID_V
    total = N * steps
    q = torch.randn(1, total, H, K, device="cuda", dtype=DECODE_DTYPE) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, total, HV, V, device="cuda", dtype=DECODE_DTYPE) * 0.05
    a = torch.zeros(1, total, HV, device="cuda", dtype=DECODE_DTYPE)
    b = torch.zeros_like(a)
    A_log = torch.zeros(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(HV, device="cuda", dtype=DECODE_DTYPE)
    state = torch.zeros(total, HV, V, K, device="cuda", dtype=DECODE_DTYPE)
    indices = torch.arange(total, device="cuda", dtype=torch.int32).view(N, steps)
    accepted = torch.ones(N, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, total + 1, steps, device="cuda", dtype=torch.int32)

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

    flops = 4.0 * HV * K * V * total
    return Case(
        label=f"sigmoid_update {name}",
        params={
            **common,
            "entry": "sigmoid_update",
            "shape": name,
            "tokens": total,
            "seqs": N,
            "h_qk": H,
            "h_v": HV,
            "K": K,
            "V": V,
            "dtype": dtype_str(DECODE_DTYPE),
        },
        run=run,
        flops=flops,
    )


def _wu_case(
    name: str, B: int, T: int, h_qk: int, h_v: int, K: int, V: int,
    BT: int, lengths: tuple[int, ...] | None, common: dict,
) -> Case:
    from boltops.fla.gdn import recompute_w_u_fwd

    if lengths is None:
        cu: torch.Tensor | None = None
        ranges = [(0, T)]
    else:
        assert sum(lengths) == T
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda", dtype=torch.int32,
        )
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        ranges = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]

    k = torch.randn(B, T, h_qk, K, device="cuda", dtype=WU_DTYPE)
    v = torch.randn(B, T, h_v, V, device="cuda", dtype=WU_DTYPE)
    beta = torch.sigmoid(torch.randn(B, T, h_v, device="cuda", dtype=torch.float32))
    g = torch.empty(B, T, h_v, device="cuda", dtype=torch.float32)
    A = torch.zeros(B, T, h_v, BT, device="cuda", dtype=WU_DTYPE)
    for bos, eos in ranges:
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            length = end - start
            lower = torch.tril(torch.randn(h_v, length, length, device="cuda") * 0.05) \
                + torch.eye(length, device="cuda").unsqueeze(0)
            A[:, start:end, :, :length] = lower.permute(1, 0, 2).to(WU_DTYPE)
            g[:, start:end] = (-torch.rand(length, h_v, device="cuda") * 0.05).cumsum(0)

    def run() -> None:
        recompute_w_u_fwd(k, v, beta, g, A, cu)

    flops = 0.0
    for bos, eos in ranges:
        for start in range(bos, eos, BT):
            flops += 2.0 * h_v * min(BT, eos - start) ** 2 * (K + V)
    return Case(
        label=f"recompute_w_u {name}",
        params={
            **common,
            "entry": "recompute_w_u",
            "shape": name,
            "tokens": T,
            "seqs": len(ranges),
            "h_qk": h_qk,
            "h_v": h_v,
            "K": K,
            "V": V,
            "dtype": dtype_str(WU_DTYPE),
        },
        run=run,
        flops=flops,
    )


def cases() -> Iterator[Case]:
    common = {
        "operator": "gdn",
    }
    for spec in CHUNK_SHAPES:
        yield _chunk_case(*spec, common=common)
    for name, B in DECODE_SHAPES:
        yield _packed_case(name, B, common)
    for name, N, steps in SIGMOID_SHAPES:
        yield _sigmoid_update_case(name, N, steps, common)
    for spec in WU_SHAPES:
        yield _wu_case(*spec, common=common)


SPEC = KernelSpec(
    backend="triton",
    kernel="gdn",
    fieldnames=(
        "operator", "entry", "shape", "tokens", "seqs",
        "h_qk", "h_v", "K", "V",
        "dtype", "triton_version", "arch",
        "latency_ms", "tflops",
    ),
    cases=cases,
    warmup_ms=10,
    repeat_ms=30,
)
