"""Performance spec: triton KDA decode (dense + packed recurrent).

Times the two production decode entries on the KDA geometry
(H=24, K=V=128, BF16 input / FP32 state):

- ``fused_recurrent_kda`` -- dense wrapper with the gate fused, N
  independent one-token sequences (vLLM ABI: ``null_state_index=0``,
  so slot 0 of the state pool is the null slot and live state sits on
  1-based rows);
- ``fused_recurrent_kda_packed_decode`` -- packed QKV, one row per
  token;
- ``fused_recurrent_kda_fwd`` -- dense, pre-gated (SGLang path): gate
  and beta are materialized once via ``fused_kda_gate`` / sigmoid
  outside the timed call.

The N grid mirrors the decode evidence table of
``docs/ops/fla/kda/kda_decode.md`` (spec batch tokens=6, large batches
N=32/64) -- keep them in sync.  FLOPS approximates the two per-token
matmuls (output ``q @ state`` plus the ``v x k`` state update):

  4 * H * K * V * N

The state pool is updated in place, which is the real decode-loop
semantics (each iteration reads/writes the same rows).
"""

from __future__ import annotations

from typing import Iterator

import torch
import triton

from ..bench import Case, KernelSpec

DTYPE = torch.bfloat16
HEADS = 24
HEAD_DIM = 128
# N = number of independent one-token sequences (decode) / tokens (packed)
SHAPES = (
    ("t1", 1),
    ("t6", 6),
    ("t32", 32),
    ("t64", 64),
)


def _dense_case(name: str, N: int, common: dict) -> Case:
    from boltops.fla.kda import fused_recurrent_kda

    H, K, V = HEADS, HEAD_DIM, HEAD_DIM
    q = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, N, H, V, device="cuda", dtype=DTYPE) * 0.1
    raw_g = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    raw_beta = torch.randn(1, N, H, device="cuda", dtype=DTYPE) * 0.1
    A_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device="cuda", dtype=torch.float32) * 0.1
    state = torch.zeros(N + 1, H, V, K, device="cuda", dtype=torch.float32)
    state[1:] = torch.randn(N, H, V, K, device="cuda", dtype=torch.float32) * 0.01
    cu = torch.arange(N + 1, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, N + 1, device="cuda", dtype=torch.int32)

    def run() -> None:
        fused_recurrent_kda(
            q, k, v, raw_g, raw_beta, A_log, dt_bias, None,
            state, cu, indices,
        )

    flops = 4.0 * H * K * V * N
    return Case(
        label=f"dense {name}",
        params={
            **common,
            "entry": "dense",
            "shape": name,
            "N": N,
            "H": H,
            "K": K,
            "V": V,
        },
        run=run,
        flops=flops,
    )


def _packed_case(name: str, N: int, common: dict) -> Case:
    from boltops.fla.kda import fused_recurrent_kda_packed_decode

    H, K, V = HEADS, HEAD_DIM, HEAD_DIM
    q = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, N, H, V, device="cuda", dtype=DTYPE) * 0.1
    mixed_qkv = torch.cat(
        (q[0].reshape(N, H * K), k[0].reshape(N, H * K),
         v[0].reshape(N, H * V)),
        dim=-1,
    ).contiguous()
    raw_g = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    raw_beta = torch.randn(1, N, H, device="cuda", dtype=DTYPE) * 0.1
    A_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device="cuda", dtype=torch.float32) * 0.1
    state = torch.zeros(N + 1, H, V, K, device="cuda", dtype=torch.float32)
    state[1:] = torch.randn(N, H, V, K, device="cuda", dtype=torch.float32) * 0.01
    indices = torch.arange(1, N + 1, device="cuda", dtype=torch.int32)

    def run() -> None:
        fused_recurrent_kda_packed_decode(
            mixed_qkv, raw_g, raw_beta, A_log, dt_bias, None,
            state, indices,
        )

    flops = 4.0 * H * K * V * N
    return Case(
        label=f"packed {name}",
        params={
            **common,
            "entry": "packed",
            "shape": name,
            "N": N,
            "H": H,
            "K": K,
            "V": V,
        },
        run=run,
        flops=flops,
    )


def _pregated_case(name: str, N: int, common: dict) -> Case:
    from boltops.fla.kda import fused_kda_gate, fused_recurrent_kda_fwd

    H, K, V = HEADS, HEAD_DIM, HEAD_DIM
    q = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, N, H, V, device="cuda", dtype=DTYPE) * 0.1
    raw_g = torch.randn(1, N, H, K, device="cuda", dtype=DTYPE) * 0.1
    raw_beta = torch.randn(1, N, H, device="cuda", dtype=DTYPE) * 0.1
    A_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, K, device="cuda", dtype=torch.float32) * 0.1
    gate = fused_kda_gate(
        raw_g.reshape(N, H * K), A_log, K, g_bias=dt_bias
    ).reshape(1, N, H, K)
    beta = raw_beta.float().sigmoid()
    state = torch.zeros(N + 1, H, V, K, device="cuda", dtype=torch.float32)
    state[1:] = torch.randn(N, H, V, K, device="cuda", dtype=torch.float32) * 0.01
    cu = torch.arange(N + 1, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, N + 1, device="cuda", dtype=torch.int32)

    def run() -> None:
        fused_recurrent_kda_fwd(
            q, k, v, gate, beta, initial_state=state,
            cu_seqlens=cu, ssm_state_indices=indices,
        )

    flops = 4.0 * H * K * V * N
    return Case(
        label=f"dense_pregated {name}",
        params={
            **common,
            "entry": "dense_pregated",
            "shape": name,
            "N": N,
            "H": H,
            "K": K,
            "V": V,
        },
        run=run,
        flops=flops,
    )


def cases() -> Iterator[Case]:
    from boltops.utility.arch_info import get_arch

    common = {
        "backend": "triton",
        "operator": "kda_decode",
        "dtype": str(DTYPE).replace("torch.", ""),
        "triton_version": triton.__version__,
        "arch": get_arch(),
    }
    for name, N in SHAPES:
        yield _dense_case(name, N, common)
        yield _packed_case(name, N, common)
        yield _pregated_case(name, N, common)


SPEC = KernelSpec(
    backend="triton",
    kernel="kda_decode",
    fieldnames=(
        "backend", "operator", "entry", "shape", "N", "H", "K", "V",
        "dtype", "triton_version", "arch", "latency_ms", "tflops",
    ),
    cases=cases,
    warmup_ms=10,
    repeat_ms=30,
)
