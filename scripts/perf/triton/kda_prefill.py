"""Performance spec: triton KDA prefill (fused-gate mainline).

Times the end-to-end KDA prefill entry ``chunk_kda_with_fused_gate_fwd``
with the shipped configs on representative shapes (fixed grid, one
varlen case, one B=4 case); writes ``triton_kda_prefill.csv``.

The shape grid mirrors the ``SHAPE_PRESETS`` of
``scripts/tune/triton/tune_kda_prefill.py`` -- keep them in sync.
FLOPS is the approximate matmul-equivalent work of the linear-attention
mainline (chunk size BT=64):

  4 * H * (T/BT) * K * V   # inter-chunk state propagation
+ 2 * H * T * (K + V)      # intra-chunk qk / pv

``prepare_chunk_indices`` (varlen only) syncs to the host, but the helper
is cached by tensor identity: the spec builds ``cu_seqlens`` once and
reuses it across warmup/capture, so the sync stays in warmup and the
varlen case is graph-captured like the rest. If the cache ever misses,
the framework falls back to event timing automatically.
"""

from __future__ import annotations

from typing import Iterator

import torch

from ..bench import Case, KernelSpec, dtype_str

DTYPE = torch.bfloat16
CHUNK_SIZE = 64
# (shape, B, total tokens, H, K, V, varlen)
SHAPES = (
    ("t512", 1, 512, 24, 128, 128, False),
    ("t2048", 1, 2048, 24, 128, 128, False),
    ("t4096", 1, 4096, 24, 128, 128, False),
    ("t8192", 1, 8192, 24, 128, 128, False),
    ("t16384", 1, 16384, 24, 128, 128, False),
    ("varlen_t8192", 1, 8192, 24, 128, 128, True),
    ("batch4_t2048", 4, 2048, 24, 128, 128, False),
)


def _case(
    name: str, B: int, total: int, H: int, K: int, V: int, varlen: bool,
    common: dict,
) -> Case:
    from boltops.fla.kda import chunk_kda_with_fused_gate_fwd

    if varlen:
        split = max(CHUNK_SIZE, (total // 2 // CHUNK_SIZE) * CHUNK_SIZE)
        lengths = [length for length in (split, total - split) if length > 0]
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda", dtype=torch.int32,
        )
        n_sequences = len(lengths)
    else:
        lengths, cu, n_sequences = [total // B] * B, None, B
    seq_len = total // B

    q = torch.randn(B, total, H, K, device="cuda", dtype=DTYPE) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(B, total, H, V, device="cuda", dtype=DTYPE) * 0.2
    raw_g = -torch.rand(B, total, H, K, device="cuda", dtype=DTYPE) * 0.2
    raw_beta = torch.rand(B, total, H, device="cuda", dtype=DTYPE)
    a_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.2
    initial = (
        torch.randn(n_sequences, H, V, K, device="cuda", dtype=torch.float32)
        * 0.05
    )

    def run() -> None:
        chunk_kda_with_fused_gate_fwd(
            q=q, k=k, v=v, raw_g=raw_g,
            raw_beta=raw_beta, A_log=a_log, g_bias=None,
            scale=K ** -0.5, initial_state=initial,
            output_final_state=True, lower_bound=None,
            cu_seqlens=cu,
        )

    # Approximate linear-attention matmul work over the total tokens.
    flops = 4.0 * H * (total / CHUNK_SIZE) * K * V + 2.0 * H * total * (K + V)
    return Case(
        label=f"{name} B={B} T={seq_len} H={H} varlen={varlen}",
        params={
            **common,
            "shape": name,
            "B": B,
            "T": seq_len,
            "H": H,
            "K": K,
            "V": V,
            "varlen": varlen,
        },
        run=run,
        flops=flops,
    )


def cases() -> Iterator[Case]:
    common = {
        "operator": "kda_prefill",
        "dtype": dtype_str(DTYPE),
    }
    for spec in SHAPES:
        yield _case(*spec, common=common)


SPEC = KernelSpec(
    backend="triton",
    kernel="kda_prefill",
    fieldnames=(
        "operator", "shape", "B", "T", "H", "K", "V", "varlen",
        "dtype", "triton_version", "arch", "latency_ms", "tflops",
    ),
    cases=cases,
    warmup_ms=10,
    repeat_ms=30,
)
