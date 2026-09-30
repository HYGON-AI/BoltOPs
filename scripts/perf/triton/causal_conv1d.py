"""Performance spec: triton causal_conv1d (SGLang + vLLM ports).

Times ``causal_conv1d_fn`` (extend / prefill) and ``causal_conv1d_update``
(decode) on the measured Kimi-K3 shapes across both ports; writes
``triton_causal_conv1d.csv``.

The shape tables and the useful-FLOPS formula mirror the bench shapes of
``test/test_generic_causal_conv1d.py`` -- keep them in sync.  ``x`` must
be channel-last (``stride(0) == 1``); a row-major ``x`` would defeat the
vec8 path and skew the numbers.

The vllm port with ``metadata=None`` copies ``query_start_loc`` back to
the CPU for host-side grid scheduling, so its fwd rows cannot be
graph-captured; the framework falls back to event timing for those rows
(sglang receives ``seq_lens_cpu`` and is fully capturable).
"""

from __future__ import annotations

from typing import Iterator

import torch

from ..bench import Case, KernelSpec, dtype_str

X_DTYPE = torch.bfloat16
WEIGHT_DTYPE = torch.float32
WIDTH = 4
DIM_PER_RANK = 1536  # TP=8: 12 heads x 128
DIM_DECODE = 3 * DIM_PER_RANK
MAX_DECODE_BATCH = 32


def _varlen_lengths(batch: int, seed: int = 7) -> tuple[int, ...]:
    rng = torch.Generator().manual_seed(seed)
    return tuple(int(v) for v in torch.randint(40, 250, (batch,), generator=rng))


# (name, dim, lengths) -- fwd / extend / prefill
FWD_SHAPES = (
    ("measured-single", DIM_PER_RANK, (46,)),
    ("varlen-b8", DIM_PER_RANK, (46, 91, 137, 182, 57, 203, 118, 76)),
    ("varlen-max-batch", DIM_PER_RANK, _varlen_lengths(MAX_DECODE_BATCH)),
    ("long-prefill", DIM_PER_RANK, (4096,)),
    ("full-model", 12288, (4156,)),
)
# (name, dim, batch, seqlen) -- decode update
UPDATE_SHAPES = (
    ("decode-b1-measured", DIM_DECODE, 1, 1),
    ("decode-b8", DIM_DECODE, 8, 1),
    ("decode-b32-max", DIM_DECODE, MAX_DECODE_BATCH, 1),
    ("decode-multi-token", DIM_DECODE, 8, 5),
)


def _fwd_case(name: str, dim: int, lengths: tuple[int, ...], port: str, common: dict) -> Case:
    from boltops.generic import causal_conv1d_fn
    from boltops.generic.triton.causal_conv1d.core import PAD_SLOT_ID

    state_len = WIDTH - 1
    cu_seqlen = sum(lengths)
    batch = len(lengths)
    # Production layout: channel-last, x.stride(0) == 1.
    x = torch.randn(cu_seqlen, dim, dtype=X_DTYPE, device="cuda").t()
    weight = torch.randn(dim, WIDTH, dtype=WEIGHT_DTYPE, device="cuda")
    pool = torch.randn(2 * batch, state_len, 3 * dim, dtype=X_DTYPE, device="cuda")
    conv_states = pool.transpose(-1, -2)[:, :dim]
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cache_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    has_initial_state = torch.ones(batch, device="cuda", dtype=torch.bool)

    kwargs = dict(
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        backend=port,
    )
    if port == "sglang":
        kwargs["seq_lens_cpu"] = list(lengths)

    def run() -> None:
        causal_conv1d_fn(x, weight, None, conv_states, query_start_loc, **kwargs)

    flops = 2.0 * dim * cu_seqlen * WIDTH
    return Case(
        label=f"fwd {name} {port}",
        params={
            **common,
            "op": "fwd",
            "port": port,
            "shape": name,
            "dim": dim,
            "width": WIDTH,
            "batch": batch,
            "tokens": cu_seqlen,
        },
        run=run,
        flops=flops,
    )


def _update_case(name: str, dim: int, batch: int, seqlen: int, port: str, common: dict) -> Case:
    from boltops.generic import causal_conv1d_update

    state_len = WIDTH - 1
    x = torch.randn(batch, dim, seqlen, dtype=X_DTYPE, device="cuda")
    weight = torch.randn(dim, WIDTH, dtype=WEIGHT_DTYPE, device="cuda")
    pool = torch.randn(batch, state_len, dim, dtype=X_DTYPE, device="cuda")
    conv_state = pool.transpose(-1, -2)

    def run() -> None:
        causal_conv1d_update(x, conv_state, weight, None, activation="silu", backend=port)

    flops = 2.0 * batch * dim * seqlen * WIDTH
    return Case(
        label=f"update {name} {port}",
        params={
            **common,
            "op": "update",
            "port": port,
            "shape": name,
            "dim": dim,
            "width": WIDTH,
            "batch": batch,
            "tokens": batch * seqlen,
        },
        run=run,
        flops=flops,
    )


def cases() -> Iterator[Case]:
    common = {
        "operator": "causal_conv1d",
        "dtype": dtype_str(X_DTYPE),
    }
    for port in ("sglang", "vllm"):
        for name, dim, lengths in FWD_SHAPES:
            yield _fwd_case(name, dim, lengths, port, common)
        for name, dim, batch, seqlen in UPDATE_SHAPES:
            yield _update_case(name, dim, batch, seqlen, port, common)


SPEC = KernelSpec(
    backend="triton",
    kernel="causal_conv1d",
    fieldnames=(
        "operator", "op", "port", "shape", "dim", "width",
        "batch", "tokens",
        "dtype", "triton_version", "arch",
        "latency_ms", "tflops",
    ),
    cases=cases,
    warmup_ms=10,
    repeat_ms=30,
)
