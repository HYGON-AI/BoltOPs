"""Performance spec: triton batched_gemm_bf16.

Times the shipped (tuned) config on both NT and TT layouts across the
configured size grid; writes ``triton_batched_gemm_bf16.csv``.
"""

from __future__ import annotations

from typing import Iterator

import torch
import triton

from ..bench import Case, KernelSpec

DTYPE = torch.float16
M_VALUES = (
    1, 8, 16, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384,
)
# (B, (M, ...), N, K)
SHAPES = (
    (16, M_VALUES, 1024, 4096),
    (16, M_VALUES, 8192, 1024),
    (16, M_VALUES, 1280, 8192),
    (56, (234,), 4928, 1232),
    (56, (234,), 1232, 2464),
)


def _case(B: int, M: int, N: int, K: int, layout: str, common: dict) -> Case:
    from boltops.gemm import batched_gemm_bf16
    from boltops.gemm.triton.batched_gemm_bf16 import _get_config

    x = torch.empty((B, M, K), device="cuda", dtype=DTYPE)
    if layout == "tn":
        weight = torch.empty((B, N, K), device="cuda", dtype=DTYPE)
    else:
        weight = torch.empty((B, K, N), device="cuda", dtype=DTYPE).transpose(1, 2)
    output = torch.empty((B, M, N), device="cuda", dtype=DTYPE)
    config = _get_config(M, N, K, DTYPE, B=B, layout=layout)
    config["BATCH_LAST_GRID"] = True

    def run() -> None:
        batched_gemm_bf16(x, weight, dtype=DTYPE, y=output, config=config)

    return Case(
        label=f"{layout} B={B} M={M} N={N} K={K}",
        params={
            **common,
            "layout": layout,
            "mode": "tuned",
            "B": B,
            "M": M,
            "N": N,
            "K": K,
            "dtype": str(DTYPE).replace("torch.", ""),
            "block_pingpong": config.get("use_block_pingpong", False),
            "batch_last_grid": config.get("BATCH_LAST_GRID", False),
        },
        run=run,
        flops=2.0 * B * M * N * K,
    )


def cases() -> Iterator[Case]:
    from boltops.utility.arch_info import get_arch

    common = {
        "backend": "triton",
        "operator": "batched_gemm_bf16",
        "dtype": str(DTYPE).replace("torch.", ""),
        "triton_version": triton.__version__,
        "arch": get_arch(),
    }
    for layout in ("tn", "tt"):
        for B, m_values, N, K in SHAPES:
            for M in m_values:
                yield _case(B, M, N, K, layout, common)


SPEC = KernelSpec(
    backend="triton",
    kernel="batched_gemm_bf16",
    fieldnames=(
        "backend", "operator", "layout", "mode", "B", "M", "N", "K",
        "dtype", "triton_version", "arch",
        "latency_ms", "tflops",
        "block_pingpong", "batch_last_grid",
    ),
    cases=cases,
    warmup_ms=10,
    repeat_ms=30,
)
