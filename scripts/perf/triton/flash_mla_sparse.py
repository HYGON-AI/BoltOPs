"""Performance spec: triton flash_mla_sparse.

Times the FlashMLA sparse forward on representative vLLM shapes with
both the baseline and the shipped tuned config (same inputs per shape,
so the rows are comparable); writes ``triton_flash_mla_sparse.csv``.
"""

from __future__ import annotations

import json
import math
from typing import Iterator

import torch
import triton

from ..bench import Case, KernelSpec

DTYPE = torch.bfloat16
# (query tokens, KV tokens, query heads, KV heads, top-k, d_qk, d_v)
# Last two rows track the real-model sizes recorded in
# docs/ops/mla_sparse_performance.md (chunked prefill + long context).
SHAPES = (
    (1, 16384, 16, 1, 2048, 576, 512),
    (4, 16384, 16, 1, 2048, 576, 512),
    (32, 16384, 16, 1, 2048, 576, 512),
    (4, 32768, 16, 1, 4096, 576, 512),
    (128, 16384, 16, 1, 2048, 576, 512),
    (32, 131072, 16, 1, 16384, 576, 512),
)
BASELINE_CONFIG = {
    "BLOCK_N": 16,
    "BLOCK_H": 16,
    "BLOCK_DMODEL": 128,
    "num_warps": 4,
    "num_stages": 1,
}


def _case(
    shape: tuple[int, ...],
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mode: str,
    config: dict,
    common: dict,
) -> Case:
    from boltops.mla import flash_mla_sparse_fwd

    q, kv, indices, topk_length = inputs
    s_q, s_kv, h_q, h_kv, topk, d_qk, d_v = shape
    softmax_scale = 1.0 / math.sqrt(d_qk)

    def run() -> None:
        flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            softmax_scale=softmax_scale,
            d_v=d_v,
            topk_length=topk_length,
            config=config,
        )

    # QK and PV each perform one multiply-add per selected value.
    flops = 2.0 * s_q * h_q * topk * (d_qk + d_v)
    return Case(
        label=f"{mode} Sq={s_q} Skv={s_kv} Hq={h_q} topk={topk}",
        params={
            **common,
            "mode": mode,
            "s_q": s_q,
            "s_kv": s_kv,
            "h_q": h_q,
            "h_kv": h_kv,
            "topk": topk,
            "d_qk": d_qk,
            "d_v": d_v,
            "config": json.dumps(config, sort_keys=True),
        },
        run=run,
        flops=flops,
    )


def cases() -> Iterator[Case]:
    from boltops.mla.triton.flash_mla_sparse import (
        _device_identity,
        _get_tuned_config,
    )

    arch, device_name = _device_identity(torch.device("cuda"))
    common = {
        "backend": "triton",
        "operator": "flash_mla_sparse",
        "triton_version": triton.__version__,
        "device_name": device_name,
        "arch": arch,
        "dtype": str(DTYPE).replace("torch.", ""),
    }
    for shape in SHAPES:
        s_q, s_kv, h_q, h_kv, topk, d_qk, d_v = shape
        # One input set per shape, shared by baseline and tuned.
        inputs = (
            torch.randn((s_q, h_q, d_qk), device="cuda", dtype=DTYPE),
            torch.randn((s_kv, h_kv, d_qk), device="cuda", dtype=DTYPE),
            torch.randint(
                0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
            ),
            torch.full((s_q,), topk, device="cuda", dtype=torch.int32),
        )
        tuned_config = _get_tuned_config(*inputs[:3], d_v)
        for mode, config in (
            ("baseline", BASELINE_CONFIG),
            ("tuned", tuned_config),
        ):
            yield _case(shape, inputs, mode, config, common)


SPEC = KernelSpec(
    backend="triton",
    kernel="flash_mla_sparse",
    fieldnames=(
        "backend", "operator", "mode", "triton_version", "device_name",
        "arch", "s_q", "s_kv", "h_q", "h_kv", "topk", "d_qk", "d_v",
        "dtype", "latency_ms", "tflops", "config",
    ),
    cases=cases,
    warmup_ms=100,
    # Graph replay budget; 300 ms keeps >=300 unrolled iterations even for
    # the fastest case, which is plenty for a stable median.
    repeat_ms=300,
)
