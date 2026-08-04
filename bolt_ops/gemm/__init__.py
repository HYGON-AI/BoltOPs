"""GEMM operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.

Public APIs (aligned with vLLM / aiter call sites); docs/tests are per-family
like ``fused_moe`` / ``moe_activation``:

* docs: ``docs/ops/gemm.md``
* tests: ``test/gemm/``

Public operators:

* ``matmul_persistent``
* ``gemm_a16w16``
* ``w8a8_triton_block_scaled_mm``
* ``triton_scaled_mm``
* ``awq_gemm_triton``
* ``gemm_a8w8_blockscale``
* ``gemm_afp4wfp4``
* ``batched_gemm_a16wfp4``
* ``w8a8_block_int8_matmul``

Internal-only (not exported): ``gemm_fp16`` (HCU WASP/WDRA).
"""

from __future__ import annotations

from typing import Optional

import torch


def matmul_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """vLLM Triton persistent GEMM: ``C = A @ B [+ bias]`` (NN)."""
    from bolt_ops.gemm.triton.matmul_persistent import matmul_persistent as impl

    return impl(a, b, bias)


def gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: bool = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """aiter / vLLM ROCm Triton GEMM: ``Y = X @ W^T`` with ``W[N, K]``."""
    from bolt_ops.gemm.triton.gemm_a16w16 import gemm_a16w16 as impl

    return impl(
        x,
        w,
        bias=bias,
        dtype=dtype,
        y=y,
        config=config,
        activation=activation,
        skip_reduce=skip_reduce,
        kernel_type=kernel_type,
        backend=backend,
    )


def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """vLLM Triton W8A8 / FP8 block-scaled matmul."""
    from bolt_ops.gemm.triton.w8a8_triton_block_scaled_mm import (
        w8a8_triton_block_scaled_mm as impl,
    )

    return impl(A, B, As, Bs, block_size, output_dtype)


def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype] | torch.dtype,
    bias: Optional[torch.Tensor] = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """vLLM compressed-tensors Triton scaled matmul."""
    from bolt_ops.gemm.triton.triton_scaled_mm import triton_scaled_mm as impl

    return impl(
        input,
        weight,
        scale_a,
        scale_b,
        out_dtype,
        bias=bias,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        use_heuristic=use_heuristic,
    )


def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    config=None,
) -> torch.Tensor:
    """vLLM Triton AWQ GEMM (int4 packed weights)."""
    from bolt_ops.gemm.triton.awq_gemm_triton import awq_gemm_triton as impl

    return impl(input, qweight, scales, qzeros, split_k_iters, config)


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: bool = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """aiter Triton A8W8 block-scale GEMM: ``Y = X @ W^T``."""
    from bolt_ops.gemm.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale as impl

    return impl(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        y=y,
        config=config,
        skip_reduce=skip_reduce,
        kernel_type=kernel_type,
        backend=backend,
    )


def gemm_afp4wfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: bool = False,
) -> torch.Tensor:
    """aiter Triton MXFP4 GEMM: ``Y = X @ W^T`` (requires FP4-capable GPU)."""
    from bolt_ops.gemm.triton.gemm_afp4wfp4 import gemm_afp4wfp4 as impl

    return impl(
        x,
        w,
        x_scales,
        w_scales,
        dtype=dtype,
        y=y,
        config=config,
        skip_reduce=skip_reduce,
    )


def batched_gemm_a16wfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    transpose_bm: bool = False,
    prequant: bool = True,
    y_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """aiter Triton batched A16WFP4 GEMM (requires FP4-capable GPU)."""
    from bolt_ops.gemm.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4 as impl

    return impl(
        x,
        w,
        w_scales,
        dtype=dtype,
        y=y,
        config=config,
        transpose_bm=transpose_bm,
        prequant=prequant,
        y_scale=y_scale,
    )


def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """SGLang Triton W8A8 block-scaled INT8 matmul."""
    from bolt_ops.gemm.triton.w8a8_block_int8_matmul import (
        w8a8_block_int8_matmul as impl,
    )

    return impl(A, B, As, Bs, block_size, output_dtype)


__all__ = [
    "awq_gemm_triton",
    "batched_gemm_a16wfp4",
    "gemm_a16w16",
    "gemm_a8w8_blockscale",
    "gemm_afp4wfp4",
    "matmul_persistent",
    "triton_scaled_mm",
    "w8a8_block_int8_matmul",
    "w8a8_triton_block_scaled_mm",
]
