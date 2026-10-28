# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/gemm/basic/gemm_a16w16.py
"""Vendored aiter Triton ``gemm_a16w16`` (FP16/BF16).

Upstream: ROCm/aiter ``aiter.ops.triton.gemm.basic.gemm_a16w16``.
vLLM call site: ``from aiter.ops.triton.gemm_a16w16 import gemm_a16w16`` then
``gemm_a16w16(x, weight, bias)`` with ``weight`` shape ``[N, K]`` (computes
``Y = X @ W^T``).

Gluon backend and ``torch_compile_guard`` are not vendored; this wrapper always
uses the Triton kernel path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from boltops.gemm.triton.activation import (
    _get_activation_from_str,
)
from boltops.gemm.triton.gemm_a16w16_op import (
    _gemm_a16_w16_kernel,
)
from boltops.gemm.triton.gemm_a16w16_op import (
    _get_config as _get_triton_config,
)
from boltops.gemm.triton.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)


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
    """16-bit matmul ``Y = X @ W^T`` (aiter / vLLM ROCm Triton path).

    Args:
        x: ``[M, K]`` FP16/BF16
        w: ``[N, K]`` FP16/BF16 (internally transposed)
        bias: optional ``[N]``
        dtype: output dtype; defaults to ``x.dtype``
        y: optional preallocated ``[M, N]``
        config: optional launch config dict (else JSON table lookup)
        activation: optional epilogue activation name
        skip_reduce: if True and split-K, return partials ``[NUM_KSPLIT, M, N]``
        kernel_type / backend: accepted for API compatibility; Triton-only here
    """
    del kernel_type  # gluon-only upstream
    if backend is not None and backend.lower() not in ("triton",):
        raise ValueError(
            f"boltops vendored gemm_a16w16 only supports backend='triton'; "
            f"got {backend!r}"
        )

    if dtype is None or not isinstance(dtype, torch.dtype):
        dtype = x.dtype

    assert x.dtype in (torch.float16, torch.bfloat16), (
        f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    )
    assert w.dtype in (torch.float16, torch.bfloat16), (
        f"Weights (w) must be fp16 or bf16, got {w.dtype}"
    )
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, _ = w.shape
    w_t = w.T

    if config is None:
        config, _ = _get_triton_config(M, N, K)
    else:
        config = dict(config)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        w_t,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=(bias is not None),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            KERNEL_NAME="_gemm_a16w16_reduce_kernel",
        )

    return y


__all__ = ["gemm_a16w16"]
