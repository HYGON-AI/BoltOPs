# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared validation helpers for the GDN TileLang backend."""

from __future__ import annotations

from typing import Optional

import torch


def tilelang_dtype(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int32: "int32",
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TileLang dtype: {dtype}")
    return mapping[dtype]


def require_contiguous(**tensors: Optional[torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the TileLang backend.")


def require_first_version(
    *,
    chunk_size: int,
    use_exp2: bool,
    transpose_state_layout: bool,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    kernel_cfg: Optional[dict],
) -> None:
    if chunk_size != 64:
        raise NotImplementedError(
            "the first TileLang version only supports chunk_size=64."
        )
    if use_exp2:
        raise NotImplementedError(
            "the first TileLang version only supports use_exp2=False."
        )
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError(
            "the first TileLang version only supports fixed sequences."
        )
    if kernel_cfg is not None:
        raise NotImplementedError(
            "kernel_cfg is not supported by the first TileLang version."
        )
