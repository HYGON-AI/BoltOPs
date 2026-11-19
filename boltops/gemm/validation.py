# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Tensor checks shared by GEMM entry points and backends."""

import torch


def validate_input_tensors(device: torch.device, *tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and min(tensor.stride(), default=1) <= 0:
            raise ValueError("inputs must have positive strides")


def validate_output(
    out: torch.Tensor, shape: tuple[int, ...], device: torch.device,
    dtypes: tuple[torch.dtype, ...], *, contiguous: bool = False,
) -> None:
    if out.shape != shape or out.device != device:
        raise ValueError("preallocated output has incompatible shape or device")
    if out.dtype not in dtypes:
        raise TypeError(f"unsupported output dtype: {out.dtype}")
    if contiguous and not out.is_contiguous():
        raise ValueError("preallocated output must be contiguous for this kernel")
    if out.numel() and min(out.stride()) <= 0:
        raise ValueError("preallocated output must have positive strides")
