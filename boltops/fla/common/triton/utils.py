# Additional upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/vllm/third_party/flash_linear_attention/ops/index.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/vllm/third_party/flash_linear_attention/ops/utils.py
"""Small standalone helpers used by the vendored KDA Triton kernels."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl

FLA_CHUNK_SIZE = 64
is_amd = torch.version.hip is not None
is_gather_supported = hasattr(tl, "gather")


class _CurrentPlatform:
    """Subset of vLLM's platform API used by the upstream KDA wrapper."""

    @staticmethod
    def is_cuda() -> bool:
        return torch.cuda.is_available() and torch.version.hip is None

    @staticmethod
    def has_device_capability(capability: int) -> bool:
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor >= capability


current_platform = _CurrentPlatform()


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache recent results using tensor identity, matching the upstream helper."""
    cache_entries: list[tuple[tuple, dict, Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for index, (last_args, last_kwargs, last_result) in enumerate(cache_entries):
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(arg is last for arg, last in zip(args, last_args))
                and all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:index]
                    + cache_entries[index + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result
        result = fn(*args, **kwargs)
        cache_entries = (cache_entries + [(args, kwargs, result)])[-cache_size:]
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(num_chunks)
            for num_chunks in triton.cdiv(
                prepare_lens(cu_seqlens), chunk_size
            ).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(prepare_lens(cu_seqlens), chunk_size),
        ]
    ).cumsum(-1)


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None

else:
    gather = tl.gather
