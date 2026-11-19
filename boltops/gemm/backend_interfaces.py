# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Callable contracts for GEMM backends."""

from typing import Protocol

import torch

from boltops.utility.arch_info import DeviceInfo


class A16W16Backend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        bias: torch.Tensor | None, activation: str | None,
        device: DeviceInfo,
    ) -> None: ...


class BiasGemmBackend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        bias: torch.Tensor | None, device: DeviceInfo,
    ) -> None: ...


class ScaledGemmBackend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        a_scale: torch.Tensor, b_scale: torch.Tensor,
        bias: torch.Tensor | None, device: DeviceInfo,
    ) -> None: ...


class BlockScaledGemmBackend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        a_scale: torch.Tensor, b_scale: torch.Tensor,
        block_size: tuple[int, int], device: DeviceInfo,
    ) -> None: ...


class AWQGemmBackend(Protocol):
    def __call__(
        self, a: torch.Tensor, packed_b: torch.Tensor, out: torch.Tensor, /, *,
        b_scale: torch.Tensor, b_zero_points: torch.Tensor | None,
        group_size: int, device: DeviceInfo,
    ) -> None: ...


class MXFP4GemmBackend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        a_scale: torch.Tensor, b_scale: torch.Tensor, device: DeviceInfo,
    ) -> None: ...


class BatchedA16W4Backend(Protocol):
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, /, *,
        b_scale: torch.Tensor, out_scale: torch.Tensor | None,
        transpose_bm: bool, device: DeviceInfo,
    ) -> None: ...
