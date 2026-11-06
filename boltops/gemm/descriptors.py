# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""GEMM semantics shared across dense, batched and grouped implementations."""
from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScaleSpec:
    dtype: torch.dtype
    block: tuple[int, int]
    encoding: str = "numeric"


@dataclass(frozen=True)
class OperandSpec:
    storage_dtype: torch.dtype
    format: str = "native"
    packing: str | None = None
    pack_axis: int | None = None
    values_per_storage: int = 1
    zero_point_dtype: torch.dtype | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.format == "native":
            if (self.packing is not None or self.pack_axis is not None or
                    self.values_per_storage != 1 or self.zero_point_dtype is not None):
                raise ValueError("native operands cannot carry packed-format metadata")
        elif (self.format not in ("int4", "fp4") or not self.packing or
              self.pack_axis is None or self.values_per_storage <= 1):
            raise ValueError("packed operands require a format, packing name, axis and pack factor")


@dataclass(frozen=True)
class GroupSpec:
    # M denotes total flattened rows; equal groups each contain M/count rows.
    mode: str
    count: int

    def __post_init__(self):
        if (self.mode not in ("equal", "offsets", "masked", "indexed") or
                type(self.count) is not int or self.count <= 0):
            raise ValueError("invalid grouped GEMM addressing contract")


@dataclass(frozen=True)
class GemmProblem:
    kind: str
    m: int
    n: int
    k: int
    a: OperandSpec
    b: OperandSpec
    out_dtype: torch.dtype
    arch: str
    a_scale: ScaleSpec | None = None
    b_scale: ScaleSpec | None = None
    batch: int = 1
    groups: GroupSpec | None = None
    a_aligned_16: bool = True
    b_aligned_16: bool = True
    num_cus: int | None = None
    activation_quantization: str | None = None
    has_bias: bool = False

    def __post_init__(self):
        if self.num_cus is not None and (type(self.num_cus) is not int or self.num_cus <= 0):
            raise ValueError("num_cus must be a positive integer when specified")
        if self.kind not in ("dense", "batched", "grouped") or min(self.m, self.n, self.k) < 0:
            raise ValueError("invalid GEMM kind or shape")
        if self.batch <= 0 or (self.kind != "batched" and self.batch != 1):
            raise ValueError("batch count belongs only to batched GEMM")
        if (self.kind == "grouped") != (self.groups is not None):
            raise ValueError("group metadata belongs only to grouped GEMM")
        if self.groups and self.groups.mode == "equal" and self.m % self.groups.count:
            raise ValueError("equal groups must divide the total M rows")

    @property
    def interface(self):
        if self.kind == "batched" and self.activation_quantization == "mxfp4":
            return "batched_a16wfp4_online_quant"
        if self.kind in ("batched", "grouped"):
            return self.kind
        return "dense" if self.unscaled else "scaled"

    @property
    def a_dtype(self):
        return self.a.storage_dtype

    @property
    def b_dtype(self):
        return self.b.storage_dtype

    @property
    def native_operands(self):
        return self.a.format == self.b.format == "native"

    @property
    def unscaled(self):
        return self.a_scale is None and self.b_scale is None
