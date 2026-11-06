# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Framework contracts do not imply additional executable GEMM implementations."""
import pytest
import torch

from boltops.gemm.descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec
from boltops.gemm.registry import KernelConfig, select_plan


def test_unscaled_dense_uses_existing_native_kernel():
    p = GemmProblem("dense", 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(torch.bfloat16), torch.float32, "gfx946", num_cus=48)
    assert p.unscaled
    assert select_plan(p).implementation.name == "triton.dense.native"


def test_batched_is_not_silently_reinterpreted_as_grouped():
    p = GemmProblem("batched", 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(torch.bfloat16), torch.float32, "gfx946", batch=2)
    with pytest.raises(NotImplementedError, match="no registered batched GEMM"):
        select_plan(p)


@pytest.mark.parametrize("mode", ["equal", "offsets", "masked", "indexed"])
def test_unimplemented_group_addressing_never_uses_dense(mode):
    p = GemmProblem("grouped", 32, 64, 64, OperandSpec(torch.bfloat16, strides=(1024, 64, 1)),
                    OperandSpec(torch.bfloat16, strides=(4096, 64, 1)), torch.bfloat16, "gfx946", groups=GroupSpec(mode, 2))
    if mode == "equal":
        assert select_plan(p).implementation.name == "triton.grouped.native16"
    else:
        with pytest.raises(NotImplementedError, match="no registered grouped GEMM"):
            select_plan(p)


@pytest.mark.parametrize("format,packing,dtype,factor", [
    ("int4", "awq", torch.int32, 8), ("fp4", "e2m1_low_high", torch.uint8, 2),
])
def test_packed_storage_never_uses_native_unpacked_kernel(format, packing, dtype, factor):
    p = GemmProblem("dense", 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(dtype, format, packing, -1, factor), torch.bfloat16, "gfx946",
                    b_scale=ScaleSpec(torch.bfloat16, (1, 32)))
    with pytest.raises(NotImplementedError, match="no registered dense GEMM"):
        select_plan(p)


def test_packing_must_be_explicit():
    with pytest.raises(ValueError, match="packed operands require"):
        OperandSpec(torch.uint8, format="fp4")


@pytest.mark.parametrize("m", [0, 32.0])
def test_tile_configuration_requires_integer_geometry(m):
    with pytest.raises(ValueError, match="positive integers"):
        KernelConfig(m, 64, 64)


def test_group_count_requires_an_integer():
    with pytest.raises(ValueError, match="grouped GEMM addressing contract"):
        GroupSpec("equal", 2.5)


def test_public_scaled_interface_is_unified():
    import boltops.gemm as gemm

    assert all(x in gemm.__all__ for x in ("gemm_scaled", "gemm_dense", "gemm_batched", "gemm_grouped"))
    for existing in ("gemm_a8w8_blockscale", "w8a8_triton_block_scaled_mm", "triton_scaled_mm"):
        assert existing not in gemm.__all__ and not hasattr(gemm, existing)
