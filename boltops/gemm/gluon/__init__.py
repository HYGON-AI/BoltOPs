# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Gluon capability checks and on-demand backend loading."""

from functools import partial

import torch

from boltops.utility.arch_info import DeviceInfo
from ..backend_interfaces import BlockScaledGemmBackend
from ..config import get_gemm_config


def supports_a8w8_blockscale() -> bool:
    try:
        from triton.experimental.gluon.language.amd.hcu import packed_scale_fma
    except ImportError:
        return False
    return callable(packed_scale_fma)


def resolve_a8w8_blockscale_backend(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *,
    a_scale: torch.Tensor, b_scale: torch.Tensor, block_size: tuple[int, int],
    device: DeviceInfo, automatic: bool,
) -> BlockScaledGemmBackend | None:
    m, k = a.shape
    n = b.shape[0]
    bn, bk = block_size
    if not all(tensor.is_contiguous() for tensor in (a, b, a_scale, b_scale)):
        if automatic:
            return None
        raise NotImplementedError(
            "the Gluon blockscale pipeline requires contiguous operands and scales"
        )
    dtypes = "-".join(
        f"{name}={str(tensor.dtype).removeprefix('torch.')}"
        for name, tensor in (("DTYPE", a), ("AS", a_scale), ("BS", b_scale), ("OUT", out))
    )
    profile = f"{dtypes}-RECIPE_A=1x{bk}-RECIPE_B={bn}x{bk}"
    try:
        default, tuned = get_gemm_config(
            "GEMM-A8W8_BLOCKSCALE", m, n, k, profile=profile,
            backend="gluon", arch=device.arch, cu_num=device.cu_count,
        )
    except FileNotFoundError:
        if automatic:
            return None
        raise
    except KeyError:
        if automatic:
            return None
        raise NotImplementedError(
            "gemm_a8w8_blockscale: no Gluon configuration for the requested precision and scale recipe"
        ) from None
    if automatic and not tuned:
        return None
    if not supports_a8w8_blockscale():
        if automatic:
            return None
        raise NotImplementedError(
            "the Gluon blockscale pipeline requires Triton HCU packed_scale_fma support"
        )

    from .gemm_a8w8_blockscale import gemm_a8w8_blockscale

    return partial(gemm_a8w8_blockscale, config=default)
