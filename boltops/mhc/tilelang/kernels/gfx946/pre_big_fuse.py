# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""gfx946 entry point for MHC pre-big-fuse.

Small shapes use the occupancy-3 common split-K kernel.  Larger shapes use the
persistent warp-specialized kernel selected by the gfx946 configuration policy.
"""

from boltops.mhc.tilelang.kernels.common.pre_big_fuse import (
    mhc_pre_big_fuse as _mhc_pre_big_fuse,
)
from boltops.mhc.tilelang.kernels.gfx946.pre_norm_splitk_wasp_persistent import (
    mhc_pre_gemm_sqrsum_splitk_persistent_kernel,
)


def mhc_pre_big_fuse(*args, **kwargs):
    residual = args[0] if args else kwargs["residual"]
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    if num_tokens > 128:
        kwargs.setdefault(
            "_splitk_kernel_factory",
            mhc_pre_gemm_sqrsum_splitk_persistent_kernel,
        )
    return _mhc_pre_big_fuse(*args, **kwargs)


__all__ = ["mhc_pre_big_fuse"]
