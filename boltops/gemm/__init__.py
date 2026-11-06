# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""GEMM entry points grouped by computation, independently of backend kernels."""

__all__ = ["gemm_dense", "gemm_scaled", "gemm_batched", "gemm_grouped",
           "gemm_batched_a16wfp4_online_quant"]


def gemm_scaled(a, b, a_scale=None, b_scale=None, *, scale_recipe=((1, 128), (128, 128)),
                      scale_encodings=("numeric", "numeric"), out_dtype=None, a_format="native", b_format="native",
                      b_zero_points=None, bias=None, backend="auto"):
    """Native dense GEMM with independent scale blocks/encodings and selectable output dtype."""
    from boltops.gemm.scaled import gemm_scaled as impl
    return impl(a, b, a_scale, b_scale, scale_recipe=scale_recipe,
                scale_encodings=scale_encodings, out_dtype=out_dtype, a_format=a_format, b_format=b_format,
                b_zero_points=b_zero_points, bias=bias, backend=backend)


def gemm_dense(a, b, *, bias=None, out_dtype=None, backend="auto"):
    """Unscaled A[M,K] @ B[N,K].T with optional bias[N]."""
    from boltops.gemm.dense import gemm_dense as impl
    return impl(a, b, bias=bias, out_dtype=out_dtype, backend=backend)


def gemm_grouped(a, b, *, group_offsets=None, a_scale=None, b_scale=None,
                 scale_recipe=((1, 128), (1, 32)),
                 scale_encodings=("numeric", "numeric"), b_format="native",
                 bias=None, out_dtype=None, backend="auto"):
    """Equal or offset-grouped GEMM; scale and packed format are explicit."""
    from boltops.gemm.grouped import gemm_grouped as impl
    return impl(a, b, group_offsets=group_offsets, a_scale=a_scale, b_scale=b_scale,
                scale_recipe=scale_recipe, scale_encodings=scale_encodings,
                b_format=b_format, bias=bias, out_dtype=out_dtype, backend=backend)


def gemm_batched(a, b, *, bias=None, out_dtype=None, backend="auto"):
    """Native batched A[B,M,K] @ B[B,N,K].T."""
    from .batched import gemm_batched as impl
    return impl(a, b, bias=bias, out_dtype=out_dtype, backend=backend)


def gemm_batched_a16wfp4_online_quant(a, b, b_scale, *, out_dtype=None, backend="auto"):
    """Batched A16WFP4 with the existing online MXFP4 activation quantization."""
    from .specialized import gemm_batched_a16wfp4_online_quant as impl
    return impl(a, b, b_scale, out_dtype=out_dtype, backend=backend)
