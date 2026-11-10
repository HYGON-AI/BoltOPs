# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""AWQ packed INT4 GEMM with explicit group size."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_a16w4(
    x, w, w_scales, w_zeros=None, dtype=None, out=None, config=None, backend=None
):
    """Compute A16 times AWQ INT4 weights with group scales and zero points.

    The logical weight is [K,N]; unpack its INT4 values and zero points
    using the existing AWQ nibble order, subtract zero points, multiply
    by the corresponding K-group scale, then compute ``x @ weight``.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [M,K]. Any positive matrix
            strides are supported, including transpose, padded and sliced views.
        w (torch.Tensor): int32 GPU weights [K,N/8] on x.device, with positive stored
            word strides. Each integer packs eight unsigned INT4 values in AWQ
            order along logical N. The carrier tensor may be transposed or
            strided physically; the logical packing remains AWQ [K,N/8]. This
            is distinct from uint8 E2M1 FP4 storage.
        w_scales (torch.Tensor): [K/G,N] GPU scales, same dtype/device as x, with positive
            matrix strides. G is inferred from the first dimension and must
            divide K, with G in {32,64,128,K}; K and N must be positive.
        w_zeros (torch.Tensor | None): int32 [K/G,N/8] GPU packed zero points on x.device,
            using the same AWQ order as w and positive stored word strides.
            None means integer zero, not an implicit symmetric zero of eight.
        dtype (torch.dtype | None): Must equal w_scales.dtype when supplied.
            None selects that dtype, which also equals x.dtype.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in x.dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        This operation preserves the repository AWQ packing contract and does
        not infer or repack other INT4 formats.
    """
    a, packed_b, b_scale, b_zero_points = (x, w, w_scales, w_zeros)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("expected rank-two activation and packed weight tensors")
    if w_scales.ndim != 2 or w_scales.shape[0] <= 0 or x.shape[1] % w_scales.shape[0]:
        raise ValueError("AWQ scales must partition K into equal groups")
    group_size = x.shape[1] // w_scales.shape[0]
    if dtype is not None and dtype != w_scales.dtype:
        raise TypeError("AWQ output dtype must match scales")
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 2 or packed_b.ndim != 2 or packed_b.shape[0] != a.shape[1]:
        raise ValueError("AWQ expects A[M,K] and packed B[K,N/8]")
    m, k = a.shape
    n = packed_b.shape[1] * 8
    if (
        k <= 0
        or n <= 0
        or group_size <= 0
        or k % group_size
        or (group_size not in (32, 64, 128, k))
    ):
        raise ValueError(
            "AWQ requires positive K/N and a dividing group size of 32/64/128 or K"
        )
    if (
        a.dtype not in (torch.float16, torch.bfloat16)
        or b_scale.dtype != a.dtype
        or packed_b.dtype != torch.int32
        or (b_zero_points is not None and b_zero_points.dtype != torch.int32)
    ):
        raise TypeError(
            "AWQ requires matching 16-bit A/scales and int32 packed B/zero points"
        )
    if tuple(b_scale.shape) != (k // group_size, n) or (
        b_zero_points is not None
        and tuple(b_zero_points.shape) != (k // group_size, n // 8)
    ):
        raise ValueError("AWQ scale/zero-point shapes must match explicit group_size")
    for tensor in (a, packed_b, b_scale):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("inputs must have positive strides")
    if b_zero_points is None:
        b_zero_points = torch.zeros(
            (k // group_size, n // 8), dtype=torch.int32, device=a.device
        )
    for tensor in (b_zero_points,):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("packed zero points must have positive strides")
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty((m, n), dtype=b_scale.dtype, device=a.device)
        )
        if (
            tuple(out.shape) != (m, n)
            or out.dtype != b_scale.dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not m:
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                config, _ = get_gemm_config(
                    "GEMM-A16W4",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.gemm_a16w4 import awq_gemm_kernel

            if config.get("SPLIT_K") != 1:
                raise ValueError("AWQ public interface requires SPLIT_K=1")
            M, N, K, b, result = (m, n, k, packed_b, out)
            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                1,
            )
            awq_gemm_kernel[grid](
                a, b, result, b_zero_points, b_scale, M, N, K, group_size,
                *a.stride(), *b_scale.stride(), *packed_b.stride(), *b_zero_points.stride(), **config
            )
        elif backend == "gluon":
            raise NotImplementedError("gemm_a16w4: no Gluon kernel is implemented")
        elif backend == "tilelang":
            raise NotImplementedError("gemm_a16w4: no TileLang kernel is implemented")
    return out
