# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""AWQ packed INT4 GEMM with explicit group size."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import AWQGemmBackend
from .validation import validate_input_tensors


def select_backend(backend: str) -> AWQGemmBackend:
    if backend == "gluon":
        raise NotImplementedError("gemm_a16w4: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_a16w4: no TileLang kernel is implemented")

    from .triton.gemm_a16w4 import gemm_a16w4 as triton_gemm_a16w4

    return triton_gemm_a16w4


def _validate_operands(a, packed_b, b_scale, b_zero_points) -> int:
    if a.ndim != 2 or packed_b.ndim != 2:
        raise ValueError("expected rank-two activation and packed weight tensors")
    if b_scale.ndim != 2 or b_scale.shape[0] <= 0 or a.shape[1] % b_scale.shape[0]:
        raise ValueError("AWQ scales must partition K into equal groups")
    group_size = a.shape[1] // b_scale.shape[0]
    if packed_b.shape[0] != a.shape[1]:
        raise ValueError("AWQ expects A[M,K] and packed B[K,N/8]")
    k = a.shape[1]
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
    validate_input_tensors(
        a.device, a, packed_b, b_scale,
        *(() if b_zero_points is None else (b_zero_points,)),
    )
    return group_size


def gemm_a16w4(
    x, w, w_scales, w_zeros=None, dtype=None, out=None, backend=None
):
    """Compute A16 times AWQ INT4 weights with group scales and zero points.

    The logical weight is [K,N]; unpack its INT4 values and zero points
    using the existing AWQ nibble order, subtract zero points, multiply
    by the corresponding K-group scale, then compute ``x @ weight``.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [M,K] with positive strides.
        w (torch.Tensor): int32 weights [K,N/8] on x.device with positive
            strides. Each word packs eight unsigned INT4 values along N in
            AWQ order.
        w_scales (torch.Tensor): Scales [K/G,N] in x.dtype on x.device, with
            positive strides. K and N must be positive. G is inferred from
            the scale shape and must divide K and be 32, 64, 128, or K.
        w_zeros (torch.Tensor | None): Optional int32 packed zero points
            [K/G,N/8] on x.device with positive strides and the same AWQ
            packing as w. None uses zero.
        dtype (torch.dtype | None): Dtype used to allocate the output. Uses w_scales.dtype when None.
            An explicit dtype must match w_scales.dtype.
        out (torch.Tensor | None): Optional contiguous [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

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
    backend = "auto" if backend is None else backend
    group_size = _validate_operands(a, packed_b, b_scale, b_zero_points)
    if out is None:
        out_dtype = b_scale.dtype if dtype is None else dtype
        out = torch.empty((a.shape[0], packed_b.shape[1] * 8), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(backend)
        implementation(
            a, packed_b, out, b_scale=b_scale, b_zero_points=b_zero_points, group_size=group_size,
            device=device,
        )
    return out
