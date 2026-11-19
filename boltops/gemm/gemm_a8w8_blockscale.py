# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Native GEMM with A row/K and B N/K block scales."""

import torch
from boltops.utility.arch_info import get_device_info
from .backend_interfaces import BlockScaledGemmBackend
from .validation import validate_input_tensors


def select_backend(
    a, b, out, *, a_scale, b_scale, block_size, backend, device,
) -> BlockScaledGemmBackend:
    if backend in ("auto", "gluon"):
        from .gluon import resolve_a8w8_blockscale_backend

        implementation = resolve_a8w8_blockscale_backend(
            a, b, out, a_scale=a_scale, b_scale=b_scale, block_size=block_size,
            device=device, automatic=backend == "auto",
        )
        if implementation is not None:
            return implementation
    if backend == "gluon":
        raise NotImplementedError("gemm_a8w8_blockscale: no Gluon kernel is implemented")
    if backend == "tilelang":
        raise NotImplementedError("gemm_a8w8_blockscale: no TileLang kernel is implemented")

    from .triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale

    return triton_gemm_a8w8_blockscale


def _validate_operands(a, b, a_scale, b_scale) -> tuple[int, int]:
    if a.ndim != 2 or b.ndim != 2 or b_scale.ndim != 2 or (min(b_scale.shape) <= 0):
        raise ValueError("expected rank-two operands and block scales")
    m, k = a.shape
    n = b.shape[0]
    if k != b.shape[1] or a.dtype != b.dtype:
        raise ValueError("expected same-dtype A[M,K] and B[N,K]")
    bk = max(16, 1 << ((k + b_scale.shape[1] - 1) // b_scale.shape[1] - 1).bit_length())
    bn = max(16, 1 << ((n + b_scale.shape[0] - 1) // b_scale.shape[0] - 1).bit_length())
    block_size = (bn, bk)
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)
    if a.dtype not in allowed:
        raise TypeError("unsupported input dtype")
    validate_input_tensors(a.device, a, b, a_scale, b_scale)
    if any(
        (
            scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            for scale in (a_scale, b_scale)
        )
    ):
        raise TypeError("numeric scales must be FP16/BF16/FP32")
    expected = ((m, (k + bk - 1) // bk), ((n + bn - 1) // bn, (k + bk - 1) // bk))
    if tuple(a_scale.shape) != expected[0] or tuple(b_scale.shape) != expected[1]:
        raise ValueError(f"scale shapes must be {expected}")
    return block_size


def gemm_a8w8_blockscale(
    x,
    w,
    x_scale,
    w_scale,
    dtype=torch.bfloat16,
    out=None,
    backend=None,
):
    """Compute FP8/INT8 GEMM with A row/K-group and B N/K-block scales.

    For each output element, sum products over K after applying the A/B
    scales for that K group. The A recipe is (1,BK_SCALE) and the B recipe
    is (BN_SCALE,BK_SCALE).

    Args:
        x (torch.Tensor): GPU input [M,K] in float8_e4m3fn, float8_e5m2, or int8.
        w (torch.Tensor): GPU weights [N,K] with the same dtype and device as x.
        x_scale (torch.Tensor): float16, bfloat16, or float32 scales
            [M,ceil(K/BK_SCALE)] on x.device. Each scale covers one input row
            and BK_SCALE consecutive K elements.
        w_scale (torch.Tensor): float16, bfloat16, or float32 scales
            [ceil(N/BN_SCALE),ceil(K/BK_SCALE)] on x.device. Each scale covers
            BN_SCALE weight rows and BK_SCALE consecutive K elements. Group
            sizes are inferred from scale shapes and must be powers of two
            of at least 16. The K grouping must match x_scale.
        dtype (torch.dtype | None): Allocation dtype (float16, bfloat16, or float32).
            Defaults to bfloat16. None also selects bfloat16.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" or "gluon" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, a_scale, b_scale, out_dtype = (x, w, x_scale, w_scale, dtype)
    backend = "auto" if backend is None else backend
    block_size = _validate_operands(a, b, a_scale, b_scale)
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)

    with torch.cuda.device(a.device):
        device = get_device_info(a.device)
        implementation = select_backend(
            a, b, out, a_scale=a_scale, b_scale=b_scale, block_size=block_size,
            backend=backend, device=device,
        )
        implementation(
            a, b, out, a_scale=a_scale, b_scale=b_scale, block_size=block_size,
            device=device,
        )
    return out
