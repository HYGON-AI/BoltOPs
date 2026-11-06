# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP32 dense GEMM with backend selection."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_a32w32(x, w, bias=None, dtype=None, out=None, config=None, backend=None):
    """Compute dense FP32 ``x @ w.T + bias`` using the persistent kernel.

    Args:
        x (torch.Tensor): float32 GPU activation [M,K]. Positive-stride views are allowed:
            (K,1) is K-contiguous, (1,M) is M-contiguous; padded positive strides
            are supported.
        w (torch.Tensor): float32 GPU weights [N,K] on x.device. Positive-stride views are
            allowed: (K,1) is K-contiguous, (1,N) is N-contiguous; padded positive
            strides are supported. The logical transpose w.T is internal.
        bias (torch.Tensor | None): Optional float32 [N] bias on x.device with unit
            stride, broadcast across M.
        dtype (torch.dtype | None): Only torch.float32 is supported; None selects it.
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
        torch.Tensor: Contiguous [M,N] result in torch.float32.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, out_dtype = (x, w, torch.float32 if dtype is None else dtype)
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    if a.dtype not in (torch.float32,):
        raise TypeError("gemm_a32w32 received an unsupported input dtype")
    m, k = a.shape
    n = b.shape[0]
    if bias is not None and (bias.shape != (n,) or bias.dtype != a.dtype):
        raise ValueError("bias must have shape [N] and the input dtype")
    for tensor in (a, b, *(() if bias is None else (bias,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any((s <= 0 for s in tensor.stride())):
            raise ValueError("dense tensors must have positive strides")
    if bias is not None and bias.stride(-1) != 1:
        raise ValueError("bias must have unit stride along N")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if out_dtype is not None and out_dtype not in (torch.float32,):
        raise TypeError("gemm_a32w32 received an unsupported output dtype")
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty((m, n), dtype=out_dtype, device=a.device)
        )
        if (
            tuple(out.shape) != (m, n)
            or out.dtype != out_dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not m or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand(m, n))
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                config, _ = get_gemm_config(
                    "GEMM-A32W32",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.gemm_a32w32 import matmul_kernel_persistent

            b = b.T

            def grid(meta):
                return (
                    min(
                        device.cu_count,
                        triton.cdiv(m, meta["BLOCK_SIZE_M"])
                        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                    ),
                )

            matmul_kernel_persistent[grid](
                a,
                b,
                out,
                bias,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                NUM_SMS=device.cu_count,
                A_LARGE=a.numel() > 2**31,
                B_LARGE=b.numel() > 2**31,
                C_LARGE=out.numel() > 2**31,
                HAS_BIAS=bias is not None,
                **config,
            )
        elif backend == "gluon":
            raise NotImplementedError("gemm_a32w32: no Gluon kernel is implemented")
        elif backend == "tilelang":
            raise NotImplementedError("gemm_a32w32: no TileLang kernel is implemented")
    return out
