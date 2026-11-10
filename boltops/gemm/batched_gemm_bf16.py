# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Batched GEMM with independent args/configuration/backend selection."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def batched_gemm_bf16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    backend=None,
):
    """Compute batched FP16/BF16 ``x[b] @ w[b].T + bias[b]``.

    FP16 and BF16 share this interface despite its historical name.
    Inputs have equal batch counts; batch broadcasting is not supported.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [B,M,K], with positive strides.
            K-contiguous, M-contiguous and padded/strided matrix views are
            supported. Standard contiguous strides are (M*K,K,1). Batch-only
            contiguous storage (only batch stride is 1) is rejected.
        w (torch.Tensor): GPU weights [B,N,K], same B, K, dtype/device as x, with
            positive strides. K-contiguous, N-contiguous and padded/strided
            matrix views are supported. Standard strides are (N*K,K,1).
            Batch-only contiguous storage is rejected; the logical transpose
            is internal and batch broadcasting is not supported.
        bias (torch.Tensor | None): Optional [B,1,N] bias in x.dtype on the
            same device, unit stride along N. Broadcast across M, not B.
        dtype (torch.dtype | None): Output float16 or bfloat16. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [B,M,N], on the input
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
        torch.Tensor: Contiguous [B,M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Equal-length grouped GEMM with this same storage contract uses this
        entry. Ragged offsets, indexed tokens and group masks are not supported.
    """
    a, b = (x, w)
    out_dtype = x.dtype if dtype is None else dtype
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("batched expects rank-three A/B with equal batch counts")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("batched_gemm_bf16 requires FP16/BF16 inputs")
    batch, m, k = a.shape
    n = b.shape[1]
    if b.shape[2] != k or a.dtype != b.dtype:
        raise ValueError("batched A/B must have matching K and dtype")
    if bias is not None and (
        tuple(bias.shape) != (batch, 1, n) or bias.dtype != a.dtype
    ):
        raise ValueError("batched bias must be [B,1,N] in the input dtype")
    for tensor in (a, b, *(() if bias is None else (bias,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any((s <= 0 for s in tensor.stride())):
            raise ValueError("batched inputs must have positive strides")
    if bias is not None and bias.stride(-1) != 1:
        raise ValueError("bias must have unit stride along N")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if out_dtype is not None and out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("batched_gemm_bf16 received an unsupported output dtype")
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty((batch, m, n), device=a.device, dtype=out_dtype)
        )
        if (
            tuple(out.shape) != (batch, m, n)
            or out.dtype != out_dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not batch or not m or (not n):
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand_as(out))
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                layout = "TN" if b.stride(-1) == 1 else "TT"
                use_tuned_shape = (a.dtype == out_dtype and a.stride(-1) == 1
                                  and (b.stride(-1) == 1 or b.stride(-2) == 1))
                config, _ = get_gemm_config(
                    "BATCHED_GEMM-A16W16",
                    m,
                    n if use_tuned_shape else None,
                    k if use_tuned_shape else None,
                    B=batch if use_tuned_shape else None,
                    layout=layout,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton
            from .triton.batched_gemm_a16w16 import _batched_gemm_a16w16_kernel

            for tensor in (a, b):
                if batch > 1 and tensor.stride(0) == 1 and tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
                    raise ValueError("batch-only contiguous storage does not provide a contiguous matrix axis")
            b = b.transpose(1, 2)

            def grid(meta):
                tiles = triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(
                    n, meta["BLOCK_SIZE_N"]
                )
                return (
                    (tiles, 1, batch)
                    if meta.get("BATCH_LAST_GRID", False)
                    else (batch, tiles)
                )

            _batched_gemm_a16w16_kernel[grid](
                a,
                b,
                out,
                bias if bias is not None else out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                bias.stride(0) if bias is not None else 1,
                bias is not None,
                **config,
            )
        elif backend == "gluon":
            raise NotImplementedError(
                "batched_gemm_bf16: no Gluon kernel is implemented"
            )
        elif backend == "tilelang":
            raise NotImplementedError(
                "batched_gemm_bf16: no TileLang kernel is implemented"
            )
    return out
