# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP16/BF16 dense GEMM with backend selection."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_a16w16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    activation=None,
    backend=None,
):
    """Compute dense FP16/BF16 GEMM, optional bias and fused activation.

    Args:
        x (torch.Tensor): GPU activation [M,K], float16 or bfloat16. The general Triton kernel
            accepts positive-stride views: row-major (K,1) makes K contiguous;
            column-major (1,M) makes M contiguous. Padded positive strides are
            also allowed.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Row-major (K,1)
            makes K contiguous; column-major (1,N) makes N contiguous. The general
            Triton kernel accepts positive-stride views. The recorded MLS/WDRA
            pipeline specifically requires x.stride()=(K,1) and w.stride()=(1,N).
            The logical transpose w.T is internal.
        bias (torch.Tensor | None): Optional [N] bias in x.dtype on the same device,
            with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        activation (str | None): Optional "gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu" or "relu6", applied to FP32 results after bias
            and the complete K reduction, before output conversion. None is
            identity. Explicit MLS/WDRA configurations do not support activation.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, out_dtype = (x, w, x.dtype if dtype is None else dtype)
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    activation_fn = ""
    if activation is not None:
        from .triton.activation import _get_activation_from_str

        try:
            activation_fn = _get_activation_from_str(activation)
        except (KeyError, TypeError):
            raise ValueError(f"unsupported activation: {activation!r}") from None
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("gemm_a16w16 received an unsupported input dtype")
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
    if out_dtype is not None and out_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("gemm_a16w16 received an unsupported output dtype")
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
            if activation is not None:
                import triton
                from .triton.activation import _apply_activation_kernel

                _apply_activation_kernel[(triton.cdiv(out.numel(), 256),)](
                    out, out.numel(), activation_fn, BLOCK_SIZE=256
                )
            return out
        device = get_device_info(a.device)
        backend = "triton" if backend == "auto" else backend
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                use_tuned_shape = (
                    a.dtype == out_dtype == torch.float16
                    and bias is None
                    and activation is None
                    and (a.stride() == (k, 1))
                    and (b.stride() == (1, n))
                )
                config, _ = get_gemm_config(
                    "GEMM-A16W16",
                    m,
                    n if use_tuned_shape else None,
                    k if use_tuned_shape else None,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton

            if "use_mls" in config:
                from .triton import gemm_a16w16_wasp as kernels
                import os

                if (
                    device.arch != "gfx946"
                    or a.dtype != torch.float16
                    or out_dtype != torch.float16
                    or (bias is not None)
                    or activation is not None
                    or (a.stride() != (k, 1))
                    or (b.stride() != (1, n))
                ):
                    raise ValueError(
                        "recorded FP16 pipeline requires gfx946, FP16 output and TT operands without bias or activation"
                    )
                wasp, wdra, use_mls, use_matrix_store = (
                    config.pop(key)
                    for key in ("wasp", "wdra", "use_mls", "use_matrix_store")
                )
                if wdra and (not wasp) or (use_matrix_store and (not use_mls)):
                    raise ValueError("invalid WDRA/MLS configuration")
                if (wasp or wdra) and (not kernels.supports_wasp_wdra(device.arch)):
                    raise ValueError("target does not support WASP/WDRA")
                if use_mls and (not kernels.arch_info.is_mls_avail()):
                    raise ValueError("target does not support MLS")
                load_warps, mma_warps = kernels._resolve_wasp_topology(
                    wasp=wasp, wdra=wdra, wasp_num_load_warps=None, wasp_num_mma_warps=None
                )
                kernels._mode_launch_kwargs(
                    wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
                )
                os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
                config = {
                    key: tuple(value) if isinstance(value, list) else value
                    for (key, value) in config.items()
                }
                config["HAS_BIAS"] = False
                if use_mls:
                    config["USE_MATRIX_STORE"] = bool(use_matrix_store)
                kernel = (
                    kernels._gemm_fp16_mls_kernel
                    if use_mls
                    else kernels._gemm_fp16_load_kernel
                )
                b = b.T
                grid = lambda meta: (
                    triton.cdiv(m, meta["BLOCK_SIZE_M"])
                    * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                )
                kernel[grid](
                    a, b, out, b, m, n, k, *a.stride(), *b.stride(), *out.stride(), **config
                )
            else:
                from .triton.gemm_a16w16 import _gemm_a16_w16_kernel
                from .triton.splitk_reduce import _gemm_splitk_reduce_kernel
                from .triton.gemm_config_utils import compute_splitk_params

                if user_config is None:
                    config.setdefault("NUM_KSPLIT", 1)
                    config = compute_splitk_params(config, k)
                M, N, K = (m, n, k)
                w_t = b.T
                if config["NUM_KSPLIT"] > 1:
                    y_pp = torch.empty(
                        (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=out.device
                    )
                else:
                    y_pp = None
                grid = lambda META: (
                    META["NUM_KSPLIT"]
                    * triton.cdiv(M, META["BLOCK_SIZE_M"])
                    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                )
                _gemm_a16_w16_kernel[grid](
                    a,
                    w_t,
                    bias,
                    out if config["NUM_KSPLIT"] == 1 else y_pp,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    w_t.stride(0),
                    w_t.stride(1),
                    0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
                    out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
                    out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
                    activation=activation_fn,
                    use_activation=activation is not None,
                    ADD_BIAS=bias is not None,
                    SKIP_REDUCE=False,
                    **config,
                )
                if config["NUM_KSPLIT"] > 1:
                    REDUCE_BLOCK_SIZE_M = 32
                    REDUCE_BLOCK_SIZE_N = 32
                    ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])
                    grid_reduce = (
                        triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
                        triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
                    )
                    _gemm_splitk_reduce_kernel[grid_reduce](
                        y_pp,
                        out,
                        bias,
                        M,
                        N,
                        y_pp.stride(0),
                        y_pp.stride(1),
                        y_pp.stride(2),
                        out.stride(0),
                        out.stride(1),
                        REDUCE_BLOCK_SIZE_M,
                        REDUCE_BLOCK_SIZE_N,
                        ACTUAL_KSPLIT,
                        triton.next_power_of_2(config["NUM_KSPLIT"]),
                        ADD_BIAS=bias is not None,
                        activation=activation_fn,
                        use_activation=activation is not None,
                        KERNEL_NAME="_gemm_a16w16_reduce_kernel",
                    )
                return out
        elif backend == "gluon":
            raise NotImplementedError("gemm_a16w16: no Gluon kernel is implemented")
        elif backend == "tilelang":
            raise NotImplementedError("gemm_a16w16: no TileLang kernel is implemented")
    return out
