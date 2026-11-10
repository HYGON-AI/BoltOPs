# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Native GEMM with A row/K and B N/K block scales."""

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


def gemm_a8w8_blockscale(
    x,
    w,
    x_scale,
    w_scale,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    backend=None,
):
    """Compute FP8/INT8 GEMM with A row/K-group and B N/K-block scales.

    For each output element, sum products over K after applying the A/B
    scales for that K group. The A recipe is (1,BK_SCALE) and the B recipe
    is (BN_SCALE,BK_SCALE); these scale blocks are independent of the
    compute tile BLOCK_SIZE_M/N/K.

    Args:
        x (torch.Tensor): GPU activation [M,K], float8_e4m3fn, float8_e5m2 or int8.
            Triton accepts any positive matrix strides, including M-contiguous,
            K-contiguous, transposed and padded/sliced views.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Triton accepts
            any positive matrix strides independently of x; w.T is internal.
        x_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales
            [M,ceil(K/BK_SCALE)] on x.device. One value covers one A row and
            BK_SCALE logical K elements. Both matrix strides may be any positive
            values on the Triton path.
        w_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales
            [ceil(N/BN_SCALE),ceil(K/BK_SCALE)] on x.device, with positive matrix
            strides on Triton. One value covers BN_SCALE weight rows and
            BK_SCALE logical K elements. Effective BN_SCALE/BK_SCALE are inferred
            from the shape as powers of two at least 16; scale shapes must match.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Both the
            default and None select bfloat16.
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
            Auto selects Gluon only for a recorded precision/recipe/M/N/K/device/CU
            match, otherwise Triton. Explicit selection never retries a backend.
            TileLang is not implemented; explicit Gluon also requires a supported
            configuration and legal kernel inputs.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Scale values are numeric, not encoded
        E8M0. The current tuned Gluon recipe requires contiguous x/w/scales and uses E4M3FN inputs, FP32 A/B scales
        with A(1,128)/B(128,128), BF16 output and complete 128x256 tiles; its K
        is 256..6144 divisible by 256. Auto uses Triton for noncontiguous matrix views; explicit Gluon reports
        unsupported storage rather than copying tensors. Other combinations use the native Triton
        path when supported. There is no software matrix-instruction fallback.
    """
    a, b, a_scale, b_scale, out_dtype = (x, w, x_scale, w_scale, dtype)
    if a.ndim != 2 or b.ndim != 2 or b_scale.ndim != 2 or (min(b_scale.shape) <= 0):
        raise ValueError("expected rank-two operands and block scales")
    k, n = (a.shape[1], b.shape[0])
    bk = max(16, 1 << ((k + b_scale.shape[1] - 1) // b_scale.shape[1] - 1).bit_length())
    bn = max(16, 1 << ((n + b_scale.shape[0] - 1) // b_scale.shape[0] - 1).bit_length())
    block_size = (bn, bk)
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or (a.dtype != b.dtype):
        raise ValueError("expected same-dtype A[M,K] and B[N,K]")
    m, k = a.shape
    n = b.shape[0]
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)
    if a.dtype not in allowed:
        raise TypeError("unsupported input dtype")
    for tensor in (a, b, a_scale, b_scale):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError("inputs must have positive strides")
    if any(
        (
            scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            for scale in (a_scale, b_scale)
        )
    ):
        raise TypeError("numeric scales must be FP16/BF16/FP32")
    bn, bk = block_size
    expected = ((m, (k + bk - 1) // bk), ((n + bn - 1) // bn, (k + bk - 1) // bk))
    if tuple(a_scale.shape) != expected[0] or tuple(b_scale.shape) != expected[1]:
        raise ValueError(f"scale shapes must be {expected}")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("output must be FP16/BF16/FP32")
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
            out.zero_()
            return out
        device = get_device_info(a.device)
        dtypes = "-".join(
            f"{name}={str(tensor.dtype).removeprefix('torch.')}"
            for name, tensor in (
                ("DTYPE", a),
                ("AS", a_scale),
                ("BS", b_scale),
                ("OUT", out),
            )
        )
        gluon_profile = f"{dtypes}-RECIPE_A=1x{bk}-RECIPE_B={bn}x{bk}"
        gluon_config = None
        if backend == "auto":
            backend = "triton"
            if all(t.is_contiguous() for t in (a, b, a_scale, b_scale)):
                try:
                    gluon_config, tuned = get_gemm_config(
                        "GEMM-A8W8_BLOCKSCALE",
                        m,
                        n,
                        k,
                        profile=gluon_profile,
                        backend="gluon",
                        arch=device.arch,
                        cu_num=device.cu_count,
                    )
                except (FileNotFoundError, KeyError):
                    pass
                else:
                    if tuned:
                        backend = "gluon"
        if backend == "gluon":
            if not all(t.is_contiguous() for t in (a, b, a_scale, b_scale)):
                raise NotImplementedError(
                    "the Gluon blockscale pipeline requires contiguous operands and scales"
                )
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            elif gluon_config is not None:
                config = gluon_config
            else:
                try:
                    config, _ = get_gemm_config(
                        "GEMM-A8W8_BLOCKSCALE",
                        m,
                        n,
                        k,
                        profile=gluon_profile,
                        backend=backend,
                        arch=device.arch,
                        cu_num=device.cu_count,
                    )
                except KeyError:
                    raise NotImplementedError(
                        "gemm_a8w8_blockscale: no Gluon configuration for the requested precision and scale recipe"
                    ) from None
            import triton
            from .gluon import gemm_a8w8_blockscale as kernels

            kernel = getattr(kernels, config.pop("kernel"))
            config = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in config.items()
            }
            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_SIZE_M"]),
                triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            )
            kernel[grid](
                a,
                b,
                a_scale,
                b_scale,
                out,
                M=m,
                N=n,
                K=k,
                GROUP_N=bn,
                GROUP_K=bk,
                **config,
            )

        elif backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                config, _ = get_gemm_config(
                    "GEMM-A8W8_BLOCKSCALE",
                    m,
                    n,
                    k,
                    backend=backend,
                    arch=device.arch,
                    cu_num=device.cu_count,
                )
            import triton

            if a.dtype == torch.int8:
                from .triton.gemm_a8w8_blockscale_int8 import (
                    _w8a8_block_int8_matmul as kernel,
                )
            else:
                from .triton.gemm_a8w8_blockscale_fp8 import (
                    _w8a8_triton_block_scaled_mm as kernel,
                )
            config.setdefault("BLOCK_SIZE_N", bn)
            config.setdefault("BLOCK_SIZE_K", bk)
            if bk % config["BLOCK_SIZE_K"]:
                raise ValueError("BLOCK_SIZE_K must divide the scale K block")
            M, N, K, C, block_n, block_k = (m, n, k, out, bn, bk)

            def grid(META):
                return (
                    triton.cdiv(M, META["BLOCK_SIZE_M"])
                    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                )

            kernel[grid](
                a,
                b,
                C,
                a_scale,
                b_scale,
                M,
                N,
                K,
                block_n,
                block_k,
                a.stride(-2),
                a.stride(-1),
                b.stride(1),
                b.stride(0),
                C.stride(-2),
                C.stride(-1),
                a_scale.stride(-2),
                a_scale.stride(-1),
                b_scale.stride(1),
                b_scale.stride(0),
                **config,
            )
        elif backend == "tilelang":
            raise NotImplementedError(
                "gemm_a8w8_blockscale: no TileLang kernel is implemented"
            )
    return out
