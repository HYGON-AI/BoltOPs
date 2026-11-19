# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py
import os
from copy import deepcopy

import torch
import triton
import triton.language as tl

from boltops.utility.arch_info import DeviceInfo
from ..config import get_gemm_config
from ..validation import validate_output

from boltops.gemm.triton.kernel_repr import make_kernel_repr
from boltops.gemm.triton.pid_preprocessing import pid_grid, remap_xcd

_gemm_a16w16_repr = make_kernel_repr(
    "_gemm_a16_w16_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
        "activation",
        "use_activation",
        "ADD_BIAS",
        "SKIP_REDUCE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_gemm_a16w16_repr,
    do_not_specialize=["M", "N"],
)
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = remap_xcd(pid_unified, num_pid_m * num_pid_n * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        if EVEN_MN:
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        if ADD_BIAS:
            if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                accumulator = tl.broadcast_to(
                    accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
            else:
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
            accumulator = tl.dot(a, b, acc=accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_activation and NUM_KSPLIT == 1:
            accumulator = activation(accumulator)

        # Write back the block of the output matrix C with masks.
        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        if EVEN_MN:
            tl.store(c_ptrs, c)
        else:
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)


def _resolve_activation(activation):
    if activation is None:
        return ""

    from .activation import _get_activation_from_str

    try:
        return _get_activation_from_str(activation)
    except (KeyError, TypeError):
        raise ValueError(f"unsupported activation: {activation!r}") from None


def gemm_a16w16(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *,
    bias: torch.Tensor | None, activation: str | None,
    device: DeviceInfo,
    config: dict[str, object] | None = None,
) -> None:
    activation_fn = _resolve_activation(activation)
    m, k = a.shape
    n = b.shape[0]
    out_dtype = out.dtype
    validate_output(out, (m, n), a.device, (torch.float16, torch.bfloat16, torch.float32))
    user_config = config
    if user_config is not None:
        config = deepcopy(user_config)
    else:
        use_tuned_shape = (
            a.dtype == out_dtype == torch.float16
            and bias is None
            and activation is None
            and out.is_contiguous()
            and (a.stride() == (k, 1))
            and (b.stride() == (1, n))
        )
        config, _ = get_gemm_config(
            "GEMM-A16W16",
            m,
            n if use_tuned_shape else None,
            k if use_tuned_shape else None,
            backend="triton",
            arch=device.arch,
            cu_num=device.cu_count,
        )

    if "use_mls" in config:
        from . import gemm_a16w16_wasp as kernels

        if (
            device.arch != "gfx946"
            or a.dtype != torch.float16
            or out_dtype != torch.float16
            or bias is not None
            or activation is not None
            or a.stride() != (k, 1)
            or b.stride() != (1, n)
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
        from .splitk_reduce import _gemm_splitk_reduce_kernel
        from .gemm_config_utils import compute_splitk_params

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
            # The kernel assumes every stride is positive, even when
            # the split dimension is unused.
            M * N if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
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
