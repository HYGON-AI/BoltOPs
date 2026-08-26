# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Vendored aiter Triton batched BF16/FP16 GEMM.

Upstream: ROCm/aiter ``aiter.ops.triton.batched_gemm_bf16``.

Computes ``Y[i] = X[i] @ W[i]^T`` with ``W`` shaped ``[B, N, K]``.

HCU tuned JSON (optional), searched in this order::

    configs/batched_gemm_bf16/BATCHED_BF16-B={B}-N={N}-K={K}_{dtype}_{device}.json
    configs/batched_gemm_bf16/BATCHED_BF16-N={N}-K={K}_{dtype}_{device}.json
    configs/batched_gemm_bf16/BATCHED_BF16-DEFAULT_{dtype}_{device}.json

For BF16, each lookup level tries a native ``bf16`` file first and then the
corresponding ``fp16`` file. Both use the same 16-bit kernel configuration, so
existing FP16 tuning remains usable until a BF16-specific table is provided.

Flat ``{M: launch_config, ...}`` or AWQ-style
``{ "{N}_{K}": {M: launch_config, ...} }``; nearest ``M`` wins. This keeps
the existing N/K files compatible while allowing batch-specialized tuning.
Missing files fall back to built-in large/small tiles (AMD-only keys filtered
at launch).
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "batched_gemm_bf16",
)

_LAUNCH_KEYS = frozenset(
    {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "use_block_pingpong",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "optimize_epilogue",
    }
)

# aiter MI large/small without waves_per_eu / matrix_instr_nonkdim
_FALLBACK_LARGE = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
}
_FALLBACK_SMALL = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
}


def _device_name_for_config() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name().replace(" ", "_")
    low = name.lower()
    if low.startswith("bw") or "bw200" in low:
        return "BW200"
    if "k100" in low:
        return "K100_AI"
    return name


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype).replace("torch.", "")


def _filter_launch_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in _LAUNCH_KEYS}


@functools.lru_cache
def get_batched_gemm_bf16_configs(
    B: int, N: int, K: int, dtype_tag: str
) -> Optional[dict[int, Any]]:
    """Load the most specialized HCU table available for ``(B, N, K)``.

    Lookup order mirrors upstream AITER: B/N/K-specialized, N/K-specialized,
    then device default. At each level BF16 falls back to FP16 because both
    dtypes share this 16-bit kernel. Existing N/K JSON files remain valid.
    """
    device = _device_name_for_config()
    dtype_tags = (dtype_tag, "fp16") if dtype_tag == "bf16" else (dtype_tag,)
    filenames = tuple(
        filename
        for pattern in (
            "BATCHED_BF16-B={B}-N={N}-K={K}_{dtype}_{device}.json",
            "BATCHED_BF16-N={N}-K={K}_{dtype}_{device}.json",
            "BATCHED_BF16-DEFAULT_{dtype}_{device}.json",
        )
        for candidate_dtype in dtype_tags
        for filename in (
            pattern.format(
                B=B,
                N=N,
                K=K,
                dtype=candidate_dtype,
                device=device,
            ),
        )
    )
    for filename in filenames:
        path = os.path.join(_CONFIG_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        # Accept flat M tables plus the wrappers used by existing/tuning tools.
        for shape_key in (f"{B}_{N}_{K}", f"{N}_{K}", "any"):
            if shape_key in data and isinstance(data[shape_key], dict):
                data = data[shape_key]
                break
        return {int(k): v for k, v in data.items()}
    return None


def _get_config(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    B: int = 1,
) -> dict:
    configs = get_batched_gemm_bf16_configs(B, N, K, _dtype_tag(dtype))
    if configs:
        return _filter_launch_config(
            configs[min(configs.keys(), key=lambda x: abs(x - M))]
        )
    if M + N >= 4096:
        return dict(_FALLBACK_LARGE)
    return dict(_FALLBACK_SMALL)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _batched_gemm_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """Batched matmul ``C[i] = A[i] @ B[i]`` (+ optional bias). Not for direct use."""

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    batch_id = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        accumulator += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def batched_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Batched GEMM ``Y[i] = X[i] @ W[i]^T`` (+ optional bias).

    Args:
        x: ``[B, M, K]`` BF16/FP16
        w: ``[B, N, K]`` BF16/FP16 (internally transposed to ``[B, K, N]``)
        bias: optional ``[B, 1, N]``
        dtype: output dtype (``bfloat16`` or ``float16``)
        splitK: must be ``None`` (unsupported on this Triton path)
        y: optional preallocated ``[B, M, N]``
        config: optional launch dict; else HCU JSON or built-in fallback
    """
    x = x.contiguous()
    w = w.contiguous()

    assert x.shape[0] == w.shape[0], "Incompatible Batch dimensions!!!"
    assert x.shape[2] == w.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_bf16"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    w = w.transpose(1, 2)

    B = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    N = w.shape[2]

    has_bias = bias is not None
    if y is None:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K, dtype, B=B)
    else:
        config = _filter_launch_config(config)

    # Triton needs a real pointer; unused when HAS_BIAS is False.
    bias_arg = bias if has_bias else y
    bias_stride = bias.stride(0) if has_bias else 1

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _batched_gemm_bf16_kernel[grid](
        x,
        w,
        y,
        bias_arg,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        bias_stride,
        has_bias,
        **config,
    )

    return y
