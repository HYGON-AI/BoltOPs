# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "awq_gemm_triton"
)

_LAUNCH_KEYS = frozenset(
    {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "SPLIT_K",
        "num_warps",
        "num_stages",
        "num_ldmatrixes",
    }
)


def _device_name_for_awq() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name().replace(" ", "_")
    low = name.lower()
    if low.startswith("bw") or "bw200" in low:
        return "BW200"
    if "k100" in low:
        return "K100_AI"
    return name


@functools.lru_cache
def get_awq_triton_configs(N: int, K: int) -> Optional[dict[int, Any]]:
    """Load HCU AWQ tables ``AWQ_{N}_{K}_{device}.json`` (vLLM naming)."""
    device = _device_name_for_awq()
    path = os.path.join(_CONFIG_DIR, f"AWQ_{N}_{K}_{device}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # File is ``{f"{N}_{K}": {M: config, ...}}`` or flat ``{M: config}``.
    if f"{N}_{K}" in data and isinstance(data[f"{N}_{K}"], dict):
        data = data[f"{N}_{K}"]
    return {int(k): v for k, v in data.items()}


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Set up the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks)
    
    iweights =tl.join(iweights, iweights).reshape(iweights.shape[:-1] + [2 * iweights.shape[-1]])
    iweights =tl.join(iweights, iweights).reshape(iweights.shape[:-1] + [2 * iweights.shape[-1]])
    iweights =tl.join(iweights, iweights).reshape(iweights.shape[:-1] + [2 * iweights.shape[-1]])
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)
    zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    group_size, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr,SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    
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

    # accumulator_dtype = c_ptr.type.element_ty
    BLOCK_SIZE_N_8 = BLOCK_SIZE_N // 8
    N_8 = N // 8
    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=tl.float32)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    shifts = ((tl.arange(0, 2) * 16)[None, :] +
                                (tl.arange(0, 4) * 4)[:, None]).reshape(1,8)

    # Create the necessary shifts to use to unpack.
    # shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts,
                             (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bzn = pid_n * (BLOCK_SIZE_N_8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bzn = offsets_bzn < N_8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N_8) * offsets_k[:, None] + offsets_bzn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bzn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)
        # b = tl.interleave(b, b)
        # b = tl.interleave(b, b)
        # b = tl.interleave(b, b)
        b =tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
        b =tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
        b =tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size 
        offsets_szk = offsets_szk + (tl.arange(0, BLOCK_SIZE_K) // group_size) 
        offsets_z = (N_8) * offsets_szk[:, None] + offsets_bzn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_bzn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z)
        # zeros = tl.interleave(zeros, zeros)
        # zeros = tl.interleave(zeros, zeros)
        # zeros = tl.interleave(zeros, zeros)
        
        zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        zeros =tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N_8)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + N * offsets_am[:, None] + offsets_sn[None, :]
    c_mask = masks_am[:, None] & masks_sn[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
        # tl.store(c_ptrs, c)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(input: torch.Tensor,
                    qweight: torch.Tensor,
                    scales: torch.Tensor,
                    qzeros: torch.Tensor,
                    split_k_iters: int,
                    config=None) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )

    if config is None:
        configs = get_awq_triton_configs(N, K)
        if configs:
            config = {
                k: v
                for k, v in configs[
                    min(configs.keys(), key=lambda x: abs(x - M))
                ].items()
                if k in _LAUNCH_KEYS
            }
        else:
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 8,
            }
            if M > 256:
                config = {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                    "SPLIT_K": 1,
                }
        if "SPLIT_K" not in config:
            config["SPLIT_K"] = split_k_iters

    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        **config,
    )

    return result
