# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0
"""Shared MoE activation implementation and public Triton entry points.

Terminology:
- "gated" activation means gate-up fusion: input is interpreted as [gate, up]
  (typically shape [..., 2D]) and output is activation(gate) * up (shape [..., D]).
- "non-gated" activation means no gate multiplication: input/output keep the
  same last-dimension size (typically [..., D] -> [..., D]).
"""

from enum import IntEnum
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


class _FusedActivationQuantKind(IntEnum):
    GATED_SILU = 0
    GATED_GELU = 1
    GATED_GELU_TANH = 2
    GATED_SILU_LIMIT = 3
    GATED_SITU = 4
    GATED_SWIGLU_STEP = 5
    SILU = 6
    GELU = 7
    GELU_TANH = 8
    RELU2 = 9


# Triton JIT functions can only capture globals instantiated as tl.constexpr.
_FUSED_ACT_GATED_SILU = tl.constexpr(
    _FusedActivationQuantKind.GATED_SILU.value
)
_FUSED_ACT_GATED_GELU = tl.constexpr(
    _FusedActivationQuantKind.GATED_GELU.value
)
_FUSED_ACT_GATED_GELU_TANH = tl.constexpr(
    _FusedActivationQuantKind.GATED_GELU_TANH.value
)
_FUSED_ACT_GATED_SILU_LIMIT = tl.constexpr(
    _FusedActivationQuantKind.GATED_SILU_LIMIT.value
)
_FUSED_ACT_GATED_SITU = tl.constexpr(
    _FusedActivationQuantKind.GATED_SITU.value
)
_FUSED_ACT_GATED_SWIGLU_STEP = tl.constexpr(
    _FusedActivationQuantKind.GATED_SWIGLU_STEP.value
)
_FUSED_ACT_SILU = tl.constexpr(_FusedActivationQuantKind.SILU.value)
_FUSED_ACT_GELU = tl.constexpr(_FusedActivationQuantKind.GELU.value)
_FUSED_ACT_GELU_TANH = tl.constexpr(
    _FusedActivationQuantKind.GELU_TANH.value
)
_FUSED_ACT_RELU2 = tl.constexpr(_FusedActivationQuantKind.RELU2.value)


@triton.heuristics(
    {
        "N_DIV": lambda args: (args["N"] % args["BLOCK_SIZE_N"]) == 0,
    }
)
@triton.jit
def activation_and_mul_kernel(
    out_ptr,  # [M, N]
    in_ptr,  # [M, 2N], chunked layout [gate(0:N), up(N:2N)]
    M,
    N: tl.constexpr,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    ACT: tl.constexpr,
    N_DIV: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 1D launch over [M, N] tiles:
    # pid -> (pid_m, pid_n), each program computes one [1, BLOCK_SIZE_N] slice.
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_n = offs_n < N
    row_in = in_ptr + pid_m.to(tl.int64) * stride_in0
    gate_ptrs = row_in + offs_n * stride_in1
    up_ptrs = row_in + (offs_n + N) * stride_in1
    if N_DIV:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        gate = tl.load(gate_ptrs, mask=mask_n, other=0.0)
        up = tl.load(up_ptrs, mask=mask_n, other=0.0)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    # ACT=0: SiLU(x) * y, where SiLU(x)=x*sigmoid(x)
    # ACT=1: GELU(x) * y, exact erf form:
    #        GELU(x)=0.5*x*(1+erf(x/sqrt(2)))
    if ACT == 0:
        act = gate_f * (1.0 / (1.0 + tl.exp(-gate_f)))
    else:
        act = gate_f * 0.5 * (1.0 + tl.erf(gate_f * 0.7071067811865476))
    y = act * up_f

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_n * stride_out1
    if N_DIV:
        tl.store(out_ptrs, y.to(gate.dtype))
    else:
        tl.store(out_ptrs, y.to(gate.dtype), mask=mask_n)


def get_triton_activation_and_mul_config(M, _N):
    # 2 configs tuned on gfx938 (BW1000B), bf16:
    # small-M: BS128_W1 — stable, low overhead for decode
    # large-M: BS1024_W1 — ~1.4-1.7x faster than old BS512_W4 at M>=4K
    if M <= 16:
        return {"BLOCK_SIZE_N": 128, "num_warps": 1}
    return {"BLOCK_SIZE_N": 1024, "num_warps": 1}


def get_triton_activation_no_mul_config(M, _N):
    # Exactly two configs for the unified flat no-mul kernel:
    # small-M: BS512/W1 won on BF16 [1,1536] and [8,1536].
    # large-M: BS1024/W1 won on BF16 [3394,1536].
    if M <= 16:
        return {"BLOCK_SIZE_N": 512, "num_warps": 1}
    return {"BLOCK_SIZE_N": 1024, "num_warps": 1}


@triton.heuristics(
    {
        "FLAT_DIV": lambda args: (
            args["n_elements"] % args["BLOCK_SIZE_N"]
        ) == 0,
        "N_DIV": lambda args: (args["N"] % args["BLOCK_SIZE_N"]) == 0,
    }
)
@triton.jit
def activation_no_mul_1d_kernel(
    out_ptr,  # contiguous [M, N]
    in_ptr,  # contiguous [M, N]
    M,
    N: tl.constexpr,
    n_elements,
    ACT: tl.constexpr,  # 0:silu 1:gelu 2:relu2
    BLOCK_SIZE_N: tl.constexpr,
    FLAT_DIV: tl.constexpr,
    N_DIV: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(n_elements > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    block_offsets = tl.arange(0, BLOCK_SIZE_N)
    if FLAT_DIV:
        offsets = pid * BLOCK_SIZE_N + block_offsets
        x = tl.load(in_ptr + offsets)
    else:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offsets_n = pid_n * BLOCK_SIZE_N + block_offsets
        offsets = pid_m * N + offsets_n
        mask = offsets_n < N
        if N_DIV:
            x = tl.load(in_ptr + offsets)
        else:
            x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    # ACT=0: SiLU(x)=x*sigmoid(x)
    # ACT=1: GELU(x)=0.5*x*(1+erf(x/sqrt(2)))
    # ACT=2: ReLU^2(x)=max(x,0)^2
    if ACT == 0:
        y = xf * (1.0 / (1.0 + tl.exp(-xf)))
    elif ACT == 1:
        y = xf * 0.5 * (1.0 + tl.erf(xf * 0.7071067811865476))
    else:
        y = tl.where(xf > 0.0, xf * xf, 0.0)

    if FLAT_DIV or N_DIV:
        tl.store(out_ptr + offsets, y.to(x.dtype))
    else:
        tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def _triton_activation_no_mul(out: torch.Tensor, inp: torch.Tensor, act: int) -> None:
    assert inp.shape == out.shape
    assert inp.is_contiguous() and out.is_contiguous()
    M, N = inp.shape
    config = get_triton_activation_no_mul_config(M, N)
    n_elements = inp.numel()
    if n_elements % config["BLOCK_SIZE_N"] == 0:
        grid = (triton.cdiv(n_elements, config["BLOCK_SIZE_N"]),)
    else:
        grid = (M * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    activation_no_mul_1d_kernel[grid](
        out,
        inp,
        M,
        N,
        n_elements,
        ACT=act,
        **config,
    )


def triton_relu2(out: torch.Tensor, inp: torch.Tensor) -> None:
    # ReLU^2(x) = max(x, 0)^2
    _triton_activation_no_mul(out, inp, act=2)


def triton_silu_no_mul(out: torch.Tensor, inp: torch.Tensor) -> None:
    # SiLU(x) = x * sigmoid(x)
    _triton_activation_no_mul(out, inp, act=0)


def triton_gelu_no_mul(out: torch.Tensor, inp: torch.Tensor) -> None:
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    _triton_activation_no_mul(out, inp, act=1)


def triton_silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    # Split input into [x, y] along last dim and compute:
    # out = SiLU(x) * y
    # Shapes: input=[M, 2N], out=[M, N]
    assert input.shape[-1] % 2 == 0
    assert input.is_contiguous()
    assert out.is_contiguous()
    M = input.numel() // input.shape[-1]
    N = input.shape[-1] // 2
    input_2d = input.view(M, input.shape[-1])
    out_2d = out.view(M, N)
    config = get_triton_activation_and_mul_config(M, N)
    grid = (M * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    activation_and_mul_kernel[grid](
        out_2d,
        input_2d,
        M,
        N,
        input_2d.stride(0),
        input_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        ACT=0,
        **config,
    )


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu_tanh(x):
    # PyTorch GELU with approximate="tanh":
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def activation_quant_kernel(
    q_ptr,
    scale_ptr,
    in_ptr,
    row_multiplier_ptr,
    M,
    D: tl.constexpr,
    ACT: tl.constexpr,
    IS_GATED: tl.constexpr,
    LIMIT: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    QUANT_INT8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_D: tl.constexpr,
    APPLY_ROW_MULTIPLIER: tl.constexpr,
):
    """Fuse a whitelisted MoE activation with per-token INT8/FP8 quant."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < M
    col_mask = cols < D
    input_width = 2 * D if IS_GATED else D
    input_offs = rows[:, None] * input_width + cols[None, :]

    if EVEN_M and EVEN_D:
        x = tl.load(in_ptr + input_offs)
    elif EVEN_D:
        x = tl.load(
            in_ptr + input_offs,
            mask=row_mask[:, None],
            other=0.0,
        )
    else:
        x = tl.load(
            in_ptr + input_offs,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

    xf = x.to(tl.float32)
    # ACT values are private implementation details of the explicit Python
    # whitelist below. Compile-time branches keep each generated kernel small.
    if ACT == _FUSED_ACT_GATED_SILU:
        y = xf * (1.0 / (1.0 + tl.exp(-xf)))
    elif ACT == _FUSED_ACT_GATED_GELU:
        y = xf * 0.5 * (1.0 + tl.erf(xf * 0.7071067811865476))
    elif ACT == _FUSED_ACT_GATED_GELU_TANH:
        y = _gelu_tanh(xf)
    elif ACT == _FUSED_ACT_GATED_SILU_LIMIT:
        gate = tl.minimum(xf, LIMIT)
        y = gate * (1.0 / (1.0 + tl.exp(-gate)))
    elif ACT == _FUSED_ACT_GATED_SITU:
        y = BETA * _tanh(xf / BETA) * tl.sigmoid(xf)
    elif ACT == _FUSED_ACT_GATED_SWIGLU_STEP:
        silu = xf * (1.0 / (1.0 + tl.exp(-xf)))
        y = tl.minimum(silu, LIMIT)
    elif ACT == _FUSED_ACT_SILU:
        y = xf * (1.0 / (1.0 + tl.exp(-xf)))
    elif ACT == _FUSED_ACT_GELU:
        y = xf * 0.5 * (1.0 + tl.erf(xf * 0.7071067811865476))
    elif ACT == _FUSED_ACT_GELU_TANH:
        y = _gelu_tanh(xf)
    else:  # _FUSED_ACT_RELU2
        y = tl.where(xf > 0.0, xf * xf, 0.0)

    if IS_GATED:
        up_offs = input_offs + D
        if EVEN_M and EVEN_D:
            up = tl.load(in_ptr + up_offs)
        elif EVEN_D:
            up = tl.load(
                in_ptr + up_offs,
                mask=row_mask[:, None],
                other=0.0,
            )
        else:
            up = tl.load(
                in_ptr + up_offs,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
        upf = up.to(tl.float32)
        if (
            ACT == _FUSED_ACT_GATED_SILU_LIMIT
            or ACT == _FUSED_ACT_GATED_SWIGLU_STEP
        ):
            upf = tl.minimum(tl.maximum(upf, -LIMIT), LIMIT)
        elif ACT == _FUSED_ACT_GATED_SITU and HAS_LINEAR_BETA:
            upf = LINEAR_BETA * _tanh(upf / LINEAR_BETA)
        y *= upf

    # Match the unfused path: activation stores to the GEMM output dtype,
    # then per-token quantization reloads that rounded value.
    y = y.to(x.dtype)
    absmax = tl.max(tl.abs(y), axis=-1)
    scale = tl.maximum(absmax.to(tl.float32), 1.0e-10) / DTYPE_MAX
    quantized = y * (1.0 / scale)[:, None]
    if QUANT_INT8:
        quantized = libdevice.nearbyint(quantized)
    q = quantized.to(q_ptr.dtype.element_ty)

    if APPLY_ROW_MULTIPLIER:
        multiplier = tl.load(
            row_multiplier_ptr + rows,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        scale *= multiplier

    q_offs = rows[:, None] * D + cols[None, :]
    if EVEN_M and EVEN_D:
        tl.store(q_ptr + q_offs, q, cache_modifier=".cs")
        tl.store(scale_ptr + rows, scale)
    elif EVEN_D:
        tl.store(
            q_ptr + q_offs,
            q,
            mask=row_mask[:, None],
            cache_modifier=".cs",
        )
        tl.store(scale_ptr + rows, scale, mask=row_mask)
    else:
        mask = row_mask[:, None] & col_mask[None, :]
        tl.store(q_ptr + q_offs, q, mask=mask, cache_modifier=".cs")
        tl.store(scale_ptr + rows, scale, mask=row_mask)


def get_triton_gelu_tanh_and_mul_config(M, _N):
    # 2 configs tuned on gfx930 (BW1000B), bf16:
    # small-M: BS128_W1 — stable, low overhead for decode
    # large-M: BS1024_W1 — same access pattern as gated GELU/SiLU
    if M <= 16:
        return {"BLOCK_SIZE_N": 128, "num_warps": 1}
    return {"BLOCK_SIZE_N": 1024, "num_warps": 1}


def get_triton_gelu_tanh_no_mul_config(M, N):
    # 2 configs tuned on gfx930 (BW1000B), bf16:
    # small-M: BS128_W1 — stable, low overhead for decode
    # large-M: BS1024_W2 — same access pattern as non-gated GELU/SiLU
    if M <= 16:
        return {"BLOCK_SIZE_N": 128, "num_warps": 1}
    return {"BLOCK_SIZE_N": 1024, "num_warps": 2}


@triton.heuristics(
    {
        "N_DIV": lambda args: (args["N"] % args["BLOCK_SIZE_N"]) == 0,
    }
)
@triton.jit
def gelu_tanh_and_mul_kernel(
    out_ptr,  # [M, N]
    in_ptr,  # [M, 2N], chunked layout [gate(0:N), up(N:2N)]
    M,
    N: tl.constexpr,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    N_DIV: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    row_in = in_ptr + pid_m.to(tl.int64) * stride_in0
    gate_ptrs = row_in + offs_n * stride_in1
    up_ptrs = row_in + (offs_n + N) * stride_in1
    if N_DIV:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        gate = tl.load(gate_ptrs, mask=mask_n, other=0.0)
        up = tl.load(up_ptrs, mask=mask_n, other=0.0)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    act = _gelu_tanh(gate_f)
    y = act * up_f

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_n * stride_out1
    if N_DIV:
        tl.store(out_ptrs, y.to(gate.dtype))
    else:
        tl.store(out_ptrs, y.to(gate.dtype), mask=mask_n)


@triton.heuristics(
    {
        "N_DIV": lambda args: (args["N"] % args["BLOCK_SIZE_N"]) == 0,
    }
)
@triton.jit
def gelu_tanh_no_mul_1d_kernel(
    out_ptr,  # [M, N]
    in_ptr,  # [M, N]
    M,
    N: tl.constexpr,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    BLOCK_SIZE_N: tl.constexpr,
    N_DIV: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    in_ptrs = in_ptr + pid_m.to(tl.int64) * stride_in0 + offs_n * stride_in1
    if N_DIV:
        x = tl.load(in_ptrs)
    else:
        x = tl.load(in_ptrs, mask=mask_n, other=0.0)

    y = _gelu_tanh(x.to(tl.float32))

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_n * stride_out1
    if N_DIV:
        tl.store(out_ptrs, y.to(x.dtype))
    else:
        tl.store(out_ptrs, y.to(x.dtype), mask=mask_n)


def triton_gelu_tanh_no_mul(out: torch.Tensor, inp: torch.Tensor) -> None:
    # GELU(x, approximate="tanh")
    assert inp.shape == out.shape
    assert inp.is_contiguous() and out.is_contiguous()
    M, N = inp.shape
    config = get_triton_gelu_tanh_no_mul_config(M, N)
    grid = (M * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    gelu_tanh_no_mul_1d_kernel[grid](
        out,
        inp,
        M,
        N,
        inp.stride(0),
        inp.stride(1),
        out.stride(0),
        out.stride(1),
        **config,
    )


def triton_gelu_tanh_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    # Split input into [x, y] along last dim and compute:
    # out = GELU(x, approximate="tanh") * y
    # Shapes: input=[M, 2N], out=[M, N]
    assert input.shape[-1] % 2 == 0
    assert input.is_contiguous()
    assert out.is_contiguous()
    M = input.numel() // input.shape[-1]
    N = input.shape[-1] // 2
    input_2d = input.view(M, input.shape[-1])
    out_2d = out.view(M, N)
    config = get_triton_gelu_tanh_and_mul_config(M, N)
    grid = (M * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    gelu_tanh_and_mul_kernel[grid](
        out_2d,
        input_2d,
        M,
        N,
        input_2d.stride(0),
        input_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        **config,
    )


def triton_gelu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    # Split input into [x, y] along last dim and compute:
    # out = GELU(x) * y
    # Shapes: input=[M, 2N], out=[M, N]
    assert input.shape[-1] % 2 == 0
    assert input.is_contiguous()
    assert out.is_contiguous()
    M = input.numel() // input.shape[-1]
    N = input.shape[-1] // 2
    input_2d = input.view(M, input.shape[-1])
    out_2d = out.view(M, N)
    config = get_triton_activation_and_mul_config(M, N)
    grid = (M * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    activation_and_mul_kernel[grid](
        out_2d,
        input_2d,
        M,
        N,
        input_2d.stride(0),
        input_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        ACT=1,
        **config,
    )


def get_triton_swiglu_variant_config(M, D):
    # 2 configs tuned on gfx930 (BW1000B), bf16:
    # small-M: BS128_W1 — stable, low overhead for decode
    # large-M: BS1024_W1 — ~1.3-2.1x faster than old BS256_W2 at M>=4K
    if M <= 16:
        return {"BLOCK_SIZE_D": 128, "num_warps": 1}
    return {"BLOCK_SIZE_D": 1024, "num_warps": 1}


def get_triton_situ_and_mul_config(M, D):
    # Tuned on gfx938, bf16. Decode-sized inputs are launch/latency bound,
    # while larger inputs benefit from a wider tile without the occupancy
    # loss measured at 1024 elements.
    if M <= 16:
        return {"BLOCK_SIZE_D": 128, "num_warps": 1}
    return {"BLOCK_SIZE_D": 512, "num_warps": 1}


def get_triton_swiglu_oai_interleaved_config(M, D):
    # BLOCK_SIZE_D >= 128 produces incorrect results due to interleaved
    # address pattern (offs_d*2) failing to generate correct buffer_load.
    # Must keep BS64. small-M: W1; large-M: W4 for better occupancy.
    if M <= 16:
        return {"BLOCK_SIZE_D": 64, "num_warps": 1}
    return {"BLOCK_SIZE_D": 64, "num_warps": 4}


_SWIGLU_GPT_OSS_FLAT_BLOCK_SIZE = 1024
_SWIGLU_GPT_OSS_FLAT_MIN_ELEMENTS = 256 * 1024


def _use_swiglu_gpt_oss_flat_kernel(M: int, D: int) -> bool:
    """Select flat mapping when it removes enough row-tile launch overhead."""
    total = M * D
    if total < _SWIGLU_GPT_OSS_FLAT_MIN_ELEMENTS or total >= 2**31:
        return False

    row_block = get_triton_swiglu_variant_config(M, D)["BLOCK_SIZE_D"]
    row_programs = M * triton.cdiv(D, row_block)
    flat_programs = triton.cdiv(total, _SWIGLU_GPT_OSS_FLAT_BLOCK_SIZE)

    # Require at least a 25% program-count reduction. Below this point the
    # flat mapping's per-element row/column calculation offsets its benefit.
    return flat_programs * 4 <= row_programs * 3


@triton.heuristics(
    {
        "EVEN_TOTAL": lambda args: (
            (args["M"] * args["D"]) % args["BLOCK_SIZE"]
        )
        == 0,
    }
)
@triton.jit
def swiglu_gpt_oss_flat_kernel(
    out_ptr,  # contiguous [M, D]
    in_ptr,  # contiguous [M, 2*D]
    M,
    D: tl.constexpr,
    ALPHA,
    LIMIT,
    BLOCK_SIZE: tl.constexpr,
    EVEN_TOTAL: tl.constexpr,
):
    """Process the logical output as one flat array with only one tail tile."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = offs // D
    col = offs - row * D
    gate_ptrs = in_ptr + row.to(tl.int64) * (2 * D) + col
    up_ptrs = gate_ptrs + D

    if EVEN_TOTAL:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        mask = offs < M * D
        gate = tl.load(gate_ptrs, mask=mask, other=0.0)
        up = tl.load(up_ptrs, mask=mask, other=0.0)

    gate_f = tl.minimum(gate.to(tl.float32), LIMIT)
    up_f = tl.minimum(tl.maximum(up.to(tl.float32), -LIMIT), LIMIT)
    y = (
        gate_f
        * (1.0 / (1.0 + tl.exp(-(gate_f * ALPHA))))
        * (up_f + 1.0)
    )

    if EVEN_TOTAL:
        tl.store(out_ptr + offs, y.to(gate.dtype))
    else:
        tl.store(out_ptr + offs, y.to(gate.dtype), mask=mask)


@triton.heuristics(
    {
        "D_DIV": lambda args: (args["D"] % args["BLOCK_SIZE_D"]) == 0,
    }
)
@triton.jit
def swiglu_variant_1d_kernel(
    out_ptr,  # [M, D]
    in_ptr,  # [M, 2*D]
    M,
    D: tl.constexpr,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    ALPHA,
    LIMIT,
    MODE: tl.constexpr,  # 0:gpt_oss 1:silu_clamp_mul 2:step
    BLOCK_SIZE_D: tl.constexpr,
    D_DIV: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(D > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    pid_m = pid // num_pid_d
    pid_d = pid % num_pid_d

    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D

    row_in = in_ptr + pid_m.to(tl.int64) * stride_in0
    gate_ptrs = row_in + offs_d * stride_in1
    up_ptrs = row_in + (offs_d + D) * stride_in1

    if D_DIV:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        gate = tl.load(gate_ptrs, mask=mask_d, other=0.0)
        up = tl.load(up_ptrs, mask=mask_d, other=0.0)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    gate_clamp = tl.minimum(gate_f, LIMIT)
    up_clamp = tl.minimum(tl.maximum(up_f, -LIMIT), LIMIT)

    # MODE=0 (gpt_oss/oai):
    # out = gate' * sigmoid(alpha*gate') * (up' + 1)
    # MODE=1 (silu_clamp_mul):
    # out = SiLU(gate') * up'
    # MODE=2 (step):
    # out = clamp(SiLU(gate), max=LIMIT) * up'
    if MODE == 0:
        y = gate_clamp * (1.0 / (1.0 + tl.exp(-(gate_clamp * ALPHA)))) * (up_clamp + 1.0)
    elif MODE == 1:
        y = gate_clamp * (1.0 / (1.0 + tl.exp(-gate_clamp))) * up_clamp
    else:
        silu_gate = gate_f * (1.0 / (1.0 + tl.exp(-gate_f)))
        silu_clamp = tl.minimum(silu_gate, LIMIT)
        y = silu_clamp * up_clamp

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_d * stride_out1
    if D_DIV:
        tl.store(out_ptrs, y.to(gate.dtype))
    else:
        tl.store(out_ptrs, y.to(gate.dtype), mask=mask_d)


def _triton_chunked_swiglu_variant(
    out: torch.Tensor,
    inp: torch.Tensor,
    *,
    alpha: float,
    limit: float,
    mode: int,
) -> None:
    assert inp.shape[-1] % 2 == 0
    assert inp.is_contiguous() and out.is_contiguous()
    M = inp.numel() // inp.shape[-1]
    D = inp.shape[-1] // 2
    out_rows = out.view(M, D)
    inp_rows = inp.view(M, inp.shape[-1])
    config = get_triton_swiglu_variant_config(M, D)
    # 1D launch: a single pid is mapped to (row_id, d_tile_id) inside kernel.
    grid = (M * triton.cdiv(D, config["BLOCK_SIZE_D"]),)
    swiglu_variant_1d_kernel[grid](
        out_rows,
        inp_rows,
        M,
        D,
        inp_rows.stride(0),
        inp_rows.stride(1),
        out_rows.stride(0),
        out_rows.stride(1),
        alpha,
        limit,
        MODE=mode,
        **config,
    )


def triton_swiglu_silu_clamp_mul(out: torch.Tensor, inp: torch.Tensor, limit: float) -> None:
    # out = SiLU(clamp(gate,max=limit)) * clamp(up,-limit,limit)
    _triton_chunked_swiglu_variant(out, inp, alpha=1.0, limit=limit, mode=1)


def triton_swiglu_gpt_oss_sigmoid_alpha(
    out: torch.Tensor, inp: torch.Tensor, alpha: float, limit: float
) -> None:
    # out = gate' * sigmoid(alpha * gate') * (up' + 1), chunked layout
    assert inp.shape[-1] % 2 == 0
    assert inp.is_contiguous() and out.is_contiguous()
    M = inp.numel() // inp.shape[-1]
    D = inp.shape[-1] // 2

    if not _use_swiglu_gpt_oss_flat_kernel(M, D):
        _triton_chunked_swiglu_variant(
            out, inp, alpha=alpha, limit=limit, mode=0
        )
        return

    block_size = _SWIGLU_GPT_OSS_FLAT_BLOCK_SIZE
    grid = (triton.cdiv(M * D, block_size),)
    swiglu_gpt_oss_flat_kernel[grid](
        out,
        inp,
        M,
        D,
        alpha,
        limit,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )


def triton_swiglu_step_and_mul(out: torch.Tensor, inp: torch.Tensor, limit: float) -> None:
    # out = clamp(SiLU(gate),max=limit) * clamp(up,-limit,limit)
    _triton_chunked_swiglu_variant(out, inp, alpha=1.0, limit=limit, mode=2)


@triton.heuristics(
    {
        "D_DIV": lambda args: (args["D"] % args["BLOCK_SIZE_D"]) == 0,
    }
)
@triton.jit
def swiglu_oai_interleaved_1d_kernel(
    out_ptr,  # [M, D]
    in_ptr,  # [M, 2*D], interleaved layout [g0, u0, g1, u1, ...]
    M,
    D,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    ALPHA,
    LIMIT,
    BLOCK_SIZE_D: tl.constexpr,
    D_DIV: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(D > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)
    tl.assume(in_ptr.to(tl.int64) >= 0)
    tl.assume(out_ptr.to(tl.int64) >= 0)

    pid = tl.program_id(axis=0)
    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    pid_m = pid // num_pid_d
    pid_d = pid % num_pid_d

    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D

    row_in = in_ptr + pid_m.to(tl.int64) * stride_in0
    gate_ptrs = row_in + (offs_d * 2) * stride_in1
    up_ptrs = row_in + (offs_d * 2 + 1) * stride_in1

    if D_DIV:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        gate = tl.load(gate_ptrs, mask=mask_d, other=0.0)
        up = tl.load(up_ptrs, mask=mask_d, other=0.0)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    gate_clamp = tl.minimum(gate_f, LIMIT)
    up_clamp = tl.minimum(tl.maximum(up_f, -LIMIT), LIMIT)

    # OpenAI interleaved SwiGLU variant:
    # out = gate' * sigmoid(alpha * gate') * (up' + 1)
    y = gate_clamp * (1.0 / (1.0 + tl.exp(-(gate_clamp * ALPHA)))) * (up_clamp + 1.0)

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_d * stride_out1
    if D_DIV:
        tl.store(out_ptrs, y.to(gate.dtype))
    else:
        tl.store(out_ptrs, y.to(gate.dtype), mask=mask_d)


def triton_swiglu_oai_and_mul(out: torch.Tensor, inp: torch.Tensor, alpha: float, limit: float) -> None:
    # out = gate' * sigmoid(alpha * gate') * (up' + 1), interleaved layout
    assert inp.shape[-1] % 2 == 0
    assert inp.is_contiguous() and out.is_contiguous()
    M = inp.numel() // inp.shape[-1]
    D = inp.shape[-1] // 2
    out_2d = out.view(M, D)
    inp_2d = inp.view(M, inp.shape[-1])
    config = get_triton_swiglu_oai_interleaved_config(M, D)
    grid = (M * triton.cdiv(D, config["BLOCK_SIZE_D"]),)
    swiglu_oai_interleaved_1d_kernel[grid](
        out_2d,
        inp_2d,
        M,
        D,
        inp_2d.stride(0),
        inp_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        alpha,
        limit,
        **config,
    )


@triton.heuristics(
    {
        "D_DIV": lambda args: (args["D"] % args["BLOCK_SIZE_D"]) == 0,
    }
)
@triton.jit
def situ_and_mul_kernel(
    out_ptr,  # [M, D]
    in_ptr,  # [M, 2D], chunked layout [gate(0:D), up(D:2D)]
    M,
    D: tl.constexpr,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    D_DIV: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Kimi SituGLU: soft-clipped SiLU gate times an optional clipped up."""
    tl.assume(M > 0)
    tl.assume(D > 0)
    tl.assume(stride_in0 > 0)
    tl.assume(stride_in1 > 0)
    tl.assume(stride_out0 > 0)
    tl.assume(stride_out1 > 0)

    pid = tl.program_id(axis=0)
    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    pid_m = pid // num_pid_d
    pid_d = pid % num_pid_d
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D

    row_in = in_ptr + pid_m.to(tl.int64) * stride_in0
    gate_ptrs = row_in + offs_d * stride_in1
    up_ptrs = row_in + (offs_d + D) * stride_in1
    if D_DIV:
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
    else:
        gate = tl.load(gate_ptrs, mask=mask_d, other=0.0)
        up = tl.load(up_ptrs, mask=mask_d, other=0.0)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    gate_out = BETA * _tanh(gate_f / BETA) * tl.sigmoid(gate_f)
    if HAS_LINEAR_BETA:
        up_f = LINEAR_BETA * _tanh(up_f / LINEAR_BETA)
    y = gate_out * up_f

    out_ptrs = out_ptr + pid_m.to(tl.int64) * stride_out0 + offs_d * stride_out1
    if D_DIV:
        tl.store(out_ptrs, y.to(gate.dtype))
    else:
        tl.store(out_ptrs, y.to(gate.dtype), mask=mask_d)


def triton_situ_and_mul(
    out: torch.Tensor,
    inp: torch.Tensor,
    beta: float,
    linear_beta: Optional[float] = None,
) -> None:
    """Apply Kimi SituGLU to a packed ``[gate, up]`` input."""
    if beta <= 0:
        raise ValueError("activation_situ_beta must be greater than 0")
    if linear_beta is not None and linear_beta <= 0:
        raise ValueError("activation_situ_linear_beta must be greater than 0")
    assert inp.shape[-1] % 2 == 0
    assert inp.is_contiguous() and out.is_contiguous()

    M = inp.numel() // inp.shape[-1]
    D = inp.shape[-1] // 2
    inp_2d = inp.view(M, inp.shape[-1])
    out_2d = out.view(M, D)
    config = get_triton_situ_and_mul_config(M, D)
    grid = (M * triton.cdiv(D, config["BLOCK_SIZE_D"]),)
    situ_and_mul_kernel[grid](
        out_2d,
        inp_2d,
        M,
        D,
        inp_2d.stride(0),
        inp_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BETA=float(beta),
        LINEAR_BETA=0.0 if linear_beta is None else float(linear_beta),
        HAS_LINEAR_BETA=linear_beta is not None,
        **config,
    )


_ACTIVATION_ALIASES = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "gelu_pytorch_tanh_no_mul": "gelu_tanh_no_mul",
}


class _ActivationSpec(NamedTuple):
    default_gated: bool
    supported_gated_modes: Tuple[bool, ...]
    # Indexed by int(is_gated): (non-gated kind, gated kind).
    fused_quant_kinds: Tuple[
        Optional[_FusedActivationQuantKind],
        Optional[_FusedActivationQuantKind],
    ]


_ACTIVATION_SPECS = {
    "silu": _ActivationSpec(
        True,
        (False, True),
        (
            _FusedActivationQuantKind.SILU,
            _FusedActivationQuantKind.GATED_SILU,
        ),
    ),
    "gelu": _ActivationSpec(
        True,
        (False, True),
        (
            _FusedActivationQuantKind.GELU,
            _FusedActivationQuantKind.GATED_GELU,
        ),
    ),
    "gelu_tanh": _ActivationSpec(
        True,
        (False, True),
        (
            _FusedActivationQuantKind.GELU_TANH,
            _FusedActivationQuantKind.GATED_GELU_TANH,
        ),
    ),
    "situ": _ActivationSpec(
        True,
        (True,),
        (None, _FusedActivationQuantKind.GATED_SITU),
    ),
    "swigluoai": _ActivationSpec(True, (True,), (None, None)),
    "swiglustep": _ActivationSpec(
        True,
        (True,),
        (None, _FusedActivationQuantKind.GATED_SWIGLU_STEP),
    ),
    "silu_no_mul": _ActivationSpec(
        False, (False,), (_FusedActivationQuantKind.SILU, None)
    ),
    "gelu_no_mul": _ActivationSpec(
        False, (False,), (_FusedActivationQuantKind.GELU, None)
    ),
    "gelu_tanh_no_mul": _ActivationSpec(
        False, (False,), (_FusedActivationQuantKind.GELU_TANH, None)
    ),
    "relu2": _ActivationSpec(
        False, (False,), (_FusedActivationQuantKind.RELU2, None)
    ),
    "relu2_no_mul": _ActivationSpec(
        False, (False,), (_FusedActivationQuantKind.RELU2, None)
    ),
}

# Publicly inspectable allowlist requested by the MoE integration. It is
# derived from the canonical registry so a new activation cannot drift between
# normalization and fused-quant dispatch.
FUSED_ACTIVATION_QUANT_ACTIVATIONS = frozenset(
    (name, is_gated)
    for name, spec in _ACTIVATION_SPECS.items()
    for is_gated in spec.supported_gated_modes
    if spec.fused_quant_kinds[int(is_gated)] is not None
)

# A per-token scale needs a reduction over the whole output row. All current
# model widths fit in one Triton block up to this limit; larger future widths
# safely retain the existing activation-then-quant path.
_MAX_FUSED_ACTIVATION_QUANT_D = 8192


def _resolve_fused_activation_quant_kind(
    activation: str,
    is_gated: bool,
    quant_dtype: torch.dtype,
    output_dim: int,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> Optional[_FusedActivationQuantKind]:
    """Resolve an allowlisted public activation to its internal kernel kind."""
    activation = _ACTIVATION_ALIASES.get(
        activation.lower(), activation.lower()
    )
    spec = _ACTIVATION_SPECS.get(activation)
    if spec is None or is_gated not in spec.supported_gated_modes:
        return None
    if quant_dtype not in (torch.int8, torch.float8_e4m3fn):
        return None
    if output_dim <= 0 or output_dim > _MAX_FUSED_ACTIVATION_QUANT_D:
        return None

    kind = spec.fused_quant_kinds[int(is_gated)]
    if kind is None:
        return None

    if not is_gated:
        return kind
    if kind == _FusedActivationQuantKind.GATED_SILU:
        # Exclude the GPT-OSS alpha+limit formula. The limit-only variant is
        # allowlisted and has separate compile-time formula selection below.
        if gemm1_alpha is not None:
            return None
        if gemm1_limit is not None:
            return _FusedActivationQuantKind.GATED_SILU_LIMIT
    elif kind in {
        _FusedActivationQuantKind.GATED_GELU,
        _FusedActivationQuantKind.GATED_GELU_TANH,
        _FusedActivationQuantKind.GATED_SITU,
    }:
        if gemm1_alpha is not None or gemm1_limit is not None:
            return None
    if kind == _FusedActivationQuantKind.GATED_SITU:
        if activation_situ_beta is None or activation_situ_beta <= 0:
            return None
        if (
            activation_situ_linear_beta is not None
            and activation_situ_linear_beta <= 0
        ):
            return None
    if (
        kind == _FusedActivationQuantKind.GATED_SWIGLU_STEP
        and gemm1_alpha is not None
    ):
        return None
    return kind


def is_fused_activation_quant_supported(
    activation: str,
    is_gated: bool,
    quant_dtype: torch.dtype,
    output_dim: int,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> bool:
    """Return whether this exact activation/quant parameter set is allowlisted."""
    return _resolve_fused_activation_quant_kind(
        activation,
        is_gated,
        quant_dtype,
        output_dim,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    ) is not None


def _get_fused_activation_quant_config(
    output_dim: int,
) -> dict:
    block_d = triton.next_power_of_2(output_dim)
    if output_dim == 128:
        block_m = 8
    else:
        block_m = max(1, min(4, 512 // block_d))
    if block_d <= 512:
        num_warps = 1
    elif block_d <= 2048:
        num_warps = 2
    else:
        num_warps = 4
    return {
        "BLOCK_M": block_m,
        "BLOCK_D": block_d,
        "num_warps": num_warps,
        "num_stages": 1,
    }


def triton_activation_quant(
    input: torch.Tensor,
    activation: str,
    is_gated: bool,
    quant_dtype: torch.dtype,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
    row_multiplier: Optional[torch.Tensor] = None,
):
    """Fuse an explicitly allowlisted MoE activation and per-token quant."""
    if input.ndim != 2 or not input.is_contiguous():
        raise ValueError("input must be a contiguous 2D tensor")
    if is_gated and input.shape[-1] % 2:
        raise ValueError("gated input last dimension must be even")

    activation = _ACTIVATION_ALIASES.get(activation.lower(), activation.lower())
    output_dim = adjust_N_for_activation(input.shape[-1], is_gated)
    act = _resolve_fused_activation_quant_kind(
        activation,
        is_gated,
        quant_dtype,
        output_dim,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )
    if act is None:
        raise ValueError(
            "activation/quant parameters are not supported by the fused kernel"
        )

    rows = input.shape[0]
    if rows <= 0:
        raise ValueError("input must contain at least one row")
    if row_multiplier is not None:
        if row_multiplier.device != input.device:
            raise ValueError("row_multiplier must be on the input device")
        if not row_multiplier.dtype.is_floating_point:
            raise ValueError("row_multiplier must have a floating-point dtype")
        if (
            row_multiplier.ndim != 1
            or not row_multiplier.is_contiguous()
            or row_multiplier.numel() < rows
        ):
            raise ValueError(
                "row_multiplier must be a contiguous vector containing every input row"
            )
    limit = 0.0
    if act == _FusedActivationQuantKind.GATED_SWIGLU_STEP:
        limit = 7.0 if gemm1_limit is None else float(gemm1_limit)
    elif act == _FusedActivationQuantKind.GATED_SILU_LIMIT:
        limit = float(gemm1_limit)

    q = torch.empty(
        (rows, output_dim), dtype=quant_dtype, device=input.device
    )
    scale = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    config = _get_fused_activation_quant_config(output_dim)
    block_m = config["BLOCK_M"]
    activation_quant_kernel[(triton.cdiv(rows, block_m),)](
        q,
        scale,
        input,
        row_multiplier if row_multiplier is not None else input,
        rows,
        D=output_dim,
        ACT=int(act),
        IS_GATED=is_gated,
        LIMIT=limit,
        BETA=(
            0.0 if activation_situ_beta is None else float(activation_situ_beta)
        ),
        LINEAR_BETA=(
            0.0
            if activation_situ_linear_beta is None
            else float(activation_situ_linear_beta)
        ),
        HAS_LINEAR_BETA=activation_situ_linear_beta is not None,
        DTYPE_MAX=(
            torch.finfo(quant_dtype).max
            if quant_dtype.is_floating_point
            else torch.iinfo(quant_dtype).max
        ),
        QUANT_INT8=quant_dtype == torch.int8,
        EVEN_M=rows % block_m == 0,
        EVEN_D=output_dim == config["BLOCK_D"],
        APPLY_ROW_MULTIPLIER=row_multiplier is not None,
        **config,
    )
    return q, scale


def _normalize_activation_and_gate(
    activation: str,
    is_gated: Optional[bool],
) -> Tuple[str, bool]:
    # Match upstream style: keep (activation, is_gated) as a direct pair.
    # 1) validate string, 2) infer is_gated when not explicitly given,
    # 3) validate gated/non-gated constraints.
    activation_name = _ACTIVATION_ALIASES.get(activation.lower(), activation.lower())
    spec = _ACTIVATION_SPECS.get(activation_name)
    if spec is None:
        raise ValueError(f"Unsupported activation: {activation}")

    if is_gated is None:
        is_gated = spec.default_gated

    if is_gated not in spec.supported_gated_modes:
        if spec.supported_gated_modes == (False,):
            raise ValueError(
                f"Activation '{activation}' is non-gated but is_gated=True"
            )
        raise ValueError(f"Activation '{activation}' requires gated mode")

    return activation_name, is_gated


def adjust_N_for_activation(n: int, is_gated: bool) -> int:
    """Align with vLLM naming: output dim after expert activation."""
    return n // 2 if is_gated else n


def normalize_moe_activation(
    activation: str,
    is_gated: Optional[bool] = None,
) -> Tuple[str, bool]:
    """Normalize an activation name and resolve whether it is gated."""
    return _normalize_activation_and_gate(activation, is_gated)


def moe_activation_output_size(n: int, is_gated: bool) -> int:
    """Return the last-dimension size produced by an MoE activation."""
    if n <= 0:
        raise ValueError("input last dimension must be greater than zero")
    if is_gated and n % 2:
        raise ValueError("gated input last dimension must be even")
    return adjust_N_for_activation(n, is_gated)


def _apply_activation(
    activation: str,
    is_gated: bool,
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> None:
    """Apply MoE expert activation.

    Shapes:
    - gated path: `ffn1_out_2d` is [rows, 2D], `activated_out` is [rows, D]
    - non-gated path: `ffn1_out_2d` and `activated_out` are both [rows, D]

    Meaning of `is_gated`:
    - True: run gate-up fusion (activation(gate) * up).
    - False: run plain activation without gate multiplication.
    """
    if is_gated:
        if activation == "silu":
            # Gated SiLU family:
            # base: out = SiLU(gate) * up
            # gpt-oss(alpha,limit): out = gate'*sigmoid(alpha*gate')*(up'+1)
            # limit-only: out = SiLU(gate')*up'
            if gemm1_alpha is not None:
                if gemm1_limit is None:
                    raise ValueError("gemm1_limit must be set when gemm1_alpha is set")
                triton_swiglu_gpt_oss_sigmoid_alpha(
                    activated_out,
                    ffn1_out_2d,
                    gemm1_alpha,
                    gemm1_limit,
                )
            elif gemm1_limit is not None:
                triton_swiglu_silu_clamp_mul(
                    activated_out,
                    ffn1_out_2d,
                    gemm1_limit,
                )
            else:
                triton_silu_and_mul(activated_out, ffn1_out_2d)
        elif activation == "gelu":
            # Gated GELU: out = GELU(gate) * up
            if gemm1_alpha is not None or gemm1_limit is not None:
                raise ValueError("gemm1_alpha/gemm1_limit are not supported for gated GELU")
            triton_gelu_and_mul(activated_out, ffn1_out_2d)
        elif activation == "gelu_tanh":
            # Gated GELU (tanh approx): out = GELU(gate, approximate="tanh") * up
            if gemm1_alpha is not None or gemm1_limit is not None:
                raise ValueError("gemm1_alpha/gemm1_limit are not supported for gated GELU_TANH")
            triton_gelu_tanh_and_mul(activated_out, ffn1_out_2d)
        elif activation == "situ":
            if gemm1_alpha is not None or gemm1_limit is not None:
                raise ValueError("gemm1_alpha/gemm1_limit are not supported for SITU")
            if activation_situ_beta is None:
                raise ValueError("SITU requires activation_situ_beta")
            triton_situ_and_mul(
                activated_out,
                ffn1_out_2d,
                activation_situ_beta,
                activation_situ_linear_beta,
            )
        elif activation == "swigluoai":
            # OpenAI interleaved SwiGLU variant:
            # out = gate'*sigmoid(alpha*gate')*(up'+1)
            alpha = 1.702 if gemm1_alpha is None else gemm1_alpha
            limit = 7.0 if gemm1_limit is None else gemm1_limit
            triton_swiglu_oai_and_mul(activated_out, ffn1_out_2d, alpha, limit)
        elif activation == "swiglustep":
            # Step variant:
            # out = clamp(SiLU(gate),max=limit) * clamp(up,-limit,limit)
            limit = 7.0 if gemm1_limit is None else gemm1_limit
            if gemm1_alpha is not None:
                raise ValueError("gemm1_alpha is not supported for swiglustep")
            triton_swiglu_step_and_mul(activated_out, ffn1_out_2d, limit)
        else:
            raise ValueError(f"Unsupported gated activation: {activation}")
        return

    if activation in {"silu", "silu_no_mul"}:
        # Non-gated: out = SiLU(x)
        triton_silu_no_mul(activated_out, ffn1_out_2d)
    elif activation in {"gelu", "gelu_no_mul"}:
        # Non-gated: out = GELU(x)
        triton_gelu_no_mul(activated_out, ffn1_out_2d)
    elif activation in {"gelu_tanh", "gelu_tanh_no_mul"}:
        # Non-gated: out = GELU(x, approximate="tanh")
        triton_gelu_tanh_no_mul(activated_out, ffn1_out_2d)
    elif activation in {"relu2", "relu2_no_mul"}:
        # Non-gated: out = ReLU(x)^2
        triton_relu2(activated_out, ffn1_out_2d)
    else:
        raise ValueError(f"Unsupported non-gated activation: {activation}")


def moe_activation(
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    activation: str = "silu",
    is_gated: Optional[bool] = None,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> None:
    """Apply an MoE activation into a caller-owned output tensor.

    This is the public integration API for aiter's ``moe_c``, ``moe_asm`` and
    ``moe_triton`` paths.
    """
    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)
    if ffn1_out_2d.ndim != 2 or activated_out.ndim != 2:
        raise ValueError("ffn1_out_2d and activated_out must both be 2D")
    if not ffn1_out_2d.is_cuda or not activated_out.is_cuda:
        raise ValueError("moe_activation only supports CUDA/HIP tensors")
    if ffn1_out_2d.device != activated_out.device:
        raise ValueError("activated_out and ffn1_out_2d must be on the same device")
    if not ffn1_out_2d.is_floating_point() or not activated_out.is_floating_point():
        raise TypeError("activation input and output must have floating-point dtypes")
    if not ffn1_out_2d.is_contiguous() or not activated_out.is_contiguous():
        raise ValueError("activation input and output must be contiguous")

    rows, n = ffn1_out_2d.shape
    output_n = moe_activation_output_size(n, is_gated)
    if activated_out.shape != (rows, output_n):
        raise ValueError(
            f"activated_out shape must be {(rows, output_n)}, got {tuple(activated_out.shape)}"
        )
    if ffn1_out_2d.numel() == 0:
        return

    _apply_activation(
        activation=activation,
        is_gated=is_gated,
        activated_out=activated_out,
        ffn1_out_2d=ffn1_out_2d,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )
