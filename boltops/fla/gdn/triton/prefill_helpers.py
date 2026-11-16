"""Projection unpacking, scalar gates and output normalization for GDN."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_matrix(tensor, name):
    if tensor.ndim != 2 or tensor.dtype not in _FLOAT_DTYPES:
        raise ValueError(f'{name} must be a 2D FP16, BF16 or FP32 tensor')
    if tensor.layout != torch.strided or any(stride <= 0 for stride in tensor.stride()):
        raise ValueError(f'{name} must have positive storage strides')


def _check_vector(tensor, name, width, device):
    if tensor.ndim != 1 or tensor.shape[0] != width or tensor.dtype not in _FLOAT_DTYPES:
        raise ValueError(f'{name} must be a floating vector of length {width}')
    if tensor.device != device or tensor.stride(0) <= 0:
        raise ValueError(f'{name} must have a positive stride and share the input device')


@triton.jit
def _gdn_qkvzba_split_kernel(
    projection, gate_projection, qkv_output, z_output, b_output, a_output,
    projection_row_stride: tl.constexpr, projection_col_stride: tl.constexpr,
    gate_row_stride: tl.constexpr, gate_col_stride: tl.constexpr,
    QKV_WIDTH: tl.constexpr, Z_WIDTH: tl.constexpr, HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(projection + row * projection_row_stride + columns * projection_col_stride,
                     mask=columns < QKV_WIDTH + Z_WIDTH, other=0)
    tl.store(qkv_output + row * QKV_WIDTH + columns, values, mask=columns < QKV_WIDTH)
    tl.store(z_output + row * Z_WIDTH + columns - QKV_WIDTH, values,
             mask=(columns >= QKV_WIDTH) & (columns < QKV_WIDTH + Z_WIDTH))
    if tl.program_id(0) * BLOCK < 2 * HEADS:
        gates = tl.load(gate_projection + row * gate_row_stride + columns * gate_col_stride,
                        mask=columns < 2 * HEADS, other=0)
        tl.store(b_output + row * HEADS + columns, gates, mask=columns < HEADS)
        tl.store(a_output + row * HEADS + columns - HEADS, gates,
                 mask=(columns >= HEADS) & (columns < 2 * HEADS))


def gdn_qkvzba_split(
    projected_qkvz, projected_ba, num_heads_qk, num_heads_v,
    head_size_qk=128, head_size_v=128,
):
    """Materialize contiguous ``qkv, z, b, a`` from Qwen3.5 projections.

    Inputs are token-first matrices with contiguous logical segments
    ``[all_q | all_k | all_v | all_z]`` and ``[all_b | all_a]``.
    Both head counts and both head dimensions must be positive integers.
    ``z`` is returned flattened to ``[tokens, num_heads_v * head_size_v]``.
    Positive row/column storage strides are supported; outputs do not alias inputs.
    """
    _check_matrix(projected_qkvz, 'projected_qkvz')
    _check_matrix(projected_ba, 'projected_ba')
    dimensions = (num_heads_qk, num_heads_v, head_size_qk, head_size_v)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError('head counts and head sizes must be positive integers')
    if num_heads_v % num_heads_qk:
        raise ValueError('num_heads_v must be divisible by num_heads_qk')
    if projected_qkvz.device != projected_ba.device or projected_qkvz.dtype != projected_ba.dtype:
        raise ValueError('projection inputs must share device and dtype')
    tokens = projected_qkvz.shape[0]
    value_width = num_heads_v * head_size_v
    qkv_width = 2 * num_heads_qk * head_size_qk + value_width
    if projected_qkvz.shape[1] != qkv_width + value_width or projected_ba.shape != (tokens, 2 * num_heads_v):
        raise ValueError('projection shapes do not match the supplied head counts and sizes')
    outputs = tuple(torch.empty((tokens, width), dtype=projected_qkvz.dtype, device=projected_qkvz.device)
                    for width in (qkv_width, value_width, num_heads_v, num_heads_v))
    if tokens:
        _gdn_qkvzba_split_kernel[(triton.cdiv(qkv_width + value_width, 512), tokens)](
            projected_qkvz, projected_ba, *outputs,
            *projected_qkvz.stride(), *projected_ba.stride(),
            QKV_WIDTH=qkv_width, Z_WIDTH=value_width, HEADS=num_heads_v, BLOCK=512,
            num_warps=4,
        )
    return outputs


@triton.jit
def _gdn_fused_gating_kernel(
    a_projection, b_projection, log_decay, time_bias, decay_output, beta_output,
    a_row_stride: tl.constexpr, a_col_stride: tl.constexpr,
    b_row_stride: tl.constexpr, b_col_stride: tl.constexpr,
    decay_stride: tl.constexpr, bias_stride: tl.constexpr,
    TOTAL: tl.constexpr, HEADS: tl.constexpr, SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // HEADS
    heads = offsets % HEADS
    valid = offsets < TOTAL
    a_value = tl.load(a_projection + rows * a_row_stride + heads * a_col_stride,
                      mask=valid, other=0).to(tl.float32)
    b_value = tl.load(b_projection + rows * b_row_stride + heads * b_col_stride,
                      mask=valid, other=0).to(tl.float32)
    log_value = tl.load(log_decay + heads * decay_stride, mask=valid, other=0).to(tl.float32)
    bias_value = tl.load(time_bias + heads * bias_stride, mask=valid, other=0).to(tl.float32)
    shifted = a_value + bias_value
    scaled = SOFTPLUS_BETA * shifted
    softplus = (tl.maximum(scaled, 0) + libdevice.log1p(tl.exp(-tl.abs(scaled)))) / SOFTPLUS_BETA
    softplus = tl.where(scaled > SOFTPLUS_THRESHOLD, shifted, softplus)
    tl.store(decay_output + offsets, -tl.exp(log_value) * softplus, mask=valid)
    tl.store(beta_output + offsets, tl.sigmoid(b_value), mask=valid)


def gdn_fused_gating(a_projection, b_projection, A_log, dt_bias,
                     softplus_beta=1.0, softplus_threshold=20.0):
    """Return FP32 ``g, beta`` without state update or cumulative summation.

    ``g = -exp(A_log) * softplus(a_projection + dt_bias)`` and
    ``beta = sigmoid(b_projection)``. Inputs are ``[tokens, value_heads]``;
    softplus parameters have the same meaning as ``torch.nn.functional.softplus``.
    """
    _check_matrix(a_projection, 'a_projection')
    _check_matrix(b_projection, 'b_projection')
    if (a_projection.shape != b_projection.shape or a_projection.dtype != b_projection.dtype
            or a_projection.device != b_projection.device):
        raise ValueError('gate projections must share shape, dtype and device')
    tokens, heads = a_projection.shape
    if heads == 0:
        raise ValueError('gate projections must have a nonzero head dimension')
    _check_vector(A_log, 'A_log', heads, a_projection.device)
    _check_vector(dt_bias, 'dt_bias', heads, a_projection.device)
    if (not math.isfinite(softplus_beta) or softplus_beta <= 0
            or not math.isfinite(softplus_threshold)):
        raise ValueError('softplus beta must be positive and both parameters must be finite')
    outputs = tuple(torch.empty((tokens, heads), device=a_projection.device, dtype=torch.float32)
                    for _ in range(2))
    if tokens:
        _gdn_fused_gating_kernel[(triton.cdiv(tokens * heads, 256),)](
            a_projection, b_projection, A_log, dt_bias, *outputs,
            *a_projection.stride(), *b_projection.stride(), A_log.stride(0), dt_bias.stride(0),
            TOTAL=tokens * heads, HEADS=heads, SOFTPLUS_BETA=softplus_beta,
            SOFTPLUS_THRESHOLD=softplus_threshold, BLOCK=256, num_warps=4,
        )
    return outputs


@triton.jit
def _gdn_rmsnorm_gated_kernel(
    hidden, gate, weight, output,
    hidden_row_stride: tl.constexpr, hidden_col_stride: tl.constexpr,
    gate_row_stride: tl.constexpr, gate_col_stride: tl.constexpr,
    weight_stride: tl.constexpr, WIDTH: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(hidden + row * hidden_row_stride + columns * hidden_col_stride,
                     mask=columns < WIDTH, other=0).to(tl.float32)
    gates = tl.load(gate + row * gate_row_stride + columns * gate_col_stride,
                    mask=columns < WIDTH, other=0).to(tl.float32)
    weights = tl.load(weight + columns * weight_stride, mask=columns < WIDTH, other=0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, 0) / WIDTH + EPS)
    normalized = values * inverse_rms * weights
    result = normalized * (gates * tl.sigmoid(gates))
    tl.store(output + row * WIDTH + columns, result, mask=columns < WIDTH)


def gdn_rmsnorm_gated(hidden, gate, weight, eps=1e-6):
    """Return ``RMSNorm(hidden, weight, eps) * SiLU(gate)`` in hidden dtype.

    Normalization and gating use FP32 intermediates; normalization precedes
    gating. Inputs are ``[rows, head_size]``, with a shared ``[head_size]`` weight.
    """
    _check_matrix(hidden, 'hidden')
    _check_matrix(gate, 'gate')
    if hidden.shape != gate.shape or hidden.dtype != gate.dtype or hidden.device != gate.device:
        raise ValueError('hidden and gate must share shape, dtype and device')
    rows, width = hidden.shape
    if width == 0 or width > 65536 // hidden.element_size():
        raise ValueError('head size must be positive and fit within 64 KiB')
    _check_vector(weight, 'weight', width, hidden.device)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    output = torch.empty(hidden.shape, device=hidden.device, dtype=hidden.dtype)
    if rows:
        _gdn_rmsnorm_gated_kernel[(rows,)](
            hidden, gate, weight, output, *hidden.stride(), *gate.stride(), weight.stride(0),
            WIDTH=width, EPS=eps, BLOCK=triton.next_power_of_2(width),
            num_warps=4 if width <= 1024 else 8,
        )
    return output
