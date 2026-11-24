# SPDX-License-Identifier: MIT
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/_triton_kernels/common/activation.py
"""Activation helpers used by vendored aiter GEMM-A16W16 (slim copy).

Source: ``aiter/ops/triton/_triton_kernels/activation.py`` (activation mapping
only; quant / fused kernels omitted).
"""

import tilelang.language as T


def _silu_exp2(x):
    return x / (1.0 + T.exp2(-(x * 1.44269504089)))


def _silu(x):
    return _silu_exp2(x)


def _tanh(x):
    return 2 * T.sigmoid(2 * x) - 1


def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + T.erf(x * ALPHA))


def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


def _relu(x):
    return T.max(0.0, x)


def _relu6(x):
    return T.min(T.max(0.0, x), 6.0)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
        "relu6": _relu6,
    }
    return mapping[activation]
