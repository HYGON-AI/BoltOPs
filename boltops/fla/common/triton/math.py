# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/vllm/third_party/flash_linear_attention/ops/op.py
"""Triton math aliases used by the upstream KDA kernels."""

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tl.exp2
    log = tldevice.fast_logf
else:
    exp = tl.exp
    exp2 = tl.exp2
    log = tl.log


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit
def safe_exp2(x):
    return tl.math.exp2(tl.where(x <= 0, x, float("-inf")))
