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
