# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Reuse existing batched native16 and explicitly quantized A16WFP4 kernels."""
import torch
from ..registry import Implementation, KernelConfig


def _native_configs(p):
    if (p.kind != "batched" or not p.native_operands or not p.unscaled or
            p.a_dtype != p.b_dtype or p.a_dtype not in (torch.float16, torch.bfloat16) or
            p.out_dtype not in (torch.float16, torch.bfloat16) or p.b.strides is None):
        return ()
    layout = "tn" if p.b.strides[-1] == 1 else "tt" if p.b.strides[-2] == 1 else None
    if layout is None:
        return ()
    from .batched_gemm_bf16 import _get_config
    cfg = _get_config(p.m, p.n, p.k, p.out_dtype, B=p.batch, layout=layout)
    return (KernelConfig(*(cfg["BLOCK_SIZE_" + d] for d in ("M", "N", "K"))),)


def launch_native(config, p, a, b, bias, out):
    from .batched_gemm_bf16 import batched_gemm_bf16, _get_config, _infer_weight_layout
    cfg = _get_config(p.m, p.n, p.k, out.dtype, B=p.batch, layout=_infer_weight_layout(a, b))
    batched_gemm_bf16(a, b, bias, dtype=out.dtype, y=out, config=cfg)


IMPLEMENTATIONS = (
    Implementation("triton.batched.native16", "batched", __name__, "launch_native", _native_configs,
                   lambda p, c: c in _native_configs(p), backend="triton"),
)
