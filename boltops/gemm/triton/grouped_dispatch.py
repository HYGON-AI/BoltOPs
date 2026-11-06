# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Reuse equal 16-bit batched GEMM and existing equal/ragged FP8 x FP4 GEMM."""
import torch

from ..registry import Implementation, KernelConfig


def _native(p):
    return (p.kind == 'grouped' and p.groups.mode == 'equal' and p.native_operands and p.unscaled and
            p.a_dtype == p.b_dtype and p.a_dtype in (torch.float16, torch.bfloat16) and
            p.out_dtype in (torch.float16, torch.bfloat16))


def _native_config(p):
    if not _native(p):
        return ()
    if p.b.strides is None:
        return ()
    layout = 'tn' if p.b.strides[-1] == 1 else 'tt' if p.b.strides[-2] == 1 else None
    if layout is None:
        return ()
    from .batched_gemm_bf16 import _get_config
    cfg = _get_config(p.m // p.groups.count, p.n, p.k, p.out_dtype, B=p.groups.count, layout=layout)
    return (KernelConfig(*(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K'))),)


def launch_grouped_native(config, problem, a, b, a_scale, b_scale, group_offsets, bias, out):
    from .batched_gemm_bf16 import batched_gemm_bf16, _get_config, _infer_weight_layout
    cfg = _get_config(a.shape[1], problem.n, problem.k, out.dtype,
                      B=problem.groups.count, layout=_infer_weight_layout(a, b))
    if tuple(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K')) != (config.block_m, config.block_n, config.block_k):
        raise ValueError('selected grouped tile disagrees with backend configuration')
    batched_gemm_bf16(a, b, bias, dtype=out.dtype, y=out, config=cfg)


IMPLEMENTATIONS = (
    Implementation('triton.grouped.native16', 'grouped', __name__, 'launch_grouped_native', _native_config,
                   lambda p, c: c in _native_config(p), backend='triton'),
)
