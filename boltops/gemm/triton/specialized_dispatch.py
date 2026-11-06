# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Adapters for distinct native quantized computations already in the backend."""
import torch
from ..registry import Implementation, KernelConfig


def _awq(p):
    return (p.interface == "scaled" and p.b.packing == "awq" and p.a_dtype in (torch.float16, torch.bfloat16) and
            p.b_dtype == torch.int32 and p.b.zero_point_dtype == torch.int32 and
            p.b_scale is not None and p.b_scale.dtype == p.out_dtype == p.a_dtype)


def _awq_configs(p):
    return (KernelConfig(16, 64, 32),) if _awq(p) else ()


def launch_awq(c, p, a, b, b_scale, b_zero_points, out):
    from .awq_gemm_triton import awq_gemm_kernel
    grid = ((p.m + c.block_m - 1) // c.block_m * ((p.n + c.block_n - 1) // c.block_n), 1)
    awq_gemm_kernel[grid](a, b, out, b_zero_points, b_scale, p.m, p.n, p.k, p.b_scale.block[1],
                         BLOCK_SIZE_M=c.block_m, BLOCK_SIZE_N=c.block_n, BLOCK_SIZE_K=c.block_k,
                         GROUP_SIZE_M=8, SPLIT_K=1)


def fp4_config(p):
    from .gemm_afp4wfp4 import _get_config
    cfg = dict(_get_config(p.m, p.n, p.k // 2)[0])
    cfg['NUM_KSPLIT'] = 1
    cfg['BLOCK_SIZE_K'] = min(cfg['BLOCK_SIZE_K'], 1 << (p.k - 1).bit_length())
    cfg['SPLITK_BLOCK_SIZE'] = p.k
    return cfg


def _fp4_configs(p):
    if (p.arch != 'gfx946' or p.interface not in ('scaled', 'batched_a16wfp4_online_quant') or
            p.out_dtype not in (torch.float16, torch.bfloat16)):
        return ()
    if (p.k % 32 or p.b_dtype != torch.uint8 or p.b.format != 'fp4' or
            p.b.packing != 'e2m1_low_high' or p.b_scale is None or
            p.b_scale.dtype != torch.uint8 or p.b_scale.encoding != 'e8m0' or
            p.b_scale.block != (1, 32)):
        return ()
    if p.interface == 'scaled':
        if (p.a_dtype != torch.uint8 or p.a.packing != 'e2m1_low_high' or p.a_scale is None or
                p.a_scale.dtype != torch.uint8 or p.a_scale.encoding != 'e8m0' or
                p.a_scale.block != (1, 32)):
            return ()
    elif p.a_dtype not in (torch.float16, torch.bfloat16) or p.a_scale is not None:
        return ()
    cfg = fp4_config(p)
    return (KernelConfig(*(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K'))),)


def launch_mxfp4(c, p, a, b, a_scale, b_scale, out):
    from .gemm_afp4wfp4 import gemm_afp4wfp4
    gemm_afp4wfp4(a, b, a_scale, b_scale, dtype=out.dtype, y=out, config=fp4_config(p))


def launch_a16wfp4(c, p, a, b, b_scale, out):
    from .batched_gemm_a16wfp4 import batched_gemm_a16wfp4
    batched_gemm_a16wfp4(a, b, b_scale, dtype=out.dtype, y=out, config=fp4_config(p), prequant=True)


SCALED_IMPLEMENTATIONS = (
    Implementation('triton.scaled.awq', 'dense', __name__, 'launch_awq', _awq_configs,
                   lambda p, c: c in _awq_configs(p), backend='triton'),
    Implementation('triton.scaled.mxfp4', 'dense', __name__, 'launch_mxfp4', _fp4_configs,
                   lambda p, c: p.interface == 'scaled' and p.a.format == 'fp4' and c in _fp4_configs(p), backend='triton'),
)
IMPLEMENTATIONS = {
    'batched_a16wfp4_online_quant': (Implementation('triton.batched_a16wfp4_online_quant', 'batched', __name__, 'launch_a16wfp4',
                       _fp4_configs, lambda p, c: p.interface == 'batched_a16wfp4_online_quant' and c in _fp4_configs(p), backend='triton'),),
}
