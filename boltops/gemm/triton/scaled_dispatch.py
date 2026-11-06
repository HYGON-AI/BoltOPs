# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Adapt existing scaled kernels without adding a device implementation."""
import torch

from ..registry import Implementation, KernelConfig

_FLOAT_SCALES = (torch.float16, torch.bfloat16, torch.float32)
_INPUTS = (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16, torch.int8)
_OUTPUTS = (torch.float16, torch.bfloat16, torch.float32)


def _common(p):
    return (p.kind == 'dense' and p.native_operands and p.a_dtype == p.b_dtype and
            p.a_dtype in _INPUTS and p.out_dtype in _OUTPUTS and
            p.a.strides == (p.k, 1) and p.b.strides == (p.k, 1) and
            all(s is not None and s.encoding == 'numeric' and s.dtype in _FLOAT_SCALES
                for s in (p.a_scale, p.b_scale)))


def _block_config(p):
    if not _common(p) or p.has_bias:
        return None
    am, ak = p.a_scale.block
    bn, bk = p.b_scale.block
    if am != 1 or ak != bk or min(bn, bk) < 16 or bn & (bn - 1) or bk & (bk - 1):
        return None
    if p.a_dtype == torch.int8:
        from .w8a8_block_int8_matmul import get_w8a8_block_int8_configs as get_configs
        from .w8a8_block_int8_matmul import _filter_launch_config
    else:
        from .w8a8_triton_block_scaled_mm import get_w8a8_block_fp8_configs as get_configs
        _filter_launch_config = dict
    configs = get_configs(p.n, p.k, bn, bk)
    cfg = (_filter_launch_config(configs[min(configs, key=lambda m: abs(m - p.m))]) if configs else
           {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': bn, 'BLOCK_SIZE_K': bk})
    # An iteration must not cross a scale-group boundary.
    if cfg['BLOCK_SIZE_K'] > bk or bk % cfg['BLOCK_SIZE_K']:
        return None
    return cfg


def _block_configs(p):
    config = _block_config(p)
    return () if config is None else (KernelConfig(*(config["BLOCK_SIZE_" + d] for d in ("M", "N", "K"))),)


def _row_configs(p):
    if not _common(p):
        return ()
    am, ak = p.a_scale.block
    bn, bk = p.b_scale.block
    if (am not in (1, p.m) or bn not in (1, p.n) or ak < p.k or bk < p.k or
            p.a_scale.dtype != p.b_scale.dtype):
        return ()
    return (KernelConfig(32, 64, 32),)


def launch_block_scaled(config, problem, a, b, a_scale, b_scale, out):
    if a.dtype == torch.int8:
        from .w8a8_block_int8_matmul import _w8a8_block_int8_matmul as kernel
    else:
        from .w8a8_triton_block_scaled_mm import _w8a8_triton_block_scaled_mm as kernel
    cfg = dict(_block_config(problem))
    cfg.setdefault('GROUP_SIZE_M', 32)
    cfg.setdefault('num_warps', 4)
    cfg.setdefault('num_stages', 2)
    m, n, k = problem.m, problem.n, problem.k
    grid = ((m + config.block_m - 1) // config.block_m *
            ((n + config.block_n - 1) // config.block_n),)
    kernel[grid](
        a, b, out, a_scale, b_scale, m, n, k, *problem.b_scale.block,
        *a.stride(), b.stride(1), b.stride(0), *out.stride(),
        *a_scale.stride(), b_scale.stride(1), b_scale.stride(0), **cfg)


def launch_row_scaled(config, problem, a, b, a_scale, b_scale, out, bias=None):
    import triton.language as tl
    from .triton_scaled_mm import scaled_mm_kernel
    m, n, k = problem.m, problem.n, problem.k
    grid = ((m + config.block_m - 1) // config.block_m *
            ((n + config.block_n - 1) // config.block_n),)
    scaled_mm_kernel[grid](
        a, b, a_scale, b_scale, out, bias, m, n, k,
        *a.stride(), b.stride(1), b.stride(0), *out.stride(), tl.int32 if a.dtype == torch.int8 else tl.float32,
        BLOCK_SIZE_M=config.block_m, BLOCK_SIZE_N=config.block_n, BLOCK_SIZE_K=config.block_k,
        BLOCK_SIZE_SCALE_A=1 if a_scale.shape[0] == 1 else config.block_m,
        BLOCK_SIZE_SCALE_B=1 if b_scale.shape[0] == 1 else config.block_n)


IMPLEMENTATIONS = (
    Implementation('triton.scaled.block_fp8', 'dense', __name__, 'launch_block_scaled',
                   _block_configs, lambda p, c: c in _block_configs(p), backend='triton'),
    Implementation('triton.scaled.row_fp8', 'dense', __name__, 'launch_row_scaled',
                   _row_configs, lambda p, c: c in _row_configs(p), backend='triton'),
)
