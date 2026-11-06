# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Adapters reuse the existing dense kernels and gfx946 FP16 recipes."""
import json
from pathlib import Path

import torch

from ..performance_map import tuned_configs
from ..registry import Implementation, KernelConfig

_TUNED = "triton.dense.fp16"


def _native_dense(p):
    return p.kind == "dense" and p.native_operands and p.unscaled and p.a_dtype == p.b_dtype


def _tuned_configurations(p):
    return tuned_configs(p, _TUNED, backend="triton") if _native_dense(p) else ()


def _general_config(p):
    if p.a_dtype in (torch.float16, torch.bfloat16):
        from .gemm_a16w16_op import _get_config
        from .gemm_config_utils import compute_splitk_params
        cfg = dict(_get_config(p.m, p.n, p.k)[0])
        cfg["NUM_KSPLIT"] = 1
        return compute_splitk_params(cfg, p.k)
    from .matmul_persistent import _launch_config
    return dict(_launch_config(p.a_dtype))


def _general_configurations(p):
    if not _native_dense(p) or p.a_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return ()
    if p.out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return ()
    if p.a_dtype == torch.float32 and (p.out_dtype != torch.float32 or not p.num_cus):
        return ()
    cfg = _general_config(p)
    return (KernelConfig(*(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K'))),)


def launch_dense(config, problem, a, b, bias, out):
    if a.dtype in (torch.float16, torch.bfloat16):
        from .gemm_a16w16 import gemm_a16w16
        gemm_a16w16(a, b, bias, dtype=out.dtype, y=out, config=_general_config(problem))
    else:
        from .matmul_persistent import matmul_kernel_persistent
        cfg = _general_config(problem)
        grid = (min(problem.num_cus, ((problem.m + config.block_m - 1) // config.block_m) * ((problem.n + config.block_n - 1) // config.block_n)),)
        matmul_kernel_persistent[grid](
            a, b.T, out, bias, problem.m, problem.n, problem.k,
            *a.stride(), *b.T.stride(), *out.stride(), NUM_SMS=problem.num_cus,
            A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31,
            C_LARGE=out.numel() > 2**31, HAS_BIAS=bias is not None, **cfg)


def launch_dense_fp16(config, problem, a, b, bias, out):
    from .gemm_fp16 import gemm_fp16, mode_launch_kwargs
    root = Path(__file__).resolve().parents[1]
    path = (root / config.source_config).resolve()
    if root not in path.parents:
        raise ValueError("dense config must be inside the GEMM package")
    recipe = json.loads(path.read_text())
    if tuple(recipe['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K')) != (config.block_m, config.block_n, config.block_k):
        raise ValueError("selected tile disagrees with the existing FP16 recipe")
    launch = mode_launch_kwargs(recipe["mode"])
    keys = {'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M',
            'WARP_SPECIALIZE', 'num_warps', 'num_stages', 'empty_arrive_after_mmac',
            'enable_v_mmac_cluster', 'mmac_layout_force', 'wasp_wdra',
            'wasp_partition_warps', 'wasp_partition_regs'}
    launch.update({k: tuple(v) if isinstance(v, list) else v for k, v in recipe.items() if k in keys})
    if "TRITON_BUFFER_CACHE_SWIZZLE" in recipe:
        launch['buffer_cache_swizzle'] = recipe['TRITON_BUFFER_CACHE_SWIZZLE']
    gemm_fp16(a, b.T, bias, out=out, wasp=recipe['mode'] != 'nowasp',
              wdra=recipe['mode'].startswith('wdra'),
              use_mls=recipe.get('use_mls', False),
              use_matrix_store=recipe.get('use_matrix_store', False),
              autotune=False, config=launch)


IMPLEMENTATIONS = (
    Implementation(_TUNED, "dense", __name__, "launch_dense_fp16", _tuned_configurations,
                   lambda p, c: c in _tuned_configurations(p), backend="triton"),
    Implementation("triton.dense.native", "dense", __name__, "launch_dense", _general_configurations,
                   lambda p, c: c in _general_configurations(p), backend="triton"),
)
