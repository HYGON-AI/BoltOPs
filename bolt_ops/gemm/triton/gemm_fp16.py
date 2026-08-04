"""FP16 GEMM Triton kernel with optional WASP/WDRA (gfx946) and barrier knobs.

Public call shape follows vLLM's Triton matmul wrappers
(``matmul_persistent`` / ``matmul_batch_invariant``):

    gemm_fp16(a, b, bias=None, *, out=None, ...)

Kernel launch options follow ``triton-staging`` HCU regression
(``test_wasp_mls_regression.py`` / ``bench_gemm_wasp_barrier_autotune.py``).

Autotune dimensions (MR !208 HIPOptions):
  * ``empty_arrive_after_mmac`` — bool
  * ``enable_v_mmac_cluster`` — 0 | 1 | 2
plus a small BLOCK_SIZE_{M,N,K} sweep for the selected WASP/WDRA mode.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from bolt_ops.utility import arch_info

# WASP / WDRA are opt-in only on gfx946 (HCU).
_WASP_WDRA_ARCHS = frozenset({"gfx946"})

# Default WDRA VGPR budgets from triton-staging GEMM regression.
_DEFAULT_WDRA_LOAD_REGS = 88
_DEFAULT_WDRA_MMA_MAIN = 144
_DEFAULT_WDRA_MMA_TAIL = 140
_DEFAULT_WDRA_MMA_TAIL_8 = 160  # TwoPTwoC: load+load+main+tail % 32 == 0

# Block candidates: wasp-friendly 128 tiles; WDRA often prefers smaller LDS.
_BLOCK_CANDIDATES_NOWASP = (
    (128, 128, 64),
    (128, 128, 128),
    (64, 64, 64),
)
_BLOCK_CANDIDATES_WASP = (
    (128, 128, 64),
    (128, 128, 128),
    (64, 64, 64),
)
_BLOCK_CANDIDATES_WDRA = (
    (64, 64, 64),
    (128, 128, 64),
    (64, 64, 128),
)

_KERNEL_CACHE: dict[tuple, Any] = {}


def default_launch_config(
    *,
    wasp: bool = False,
    wdra: bool = False,
    load_warps: int = 4,
    mma_warps: int = 4,
    empty_arrive_after_mmac: bool = False,
    enable_v_mmac_cluster: int = 0,
) -> dict[str, Any]:
    """Single fixed launch config (no autotune)."""
    if wdra:
        bm = bn = bk = 64
    else:
        bm = bn = 128
        bk = 64
    cfg = _mode_launch_kwargs(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )
    cfg.update(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "empty_arrive_after_mmac": bool(empty_arrive_after_mmac),
            "enable_v_mmac_cluster": int(enable_v_mmac_cluster),
        }
    )
    if not wasp:
        cfg["num_warps"] = 4
        cfg["num_stages"] = 2
    return cfg


def supports_wasp_wdra(arch: Optional[str] = None) -> bool:
    """Return True when the current (or given) arch may enable WASP/WDRA."""
    arch = arch or arch_info.get_arch()
    return arch in _WASP_WDRA_ARCHS


def _mode_launch_kwargs(
    *,
    wasp: bool,
    wdra: bool,
    load_warps: int,
    mma_warps: int,
) -> dict[str, Any]:
    if wdra and not wasp:
        raise ValueError("wdra requires wasp=True")
    if wasp and load_warps not in (4, 8):
        raise ValueError(f"wasp_num_load_warps must be 4 or 8; got {load_warps}")
    if wasp and mma_warps not in (4, 8):
        raise ValueError(f"wasp_num_mma_warps must be 4 or 8; got {mma_warps}")
    if wasp and load_warps == 8 and mma_warps != 8:
        raise ValueError("wasp load_warps=8 requires mma_warps=8 (TwoPTwoC)")

    cfg: dict[str, Any] = {
        "GROUP_SIZE_M": 1,
        "WARP_SPECIALIZE": wasp,
        "wasp_enabled": wasp,
        "wdra_enabled": wdra,
    }
    if wasp:
        cfg["wasp_num_load_warps"] = load_warps
        cfg["wasp_num_mma_warps"] = mma_warps
        if wdra:
            mma_tail = (
                _DEFAULT_WDRA_MMA_TAIL_8 if load_warps >= 8 else _DEFAULT_WDRA_MMA_TAIL
            )
            cfg.update(
                {
                    "wdra_num_load_regs": _DEFAULT_WDRA_LOAD_REGS,
                    "wdra_num_mma_regs_main": _DEFAULT_WDRA_MMA_MAIN,
                    "wdra_num_mma_regs_tail": mma_tail,
                }
            )
    return cfg


def _block_candidates(*, wasp: bool, wdra: bool) -> tuple[tuple[int, int, int], ...]:
    if wdra:
        return _BLOCK_CANDIDATES_WDRA
    if wasp:
        return _BLOCK_CANDIDATES_WASP
    return _BLOCK_CANDIDATES_NOWASP


def build_autotune_configs(
    *,
    wasp: bool = False,
    wdra: bool = False,
    load_warps: int = 4,
    mma_warps: int = 4,
) -> list[triton.Config]:
    """Build ``@triton.autotune`` configs for the selected mode.

    Sweep ``empty_arrive_after_mmac`` × ``enable_v_mmac_cluster`` (MR !208)
    and a small block-size grid. Mode / WASP topology is fixed per call.
    """
    mode = _mode_launch_kwargs(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )
    configs: list[triton.Config] = []
    for bm, bn, bk in _block_candidates(wasp=wasp, wdra=wdra):
        for empty_after in (False, True):
            for cluster in (0, 1, 2):
                kwargs = {
                    **mode,
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "BLOCK_SIZE_K": bk,
                    "empty_arrive_after_mmac": empty_after,
                    "enable_v_mmac_cluster": cluster,
                }
                # Plain num_warps only for non-WASP; compiler ignores it after WS.
                if wasp:
                    configs.append(triton.Config(kwargs))
                else:
                    configs.append(triton.Config(kwargs, num_warps=4, num_stages=2))
    return configs


@triton.jit
def _gemm_fp16_load_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        accumulator += bias
    c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _gemm_fp16_mls_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        mls_offs_k = k * BLOCK_SIZE_K
        a = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_am, mls_offs_k],
        )
        b = tl.matrix_load(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            offsets=[mls_offs_k, mls_offs_bn],
        )
        accumulator = tl.dot(a, b, accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        accumulator += bias
    c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _get_autotuned_kernel(
    *,
    wasp: bool,
    wdra: bool,
    load_warps: int,
    mma_warps: int,
    use_mls: bool,
):
    key = (wasp, wdra, load_warps, mma_warps, use_mls)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    base = _gemm_fp16_mls_kernel if use_mls else _gemm_fp16_load_kernel
    configs = build_autotune_configs(
        wasp=wasp,
        wdra=wdra,
        load_warps=load_warps,
        mma_warps=mma_warps,
    )
    tuned = triton.autotune(configs=configs, key=["M", "N", "K"])(base)
    _KERNEL_CACHE[key] = tuned
    return tuned


def _resolve_wasp_topology(
    *,
    wasp: bool,
    wdra: bool,
    wasp_num_load_warps: Optional[int],
    wasp_num_mma_warps: Optional[int],
) -> tuple[int, int]:
    if not wasp:
        return 4, 4
    load_warps = 4 if wasp_num_load_warps is None else wasp_num_load_warps
    if wasp_num_mma_warps is None:
        mma_warps = 8 if wdra else 4
    else:
        mma_warps = wasp_num_mma_warps
    return load_warps, mma_warps


def _prepare_ab(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Flatten leading dims of ``a`` like vLLM ``matmul_batch_invariant``.

    Supports:
      * 2D × 2D: ``[M,K] @ [K,N]``
      * ND × 2D: ``[...,K] @ [K,N]`` → reshape ``a`` to ``[-1,K]``
    """
    if b.dim() != 2:
        raise ValueError(f"gemm_fp16 expects 2D b [K,N]; got b={tuple(b.shape)}")
    if a.dim() < 2:
        raise ValueError(f"gemm_fp16 expects a with dim>=2; got a={tuple(a.shape)}")
    if a.shape[-1] != b.shape[0]:
        raise ValueError(
            f"incompatible GEMM shapes: a={tuple(a.shape)}, b={tuple(b.shape)}"
        )
    out_shape = a.shape[:-1] + (b.shape[1],)
    a_2d = a.reshape(-1, a.shape[-1]) if a.dim() != 2 else a
    return a_2d, b, out_shape


def gemm_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    wasp: bool = False,
    wdra: bool = False,
    use_mls: bool = False,
    wasp_num_load_warps: Optional[int] = None,
    wasp_num_mma_warps: Optional[int] = None,
    autotune: bool = True,
    config: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """FP16 GEMM aligned with vLLM Triton matmul call shape.

    Computes ``out = a @ b [+ bias]`` with FP32 accumulate. Layout matches
    vLLM ``matmul_persistent(a, b, bias)`` / ``VLLM_USE_NN`` linear path:
    ``a[..., K]``, ``b[K, N]``, optional 1D ``bias[N]``.
    """
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError(
            f"gemm_fp16 requires float16 inputs; got a={a.dtype}, b={b.dtype}"
        )
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if not a.is_cuda:
        raise ValueError("gemm_fp16 requires CUDA/HIP tensors")

    a_2d, b_2d, out_shape = _prepare_ab(a, b)

    if bias is not None:
        if bias.dim() != 1:
            raise ValueError(
                f"bias must be 1D [N]; got bias.shape={tuple(bias.shape)}"
            )
        if bias.shape[0] != b_2d.shape[1]:
            raise ValueError(
                f"bias length must equal N={b_2d.shape[1]}; got {bias.shape[0]}"
            )
        if bias.device != a.device:
            raise ValueError("bias must be on the same device as a/b")
        if bias.dtype != torch.float16:
            raise TypeError(f"bias must be float16; got {bias.dtype}")

    if wdra and not wasp:
        raise ValueError("wdra requires wasp=True")
    if wasp or wdra:
        arch = arch_info.get_arch()
        if not supports_wasp_wdra(arch):
            raise ValueError(
                f"wasp/wdra are only supported on {_WASP_WDRA_ARCHS}; got arch={arch!r}"
            )
    if use_mls and not arch_info.is_mls_avail():
        raise ValueError(
            f"use_mls=True requires an MLS-capable arch; got {arch_info.get_arch()!r}"
        )
    if config is not None and not isinstance(config, dict):
        raise TypeError(f"config must be a dict or None; got {type(config)!r}")

    # Match staging regression defaults that help buffer / epilogue paths.
    os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")

    load_warps, mma_warps = _resolve_wasp_topology(
        wasp=wasp,
        wdra=wdra,
        wasp_num_load_warps=wasp_num_load_warps,
        wasp_num_mma_warps=wasp_num_mma_warps,
    )
    _mode_launch_kwargs(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )

    M, K = a_2d.shape
    _, N = b_2d.shape
    has_bias = bias is not None
    # Triton needs a real pointer even when HAS_BIAS=False; use ``b`` as dummy.
    bias_arg = bias if has_bias else b_2d

    if out is None:
        out_2d = torch.empty((M, N), device=a.device, dtype=torch.float16)
        out = out_2d.view(out_shape) if a.dim() != 2 else out_2d
    else:
        if out.dtype != torch.float16:
            raise TypeError(f"out must be float16; got {out.dtype}")
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(
                f"out shape must be {tuple(out_shape)}; got {tuple(out.shape)}"
            )
        if out.device != a.device:
            raise ValueError("out must be on the same device as a/b")
        out_2d = out.reshape(M, N)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    args = (
        a_2d,
        b_2d,
        out_2d,
        bias_arg,
        M,
        N,
        K,
        a_2d.stride(0),
        a_2d.stride(1),
        b_2d.stride(0),
        b_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
    )

    if config is not None or not autotune:
        launch = (
            dict(config)
            if config is not None
            else default_launch_config(
                wasp=wasp,
                wdra=wdra,
                load_warps=load_warps,
                mma_warps=mma_warps,
            )
        )
        launch["HAS_BIAS"] = has_bias
        base = _gemm_fp16_mls_kernel if use_mls else _gemm_fp16_load_kernel
        base[grid](*args, **launch)
    else:
        kernel = _get_autotuned_kernel(
            wasp=wasp,
            wdra=wdra,
            load_warps=load_warps,
            mma_warps=mma_warps,
            use_mls=use_mls,
        )
        kernel[grid](*args, HAS_BIAS=has_bias)
    return out


__all__ = [
    "build_autotune_configs",
    "default_launch_config",
    "gemm_fp16",
    "supports_wasp_wdra",
]
