# SPDX-License-Identifier: MIT
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/python/tutorials/03-matrix-multiplication.py
"""FP16 GEMM Triton kernel with optional WASP/WDRA (gfx946) and barrier knobs.

The public host entry is ``boltops.gemm.gemm_a16w16``.

Kernel launch options follow ``triton-release-3.6.x`` HCU regression /
barrier autotune
(``test_wasp_mls_regression.py`` / ``bench_gemm_wasp_barrier_autotune.py``).

Autotune HIPOptions (consolidated; sched barriers are not independent):
  * ``empty_arrive_after_mmac`` — bool
  * ``enable_v_mmac_cluster`` — 0 | 1 | 2
      0: off
      1: cluster (+ SchedBarrier before mmac and between A/B loads)
      2: cluster + cross-region A/B load analysis (implies 1)
  * ``wasp_partition_warps`` / ``wasp_wdra`` / ``wasp_partition_regs`` —
    Load-first topology, WDRA selection and optional per-partition VGPR quotas.
    Consumer ping-pong is managed by Triton for two-consumer topologies.
  * ``hcu_use_gfx946_sched_model_v2`` — bool. This GEMM sets it to False to
    retain the legacy gfx946 scheduling model with newer HCU Clang builds.
  * ``num_stages`` — WASP/WDRA LDS buffer depth ``2`` (default) or ``4``
    (``wdra88`` / TwoPTwoC does not support ``4``)
  * ``mmac_layout_force`` — ``-1`` (default / auto) or ``0..4``
      0: mfma; 1: mmac legacy; 2: interleave; 3: transpose;
      4: interleave+transpose
plus a small BLOCK_SIZE_{M,N,K} sweep for the selected WASP/WDRA mode.

Named modes (same as the release bench): nowasp | wasp44 | wdra48 | wdra88.

MLS path (``use_mls`` / ``--mls``): A/B use ``tl.matrix_load``. C uses
``tl.matrix_store`` when ``use_matrix_store`` / ``--matrix-store`` is set
(needs ``triton-mls-store``; prefer ``mmac_layout_force=3``); otherwise
``tl.store``.
"""

from __future__ import annotations

from typing import Any, Optional

import triton
import triton.language as tl

from boltops.utility import arch_info

# WASP / WDRA are opt-in only on gfx946 (HCU).
_WASP_WDRA_ARCHS = frozenset({"gfx946"})

# Default WDRA VGPR budgets (load / mma_main / mma_tail).
# OnePTwoC must keep sum % 12 == 0. TN MLS BK=64 + consumer pingpong uses
# 16/244/244 (sum 504); prior non-pingpong TN was 88/208/208. TT large tiles
# often use 112/196/196.
_DEFAULT_WDRA_LOAD_REGS = 16
_DEFAULT_WDRA_MMA_MAIN = 244
_DEFAULT_WDRA_MMA_TAIL = 244
# _DEFAULT_WDRA_LOAD_REGS = 88
# _DEFAULT_WDRA_MMA_MAIN = 208
# _DEFAULT_WDRA_MMA_TAIL = 208
_DEFAULT_WDRA_MMA_TAIL_8 = 160  # TwoPTwoC quotas are normalized by Triton to sum % 16 == 0

# Named modes mirrored from scripts/hcu/bench_gemm_wasp_barrier_autotune.py.
MODES = ("nowasp", "wasp44", "wdra48", "wdra88")

# Block candidates: wasp-friendly 128 tiles; WDRA often prefers smaller LDS.
# Defaults match regression ``gemm_config`` (square tile: 128 or 64).
_BLOCK_CANDIDATES_NOWASP = (
    (128, 128, 128),
    (128, 128, 64),
    (64, 64, 64),
)
_BLOCK_CANDIDATES_WASP = (
    (128, 128, 128),
    (128, 128, 64),
    (64, 64, 64),
)
_BLOCK_CANDIDATES_WDRA = (
    (64, 64, 64),
    (128, 128, 64),
    (64, 64, 128),
)


def _validate_mmac_cluster(enable_v_mmac_cluster: int) -> int:
    cluster = int(enable_v_mmac_cluster)
    if cluster not in (0, 1, 2):
        raise ValueError(
            f"enable_v_mmac_cluster must be 0, 1, or 2; got {cluster}"
        )
    return cluster


def _validate_mmac_layout_force(mmac_layout_force: int) -> int:
    """HIPOptions ``mmac_layout_force``: ``-1`` auto, else ``0..4`` (see module doc)."""
    layout = int(mmac_layout_force)
    if layout < -1 or layout > 4:
        raise ValueError(
            f"mmac_layout_force must be -1 or 0..4; got {layout}"
        )
    return layout


def _validate_num_stages(num_stages: int, *, wasp: bool, wdra: bool, load_warps: int) -> int:
    """WASP/WDRA LDS buffer depth is 2 or 4; TwoPTwoC (wdra88) forbids 4."""
    stages = int(num_stages)
    if wasp:
        if stages not in (2, 4):
            raise ValueError(f"wasp requires num_stages in (2, 4); got {stages}")
        if load_warps >= 8 and stages == 4:
            raise ValueError(
                "wdra TwoPTwoC (load_warps=8) does not support num_stages=4 "
                "(abarrier ID limit)"
            )
    elif stages < 1:
        raise ValueError(f"num_stages must be >= 1; got {stages}")
    return stages


def default_launch_config(
    *,
    wasp: bool = False,
    wdra: bool = False,
    load_warps: int = 4,
    mma_warps: int = 4,
    empty_arrive_after_mmac: bool = False,
    enable_v_mmac_cluster: int = 0,
    num_stages: int = 2,
    mmac_layout_force: int = -1,
) -> dict[str, Any]:
    """Single fixed launch config (no autotune).

    Tile defaults match ``test_wasp_mls_regression.gemm_config``: square
    ``128`` without WDRA, square ``64`` with WDRA (smaller LDS).

    Under WASP/WDRA, ``num_stages`` is the LDS buffer depth (2 or 4).
    """
    block = 64 if wdra else 128
    stages = _validate_num_stages(
        num_stages, wasp=wasp, wdra=wdra, load_warps=load_warps
    )
    cfg = _mode_launch_kwargs(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )
    cfg.update(
        {
            "BLOCK_SIZE_M": block,
            "BLOCK_SIZE_N": block,
            "BLOCK_SIZE_K": block,
            "empty_arrive_after_mmac": bool(empty_arrive_after_mmac),
            "enable_v_mmac_cluster": _validate_mmac_cluster(enable_v_mmac_cluster),
            "num_stages": stages,
            "mmac_layout_force": _validate_mmac_layout_force(mmac_layout_force),
        }
    )
    if not wasp:
        cfg["num_warps"] = 4
    return cfg


def mode_launch_kwargs(
    mode: str,
    *,
    empty_arrive_after_mmac: bool = False,
    enable_v_mmac_cluster: int = 0,
    num_stages: int = 2,
    mmac_layout_force: int = -1,
) -> dict[str, Any]:
    """Fixed launch kwargs for a named release-bench mode.

    Modes: ``nowasp`` | ``wasp44`` | ``wdra48`` | ``wdra88``.
    """
    kw = dict(
        empty_arrive_after_mmac=empty_arrive_after_mmac,
        enable_v_mmac_cluster=enable_v_mmac_cluster,
        num_stages=num_stages,
        mmac_layout_force=mmac_layout_force,
    )
    if mode == "nowasp":
        return default_launch_config(wasp=False, wdra=False, **kw)
    if mode == "wasp44":
        return default_launch_config(
            wasp=True, wdra=False, load_warps=4, mma_warps=4, **kw
        )
    if mode == "wdra48":
        return default_launch_config(
            wasp=True, wdra=True, load_warps=4, mma_warps=8, **kw
        )
    if mode == "wdra88":
        return default_launch_config(
            wasp=True, wdra=True, load_warps=8, mma_warps=8, **kw
        )
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


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
        "wasp_wdra": wdra,
        "amdgpu_enable_max_ilp_scheduling_strategy": True,
        "hcu_use_gfx946_sched_model_v2": False,
    }
    if wasp:
        producers = 2 if load_warps == 8 else 1
        consumers = 2 if mma_warps == 8 else 1
        cfg["wasp_partition_warps"] = (4,) * (producers + consumers)
        if wdra:
            consumers_regs = (_DEFAULT_WDRA_MMA_MAIN,)
            if consumers == 2:
                consumers_regs += (
                    _DEFAULT_WDRA_MMA_TAIL_8 if producers == 2 else _DEFAULT_WDRA_MMA_TAIL,
                )
            cfg["wasp_partition_regs"] = (
                (_DEFAULT_WDRA_LOAD_REGS,) * producers + consumers_regs
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
    mmac_layout_force: tuple[int, ...] = (-1,),
) -> list[triton.Config]:
    """Build ``@triton.autotune`` configs for the selected mode.

    Mirrors ``bench_gemm_wasp_barrier_autotune.py`` HIPOptions sweep inside one
    topology: ``empty_arrive_after_mmac`` × ``enable_v_mmac_cluster`` (0/1/2)
    × ``mmac_layout_force``, plus a small block-size grid. Sched barriers are
    not swept separately — ``enable_v_mmac_cluster >= 1`` implies before-mmac
    and A/B-load fences. Default ``mmac_layout_force=(-1,)`` keeps auto layout
    only; pass ``(-1, 0, 1, 2, 3, 4)`` (or a subset) to sweep. Mode / WASP
    topology stays fixed per call (use ``mode_launch_kwargs`` /
    ``wasp``+``wdra``+warp counts to select nowasp/wasp44/wdra48/wdra88).
    """
    mode = _mode_launch_kwargs(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )
    layouts = tuple(
        _validate_mmac_layout_force(v) for v in mmac_layout_force
    ) or (-1,)
    configs: list[triton.Config] = []
    for bm, bn, bk in _block_candidates(wasp=wasp, wdra=wdra):
        for empty_after in (False, True):
            for cluster in (0, 1, 2):
                for layout in layouts:
                    kwargs = {
                        **mode,
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_N": bn,
                        "BLOCK_SIZE_K": bk,
                        "empty_arrive_after_mmac": empty_after,
                        "enable_v_mmac_cluster": cluster,
                        "mmac_layout_force": layout,
                    }
                    # Plain num_warps only for non-WASP; compiler ignores it after WS.
                    if wasp:
                        configs.append(triton.Config(kwargs))
                    else:
                        configs.append(
                            triton.Config(kwargs, num_warps=4, num_stages=2)
                        )
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
        remaining_k = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < remaining_k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < remaining_k, other=0.0)
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
    USE_MATRIX_STORE: tl.constexpr,
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

    if HAS_BIAS:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        accumulator += bias
    c = accumulator.to(tl.float16)

    if USE_MATRIX_STORE:
        # VGPR→GMEM matrix_store (prefer mmac_layout_force=3).
        tl.matrix_store(
            c_ptr,
            c,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            offsets=[mls_offs_am, mls_offs_bn],
        )
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


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
