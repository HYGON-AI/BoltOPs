# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501


import torch

import triton
import triton.language as tl

from .chunk_intra import chunk_kda_fwd_intra
from .config_utils import dtype_name, get_kda_tuned_config
from .chunk_state import (
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_fwd_state_output_fused,
)
from .prefill_optimized import can_run_prefill_optimized, prefill_optimized_fwd
from boltops.fla.common.triton.cumsum import chunk_local_cumsum
from boltops.fla.common.triton.l2norm import l2norm_fwd
from boltops.fla.common.triton.math import exp2, log
from boltops.fla.common.triton.utils import FLA_CHUNK_SIZE, prepare_chunk_indices

RCP_LN2 = 1.4426950408889634
cdiv = triton.cdiv
next_power_of_2 = triton.next_power_of_2

# Offline candidates retained on BW1000_H.  The new recompute/output champions
# were selected at B=1, T=8192, H=24, K=V=128; wrappers select one config
# directly and never run online autotuning.
RECOMPUTE_W_U_FWD_CONFIGS = (
    {"BK": 64, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 64, "BV": 128, "num_warps": 4, "num_stages": 1},
    {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 2},
)
CHUNK_GLA_FWD_O_CONFIGS = (
    {"BK": 128, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 128, "BV": 128, "num_warps": 8, "num_stages": 2},
    {"BK": 64, "BV": 64, "num_warps": 8, "num_stages": 2},
)
# Retuned on BW1100 (72 CU) against a torch reference over
# BS x num_warps x num_stages.  The kernel is latency-bound rather than
# bandwidth-bound (33% of HBM roofline at its best), and a single wavefront
# per program wins at every measured shape: widening to 2-8 warps splits the
# 64-row cumsum across wavefronts and costs 10-25%.  Larger BS then amortizes
# the per-program prologue, so the tile grows with the launch.
KDA_GATE_CHUNK_CUMSUM_VECTOR_CONFIGS = (
    {"BS": 32, "num_warps": 1, "num_stages": 1},
    {"BS": 64, "num_warps": 1, "num_stages": 2},
    {"BS": 128, "num_warps": 1, "num_stages": 1},
)
KDA_GATE_FWD_CONFIGS = (
    {"BT": 32, "num_warps": 16, "num_stages": 2},
    {"BT": 64, "num_warps": 4, "num_stages": 2},
    {"BT": 128, "num_warps": 4, "num_stages": 2},
)


def get_recompute_w_u_fwd_config(
    _T: int,
    _H: int,
    _K: int,
    V: int,
    _is_varlen: bool,
    *,
    b: int,
    dtype: torch.dtype,
) -> dict:
    return get_kda_tuned_config(
        "recompute_w_u_fwd",
        f"B={b},dtype={dtype_name(dtype)},T={_T},H={_H},K={_K},V={V},varlen={_is_varlen}",
        fallback=static_recompute_w_u_fwd_config(_T, _H, _K, V, _is_varlen),
    )


def static_recompute_w_u_fwd_config(
    _T: int, _H: int, _K: int, V: int, _is_varlen: bool
) -> dict:
    if V < 128:
        return RECOMPUTE_W_U_FWD_CONFIGS[2]
    return RECOMPUTE_W_U_FWD_CONFIGS[0]


def get_chunk_gla_fwd_o_config(
    T: int,
    H: int,
    K: int,
    V: int,
    is_varlen: bool,
    *,
    b: int,
    dtype: torch.dtype,
) -> dict:
    return get_kda_tuned_config(
        "chunk_gla_fwd_o_gk",
        f"B={b},dtype={dtype_name(dtype)},T={T},H={H},K={K},V={V},varlen={is_varlen}",
        fallback=static_chunk_gla_fwd_o_config(T, H, K, V, is_varlen),
    )


def static_chunk_gla_fwd_o_config(
    T: int, H: int, K: int, V: int, is_varlen: bool
) -> dict:
    del T, H, is_varlen
    if K >= 128 and V >= 128:
        # On BW200, the output kernel is large enough at the model shape to
        # benefit from twice the wavefront-level parallelism.  The BK=128 /
        # BV=128 tile is unchanged; only the launch width moves from 4 to 8
        # warps.  The smaller-tile candidate is slower because it increases
        # the number of programs and repeats the h-tile loads.
        return CHUNK_GLA_FWD_O_CONFIGS[1]
    return CHUNK_GLA_FWD_O_CONFIGS[2]


def get_kda_gate_chunk_cumsum_vector_config(
    T: int,
    H: int,
    S: int,
    is_varlen: bool,
    *,
    b: int,
    dtype: torch.dtype,
) -> dict:
    return get_kda_tuned_config(
        "kda_gate_chunk_cumsum",
        f"B={b},dtype={dtype_name(dtype)},T={T},H={H},S={S},varlen={is_varlen}",
        fallback=static_kda_gate_chunk_cumsum_vector_config(T, H, S, is_varlen),
    )


def static_kda_gate_chunk_cumsum_vector_config(
    T: int, H: int, S: int, is_varlen: bool
) -> dict:
    del is_varlen  # the index math differs, the tile economics do not
    # Measured best-of-80 (BS x warps x stages) per shape, K=V=128:
    #   T*H >= 512K  -> BS=128 (0.85 ms vs 0.99 ms at T=8192 H=64)
    #   T*H >=  32K  -> BS= 64 (0.26 ms vs 0.30 ms at T=16384 H=16)
    #   else         -> BS= 32 (0.088 ms vs 0.097 ms at T=1024 H=16)
    work = T * H
    if work >= 512 * 1024:
        config = KDA_GATE_CHUNK_CUMSUM_VECTOR_CONFIGS[2]
    elif work >= 32 * 1024:
        config = KDA_GATE_CHUNK_CUMSUM_VECTOR_CONFIGS[1]
    else:
        config = KDA_GATE_CHUNK_CUMSUM_VECTOR_CONFIGS[0]
    if config["BS"] > S:
        # A tile wider than the gate dimension is all boundary-masked waste.
        return {**config, "BS": max(16, next_power_of_2(S))}
    return config


def get_kda_gate_fwd_config(
    T: int,
    H: int,
    _D: int,
    *,
    b: int,
    dtype: torch.dtype,
) -> dict:
    return get_kda_tuned_config(
        "kda_gate_fwd",
        f"B={b},dtype={dtype_name(dtype)},T={T},H={H}",
        fallback=static_kda_gate_fwd_config(T, H, _D),
    )


def static_kda_gate_fwd_config(T: int, H: int, _D: int) -> dict:
    if H <= 3 or T <= 512:
        return KDA_GATE_FWD_CONFIGS[0]
    # ``T`` is flattened B*T at this call site.  Long prefill launches on
    # BW1000_H favors the smaller BT32 tile; the BT64/BT128 choices remain for
    # medium decode-like launches.
    if T >= 8192:
        return KDA_GATE_FWD_CONFIGS[0]
    return KDA_GATE_FWD_CONFIGS[1]


def _use_kda_state_output_fused(
    q: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    kg: torch.Tensor,
    Aqk: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool = False,
) -> bool:
    """Select the standalone same-head KDA state/output fusion fast path."""
    del output_final_state  # both paths now produce a bit-identical final state
    if chunk_size != 64 or q.ndim != 4 or u.ndim != 4:
        return False
    if q.shape[2] != u.shape[2] or kg.shape != q.shape:
        return False
    if w.shape != q.shape or g.shape != q.shape or Aqk.shape[:3] != q.shape[:3]:
        return False
    if q.dtype != torch.bfloat16 or w.dtype != q.dtype or u.dtype != q.dtype:
        return False
    if kg.dtype != q.dtype or Aqk.dtype != q.dtype:
        return False
    if g.dtype not in (torch.bfloat16, torch.float32):
        return False
    kdim, vdim = q.shape[-1], u.shape[-1]
    # KDA's ordinary prefill shapes are K/V=128 and 256.  K/V=64 remains on
    # the split path because its smaller state tile does not amortize fusion.
    if kdim < 128 or vdim < 128:
        return False
    if cu_seqlens is not None and cu_seqlens.ndim != 1:
        return False
    n = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    if kdim < 256 or vdim < 256:
        # This used to require N*H >= 40.  That crossover belonged to the old
        # fixed BV=64 tile, whose grid was 2*N*H workgroups and could not fill
        # the device at small N*H.  The tile is now chosen from N*H (see
        # get_fused_state_output_config), which keeps the grid near one CU
        # wave at every head count, and fusion wins throughout.  Measured on
        # BW1100 (72 CU) at T=16384 -- the worst case for the fused path,
        # since it has the most serial chunks and no chunk-level parallelism
        # to lose -- split/fused is 1.25x at N*H=2, 1.37x at 8, 1.48x at 16,
        # 1.66x at 32 and 1.65x at 48.
        return True
    return q.shape[1] * q.shape[2] >= 96 * 8192


def _use_chunk_gla_fwd_o_local_tri32(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Whether the 32-row triangular local-causal path is shape-safe.

    The prototype relies on a fixed 64-token chunk split into two 32-row
    programs.  It is deliberately restricted to the matrix-heavy K/V class;
    all other shapes keep using the upstream-compatible dense-mask kernel.
    """
    if chunk_size != 64 or q.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        return False
    if h.ndim != 5 or A.ndim != 4:
        return False
    if q.shape[:3] != v.shape[:3] or q.shape[:3] != g.shape[:3]:
        return False
    if h.shape[0] != q.shape[0] or h.shape[2] != q.shape[2]:
        return False
    if h.shape[-2:] != (v.shape[-1], q.shape[-1]):
        return False
    if A.shape[:3] != q.shape[:3] or A.shape[-1] != chunk_size:
        return False
    if q.shape[-1] < 128 or v.shape[-1] < 128:
        return False
    return cu_seqlens is None or cu_seqlens.ndim == 1


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_beta_factored_head_first(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Head-first sibling that factors beta into the shared A tile."""
    i_bh, i_tc = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_tc * 2).to(tl.int32),
            tl.load(chunk_indices + i_tc * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        i_t = i_tc
        bos, eos = i_b * T, i_b * T + T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_Ab = (
        tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        * b_beta[None, :]
    ).to(tl.bfloat16)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Ab, b_v, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kgate = (b_k * exp2(b_gk)).to(b_k.dtype)
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_w = tl.dot(b_Ab, b_kgate, input_precision=DOT_PRECISION)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            tl.store(
                p_qg,
                (b_q * exp2(b_gk)).to(p_qg.dtype.element_ty),
                boundary_check=(0, 1),
            )

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            tl.store(
                p_kg,
                b_kg.to(p_kg.dtype.element_ty),
                boundary_check=(0, 1),
            )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    u_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
    """Rebuild the WY factors ``w``/``u`` and the decayed key ``kg``."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    if u_out is not None:
        if u_out.shape != v.shape or u_out.dtype != v.dtype or not u_out.is_contiguous():
            raise ValueError("u_out must be contiguous and match v")
        u = u_out
    else:
        u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None
    config = get_recompute_w_u_fwd_config(
        T, H, K, V, cu_seqlens is not None,
        b=B, dtype=k.dtype,
    )
    use_beta_factored_head_first = (
        k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and A.dtype == torch.bfloat16
        and gk is not None
        and BT == 64
        and K <= 128
        and V <= 128
        and (cu_seqlens is None or B == 1)
    )
    kernel = (
        recompute_w_u_fwd_kernel_beta_factored_head_first
        if use_beta_factored_head_first
        else recompute_w_u_fwd_kernel
    )
    grid = (B * H, NT) if use_beta_factored_head_first else (NT, B * H)
    kernel[grid](
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        DOT_PRECISION="ieee",
        **config,
    )
    return w, u, None, kg


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_local_first(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Chunk output with the causal local term scheduled before state."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos, eos = i_b * T, i_b * T + T

    rows = tl.arange(0, BT)
    causal = rows[:, None] >= rows[None, :]
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(causal, b_A, 0.0).to(b_v.dtype)
    b_o = tl.dot(b_A, b_v, allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))

    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_local_tri32(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Local-causal output with a 64 -> 32x32 triangular decomposition.

    The dense implementation performs one 64x64 dot after masking its upper
    triangle.  This kernel computes the upper row half from the lower-left
    diagonal block and the lower row half from the full lower-left plus the
    lower-right diagonal block, so the zero upper-right block never reaches a
    matrix multiply.  State contribution and addressing remain identical to
    ``chunk_gla_fwd_kernel_o_local_first``.
    """
    i_v, i_ts, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_row = i_ts % 2
    i_tg = i_ts // 2
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_tg * 2).to(tl.int32),
            tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_t = i_tg % NT
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    row_base = i_row * 32
    rows = tl.arange(0, 32)
    b_o = tl.zeros([32, BV], dtype=tl.float32)

    # Only the blocks at or below the diagonal are materialized as dots.
    for i_col in range(2):
        col_base = i_col * 32
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT + col_base, i_v * BV),
            (32, BV),
            (1, 0),
        )
        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + row_base, col_base),
            (32, 32),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        causal = (row_base + rows)[:, None] >= (col_base + rows)[None, :]
        b_A = tl.where(causal, b_A, 0.0).to(b_v.dtype)
        b_o += tl.dot(b_A, b_v, allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + row_base, i_k * BK),
            (32, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + row_base, i_k * BK),
            (32, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))

    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT + row_base, i_v * BV),
        (32, BV),
        (1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    config = get_chunk_gla_fwd_o_config(
        T, H, K, V, cu_seqlens is not None,
        b=B, dtype=q.dtype,
    )
    use_tri32 = _use_chunk_gla_fwd_o_local_tri32(
        q, v, g, h, A, BT, cu_seqlens
    )
    grid = (cdiv(V, config["BV"]), NT * (2 if use_tri32 else 1), B * H)

    if use_tri32:
        kernel = chunk_gla_fwd_kernel_o_local_tri32
    else:
        kernel = (
            chunk_gla_fwd_kernel_o_local_first
            if K >= 128 and V >= 128 and BT == 64
            else chunk_gla_fwd_kernel_o
        )
    kernel[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        **config,
    )
    return o


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    raw_beta,
    A_log,
    g_bias,
    o,
    beta_out,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound,
    beta,
    threshold,
    T,
    stride_beta_batch,
    stride_beta_token,
    stride_beta_head,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    if i_s == 0:
        o_beta_t = tl.arange(0, BT)
        m_beta = i_t * BT + o_beta_t < T
        if IS_VARLEN:
            p_beta = (
                raw_beta
                + (bos + i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        else:
            p_beta = (
                raw_beta
                + i_b * stride_beta_batch
                + (i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        b_beta = tl.load(p_beta, mask=m_beta, other=0.0).to(tl.float32)
        p_beta_out = beta_out + (bos + i_t * BT + o_beta_t) * H + i_h
        tl.store(p_beta_out, tl.sigmoid(b_beta), mask=m_beta)
        return

    i_s -= 1

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_bias = tl.make_block_ptr(
            g_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_bias, boundary_check=(0,)).to(tl.float32)
        b_s += b_bias[None, :]

    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_s)
    else:
        b_g_scaled = b_s * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_s,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = -b_a * b_softplus

    # Boundary loads return zero, but bias and gate activation can make padded
    # rows nonzero. Padding trails valid rows, so it only affects masked stores.
    b_o = tl.cumsum(b_gate, axis=0) * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    B, T, H, D = raw_g.shape
    if raw_beta.shape != (B, T, H):
        raise ValueError(f"Expected raw_beta shape {(B, T, H)}, got {raw_beta.shape}")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    beta_out = torch.empty(raw_beta.shape, device=raw_beta.device, dtype=torch.float32)
    config = get_kda_gate_chunk_cumsum_vector_config(
        T, H, D, cu_seqlens is not None,
        b=B, dtype=raw_g.dtype,
    )
    # For each (chunk, head), program 0 computes beta without extending a gate
    # tile's critical path. The remaining programs cover the gate dimension.
    grid = (cdiv(D, config["BS"]) + 1, NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        o=y,
        beta_out=beta_out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g). Keep this in sync with the
        # `use_exp2=True` path in `_chunk_kda_fwd_with_cumulative_g`.
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound or 0.0,
        beta=beta,
        threshold=threshold,
        T=T,
        stride_beta_batch=raw_beta.stride(0),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        H=H,
        S=D,
        BT=chunk_size,
        USE_LOWER_BOUND=lower_bound is not None,
        **config,
    )
    return y, beta_out


def _chunk_kda_fwd_with_cumulative_g(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
    normalize_qk: bool = False,
    null_state_index: int = -1,
):
    Aqk, A = chunk_kda_fwd_intra(
        q=q, k=k, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, safe_gate=safe_gate,
        normalize_qk=normalize_qk,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, gk=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        u_out=v,
    )
    if kg is not None and _use_kda_state_output_fused(
        q, w, u, g, kg, Aqk, chunk_size, cu_seqlens,
        output_final_state=output_final_state,
    ):
        return chunk_kda_fwd_state_output_fused(
            q=q,
            w=w,
            u=u,
            g=g,
            kg=kg,
            Aqk=Aqk,
            o=v,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_final_state=output_final_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            null_state_index=null_state_index,
        )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, use_exp2=True,
        null_state_index=null_state_index,
    )
    o = chunk_gla_fwd_o_gk(
        q=q, v=v_new, g=g, A=Aqk, h=h, o=v, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return o, final_state


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    null_state_index: int = -1,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None else None
    )
    # RCP_LN2 (natural-log -> log2) is folded into the cumsum store rather
    # than applied as a torch multiply: the tensor is FP32 [B,T,H,K], so the
    # separate pass cost a 1074 MB round trip (0.61 ms at T=16384, H=64).
    # Keep in sync with `cumsum_scale` in the fused-gate cumsum kernel.
    g = chunk_local_cumsum(
        g, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, cumsum_scale=RCP_LN2,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        initial_state_indices=initial_state_indices,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, null_state_index=null_state_index,
    )


def chunk_kda_with_fused_gate_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    normalize_qk: bool = False,
    null_state_index: int = -1,
):
    chunk_size = FLA_CHUNK_SIZE
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None else None
    )
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g, raw_beta=raw_beta, A_log=A_log, g_bias=g_bias,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, lower_bound=lower_bound,
    )
    return _chunk_kda_fwd_with_cumulative_g(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        initial_state_indices=initial_state_indices,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, safe_gate=lower_bound is not None,
        normalize_qk=normalize_qk, null_state_index=null_state_index,
    )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    null_state_index: int = -1,
    **kwargs,
):
    del kwargs
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
    if initial_state is not None and initial_state.stride()[1:] != (
        initial_state.shape[-2] * initial_state.shape[-1],
        initial_state.shape[-1],
        1,
    ):
        initial_state = initial_state.contiguous()
    return chunk_kda_fwd(
        q=q, k=k, v=v.contiguous(), g=g.contiguous(), beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        null_state_index=null_state_index,
    )


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    null_state_index: int = -1,
    **kwargs,
):
    del kwargs
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The fixed-schedule prefill is single-sequence: route pooled or
    # in-place state through the standard pipeline instead.
    if (
        initial_state_indices is None
        and not inplace_final_state
        and q.shape[1] <= 1024
        and q.shape[2] <= 64
        and can_run_prefill_optimized(
            q,
            k,
            v,
            raw_g,
            raw_beta,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )
    ):
        return prefill_optimized_fwd(
            q=q,
            k=k,
            v=v,
            g=raw_g,
            beta=raw_beta,
            scale=scale,
            initial_state=(
                initial_state.contiguous() if initial_state is not None else None
            ),
            output_final_state=output_final_state,
            A_log=A_log.contiguous(),
            g_bias=g_bias.contiguous() if g_bias is not None else None,
            lower_bound=lower_bound,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
            apply_beta_sigmoid=True,
        )
    normalize_qk = False
    if use_qk_l2norm_in_kernel:
        # The K=128, BT=64 intra polynomial kernel can normalize q/k while
        # computing Aqk/Akk. Keep contiguous storage for its write-back; other
        # shapes retain the standalone l2norm path below.
        if q.shape[-1] == 128 and k.shape[-1] == 128 and q.shape[1] % 64 == 0:
            # The intra stage writes the normalized rows back into these
            # buffers (in-kernel when the sub-chunk kernel runs, in place via
            # the l2norm helper otherwise) for recompute/output consumers,
            # so do not alias the caller's q/k tensors.
            # If q is already contiguous, clone directly (1 copy).  If not,
            # .contiguous() creates a fresh private buffer — no second clone.
            q = q.clone() if q.is_contiguous() else q.contiguous()
            k = k.clone() if k.is_contiguous() else k.contiguous()
            normalize_qk = True
        else:
            q = l2norm_fwd(q.contiguous())
            k = l2norm_fwd(k.contiguous())
    if initial_state is not None and initial_state.stride()[1:] != (
        initial_state.shape[-2] * initial_state.shape[-1],
        initial_state.shape[-1],
        1,
    ):
        initial_state = initial_state.contiguous()
    return chunk_kda_with_fused_gate_fwd(
        q=q, k=k, v=v.contiguous(), raw_g=raw_g.contiguous(),
        raw_beta=raw_beta, A_log=A_log, g_bias=g_bias, scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        normalize_qk=normalize_qk,
        null_state_index=null_state_index,
    )


@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    FAST_DIV: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    if USE_LOWER_BOUND:
        b_y = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        if FAST_DIV:
            inv_beta = tl.fdiv_fast(1.0, beta)
        else:
            inv_beta = 1.0 / beta
        sp = tl.where(use_linear, b_g, inv_beta * log(1.0 + tl.exp(g_scaled)))
        b_y = -b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)
    config = get_kda_gate_fwd_config(
        T, H, head_k_dim, b=g.shape[0], dtype=g.dtype
    )
    grid = (cdiv(T, config["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        lower_bound or 0.0,
        beta,
        threshold,
        T,
        H,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
        FAST_DIV=hasattr(tl, "fdiv_fast"),
        **config,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y
