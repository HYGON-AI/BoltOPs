# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Forward-only adaptation of flash-linear-attention 0.5.0.
# ruff: noqa: E501

import functools

import torch

import triton
import triton.language as tl

from .chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from ...common.triton.math import exp2
from ...common.triton.utils import (
    current_platform,
    gather,
    is_gather_supported,
    prepare_chunk_indices,
)


# BW1000_H candidates retained for the KDA Prefill model shape.  Optimized
# siblings also passed fixed/varlen multi-sequence tail smoke tests.
# Build the inverted diagonal blocks inside the inter+solve kernel instead of
# in a separate polynomial pass.  Both kernels are bandwidth bound at the
# measured HBM peak, and they read the same q/k/g rows, so the second pass is
# almost pure duplicated traffic.  Kept as a named switch so the two paths stay
# A/B comparable.
FUSE_INTRA_DIAGONAL = True

CHUNK_KDA_INTER_SOLVE_CONFIGS: tuple[dict[str, int], ...] = (
    {"BK": 64, "num_warps": 1, "num_stages": 2},
    {"BK": 64, "num_warps": 1, "num_stages": 1},
    {"BK": 32, "num_warps": 1, "num_stages": 1},
    {"BK": 32, "num_warps": 1, "num_stages": 2},
)

CHUNK_KDA_INTRA_SUB_CHUNK_CONFIGS: tuple[dict[str, int | bool], ...] = (
    {"BK": 128, "EARLY_AQK_STORE": False, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "EARLY_AQK_STORE": True, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "EARLY_AQK_STORE": True, "num_warps": 1, "num_stages": 2},
)


def get_chunk_kda_inter_solve_config(
    T: int,
    H: int,
    HV: int,
    is_varlen: bool,
    safe_gate: bool,
    fuse_diag: bool = False,
) -> dict[str, int]:
    """Select one of four offline configs for the inter/solve kernel.

    ``BK`` flips with ``fuse_diag``.  Without the fusion the kernel is
    read-bound, so the wide tile wins by halving the number of passes over the
    q/k/g rows.  Folding the diagonal in adds eight ``[BC, BC]`` accumulators,
    a ``q0`` tile and the A2/A4/A8 chain, which pushes it over into being
    register-bound: measured on BW1100 at ``T=16384, H=64`` the fused kernel
    runs 2.46 ms at ``BK=32`` against 2.90 ms at ``BK=64`` (546 vs 463 GB/s),
    even though the narrow tile reads the rows four times instead of twice.
    """
    del H, HV, safe_gate
    if fuse_diag:
        return CHUNK_KDA_INTER_SOLVE_CONFIGS[3]
    if T <= 128:
        return CHUNK_KDA_INTER_SOLVE_CONFIGS[1]
    if is_varlen:
        return CHUNK_KDA_INTER_SOLVE_CONFIGS[1]
    # Fixed length used to take the BK=32 tile.  With K=128 that is four
    # passes over the q/k/g rows where BK=64 needs two, and the unfused kernel
    # is read-bound rather than register-bound, so the wider tile is a strict
    # win: bit-identical output at 1.16-1.26x on BW1100 (T=512..16384,
    # H=16/64).  Varlen already ran a BK=64 tile.
    return CHUNK_KDA_INTER_SOLVE_CONFIGS[0]


def get_chunk_kda_intra_sub_chunk_config(
    T: int, H: int, HV: int, K: int, is_varlen: bool
) -> dict[str, int | bool]:
    """Select one of three offline configs for the polynomial sibling."""
    del T, H, HV, is_varlen
    config = dict(CHUNK_KDA_INTRA_SUB_CHUNK_CONFIGS[1])
    config["BK"] = triton.next_power_of_2(K)
    return config


################################################################################
# Diagonal 16x16 block: products and nilpotent-polynomial inverse
################################################################################


@triton.jit
def _diag_block_products(
    b_q,
    b_k,
    b_g,
    g,
    i_row,
    o_k,
    m_k,
    m_row,
    HV: tl.constexpr,
    K: tl.constexpr,
):
    """The two ``[BC, BC]`` products of one diagonal sub-chunk block.

    ``i_row`` is the *midpoint* row of the sub-chunk, not its first row.  The
    off-diagonal blocks can reference their own first row because every
    exponent they form is then non-positive, but a diagonal block needs both
    ``exp2(+gm)`` and ``exp2(-gm)``; referencing the midpoint halves the span
    each direction has to cover, which is what keeps the discarded upper
    triangle from overflowing FP32.  Measured KDA gate spans reach ~105 in
    log2 over 16 tokens against a 2^128 ceiling, so the halving is load
    bearing, not a micro-optimization.
    """
    b_gn = tl.load(g + i_row * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_gm = (b_g - b_gn[None, :]).to(tl.float32)
    b_gq = tl.where(m_row[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_row[:, None], exp2(-b_gm), 0.0)
    b_right = tl.trans(b_k * b_gk)
    return (
        tl.dot(b_q * b_gq, b_right, input_precision="ieee"),
        tl.dot(b_k * b_gq, b_right, input_precision="ieee"),
    )


@triton.jit
def _invert_strict_lower(A, BC: tl.constexpr):
    """Invert ``I - A`` for strict-lower ``A`` via its nilpotent polynomial.

    Every input and intermediate stays FP32: the published inverse is BF16 but
    narrowing between products compounds the error along the A2/A4/A8 chain.
    """
    o_i = tl.arange(0, BC)
    identity = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    A2 = tl.dot(A, A, input_precision="ieee", out_dtype=tl.float32)
    X = tl.dot(
        identity - A, identity + A2, input_precision="ieee", out_dtype=tl.float32
    )
    A4 = tl.dot(A2, A2, input_precision="ieee", out_dtype=tl.float32)
    X = tl.dot(X, identity + A4, input_precision="ieee", out_dtype=tl.float32)
    A8 = tl.dot(A4, A4, input_precision="ieee", out_dtype=tl.float32)
    return tl.dot(X, identity + A8, input_precision="ieee", out_dtype=tl.float32)


################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    FUSE_DIAG: tl.constexpr,
    SOLVE_TRIL_DOT_PRECISION: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: the diagonal Akk blocks are already inverted in ``Akkd`` --
    unless ``FUSE_DIAG``, in which case this kernel builds them itself and
    ``Akkd`` is never touched.  The separate polynomial kernel reads the same
    ``q``/``k``/``g`` rows this one already has in registers, and both kernels
    run at the measured HBM peak (838 and 782 GB/s on BW1100), so the second
    read pass is pure cost: folding it in removes ~1073 MB of reads plus the
    ``Akkd`` round trip.  ``FUSE_DIAG`` requires ``USE_SAFE_GATE`` since it
    produces the already-inverted form.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

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

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    if FUSE_DIAG:
        b_Adq0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adk0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adq1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adk1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adq2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adk2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adq3 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Adk3 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if FUSE_DIAG:
            p_q0 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_q0 = tl.load(p_q0, boundary_check=(0, 1)).to(tl.float32)
            b_dq, b_dk = _diag_block_products(
                b_q0, b_k0, b_g0, g,
                i_tc0 + min(BC // 2, T - i_tc0 - 1),
                o_k, m_k, m_tc0, HV, K,
            )
            b_Adq0 += b_dq
            b_Adk0 += b_dk

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (HV * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            if FUSE_DIAG:
                b_dq, b_dk = _diag_block_products(
                    b_q1, b_k1, b_g1, g,
                    i_tc1 + min(BC // 2, T - i_tc1 - 1),
                    o_k, m_k, m_tc1, HV, K,
                )
                b_Adq1 += b_dq
                b_Adk1 += b_dk

            b_gn1 = tl.load(g + i_tc1 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (HV * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                if FUSE_DIAG:
                    b_dq, b_dk = _diag_block_products(
                        b_q2, b_k2, b_g2, g,
                        i_tc2 + min(BC // 2, T - i_tc2 - 1),
                        o_k, m_k, m_tc2, HV, K,
                    )
                    b_Adq2 += b_dq
                    b_Adk2 += b_dk

                b_gn2 = tl.load(g + i_tc2 * HV * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (HV * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    if FUSE_DIAG:
                        b_dq, b_dk = _diag_block_products(
                            b_q3, b_k3, b_g3, g,
                            i_tc3 + min(BC // 2, T - i_tc3 - 1),
                            o_k, m_k, m_tc3, HV, K,
                        )
                        b_Adq3 += b_dq
                        b_Adk3 += b_dk

                    b_gn3 = tl.load(g + i_tc3 * HV * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b1 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc1,), (BC,), (0,)
        )
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b2 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc2,), (BC,), (0,)
        )
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b3 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc3,), (BC,), (0,)
        )
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    if FUSE_DIAG:
        m_le = o_i[:, None] >= o_i[None, :]
        m_lt = o_i[:, None] > o_i[None, :]
        # Rows past the sequence end never accumulated anything, so their
        # blocks are zero and every store below is boundary checked; the
        # inverse of a zero block is the identity rather than the zero the
        # ``Akkd`` path would have loaded, which is equally discarded.
        p_bd0 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc0,), (BC,), (0,)
        )
        p_bd1 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc1,), (BC,), (0,)
        )
        p_bd2 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc2,), (BC,), (0,)
        )
        p_bd3 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc3,), (BC,), (0,)
        )
        b_bd0 = tl.load(p_bd0, boundary_check=(0,)).to(tl.float32)
        b_bd1 = tl.load(p_bd1, boundary_check=(0,)).to(tl.float32)
        b_bd2 = tl.load(p_bd2, boundary_check=(0,)).to(tl.float32)
        b_bd3 = tl.load(p_bd3, boundary_check=(0,)).to(tl.float32)

        p_Aqk00 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Aqk11 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
        )
        p_Aqk22 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
        )
        p_Aqk33 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk00,
            (tl.where(m_le, b_Adq0, 0.0) * scale).to(Aqk.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Aqk11,
            (tl.where(m_le, b_Adq1, 0.0) * scale).to(Aqk.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Aqk22,
            (tl.where(m_le, b_Adq2, 0.0) * scale).to(Aqk.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Aqk33,
            (tl.where(m_le, b_Adq3, 0.0) * scale).to(Aqk.dtype.element_ty),
            boundary_check=(0, 1),
        )

        b_Ai00 = _invert_strict_lower(
            tl.where(m_lt, b_Adk0 * b_bd0[:, None], 0.0), BC
        )
        b_Ai11 = _invert_strict_lower(
            tl.where(m_lt, b_Adk1 * b_bd1[:, None], 0.0), BC
        )
        b_Ai22 = _invert_strict_lower(
            tl.where(m_lt, b_Adk2 * b_bd2[:, None], 0.0), BC
        )
        b_Ai33 = _invert_strict_lower(
            tl.where(m_lt, b_Adk3 * b_bd3[:, None], 0.0), BC
        )
    else:
        p_Akk00 = tl.make_block_ptr(
            Akkd, (T, BC), (HV * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akkd, (T, BC), (HV * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akkd, (T, BC), (HV * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akkd, (T, BC), (HV * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # store full Akk_inv to Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
    )
    p_Akk10 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
    )
    p_Akk11 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
    )
    p_Akk20 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
    )
    p_Akk21 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk30 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
    )
    p_Akk31 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
    )
    p_Akk32 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # Zero the strictly-upper triangle so callers get a full BT x BT tile
    # without a separate device-wide memset of ``Akk`` (which is B*T*HV*BT and
    # costs a full extra pass over the largest intra buffer).  Block shapes
    # must stay powers of two, so the six 16x16 blocks above the diagonal are
    # covered by two 16x32 stores and two 16x16 stores.
    b_zero_wide = tl.zeros([BC, 2 * BC], dtype=Akk.dtype.element_ty)
    b_zero = tl.zeros([BC, BC], dtype=Akk.dtype.element_ty)
    tl.store(
        tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc0, BC), (BC, 2 * BC), (1, 0)),
        b_zero_wide,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc0, 3 * BC), (BC, BC), (1, 0)),
        b_zero,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc1, 2 * BC), (BC, 2 * BC), (1, 0)),
        b_zero_wide,
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, 3 * BC), (BC, BC), (1, 0)),
        b_zero,
        boundary_check=(0, 1),
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["B", "T"])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

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

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    if USE_GATHER:
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        # caculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * HV * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    EARLY_AQK_STORE: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Invert a strict-lower 16x16 block with its nilpotent polynomial."""
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_hv = i_bh
    else:
        i_b, i_hv = i_bh // HV, i_bh % HV
        bos = i_b * T
    i_h = i_hv // (HV // H)
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BC
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    if NORMALIZE_QK:
        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)
        b_q = (b_qf * tl.rsqrt(tl.sum(b_qf * b_qf, axis=1)[:, None] + 1e-6)).to(
            tl.bfloat16
        )
        b_k = (b_kf * tl.rsqrt(tl.sum(b_kf * b_kf, axis=1)[:, None] + 1e-6)).to(
            tl.bfloat16
        )
        # The following inter-solve and recompute kernels consume q/k again.
        tl.store(p_q, b_q, boundary_check=(0, 1))
        tl.store(p_k, b_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    o_k = tl.arange(0, BK)
    center = min(BC // 2, T - i_ti - 1)
    b_gn = tl.load(
        g + (i_ti + center) * HV * K + o_k, mask=o_k < K, other=0.0
    )[None, :]
    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)
    b_right = tl.trans(b_k * b_gk)
    b_Aqk = tl.dot(b_q * b_gq, b_right, input_precision="ieee") * scale
    b_Akk = tl.dot(b_k * b_gq, b_right, input_precision="ieee") * b_beta[:, None]
    o_i = tl.arange(0, BC)
    lower_equal = o_i[:, None] >= o_i[None, :]
    lower_strict = o_i[:, None] > o_i[None, :]
    identity = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    b_Aqk = tl.where(lower_equal, b_Aqk, 0.0)
    A = tl.where(lower_strict, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    if EARLY_AQK_STORE:
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Keep every polynomial input and intermediate in FP32.  Although the
    # public inverse output is BF16, narrowing between products compounds the
    # error across the A2/A4/A8 chain.
    A2 = tl.dot(
        A,
        A,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        identity - A,
        identity + A2,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    A4 = tl.dot(
        A2,
        A2,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        X,
        identity + A4,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    A8 = tl.dot(
        A4,
        A4,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        X,
        identity + A8,
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    p_Akk = tl.make_block_ptr(
        Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0)
    )
    if not EARLY_AQK_STORE:
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, X.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@functools.lru_cache(maxsize=1)
def _solve_tril_dot_precision() -> str:
    """Cached device query: it costs ~microseconds and cannot change."""
    if current_platform.is_cuda() and current_platform.has_device_capability(80):
        return "tf32"
    return "ieee"


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    normalize_qk: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    BC = 16
    if BT != 4 * BC:
        # ``chunk_kda_fwd_kernel_inter_solve_fused`` is hand-unrolled for
        # exactly four BC=16 sub-chunks and is on the unconditional path.  Its
        # sub-chunk origins are only guarded against T, not against the chunk
        # boundary, so at BT=16 they address the *next three chunks* and four
        # programs race to overwrite each other's tiles -- silently, since the
        # out-of-range column offsets are clipped by ``boundary_check``.  Fail
        # loudly instead of returning corrupt coefficients.
        raise ValueError(
            f"KDA intra requires chunk_size == {4 * BC}, got {BT}"
        )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # ``chunk_kda_fwd_kernel_inter_solve_fused`` below is the only writer of
    # ``Akk`` and it stores every block of the BT x BT tile, zeros included,
    # so no pre-zeroing pass is needed.
    Akk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)

    # Compute diagonal blocks into Akkd in fp32.  For the model-shape class the
    # polynomial sibling writes the already-inverted 16x16 diagonal blocks, so
    # the inter kernel can skip four serial forward-substitution loops -- or,
    # under ``FUSE_DIAG``, builds them itself from the q/k/g rows it is already
    # reading and never materializes ``Akkd`` at all.  The public
    # token-parallel wrapper still retains its raw-Akkd contract.
    use_poly_solve = K == 128 and BT == 64 and BC == 16
    fuse_diag = use_poly_solve and FUSE_INTRA_DIAGONAL
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = (
        k.new_empty(1, dtype=torch.float32)
        if fuse_diag
        else torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)
    )

    if fuse_diag:
        pass
    elif use_poly_solve:
        grid = (NT, NC, B * HV if cu_seqlens is None else HV)
        config = get_chunk_kda_intra_sub_chunk_config(
            T=T,
            H=H,
            HV=HV,
            K=K,
            is_varlen=cu_seqlens is not None,
        )
        chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            NORMALIZE_QK=normalize_qk,
            **config,
        )
    elif safe_gate:
        grid = (NT, NC, B * HV)
        BK = triton.next_power_of_2(K)
        fallback_config = {"num_warps": 1, "num_stages": 2}
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=is_gather_supported,
            **fallback_config,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    solve_tril_dot_precision = _solve_tril_dot_precision()
    grid = (NT, B * HV)
    diagonal_is_presolved = safe_gate or use_poly_solve
    config = get_chunk_kda_inter_solve_config(
        T=T,
        H=H,
        HV=HV,
        is_varlen=cu_seqlens is not None,
        safe_gate=diagonal_is_presolved,
        fuse_diag=fuse_diag,
    )
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=diagonal_is_presolved,
        FUSE_DIAG=fuse_diag,
        SOLVE_TRIL_DOT_PRECISION=solve_tril_dot_precision,
        **config,
    )
    return Aqk, Akk
