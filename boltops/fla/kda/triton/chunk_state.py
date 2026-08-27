"""KDA chunk state propagation (``chunk_gated_delta_rule_fwd_h``).

Port of the vLLM/FLA delta-rule chunk state kernel used by the ``chunk_kda``
orchestration. It consumes the recomputed ``w/u/kg`` chunk intermediates and
produces per-chunk states ``h``, the decayed ``v_new`` and the optional final
state. Kernel body is kept verbatim from upstream; only the autotune decorator
is replaced by the repository's bounded static-config convention.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...common.triton.math import exp, exp2
from ...common.triton.utils import (
    FLA_CHUNK_SIZE,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

# BW1100
_NUM_CU = 72

CHUNK_STATE_CONFIGS: tuple[dict[str, int], ...] = (
    {"BV": 64, "num_warps": 4, "num_stages": 2},
    {"BV": 32, "num_warps": 2, "num_stages": 2},
)


def get_chunk_state_config(
    N: int,
    H: int,
    V: int,
    K: int,
    is_varlen: bool,
    T: int | None = None,
) -> dict[str, int]:
    del N
    # On BW1000_H the packed varlen launch has enough independent programs
    # that the smaller BV32/s2 tile wins for the K/V=128 class.  Fixed length
    # still benefits from the wider BV64 tile.  K/V=256 remains BV32/s2
    # because the wider alternatives exceed the 64 KiB LDS limit at large H.
    if K >= 256:
        # The K=256 accumulator carries four 64-column FP32 tiles.  Two
        # software-pipeline stages exceed BW1000_H's 64 KiB LDS limit even
        # with BV32, so use a single stage as the resource-safe fallback.
        return {"BV": 32, "num_warps": 2, "num_stages": 1}
    if (
        not is_varlen
        and T is not None
        and T >= 8192
        and V >= 64
        and K <= 128
    ):
        if H <= 16:
            return CHUNK_STATE_CONFIGS[1]
        if H <= 32:
            return {"BV": 32, "num_warps": 4, "num_stages": 1}
    if is_varlen and V >= 64 and K <= 128:
        return CHUNK_STATE_CONFIGS[1]
    if V >= 64 and K <= 128:
        return CHUNK_STATE_CONFIGS[0]
    return CHUNK_STATE_CONFIGS[1]



def get_fused_state_output_config(n_programs: int, V: int) -> dict[str, int]:
    """Pick the V-block for the fused state/output kernel.

    The fused grid is ``(V/BV, N*H)``.  Splitting V costs real bandwidth --
    every V-block re-reads the whole K side (``q``, ``kg``, ``w``, ``g``,
    ``Aqk``) -- so the only reason to split is to put more workgroups on the
    device.  The wide ``BV=V`` tile also carries the full ``[V,K]`` FP32
    accumulator (64 KiB at K/V=128), so it can never co-schedule two
    workgroups on one CU: once its grid exceeds a single wave it stops hiding
    latency and the narrower tile wins outright.

    Measured on BW1100 (72 CU, K/V=128, T=16384), best config per ``N*H``:

    | ``N*H`` | 16 | 24 | 32 | 40 | 48 | 64 | 72 | 96 |
    |---|---|---|---|---|---|---|---|---|
    | ``BV`` | 32 | 64 | 64 | 128 | 128 | 128 | 128 | 64 |

    The thresholds below are those measurements expressed as fractions of one
    CU wave; ``_NUM_CU`` only steers the choice, so a stale value costs
    performance, never correctness.
    """
    # ``num_warps=16`` beat ``num_warps=8`` on the wide ``BV=V`` tile by
    # 1.15-1.20x across every H that lands there (T=16384, K/V=128, BW1100:
    # H=32/48/64 all go 2.89 -> ~2.41 ms).  The chunk loop is serial in time
    # (the ``[V,K]`` accumulator has cross-chunk deps), so extra CUs don't
    # help; the only lever is to widen the workgroup so more lanes cover the
    # loop's latency together.  On the narrower tiles this same widening
    # spills registers and loses (H=24 at BV=64 goes 1.66 -> 2.35 ms, and
    # H=16 at BV=32 goes 0.46 -> 0.64 ms), so those branches keep their
    # previous warp counts.
    if n_programs > _NUM_CU:
        # More than one wave of the wide tile: fall back to the narrow tile,
        # which co-resides and hides its duplicated K-side reads.
        return {"BV": min(64, V), "num_warps": 4, "num_stages": 1}
    if n_programs * 2 >= _NUM_CU:
        return {"BV": V, "num_warps": 16, "num_stages": 2}
    if n_programs * 3 >= _NUM_CU:
        return {"BV": min(64, V), "num_warps": 8, "num_stages": 2}
    return {"BV": min(32, V), "num_warps": 4, "num_stages": 2}

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, 64),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 128),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 192),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_state_output_fused(
    q,
    k,
    w,
    u,
    g,
    Aqk,
    o,
    h0,
    ht,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fuse KDA chunk state propagation and output for same-head KDA inputs."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0 += i_nh * V * K
        p_h0 = tl.make_block_ptr(
            h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        b_h1 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    stride_qk = H * K
    stride_v = H * V
    stride_a = H * BT
    for i_t in range(NT):
        q_base = q + (bos * H + i_h) * K
        k_base = k + (bos * H + i_h) * K
        g_base = g + (bos * H + i_h) * K
        a_base = Aqk + (bos * H + i_h) * BT
        o_base = o + (bos * H + i_h) * V
        w_base = w + (bos * H + i_h) * K
        u_base = u + (bos * H + i_h) * V

        b_v = tl.zeros([BT, BV], dtype=tl.float32)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in tl.static_range(0, K // 64 if K % 64 == 0 else K // 64 + 1):
            i_k0 = i_k * 64
            o_k = i_k0 + tl.arange(0, 64)
            m_k = o_k < K
            p_q = tl.make_block_ptr(
                q_base, (T, K), (stride_qk, 1), (i_t * BT, i_k0),
                (BT, 64), (1, 0)
            )
            p_g = tl.make_block_ptr(
                g_base, (T, K), (stride_qk, 1), (i_t * BT, i_k0),
                (BT, 64), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            b_qg = (b_q * tl.exp2(b_g)).to(b_q.dtype)
            if i_k == 0:
                b_h = b_h1.to(b_q.dtype)
            elif i_k == 1:
                b_h = b_h2.to(b_q.dtype)
            elif i_k == 2:
                b_h = b_h3.to(b_q.dtype)
            else:
                b_h = b_h4.to(b_q.dtype)
            b_o += tl.dot(b_qg, tl.trans(b_h))
            p_w = tl.make_block_ptr(
                w_base, (T, K), (stride_qk, 1), (i_t * BT, i_k0),
                (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h))

        p_u = tl.make_block_ptr(
            u_base, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
            (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32) - b_v

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        o_k = tl.arange(0, 64)
        b_g_last1 = tl.load(
            g + (bos + last_idx) * H * K + i_h * K + o_k,
            mask=o_k < K,
            other=0.0,
        )
        b_h1 *= tl.exp2(b_g_last1)[None, :]
        if K > 64:
            o_k2 = 64 + o_k
            b_g_last2 = tl.load(
                g + (bos + last_idx) * H * K + i_h * K + o_k2,
                mask=o_k2 < K,
                other=0.0,
            )
            b_h2 *= tl.exp2(b_g_last2)[None, :]
        if K > 128:
            o_k3 = 128 + o_k
            b_g_last3 = tl.load(
                g + (bos + last_idx) * H * K + i_h * K + o_k3,
                mask=o_k3 < K,
                other=0.0,
            )
            b_h3 *= tl.exp2(b_g_last3)[None, :]
        if K > 192:
            o_k4 = 192 + o_k
            b_g_last4 = tl.load(
                g + (bos + last_idx) * H * K + i_h * K + o_k4,
                mask=o_k4 < K,
                other=0.0,
            )
            b_h4 *= tl.exp2(b_g_last4)[None, :]

        b_v = b_v.to(k.dtype.element_ty)
        p_A = tl.make_block_ptr(
            a_base, (T, BT), (stride_a, 1), (i_t * BT, 0),
            (BT, BT), (1, 0)
        )
        rows = tl.arange(0, BT)
        causal = rows[:, None] >= rows[None, :]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(tl.where(causal, b_A, 0).to(b_v.dtype), b_v)
        p_o = tl.make_block_ptr(
            o_base, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
            (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        for i_k in tl.static_range(0, K // 64 if K % 64 == 0 else K // 64 + 1):
            i_k0 = i_k * 64
            p_k = tl.make_block_ptr(
                k_base, (K, T), (1, stride_qk), (i_k0, i_t * BT),
                (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if i_k == 0:
                b_h1 += tl.trans(tl.dot(b_k, b_v))
            elif i_k == 1:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            elif i_k == 2:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        ht += i_nh * V * K
        p_ht = tl.make_block_ptr(
            ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_ht, b_h1, boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2, boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3, boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4, boundary_check=(0, 1))


def chunk_kda_fwd_state_output_fused(
    q: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    kg: torch.Tensor,
    Aqk: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse KDA chunk state propagation and output.

    Takes the materialized WY factors ``w = Ab @ (k * exp2(g))`` and
    ``u = Ab @ v``.  Rebuilding them here from ``k``/``v``/``A`` was measured
    slower: it forces an extra ``Ab`` tile and matmul into the serial chunk
    loop, and this kernel runs at ~700 GB/s, so it has no bandwidth headroom
    to trade for arithmetic.
    """
    B, T, H, K = q.shape
    if kg.shape != (B, T, H, K):
        raise ValueError("KDA fused state-output requires same-head kg")
    if g.shape != kg.shape:
        raise ValueError("KDA fused state-output expects matching K tensors")
    if w.shape != kg.shape:
        raise ValueError("KDA fused state-output expects matching K tensors")
    value = u
    if value.shape[:3] != (B, T, H) or o.shape != value.shape:
        raise ValueError("KDA fused state-output expects matching value tensors")
    V = value.shape[-1]
    BT = chunk_size
    if BT != 64 or K > 256 or V > 256:
        raise ValueError("KDA fused state-output supports BT=64 and K/V<=256")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, BT)
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    final_state = (
        kg.new_empty(N, H, V, K, dtype=torch.float32)
        if output_final_state else None
    )
    config = get_chunk_state_config(N, H, V, K, cu_seqlens is not None, T=T)
    # The fused path keeps both the state and output tiles live; its dot-heavy
    # body benefits from one additional wave of issue parallelism at K=256.
    if K <= 128 and V >= 64:
        config = get_fused_state_output_config(N * H, V)
    elif K >= 256:
        config = {**config, "num_warps": 4}
    else:
        config = {"BV": min(config["BV"], 32), "num_warps": 2, "num_stages": 1}

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_kda_fwd_kernel_state_output_fused[grid](
        q=q,
        k=kg,
        w=w,
        u=u,
        g=g,
        Aqk=Aqk,
        o=o,
        h0=initial_state,
        ht=final_state,
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
    return o, final_state


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None
    config = get_chunk_state_config(N, H, V, K, cu_seqlens is not None, T=T)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        **config,
    )
    return h, v_new, final_state
