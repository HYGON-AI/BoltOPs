# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/fla-org/flash-linear-attention/blob/3a9ce1c83a13994d824dbb3421e2989d330bb38b/fla/ops/common/chunk_delta_h.py

from __future__ import annotations

import functools
import json
import os

import torch
import triton
import triton.language as tl

from boltops.fla.common.triton.math import exp, exp2, safe_exp, safe_exp2
from boltops.fla.common.triton.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from boltops.utility import arch_info, logger

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")

TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"


_DEFAULT_CHUNK_DELTA_H_CONFIG = {
    "BV": 32,
    "num_warps": 8,
    "num_stages": 2,
}


@functools.lru_cache(maxsize=1)
def _load_chunk_delta_h_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "chunk_gated_delta_rule_fwd_h",
        f"chunk_gated_delta_rule_fwd_h-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"chunk_gated_delta_rule_fwd_h config not found at {path}, using default {_DEFAULT_CHUNK_DELTA_H_CONFIG}."
        )
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("config", {}) if isinstance(payload, dict) else {}


@functools.lru_cache
def _get_chunk_delta_h_config(
    K: int, V: int, BT: int, H: int, is_varlen: bool = False
) -> dict:
    cfgs = _load_chunk_delta_h_configs()
    shape_key = f"K={K},V={V},BT={BT},H={H}"
    mode_key = f"{shape_key},IS_VARLEN=True"
    cfg = cfgs.get(mode_key) if is_varlen else None
    if cfg is None:
        cfg = cfgs.get(shape_key)

    if cfg is None:
        default_cfg = cfgs.get("default", _DEFAULT_CHUNK_DELTA_H_CONFIG)
        if TRITON_CONFIG_CHECK:
            logger.warning(
                "chunk_gated_delta_rule_fwd_h config missing for "
                f"{mode_key if is_varlen else shape_key}, using default config "
                f"{default_cfg}."
            )
        cfg = default_cfg
    merged = dict(_DEFAULT_CHUNK_DELTA_H_CONFIG)
    merged.update(cfg)
    return merged


def launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    *,
    k: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    v_new: torch.Tensor | None,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    h: torch.Tensor,
    initial_state: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None,
    final_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_offsets: torch.LongTensor | None,
    N: int,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    BT: int,
    use_exp2: bool,
    transpose_state_layout: bool,
    inplace_final_state: bool,
    null_state_index: int,
    kernel_cfg: dict | None,
):
    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    cfg = (
        kernel_cfg
        if kernel_cfg is not None
        else _get_chunk_delta_h_config(
            K, V, BT, H, is_varlen=cu_seqlens is not None
        )
    )
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        initial_state_indices=initial_state_indices,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=cfg["BV"],
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        INPLACE_FINAL_STATE=inplace_final_state,
        STATE_ROWS=initial_state.shape[0] if initial_state is not None else 0,
        NULL_STATE_INDEX=null_state_index,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "USE_INITIAL_STATE_INDICES": lambda args: args["initial_state_indices"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
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
    initial_state_indices,
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
    USE_INITIAL_STATE_INDICES: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    i_s = i_n
    if USE_INITIAL_STATE_INDICES:
        i_s = tl.load(initial_state_indices + i_n).to(tl.int32)
    valid_initial_state = (i_s > NULL_STATE_INDEX) & (i_s < STATE_ROWS)

    if USE_INITIAL_STATE:
        if valid_initial_state:
            h0 = h0 + (i_s * H + i_h).to(tl.int64) * K * V
    if STORE_FINAL_STATE:
        i_ht = i_n
        valid_final_state = True
        if INPLACE_FINAL_STATE:
            i_ht = i_s
            valid_final_state = valid_initial_state
        if valid_final_state:
            ht = ht + (i_ht * H + i_h).to(tl.int64) * K * V

    if USE_INITIAL_STATE:
        if valid_initial_state:
            if TRANSPOSE_STATE:
                p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            else:
                p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                if TRANSPOSE_STATE:
                    p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                else:
                    p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                if TRANSPOSE_STATE:
                    p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                else:
                    p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                if TRANSPOSE_STATE:
                    p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                else:
                    p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, safe_exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, safe_exp(b_g_last - b_g), 0)[:, None]
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
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if valid_final_state:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                if TRANSPOSE_STATE:
                    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                else:
                    p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                if TRANSPOSE_STATE:
                    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                else:
                    p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                if TRANSPOSE_STATE:
                    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                else:
                    p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = True,
    null_state_index: int = -1,
    kernel_cfg: dict | None = None,
):
    """Compute chunk states and optionally persist the final recurrent state.

    ``null_state_index`` selects the state-index ABI for
    ``initial_state_indices`` so both frameworks can call the same entry:
    ``-1`` (the default, SGLang) treats only negative indices as invalid --
    SGLang reserves slot 0 as the dummy write target for padded requests and
    never allocates it to a live request; ``0`` (vLLM NULL_BLOCK_ID) treats
    indices ``<= 0`` as invalid and valid slots are 1-based.

    With ``inplace_final_state=True``, the kernel writes directly into
    ``initial_state`` and the third return value is that same tensor object.
    The caller must ensure that ``initial_state_indices`` contains no duplicate
    valid indices in this mode, or concurrent sequences may race on one row.
    """
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size
    if null_state_index not in (-1, 0):
        raise ValueError("null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI).")

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if inplace_final_state and initial_state is None:
        raise ValueError("initial_state is required when inplace_final_state=True.")
    if inplace_final_state and not output_final_state:
        raise ValueError(
            "output_final_state must be True when inplace_final_state=True."
        )

    expected_state_shape = (H, V, K) if transpose_state_layout else (H, K, V)
    if initial_state is not None:
        if (
            initial_state.ndim != 4
            or tuple(initial_state.shape[1:]) != expected_state_shape
        ):
            raise ValueError(
                f"initial_state trailing shape must be {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}."
            )
        if initial_state.device != k.device:
            raise ValueError("initial_state and k must be on the same device.")
    if initial_state_indices is not None:
        if initial_state is None:
            raise ValueError("initial_state_indices requires initial_state.")
        if initial_state_indices.ndim != 1 or len(initial_state_indices) != N:
            raise ValueError(f"initial_state_indices must have shape ({N},).")
        if initial_state_indices.device != k.device:
            raise ValueError("initial_state_indices and k must be on the same device.")
        if initial_state_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("initial_state_indices must have int32 or int64 dtype.")

    state_shape = (N, *expected_state_shape)
    h = k.new_empty(B, NT, *state_shape[1:])
    if not output_final_state:
        final_state = None
    elif inplace_final_state:
        final_state = initial_state
    else:
        final_state = k.new_empty(*state_shape, dtype=torch.float32)
    v_new = torch.empty_like(u) if save_new_value else None

    launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
        k=k,
        u=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        N=N,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        inplace_final_state=inplace_final_state,
        null_state_index=null_state_index,
        kernel_cfg=kernel_cfg,
    )
    return h, v_new, final_state
