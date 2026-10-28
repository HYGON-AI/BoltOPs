# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/fla-org/flash-linear-attention/blob/3a9ce1c83a13994d824dbb3421e2989d330bb38b/fla/ops/common/chunk_o.py

from __future__ import annotations

import functools
import json
import os

import torch
import triton
import triton.language as tl

from boltops.fla.common.triton.math import exp, exp2, safe_exp, safe_exp2
from boltops.fla.common.triton.utils import prepare_chunk_indices
from boltops.utility import arch_info, logger

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")

TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"


_DEFAULT_CHUNK_O_CONFIG = {
    "BK": 128,
    "BV": 64,
    "num_warps": 4,
    "num_stages": 2,
}


@functools.lru_cache(maxsize=1)
def _load_chunk_o_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "chunk_fwd_o",
        f"chunk_fwd_o-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"chunk_fwd_o config not found at {path}, using default {_DEFAULT_CHUNK_O_CONFIG}."
        )
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("config", {}) if isinstance(payload, dict) else {}


@functools.lru_cache
def _get_chunk_o_config(K: int, V: int, BT: int, transpose_state_layout: bool) -> dict:
    cfgs = _load_chunk_o_configs()
    key = f"K={K},V={V},BT={BT}"
    cfg = cfgs.get(key)
    if cfg is None:
        default_cfg = cfgs.get("default", _DEFAULT_CHUNK_O_CONFIG)
        if TRITON_CONFIG_CHECK:
            logger.warning(
                "chunk_fwd_o config missing for "
                f"{key}, using default config {default_cfg}."
            )
        cfg = default_cfg
    merged = dict(_DEFAULT_CHUNK_O_CONFIG)
    merged.update(cfg)
    return merged


def launch_chunk_fwd_kernel_o(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    o: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    scale: float,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    BT: int,
    NT: int,
    B: int,
    use_exp2: bool,
    transpose_state_layout: bool,
    kernel_cfg: dict | None,
):
    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    cfg = kernel_cfg if kernel_cfg is not None else _get_chunk_o_config(K, V, BT, transpose_state_layout)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=cfg["BK"],
        BV=cfg["BV"],
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        if USE_EXP2:
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * safe_exp2(b_g[:, None] - b_g[None, :])
        else:
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        if USE_EXP2:
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * safe_exp2(b_g[:, None] - b_g[None, :])
        else:
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    kernel_cfg: dict | None = None,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    launch_chunk_fwd_kernel_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        NT=NT,
        B=B,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        kernel_cfg=kernel_cfg,
    )
    return o
