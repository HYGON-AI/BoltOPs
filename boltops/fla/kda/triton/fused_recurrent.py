# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

import torch

import triton
import triton.language as tl

from boltops.fla.common.triton.math import exp, log

from .config_utils import dtype_name, get_kda_tuned_config

cdiv = triton.cdiv
next_power_of_2 = triton.next_power_of_2


# Offline-tuned on BW1000_H, BF16/FP32, D=128.  These candidate sets cover
# decode (tokens, heads)={(1,3),(4,3),(6,3/6/24/96),(8,12),(16,24)}.
KDA_GATE_BETA_CONFIGS = (
    {"BT": 2, "num_warps": 2, "num_stages": 3},  # T=1 and T=6
    {"BT": 2, "num_warps": 8, "num_stages": 2},  # T=4 and T=8
    {"BT": 8, "num_warps": 4, "num_stages": 3},  # T=16
)

FUSED_RECURRENT_KDA_CONFIGS = (
    {"BV": 8, "num_warps": 1, "num_stages": 2},  # B=1
    {"BV": 16, "num_warps": 1, "num_stages": 1},  # fused B=2..31 or H>=24
    {"BV": 8, "num_warps": 1, "num_stages": 1},  # B=4 or B=8
    {"BV": 32, "num_warps": 4, "num_stages": 1},  # fused gate, B>=32
)

FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS = (
    {"BV": 8, "num_warps": 1, "num_stages": 2},  # B=8
    {"BV": 8, "num_warps": 1, "num_stages": 1},  # B<=6
    {"BV": 32, "num_warps": 2, "num_stages": 2},  # B=1 or B>=16
)


def get_kda_gate_beta_config(
    T: int,
    H: int,
    _D: int,
    *,
    b: int,
    dtype: torch.dtype,
) -> dict[str, int]:
    return get_kda_tuned_config(
        "kda_gate_beta",
        f"B={b},dtype={dtype_name(dtype)},T={T},H={H}",
        fallback=static_kda_gate_beta_config(T, H, _D),
    )


def static_kda_gate_beta_config(T: int, H: int, _D: int) -> dict[str, int]:
    if T >= 16:
        return KDA_GATE_BETA_CONFIGS[2]
    if T >= 6 and H >= 24:
        return KDA_GATE_BETA_CONFIGS[0]
    if T >= 4:
        return KDA_GATE_BETA_CONFIGS[1]
    return KDA_GATE_BETA_CONFIGS[0]


def get_fused_recurrent_kda_config(
    B: int,
    _T: int,
    H: int,
    use_gate_in_kernel: bool,
    *,
    dtype: torch.dtype,
) -> dict[str, int]:
    return get_kda_tuned_config(
        "fused_recurrent_kda",
        f"B={B},dtype={dtype_name(dtype)},T={_T},H={H},gate={use_gate_in_kernel}",
        fallback=static_fused_recurrent_kda_config(B, _T, H, use_gate_in_kernel),
    )


def static_fused_recurrent_kda_config(
    B: int,
    _T: int,
    H: int,
    use_gate_in_kernel: bool,
) -> dict[str, int]:
    if use_gate_in_kernel:
        if B <= 1:
            return FUSED_RECURRENT_KDA_CONFIGS[0]
        if B >= 32 and H >= 24:
            return FUSED_RECURRENT_KDA_CONFIGS[3]
        if H < 24:
            return FUSED_RECURRENT_KDA_CONFIGS[2]
        return FUSED_RECURRENT_KDA_CONFIGS[1]

    if B <= 1 or B >= 8:
        return FUSED_RECURRENT_KDA_CONFIGS[1]
    if H >= 24:
        return FUSED_RECURRENT_KDA_CONFIGS[0]
    return FUSED_RECURRENT_KDA_CONFIGS[2]


def get_fused_recurrent_kda_packed_decode_config(
    B: int,
    H: int,
    K: int,
    V: int,
    *,
    dtype: torch.dtype,
) -> dict[str, int]:
    return get_kda_tuned_config(
        "fused_recurrent_kda_packed_decode",
        f"B={B},dtype={dtype_name(dtype)},H={H},K={K},V={V}",
        fallback=static_fused_recurrent_kda_packed_decode_config(B, H, K, V),
    )


def static_fused_recurrent_kda_packed_decode_config(
    B: int,
    H: int,
    K: int,
    V: int,
) -> dict[str, int]:
    if K < 128 or V < 128:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[1]
    if B <= 1:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]
    if B <= 6:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[1]
    if B < 16:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[0]
    return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]


@triton.heuristics(
    {
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit
def _kda_gate_beta_fwd_kernel(
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    gate,
    beta_out,
    lower_bound,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    T,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D

    p_g = raw_g + o_t[:, None] * stride_g_token + i_h * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(
            dt_bias + i_h * D + o_d,
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        b_g += b_bias[None, :]

    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_scaled = b_g * softplus_beta
        b_softplus = tl.where(
            b_scaled > softplus_threshold,
            b_g,
            log(1.0 + tl.exp(b_scaled)) / softplus_beta,
        )
        b_gate = -b_a * b_softplus

    p_gate = gate + (o_t[:, None] * H + i_h) * D + o_d[None, :]
    tl.store(
        p_gate,
        b_gate,
        mask=m_t[:, None] & m_d[None, :],
    )

    b_beta = tl.load(
        raw_beta + o_t * stride_beta_token + i_h,
        mask=m_t,
        other=0.0,
    ).to(tl.float32)
    tl.store(beta_out + o_t * H + i_h, tl.sigmoid(b_beta), mask=m_t)


def _fused_kda_gate_beta(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = raw_g.shape
    assert B == 1
    assert raw_beta.shape == (B, T, H)
    assert raw_g.stride()[2:] == (D, 1)
    assert raw_beta.stride(2) == 1
    gate = torch.empty((B, T, H, D), dtype=torch.float32, device=raw_g.device)
    beta = torch.empty((B, T, H), dtype=torch.float32, device=raw_beta.device)

    config = get_kda_gate_beta_config(
        T, H, D, b=B, dtype=raw_g.dtype,
    )
    grid = (cdiv(T, config["BT"]), H)
    _kda_gate_beta_fwd_kernel[grid](
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        gate=gate,
        beta_out=beta,
        lower_bound=lower_bound,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        T=T,
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        H=H,
        D=D,
        BD=next_power_of_2(D),
        **config,
    )
    return gate, beta


@triton.heuristics(
    {
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    out,
    state,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qkv_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    FAST_L2_NORM: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    i_v = pid % tl.cdiv(V, BV)
    i_nh = pid // tl.cdiv(V, BV)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    sequence_length = eos - bos
    if sequence_length == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_state = m_v[:, None] & m_k[None, :]

    if IS_SPEC_DECODING:
        initial_token = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    else:
        initial_token = 0
    state_index = tl.load(state_indices + i_n * stride_indices_seq + initial_token).to(
        tl.int64
    )
    p_out = out + bos * stride_out_token + i_h * V + o_v
    if state_index <= NULL_STATE_INDEX:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=m_v)
        return

    p_state = (
        state
        + state_index * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_state = tl.load(p_state, mask=m_state, other=0.0).to(tl.float32)

    p_q = q + bos * stride_qkv_token + i_h * K + o_k
    p_k = k + bos * stride_qkv_token + i_h * K + o_k
    p_v = v + bos * stride_qkv_token + i_h * V + o_v
    p_g = g + bos * stride_g_token + i_h * K + o_k
    p_beta = beta + bos * stride_beta_token + i_h
    for i_t in tl.range(0, sequence_length, num_stages=num_stages):
        b_q = tl.load(p_q, mask=m_k, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=m_k, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=m_v, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        if USE_QK_L2NORM_IN_KERNEL:
            if FAST_L2_NORM:
                b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            else:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale

        b_gate = tl.load(
            p_g,
            mask=m_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                b_bias = tl.load(
                    dt_bias + i_h * K + o_k,
                    mask=m_k,
                    other=0.0,
                ).to(tl.float32)
                b_gate += b_bias
            b_a = exp(tl.load(A_log + i_h).to(tl.float32))
            if USE_LOWER_BOUND:
                b_gate = lower_bound * tl.sigmoid(b_a * b_gate)
            else:
                b_softplus = tl.where(
                    b_gate > 20.0,
                    b_gate,
                    log(1.0 + tl.exp(b_gate)),
                )
                b_gate = -b_a * b_softplus

        b_state *= exp(b_gate[None, :])
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]
        b_out = tl.sum(b_state * b_q[None, :], axis=1)
        tl.store(
            p_out,
            b_out.to(p_out.dtype.element_ty),
            mask=m_v,
            eviction_policy="evict_first",
        )

        final_state_index = tl.load(state_indices + i_n * stride_indices_seq + i_t).to(
            tl.int64
        )
        if final_state_index > NULL_STATE_INDEX:
            p_final_state = (
                state
                + final_state_index * stride_state_token
                + i_h * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_final_state,
                b_state.to(p_final_state.dtype.element_ty),
                mask=m_state,
            )

        p_q += stride_qkv_token
        p_k += stride_qkv_token
        p_v += stride_qkv_token
        p_g += stride_g_token
        p_beta += stride_beta_token
        p_out += stride_out_token


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
    null_state_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch recurrent KDA with dense inner dimensions and row strides.

    ``null_state_index`` selects the state-index ABI: ``0`` (the default,
    vLLM NULL_BLOCK_ID) treats indices ``<= 0`` as invalid, so valid slots
    are 1-based; ``-1`` (SGLang) treats only negative indices as invalid --
    SGLang's allocator never hands slot 0 to a live request and reserves it
    as the dummy write target for padded requests.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert B == 1 and k.shape == q.shape
    assert v.shape == (B, T, H, V) and g.shape == (B, T, H, K)
    assert beta.shape == (B, T, H)
    assert initial_state is not None
    assert cu_seqlens is not None
    assert ssm_state_indices is not None
    assert inplace_final_state
    if out is None:
        out = torch.empty_like(v)
    assert out.shape == v.shape
    assert initial_state.shape[1:] == (H, V, K)
    assert ssm_state_indices.ndim in (1, 2)

    assert q.stride()[2:] == k.stride()[2:] == (K, 1)
    assert v.stride()[2:] == out.stride()[2:] == (V, 1)
    assert g.stride()[2:] == (K, 1)
    assert beta.stride(2) == 1
    assert q.stride(1) == k.stride(1) == v.stride(1)
    assert initial_state.stride()[1:] == (V * K, K, 1)
    if null_state_index not in (-1, 0):
        raise ValueError("null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI).")
    N = cu_seqlens.numel() - 1
    if ssm_state_indices.ndim == 1:
        assert T == N
        assert num_accepted_tokens is None
    else:
        assert ssm_state_indices.stride(1) == 1
    assert cu_seqlens.is_contiguous()
    if use_gate_in_kernel:
        assert A_log is not None and A_log.is_contiguous()
        assert dt_bias is None or dt_bias.is_contiguous()

    if scale is None:
        scale = K**-0.5

    config = get_fused_recurrent_kda_config(
        N, T, H, use_gate_in_kernel, dtype=q.dtype
    )
    grid = (cdiv(V, config["BV"]) * N * H,)
    fused_recurrent_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
        N=N,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=next_power_of_2(K),
        stride_qkv_token=q.stride(1),
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        stride_out_token=out.stride(1),
        stride_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        FAST_L2_NORM=H >= 24,
        NULL_STATE_INDEX=null_state_index,
        **config,
    )
    return out, initial_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
    null_state_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent KDA from raw gate and beta inputs.

    This vLLM wrapper applies the gate activation and beta sigmoid, selecting
    whether to materialize them before launching the recurrent kernel.

    ``null_state_index`` selects the state-index ABI: ``0`` (the default,
    vLLM NULL_BLOCK_ID) treats indices ``<= 0`` as invalid; ``-1`` (SGLang)
    treats only negative indices as invalid -- SGLang reserves slot 0 as the
    dummy write target for padded requests and never allocates it to a live
    request.
    """
    if fuse_gate is None:
        # gfx950: always fuse the gate.
        fuse_gate = True

    if fuse_gate:
        gate = raw_g
        beta = raw_beta
    else:
        gate, beta = _fused_kda_gate_beta(
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound,
        )
    return fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        A_log=A_log if fuse_gate else None,
        dt_bias=dt_bias if fuse_gate else None,
        lower_bound=lower_bound if fuse_gate else None,
        use_gate_in_kernel=fuse_gate,
        use_beta_sigmoid_in_kernel=fuse_gate,
        out=out,
        null_state_index=null_state_index,
    )


@triton.jit
def fused_recurrent_kda_packed_decode_kernel(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    FAST_L2_NORM: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= NULL_STATE_INDEX:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_state = state + state_idx * stride_state_token
    p_state += i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    # Q, K, and V occupy consecutive channel ranges, while the token stride
    # may also include the output-gate projection that follows packed QKV.
    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    if FAST_L2_NORM:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    else:
        b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = raw_g + i_n * stride_g_token + i_h * K + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_g += b_bias
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_softplus = tl.where(
            b_g > SOFTPLUS_THRESHOLD,
            b_g,
            log(1.0 + tl.exp(b_g)),
        )
        b_gate = -b_a * b_softplus

    b_state *= exp(b_gate[None, :])
    b_v -= tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )
    b_v *= b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


@triton.jit
def fused_recurrent_kda_packed_decode_3d_kernel(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FAST_L2_NORM: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
):
    """Bounded-gate packed decode with direct (V tile, head, token) IDs."""
    i_v, i_h, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= NULL_STATE_INDEX:
        tl.store(p_out, 0.0, mask=mask_v)
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(
        p_mixed + i_h * K + o_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    if FAST_L2_NORM:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    else:
        b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    b_g = tl.load(
        raw_g + i_n * stride_g_token + i_h * K + o_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    b_bias = tl.load(
        dt_bias + i_h * K + o_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_gate = lower_bound * tl.sigmoid(b_a * (b_g + b_bias))
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )

    p_state = (
        state
        + state_idx * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    mask_state = mask_v[:, None] & mask_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0.0).to(tl.float32)
    b_state *= exp(b_gate)[None, :]
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)
    b_v = (b_v - tl.sum(b_state * b_k[None, :], axis=1)) * b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
    null_state_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-token KDA decode directly from packed post-conv QKV.

    ``null_state_index`` selects the state-index ABI so both frameworks can
    call the same entry: ``0`` (the default, vLLM NULL_BLOCK_ID) treats
    indices ``<= 0`` as invalid and valid slots are 1-based; ``-1`` (SGLang)
    treats only negative indices as invalid -- SGLang reserves slot 0 as the
    dummy write target for padded requests and never allocates it to a live
    request.
    """
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be 2D and contiguous in its last dim.")
    if raw_g.ndim != 4 or raw_g.shape[0] != 1:
        raise ValueError("`raw_g` must have shape [1, B, H, K].")
    if raw_beta.ndim != 3 or raw_beta.shape[0] != 1:
        raise ValueError("`raw_beta` must have shape [1, B, H].")
    if initial_state.ndim != 4:
        raise ValueError("`initial_state` must have shape [cache, H, V, K].")
    _, H, V, K = initial_state.shape
    if raw_g.stride()[2:] != (K, 1):
        raise ValueError("`raw_g` must be contiguous within each token.")
    if raw_beta.stride(2) != 1:
        raise ValueError("`raw_beta` heads must be contiguous.")
    if initial_state.stride()[1:] != (V * K, K, 1):
        raise ValueError("`initial_state` must be contiguous within each cache slot.")
    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        raise ValueError("`state_indices` must be contiguous and one-dimensional.")
    if A_log.ndim != 1 or not A_log.is_contiguous():
        raise ValueError("`A_log` must be contiguous and one-dimensional.")
    if not dt_bias.is_contiguous():
        raise ValueError("`dt_bias` must be contiguous.")

    device = mixed_qkv.device
    if any(
        x.device != device
        for x in (raw_g, raw_beta, A_log, dt_bias, initial_state, state_indices)
    ):
        raise ValueError("All packed KDA inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if raw_g.shape != (1, B, H, K):
        raise ValueError(f"Unexpected raw gate shape {tuple(raw_g.shape)}.")
    if raw_beta.shape != (1, B, H):
        raise ValueError(f"Unexpected raw beta shape {tuple(raw_beta.shape)}.")
    if mixed_qkv.shape[1] != 2 * H * K + H * V:
        raise ValueError(f"Unexpected packed QKV shape {tuple(mixed_qkv.shape)}.")
    if A_log.numel() != H or dt_bias.numel() != H * K:
        raise ValueError("`A_log` or `dt_bias` has an incompatible shape.")
    if state_indices.shape[0] != B:
        raise ValueError("`state_indices` must contain one entry per token.")

    BK = next_power_of_2(K)
    if scale is None:
        scale = K**-0.5
    if null_state_index not in (-1, 0):
        raise ValueError("null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI).")

    out = torch.empty((1, B, H, V), dtype=mixed_qkv.dtype, device=device)
    config = get_fused_recurrent_kda_packed_decode_config(
        B, H, K, V, dtype=mixed_qkv.dtype
    )
    use_direct_3d = (
        lower_bound is not None
        and K == 128
        and V == 128
        and H >= 24
        and initial_state.dtype == torch.float32
    )
    if use_direct_3d:
        grid = (cdiv(V, config["BV"]), H, B)
        fused_recurrent_kda_packed_decode_3d_kernel[grid](
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            state=initial_state,
            state_indices=state_indices,
            lower_bound=lower_bound,
            scale=scale,
            stride_mixed_token=mixed_qkv.stride(0),
            stride_g_token=raw_g.stride(1),
            stride_beta_token=raw_beta.stride(1),
            stride_state_token=initial_state.stride(0),
            H=H,
            K=K,
            V=V,
            BK=BK,
            FAST_L2_NORM=H >= 24,
            NULL_STATE_INDEX=null_state_index,
            **config,
        )
    else:
        # Preserve the upstream softplus and generic-shape implementation.
        # Keep the upstream generic launch for small heads, softplus gates,
        # non-square dimensions, and non-FP32 state.  The direct-3D sibling
        # is deliberately restricted to the shape family where it wins.
        config = {
            "BV": min(next_power_of_2(V), 32),
            "num_warps": 4,
            "num_stages": 2,
        }
        grid = (cdiv(V, config["BV"]), B * H)
        fused_recurrent_kda_packed_decode_kernel[grid](
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            state=initial_state,
            state_indices=state_indices,
            lower_bound=lower_bound or 0.0,
            scale=scale,
            stride_mixed_token=mixed_qkv.stride(0),
            stride_g_token=raw_g.stride(1),
            stride_beta_token=raw_beta.stride(1),
            stride_state_token=initial_state.stride(0),
            H=H,
            K=K,
            V=V,
            BK=BK,
            SOFTPLUS_THRESHOLD=20.0,
            USE_LOWER_BOUND=lower_bound is not None,
            FAST_L2_NORM=H >= 24,
            NULL_STATE_INDEX=null_state_index,
            **config,
        )
    return out, initial_state
