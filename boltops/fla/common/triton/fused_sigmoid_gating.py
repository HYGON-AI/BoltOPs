# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Tuple

import functools
import json
import os

import torch
import triton
import triton.language as tl

from boltops.utility import arch_info, logger

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")

# HAS_DUMPED_SIGMOID_GATING_KERNEL_METADATA = False
TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"

_DEFAULT_FUSED_SIGMOID_GATING_CONFIG = {
    "BV": 32,
    "num_warps": 1,
}


@functools.lru_cache(maxsize=1)
def _load_fused_sigmoid_gating_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "fused_sigmoid_gating_delta_rule_update",
        f"fused_sigmoid_gating_delta_rule_update-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"fused_sigmoid_gating_delta_rule_update config not found at {path}, "
            f"using default {_DEFAULT_FUSED_SIGMOID_GATING_CONFIG}."
        )
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("config", {}) if isinstance(payload, dict) else {}


@functools.lru_cache
def _get_fused_sigmoid_gating_config(T: int, H: int, HV: int) -> dict:
    cfgs = _load_fused_sigmoid_gating_configs()
    key = f"T={T},H={H},HV={HV}"
    cfg = cfgs.get(key)
    if cfg is None:
        candidates = []
        for k, v in cfgs.items():
            if k == "default":
                continue
            try:
                parts = {x.split("=")[0]: int(x.split("=")[1]) for x in k.split(",")}
            except Exception:
                continue
            if parts.get("H") == H and parts.get("HV") == HV and "T" in parts:
                candidates.append((abs(parts["T"] - T), parts["T"], v))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, nearest_t, cfg = candidates[0]
            if TRITON_CONFIG_CHECK:
                logger.warning(
                    f"fused_sigmoid_gating config key '{key}' not found, "
                    f"using nearest-T config with T={nearest_t}: {cfg}."
                )
    if cfg is None:
        default_cfg = cfgs.get("default", _DEFAULT_FUSED_SIGMOID_GATING_CONFIG)
        if TRITON_CONFIG_CHECK:
            logger.warning(
                f"fused_sigmoid_gating config key '{key}' not found, "
                f"using default config: {default_cfg}."
            )
        cfg = default_cfg
    merged = dict(_DEFAULT_FUSED_SIGMOID_GATING_CONFIG)
    merged.update(cfg)
    return merged


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_A_log = tl.exp(tl.load(p_A_log).to(tl.float32))
    if not IS_KDA:
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                accepted = tl.load(num_accepted_tokens + i_n).to(tl.int64)
                if accepted <= 0:
                    return
                i_t = accepted - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        if not IS_KDA:
            x = tl.load(p_a).to(tl.float32) + b_dt_bias
        else:
            x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -b_A_log * softplus_x
        b_beta = tl.sigmoid(b_b)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        # KDA gate input is [B, T, HV, K]: the token pitch is HV*K, not HV.
        if IS_KDA:
            p_a += HV * K
        else:
            p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    kernel_cfg: dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global HAS_DUMPED_SIGMOID_GATING_KERNEL_METADATA

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    cfg = kernel_cfg if kernel_cfg is not None else _get_fused_sigmoid_gating_config(T, H, HV)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_warps = int(cfg["num_warps"])
    if NK != 1:
        raise ValueError(f"NK > 1 is not supported (K={K}, BK={BK}, NK={NK}).")

    if scale is None:
        scale = K**-0.5
    elif scale <= 0:
        raise ValueError("scale must be positive.")

    if initial_state is None:
        raise ValueError("initial_state must not be None.")

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"q.shape[0] must be 1 when using cu_seqlens, got {q.shape[0]}."
        )

    o = q.new_empty(NK, *v.shape)
    final_state = initial_state if inplace_final_state else q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    elif ssm_state_indices.ndim == 2:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
    else:
        raise ValueError(
            f"ssm_state_indices must be 1D/2D when provided, got ndim={ssm_state_indices.ndim}."
        )

    grid = (NK, NV, N * HV)
    compiled_kernel = fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=1,
    )
    '''
    if not HAS_DUMPED_SIGMOID_GATING_KERNEL_METADATA and compiled_kernel is not None:
        print("sigmoid gating kernel metadata")
        print(f"  grid: {grid}")
        print(f"  registers: {compiled_kernel.n_regs}")
        print(f"  spills: {compiled_kernel.n_spills}")
        print(f"  shared memory: {compiled_kernel.metadata.shared} bytes")
        HAS_DUMPED_SIGMOID_GATING_KERNEL_METADATA = True
    '''
    return o.squeeze(0), final_state
