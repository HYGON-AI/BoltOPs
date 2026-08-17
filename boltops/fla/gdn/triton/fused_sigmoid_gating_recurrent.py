from __future__ import annotations

from typing import Optional

import functools
import json
import os

import torch
import triton
import triton.language as tl
from boltops.utility import arch_info, logger

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")

# HAS_DUMPED_SIGMOID_GATING_REC_KERNEL_METADATA = False
TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"
_DEFAULT_FUSED_SIGMOID_GATING_REC_CONFIG = {
    "BV": 32,
    "num_warps": 1,
}


@functools.lru_cache(maxsize=1)
def _load_fused_sigmoid_gating_recurrent_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "fused_sigmoid_gating_delta_rule_update_recurrent",
        f"fused_sigmoid_gating_delta_rule_update_recurrent-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"fused_sigmoid_gating_delta_rule_update_recurrent config not found at {path}, "
            f"using default {_DEFAULT_FUSED_SIGMOID_GATING_REC_CONFIG}."
        )
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("config", {}) if isinstance(payload, dict) else {}

@functools.lru_cache
def _get_fused_sigmoid_gating_recurrent_config(T: int, H: int, HV: int) -> dict:
    cfgs = _load_fused_sigmoid_gating_recurrent_configs()
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
                    f"fused_sigmoid_gating_recurrent config key '{key}' not found, "
                    f"using nearest-T config with T={nearest_t}: {cfg}."
                )
    if cfg is None:
        default_cfg = cfgs.get("default", _DEFAULT_FUSED_SIGMOID_GATING_REC_CONFIG)
        if TRITON_CONFIG_CHECK:
            logger.warning(
                f"fused_sigmoid_gating_recurrent config key '{key}' not found, "
                f"using default config: {default_cfg}."
            )
        cfg = default_cfg
    merged = dict(_DEFAULT_FUSED_SIGMOID_GATING_REC_CONFIG)
    merged.update(cfg)
    return merged


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_recurrent_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
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

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)
    # Invariant across timesteps.
    b_A = tl.exp(tl.load(p_A_log).to(tl.float32))
    if not IS_KDA:
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load k first; q is loaded later right before output to reduce register live range.
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)

        # Compute g with tighter live ranges for intermediates.
        x = b_a + b_dt_bias
        x_scaled = softplus_beta * x
        x = tl.where(
            x_scaled <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(x_scaled)),
            x,
        )
        b_g = -b_A * x

        # Apply L2 normalization to k early; q normalization is deferred until q is loaded.
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= tl.sigmoid(tl.load(p_b).to(tl.float32))

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Load q late to shorten q live range and lower peak register pressure.
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        if IS_KDA:
            p_a += HV * K
        else:
            p_a += HV

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update_recurrent(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    kernel_cfg: dict | None = None,
):
    global HAS_DUMPED_SIGMOID_GATING_REC_KERNEL_METADATA
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    cfg = kernel_cfg if kernel_cfg is not None else _get_fused_sigmoid_gating_recurrent_config(T, H, HV)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_warps = int(cfg["num_warps"])

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    compiled_kernel = fused_sigmoid_gating_delta_rule_update_recurrent_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=0 if cache_steps is None else cache_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        num_warps=num_warps,
        num_stages=1,
    )
    '''
    if not HAS_DUMPED_SIGMOID_GATING_REC_KERNEL_METADATA and compiled_kernel is not None:
        print("sigmoid gating recurrent kernel metadata")
        print(f"  grid: {grid}")
        print(f"  registers: {compiled_kernel.n_regs}")
        print(f"  spills: {compiled_kernel.n_spills}")
        print(f"  shared memory: {compiled_kernel.metadata.shared} bytes")
        HAS_DUMPED_SIGMOID_GATING_REC_KERNEL_METADATA = True
    '''
    o = o.squeeze(0)
    return o
