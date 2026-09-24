# SPDX-License-Identifier: MIT

from __future__ import annotations

import functools
import json
import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from boltops.utility import arch_info, logger

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")


# HAS_DUMPED_PACKED_DECODE_KERNEL_METADATA = False
TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"

_DEFAULT_FUSED_RECURRENT_PACKED_DECODE_CONFIG = {
    "BV": 32,
    "num_warps": 1,
    "num_stages": 1,
}


@functools.lru_cache(maxsize=1)
def _load_fused_recurrent_packed_decode_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "fused_recurrent_gated_delta_rule_packed_decode",
        f"fused_recurrent_gated_delta_rule_packed_decode-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"fused_recurrent_gated_delta_rule_packed_decode config not found at {path}, "
            f"using default {_DEFAULT_FUSED_RECURRENT_PACKED_DECODE_CONFIG}."
        )
        return {}
    with open(path) as f:
        payload = json.load(f)
    return payload.get("config", {}) if isinstance(payload, dict) else {}


@functools.lru_cache
def _get_fused_recurrent_packed_decode_config(B: int, H: int, HV: int) -> dict:
    cfgs = _load_fused_recurrent_packed_decode_configs()
    key = f"B={B},H={H},HV={HV}"
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
            if parts.get("H") == H and parts.get("HV") == HV and "B" in parts:
                candidates.append((abs(parts["B"] - B), parts["B"], v))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, nearest_b, cfg = candidates[0]
            if TRITON_CONFIG_CHECK:
                logger.warning(
                    f"fused_recurrent_packed_decode config key '{key}' not found, "
                    f"using nearest-B config with B={nearest_b}: {cfg}."
                )
    if cfg is None:
        default_cfg = cfgs.get("default", _DEFAULT_FUSED_RECURRENT_PACKED_DECODE_CONFIG)
        if TRITON_CONFIG_CHECK:
            logger.warning(
                f"fused_recurrent_packed_decode config key '{key}' not found, "
                f"using default config: {default_cfg}."
            )
        cfg = default_cfg
    merged = dict(_DEFAULT_FUSED_RECURRENT_PACKED_DECODE_CONFIG)
    merged.update(cfg)
    return merged


@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx <= NULL_STATE_INDEX:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    # [BV, BK]
    b_h = tl.load(p_h0, mask=(mask_v[:, None] & mask_k[None, :]), other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        k_norm_inv = tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_k = b_k * k_norm_inv

    x = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    x += tl.load(dt_bias + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    beta_val = tl.sigmoid(tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32))

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]

    q_off = i_h * K + o_k
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        q_norm_inv = tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * q_norm_inv
    b_o = tl.sum(b_h * b_q[None, :], 1)
    b_o = b_o * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=(mask_v[:, None] & mask_k[None, :]))


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    null_state_index: int = -1,
    kernel_cfg: dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-token GDN decode from packed post-conv QKV.

    ``null_state_index`` selects the state-index ABI so both frameworks can
    call the same entry: ``-1`` (the default, SGLang) treats only negative
    indices as invalid -- SGLang reserves slot 0 as the dummy write target
    for padded requests and never allocates it to a live request; ``0``
    (vLLM NULL_BLOCK_ID) treats indices ``<= 0`` as invalid and valid slots
    are 1-based.
    """
    global HAS_DUMPED_PACKED_DECODE_KERNEL_METADATA

    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be 2D, got ndim={mixed_qkv.ndim}.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"`a` and `b` must be 2D, got a.ndim={a.ndim}, b.ndim={b.ndim}.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log` and `dt_bias` must be 1D.")
    if ssm_state_indices.ndim != 1:
        raise ValueError("`ssm_state_indices` must be 1D.")
    if initial_state.ndim != 4:
        raise ValueError(f"`initial_state` must be 4D, got ndim={initial_state.ndim}.")

    dev = mixed_qkv.device
    if any(t.device != dev for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)):
        raise ValueError("All tensors must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B or ssm_state_indices.shape[0] != B:
        raise ValueError("Batch dimensions of mixed_qkv/a/b/ssm_state_indices must match.")
    if null_state_index not in (-1, 0):
        raise ValueError("null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI).")

    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError("`a` and `b` second dim must match HV from initial_state.")
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError("`A_log` and `dt_bias` numel must equal HV.")
    if out.shape != (B, 1, HV, V):
        raise ValueError(f"`out` must have shape {(B, 1, HV, V)}, got {tuple(out.shape)}.")

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError("Invalid mixed_qkv layout for packed decode.")
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError("Inferred q_dim must be divisible by K.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"Invalid inferred heads: H={H}, HV={HV}.")

    BK = triton.next_power_of_2(K)
    cfg = kernel_cfg if kernel_cfg is not None else _get_fused_recurrent_packed_decode_config(B, H, HV)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    launch_kwargs = dict(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_STATE_INDEX=null_state_index,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    compiled_kernel = fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](**launch_kwargs)

    '''
    if not HAS_DUMPED_PACKED_DECODE_KERNEL_METADATA and compiled_kernel is not None:
        print("packed decode kernel metadata")
        print(f"  grid: {grid}")
        print(f"  registers: {compiled_kernel.n_regs}")
        print(f"  spills: {compiled_kernel.n_spills}")
        print(f"  shared memory: {compiled_kernel.metadata.shared} bytes")
        HAS_DUMPED_PACKED_DECODE_KERNEL_METADATA = True
    '''
    return out, initial_state
