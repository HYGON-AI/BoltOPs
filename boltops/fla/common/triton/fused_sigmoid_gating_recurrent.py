# SPDX-License-Identifier: MIT

"""Fused sigmoid-gating + recurrent delta-rule update (decode and verify).

KDA+GDN shared recurrent decode+verify kernel. The full-parameter entry
``fused_sigmoid_gating_delta_rule_update`` keeps the SGLang
``fla/fused_sigmoid_gating_recurrent`` name: it supports decode and verify
with tunable softplus params and ``disable_state_update``. Framework
adapters own named verify entries and their presets.

The supported input contract is documented in ``docs/ops/fla/gdn.md``
(section 3.3); framework adapters own fallback decisions.

Implements the SGLang ``fused_sigmoid_gating_delta_rule_update`` logical
interface for Kimi K3 / GDN speculative verify:

- KDA per-K gate (including the ``lower_bound`` safe gate) and GDN scalar gate;
- varlen packed ``[1, total_tokens, H, K]`` / ``[1, total_tokens, HV, V]``;
- committed state pool with int64 slot indices and envelope slot stride
  (``initial_state_source.stride(0)``);
- per-draft-token intermediate state cache with adaptive row stride
  (``intermediate_states_buffer.stride(0)``);
- chain and tree verify (``retrieve_parent_token``);
- ReplaySSM fused ring write (``cache_ring``).

See ``docs/ops/fla/gdn.md`` (section 3.3) for the full operator contract.
"""

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

TRITON_CONFIG_CHECK = os.environ.get("TRITON_CONFIG_CHECK", "0") == "1"
_DEFAULT_FUSED_SIGMOID_GATING_REC_CONFIG = {
    "BV": 32,
    "num_warps": 1,
    "num_stages": 1,
}


@functools.lru_cache(maxsize=1)
def _load_fused_sigmoid_gating_recurrent_configs() -> dict:
    device_name = arch_info.get_arch()
    path = os.path.join(
        _CONFIGS_PATH,
        "fused_sigmoid_gating_recurrent",
        f"fused_sigmoid_gating_recurrent-{device_name}.json",
    )
    if not os.path.exists(path):
        logger.warning(
            f"fused_sigmoid_gating_recurrent config not found at {path}, "
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
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for speculative verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off. The gate ring layout follows IS_KDA: per-K vector
    # [.., MAX_CACHE_LEN, K] for KDA, per-head scalar [.., MAX_CACHE_LEN]
    # for GDN.
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
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
    USE_LOWER_BOUND: tl.constexpr,
    # Optional flags for speculative verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    CACHE_RING: tl.constexpr = False,
    NULL_STATE_INDEX: tl.constexpr = -1,
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

    # Gating computation pointers. ``stride_a`` is the real per-token pitch of
    # the gate input (a.stride(1) for [B, T, ...], a.stride(0) for [T, ...]);
    # for KDA packed [1, T, HV, K] it is HV*K, never a constant HV.
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    slot_idx = -1
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state
        # pool may be an envelope-strided view (page-major / unified memory),
        # where the per-slot pitch spans more than HV*K*V. int64: envelope
        # pitches overflow an int32 index product for large slot counts.
        slot_idx = tl.load(h0_indices + i_n).to(tl.int64)
        if slot_idx > NULL_STATE_INDEX:
            p_h0 = (
                h0_source
                + slot_idx * stride_h0_source
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

    # Prepare intermediate state cache index if enabled. int64:
    # `cache_idx * cache_steps * HV * K * V` can exceed int32 for large
    # slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)
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

        # Load k first (still pre-l2norm here; the ring write below needs it).
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)

        # Compute sigmoid gating
        if IS_KDA:
            b_x = (
                tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
                + tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
            )
        else:
            b_x = tl.load(p_a).to(tl.float32) + b_dt_bias

        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias)).
            # Replacing this with the softplus gate changes model semantics.
            b_g = lower_bound * tl.sigmoid(b_A * b_x)
        else:
            x_scaled = softplus_beta * b_x
            # Apply softplus with numerical stability
            b_x = tl.where(
                x_scaled <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(x_scaled)),
                b_x,
            )
            b_g = -b_A * b_x

        b_beta = 1.0 / (1.0 + tl.exp(-tl.load(p_b).to(tl.float32)))

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. b_k is still
        # pre-l2norm and b_v still pre-delta, so the fold's replay is
        # bit-identical to the update below. rawk uses the k-head i_h (shared
        # across a GQA group); rawv/g/beta use the v-head i_hv.
        # step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the ring;
        # the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if slot_idx > NULL_STATE_INDEX and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + slot_idx * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + slot_idx * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # The gate ring row is K wide for KDA (per-K vector) and
                    # a scalar for GDN; the two layouts are not interchangeable.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + slot_idx * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + slot_idx * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + slot_idx * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        if not CACHE_RING:
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Load q late to shorten q live range and lower peak register pressure.
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
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
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            if slot_idx > NULL_STATE_INDEX:
                p_h0 = (
                    h0_source
                    + slot_idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
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
    lower_bound: Optional[float] = None,
    # Optional parameters for speculative verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft
    # step stores pre-norm k / raw v / gate / beta into these per-slot rings
    # (per-layer views), replacing the eager per-step state snapshots.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    null_state_index: int = -1,
    kernel_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and speculative verify modes:
    - decode: standard single-step update with state write-back
    - speculative verify: multi-step with intermediate state caching, optional tree attention,
                     optional state update disable, and optional ReplaySSM ring write

    ``null_state_index`` selects the state-index ABI so both frameworks can
    call the same entry: ``-1`` (the default, SGLang) treats only negative
    ``initial_state_indices`` values as invalid -- SGLang reserves slot 0 as
    the dummy write target for padded requests and never allocates it to a
    live request; ``0`` (vLLM NULL_BLOCK_ID) treats indices ``<= 0`` as
    invalid and valid slots are 1-based.

    Returns a single attention output tensor ``[1, total_tokens, HV, V]``
    (the SGLang verify dispatcher does not consume a final-state tuple).
    """
    global HAS_DUMPED_SIGMOID_GATING_REC_KERNEL_METADATA
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis
    # stride. For 2D a ([T, ...]) this is stride(0); for 3D/4D a
    # ([B, T, ...]) this is stride(1). For KDA packed [1, T, HV, K] it is
    # HV*K -- the real pitch, never a constant HV.
    stride_a = a.stride()[1] if a.ndim == 4 else a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    cfg = kernel_cfg if kernel_cfg is not None else _get_fused_sigmoid_gating_recurrent_config(T, H, HV)
    BV = min(triton.next_power_of_2(V), int(cfg["BV"]))
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    if NK != 1:
        raise ValueError(f"NK > 1 is not supported yet (K={K}).")
    num_warps = int(cfg["num_warps"])
    num_stages = int(cfg["num_stages"])

    if scale is None:
        scale = k.shape[-1] ** -0.5
    elif scale <= 0:
        raise ValueError("scale must be positive.")

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"q.shape[0] must be 1 when using cu_seqlens, got {q.shape[0]}."
        )
    if null_state_index not in (-1, 0):
        raise ValueError("null_state_index must be 0 (vLLM ABI) or -1 (SGLang ABI).")
    if retrieve_parent_token is not None:
        if retrieve_parent_token.dim() != 2:
            raise ValueError(
                "retrieve_parent_token must be 2D [N, T], "
                f"got ndim={retrieve_parent_token.dim()}."
            )
        if intermediate_states_buffer is None:
            raise ValueError(
                "retrieve_parent_token (tree verify) requires "
                "intermediate_states_buffer."
            )
    if intermediate_states_buffer is not None:
        if intermediate_state_indices is None:
            raise ValueError(
                "intermediate_state_indices is required with "
                "intermediate_states_buffer."
            )
        if retrieve_parent_token is not None and retrieve_parent_token.shape[0] != N:
            raise ValueError(
                f"retrieve_parent_token row count {retrieve_parent_token.shape[0]} "
                f"must match the request count {N}."
            )
    if cache_ring:
        g_ndim = 4 if is_kda else 3
        if (
            replayssm_rawv is None
            or replayssm_rawk is None
            or replayssm_g is None
            or replayssm_beta is None
        ):
            raise ValueError(
                "cache_ring requires replayssm_rawv/rawk/g/beta per-layer "
                "ring views."
            )
        if not (
            replayssm_rawv.dim() == 4
            and replayssm_rawk.dim() == 4
            and replayssm_g.dim() == g_ndim
            and replayssm_beta.dim() == 3
        ):
            raise ValueError(
                f"cache_ring expects per-layer ring views (rawv/rawk 4D, g "
                f"{g_ndim}D, beta 3D); got rawv={replayssm_rawv.dim()}, "
                f"rawk={replayssm_rawk.dim()}, g={replayssm_g.dim()}, "
                f"beta={replayssm_beta.dim()}."
            )
        if initial_state_indices is None:
            raise ValueError(
                "cache_ring requires initial_state_indices (ring slot index)."
            )

    # Adaptive spec changes the runtime draft count without changing the
    # allocated per-request pitch, which is preserved in stride(0).
    if intermediate_states_buffer is not None:
        row_elems = HV * K * V
        if intermediate_states_buffer.stride(0) % row_elems != 0:
            raise ValueError(
                "intermediate_states_buffer.stride(0) must be a multiple of "
                f"HV*K*V ({row_elems}), got {intermediate_states_buffer.stride(0)}."
            )
        cache_stride_steps = intermediate_states_buffer.stride(0) // row_elems
    elif cache_steps is not None and cache_steps > 0:
        cache_stride_steps = cache_steps
    else:
        cache_stride_steps = 0

    # ring strides (per-slot rings are contiguous [num_slots, heads, L, dim];
    # the kernel offsets within a slot with MAX_CACHE_LEN and the dim extents).
    if cache_ring:
        max_cache_len = replayssm_rawv.shape[-2]
        stride_rawv_slot = replayssm_rawv.stride(0)
        stride_rawk_slot = replayssm_rawk.stride(0)
        stride_g_slot = replayssm_g.stride(0)
        stride_beta_slot = replayssm_beta.stride(0)
    else:
        max_cache_len = 0
        stride_rawv_slot = stride_rawk_slot = stride_g_slot = stride_beta_slot = 0

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
    compiled_kernel = fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != HV*K*V; contiguous pools pass exactly HV*K*V.
        stride_h0_source=(
            initial_state_source.stride(0) if initial_state_source is not None else 0
        ),
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        stride_rawv_slot=stride_rawv_slot,
        stride_rawk_slot=stride_rawk_slot,
        stride_g_slot=stride_g_slot,
        stride_beta_slot=stride_beta_slot,
        MAX_CACHE_LEN=max_cache_len,
        scale=scale,
        T=T,
        stride_a=stride_a,
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
        USE_LOWER_BOUND=lower_bound is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        CACHE_RING=cache_ring,
        NULL_STATE_INDEX=null_state_index,
        num_warps=num_warps,
        num_stages=num_stages,
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
