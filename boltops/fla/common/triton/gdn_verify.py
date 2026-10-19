"""FP32 chain/tree verify kernel.

The state tile is [BV, K], so loads, snapshot stores and K reductions use
the physical VK layout directly. All recurrence arithmetic stays FP32.
One program owns one value head, avoiding the register growth of head grouping.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from boltops.utility import arch_info


@functools.lru_cache(maxsize=1)
def _load_configs() -> dict:
    path = (
        Path(__file__).parent / "configs" / "fused_sigmoid_gating_recurrent"
        / f"fused_sigmoid_gating_recurrent-{arch_info.get_arch()}.json"
    )
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("verify_config", {})


def get_config(
    *, q, k, v, a, b, dt_bias, initial_state_source,
    initial_state_indices, cu_seqlens, intermediate_states_buffer,
    intermediate_state_indices, is_kda, lower_bound,
    disable_state_update, use_qk_l2norm_in_kernel, cache_ring,
    retrieve_parent_token, kernel_cfg,
) -> dict | None:
    """Select a qualified shape family; explicit kernel_cfg forces fallback."""
    if (
        kernel_cfg is not None or is_kda or lower_bound is not None or cache_ring
        or not disable_state_update or not use_qk_l2norm_in_kernel
        or cu_seqlens is None or initial_state_source is None
        or initial_state_indices is None or intermediate_states_buffer is None
        or intermediate_state_indices is None
    ):
        return None
    if cu_seqlens.ndim != 1 or initial_state_source.ndim != 4:
        return None
    n = cu_seqlens.numel() - 1
    tokens = q.shape[1] if q.ndim == 4 else -1
    h, k_dim = q.shape[-2:] if q.ndim == 4 else (-1, -1)
    hv, v_dim = v.shape[-2:] if v.ndim == 4 else (-1, -1)
    if (
        n <= 0 or tokens <= 0 or q.ndim != 4 or q.shape[0] != 1
        or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]
        or h <= 0 or hv <= 0 or hv % h != 0
        or k_dim not in (16, 32, 64, 128) or v_dim not in (16, 32, 64, 128)
        or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
    ):
        return None
    if (
        q.stride(-1) != 1 or q.stride(-2) != k_dim
        or k.stride(-1) != 1 or k.stride(-2) != k_dim
        or v.stride(-1) != 1 or v.stride(-2) != v_dim
    ):
        return None
    if a.shape[-1] != hv or b.shape[-1] != hv or a.stride(-1) != 1 or b.stride(-1) != 1:
        return None
    if dt_bias.ndim != 1 or dt_bias.shape[0] != hv or dt_bias.stride(0) != 1:
        return None
    for tensor in (initial_state_source, intermediate_states_buffer):
        if (
            tensor.dtype != torch.float32 or tensor.shape[-3:] != (hv, v_dim, k_dim)
            or tensor.stride(-1) != 1 or tensor.stride(-2) != k_dim
            or tensor.stride(-3) != v_dim * k_dim
        ):
            return None
    if (
        intermediate_states_buffer.ndim != 5
        or intermediate_states_buffer.stride(-4) % (hv * v_dim * k_dim) != 0
    ):
        return None
    tree = int(retrieve_parent_token is not None)
    tokens_per_request = tokens // n
    token_bucket = 4 if tokens_per_request <= 4 else 8
    configs = _load_configs()
    # The short-chain large-batch tail is bandwidth bound; measured BV=16
    # launches lose to the established generic kernel there.
    if (
        tree == 0 and token_bucket == 4
        and n >= configs.get("fallback_chain_T4_from_N", float("inf"))
    ):
        return None
    # Batch size never selects a separate launch configuration. Keep one
    # architecture-qualified config per token-count bucket and topology.
    config = configs.get(f"default_{'tree' if tree else 'chain'}_T{token_bucket}")
    if not isinstance(config, dict):
        return None
    if (
        config.get("BV") not in (1, 2, 4, 8, 16, 32)
        or config["BV"] > v_dim
        or config.get("num_warps") not in (1, 2, 4)
        or config.get("num_stages") != 1
        or type(config.get("reuse_parent", False)) is not bool
    ):
        return None
    return config


@triton.jit
def gdn_verify_kernel(
    Q, K, V, A, B, ALog, Bias, State, StateIndices, CuSeq,
    Cache, CacheIndices, Parents, Out,
    stride_q: tl.constexpr, stride_k: tl.constexpr, stride_v: tl.constexpr,
    stride_a: tl.constexpr, stride_b: tl.constexpr,
    stride_state: tl.constexpr, stride_cache: tl.constexpr,
    stride_parent_n: tl.constexpr, stride_parent_t: tl.constexpr,
    stride_out: tl.constexpr, state_head_stride: tl.constexpr,
    cache_head_stride: tl.constexpr, cache_token_stride: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr,
    SCALE: tl.constexpr, SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr, NULL_INDEX: tl.constexpr,
    HAS_TREE: tl.constexpr, BV: tl.constexpr, REUSE_PARENT: tl.constexpr = False,
):
    iv, nh = tl.program_id(0), tl.program_id(1)
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    kk = tl.arange(0, K_DIM)
    vv = iv * BV + tl.arange(0, BV)
    offsets = vv[:, None] * K_DIM + kk[None, :]
    start = tl.load(CuSeq + n).to(tl.int64)
    end = tl.load(CuSeq + n + 1).to(tl.int64)
    slot = tl.load(StateIndices + n).to(tl.int64)
    cache_slot = tl.load(CacheIndices + n).to(tl.int64)
    state = tl.load(
        State + slot * stride_state + hv * state_head_stride + offsets,
        mask=slot > NULL_INDEX, other=0,
    )
    cache_base = Cache + cache_slot * stride_cache + hv * cache_head_stride + offsets
    log_a = tl.exp(tl.load(ALog + hv).to(tl.float32))
    bias = tl.load(Bias + hv).to(tl.float32)
    if HAS_TREE and REUSE_PARENT:
        # Keep the last parent's base, not the last token's updated state.
        # Sibling tokens often read the same base snapshot consecutively.
        parent_state = state
        previous_parent = -1
    for t in range(end - start):
        token = start + t
        if HAS_TREE:
            if t > 0 and cache_slot >= 0:
                parent = tl.load(Parents + n * stride_parent_n + t * stride_parent_t)
                if REUSE_PARENT:
                    if parent == t - 1:
                        parent_state = state
                    elif parent != previous_parent:
                        parent_state = tl.load(cache_base + parent.to(tl.int64) * cache_token_stride)
                    state = parent_state
                    previous_parent = parent
                elif parent != t - 1:
                    state = tl.load(cache_base + parent.to(tl.int64) * cache_token_stride)
        key = tl.load(K + token * stride_k + h * K_DIM + kk).to(tl.float32)
        key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
        value = tl.load(V + token * stride_v + hv * V_DIM + vv).to(tl.float32)
        a = tl.load(A + token * stride_a + hv).to(tl.float32) + bias
        x = a * SOFTPLUS_BETA
        softplus = tl.where(
            x <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(x)) / SOFTPLUS_BETA, a,
        )
        decay = tl.exp(-log_a * softplus)
        beta = tl.sigmoid(tl.load(B + token * stride_b + hv).to(tl.float32))
        state *= decay
        value = (value - tl.sum(state * key[None, :], 1)) * beta
        state += value[:, None] * key[None, :]
        query = tl.load(Q + token * stride_q + h * K_DIM + kk).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6)
        output = tl.sum(state * query[None, :], 1) * SCALE
        tl.store(Out + token * stride_out + hv * V_DIM + vv, output)
        if cache_slot >= 0:
            tl.store(cache_base + t * cache_token_stride, state)
