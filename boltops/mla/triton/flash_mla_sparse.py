# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""Triton implementation of the FlashMLA sparse forward interface."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

from boltops.utility.triton_capability import get_triton_capabilities


_CONFIG_ROOT = Path(__file__).with_name("configs")


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _triton_version_tag() -> str:
    match = re.match(r"(\d+)\.(\d+)", triton.__version__)
    return f"triton_{match.group(1)}_{match.group(2)}" if match else "triton_unknown"


def _device_identity(device: torch.device) -> tuple[str, str]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    arch_value = getattr(props, "gcnArchName", "") or getattr(
        props, "arch_name", ""
    )
    arch_match = re.search(r"gfx[0-9a-f]+", str(arch_value).lower())
    arch = arch_match.group(0) if arch_match else "unknown"
    return arch, str(props.name)


@functools.lru_cache(maxsize=None)
def _load_device_configs(version_tag: str, arch: str, device_name: str) -> dict:
    filename = f"flash_mla_sparse_{arch}_{_slug(device_name)}.json"
    path = _CONFIG_ROOT / version_tag / filename
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_match_score(match: dict, shape: dict) -> Optional[tuple[int, int, float]]:
    """Return match specificity, or ``None`` when a shape does not match."""
    exact = 0
    bounds = 0
    widths = {}
    for key, expected in match.items():
        if key.endswith("_min") or key.endswith("_max"):
            field, kind = key.rsplit("_", 1)
            actual = shape.get(field)
            if actual is None:
                return None
            if kind == "min" and actual < expected:
                return None
            if kind == "max" and actual > expected:
                return None
            widths.setdefault(field, {})[kind] = expected
            bounds += 1
        elif shape.get(key) != expected:
            return None
        else:
            exact += 1
    total_width = 0.0
    for limits in widths.values():
        if "min" in limits and "max" in limits:
            total_width += limits["max"] - limits["min"]
        else:
            total_width += float("inf")
    return exact, bounds, -total_width


def _get_tuned_config(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
) -> dict:
    arch, device_name = _device_identity(q.device)
    data = _load_device_configs(_triton_version_tag(), arch, device_name)
    shape = {
        "s_q": q.shape[0],
        "s_kv": kv.shape[0],
        "h_q": q.shape[1],
        "h_kv": kv.shape[1],
        "topk": indices.shape[2],
        "d_qk": q.shape[2],
        "d_v": d_v,
        "dtype": str(q.dtype).replace("torch.", ""),
    }
    selected = None
    selected_score = None
    for entry in data.get("configs", ()):
        match = entry.get("match", {})
        if not isinstance(match, dict):
            continue
        score = _config_match_score(match, shape)
        if score is not None and (selected_score is None or score > selected_score):
            selected, selected_score = entry, score
    if selected is not None:
        config = selected.get("config", {})
        return dict(config) if isinstance(config, dict) else {}
    return {}


@triton.jit
def _flash_mla_sparse_fwd_kernel(
    Q,
    KV,
    Indices,
    TopkLength,
    Out,
    MaxLogits,
    Lse,
    softmax_scale,
    stride_qs,
    stride_qh,
    stride_kvs,
    stride_kvh,
    stride_is,
    stride_ih,
    stride_os,
    stride_oh,
    S_KV,
    TOPK: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_SPLITS: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """One program handles a query token and a group of query heads."""
    q_idx = tl.program_id(0)
    head_block = tl.program_id(1)

    kv_group_size: tl.constexpr = H_Q // H_KV
    heads_per_program: tl.constexpr = (
        BLOCK_H if BLOCK_H < kv_group_size else kv_group_size
    )
    heads = head_block * heads_per_program + tl.arange(0, BLOCK_H)
    head_mask = (heads < (head_block + 1) * heads_per_program) & (heads < H_Q)
    kv_head = head_block // tl.cdiv(kv_group_size, BLOCK_H)

    if HAS_TOPK_LENGTH:
        valid_topk = tl.load(TopkLength + q_idx)
    else:
        valid_topk = TOPK
    valid_topk = tl.maximum(0, tl.minimum(valid_topk, TOPK))

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_v = tl.arange(0, BLOCK_DV)
    mask_v = offs_v < Lv

    if BLOCK_DMODEL_SPLITS == 1:
        mask_d = offs_d < Lk
        q = tl.load(
            Q + q_idx * stride_qs + heads[:, None] * stride_qh + offs_d[None, :],
            mask=head_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
    if BLOCK_DPE > 0:
        offs_pe = BLOCK_DMODEL * BLOCK_DMODEL_SPLITS + tl.arange(0, BLOCK_DPE)
        mask_pe = offs_pe < Lk
        q_pe = tl.load(
            Q + q_idx * stride_qs + heads[:, None] * stride_qh + offs_pe[None, :],
            mask=head_mask[:, None] & mask_pe[None, :],
            other=0.0,
        )

    row_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    row_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        prefix_mask = offs_n < valid_topk
        token = tl.load(
            Indices + q_idx * stride_is + kv_head * stride_ih + offs_n,
            mask=prefix_mask,
            other=-1,
        )
        valid = prefix_mask & (token >= 0) & (token < S_KV)
        safe_token = tl.where(valid, token, 0)

        if BLOCK_DMODEL_SPLITS == 1:
            k = tl.load(
                KV
                + safe_token[None, :] * stride_kvs
                + kv_head * stride_kvh
                + offs_d[:, None],
                mask=valid[None, :] & mask_d[:, None],
                other=0.0,
            )
            scores = tl.dot(q, k.to(q.dtype))
        else:
            scores = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            for start_d in range(
                0, BLOCK_DMODEL_SPLITS * BLOCK_DMODEL, BLOCK_DMODEL
            ):
                split_d = start_d + tl.arange(0, BLOCK_DMODEL)
                split_mask = split_d < Lk
                q_part = tl.load(
                    Q
                    + q_idx * stride_qs
                    + heads[:, None] * stride_qh
                    + split_d[None, :],
                    mask=head_mask[:, None] & split_mask[None, :],
                    other=0.0,
                )
                k_part = tl.load(
                    KV
                    + safe_token[None, :] * stride_kvs
                    + kv_head * stride_kvh
                    + split_d[:, None],
                    mask=valid[None, :] & split_mask[:, None],
                    other=0.0,
                )
                scores += tl.dot(q_part, k_part.to(q_part.dtype))
        if BLOCK_DPE > 0:
            k_pe = tl.load(
                KV
                + safe_token[None, :] * stride_kvs
                + kv_head * stride_kvh
                + offs_pe[:, None],
                mask=valid[None, :] & mask_pe[:, None],
                other=0.0,
            )
            scores += tl.dot(q_pe, k_pe.to(q_pe.dtype))

        scores *= softmax_scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        has_old = row_sum > 0
        has_tile = tile_max > -float("inf")
        has_row = has_old | has_tile
        new_max = tl.maximum(row_max, tile_max)
        safe_max = tl.where(has_row, new_max, 0.0)
        old_scale = tl.where(has_old, tl.exp(row_max - safe_max), 0.0)
        probs = tl.exp(scores - safe_max[:, None])
        probs = tl.where(scores > -float("inf"), probs, 0.0)

        values = tl.load(
            KV
            + safe_token[:, None] * stride_kvs
            + kv_head * stride_kvh
            + offs_v[None, :],
            mask=valid[:, None] & mask_v[None, :],
            other=0.0,
        )
        acc = acc * old_scale[:, None] + tl.dot(probs.to(values.dtype), values)
        row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
        row_max = tl.where(has_row, new_max, row_max)

    safe_sum = tl.where(row_sum > 0, row_sum, 1.0)
    output = acc / safe_sum[:, None]
    lse = tl.where(row_sum > 0, row_max + tl.log(safe_sum), -float("inf"))
    tl.store(
        Out + q_idx * stride_os + heads[:, None] * stride_oh + offs_v[None, :],
        output,
        mask=head_mask[:, None] & mask_v[None, :],
    )
    tl.store(MaxLogits + q_idx * H_Q + heads, row_max, mask=head_mask)
    tl.store(Lse + q_idx * H_Q + heads, lse, mask=head_mask)


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    topk_length: Optional[torch.Tensor],
) -> None:
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("q, kv, and indices must all be 3-dimensional")
    if q.device != kv.device or q.device != indices.device:
        raise ValueError("q, kv, and indices must be on the same device")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("q must have dtype torch.bfloat16 or torch.float16")
    if kv.dtype != q.dtype:
        raise TypeError("kv must have the same dtype as q")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have dtype torch.int32 or torch.int64")
    if q.shape[0] != indices.shape[0] or kv.shape[1] != indices.shape[1]:
        raise ValueError("indices must have shape [s_q, h_kv, topk]")
    if q.shape[2] != kv.shape[2]:
        raise ValueError("q and kv must have the same d_qk")
    if kv.shape[1] == 0 or q.shape[1] % kv.shape[1] != 0:
        raise ValueError("h_q must be divisible by h_kv")
    if not 0 < d_v <= kv.shape[2]:
        raise ValueError("d_v must be positive and no larger than d_qk")
    if indices.shape[2] == 0:
        raise ValueError("topk must be positive")
    if topk_length is not None:
        if topk_length.device != q.device:
            raise ValueError("topk_length must be on the same device as q")
        if topk_length.ndim != 1 or topk_length.shape[0] != q.shape[0]:
            raise ValueError("topk_length must have shape [s_q]")
        if topk_length.dtype not in (torch.int32, torch.int64):
            raise TypeError("topk_length must have dtype torch.int32 or torch.int64")


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sparse MLA attention using token indices.

    ``q`` is ``[s_q, h_q, d_qk]``, ``kv`` is ``[s_kv, h_kv, d_qk]``, and
    ``indices`` is ``[s_q, h_kv, topk]``. Values are read from the first
    ``d_v`` channels of ``kv``. Invalid indices are ignored. The results are
    output, maximum logits, and log-sum-exp with shapes ``[s_q, h_q, d_v]``,
    ``[s_q, h_q]``, and ``[s_q, h_q]`` respectively.
    """
    _validate_inputs(q, kv, indices, d_v, topk_length)
    s_q, h_q, d_qk = q.shape
    h_kv = kv.shape[1]

    output = torch.empty((s_q, h_q, d_v), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((s_q, h_q), device=q.device, dtype=torch.float32)
    lse = torch.empty_like(max_logits)

    tuned = _get_tuned_config(q, kv, indices, d_v) if config is None else dict(config)
    # Splitting the 512-dimensional content part keeps register use practical.
    # The final 64 dimensions of the common 576 layout are RoPE.
    if d_qk == 576:
        block_dmodel = int(tuned.get("BLOCK_DMODEL", 128))
        if block_dmodel not in (64, 128, 256, 512) or 512 % block_dmodel:
            raise ValueError("BLOCK_DMODEL must be one of 64, 128, 256, or 512")
        dmodel_splits, block_dpe = 512 // block_dmodel, 64
    else:
        block_dmodel, dmodel_splits, block_dpe = triton.next_power_of_2(d_qk), 1, 0
    block_dv = triton.next_power_of_2(d_v)
    default_block_n = 16 if torch.version.hip is not None and d_qk >= 576 else 32
    block_n = int(tuned.get("BLOCK_N", default_block_n))
    block_h = int(tuned.get("BLOCK_H", 16))
    num_warps = int(tuned.get("num_warps", 4))
    num_stages = int(
        tuned.get("num_stages", 1 if torch.version.hip is not None else 2)
    )
    heads_per_program = min(block_h, h_q // h_kv)
    grid = (s_q, triton.cdiv(h_q, heads_per_program))

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    topk_arg = topk_length.contiguous() if topk_length is not None else indices
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if torch.version.hip is not None:
        capabilities = get_triton_capabilities()
        compiler_options = (
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
            "amdgpu_enable_max_ilp_scheduling_strategy",
            "misched_regpressure",
            "optimize_epilogue",
            "use_block_pingpong",
        )
        for name in compiler_options:
            if name in tuned and capabilities.supports(name):
                launch_options[name] = int(tuned[name])
    _flash_mla_sparse_fwd_kernel[grid](
        q,
        kv,
        indices,
        topk_arg,
        output,
        max_logits,
        lse,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        kv.shape[0],
        TOPK=indices.shape[2],
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DMODEL_SPLITS=dmodel_splits,
        BLOCK_DPE=block_dpe,
        BLOCK_DV=block_dv,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        HAS_TOPK_LENGTH=topk_length is not None,
        H_Q=h_q,
        H_KV=h_kv,
        Lk=d_qk,
        Lv=d_v,
        **launch_options,
    )

    if attn_sink is not None:
        if attn_sink.numel() != h_q:
            raise ValueError("attn_sink must contain one value per query head")
        sink = attn_sink.to(device=q.device, dtype=torch.float32).view(1, h_q)
        output *= torch.sigmoid(lse - sink).unsqueeze(-1).to(output.dtype)
    return output, max_logits, lse


__all__ = ["flash_mla_sparse_fwd"]
