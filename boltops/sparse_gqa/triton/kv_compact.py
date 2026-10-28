# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/yht183/sglang-das/blob/4cc62ab1d0c7a63c770e2a275d42b67ae20c345c/python/sglang/srt/layers/attention/qsa/sparse_attn.py
"""Pack selected rows from the HCU-FA paged KV layout."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .config import load_config


_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


def is_fp8_kv_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def _unit_scale(scale: Optional[float]) -> float:
    return 1.0 if scale is None else float(scale)


@triton.jit
def _compact_kv_hcu_fa(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    k_sp: tl.constexpr,
    k_sh: tl.constexpr,
    k_so: tl.constexpr,
    k_sd: tl.constexpr,
    v_sp: tl.constexpr,
    v_sh: tl.constexpr,
    v_so: tl.constexpr,
    v_sd: tl.constexpr,
    ok0: tl.constexpr,
    ok1: tl.constexpr,
    ok2: tl.constexpr,
    ov0: tl.constexpr,
    ov1: tl.constexpr,
    ov2: tl.constexpr,
    k_scale,
    v_scale,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DEQUANTIZE_FP8: tl.constexpr,
    GROUPED_V: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
    PACKED_V: tl.constexpr,
):
    """Gather selected tokens from HCU FA paged KV into packed NHD rows.

    K is (pages, heads, page, dim); V is (pages, heads, dim, page). Physical
    slots are page * page_size + offset in the HCU-FA cache layout. This does
    not change the pool layout.
    """
    if TOKEN_MAJOR:
        block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(
        indices + batch * idx_stride + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    page = slots // page_size
    off = slots % page_size
    k_ptrs = (
        k
        + page[:, None] * k_sp
        + head * k_sh
        + off[:, None] * k_so
        + dims[None, :] * k_sd
    )
    v_ptrs = (
        v
        + page[:, None] * v_sp
        + head * v_sh
        + off[:, None] * v_so
        + dims[None, :] * v_sd
    )
    dst = (pack_start + cols)[:, None]
    mask = valid[:, None] & (dims[None, :] < dim)
    k_dst = out_k + dst * ok0 + head * ok1 + dims[None, :] * ok2
    v_dst = out_v + dst * ov0 + head * ov1 + dims[None, :] * ov2
    k_values = tl.load(k_ptrs, mask=mask, other=0.0)
    # Validate four-slot locality on device before using coalesced V loads.
    if GROUPED_V and BLOCK_TOPK >= 4 and page_size % 4 == 0 and v_so == 1:
        group_slots = tl.reshape(slots, (BLOCK_TOPK // 4, 4))
        lanes = tl.arange(0, 4)
        bases = tl.sum(tl.where(lanes[None, :] == 0, group_slots, 0), 1)
        expected = bases[:, None] + lanes[None, :]
        grouped_valid = tl.reshape(valid, (BLOCK_TOPK // 4, 4))
        mismatch = grouped_valid & (
            (group_slots != expected) | (bases[:, None] % 4 != 0)
        )
        contiguous = tl.sum(tl.sum(mismatch.to(tl.int32), 1), 0) == 0
        if contiguous:
            if (
                PACKED_V and v.dtype.element_ty.primitive_bitwidth == 8
                and v_sp % 4 == 0 and v_sh % 4 == 0 and v_sd % 4 == 0
            ):
                # The wrapper checks pointer alignment. The slot check above
                # guarantees all four bytes remain inside the same cache page,
                # including masked tail bytes. Preserve FP8 bits until descale.
                bases = tl.multiple_of(bases, 4)
                vp = (
                    v.to(tl.pointer_type(tl.uint32))
                    + (bases // page_size)[:, None] * (v_sp // 4)
                    + head * (v_sh // 4) + (bases % page_size)[:, None] // 4
                    + dims[None, :] * (v_sd // 4)
                )
                vm = (tl.sum(grouped_valid.to(tl.int32), 1) > 0)[:, None]
                words = tl.load(vp, mask=vm & (dims[None, :] < dim), other=0)
                bytes_v = (
                    (words[:, None, :] >> (lanes[None, :, None] * 8)) & 255
                ).to(tl.uint8)
                v_values = tl.reshape(
                    bytes_v.to(v.dtype.element_ty, bitcast=True),
                    (BLOCK_TOPK, BLOCK_D),
                )
            else:
                # [group, dim, token] exposes V's contiguous page offsets.
                vp = (
                    v + (bases // page_size)[:, None, None] * v_sp
                    + head * v_sh + (bases % page_size)[:, None, None]
                    + dims[None, :, None] * v_sd + lanes[None, None, :]
                )
                vm = grouped_valid[:, None, :] & (dims[None, :, None] < dim)
                vv = tl.load(vp, mask=vm, other=0.0)
                v_values = tl.reshape(
                    tl.permute(vv, (0, 2, 1)), (BLOCK_TOPK, BLOCK_D)
                )
        else:
            v_values = tl.load(v_ptrs, mask=mask, other=0.0)
    else:
        v_values = tl.load(v_ptrs, mask=mask, other=0.0)
    if DEQUANTIZE_FP8:
        k_values = k_values.to(tl.float32) * k_scale
        v_values = v_values.to(tl.float32) * v_scale
    tl.store(k_dst, k_values, mask=mask)
    tl.store(v_dst, v_values, mask=mask)


def _validate_compact(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size,
) -> None:
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("HCU-FA K/V cache must be rank 4")
    if k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
        raise ValueError("HCU-FA K/V page and head dimensions must match")
    if k.shape[2] != v.shape[3] or k.shape[3] != v.shape[2]:
        raise ValueError("expected K[P,H,S,D] and V[P,H,D,S]")
    if int(page_size) != k.shape[2] or int(page_size) <= 0:
        raise ValueError("page_size must equal K.shape[2] and V.shape[3]")
    if k.dtype != v.dtype or k.device != v.device:
        raise ValueError("K/V cache dtype and device must match")
    if out_k.ndim != 3 or out_v.ndim != 3 or out_k.shape != out_v.shape:
        raise ValueError("out_k/out_v must be equal-shape packed [N,H,D] tensors")
    if out_k.shape[1:] != k.shape[1::2] or out_k.dtype != out_v.dtype:
        raise ValueError("packed output head/dim shape and dtype pairs must match")
    if out_k.device != k.device or out_v.device != k.device:
        raise ValueError("all K/V tensors must use the same device")
    if not is_fp8_kv_dtype(k.dtype) and out_k.dtype != k.dtype:
        raise ValueError("non-FP8 cache and packed output dtypes must match")
    if is_fp8_kv_dtype(out_k.dtype):
        if out_k.dtype != k.dtype:
            raise ValueError("FP8 output must use the cache dtype")
    elif out_k.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("packed output must be FP16/BF16 or matching FP8")
    if req_to_token.ndim != 2 or indices.ndim != 2:
        raise ValueError("req_to_token and indices must be rank 2")
    if req_indices.ndim != 1 or seq_lens.ndim != 1 or cu_k.ndim != 1:
        raise ValueError("req_indices, seq_lens and cu_k must be rank 1")
    if (
        batch <= 0
        or topk <= 0
        or indices.shape[0] < batch
        or indices.shape[1] < topk
    ):
        raise ValueError("batch/topk exceed the supplied metadata")
    if req_indices.shape[0] < batch or seq_lens.shape[0] < batch:
        raise ValueError("request metadata is shorter than batch")
    if cu_k.shape[0] < batch + 1 or out_k.shape[0] < batch:
        raise ValueError("cu_k/output capacity is too small")
    tensors = (
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
    )
    if any(t.device != k.device for t in tensors):
        raise ValueError("all tensors must use the same device")
    for name, tensor in (
        ("req_to_token", req_to_token),
        ("req_indices", req_indices),
        ("indices", indices),
        ("seq_lens", seq_lens),
        ("cu_k", cu_k),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64")


def _launch_compact_hcu_fa(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size,
    k_scale=None,
    v_scale=None,
    *,
    config: Optional[dict] = None,
) -> None:
    _validate_compact(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        batch,
        topk,
        page_size,
    )
    tuned = (
        load_config(
            "compact",
            batch,
            topk,
            0,
            k.shape[1],
            k.shape[3],
            f"{str(k.dtype).replace('torch.', '')}_to_"
            f"{str(out_k.dtype).replace('torch.', '')}",
            page_size=page_size,
        )
        if config is None
        else config
    )
    config = {"BLOCK_TOPK": 16, "num_warps": 8, "num_stages": 1} | tuned
    block_topk = int(config["BLOCK_TOPK"])
    dequantize_fp8 = is_fp8_kv_dtype(k.dtype) and not is_fp8_kv_dtype(out_k.dtype)
    heads = k.shape[1]
    dim = k.shape[3]
    grouped_v = bool(config.get("GROUPED_V", False))
    token_major = bool(config.get("TOKEN_MAJOR", False))
    packed_v = bool(config.get("PACKED_V", False)) and v.data_ptr() % 4 == 0
    launch = {
        key: value
        for key, value in config.items()
        if key not in ("BLOCK_TOPK", "GROUPED_V", "TOKEN_MAJOR", "PACKED_V")
    }
    blocks = triton.cdiv(topk, block_topk)
    # Keep neighboring CTAs within one request's cache working set.
    grid = (blocks, heads, batch) if token_major else (batch, heads, blocks)
    _compact_kv_hcu_fa[grid](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        page_size,
        req_to_token.stride(0),
        indices.stride(0),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(3),
        v.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        _unit_scale(k_scale),
        _unit_scale(v_scale),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        DEQUANTIZE_FP8=dequantize_fp8,
        GROUPED_V=grouped_v,
        TOKEN_MAJOR=token_major,
        PACKED_V=packed_v,
        **launch,
    )


def qwen_sparse_kv_extraction_compact_hcu_fa_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size: int,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    """QSA-only compact from HCU-FA paged KV; writes outputs in place."""
    _launch_compact_hcu_fa(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        batch,
        topk,
        page_size,
        k_scale,
        v_scale,
    )
