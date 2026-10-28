# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Merge per-split sparse-decode attention states.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``sparse_attn.merge_split_states``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import _require_contiguous_last_dim

NEGATIVE_INFINITY = tl.constexpr(float("-inf"))
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _merge_split_states_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_D: tl.constexpr,
) -> None:
    """Combine per-split attention states into one, weighting by the partial lse.

    Each split produced an output already divided by its own softmax denominator, so the
    combination is a softmax over the partial log-sum-exps rather than a plain sum:
    ``out = sum_s softmax(lse)_s * out_s`` and ``lse = logsumexp_s(lse_s)``. A split that
    covered no live slot reports ``lse = -inf`` and drops out with weight zero; if *every*
    split does, the row selected nothing and must come back as a zero output with
    ``lse = -inf`` rather than ``0/0``.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, BLOCK_S)
    split_ok = splits < num_splits

    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.math.exp2((partial_lse - shift) * LOG2E), 0.0)
    total = tl.sum(weights)
    inv_total = tl.div_rn(1.0, tl.maximum(total, 1.0e-20))

    for i in tl.static_range(0, N_D):
        dims = i * BLOCK_D + tl.arange(0, BLOCK_D)
        dim_ok = dims < head_dim
        states = tl.load(
            partial_out_ptr
            + splits[:, None] * stride_po_split
            + token * stride_po_token
            + head * stride_po_head
            + dims[None, :],
            mask=split_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(states * weights[:, None], axis=0) * inv_total
        tl.store(
            out_ptr + token * stride_out_token + head * stride_out_head + dims,
            merged.to(out_ptr.dtype.element_ty),
            mask=dim_ok,
        )
    tl.store(
        lse_ptr + token * stride_lse_token + head,
        tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
    )


@triton.jit
def _merge_split_states_dgrid_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po_split,
    stride_po_token,
    stride_po_head,
    stride_pl_split,
    stride_pl_token,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    num_splits,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Merge one 64-wide head-dimension tile per program."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim_tile = tl.program_id(2)

    splits = tl.arange(0, BLOCK_S)
    split_ok = splits < num_splits
    partial_lse = tl.load(
        partial_lse_ptr + splits * stride_pl_split + token * stride_pl_token + head,
        mask=split_ok,
        other=NEGATIVE_INFINITY,
    )
    peak = tl.max(partial_lse)
    empty = peak == NEGATIVE_INFINITY
    shift = tl.where(empty, 0.0, peak)
    weights = tl.where(split_ok, tl.math.exp2((partial_lse - shift) * LOG2E), 0.0)
    total = tl.sum(weights)
    inv_total = tl.div_rn(1.0, tl.maximum(total, 1.0e-20))

    dims = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    states = tl.load(
        partial_out_ptr
        + splits[:, None] * stride_po_split
        + token * stride_po_token
        + head * stride_po_head
        + dims[None, :],
        mask=split_ok[:, None] & dim_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(states * weights[:, None], axis=0) * inv_total
    tl.store(
        out_ptr + token * stride_out_token + head * stride_out_head + dims,
        merged.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )
    if dim_tile == 0:
        tl.store(
            lse_ptr + token * stride_lse_token + head,
            tl.where(empty, NEGATIVE_INFINITY, shift + tl.log(total)),
        )


def merge_split_states(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[splits, tokens, heads, dim]`` partial states to ``[tokens, heads, dim]``.

    ``partial_out`` and the allocated ``out`` must be contiguous in the last
    dimension (``stride(-1) == 1``); the kernel addresses ``head_dim`` as
    ``base + d``. Pass ``out`` / ``lse`` to write the reduction in place and
    skip the extra allocation.

    Semantics follow ``merge_variable_split_nat_lse_states_sm90_gqa`` -- "nat" being the
    natural-log lse the sparse kernels emit. The upstream kernel additionally supports a
    *variable* split plan, where the first ``n_split4`` requests are split four ways, the
    next ``n_split2`` twice, and the rest not at all, packed contiguously. That layout
    exists to fill the SMs at small batch and carries no semantics, so the release uses a
    uniform split and pays a little occupancy for a much simpler contract.
    """
    num_splits, tokens, num_heads, head_dim = partial_out.shape
    _require_contiguous_last_dim(partial_out, "partial_out")
    if out is None:
        out = torch.empty(
            (tokens, num_heads, head_dim),
            device=partial_out.device,
            dtype=partial_out.dtype,
        )
    _require_contiguous_last_dim(out, "out")
    if lse is None:
        lse = torch.empty(
            (tokens, num_heads), device=partial_out.device, dtype=torch.float32
        )
    block_s = triton.next_power_of_2(num_splits)
    block_d = 64
    if tokens <= 32:
        # Small-token decode/prefill does not expose enough workgroups to fill
        # the device. Split the 192-wide head across D to triple grid size.
        # For larger token counts the repeated LSE reduction/load costs more
        # than the extra parallelism, so retain the original kernel below.
        num_dim_tiles = triton.cdiv(head_dim, block_d)
        _merge_split_states_dgrid_kernel[(tokens, num_heads, num_dim_tiles)](
            partial_out,
            partial_lse,
            out,
            lse,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            num_splits,
            head_dim=head_dim,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
            num_warps=1,
            num_stages=1,
        )
        return out, lse

    # Prefill merge is 1024 tiny workgroups (Q=128, H=8, S=18). Default HIP
    # num_warps=4 launched 4096 waves of 255 instructions and spent 50% of
    # cycles No-Eligible. One warp is enough for S<=32; large decode S keeps
    # more warps so BLOCK_S*BLOCK_D tiles still fit. Walk D in 64-wide power-of-two
    # tiles so a 192-wide head does not pad to 256.
    if block_s <= 32:
        num_warps = 1
    elif block_s <= 64:
        num_warps = 2
    else:
        num_warps = 4
    _merge_split_states_kernel[(tokens, num_heads)](
        partial_out,
        partial_lse,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        num_splits,
        head_dim=head_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        N_D=triton.cdiv(head_dim, block_d),
        num_warps=num_warps,
        num_stages=1,
    )
    return out, lse

