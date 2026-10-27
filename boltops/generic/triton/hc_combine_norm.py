# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for the HyperConnection combine + grouped Gemma RMSNorm.

``hc_combine_norm`` is the qwen4_exp (qwen3.8-flash) HyperConnection boundary
op: for every token row and hc-stream ``s`` it computes

    gate_s = 2 * sigmoid(injection_logits[s] / HC)
    out_s  = bf16(residual_s + block_output * gate_s)   # rounded here
    y_s    = rmsnorm(out_s) * (1 + w_s)                 # Gemma-style affine

and returns both tensors in one launch: ``out`` is the new residual stream
(read by the next combine), ``y`` feeds the next down/gate projections.

Two kernels share this module:

- ``_hc_combine_norm_kernel``: the production vLLM AMD kernel, kept verbatim
  (grid ``(N, HC)``, one program per (row, stream) tile; ``launch_pdl`` is
  always False because gfx936 has no PDL, matching the production wrapper).
- ``_hc_combine_norm_row_split_kernel``: grid ``(N,)``, one program per
  full row — four streams of loads are in flight before the RMSNorm
  reduction, the per-stream reductions collapse into one ``[HC]`` vector
  reduction, and the row is split into two power-of-two segments
  (2560 = 2048 + 512) so no lane is masked off (101 -> 65 VGPRs vs the
  padded one-row variant).

The public ``hc_combine_norm`` dispatches v4 -> baseline. v4 needs
``N >= _MIN_ROW_TOKENS`` (below that the one-row-per-program grid quantizes
badly against the 80 CUs: it loses 4-10% at N=96-192), a contiguous
row-major residual (its flat address fuses row and stream),
``HCDIM <= _MAX_HC_DIM`` (one program holds the full-row fp32 tile), and
``HCDIM = 2^k + 2^m`` (both segments load unmasked). Everything else —
including hidden sizes that do not split into two power-of-two segments —
falls back to the baseline kernel.

Measured on BW151/gfx936 (qwen3.8-flash dims, bf16; see docs/ops/generic/
hc_combine_norm.md): N=8192 goes 574 -> 448 us (1.28x, 1.22 TB/s = 90% of
the device's 1.35 TB/s copy ceiling); N < 256 stays on the baseline so
decode regresses nothing.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

# Measured crossover on BW151: the row kernel wins from N=256 up (1.12x
# at 256, 1.28x at 8192); below that the grid under-fills the 80 CUs (it
# loses 4-10% at N=96-192), so small batches stay on the fine-grained
# baseline kernel.
_MIN_ROW_TOKENS = 256
# One program holds the full-row fp32 tile ([HC, A] + [HC, B]); 4096 is the
# largest that stays spill-free at num_warps=4 on gfx936. Larger hidden
# sizes fall back to the baseline.
_MAX_HC_DIM = 4096


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    row = tl.program_id(0)
    stream = tl.program_id(1)
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    inj = tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This
    # matches the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    # Keep the two-axis reduction: flattening the padded tile is ~40% slower
    # at decode sizes.
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # Loading the weight earlier helps decode but keeps the tile live across
    # the reduction and regresses larger batches, so defer it to the norm.
    w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


@triton.jit
def _hc_combine_norm_row_split_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_inj,
    DIM: tl.constexpr,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    s = tl.arange(0, HC)
    ia = tl.arange(0, BLOCK_A)
    ib = tl.arange(0, BLOCK_B)
    s_off = s[:, None] * HC_DIM
    base = row * DIM + s_off

    offs_a = base + ia[None, :]
    offs_b = base + (BLOCK_A + ib)[None, :]
    # Weight offsets are stream-relative: the row term must NOT be added
    # here (w is DIM or HC_DIM long, not N*DIM — adding it was an instant
    # GPU VM fault during development).
    offs_w_a = ia[None, :] if W_SHARED else s_off + ia[None, :]
    offs_w_b = (BLOCK_A + ib)[None, :] if W_SHARED else s_off + (BLOCK_A + ib)[None, :]

    res_a = tl.load(res_ptr + offs_a)
    res_b = tl.load(res_ptr + offs_b)
    inj = tl.load(inj_ptr + row * stride_inj + s)
    blk_a = tl.load(block_ptr + row * stride_block + ia)
    blk_b = tl.load(block_ptr + row * stride_block + BLOCK_A + ib)

    gate = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    out_a = (res_a.to(tl.float32) + blk_a[None, :].to(tl.float32) * gate[:, None]).to(
        out_ptr.dtype.element_ty
    )
    out_b = (res_b.to(tl.float32) + blk_b[None, :].to(tl.float32) * gate[:, None]).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + offs_a, out_a)
    tl.store(out_ptr + offs_b, out_b)

    of_a = out_a.to(tl.float32)
    of_b = out_b.to(tl.float32)
    # The norm reduction is the sum of the two segment reductions; the only
    # numeric difference vs the baseline is that reduction order (bf16
    # noise-level in y).
    sum_sq = tl.sum(of_a * of_a, axis=1) + tl.sum(of_b * of_b, axis=1)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    w_a = tl.load(w_ptr + offs_w_a)
    w_b = tl.load(w_ptr + offs_w_b)
    y_a = of_a * rrms[:, None]
    y_a += y_a * w_a.to(tl.float32)
    y_b = of_b * rrms[:, None]
    y_b += y_b * w_b.to(tl.float32)
    tl.store(y_ptr + offs_a, y_a)
    tl.store(y_ptr + offs_b, y_b)


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream kernel verbatim: grid ``(N, HC)``, ``BLOCK_SIZE=512``, 4 warps."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_norm_kernel[(N, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        launch_pdl=False,
    )
    return out, y


def _split(hc_dim: int) -> tuple[int, int]:
    """Two power-of-two segments covering the row exactly, else any pair."""
    if hc_dim & (hc_dim - 1) == 0:
        half = hc_dim // 2
        return half, half
    a = 1 << (hc_dim.bit_length() - 1)
    return a, triton.next_power_of_2(hc_dim - a)


def _split_supported(hc_dim: int) -> bool:
    a, b = _split(hc_dim)
    return a + b == hc_dim and hc_dim <= _MAX_HC_DIM


def _hc_combine_norm_row_split(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One row per program, split into two unmasked power-of-two tiles;
    needs ``HCDIM = 2^k + 2^m``."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)
    assert residual.stride(1) == 1 and residual.stride(0) == DIM
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1

    block_a, block_b = _split(hc_dim)
    # Both segments are loaded unmasked (they cover the row exactly), so
    # this variant only supports HCDIM = 2^k + 2^m; anything else must use
    # the baseline.
    assert block_a + block_b == hc_dim
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    _hc_combine_norm_row_split_kernel[(N,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        injection_logits.stride(0),
        DIM,
        hc_dim,
        hc_count,
        BLOCK_A=block_a,
        BLOCK_B=block_b,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        num_warps=num_warps,
    )
    return out, y


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ship entry: v4 for large splittable rows, the baseline kernel for
    everything else (small batches, strided residuals, hidden sizes that do
    not split into two power-of-two segments or exceed _MAX_HC_DIM).

    Fallback outputs are bitwise identical to ``_hc_combine_norm``
    (same kernel); the row_split path differs from it only by bf16 reduction-order
    noise in ``y`` (``out`` is bitwise identical).
    """
    N, DIM = residual.shape
    hc_dim = DIM // hc_count
    if N >= _MIN_ROW_TOKENS and residual.is_contiguous() and _split_supported(hc_dim):
        impl = _hc_combine_norm_row_split
    else:
        impl = _hc_combine_norm
    return impl(residual, block_output, injection_logits, norm_weight, eps, hc_count)


__all__ = ["hc_combine_norm"]
