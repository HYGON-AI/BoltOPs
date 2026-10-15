"""CSA region layout and per-region FP32/FP8 summary compression.

Vendored from Step-4 DSA handoff ``inference/kernel.py`` (``dsa_kernels`` CSA).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from boltops.step4_dsa.triton.common import _NO_FP_FUSION, REGION_BLOCK_SIZE


@triton.jit
def _csa_compress_regions_kernel(
    index_k_ptr,
    index_z_ptr,
    token_start_ptr,
    token_count_ptr,
    summary_ptr,
    mean_ptr,
    stride_k_token,
    stride_k_head,
    stride_z_token,
    stride_z_head,
    stride_summary_region,
    stride_summary_head,
    num_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """One program per (region, head): compress a region's tokens into one vector.

    The weight is a softmax over the token axis of ``index_z`` taken **per dimension**,
    so this is an element-wise weighted mean, not a convex combination of whole key
    vectors:

        s_d = sum_t exp(z_td - m) * k_td / sum_t exp(z_td - m)

    ``m`` is a scalar over both axes. It cancels algebraically and only exists to keep
    ``exp`` in range -- but it is the *same* scalar the CuTe kernel uses, which is why
    the two agree in the last places rather than merely within tolerance.
    """
    region = tl.program_id(0)
    head = tl.program_id(1)

    start = tl.load(token_start_ptr + region).to(tl.int64)
    count = tl.load(token_count_ptr + region).to(tl.int64)

    tokens = tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    token_ok = tokens < count
    dim_ok = dims < proxy_dim
    mask = token_ok[:, None] & dim_ok[None, :]

    token_ids = (start + tokens.to(tl.int64))[:, None]
    k_offsets = token_ids * stride_k_token + head * stride_k_head + dims[None, :]
    z_offsets = token_ids * stride_z_token + head * stride_z_head + dims[None, :]
    logits = tl.load(index_z_ptr + z_offsets, mask=mask, other=float("-inf")).to(tl.float32)
    values = tl.load(index_k_ptr + k_offsets, mask=mask, other=0.0).to(tl.float32)

    shift = tl.max(tl.where(mask, logits, float("-inf")))
    weights = tl.where(mask, tl.exp(logits - shift), 0.0)
    denominator = tl.sum(weights, axis=0)
    numerator = tl.sum(weights * values, axis=0)
    summary = tl.where(
        denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
    )

    out = region * stride_summary_region + head * stride_summary_head + dims
    tl.store(summary_ptr + out, summary, mask=dim_ok)
    # Round through bfloat16 on the way to e4m3, matching the CuTe kernel. Going
    # straight from fp32 is observably different: bf16 keeps 8 mantissa bits and e4m3
    # keeps 3, so a value sitting on an e4m3 midpoint can round the other way once bf16
    # has already moved it. The parity gate compares bytes, so this is not cosmetic.
    tl.store(mean_ptr + out, summary.to(tl.bfloat16).to(tl.float8e4nv), mask=dim_ok)


def csa_region_layout(
    seq_lens: torch.Tensor, *, regions_per_seq: int, region_size: int = REGION_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token span of every region, for sequences packed back to back.

    ``regions_per_seq`` is the stride of the region id space, not the number of regions a
    sequence actually fills -- a sequence's regions have to land at a fixed offset so the
    summary store can be indexed without a second lookup. Regions past a sequence's end
    get a count of zero and are skipped rather than written, which keeps a stale summary
    from a previous request out of the selection.
    """
    device = seq_lens.device
    lengths = seq_lens.to(torch.int64)
    token_starts = torch.cumsum(lengths, dim=0) - lengths

    regions = torch.arange(regions_per_seq, device=device, dtype=torch.int64)
    offset = regions.view(1, -1) * region_size
    count = (lengths.view(-1, 1) - offset).clamp(0, region_size)
    start = token_starts.view(-1, 1) + offset
    return start.reshape(-1).to(torch.int32), count.reshape(-1).to(torch.int32)


def csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress packed tokens into per-region summaries.

    Args:
        index_k: ``[tokens, heads, proxy_dim]`` proxy keys. Last dim must be contiguous;
            token/head strides may differ from ``index_z``.
        index_z: ``[tokens, heads, proxy_dim]`` per-dimension softmax logits. Last dim
            must be contiguous; token/head strides are passed separately from ``index_k``.
        token_start: ``[regions]`` int32, first token of each region.
        token_count: ``[regions]`` int32, tokens present in each region; 0 = skip.

    Returns:
        ``(summary_fp32, summary_fp8)``, both ``[regions, heads, proxy_dim]``. The fp8
        tensor is what selection and sparse attention consume; the fp32 one exists
        because it is the only place the unrounded value is observable, which is what
        lets a parity failure be attributed to the reduction rather than the rounding.
    """
    if index_k.shape != index_z.shape:
        raise ValueError(
            f"index_k and index_z must match, got {tuple(index_k.shape)} and "
            f"{tuple(index_z.shape)}"
        )
    if index_k.ndim != 3:
        raise ValueError(f"index_k must be [tokens, heads, proxy_dim], got {tuple(index_k.shape)}")
    if index_k.stride(2) != 1:
        raise ValueError("index_k must be contiguous in the proxy dimension")
    if index_z.stride(2) != 1:
        raise ValueError("index_z must be contiguous in the proxy dimension")

    regions = int(token_start.numel())
    _, heads, proxy_dim = index_k.shape
    summary = torch.zeros((regions, heads, proxy_dim), device=index_k.device, dtype=torch.float32)
    mean = torch.zeros((regions, heads, proxy_dim), device=index_k.device, dtype=torch.float8_e4m3fn)

    _csa_compress_regions_kernel[(regions, heads)](
        index_k,
        index_z,
        token_start,
        token_count,
        summary,
        mean,
        index_k.stride(0),
        index_k.stride(1),
        index_z.stride(0),
        index_z.stride(1),
        summary.stride(0),
        summary.stride(1),
        num_heads=heads,
        proxy_dim=proxy_dim,
        BLOCK_T=triton.next_power_of_2(region_size),
        BLOCK_D=triton.next_power_of_2(proxy_dim),
        # ``weights * values`` feeds a tree reduction, and a backend that contracts the
        # multiply into the summation's FMA changes the fp32 result by an ulp or two.
        # That is inside the fp32 band the gate allows, but the summary is *also* stored
        # through bfloat16 to e4m3 -- a double rounding -- and a value sitting near the
        # bf16 tie boundary then lands on a different e4m3 code. The gate compares those
        # bytes, so the reduction has to round exactly like the reference.
        **_NO_FP_FUSION,
    )
    return summary, mean


def csa_prefill(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    regions_per_request: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Region summaries for one prefill launch, in logical region-id order.

    Region ids follow ``request * regions_per_request + position // region_size``.
    """
    token_start, token_count = csa_region_layout(
        seq_lens, regions_per_seq=regions_per_request, region_size=region_size
    )
    return csa_compress_regions(
        index_k, index_z, token_start, token_count, region_size=region_size
    )
