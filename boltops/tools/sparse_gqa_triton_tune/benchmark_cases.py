"""Production-shape simulations shared by Sparse-GQA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SparseGQABenchmarkCase:
    name: str
    tp: int
    batch: int
    context_len: int
    q_heads: int
    kv_heads: int = 1
    head_dim: int = 256
    token_topk: int = 2048
    compress_ratio: int = 4
    page_size: int = 64

    @property
    def topk(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    @property
    def case_id(self) -> str:
        return f"TP{self.tp}-{self.name}"


PRODUCTION_CASES = (
    SparseGQABenchmarkCase("D1", 8, 1, 4096, 3),
    SparseGQABenchmarkCase("D8", 8, 8, 32768, 3),
    SparseGQABenchmarkCase("D32", 8, 32, 32768, 3),
    SparseGQABenchmarkCase("D1", 4, 1, 4096, 6),
    SparseGQABenchmarkCase("D8", 4, 8, 32768, 6),
    SparseGQABenchmarkCase("D32", 4, 32, 32768, 6),
)


def fp8_e4m3_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None) or getattr(
        torch, "float8_e4m3fnuz", None
    )
    if dtype is None:
        raise RuntimeError("this torch build does not expose an FP8 e4m3 dtype")
    return dtype


def _expanded_qsa_indices(
    case: SparseGQABenchmarkCase,
    generator: torch.Generator,
    device: torch.device,
    access_pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_topk = math.ceil(case.token_topk / case.compress_ratio)
    complete_blocks = case.context_len // case.compress_ratio
    if complete_blocks < block_topk:
        raise ValueError("context is too short for the requested QSA token budget")

    block_rows = []
    for _ in range(case.batch):
        if access_pattern == "production":
            blocks = torch.randperm(
                complete_blocks, device=device, generator=generator
            )[:block_topk]
        elif access_pattern == "contiguous":
            blocks = torch.arange(block_topk, device=device)
        else:
            raise ValueError(f"unknown access pattern: {access_pattern}")
        block_rows.append(blocks)
    blocks = torch.stack(block_rows).to(torch.int64)
    offsets = torch.arange(case.compress_ratio, device=device, dtype=torch.int64)
    expanded = (
        blocks.unsqueeze(-1) * case.compress_ratio + offsets
    ).reshape(case.batch, -1)[:, : case.token_topk]

    tail_count = case.context_len % case.compress_ratio
    indices = torch.full(
        (case.batch, case.topk), -1, device=device, dtype=torch.int32
    )
    indices[:, : case.token_topk] = expanded.to(torch.int32)
    if tail_count:
        tail_start = case.context_len - tail_count
        indices[:, case.token_topk : case.token_topk + tail_count] = torch.arange(
            tail_start, case.context_len, device=device, dtype=torch.int32
        )
    counts = torch.full(
        (case.batch,), case.token_topk + tail_count,
        device=device, dtype=torch.int32,
    )
    return indices, counts


def make_production_inputs(
    case: SparseGQABenchmarkCase,
    *,
    access_pattern: str = "production",
    fp8_cache: bool = True,
    seed: int = 123,
    k_scale: float = 0.5,
    v_scale: float = 0.25,
) -> dict:
    """Create a deterministic simulation of the documented HCU decode path.

    ``production`` spreads selected QSA blocks across the full documented
    context and permutes physical pages. ``contiguous`` keeps both mappings
    ordered and is useful only as an ideal-locality upper bound.
    """

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = math.ceil(case.context_len / case.page_size)
    total_pages = case.batch * pages_per_request
    physical_pages = torch.arange(total_pages, device=device, dtype=torch.int64)
    if access_pattern == "production":
        physical_pages = physical_pages[
            torch.randperm(total_pages, device=device, generator=generator)
        ]
    elif access_pattern != "contiguous":
        raise ValueError(f"unknown access pattern: {access_pattern}")
    page_table = physical_pages.reshape(case.batch, pages_per_request)
    logical = torch.arange(case.context_len, device=device, dtype=torch.int64)
    req_to_token = (
        page_table[:, logical // case.page_size] * case.page_size
        + logical % case.page_size
    ).to(torch.int32)

    indices, counts = _expanded_qsa_indices(
        case, generator, device, access_pattern
    )
    cu_k = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), counts.cumsum(0))
    )
    req_indices = torch.arange(case.batch, device=device, dtype=torch.int32)
    seq_lens = torch.full(
        (case.batch,), case.context_len, device=device, dtype=torch.int32
    )

    base_k = torch.randn(
        (total_pages, case.kv_heads, case.page_size, case.head_dim),
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    base_v_nhd = torch.randn(
        (total_pages, case.kv_heads, case.page_size, case.head_dim),
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    base_v = base_v_nhd.transpose(2, 3).contiguous()
    if fp8_cache:
        cache_dtype = fp8_e4m3_dtype()
        k_cache = base_k.to(cache_dtype)
        v_cache = base_v.to(cache_dtype)
        packed_k_scale, packed_v_scale = k_scale, v_scale
    else:
        k_cache, v_cache = base_k, base_v
        packed_k_scale = packed_v_scale = 1.0
    del base_k, base_v, base_v_nhd

    packed_shape = (int(counts.sum()), case.kv_heads, case.head_dim)
    packed_k = torch.empty(packed_shape, device=device, dtype=torch.bfloat16)
    packed_v = torch.empty_like(packed_k)
    q = torch.randn(
        (case.batch, case.q_heads, case.head_dim),
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    cu_q = torch.arange(case.batch + 1, device=device, dtype=torch.int32)
    relative_indices = torch.arange(
        case.topk, device=device, dtype=torch.int32
    ).expand(case.batch, -1)
    relative_indices = relative_indices.masked_fill(
        relative_indices >= counts[:, None], -1
    ).contiguous()

    compact = {
        "k": k_cache,
        "v": v_cache,
        "req_to_token": req_to_token,
        "req_indices": req_indices,
        "indices": indices,
        "seq_lens": seq_lens,
        "cu_k": cu_k,
        "counts": counts,
        "out_k": packed_k,
        "out_v": packed_v,
        "batch": case.batch,
        "topk": case.topk,
        "page_size": case.page_size,
    }
    decode = (
        q, packed_k, packed_v, relative_indices, cu_q, cu_k, counts
    )
    return {
        "case": case,
        "compact": compact,
        "decode": decode,
        "k_scale": packed_k_scale,
        "v_scale": packed_v_scale,
        "access_pattern": access_pattern,
        "seed": seed,
    }
