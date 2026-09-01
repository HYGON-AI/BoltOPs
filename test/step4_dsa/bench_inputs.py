"""Build timed callables for Step-4 DSA benchmark cases.

Input construction follows the handoff ``operator_parity.benchmark_dsa`` helpers
so production shapes stay comparable.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from boltops.step4_dsa import (
    triton_csa_prefill,
    triton_decode_sparse_meta,
    triton_indexer_logits,
    triton_indexer_norm_rope,
    triton_merge_split_states,
    triton_prefill_sparse_meta,
    triton_sparse_attention_decode,
    triton_sparse_attention_prefill,
    triton_topk_select_and_pack,
)
from boltops.step4_dsa.geometry import STEP4, Step4Geometry

from .benchmark_cases import BenchmarkCase, tp1_geometry, tp8_geometry
from .gen import make_csa_prefill_inputs, make_indexer_inputs


def _geo(case: BenchmarkCase) -> Step4Geometry:
    return tp1_geometry() if case.case_id.startswith("tp1.") else tp8_geometry()


def _make_norm_inputs(
    geo: Step4Geometry,
    *,
    tokens: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    rotary_dim = geo.indexer_rope_dim // 2
    dtype = torch.bfloat16
    max_position = max(4096, tokens + 1)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype).contiguous()

    inverse_frequency = 1.0 / (
        10000.0
        ** (torch.arange(0, rotary_dim, device=device, dtype=torch.float32) / rotary_dim)
    )
    angles = torch.arange(max_position, device=device, dtype=torch.float32)[:, None] * (
        inverse_frequency[None, :]
    )
    return {
        "index_q": randn(tokens, geo.indexer_num_heads * geo.proxy_dim),
        "index_k": randn(tokens, geo.indexer_num_k_heads * geo.proxy_dim),
        "index_z": randn(tokens, geo.indexer_num_k_heads * geo.proxy_dim),
        "q_norm_weight": randn(geo.proxy_dim),
        "k_norm_weight": randn(geo.proxy_dim),
        "k_norm_bias": randn(geo.proxy_dim),
        "cos": angles.cos().to(dtype).contiguous(),
        "sin": angles.sin().to(dtype).contiguous(),
        "positions": torch.randint(
            0, max_position, (tokens,), generator=generator, device=device, dtype=torch.int32
        ),
    }


def _selector_inputs(case: BenchmarkCase, device: torch.device, seed: int):
    geo = _geo(case)
    rows = int(case.params["rows"])
    regions = int(case.params["regions"])
    topk = int(case.params["topk"])
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn((rows, regions), generator=generator, device=device, dtype=torch.float32)
    logits = torch.where(logits.abs() < 0.08, torch.zeros_like(logits), logits).contiguous()
    lengths = torch.full((rows,), regions, device=device, dtype=torch.int32)
    q_positions = torch.full(
        (rows,), regions * geo.region_block_size - 1, device=device, dtype=torch.int32
    )
    pages = -(-regions // geo.regions_per_page)
    table_layout = str(case.params.get("table_layout", "shuffled"))
    if table_layout == "identity":
        table = torch.arange(pages, device=device, dtype=torch.int32)
    else:
        table = torch.randperm(pages, generator=generator, device=device).to(torch.int32)
    unmapped_pages = int(case.params.get("unmapped_pages", 0))
    if unmapped_pages:
        holes = torch.randperm(pages, generator=generator, device=device)[:unmapped_pages]
        table = table.clone()
        table[holes] = -1
    block_table = table.view(1, -1).contiguous()
    return geo, logits, lengths, q_positions, block_table, topk


def _build_prefill_inputs(
    case: BenchmarkCase,
    device: torch.device,
    seed: int,
    geo: Step4Geometry | None = None,
) -> dict:
    geo = _geo(case) if geo is None else geo
    queries = int(case.params["queries"])
    cache_regions = int(case.params["cache_regions"])
    topk = int(case.params["topk"])
    heads = int(case.params["heads"])
    groups = int(case.params["groups"])
    request_lengths = tuple(case.params.get("request_lengths", (queries,)))
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (queries, heads, geo.head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    regions_per_page = geo.regions_per_page
    page_size = geo.paged_block_size
    pages_per_request = -(-cache_regions // regions_per_page)
    total_pages = pages_per_request * len(request_lengths)
    page_order = torch.randperm(total_pages, generator=generator, device=device).to(torch.int32)
    key_cache = torch.empty(
        (total_pages * page_size, groups, geo.head_dim), device=device, dtype=torch.bfloat16
    )
    value_cache = torch.empty_like(key_cache)
    tables = []
    for request in range(len(request_lengths)):
        table = page_order[request * pages_per_request : (request + 1) * pages_per_request]
        tables.append(table)
        logical_k = torch.randn(
            (pages_per_request * page_size, groups, geo.head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        logical_v = torch.randn(
            (pages_per_request * page_size, groups, geo.head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for logical_page in range(pages_per_request):
            physical_page = int(table[logical_page])
            source = slice(logical_page * page_size, (logical_page + 1) * page_size)
            target = slice(physical_page * page_size, (physical_page + 1) * page_size)
            key_cache[target] = logical_k[source]
            value_cache[target] = logical_v[source]

    width = topk + 1
    rows = groups * queries
    packed = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    logical_ids = torch.full_like(packed, -1)
    counts = torch.empty((rows,), device=device, dtype=torch.int32)
    positions_base = torch.empty((queries,), device=device, dtype=torch.int32)
    requests_base = torch.empty((queries,), device=device, dtype=torch.int32)
    query_cursor = 0
    for request, length in enumerate(request_lengths):
        start_position = max(cache_regions * geo.region_block_size - length, 0)
        positions_base[query_cursor : query_cursor + length] = torch.arange(
            start_position, start_position + length, device=device, dtype=torch.int32
        )
        requests_base[query_cursor : query_cursor + length] = request
        query_cursor += length
    q_positions = positions_base.repeat(groups).contiguous()
    request_indices = requests_base.repeat(groups).contiguous()
    for row in range(rows):
        position = int(q_positions[row])
        request = int(request_indices[row])
        current = min(position // geo.region_block_size, cache_regions - 1)
        history_count = min(current, topk)
        first = current - history_count
        selected = list(range(first, current)) + [current]
        counts[row] = len(selected)
        table = tables[request]
        for slot, logical_region in enumerate(selected):
            physical_page = int(table[logical_region // regions_per_page])
            physical = physical_page * regions_per_page + logical_region % regions_per_page
            valid = min(
                max(position + 1 - logical_region * geo.region_block_size, 0),
                geo.region_block_size,
            )
            packed[row, slot] = physical | (valid << 24)
            logical_ids[row, slot] = logical_region
    return {
        "query": query,
        "key_cache": key_cache.contiguous(),
        "value_cache": value_cache.contiguous(),
        "packed": packed,
        "counts": counts.contiguous(),
        "logical_ids": logical_ids,
        "q_positions": q_positions,
        "request_indices": request_indices,
        "geo": geo,
        "groups": groups,
    }


def _build_decode_inputs(
    case: BenchmarkCase,
    device: torch.device,
    seed: int,
    geo: Step4Geometry | None = None,
) -> dict:
    geo = _geo(case) if geo is None else geo
    batch = int(case.params["batch"])
    context = int(case.params["context"])
    groups = int(case.params["groups"])
    heads = int(case.params["heads"])
    pages_per_request = -(-context // geo.paged_block_size)
    regions = -(-context // geo.region_block_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (batch, heads, geo.head_dim), generator=generator, device=device, dtype=torch.bfloat16
    ).contiguous()
    total_pages = batch * pages_per_request
    page_order = torch.randperm(total_pages, generator=generator, device=device).to(torch.int32)
    block_table = page_order.view(batch, pages_per_request).contiguous()
    unmapped_pages = int(case.params.get("unmapped_pages", 0))
    if unmapped_pages:
        block_table = block_table.clone()
        for request in range(batch):
            holes = torch.randperm(pages_per_request, generator=generator, device=device)[
                :unmapped_pages
            ]
            block_table[request, holes] = -1
    cache_tokens = total_pages * geo.paged_block_size
    key_cache = torch.randn(
        (cache_tokens, groups, geo.head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    value_cache = torch.randn(
        (cache_tokens, groups, geo.head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    kv_seqlens = torch.full((batch,), context, device=device, dtype=torch.int32)
    logits = torch.randn(
        (groups * batch, regions), generator=generator, device=device, dtype=torch.float32
    ).contiguous()
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "logits": logits,
        "kv_seqlens": kv_seqlens,
        "block_table": block_table,
        "geo": geo,
    }


def prepare_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int = 20260831,
) -> Callable[[], object]:
    """Return a zero-arg callable that launches the case operator once."""
    op = case.operator
    geo = _geo(case)

    if op == "norm_rope":
        inputs = _make_norm_inputs(
            geo, tokens=int(case.params["tokens"]), device=device, seed=seed
        )
        ordered = (
            inputs["index_q"],
            inputs["index_k"],
            inputs["index_z"],
            inputs["q_norm_weight"],
            inputs["k_norm_weight"],
            inputs["k_norm_bias"],
            inputs["cos"],
            inputs["sin"],
            inputs["positions"],
        )
        kwargs = dict(
            head_dim=geo.proxy_dim,
            num_q_heads=geo.indexer_num_heads,
            num_k_heads=geo.indexer_num_k_heads,
            rotary_dim=geo.indexer_rope_dim // 2,
            eps=1e-6,
            q_norm_weight_bias=1.0,
        )

        def call():
            return triton_indexer_norm_rope(*ordered, **kwargs)

        return call

    if op == "csa":
        seq_lens = tuple(case.params.get("seq_lens", (int(case.params["tokens"]),)))
        regions_per_request = max(-(-length // geo.region_block_size) for length in seq_lens)
        metadata = make_csa_prefill_inputs(
            geo,
            seq_lens=seq_lens,
            regions_per_request=regions_per_request,
            device=device,
            seed=seed,
        )

        def call():
            return triton_csa_prefill(
                metadata.index_k,
                metadata.index_z,
                metadata.seq_lens,
                regions_per_request=regions_per_request,
                region_size=geo.region_block_size,
            )

        return call

    if op == "logits":
        inputs = make_indexer_inputs(
            geo,
            seq_q=int(case.params["queries"]),
            seq_k=int(case.params["regions"]),
            device=device,
            seed=seed,
        )

        def call():
            return triton_indexer_logits(*inputs)

        return call

    if op == "selector":
        geo, logits, lengths, q_positions, block_table, topk = _selector_inputs(
            case, device, seed
        )

        def call():
            return triton_topk_select_and_pack(
                logits,
                lengths,
                q_positions,
                block_table,
                topk=topk,
                region_size=geo.region_block_size,
                regions_per_page=geo.regions_per_page,
            )

        return call

    if op == "metadata":
        geo, logits, _lengths, q_positions, block_table, topk = _selector_inputs(
            case, device, seed
        )
        request_indices = torch.zeros((logits.shape[0],), device=device, dtype=torch.int32)

        def call():
            return triton_prefill_sparse_meta(
                logits,
                q_positions,
                block_table,
                request_indices,
                topk=topk,
                region_size=geo.region_block_size,
                regions_per_page=geo.regions_per_page,
                return_logical_ids=True,
            )

        return call

    if op == "decode_metadata":
        inputs = _build_decode_inputs(case, device, seed)
        topk = int(case.params["topk"])

        def call():
            return triton_decode_sparse_meta(
                inputs["logits"],
                inputs["kv_seqlens"],
                inputs["block_table"],
                topk=topk,
                region_size=geo.region_block_size,
                regions_per_page=geo.regions_per_page,
            )

        return call

    if op in {"prefill", "tiled_prefill"}:
        data = _build_prefill_inputs(case, device, seed)
        tiled = op == "tiled_prefill"
        kwargs = {
            "num_kv_groups": data["groups"],
            "region_size": geo.region_block_size,
            "softmax_scale": float(geo.head_dim) ** -0.5,
            "block_regions": 4 if tiled else 16,
        }
        if tiled:
            kwargs.update(
                logical_regions=data["logical_ids"],
                q_positions=data["q_positions"],
                request_indices=data["request_indices"],
                block_q=128,
            )

        def call():
            return triton_sparse_attention_prefill(
                data["query"],
                data["key_cache"],
                data["value_cache"],
                data["packed"],
                data["counts"],
                **kwargs,
            )

        return call

    if op == "decode":
        inputs = _build_decode_inputs(case, device, seed)
        topk = int(case.params["topk"])
        splits = int(case.params["splits"])
        packed, counts = triton_decode_sparse_meta(
            inputs["logits"],
            inputs["kv_seqlens"],
            inputs["block_table"],
            topk=topk,
            region_size=geo.region_block_size,
            regions_per_page=geo.regions_per_page,
        )

        def call():
            outputs = triton_sparse_attention_decode(
                inputs["query"],
                inputs["key_cache"],
                inputs["value_cache"],
                packed,
                counts,
                inputs["kv_seqlens"],
                num_kv_groups=geo.num_attention_groups,
                region_size=geo.region_block_size,
                softmax_scale=float(geo.head_dim) ** -0.5,
                num_splits=splits,
            )
            return triton_merge_split_states(*outputs) if splits > 1 else outputs

        return call

    if op == "merge":
        splits = int(case.params["splits"])
        batch = int(case.params["batch"])
        heads = int(case.params["heads"])
        generator = torch.Generator(device=device).manual_seed(seed)
        partial_out = torch.randn(
            (splits, batch, heads, STEP4.head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        partial_lse = torch.randn(
            (splits, batch, heads),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        def call():
            return triton_merge_split_states(partial_out, partial_lse)

        return call

    raise ValueError(f"unsupported benchmark operator {op!r}")
