# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA Triton operators: paged compressed-key scoring, causal index
expansion, combined selection, and sparse paged GQA.

Single module mirroring the vLLM ``qwen4_exp`` QSA ops layout: all kernels
and the four public entries live here, with the public entry order matching
upstream (``qsa_mqa_paged`` / ``expand_qsa_block_indices`` /
``qsa_select_paged_tokens`` / ``qsa_sparse_paged_attention``).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Shared input-contract checks
# ------------------------------------------------------------------
def _require_accelerator_tensors(*tensors: torch.Tensor | None) -> None:
    for tensor in tensors:
        if tensor is not None and not tensor.is_cuda:
            raise ValueError("QSA operators require CUDA/HIP tensors.")

def _require_same_device(reference: torch.Tensor, *tensors: torch.Tensor) -> None:
    if any(tensor.device != reference.device for tensor in tensors):
        raise ValueError("QSA tensors must all use the same device.")

def _require_int32(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise ValueError(f"QSA {name} must be int32.")

def _require_last_dim_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim and tensor.stride(-1) != 1:
        raise ValueError(f"QSA {name} must have a contiguous last dimension.")

# ------------------------------------------------------------------
# Paged compressed-key MQA scoring (qsa_mqa_paged)
# ------------------------------------------------------------------
def _select_qsa_mqa_config(
    rows: int, columns: int, head_dim: int, query_heads: int,
) -> dict[str, int]:
    """Return a measured MQA-score launch configuration without device work."""
    if not all(
        isinstance(value, int)
        for value in (rows, columns, head_dim, query_heads)
    ):
        raise TypeError("QSA MQA configuration dimensions must be integers.")
    if rows < 1 or columns < 1 or head_dim < 1 or query_heads < 1:
        raise ValueError("QSA MQA configuration dimensions must be positive.")
    use_mma = query_heads <= 32 and 16 <= head_dim <= 256
    use_grouped = (
        use_mma and query_heads in (1, 2, 4, 8)
        and query_heads * head_dim >= 512
    )
    if not use_mma:
        return {
            "block_n": 32, "num_warps": 4,
            "use_mma": 0, "use_grouped_mma": 0, "group_tokens": 0,
        }
    group_tokens = max(1, 16 // query_heads) if query_heads < 16 else 1
    return {
        "block_n": 32 if head_dim >= 256 else 64,
        "num_warps": 4,
        "use_mma": 1,
        "use_grouped_mma": int(use_grouped),
        "group_tokens": group_tokens if use_grouped else 1,
    }

@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr,
    sequence_lengths_ptr, visible_blocks_ptr, logits_ptr, stride_q_row,
    stride_q_head, stride_q_dim, stride_cache_block, stride_cache_token,
    stride_cache_dim, stride_table_req, stride_table_page, stride_logits_row,
    num_rows, num_columns, num_pages, num_requests, score_divisor,
    PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Score compressed MQA keys, matching the upstream Apache-2.0 contract."""
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests), other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows) & (columns < num_columns) & (columns < visible)
        & (request >= 0) & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid, other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    keys = tl.load(
        k_cache_ptr + safe_physical_page[:, None] * stride_cache_block
        + page_offset[:, None] * stride_cache_token
        + dims[None, :] * stride_cache_dim,
        mask=valid[:, None] & (dims[None, :] < HEAD_DIM), other=0.0,
    ).to(tl.float32)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM, other=0.0,
        ).to(tl.float32)
        score += tl.maximum(tl.sum(keys * query[None, :], axis=1), 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score * (1.0 / score_divisor), -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )

@triton.jit
def _qsa_mqa_paged_mma_kernel(
    q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr,
    sequence_lengths_ptr, visible_blocks_ptr, logits_ptr, stride_q_row,
    stride_q_head, stride_q_dim, stride_cache_block, stride_cache_token,
    stride_cache_dim, stride_table_req, stride_table_page, stride_logits_row,
    num_rows, num_columns, num_pages, num_requests, score_divisor,
    PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    GROUP_TOKENS: tl.constexpr, DOT_ROWS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
) -> None:
    """Group same-request tokens into one MMA to remove H-dimension padding.

    A program owns ``GROUP_TOKENS`` consecutive query rows.  When those rows
    belong to one request, their page table and K tile are shared and Q is
    flattened as ``token * head`` rows for one MMA.  Mixed-request groups use
    the one-token MMA path for correctness.
    """
    row_base = tl.program_id(0) * GROUP_TOKENS
    token_offsets = row_base + tl.arange(0, GROUP_TOKENS)
    token_mask = token_offsets < num_rows
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)

    requests = tl.load(
        token_to_req_ptr + token_offsets, mask=token_mask, other=-1
    )
    safe_requests = tl.minimum(tl.maximum(requests, 0), num_requests - 1)
    query_positions = tl.load(
        query_positions_ptr + token_offsets, mask=token_mask, other=0
    )
    requests_valid = (
        (requests >= 0) & (requests < num_requests) & token_mask
    )
    sequence_lengths = tl.load(
        sequence_lengths_ptr + safe_requests,
        mask=requests_valid, other=0,
    )
    group_visible = tl.minimum(
        (query_positions + 1) // COMPRESS_RATIO,
        sequence_lengths // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(
            visible_blocks_ptr + token_offsets, group_visible, mask=token_mask
        )

    logical_pages = columns // PAGE_SIZE
    page_offsets = columns % PAGE_SIZE
    group_valid = (
        token_mask[:, None] & (columns[None, :] < num_columns)
        & (columns[None, :] < group_visible[:, None]) & requests_valid[:, None]
        & (logical_pages[None, :] < PAGE_TABLE_WIDTH)
    )
    group_store_mask = token_mask[:, None] & (columns[None, :] < num_columns)
    request0 = tl.load(token_to_req_ptr + row_base)
    request0_valid = (request0 >= 0) & (request0 < num_requests)
    safe_request0 = tl.minimum(tl.maximum(request0, 0), num_requests - 1)
    same_requests = (
        (row_base + GROUP_TOKENS <= num_rows)
        & request0_valid
        & (tl.sum((requests != request0).to(tl.int32), axis=0) == 0)
    )

    safe_logical_pages = tl.minimum(logical_pages, PAGE_TABLE_WIDTH - 1)
    if same_requests.item():
        max_visible = tl.max(group_visible, axis=0)
        shared_column_valid = (
            (columns < num_columns) & (columns < max_visible)
            & request0_valid & (logical_pages < PAGE_TABLE_WIDTH)
        )
        physical_pages = tl.load(
            page_table_ptr + safe_request0 * stride_table_req
            + safe_logical_pages * stride_table_page,
            mask=shared_column_valid, other=-1,
        )
        shared_column_valid &= (
            (physical_pages >= 0) & (physical_pages < num_pages)
        )
        safe_physical_pages = tl.maximum(physical_pages, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_pages[:, None] * stride_cache_block
            + page_offsets[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=shared_column_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        q_rows = row_base * NUM_HEADS + tl.arange(0, DOT_ROWS)
        token_ids = q_rows // NUM_HEADS
        head_ids = q_rows % NUM_HEADS
        query = tl.load(
            q_ptr + token_ids[:, None] * stride_q_row
            + head_ids[:, None] * stride_q_head
            + dims[None, :] * stride_q_dim,
            mask=(token_ids[:, None] < num_rows)
            & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores_t = tl.dot(keys, tl.trans(query), out_dtype=tl.float32)
        grouped_scores = tl.reshape(
            tl.trans(tl.maximum(scores_t, 0.0)),
            (GROUP_TOKENS, NUM_HEADS, BLOCK_N),
        )
        group_scores = tl.sum(grouped_scores, axis=1)
        tl.store(
            logits_ptr + token_offsets[:, None] * stride_logits_row
            + columns[None, :],
            tl.where(
                group_valid, group_scores * (1.0 / score_divisor), -float("inf")
            ),
            mask=group_store_mask,
        )
    else:
        for group_offset in tl.static_range(GROUP_TOKENS):
            row = row_base + group_offset
            row_valid = row < num_rows
            request = tl.load(token_to_req_ptr + row, mask=row_valid, other=-1)
            request_valid = (
                row_valid & (request >= 0) & (request < num_requests)
            )
            safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
            query_position = tl.load(
                query_positions_ptr + row, mask=row_valid, other=0
            )
            sequence_length = tl.load(
                sequence_lengths_ptr + safe_request,
                mask=request_valid, other=0,
            )
            token_visible = tl.minimum(
                (query_position + 1) // COMPRESS_RATIO,
                sequence_length // COMPRESS_RATIO,
            )
            token_valid = (
                row_valid & (columns < num_columns) & (columns < token_visible)
                & request_valid & (logical_pages < PAGE_TABLE_WIDTH)
            )
            physical_pages = tl.load(
                page_table_ptr + safe_request * stride_table_req
                + safe_logical_pages * stride_table_page,
                mask=token_valid, other=-1,
            )
            token_valid &= (physical_pages >= 0) & (physical_pages < num_pages)
            safe_physical_pages = tl.maximum(physical_pages, 0).to(tl.int64)
            keys = tl.load(
                k_cache_ptr
                + safe_physical_pages[:, None] * stride_cache_block
                + page_offsets[:, None] * stride_cache_token
                + dims[None, :] * stride_cache_dim,
                mask=token_valid[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
            )
            heads = tl.arange(0, HEAD_TILE)
            query_t = tl.load(
                q_ptr + row * stride_q_row + dims[:, None] * stride_q_dim
                + heads[None, :] * stride_q_head,
                mask=row_valid & (dims[:, None] < HEAD_DIM)
                & (heads[None, :] < NUM_HEADS),
                other=0.0,
            )
            scores_t = tl.dot(keys, query_t, out_dtype=tl.float32)
            token_scores = tl.sum(tl.maximum(scores_t, 0.0), axis=1)
            tl.store(
                logits_ptr + row * stride_logits_row + columns,
                tl.where(
                    token_valid, token_scores * (1.0 / score_divisor),
                    -float("inf"),
                ),
                mask=row_valid & (columns < num_columns),
            )

def _validate_mqa_inputs(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, compress_ratio: int,
    num_columns: int | None, score_scale: float | None,
) -> int:
    _require_accelerator_tensors(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths
    )
    _require_same_device(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths
    )
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim].")
    if q.dtype != torch.bfloat16:
        raise ValueError("QSA query must be bfloat16.")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim].")
    if k_cache.dtype != torch.bfloat16:
        raise ValueError("QSA compressed-key cache must be bfloat16.")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache head dimensions must match.")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional.")
    if q.shape[0] and (
        k_cache.shape[0] == 0 or k_cache.shape[1] == 0
        or page_table.shape[0] == 0 or page_table.shape[1] == 0
    ):
        raise ValueError("QSA paged scoring cache and page table must be nonempty.")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows.")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows.")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests.")
    for name, tensor in (
        ("page table", page_table), ("request mapping", token_to_req),
        ("query positions", query_positions), ("sequence lengths", sequence_lengths),
    ):
        _require_int32(name, tensor)
        _require_last_dim_contiguous(name, tensor)
    _require_last_dim_contiguous("query", q)
    _require_last_dim_contiguous("compressed-key cache", k_cache)
    if not isinstance(compress_ratio, int) or compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be a positive integer.")
    if num_columns is not None and (
        not isinstance(num_columns, int) or num_columns < 0
    ):
        raise ValueError("QSA score width must be a non-negative integer.")
    divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if not isinstance(divisor, (int, float)) or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("QSA score scale must be finite and positive.")
    return page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns

def qsa_mqa_paged(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, compress_ratio: int,
    num_columns: int | None = None, score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 QSA scores from a paged compressed-key cache."""
    columns = _validate_mqa_inputs(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        compress_ratio, num_columns, score_scale,
    )
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    mqa_config = _select_qsa_mqa_config(
        q.shape[0], columns, q.shape[2], q.shape[1]
    )
    block_n = mqa_config["block_n"]
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else float(score_scale)
    if torch.version.hip is not None and mqa_config["use_mma"]:
        group_tokens = mqa_config["group_tokens"]
        _qsa_mqa_paged_mma_kernel[
            (triton.cdiv(q.shape[0], group_tokens),
             triton.cdiv(columns, mqa_config["block_n"]))
        ](
            q, k_cache, page_table, token_to_req, query_positions,
            sequence_lengths, visible_blocks, logits, q.stride(0), q.stride(1),
            q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
            page_table.stride(0), page_table.stride(1), logits.stride(0),
            q.shape[0], columns, k_cache.shape[0], page_table.shape[0],
            score_divisor, PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2], BLOCK_N=mqa_config["block_n"],
            BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
            COMPRESS_RATIO=compress_ratio, GROUP_TOKENS=group_tokens,
            DOT_ROWS=group_tokens * q.shape[1],
            HEAD_TILE=max(16, triton.next_power_of_2(q.shape[1])),
            num_warps=mqa_config["num_warps"],
        )
        return logits, visible_blocks
    _qsa_mqa_paged_kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
        page_table.stride(0), page_table.stride(1), logits.stride(0), q.shape[0],
        columns, k_cache.shape[0], page_table.shape[0], score_divisor,
        PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2], BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]), COMPRESS_RATIO=compress_ratio,
        num_warps=mqa_config["num_warps"],
    )
    return logits, visible_blocks


# ------------------------------------------------------------------
# Causal index expansion (expand_qsa_block_indices)
# ------------------------------------------------------------------
@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr, query_positions_ptr, sequence_lengths_ptr,
    token_to_req_ptr, output_ptr, stride_blocks_row, stride_blocks_column,
    stride_output_row, stride_output_column, rows, num_requests,
    BLOCK_TOPK: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr, OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    """Expand complete compressed groups and the causal tail of the open group."""
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests), other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ), BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start
    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row
        + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded, other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count) & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows) & (columns < OUTPUT_WIDTH) & (is_expanded | is_tail)
        & (token >= 0) & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )

def _validate_expand_inputs(
    block_indices: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_to_req: torch.Tensor,
    compress_ratio: int, token_topk: int, out: torch.Tensor | None,
) -> tuple[int, int]:
    _require_accelerator_tensors(
        block_indices, query_positions, sequence_lengths, token_to_req, out
    )
    tensors = (query_positions, sequence_lengths, token_to_req)
    if out is not None:
        tensors += (out,)
    _require_same_device(block_indices, *tensors)
    if not isinstance(compress_ratio, int) or compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be a positive integer.")
    if not isinstance(token_topk, int) or token_topk <= 0:
        raise ValueError("QSA token top-k must be a positive integer.")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio.")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError("QSA compressed top-k has an invalid shape.")
    if query_positions.shape != (block_indices.shape[0],):
        raise ValueError("QSA query positions must match compressed top-k rows.")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions.")
    if sequence_lengths.ndim != 1 or sequence_lengths.numel() == 0:
        raise ValueError("QSA request sequence lengths must be nonempty and rank 1.")
    for name, tensor in (
        ("compressed top-k", block_indices), ("query positions", query_positions),
        ("sequence lengths", sequence_lengths), ("request mapping", token_to_req),
    ):
        _require_int32(name, tensor)
        _require_last_dim_contiguous(name, tensor)
    if out is not None:
        if out.shape != (block_indices.shape[0], output_width):
            raise ValueError("QSA expansion output has an invalid shape.")
        if out.dtype != torch.int32:
            raise ValueError("QSA expansion output must be int32.")
        _require_last_dim_contiguous("expansion output", out)
    return block_topk, output_width

def expand_qsa_block_indices(
    block_indices: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_to_req: torch.Tensor,
    compress_ratio: int, token_topk: int, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected compressed QSA block ids into causal logical token ids."""
    block_topk, output_width = _validate_expand_inputs(
        block_indices, query_positions, sequence_lengths, token_to_req,
        compress_ratio, token_topk, out,
    )
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width), dtype=torch.int32,
            device=block_indices.device,
        )
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices, query_positions, sequence_lengths, token_to_req, out,
        block_indices.stride(0), block_indices.stride(1), out.stride(0),
        out.stride(1), block_indices.shape[0], sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk, COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk, OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block, num_warps=4,
    )
    return out


# ------------------------------------------------------------------
# Combined selection (qsa_select_paged_tokens)
# ------------------------------------------------------------------
_QSA_SELECTION_WORKSPACE_BYTES = 128 * 1024 * 1024

def _validate_select_inputs(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_topk: int, compress_ratio: int,
    out: torch.Tensor | None,
) -> tuple[int, int, int]:
    columns = _validate_mqa_inputs(
        q, k_cache, page_table, token_to_req, query_positions,
        sequence_lengths, compress_ratio, num_columns=None, score_scale=None,
    )
    if not isinstance(token_topk, int) or token_topk <= 0:
        raise ValueError("QSA token top-k must be a positive integer.")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio.")
    block_topk = token_topk // compress_ratio
    if q.shape[0] and block_topk > columns:
        raise ValueError("QSA token top-k exceeds available compressed blocks.")
    output_width = token_topk + compress_ratio - 1
    if out is not None:
        _require_accelerator_tensors(out)
        _require_same_device(q, out)
        if out.shape != (q.shape[0], output_width):
            raise ValueError("QSA selection output has an invalid shape.")
        if out.dtype != torch.int32:
            raise ValueError("QSA selection output must be int32.")
        _require_last_dim_contiguous("selection output", out)
    return columns, block_topk, output_width

def qsa_select_paged_tokens(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_topk: int, compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select visible compressed blocks and expand them into causal token ids."""
    columns, block_topk, output_width = _validate_select_inputs(
        q, k_cache, page_table, token_to_req, query_positions,
        sequence_lengths, token_topk, compress_ratio, out,
    )
    if out is None:
        out = torch.empty(
            (q.shape[0], output_width), dtype=torch.int32, device=q.device
        )
    if not q.shape[0]:
        return out

    # Keep the FP32 score workspace at or below 128 MiB. The selection mask
    # and top-k result are transient per chunk and no framework workspace is used.
    score_rows = max(1, _QSA_SELECTION_WORKSPACE_BYTES // (columns * 4))
    score_columns = torch.arange(columns, device=q.device).unsqueeze(0)
    for row_start in range(0, q.shape[0], score_rows):
        row_stop = min(row_start + score_rows, q.shape[0])
        scores, visible_blocks = qsa_mqa_paged(
            q[row_start:row_stop],
            k_cache,
            page_table,
            token_to_req[row_start:row_stop],
            query_positions[row_start:row_stop],
            sequence_lengths,
            compress_ratio,
        )
        scores.masked_fill_(
            score_columns >= visible_blocks.unsqueeze(1), -float("inf")
        )
        block_indices = torch.topk(
            scores, block_topk, dim=1, largest=True, sorted=False
        ).indices.to(torch.int32)
        expand_qsa_block_indices(
            block_indices,
            query_positions[row_start:row_stop],
            sequence_lengths,
            token_to_req[row_start:row_stop],
            compress_ratio,
            token_topk,
            out=out[row_start:row_stop],
        )
    return out


# ------------------------------------------------------------------
# Sparse paged GQA (qsa_sparse_paged_attention)
# ------------------------------------------------------------------
_TARGET_SHAPE = (1, 2_051, 256)

class SparseLaunchConfig(NamedTuple):
    block_n: int
    target_splits: int
    num_warps: int
    num_stages: int
    merge_warps: int
    merge_dim_tile: int

def _sparse_baseline_values(
    rows: int, kv_heads: int, block_m: int = 8,
) -> SparseLaunchConfig:
    base_programs = rows * kv_heads
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, num_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, num_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, num_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, num_warps = 64, 4, 2
    else:
        block_n, target_splits, num_warps = 64, 1, 2
    # Small decode batches are occupancy-bound and benefit from DIM_TILE=64.
    # Larger merges are bandwidth-bound; one warp and the full head minimize
    # repeated LSE reduction work.
    merge_dim_tile = 64 if base_programs <= 8 else 256
    return SparseLaunchConfig(
        block_n, target_splits, num_warps, 1, 1, merge_dim_tile
    )

def _sparse_config(
    values: SparseLaunchConfig,
) -> dict[str, int]:
    return values._asdict()

# Only measured non-baseline segments belong here.  Rows outside a segment use
# the source-equivalent baseline dispatch above.  Segment boundaries follow the
# measured decode/prefill crossover points for the target shape on HCU.
TUNED_TARGET_SEGMENTS: tuple[
    tuple[int, int, tuple[int, int, int, int, int, int]], ...
] = (
    (2, 2, (32, 32, 4, 1, 1, 256)),
    (3, 23, (32, 16, 2, 1, 1, 256)),
    (24, 256, (32, 1, 2, 1, 1, 256)),
    (257, 512, (16, 1, 2, 1, 1, 256)),
    (513, 262_144, (16, 1, 2, 1, 1, 256)),
)

def _tuned_sparse_values(
    rows: int,
) -> SparseLaunchConfig | None:
    for start, end, values in TUNED_TARGET_SEGMENTS:
        if start <= rows <= end:
            return SparseLaunchConfig(*values)
    return None

def _validate_sparse_config_shape(
    rows: int, kv_heads: int, topk: int, head_dim: int,
) -> None:
    if not all(isinstance(value, int) for value in (rows, kv_heads, topk, head_dim)):
        raise TypeError("QSA sparse configuration dimensions must be integers.")
    if rows < 1 or kv_heads < 1 or topk < 1 or head_dim < 1:
        raise ValueError("QSA sparse configuration dimensions must be positive.")

@lru_cache(maxsize=128)
def _select_qsa_sparse_config(
    rows: int, kv_heads: int, topk: int, head_dim: int,
    block_m: int = 8,
) -> dict[str, int]:
    _validate_sparse_config_shape(rows, kv_heads, topk, head_dim)
    tuned = (
        _tuned_sparse_values(rows)
        if (kv_heads, topk, head_dim) == _TARGET_SHAPE else None
    )
    values = tuned or _sparse_baseline_values(rows, kv_heads, block_m)
    return _sparse_config(values)

def _baseline_qsa_sparse_config(
    rows: int, kv_heads: int, topk: int, head_dim: int,
    block_m: int = 8,
) -> dict[str, int]:
    _validate_sparse_config_shape(rows, kv_heads, topk, head_dim)
    return _sparse_config(_sparse_baseline_values(rows, kv_heads, block_m))

@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, indices_ptr, block_table_ptr,
    token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token,
    stride_k_head, stride_v_block, stride_v_token, stride_v_head,
    stride_indices_row, stride_table_req, stride_output_row,
    stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:
    """Sparse paged GQA, ported from vLLM's Apache-2.0 QSA kernel."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
            & (columns < TOPK)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # The block-stride product can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scale scores so the query is not re-quantized after scaling.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + partial_row * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )

@triton.jit
def _qsa_sparse_paged_gqa_splitk_clamped_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, indices_ptr, block_table_ptr,
    token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token,
    stride_k_head, stride_v_block, stride_v_token, stride_v_head,
    stride_indices_row, stride_table_req, stride_output_row,
    stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:
    """Sparse paged GQA, ported from vLLM's Apache-2.0 QSA kernel."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Row extent: last valid index column + 1 (0 for empty rows).  The
    # split tile range is clamped to it so fully-masked tiles (all -1)
    # are skipped for short rows.  Scanning the row's index entries per
    # program keeps the launcher to a single Triton dispatch; the 8 KB
    # scan is L2-resident and negligible next to the tile work.
    extent_best = tl.full((), -1, dtype=tl.int32)
    for extent_start in range(0, TOPK, 1024):
        extent_cols = extent_start + tl.arange(0, 1024)
        extent_vals = tl.load(
            indices_ptr + row * stride_indices_row + extent_cols,
            mask=extent_cols < TOPK,
            other=-1,
        )
        extent_best = tl.maximum(
            extent_best,
            tl.max(tl.where(extent_vals >= 0, extent_cols, -1), axis=0),
        )
    row_extent = extent_best + 1
    n_useful_tiles = tl.minimum(
        tl.maximum((row_extent + BLOCK_N - 1) // BLOCK_N, 0), NUM_TILES
    )
    split_tile_start = tl.minimum(
        split_id * NUM_TILES // NUM_SPLITS, n_useful_tiles
    )
    split_tile_end = tl.minimum(
        (split_id + 1) * NUM_TILES // NUM_SPLITS, n_useful_tiles
    )
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
            & (columns < TOPK)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # The block-stride product can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scale scores so the query is not re-quantized after scaling.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + partial_row * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )

@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row,
    stride_output_head, num_rows, HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr, DIM_TILE: tl.constexpr,
) -> None:
    """Merge independently normalized QSA splits in FP32 log-sum-exp space."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim_tile_id = tl.program_id(2)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = dim_tile_id * DIM_TILE + tl.arange(0, DIM_TILE)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr
        + (split_offsets * num_rows + row) * NUM_QUERY_HEADS
        + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr
        + row * stride_output_row
        + head * stride_output_head
        + dim_offsets,
        merged,
    )

def _validate_sparse_inputs(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    logical_indices: torch.Tensor, block_table: torch.Tensor,
    token_to_req: torch.Tensor, out: torch.Tensor | None,
) -> None:
    if not q.is_cuda:
        raise ValueError("QSA operators require CUDA/HIP tensors.")
    device = q.device
    if (
        k_cache.device != device or v_cache.device != device
        or logical_indices.device != device or block_table.device != device
        or token_to_req.device != device
        or (out is not None and out.device != device)
    ):
        raise ValueError("QSA tensors must all use the same device.")
    if q.ndim != 3 or q.shape[1] <= 0:
        raise ValueError("QSA sparse query must be [rows, heads, head_dim].")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse K/V caches must have matching rank-4 shapes.")
    if (
        not all(k_cache.shape[:3]) or block_table.ndim != 2
        or not all(block_table.shape)
    ):
        raise ValueError("QSA sparse cache and block table must be nonempty.")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA sparse indices must have one row per query.")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width.")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA sparse request mapping must match query rows.")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("QSA sparse query and cache head dimensions must match.")
    if q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads.")
    if q.shape[2] < 16 or q.shape[2] & (q.shape[2] - 1):
        raise ValueError(
            "QSA sparse head dimension must be a power of two at least 16."
        )
    if q.dtype is not torch.bfloat16:
        raise ValueError("QSA sparse query must be bfloat16.")
    if k_cache.dtype is not torch.bfloat16 or v_cache.dtype is not torch.bfloat16:
        raise ValueError("QSA sparse K/V caches must be bfloat16.")
    for name, tensor in (
        ("indices", logical_indices), ("block table", block_table),
        ("request mapping", token_to_req),
    ):
        if tensor.dtype is not torch.int32:
            raise ValueError(f"QSA {name} must be int32.")
        if tensor.stride(-1) != 1:
            raise ValueError(f"QSA {name} must have a contiguous last dimension.")
    for name, tensor in (
        ("query", q), ("key cache", k_cache), ("value cache", v_cache),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(f"QSA {name} must have a contiguous last dimension.")
    if out is not None:
        if out.shape != q.shape:
            raise ValueError("QSA sparse output must match its query.")
        if out.dtype is not torch.bfloat16:
            raise ValueError("QSA sparse output must be bfloat16.")
        if out.stride(-1) != 1:
            raise ValueError("QSA sparse output must have a contiguous last dimension.")

def _select_qsa_sparse_block_m(group_size: int, head_dim: int) -> int:
    """Choose a matrix-core-compatible M tile for the measured HCU GQA shape."""
    # HCU exposes 16x16 BF16 MMAC.  GROUP_SIZE=6 otherwise rounds to 8,
    # which makes both QK (8xN) and PV (8xD) dots fail the MMAC legality check.
    if torch.version.hip is not None and group_size == 6 and head_dim == 256:
        return 16
    return triton.next_power_of_2(group_size)

def _select_sparse_launch_config(
    rows: int, kv_heads: int, topk: int, head_dim: int, group_size: int,
) -> dict[str, int]:
    """Select the launch policy used by the public sparse attention API."""
    block_m = _select_qsa_sparse_block_m(group_size, head_dim)
    return _select_qsa_sparse_config(
        rows, kv_heads, topk, head_dim, block_m
    )

def _launch_qsa_sparse_paged_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    logical_indices: torch.Tensor, block_table: torch.Tensor,
    token_to_req: torch.Tensor, config: dict[str, int], out: torch.Tensor | None,
) -> torch.Tensor:
    """Launch validated sparse QSA with an explicit static configuration."""
    if out is None:
        out = torch.empty_like(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = _select_qsa_sparse_block_m(group_size, q.shape[2])
    block_n = config["block_n"]
    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, config["target_splits"])
    partial_warps = config["num_warps"]
    # The static table is calibrated for HIP.  CUDA retains the upstream
    # source stage so an unmeasured non-target path stays source-equivalent.
    partial_stages = config["num_stages"] if torch.version.hip is not None else 2

    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # Split-K partials remain FP32 until the final normalized merge.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    if q.shape[0] >= 3:
        # From three rows on, skipping fully-masked tiles outweighs their
        # in-kernel extent scan; single/double-token decode keeps the
        # constant-bound kernel.
        _qsa_sparse_paged_gqa_splitk_clamped_kernel[
            (q.shape[0], k_cache.shape[2], num_splits)
        ](
            q, k_cache, v_cache, logical_indices, block_table, token_to_req,
            partial_output, partial_lse, out, q.stride(0),
            q.stride(1), k_cache.stride(0), k_cache.stride(1),
            k_cache.stride(2), v_cache.stride(0), v_cache.stride(1),
            v_cache.stride(2), logical_indices.stride(0),
            block_table.stride(0), out.stride(0), out.stride(1), q.shape[0],
            k_cache.shape[0], block_table.shape[0],
            TOPK=logical_indices.shape[1], PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits, NUM_TILES=num_tiles, BLOCK_M=block_m,
            BLOCK_N=block_n, num_warps=partial_warps,
            num_stages=partial_stages,
        )
    else:
        _qsa_sparse_paged_gqa_splitk_kernel[
            (q.shape[0], k_cache.shape[2], num_splits)
        ](
            q, k_cache, v_cache, logical_indices, block_table, token_to_req,
            partial_output, partial_lse, out, q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            logical_indices.stride(0), block_table.stride(0), out.stride(0),
            out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
            TOPK=logical_indices.shape[1], PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=partial_warps, num_stages=partial_stages,
        )
    if num_splits == 1:
        return out

    merge_dim_tile = config["merge_dim_tile"]
    if (
        merge_dim_tile <= 0
        or merge_dim_tile > q.shape[2]
        or q.shape[2] % merge_dim_tile
    ):
        # Non-divisor tiles keep the full-head source layout.
        merge_dim_tile = q.shape[2]
    _qsa_merge_splitk_kernel[
        (q.shape[0], q.shape[1], q.shape[2] // merge_dim_tile)
    ](
        partial_output, partial_lse, out, out.stride(0), out.stride(1),
        q.shape[0], HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits, BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        DIM_TILE=merge_dim_tile, num_warps=config["merge_warps"], num_stages=1,
    )
    return out

def _qsa_sparse_paged_attention_with_config(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    logical_indices: torch.Tensor, block_table: torch.Tensor,
    token_to_req: torch.Tensor, config: dict[str, int], out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse QSA using a caller-supplied config for benchmark dispatch."""
    return _launch_qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, config, out
    )

def qsa_sparse_paged_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    logical_indices: torch.Tensor, block_table: torch.Tensor,
    token_to_req: torch.Tensor, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse grouped-query attention over paged BF16 K/V caches."""
    _validate_sparse_inputs(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
    )
    if not q.shape[0]:
        return out if out is not None else torch.empty_like(q)

    config = _select_sparse_launch_config(
        q.shape[0], k_cache.shape[2], logical_indices.shape[1], q.shape[2],
        q.shape[1] // k_cache.shape[2],
    )
    return _launch_qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, config, out
    )

