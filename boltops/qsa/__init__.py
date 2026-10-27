# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""QSA (query-specific attention) operator package.

Framework-agnostic paged QSA data flow: compressed-key scoring, compressed
block selection, causal index expansion, and sparse paged GQA over the main
K/V cache. Callers own cache allocation and writes, page tables, and request
mappings; BoltOPs does not manage framework scheduling or cache lifecycles.

* docs: ``docs/ops/qsa.md``
* tests: ``test/qsa/``

Public operators:

* ``qsa_mqa_paged`` — FP32 QSA scores over a paged compressed-key cache
* ``qsa_select_paged_tokens`` — score, top-k compressed blocks, causal expand
* ``expand_qsa_block_indices`` — expand caller-selected compressed blocks
* ``qsa_sparse_paged_attention`` — sparse grouped-query attention over
  selected logical tokens

TileLang QSA scoring entries (``tilelang_qsa_mqa_prefill`` /
``tilelang_qsa_mqa_decode``) are implemented and maintained in the
``boltops.mqa_logits`` package; see ``docs/ops/qsa.md`` (scoring backends).
"""

from __future__ import annotations

import torch


def qsa_mqa_paged(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, compress_ratio: int,
    num_columns: int | None = None, score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 QSA scores from a paged compressed-key cache."""
    from .triton.qsa import qsa_mqa_paged as impl

    return impl(
        q, k_cache, page_table, token_to_req, query_positions,
        sequence_lengths, compress_ratio,
        num_columns=num_columns, score_scale=score_scale,
    )


def qsa_select_paged_tokens(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor,
    token_to_req: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_topk: int, compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select visible compressed blocks and expand them into causal token ids."""
    from .triton.qsa import qsa_select_paged_tokens as impl

    return impl(
        q, k_cache, page_table, token_to_req, query_positions,
        sequence_lengths, token_topk, compress_ratio, out=out,
    )


def expand_qsa_block_indices(
    block_indices: torch.Tensor, query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor, token_to_req: torch.Tensor,
    compress_ratio: int, token_topk: int, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected compressed QSA block ids into causal logical token ids."""
    from .triton.qsa import expand_qsa_block_indices as impl

    return impl(
        block_indices, query_positions, sequence_lengths, token_to_req,
        compress_ratio, token_topk, out=out,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    logical_indices: torch.Tensor, block_table: torch.Tensor,
    token_to_req: torch.Tensor, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse grouped-query attention over paged BF16 K/V caches."""
    from .triton.qsa import qsa_sparse_paged_attention as impl

    return impl(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        out=out,
    )

__all__ = [
    "expand_qsa_block_indices",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
]
