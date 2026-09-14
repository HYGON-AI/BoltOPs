"""Step-4 Dynamic Sparse Attention (DSA) operator package.

Public signature and ship-time backend selection live here (not a runtime
``backend=`` switch). Change the lazy-import target below after Triton /
TileLang bake-off.

Public APIs follow the handoff vendor contract ``step4-dsa-ops-v1``
(aligned with the Step-4 / vLLM overlay numerical surface):

* docs: ``docs/ops/step4_dsa.md``
* tests: ``test/step4_dsa/``

This module is a valid ``--ops-module`` for the original handoff parity suite.
"""

from __future__ import annotations

from typing import Optional

import torch

STEP4_DSA_OPS_API_VERSION = "step4-dsa-ops-v1"
CORE_DSA_CAPABILITY = "core_dsa_v1"
LOGICAL_PREFILL_IDS_CAPABILITY = "logical_prefill_ids_v1"
TILED_PREFILL_CAPABILITY = "tiled_prefill_v1"
STEP4_DSA_CAPABILITIES = {
    CORE_DSA_CAPABILITY,
    LOGICAL_PREFILL_IDS_CAPABILITY,
    TILED_PREFILL_CAPABILITY,
}


def triton_build_rope_cache(
    *,
    rotary_span: int,
    theta: float,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaling: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX-style partial RoPE cos/sin tables ``[max_position, rotary_span // 2]``."""
    from boltops.step4_dsa.triton.norm_rope import build_rope_cache as impl

    return impl(
        rotary_span=rotary_span,
        theta=theta,
        max_position=max_position,
        device=device,
        dtype=dtype,
        scaling=scaling,
    )


def triton_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused indexer RMS/LayerNorm + partial RoPE; ``index_z`` is a passthrough."""
    from boltops.step4_dsa.triton.norm_rope import indexer_norm_rope as impl

    return impl(
        index_q,
        index_k,
        index_z,
        q_norm_weight,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        positions,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=rotary_dim,
        eps=eps,
        q_norm_weight_bias=q_norm_weight_bias,
    )


def triton_csa_region_layout(
    seq_lens: torch.Tensor,
    *,
    regions_per_seq: int,
    region_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-region token start/count for sequences packed back to back."""
    from boltops.step4_dsa.triton.csa import csa_region_layout as impl

    return impl(
        seq_lens, regions_per_seq=regions_per_seq, region_size=region_size
    )


def triton_csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    *,
    region_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress packed tokens into FP32 and FP8 e4m3 region summaries."""
    from boltops.step4_dsa.triton.csa import csa_compress_regions as impl

    return impl(
        index_k,
        index_z,
        token_start,
        token_count,
        region_size=region_size,
    )


def triton_csa_prefill(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    regions_per_request: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layout + compress convenience wrapper (not required by the core contract)."""
    from boltops.step4_dsa.triton.csa import csa_prefill as impl

    return impl(
        index_k,
        index_z,
        seq_lens,
        regions_per_request=regions_per_request,
        region_size=region_size,
    )


def triton_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores, FP32 ``[groups * Q, regions]`` group-major."""
    from boltops.step4_dsa.triton.logits import indexer_logits as impl

    return impl(index_q, weights, index_k)


def triton_region_topk_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Signed top-k over visible regions, returning logical ids (ascending)."""
    from boltops.step4_dsa.triton.topk import region_topk_ids as impl

    return impl(logits, lengths, topk=topk)


def triton_topk_select_and_pack(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: Optional[torch.Tensor] = None,
    *,
    topk: int,
    region_size: int = 8,
    regions_per_page: int,
) -> torch.Tensor:
    """Fused top-k selection and prefill packing (physical ids)."""
    from boltops.step4_dsa.triton.topk import region_topk_pack as impl

    return impl(
        logits,
        lengths,
        q_positions,
        block_table,
        request_indices,
        topk=topk,
        region_size=region_size,
        regions_per_page=regions_per_page,
    )


def triton_prefill_sparse_meta(
    logits: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: Optional[torch.Tensor] = None,
    *,
    topk: int,
    region_size: int = 8,
    regions_per_page: int,
    return_logical_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill sparse metadata of width ``topk + 1`` (current region appended)."""
    from boltops.step4_dsa.triton.metadata import prefill_sparse_meta as impl

    return impl(
        logits,
        q_positions,
        block_table,
        request_indices,
        topk=topk,
        region_size=region_size,
        regions_per_page=regions_per_page,
        return_logical_ids=return_logical_ids,
    )


def triton_decode_sparse_meta(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: Optional[torch.Tensor] = None,
    *,
    topk: int,
    region_size: int = 8,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed int64 metadata of width ``topk + 1``."""
    from boltops.step4_dsa.triton.metadata import decode_sparse_meta as impl

    return impl(
        logits,
        kv_seqlens,
        block_table,
        request_indices,
        topk=topk,
        region_size=region_size,
        regions_per_page=regions_per_page,
    )


def triton_sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    block_regions: int = 16,
    logical_regions: Optional[torch.Tensor] = None,
    q_positions: Optional[torch.Tensor] = None,
    request_indices: Optional[torch.Tensor] = None,
    block_q: int = 128,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse GQA prefill. Tiled path needs logical ids, positions, and request ids.

    Q / K / V / ``out`` last dim must be contiguous (``stride(-1) == 1``).
    """
    from boltops.step4_dsa.triton.prefill import sparse_attention_prefill as impl

    return impl(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
        block_regions=block_regions,
        logical_regions=logical_regions,
        q_positions=q_positions,
        request_indices=request_indices,
        block_q=block_q,
        config=config,
    )


def triton_sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = 8,
    softmax_scale: Optional[float] = None,
    num_splits: int = 1,
    block_regions: int = 16,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse GQA decode. ``num_splits==1`` auto KV-splits to fill CUs and returns
    merged states. ``num_splits > 1`` returns partial states for the caller to merge.

    Same calling convention as vLLM ``Step4DSAImpl`` / handoff
    ``sparse_attention_decode``. Q / K / V / ``out`` last dim must be contiguous
    (``stride(-1) == 1``)."""
    from boltops.step4_dsa.triton.decode import sparse_attention_decode as impl

    return impl(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
        num_splits=num_splits,
        block_regions=block_regions,
        config=config,
    )


def triton_merge_split_states(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge split-KV partial outputs by softmax over partial LSE.

    ``partial_out`` / ``out`` last dim must be contiguous (``stride(-1) == 1``).
    """
    from boltops.step4_dsa.triton.merge import merge_split_states as impl

    return impl(partial_out, partial_lse)


__all__ = [
    "CORE_DSA_CAPABILITY",
    "LOGICAL_PREFILL_IDS_CAPABILITY",
    "STEP4_DSA_CAPABILITIES",
    "STEP4_DSA_OPS_API_VERSION",
    "TILED_PREFILL_CAPABILITY",
    "triton_build_rope_cache",
    "triton_csa_compress_regions",
    "triton_csa_prefill",
    "triton_csa_region_layout",
    "triton_decode_sparse_meta",
    "triton_indexer_logits",
    "triton_indexer_norm_rope",
    "triton_merge_split_states",
    "triton_prefill_sparse_meta",
    "triton_region_topk_ids",
    "triton_sparse_attention_decode",
    "triton_sparse_attention_prefill",
    "triton_topk_select_and_pack",
]
