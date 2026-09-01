"""CPU-safe contract tests for the Step-4 DSA public facade."""

from __future__ import annotations

from boltops.step4_dsa import (
    CORE_DSA_CAPABILITY,
    LOGICAL_PREFILL_IDS_CAPABILITY,
    STEP4_DSA_CAPABILITIES,
    STEP4_DSA_OPS_API_VERSION,
    TILED_PREFILL_CAPABILITY,
    triton_build_rope_cache,
    triton_csa_compress_regions,
    triton_csa_prefill,
    triton_csa_region_layout,
    triton_decode_sparse_meta,
    triton_indexer_logits,
    triton_indexer_norm_rope,
    triton_merge_split_states,
    triton_prefill_sparse_meta,
    triton_region_topk_ids,
    triton_sparse_attention_decode,
    triton_sparse_attention_prefill,
    triton_topk_select_and_pack,
)

CORE_SYMBOLS = (
    "triton_build_rope_cache",
    "triton_indexer_norm_rope",
    "triton_csa_region_layout",
    "triton_csa_compress_regions",
    "triton_indexer_logits",
    "triton_region_topk_ids",
    "triton_topk_select_and_pack",
    "triton_prefill_sparse_meta",
    "triton_decode_sparse_meta",
    "triton_sparse_attention_prefill",
    "triton_sparse_attention_decode",
    "triton_merge_split_states",
)


def test_vendor_contract_constants():
    assert STEP4_DSA_OPS_API_VERSION == "step4-dsa-ops-v1"
    assert CORE_DSA_CAPABILITY in STEP4_DSA_CAPABILITIES
    assert LOGICAL_PREFILL_IDS_CAPABILITY in STEP4_DSA_CAPABILITIES
    assert TILED_PREFILL_CAPABILITY in STEP4_DSA_CAPABILITIES


def test_core_symbols_are_callable():
    namespace = {
        "triton_build_rope_cache": triton_build_rope_cache,
        "triton_indexer_norm_rope": triton_indexer_norm_rope,
        "triton_csa_region_layout": triton_csa_region_layout,
        "triton_csa_compress_regions": triton_csa_compress_regions,
        "triton_indexer_logits": triton_indexer_logits,
        "triton_region_topk_ids": triton_region_topk_ids,
        "triton_topk_select_and_pack": triton_topk_select_and_pack,
        "triton_prefill_sparse_meta": triton_prefill_sparse_meta,
        "triton_decode_sparse_meta": triton_decode_sparse_meta,
        "triton_sparse_attention_prefill": triton_sparse_attention_prefill,
        "triton_sparse_attention_decode": triton_sparse_attention_decode,
        "triton_merge_split_states": triton_merge_split_states,
    }
    for name in CORE_SYMBOLS:
        assert callable(namespace[name])
    assert callable(triton_csa_prefill)
