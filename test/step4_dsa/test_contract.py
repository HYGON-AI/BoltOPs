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


def test_config_sku_aliases_board_renames():
    from boltops.step4_dsa.triton.common import (
        _canonical_config_sku,
        _config_sku_candidates,
    )

    assert _canonical_config_sku("BW1101") == "BW1100"
    assert _canonical_config_sku("BW1100") == "BW1100"
    assert _canonical_config_sku("Hygon BW1101") == "BW1100"
    assert _canonical_config_sku("BW200, UBB BW1000") == "BW200"
    assert _config_sku_candidates("BW1101") == ("BW1100", "BW1101")
    assert _config_sku_candidates("BW1100") == ("BW1100",)


def test_occupancy_json_loaders_alias_bw1101_to_bw1100():
    from boltops.step4_dsa.triton.decode import _load_decode_device_configs
    from boltops.step4_dsa.triton.prefill import (
        _load_prefill_device_configs,
        _load_prefill_tiled_device_configs,
    )

    for loader in (
        _load_decode_device_configs,
        _load_prefill_device_configs,
        _load_prefill_tiled_device_configs,
    ):
        loader.cache_clear()
        aliased = loader("BW1101")
        canonical = loader("BW1100")
        loader.cache_clear()
        assert canonical.get("configs")
        assert aliased == canonical
