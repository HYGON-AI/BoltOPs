"""Smoke tests for package import and public API facades."""

from __future__ import annotations

from boltops import __version__
from boltops.fused_moe import (
    fused_experts,
    moe_activation,
    moe_activation_output_size,
    normalize_moe_activation,
)
from boltops.generic import (
    attn_res,
    fused_mamba_state_scatter_with_mask,
)
from boltops.qsa import (
    expand_qsa_block_indices,
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
from boltops.fla.kda import recompute_w_u_fwd as recompute_w_u_fwd_kda
from boltops.fla.gdn import recompute_w_u_fwd as recompute_w_u_fwd_gdn
from boltops.mqa_logits import (
    tilelang_qsa_mqa_decode,
    tilelang_qsa_mqa_prefill,
    triton_mqa_logits,
)
from boltops.gemm import (
    awq_gemm_triton,
    batched_gemm_a16wfp4,
    batched_gemm_bf16,
    gemm_a16w16,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    matmul_persistent,
    triton_scaled_mm,
    w8a8_block_int8_matmul,
    w8a8_triton_block_scaled_mm,
)
from boltops.mla import decode_attention_fwd, compute_num_kv_splits
from boltops.fa.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
from boltops.fa.extend_attention import (
    build_unified_kv_indices,
    can_use_dense_prefill_fp8,
    dense_prefill_attention_fwd,
    extend_attention_fwd,
    extend_attention_fwd_unified,
)
from boltops.ihc import ihc_head, ihc_post, ihc_pre
from boltops.step4_dsa import (
    triton_build_rope_cache,
    triton_csa_compress_regions,
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
from boltops.mhc import (
    hc_split_sinkhorn,
    mhc_fused_tilelang,
    mhc_post_fwd,
    mhc_pre_big_fuse,
    pre_big_fuse_tilelang,
)
from boltops.tools.fused_moe_triton_tune.tune_moe_cli import main as moe_tune_main
from boltops.tools.sparse_gqa_triton_tune.tune_sparse_gqa import (
    main as sparse_gqa_tune_main,
)
from boltops.tools.extend_attention_triton_tune.tune_extend_attention import (
    main as extend_attention_tune_main,
)


def test_version():
    assert __version__ == "0.1.0"


def test_operator_package_exports():
    assert callable(attn_res)
    assert callable(triton_mqa_logits)
    assert callable(qsa_mqa_paged)
    assert callable(qsa_select_paged_tokens)
    assert callable(expand_qsa_block_indices)
    assert callable(qsa_sparse_paged_attention)
    assert callable(tilelang_qsa_mqa_decode)
    assert callable(tilelang_qsa_mqa_prefill)
    assert callable(fused_mamba_state_scatter_with_mask)
    assert callable(recompute_w_u_fwd_kda)
    assert callable(recompute_w_u_fwd_gdn)
    assert callable(fused_experts)
    assert callable(matmul_persistent)
    assert callable(gemm_a16w16)
    assert callable(w8a8_triton_block_scaled_mm)
    assert callable(triton_scaled_mm)
    assert callable(awq_gemm_triton)
    assert callable(gemm_a8w8_blockscale)
    assert callable(gemm_afp4wfp4)
    assert callable(batched_gemm_a16wfp4)
    assert callable(batched_gemm_bf16)
    assert callable(w8a8_block_int8_matmul)
    assert callable(moe_activation)
    assert callable(moe_activation_output_size)
    assert callable(normalize_moe_activation)
    assert callable(hc_split_sinkhorn)
    assert callable(mhc_fused_tilelang)
    assert callable(mhc_post_fwd)
    assert callable(mhc_pre_big_fuse)
    assert callable(pre_big_fuse_tilelang)
    assert callable(ihc_pre)
    assert callable(ihc_post)
    assert callable(ihc_head)
    assert callable(decode_attention_fwd)
    assert callable(compute_num_kv_splits)
    assert callable(qwen_sparse_kv_extraction_compact_hcu_fa_triton)
    assert callable(sparse_gqa_packed_decode_triton)
    assert callable(build_unified_kv_indices)
    assert callable(can_use_dense_prefill_fp8)
    assert callable(dense_prefill_attention_fwd)
    assert callable(extend_attention_fwd)
    assert callable(extend_attention_fwd_unified)
    assert callable(triton_build_rope_cache)
    assert callable(triton_indexer_norm_rope)
    assert callable(triton_csa_region_layout)
    assert callable(triton_csa_compress_regions)
    assert callable(triton_indexer_logits)
    assert callable(triton_region_topk_ids)
    assert callable(triton_topk_select_and_pack)
    assert callable(triton_prefill_sparse_meta)
    assert callable(triton_decode_sparse_meta)
    assert callable(triton_sparse_attention_prefill)
    assert callable(triton_sparse_attention_decode)
    assert callable(triton_merge_split_states)


def test_root_package_does_not_reexport_operators():
    import boltops

    assert boltops.__all__ == ["__version__"]
    for name in (
        "attn_res",
        "fused_mamba_state_scatter_with_mask",
        "fused_experts",
        "moe_activation",
        "matmul_persistent",
        "gemm_a16w16",
        "triton_indexer_logits",
    ):
        assert not hasattr(boltops, name)


def test_gemm_fp16_not_public():
    import boltops.gemm as gemm

    assert not hasattr(gemm, "gemm_fp16")


def test_moe_tune_cli_is_packaged():
    assert callable(moe_tune_main)


def test_sparse_gqa_tune_cli_is_packaged():
    assert callable(sparse_gqa_tune_main)


def test_extend_attention_tune_cli_is_packaged():
    assert callable(extend_attention_tune_main)
