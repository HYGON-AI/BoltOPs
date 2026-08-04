"""Smoke tests for package import and public API facades."""

from __future__ import annotations

from bolt_ops import (
    __version__,
    awq_gemm_triton,
    batched_gemm_a16wfp4,
    fused_experts,
    fused_mamba_state_scatter_with_mask,
    gemm_a16w16,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    matmul_persistent,
    moe_activation,
    recompute_w_u_fwd,
    triton_scaled_mm,
    w8a8_block_int8_matmul,
    w8a8_triton_block_scaled_mm,
)
from bolt_ops.tools.triton_moe_tune.tune_moe_cli import main as moe_tune_main
from bolt_ops.fused_moe import (
    moe_activation as fused_moe_activation,
    moe_activation_output_size,
    normalize_moe_activation,
)


def test_version():
    assert __version__ == "0.1.0"


def test_public_api_exports():
    assert callable(recompute_w_u_fwd)
    assert callable(fused_mamba_state_scatter_with_mask)
    assert callable(fused_experts)
    assert callable(matmul_persistent)
    assert callable(gemm_a16w16)
    assert callable(w8a8_triton_block_scaled_mm)
    assert callable(triton_scaled_mm)
    assert callable(awq_gemm_triton)
    assert callable(gemm_a8w8_blockscale)
    assert callable(gemm_afp4wfp4)
    assert callable(batched_gemm_a16wfp4)
    assert callable(w8a8_block_int8_matmul)
    assert callable(moe_activation)
    assert callable(moe_activation_output_size)
    assert callable(normalize_moe_activation)
    assert fused_moe_activation is moe_activation


def test_gemm_fp16_not_public():
    import bolt_ops

    assert not hasattr(bolt_ops, "gemm_fp16")


def test_moe_tune_cli_is_packaged():
    assert callable(moe_tune_main)
