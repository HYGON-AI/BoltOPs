"""Smoke tests for package import and public API facades."""

from __future__ import annotations

from bolt_ops import __version__
from bolt_ops.fused_moe import (
    fused_experts,
    moe_activation,
    moe_activation_output_size,
    normalize_moe_activation,
)
from bolt_ops.generic import (
    attn_res,
    fused_mamba_state_scatter_with_mask,
    recompute_w_u_fwd,
)
from bolt_ops.gemm import (
    awq_gemm_triton,
    batched_gemm_a16wfp4,
    gemm_a16w16,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    matmul_persistent,
    triton_scaled_mm,
    w8a8_block_int8_matmul,
    w8a8_triton_block_scaled_mm,
)
from bolt_ops.tools.triton_moe_tune.tune_moe_cli import main as moe_tune_main


def test_version():
    assert __version__ == "0.1.0"


def test_operator_package_exports():
    assert callable(attn_res)
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


def test_root_package_does_not_reexport_operators():
    import bolt_ops

    assert bolt_ops.__all__ == ["__version__"]
    for name in (
        "attn_res",
        "recompute_w_u_fwd",
        "fused_mamba_state_scatter_with_mask",
        "fused_experts",
        "moe_activation",
        "matmul_persistent",
        "gemm_a16w16",
    ):
        assert not hasattr(bolt_ops, name)


def test_gemm_fp16_not_public():
    import bolt_ops.gemm as gemm

    assert not hasattr(gemm, "gemm_fp16")


def test_moe_tune_cli_is_packaged():
    assert callable(moe_tune_main)

