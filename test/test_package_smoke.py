"""Smoke tests for package import and public API facades."""

from __future__ import annotations

from bolt_ops import (
    __version__,
    fused_experts,
    fused_mamba_state_scatter_with_mask,
    moe_activation,
    recompute_w_u_fwd,
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
    assert callable(moe_activation)
    assert callable(moe_activation_output_size)
    assert callable(normalize_moe_activation)
    assert fused_moe_activation is moe_activation


def test_moe_tune_cli_is_packaged():
    assert callable(moe_tune_main)
