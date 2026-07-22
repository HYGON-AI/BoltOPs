"""Smoke tests for package import and public API facades."""

from __future__ import annotations

from bolt_ops import (
    __version__,
    fused_experts,
    fused_mamba_state_scatter_with_mask,
    recompute_w_u_fwd,
)
from bolt_ops.tools.triton_moe_tune.tune_moe_cli import main as moe_tune_main


def test_version():
    assert __version__ == "0.1.0"


def test_public_api_exports():
    assert callable(recompute_w_u_fwd)
    assert callable(fused_mamba_state_scatter_with_mask)
    assert callable(fused_experts)


def test_moe_tune_cli_is_packaged():
    assert callable(moe_tune_main)
