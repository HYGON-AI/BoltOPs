"""Local test-only replacements for native reference helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from boltops.utility import logger
from boltops.tools.triton_moe_tune.routing import moe_fused_gate


def silu_and_mul(output: torch.Tensor, input: torch.Tensor):
    gate, up = input.chunk(2, dim=-1)
    output.copy_(F.silu(gate) * up)
    return output


def gelu_and_mul(output: torch.Tensor, input: torch.Tensor):
    gate, up = input.chunk(2, dim=-1)
    output.copy_(F.gelu(gate, approximate="none") * up)
    return output


def moe_sum(output: torch.Tensor, input: torch.Tensor):
    output.copy_(input.sum(dim=1))
    return output


__all__ = ["gelu_and_mul", "logger", "moe_fused_gate", "moe_sum", "silu_and_mul"]

