# SPDX-License-Identifier: MIT
"""Shared data model for TileLang MQA prefill launch configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QSAMQAPrefillConfig:
    """Compile-time parameters for the gfx938 12-wave prefill kernel."""

    block_q: int
    block_n: int
    threads: int
    n_tile_phases: int
    split_n: int
    rows_divisible: bool
    heads: int = 4

    @property
    def logical_m(self) -> int:
        """Logical GEMM M covered by one CTA."""
        return self.block_q * self.heads
