# SPDX-License-Identifier: MIT
"""Architecture-aware configuration dispatch for TileLang MQA prefill."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Optional, Protocol

from boltops.mqa_logits.tilelang._arch import targets_for_arch

from .common import QSAMQAPrefillConfig


class GetConfig(Protocol):
    def __call__(
        self,
        num_rows: int,
        num_keys: int,
        cu_count: int,
        effective_range: Optional[int] = None,
        num_heads: int = 4,
    ) -> QSAMQAPrefillConfig:
        """``effective_range`` is the longest valid key span; ``None`` uses ``num_keys``."""
        ...


@lru_cache(maxsize=None)
def resolve_get_config(arch: str) -> GetConfig:
    """Resolve exact, family, then common configuration for an architecture."""
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = (
            f"boltops.mqa_logits.tilelang.configs.mqa_prefill.{target}"
        )
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return import_module(qualified_name).get_config

    raise RuntimeError(
        f"No QSA MQA prefill config module found for architecture {arch}; "
        f"tried targets {targets}"
    )


__all__ = ["QSAMQAPrefillConfig", "resolve_get_config"]
