# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Separate QSA BF16/FP8 and weighted BF16 policies for TileLang MQA prefill."""

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


def _load_prefill_config_module(arch: str):
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
            return import_module(qualified_name)
    raise RuntimeError(
        f"No QSA MQA prefill config module found for architecture {arch}; "
        f"tried targets {targets}"
    )


@lru_cache(maxsize=None)
def resolve_get_config(arch: str) -> GetConfig:
    """QSA BF16 WASP launch policy; also the weighted policy baseline."""
    return _load_prefill_config_module(arch).get_config


@lru_cache(maxsize=None)
def resolve_get_config_weighted(arch: str) -> GetConfig:
    """Weighted BF16 policy; reuse BF16 defaults without an arch overlay."""
    module = _load_prefill_config_module(arch)
    return getattr(module, "get_config_weighted", module.get_config)


@lru_cache(maxsize=None)
def resolve_get_config_fp8(arch: str) -> GetConfig:
    """FP8 WASP launch policy; falls back to BF16 if an arch has no overlay."""
    module = _load_prefill_config_module(arch)
    return getattr(module, "get_config_fp8", module.get_config)


__all__ = [
    "QSAMQAPrefillConfig",
    "resolve_get_config",
    "resolve_get_config_weighted",
    "resolve_get_config_fp8",
]
