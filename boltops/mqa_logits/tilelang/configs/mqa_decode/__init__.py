# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Architecture-aware configuration dispatch for TileLang MQA decode."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Protocol

from boltops.mqa_logits.tilelang._arch import targets_for_arch

from .common import QSAMQADecodeConfig


class GetConfig(Protocol):
    def __call__(
        self,
        batch: int,
        max_model_len: int,
        cu_count: int,
        num_heads: int,
    ) -> QSAMQADecodeConfig: ...


@lru_cache(maxsize=None)
def resolve_get_config(arch: str) -> GetConfig:
    """Resolve exact, family, then common configuration for an architecture."""
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = f"boltops.mqa_logits.tilelang.configs.mqa_decode.{target}"
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return import_module(qualified_name).get_config

    raise RuntimeError(
        f"No QSA MQA decode config module found for architecture {arch}; "
        f"tried targets {targets}"
    )


@lru_cache(maxsize=None)
def resolve_get_config_fp8(arch: str) -> GetConfig:
    """FP8 occupancy policy.  Missing overlay is an error, not a BF16 fallback."""
    targets = targets_for_arch(arch)
    tried = []
    for target in targets:
        qualified_name = f"boltops.mqa_logits.tilelang.configs.mqa_decode.{target}"
        tried.append(qualified_name)
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            continue
        module = import_module(qualified_name)
        getter = getattr(module, "get_config_fp8", None)
        if getter is not None:
            return getter

    raise RuntimeError(
        f"FP8 QSA MQA decode config is not available for architecture {arch}; "
        f"tried {tried}"
    )


__all__ = ["QSAMQADecodeConfig", "resolve_get_config", "resolve_get_config_fp8"]
