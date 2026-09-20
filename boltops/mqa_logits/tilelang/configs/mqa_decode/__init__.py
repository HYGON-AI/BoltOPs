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


__all__ = ["QSAMQADecodeConfig", "resolve_get_config"]
