# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Architecture-aware configuration dispatch for ``ihc_pre``."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

from boltops.ihc.tilelang._arch import targets_for_arch

from .common import BLOCK_N, PreConfig, get_config

GetConfig = Callable[..., PreConfig]


@lru_cache(maxsize=None)
def resolve_get_config(arch: str) -> GetConfig:
    """Resolve exact, family, then common configuration for an architecture."""
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = f"boltops.ihc.tilelang.configs.pre.{target}"
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return import_module(qualified_name).get_config

    raise RuntimeError(
        f"No ihc_pre config module found for architecture {arch}; "
        f"tried targets {targets}"
    )


__all__ = ["BLOCK_N", "PreConfig", "GetConfig", "get_config", "resolve_get_config"]
