"""Architecture-aware configuration dispatch for MHC pre-big-fuse kernels."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

from boltops.mhc.tilelang._arch import targets_for_arch

from .common import PreBigFuseConfig

GetConfig = Callable[[int, int, int], PreBigFuseConfig]


@lru_cache(maxsize=None)
def resolve_get_config(arch: str) -> GetConfig:
    """Resolve exact, family, then common configuration for an architecture."""
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = (
            f"boltops.mhc.tilelang.configs.pre_big_fuse.{target}"
        )
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return import_module(qualified_name).get_config

    raise RuntimeError(
        f"No pre_big_fuse config module found for architecture {arch}; "
        f"tried targets {targets}"
    )


__all__ = ["PreBigFuseConfig", "resolve_get_config"]
