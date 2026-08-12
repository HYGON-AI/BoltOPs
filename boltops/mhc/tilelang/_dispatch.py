"""Architecture-family dispatch for TileLang MHC kernels."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import torch

from boltops.utility.arch_info import get_device_info

from ._arch import targets_for_arch

_DEFAULT_IMPLEMENTATIONS = {
    "hc_split_sinkhorn": ("split_sinkhorn", "hc_split_sinkhorn"),
    "mhc_fused_tilelang": ("fused_post_pre", "mhc_fused_tilelang"),
    "mhc_post_fwd": ("post", "mhc_post_fwd"),
    "mhc_pre_big_fuse": ("pre_big_fuse", "mhc_pre_big_fuse"),
    "pre_big_fuse_tilelang": (
        "pre_big_fuse_kernel",
        "pre_big_fuse_tilelang",
    ),
}

def resolve(op_name: str, device: torch.device | str | int) -> Any:
    arch = get_device_info(device).arch
    return _resolve_for_arch(op_name, arch)


@lru_cache(maxsize=None)
def _resolve_for_arch(op_name: str, arch: str) -> Any:
    implementation = _DEFAULT_IMPLEMENTATIONS.get(op_name)
    if implementation is None:
        raise RuntimeError(f"Unknown MHC operator: {op_name}")

    module_name, symbol_name = implementation
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = (
            f"boltops.mhc.tilelang.kernels.{target}.{module_name}"
        )
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            module = import_module(qualified_name)
            return getattr(module, symbol_name)

    raise RuntimeError(
        f"{op_name} does not support HCU architecture {arch}; "
        f"no module found for targets {targets}"
    )
