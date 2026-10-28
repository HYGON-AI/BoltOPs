# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Architecture-family dispatch for TileLang IHC kernels."""

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec

from boltops.utility.arch_info import get_device_info

from ._arch import targets_for_arch

_OPS = {
    "ihc_head": ("head", "ihc_head"),
    "ihc_post": ("post", "ihc_post"),
    "ihc_pre": ("pre", "ihc_pre"),
}


def resolve(name, device):
    return _resolve_for_arch(name, get_device_info(device).arch)


@lru_cache(maxsize=None)
def _resolve_for_arch(name, arch):
    if name not in _OPS:
        raise RuntimeError(f"Unknown IHC operator: {name}")
    module, symbol = _OPS[name]
    targets = targets_for_arch(arch)
    for target in targets:
        qualified = f"boltops.ihc.tilelang.kernels.{target}.{module}"
        try:
            spec = find_spec(qualified)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            return getattr(import_module(qualified), symbol)
    raise RuntimeError(
        f"{name} does not support HCU architecture {arch}; "
        f"no module found for targets {targets}"
    )
