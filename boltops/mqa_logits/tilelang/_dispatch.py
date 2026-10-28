# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Architecture-family dispatch for TileLang MQA logits kernels."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import torch

from boltops.utility.arch_info import get_device_info

from ._arch import targets_for_arch

_DEFAULT_IMPLEMENTATIONS = {
    "qsa_mqa_decode": ("mqa", "qsa_mqa_decode"),
    "qsa_mqa_prefill": ("mqa", "qsa_mqa_prefill"),
}


def resolve(op_name: str, device: torch.device | str | int) -> Any:
    arch = get_device_info(device).arch
    return _resolve_for_arch(op_name, arch)


@lru_cache(maxsize=None)
def _resolve_for_arch(op_name: str, arch: str) -> Any:
    implementation = _DEFAULT_IMPLEMENTATIONS.get(op_name)
    if implementation is None:
        raise RuntimeError(f"Unknown QSA operator: {op_name}")

    module_name, symbol_name = implementation
    targets = targets_for_arch(arch)
    for target in targets:
        qualified_name = (
            f"boltops.mqa_logits.tilelang.kernels.{target}.{module_name}"
        )
        try:
            spec = find_spec(qualified_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            module = import_module(qualified_name)
            implementation = getattr(module, symbol_name, None)
            if implementation is not None:
                return implementation

    raise RuntimeError(
        f"{op_name} does not support HCU architecture {arch}; "
        f"no implementation found for targets {targets}"
    )
