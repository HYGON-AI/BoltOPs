# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Internal Triton capability detection for BoltOPs operators.

This module is intentionally not exported from :mod:`boltops.utility`. Each
operator decides whether to use it, which compiler options belong in its
autotune space, and how its config files are named.

Triton package versions map to compatible capability families as follows::

    2.1 - 3.3 -> 3.2
    3.4 - 3.6 -> 3.6
    3.7+      -> 3.8

The detected compiler options always come from the installed backend's
``HIPOptions``. HCU is preferred when present; otherwise AMD is used. Detection
is cached once per process.

Typical operator usage::

    from pathlib import Path
    import triton

    from boltops.utility.triton_capability import (
        get_compatible_triton_config_dirs,
        get_triton_capabilities,
        get_triton_config_dir,
    )

    caps = get_triton_capabilities()
    kwargs = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}

    # Unsupported compiler options must be omitted, not passed as False.
    if caps.supports("use_block_pingpong"):
        kwargs["use_block_pingpong"] = True
    config = triton.Config(kwargs, num_warps=8, num_stages=2)

    config_root = Path(__file__).parent / "configs" / "my_operator"
    write_dir = get_triton_config_dir(config_root, caps)
    read_dirs = get_compatible_triton_config_dirs(config_root, caps)

``write_dir`` is the current capability directory, for example
``triton_3_6``. ``read_dirs`` lists the current directory followed by lower
compatible versions, for example ``3.8 -> 3.6 -> 3.2``. These helpers only
return :class:`~pathlib.Path` objects; they never create directories.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from packaging.version import InvalidVersion, Version


_CAPABILITY_ORDER = ("3.2", "3.6", "3.8")


@dataclass(frozen=True)
class TritonCapabilities:
    version: str
    compiler_options: frozenset[str]
    backend: Optional[str]
    package_version: str

    def supports(self, compiler_option: str) -> bool:
        return compiler_option in self.compiler_options

    @property
    def config_dir_name(self) -> str:
        return f"triton_{self.version.replace('.', '_')}"


def _capability_from_triton_version(package_version: str) -> str:
    try:
        major_minor = Version(package_version).release[:2]
    except InvalidVersion as exc:
        raise RuntimeError(f"Invalid Triton version: {package_version!r}") from exc
    if major_minor >= (3, 7):
        return "3.8"
    if major_minor >= (3, 4):
        return "3.6"
    return "3.2"


def _select_backend(backends: dict) -> tuple[str, object]:
    if "hcu" in backends:
        return "hcu", backends["hcu"]
    if "amd" in backends:
        return "amd", backends["amd"]
    raise RuntimeError(
        "No compatible HIP Triton backend is installed"
    )


def _get_compiler_options() -> tuple[frozenset[str], str]:
    backends = importlib.import_module("triton.backends").backends
    backend_name, backend = _select_backend(backends)
    compiler_class = backend.compiler

    # New Triton versions register the compiler module normally. Older
    # versions load it dynamically, so HIPOptions must be read from the
    # parse_options function globals instead.
    compiler_module = inspect.getmodule(compiler_class)
    options_class = getattr(compiler_module, "HIPOptions", None)
    if options_class is None:
        parse_options = getattr(compiler_class, "parse_options", None)
        options_class = getattr(parse_options, "__globals__", {}).get("HIPOptions")

    if options_class is None:
        raise RuntimeError(
            "Cannot find HIPOptions for the selected Triton HIP backend"
        )
    fields = getattr(options_class, "__dataclass_fields__", {})
    if not fields:
        raise RuntimeError(
            "The selected Triton HIP backend HIPOptions has no dataclass fields"
        )
    return frozenset(fields), backend_name


@lru_cache(maxsize=1)
def get_triton_capabilities() -> TritonCapabilities:
    """Return the installed Triton backend's capability version and options."""

    triton = importlib.import_module("triton")
    package_version = getattr(triton, "__version__", None)
    if not package_version:
        raise RuntimeError("Triton does not provide __version__")

    compiler_options, backend = _get_compiler_options()

    version = _capability_from_triton_version(package_version)
    return TritonCapabilities(version, compiler_options, backend, package_version)


def get_triton_config_dir(
    base_dir: Union[str, Path],
    capabilities: TritonCapabilities,
) -> Path:
    """Return the directory where the current capability writes configs."""

    return Path(base_dir) / capabilities.config_dir_name


def get_compatible_triton_config_dirs(
    base_dir: Union[str, Path], capabilities: TritonCapabilities
) -> list[Path]:
    """Return current and lower-compatible versioned config directories."""

    base_path = Path(base_dir)
    current_index = _CAPABILITY_ORDER.index(capabilities.version)
    versioned_dirs = [
        base_path / f"triton_{version.replace('.', '_')}"
        for version in reversed(_CAPABILITY_ORDER[: current_index + 1])
    ]
    return versioned_dirs
