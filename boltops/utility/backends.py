# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Backend sub-package discovery for op packages.

BoltOps op packages follow the ``boltops/<op>/<backend>/`` layout: each
backend sub-package (``triton``, ``tilelang``, ...) exposes the op's
unified entry points under identical names.  :func:`discover_backends`
imports a package's backend sub-packages and returns the ones that
export a given entry point, so tests (and tuning scripts) can
parametrize over the backends that are actually importable in this
environment.  A sub-package whose import fails on a missing optional
dependency (e.g. ``tilelang`` not installed) is treated as *unavailable*,
not as an error; any other import failure propagates.

This is a package-layout utility, not part of the perf collection
contract (``boltops/utility/bench.py``).
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable, Union


def discover_backends(
    package: Union[str, ModuleType], operator_name: str
) -> dict[str, Callable]:
    """Backends of ``package`` that export ``operator_name``.

    ``package`` is an op package (module or import name, e.g.
    ``boltops.fla.gdn``).  Sub-packages are scanned in sorted order;
    names starting with ``_`` are skipped.  The result maps
    ``backend sub-package name -> callable`` (sorted by name).
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    found: dict[str, Callable] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(
                f"{package.__name__}.{module_info.name}"
            )
        except ModuleNotFoundError as exc:
            if exc.name == module_info.name:
                continue
            raise
        operator = getattr(module, operator_name, None)
        if callable(operator):
            found[module_info.name] = operator
    return dict(sorted(found.items()))
