"""BoltOps performance-test framework.

Layout: one module per kernel under a backend directory,
``<backend>/<kernel>.py``, exposing a module-level ``SPEC``
(``bench.KernelSpec``).  Add a benchmark by adding such a module --
nothing else to register.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Tuple

from .bench import KernelSpec  # noqa: F401  (re-exported for spec modules)


def discover_specs() -> Tuple[Dict[str, KernelSpec], List[str]]:
    """Import every ``<backend>/*.py`` module that exposes ``SPEC``.

    Returns ``(specs keyed by 'backend/kernel', import errors)``.  A
    broken module is reported, not fatal, so one bad kernel cannot take
    down the rest of the run.
    """
    specs: Dict[str, KernelSpec] = {}
    errors: List[str] = []
    root = Path(__file__).resolve().parent
    for backend_dir in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        for path in sorted(backend_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"{backend_dir.name}.{path.stem}"
            try:
                module = importlib.import_module(f".{module_name}", __name__)
            except Exception as exc:
                errors.append(f"{module_name}: {str(exc).splitlines()[-1]}")
                continue
            spec = getattr(module, "SPEC", None)
            if spec is None:
                continue  # helper module without a benchmark
            if spec.name in specs:
                errors.append(f"{module_name}: duplicate SPEC {spec.name}")
                continue
            specs[spec.name] = spec
    return specs, errors
