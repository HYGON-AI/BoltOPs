# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared kernel-benchmark helpers for BoltOps.

The optest ``per`` entry (``scripts/run_operator_benchmarks.py``) is a
pure collection/formatting layer over the operator bench tests in
``test/``: those are plain ``@pytest.mark.bench`` unit tests (their own
inputs, timing via ``triton.testing.do_bench`` & co., their own prints)
that carry no perf-specific code.  While the entry sets
:data:`RAW_DIR_ENV`, the collection hooks in ``test/conftest.py``
observe every timing call a bench test makes and emit one structured
result row per call with :func:`emit_row`; the entry collects the rows
(:func:`read_rows`) and formats the unified perf CSVs.  :func:`time_ms`
and the ``time_ms`` fixture (``test/conftest.py``) are optional
shared-timing helpers on top of the same contract.

Rows are only emitted while the perf entry sets :data:`RAW_DIR_ENV`, so
a plain ``pytest --bench`` stays print-only.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

#: Env var pointing at this run's raw-results directory.  The perf entry
#: creates it before running the op benches; :func:`get_raw_dir` returns
#: ``None`` outside such a run.
RAW_DIR_ENV = "OPTEST_BENCH_RAW_DIR"

#: Env var naming the backend the perf entry is currently running
#: (one pytest pass per declared op backend).  When set, collection
#: stamps every row with this value instead of deriving the backend from
#: the test module.  Unset outside the perf entry, so plain local runs
#: keep the module's own backend label.
BACKEND_ENV = "OPTEST_BACKEND"

#: Columns filled in by the perf entry for every row.  Op benches must
#: not provide these (see :func:`emit_row`).
FRAMEWORK_COLS = (
    "arch", "version", "timing", "stat", "kernels", "latency_ms",
    "tflops",
)

_RESERVED = frozenset({"operator", "backend", "dtype", *FRAMEWORK_COLS})

# (operator, original param name) already warned about; a case param that
# collides with a reserved column is renamed ``case_<name>`` (once per
# op+param) instead of failing the test.
_warned_param_collisions: set[tuple[str, str]] = set()


def _dedupe_reserved_params(
    operator: str, params: Mapping[str, object]
) -> dict[str, object]:
    """Case params with reserved-column names renamed to ``case_<name>``."""
    keys = set(params)
    out: dict[str, object] = {}
    for key, value in params.items():
        if key not in _RESERVED:
            out[key] = value
            continue
        renamed = f"case_{key}"
        suffix = 1
        while renamed in keys or renamed in out:
            suffix += 1
            renamed = f"case_{key}_{suffix}"
        if (operator, key) not in _warned_param_collisions:
            _warned_param_collisions.add((operator, key))
            print(
                f"[bench] {operator}: case param {key!r} collides with a "
                f"reserved CSV column; emitted as {renamed!r}",
                file=sys.stderr,
                flush=True,
            )
        out[renamed] = value
    return out


def get_raw_dir() -> Optional[Path]:
    """Raw-results directory of the current run, or ``None`` if unset."""
    raw = os.environ.get(RAW_DIR_ENV)
    return Path(raw) if raw else None


def emit_row(
    operator: str,
    backend: str,
    *,
    params: Mapping[str, object],
    latency_ms: float | None,
    dtype: Optional[object] = None,
    flops: Optional[float] = None,
    timing: Optional[str] = None,
    stat: Optional[str] = None,
    kernels: Optional[str] = None,
    error: Optional[str] = None,
    raw_dir: Optional[Path] = None,
    raw_name: Optional[str] = None,
) -> None:
    """Emit one measured result row for the perf collector.

    Appends a JSON line to ``<raw_dir>/<raw_name>.raw.jsonl`` where
    ``raw_name`` defaults to ``operator``.  Callers whose grouping name
    can differ from the test file's derived name (``PERF_OP``) pass the
    file-derived name so the perf entry's lookup (by test file) always
    finds the rows.  ``params`` are the case columns; their insertion
    order is preserved into the final CSV.  ``dtype`` and ``flops`` are
    optional (the CSV cells stay empty when absent).  A failing bench
    test emits the same row with ``latency_ms=None`` and a short
    ``error`` summary so the failure stays visible in the CSV.  No-op
    when the raw-results directory is unset (local ``pytest --bench``
    stays print-only).
    """
    directory = Path(raw_dir) if raw_dir is not None else get_raw_dir()
    if directory is None:
        return
    row = {
        "operator": operator,
        "backend": backend,
        "params": _dedupe_reserved_params(operator, params),
        "dtype": dtype_str(dtype) if dtype is not None else None,
        "latency_ms": None if latency_ms is None else float(latency_ms),
        "flops": None if flops is None else float(flops),
        "timing": timing,
        "stat": stat,
        "kernels": kernels,
        "error": error,
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (
        directory / f"{raw_name or operator}.raw.jsonl"
    ).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_rows(raw_dir: Path) -> dict[str, list[dict]]:
    """Read emitted rows grouped by operator: ``{op: [row, ...]}``."""
    rows: dict[str, list[dict]] = {}
    for path in sorted(Path(raw_dir).glob("*.raw.jsonl")):
        operator = path.name[: -len(".raw.jsonl")]
        with path.open("r", encoding="utf-8") as handle:
            rows[operator] = [json.loads(line) for line in handle if line.strip()]
    return rows


_VERSION_CACHE: dict[str, str] = {}


def _dist_version(package: str) -> str:
    """Full dist version of ``package`` (what ``pip show`` reports).

    The version string carries the toolchain's own build suffix / commit
    (e.g. ``3.6.0+git46766868.staging``), so it is the version tag rows
    use -- no separate source-commit column is needed.  Reads package
    metadata only (no import); returns ``""`` when not installed.
    """
    if package not in _VERSION_CACHE:
        try:
            _VERSION_CACHE[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            _VERSION_CACHE[package] = ""
    return _VERSION_CACHE[package]


def triton_version() -> str:
    """Triton dist version (``""`` when not installed)."""
    return _dist_version("triton")


def tilelang_version() -> str:
    """Tilelang dist version (``""`` when not installed)."""
    return _dist_version("tilelang")


def backend_version(backend: str) -> str:
    """Dist version of a row's toolchain (``""`` for unknown backends)."""
    if backend == "triton":
        return triton_version()
    if backend == "tilelang":
        return tilelang_version()
    return ""


def dtype_str(dtype: object) -> str:
    """CSV spelling of a torch dtype: ``torch.bfloat16`` -> ``bfloat16``."""
    return str(dtype).removeprefix("torch.")


def tflops(flops: float, latency_ms: float) -> float:
    """TFLOPS implied by ``flops`` completed in ``latency_ms``."""
    return flops / (latency_ms * 1e-3) / 1e12


#: Statistical modes both toolchains' ``do_bench(_cudagraph)`` accept
#: as a scalar result (triton additionally accepts "all", which returns
#: an array and is therefore not exposed).
STAT_MODES = ("min", "max", "mean", "median")


def time_ms(
    run: Callable[[], None],
    warmup_ms: float = 10.0,
    repeat_ms: float = 30.0,
    backend: str = "triton",
    return_mode: str = "median",
    use_graph: bool = True,
) -> tuple[float, str]:
    """Latency in ms plus the method used ("cudagraph"/"event").

    ``return_mode`` picks the statistic over the timed iterations
    (default ``"median"``; ``"min"`` / ``"max"`` / ``"mean"`` are
    supported by both toolchains).

    ``use_graph`` (default ``True``) prefers CUDA-graph capture; ops
    that cannot be captured fall back to plain event timing.  Errors
    unrelated to capture (e.g. a broken kernel) are re-raised so they
    fail loudly instead of being timed with the fallback path.
    ``use_graph=False`` times with plain events only (no capture
    attempt) -- the owner's explicit choice, e.g. for calls with host
    work or for side-by-side event comparisons.

    ``backend`` picks the toolchain profiler ("triton" or "tilelang");
    both expose the same ms-budget ``do_bench(_cudagraph)`` contract.
    Calls go through the toolchain's module attributes so the perf
    collector observes them exactly like a direct timing call.
    """
    if return_mode not in STAT_MODES:
        raise ValueError(f"unsupported return_mode: {return_mode!r}")
    if backend == "triton":
        import triton

        cudagraph_bench = triton.testing.do_bench_cudagraph
        event_bench = triton.testing.do_bench
    elif backend == "tilelang":
        import tilelang.profiler as tilelang_profiler

        cudagraph_bench = tilelang_profiler.do_bench_cudagraph
        event_bench = tilelang_profiler.do_bench
    else:
        raise ValueError(f"unsupported timing backend: {backend!r}")

    if not use_graph:
        return (
            float(
                event_bench(
                    run, warmup=warmup_ms, rep=repeat_ms,
                    return_mode=return_mode,
                )
            ),
            "event",
        )
    try:
        return (
            float(cudagraph_bench(run, rep=repeat_ms, return_mode=return_mode)),
            "cudagraph",
        )
    except Exception as exc:
        message = str(exc).replace("\n", " ")
        if "captur" not in message.lower():
            raise
        print(
            f"[bench] CUDA graph capture failed ({message[:200]}); "
            "falling back to event timing",
            file=sys.stderr,
            flush=True,
        )
        return (
            float(
                event_bench(
                    run, warmup=warmup_ms, rep=repeat_ms,
                    return_mode=return_mode,
                )
            ),
            "event",
        )


def _is_scalar(value: object) -> bool:
    # None is intentionally excluded: auto-derived ``None`` fields are
    # "not set" noise; owners can still pass ``None`` explicitly via
    # ``params`` (emit_row does not filter).
    return isinstance(value, (str, int, float, bool))


def _stem_op_name(stem: str) -> str:
    """Stem minus the ``test_`` / ``generic_`` prefixes."""
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    if stem.startswith("generic_"):
        stem = stem[len("generic_"):]
    return stem


#: A test file carrying a ``@pytest.mark.bench`` line is an op bench file
#: (static line scan, no import).
BENCH_MARKER = re.compile(r"^\s*@pytest\.mark\.bench\b")

#: Derived op names that are too generic to name a CSV on their own; such
#: files are qualified with their parent directory (``test/step4_dsa/
#: test_bench.py`` -> ``step4_dsa_bench``).
GENERIC_OP_NAMES = frozenset({"bench", "benchmark", "perf", "test"})


def bench_files(root) -> list[Path]:
    """Bench files under ``root`` (see :data:`BENCH_MARKER`), sorted."""
    found: list[Path] = []
    for path in sorted(Path(root).rglob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(BENCH_MARKER.match(line) for line in text.splitlines()):
            found.append(path)
    return found


def derived_op_counts(paths) -> dict[str, int]:
    """File-derived (unqualified) op name -> number of files deriving it."""
    counts: dict[str, int] = {}
    for path in paths:
        derived = op_name_for_file(path)
        counts[derived] = counts.get(derived, 0) + 1
    return counts


def auto_op_name_for_file(path, derived_counts: dict[str, int] | None = None) -> str:
    """File-derived op name, qualified when generic or ambiguous.

    A derived name that is generic (see :data:`GENERIC_OP_NAMES`) or
    shared by more than one bench file (per ``derived_counts``) is
    prefixed with the parent directory (``test/step4_dsa/test_bench.py``
    -> ``step4_dsa_bench``; ``test/ihc/test_post.py`` +
    ``test/mhc/test_post.py`` -> ``ihc_post`` / ``mhc_post``).  Files
    directly under ``test/`` keep their derived name (``PERF_OP`` is the
    only way to rename them).  Shared by the perf entry and the
    collection layer in ``test/conftest.py`` so both sides always agree
    on an op name.
    """
    path = Path(path)
    derived = op_name_for_file(path)
    ambiguous = (
        derived in GENERIC_OP_NAMES
        or (derived_counts is not None and derived_counts.get(derived, 0) > 1)
    )
    if ambiguous:
        parent = path.parent.name
        if parent != "test":
            return f"{parent}_{derived}"
    return derived


def op_name_for_file(path: Path) -> str:
    """Derived op name for a test file path (see :func:`derive_op_name`)."""
    stem = Path(path).name
    if stem.endswith(".py"):
        stem = stem[: -len(".py")]
    return _stem_op_name(stem)


def derive_op_name(
    module: object, derived_counts: dict[str, int] | None = None
) -> str:
    """Op name for perf grouping.

    A module-level ``PERF_OP`` string constant wins; otherwise the test
    module stem minus the ``test_`` / ``generic_`` prefixes, auto-
    qualified when generic or shared by two files
    (:func:`auto_op_name_for_file`).  Set the same ``PERF_OP`` in
    several modules to merge them into one op.
    """
    op = getattr(module, "PERF_OP", None)
    if isinstance(op, str) and op:
        return op
    module_file = Path(getattr(module, "__file__", "") or "")
    return auto_op_name_for_file(module_file, derived_counts)


#: Toolchains the perf framework knows about (CSV ``version`` column and
#: the ``time_ms`` fixture route to the matching profiler).
KNOWN_BACKENDS = ("triton", "tilelang")


def case_backend(params: Mapping[str, object] | None) -> str | None:
    """Backend declared by a ``(backend_name, callable)`` parametrize value.

    A bench that parametrizes its implementation over backends (one test
    file, several backends -- e.g. ``gdn_chunk``) passes the case as
    ``pytest.param(("triton", fn), id="triton")``.  The first such value
    whose name is a known backend names the row's backend, so the
    ``backend`` / ``version`` columns follow the measured implementation
    instead of the pass-level ``$OPTEST_BACKEND``.
    """
    if not params:
        return None
    for value in params.values():
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and value[0] in KNOWN_BACKENDS
            and callable(value[1])
        ):
            return value[0]
    return None


def backend_token_in_name(name: str) -> str | None:
    """The unique known backend named in a test function name.

    A dual-backend bench file written as one test per backend (§6)
    names its tests after the backend they measure
    (``test_bench_triton_op`` / ``test_bench_tilelang_op``); the token
    routes the row's ``backend`` / ``version`` and the ``time_ms``
    harness, and the perf entry skips the test in the other backend's
    pass.  A name naming zero or two known backends yields ``None``.
    """
    lowered = name.lower()
    found = {b for b in KNOWN_BACKENDS if b in lowered}
    return found.pop() if len(found) == 1 else None


def resolve_backend(
    case_params: Mapping[str, object] | None = None,
    *,
    explicit_backend: str | None = None,
    module: object | None = None,
    func_name: str | None = None,
) -> str:
    """Backend for a timing call / result row.

    Priority: case-parameterized backend (see :func:`case_backend`) >
    function-name backend token (see :func:`backend_token_in_name`) >
    ``$OPTEST_BACKEND`` (the entry's pass dimension) > module
    derivation (``PERF_BACKEND(S)`` > detection > ``triton``).
    ``explicit_backend`` remains for in-process callers that must pin a
    row's label.
    """
    if explicit_backend:
        return explicit_backend
    if case_params is not None:
        backend = case_backend(case_params)
        if backend is not None:
            return backend
    if func_name is not None:
        backend = backend_token_in_name(func_name)
        if backend is not None:
            return backend
    if os.environ.get(BACKEND_ENV):
        return os.environ[BACKEND_ENV]
    return derive_backend(module)


def detect_backends_from_file(path) -> tuple[str, ...]:
    """Toolchains a bench file opts into via module-level
    ``pytest.importorskip("<toolchain>")`` calls (AST, no import).

    A file skipping both toolchains declares both backends.  Order is
    canonical (``triton`` before ``tilelang``) regardless of the file's
    call order.  Files without such a call return ``()``.
    """
    try:
        import ast as _ast

        tree = _ast.parse(Path(path).read_text(encoding="utf-8"),
                          filename=str(path))
    except (OSError, SyntaxError):
        return ()
    found: set[str] = set()
    top_level_calls: list[object] = []
    for node in tree.body:
        if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call):
            top_level_calls.append(node.value)
        elif isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call):
            top_level_calls.append(node.value)
    for call in top_level_calls:
        func = getattr(call, "func", None)
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name != "importorskip":
            continue
        args = getattr(call, "args", ())
        if (
            args
            and isinstance(args[0], _ast.Constant)
            and isinstance(args[0].value, str)
            and args[0].value in KNOWN_BACKENDS
        ):
            found.add(args[0].value)
    return tuple(b for b in KNOWN_BACKENDS if b in found)


def derive_backend(module: object) -> str:
    """Kernel-DSL backend label for a test module.

    ``PERF_BACKEND`` (str) wins; else the first entry of ``PERF_BACKENDS``
    (multi-backend files, see the perf entry's auto-discovery); else the
    first toolchain the file opts into via module-level
    ``pytest.importorskip``; else ``"triton"``.
    """
    backend = getattr(module, "PERF_BACKEND", None)
    if isinstance(backend, str) and backend:
        return backend
    backends = getattr(module, "PERF_BACKENDS", None)
    if (
        isinstance(backends, (list, tuple))
        and backends
        and all(isinstance(b, str) and b for b in backends)
    ):
        return backends[0]
    module_file = getattr(module, "__file__", None)
    if module_file:
        detected = detect_backends_from_file(module_file)
        if detected:
            return detected[0]
    return "triton"


def extract_case_params(callspec_params: Mapping[str, object]) -> dict:
    """Flatten pytest parametrize values into CSV-able case params.

    dict / dataclass values contribute their scalar fields (str / int /
    float / bool / None); scalar values are kept under the param name;
    anything else (tensors, tuples, ...) is skipped.
    """
    flat: dict[str, object] = {}
    for name, value in callspec_params.items():
        if isinstance(value, Mapping):
            items = value.items()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            items = dataclasses.asdict(value).items()
        elif _is_scalar(value):
            flat[str(name)] = value
            continue
        else:
            continue
        for key, val in items:
            if _is_scalar(val):
                flat[str(key)] = val
    return flat
