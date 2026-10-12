"""Shared pytest hooks for correctness vs performance suites.

Default: run functional/correctness tests only.
With ``--runslow``: also collect ``@pytest.mark.slow`` GPU grids (TileLang MQA
shape sweeps, extra kernels). CI omits these.
With ``--bench`` / ``--benchmark``: also collect ``@pytest.mark.bench`` cases
(real-model sizes, timing). Bench and correctness use different shape sets on
purpose.

Perf collection (optest ``per``):
The optest entry (``scripts/run_operator_benchmarks.py``) runs the declared
op benches with ``$OPTEST_BENCH_RAW_DIR`` set.  While that variable is set,
every timing call a ``bench``-marked test makes through
``triton.testing.do_bench`` / ``do_bench_cudagraph``,
``tilelang.profiler.do_bench`` / ``do_bench_cudagraph`` (including the
``from tilelang.profiler import ...`` form, rebound in the test module for
the duration of the test) or the ``time_ms`` fixture is observed and
emitted as one raw result row -- test code stays a plain benchmark, no
perf API required.  See ``docs/optest_perf.md``.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Callable

import pytest
import torch

from boltops.utility.bench import (
    BACKEND_ENV,
    RAW_DIR_ENV,
    auto_op_name_for_file,
    backend_token_in_name,
    bench_files,
    case_backend,
    derive_backend,
    derive_op_name,
    derived_op_counts,
    emit_row,
    extract_case_params,
    resolve_backend,
    time_ms as bolt_time_ms,
)

_ACTIVE_CTX: "_PerfCollectContext | None" = None

#: Kernel names of every triton / tilelang launch since process start.
_LAUNCHED_KERNELS: list[str] = []


def _install_kernel_launch_hooks() -> None:
    """Append each triton / tilelang kernel launch to ``_LAUNCHED_KERNELS``.

    Patched once per process on the toolchain launch choke points
    (``triton JITFunction.run`` / ``tilelang JITKernel.__call__``); the
    perf collector reads the list only around the timed call, so
    launches outside a timing window never leak into rows.
    """
    if getattr(_install_kernel_launch_hooks, "_done", False):
        return
    _install_kernel_launch_hooks._done = True
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        JITFunction = None
    if JITFunction is not None and not getattr(JITFunction.run, "_ktrace", False):
        original_run = JITFunction.run

        def run(self, *args, grid=None, warmup=None, **kwargs):
            _LAUNCHED_KERNELS.append(
                getattr(self.fn, "__name__", "triton_kernel")
            )
            return original_run(self, *args, grid=grid, warmup=warmup, **kwargs)

        run._ktrace = True
        JITFunction.run = run
    try:
        from tilelang.jit.kernel import JITKernel
    except ImportError:
        JITKernel = None
    if JITKernel is not None and not getattr(JITKernel.__call__, "_ktrace", False):
        original_call = JITKernel.__call__

        def call(self, *args, **kwargs):
            prim_func = getattr(self, "prim_func", None)
            name = "tilelang_kernel"
            if prim_func is not None:
                name = prim_func.attrs.get("global_symbol", name)
            _LAUNCHED_KERNELS.append(name)
            return original_call(self, *args, **kwargs)

        call._ktrace = True
        JITKernel.__call__ = call

# File-derived op name -> how many bench files derive it (same rule the
# perf entry applies, so raw rows and raw file names always match the
# entry's op names; see auto_op_name_for_file).
_BENCH_ROOT = Path(__file__).resolve().parent
_DERIVED_OP_COUNTS = derived_op_counts(bench_files(_BENCH_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("boltops")
    group.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run @pytest.mark.slow cases (large TileLang MQA GPU grids)",
    )
    group.addoption(
        "--bench",
        "--benchmark",
        action="store_true",
        default=False,
        help="also run @pytest.mark.bench performance cases (real-model sizes)",
    )
    group.addoption(
        "--bench-warmup",
        type=int,
        default=10,
        help="CUDA warmup budget in ms for --bench/--benchmark (default: 10)",
    )
    group.addoption(
        "--bench-repeat",
        type=int,
        default=50,
        help="CUDA timed budget in ms for --bench/--benchmark (default: 50)",
    )
    group.addoption(
        "--bench-t",
        "--bench-seq-len",
        dest="bench_seq_len",
        type=int,
        default=None,
        help=(
            "override the sequence length T for benchmarks that support it "
            "(default: benchmark-specific)"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: long GPU grids skipped unless --runslow (CI default omits these)",
    )
    config.addinivalue_line(
        "markers",
        "bench: performance benchmark on real-model sizes (requires --bench/--benchmark)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_bench = pytest.mark.skip(
        reason="pass --bench or --benchmark to run performance benchmarks"
    )
    skip_slow = pytest.mark.skip(
        reason="pass --runslow to run @pytest.mark.slow GPU grids"
    )
    run_bench = config.getoption("--bench")
    run_slow = config.getoption("--runslow")
    pass_backend = os.environ.get(BACKEND_ENV)
    for item in items:
        if not run_bench and "bench" in item.keywords:
            item.add_marker(skip_bench)
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if (
            run_bench
            and pass_backend
            and "bench" in item.keywords
            and isinstance(item, pytest.Function)
        ):
            token = backend_token_in_name(_func_name(item))
            if token is not None and token != pass_backend:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{token} bench; this perf pass runs "
                            f"{pass_backend} (one pass per backend)"
                        )
                    )
                )


def _func_name(item: pytest.Item) -> str:
    'Bare test function name (before the parametrize id).'
    function = getattr(item, "function", None)
    return getattr(function, "__name__", "") or str(item.name).split("[")[0]


@pytest.fixture
def bench_warmup(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bench-warmup"))


@pytest.fixture
def bench_repeat(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bench-repeat"))


@pytest.fixture
def bench_seq_len(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--bench-seq-len")
    return None if value is None else int(value)


@pytest.fixture
def time_ms(request) -> Callable[..., float]:
    """Shared timing helper: CUDA-graph first, event fallback.

    Times through the active backend's profiler entry points -- the
    same resolution the collector uses for the row's ``backend`` field
    (case-parameterized backend > ``$OPTEST_BACKEND`` > module
    ``PERF_BACKEND`` > ``triton``) -- so triton and tilelang benches
    share one identical call site.
    ``warmup`` / ``repeat`` are millisecond budgets (same convention as
    ``triton.testing.do_bench`` / ``tilelang.profiler.do_bench``).
    The helper times through the observed ``do_bench_cudagraph`` /
    ``do_bench`` entry points, so the perf collector records exactly
    one raw row with the method actually used (``cudagraph`` /
    ``event``); a capture failure prints a warning, non-capture errors
    still raise.

    Optional keyword enrichment for the row (defaults off, other rows
    unaffected): ``return_mode=`` picks the statistic (default
    ``"median"``; ``"min"`` / ``"max"`` / ``"mean"``) and is recorded
    in the CSV ``stat`` column when it differs from the default;
    ``flops=`` adds the ``tflops`` column (flops / latency); ``dtype=``
    adds the ``dtype`` column.  ``use_graph=`` (default ``True``) keeps
    CUDA-graph-first timing; ``use_graph=False`` times with plain
    events only (no capture attempt), recorded as ``event`` in the
    CSV ``timing`` column.
    """
    callspec = getattr(request.node, "callspec", None)
    backend = resolve_backend(
        callspec.params if callspec is not None else None,
        module=request.module,
        func_name=_func_name(request.node),
    )

    def _time_ms(
        fn: Callable[[], None],
        warmup: int,
        repeat: int,
        *,
        dtype=None,
        flops: float | None = None,
        return_mode: str = "median",
        use_graph: bool = True,
    ) -> float:
        if warmup < 0 or repeat <= 0:
            raise ValueError("warmup must be >= 0 and repeat must be > 0")
        context = _ACTIVE_CTX
        if context is not None and (
            dtype is not None or flops is not None or return_mode != "median"
        ):
            context.pending = {
                "params": None,
                "backend": None,
                "dtype": dtype,
                "flops": flops,
                "stat": None if return_mode == "median" else return_mode,
            }
        latency, _timing = bolt_time_ms(
            fn,
            warmup_ms=warmup,
            repeat_ms=repeat,
            backend=backend,
            return_mode=return_mode,
            use_graph=use_graph,
        )
        return latency

    return _time_ms


def _error_summary(exc: BaseException, limit: int = 200) -> str:
    """``<ExcType>: <first line>`` -- short enough for one CSV cell."""
    message = str(exc).strip()
    first_line = message.splitlines()[0] if message else ""
    summary = type(exc).__name__
    if first_line:
        summary += f": {first_line}"
    return summary[:limit]


class _PerfCollectContext:
    """Observation state for one bench test under the perf entry.

    Buffers one record per timing call made by the test; the records are
    flushed to ``<raw_dir>/<op>.raw.jsonl`` when the call phase ends.
    A failing bench flushes error rows instead of dropping its data:
    measured rows carry the test-level error, a timing call that raised
    contributes a row with ``latency_ms=None`` and its own error, and a
    test that never measured anything contributes one synthesized
    triage row.  Skipped tests emit nothing.  ``pending`` carries the
    ``time_ms`` fixture's optional row enrichment (``dtype`` /
    ``flops`` / ``stat``) so the call is recorded once with it.
    """

    def __init__(self, item: pytest.Item) -> None:
        self.item = item
        self.records: list[dict] = []
        self.pending: dict | None = None
        self._patches: dict[tuple[object, str], Callable] = {}
        self._module_patches: dict[str, Callable] = {}
        # Raw rows are filed under the test file's derived op name so the
        # perf entry (which looks rows up by test file) always finds them,
        # even when the module overrides the grouping via PERF_OP.
        module_file = getattr(getattr(item, "module", None), "__file__", None)
        self.raw_name = (
            auto_op_name_for_file(Path(module_file), _DERIVED_OP_COUNTS)
            if module_file else None
        )

    def record(
        self,
        *,
        latency_ms: float | None = None,
        timing: str | None = None,
        explicit: dict | None = None,
        error: str | None = None,
        kernels: list[str] | None = None,
    ) -> None:
        item = self.item
        call_index = len(self.records) + 1
        callspec = getattr(item, "callspec", None)
        case_params = callspec.params if callspec is not None else {}
        if explicit is not None and explicit.get("params") is not None:
            params = dict(explicit["params"])
        else:
            params = extract_case_params(case_params)
        params["call"] = call_index
        self.records.append(
            {
                "operator": derive_op_name(item.module, _DERIVED_OP_COUNTS),
                "backend": resolve_backend(
                    case_params,
                    explicit_backend=(explicit or {}).get("backend"),
                    module=item.module,
                    func_name=_func_name(item),
                ),
                "params": params,
                "dtype": explicit.get("dtype") if explicit else None,
                "latency_ms": latency_ms,
                "flops": explicit.get("flops") if explicit else None,
                "timing": timing,
                "stat": explicit.get("stat") if explicit else None,
                "kernels": ";".join(kernels) if kernels else None,
                "error": error,
            }
        )

    def _synthesize_error_record(self, error: str) -> None:
        """Triage row for a failed test that never measured anything."""
        item = self.item
        callspec = getattr(item, "callspec", None)
        case_params = callspec.params if callspec is not None else {}
        self.records.append(
            {
                "operator": derive_op_name(item.module, _DERIVED_OP_COUNTS),
                "backend": resolve_backend(
                    case_params,
                    module=item.module,
                    func_name=_func_name(item),
                ),
                "params": extract_case_params(case_params),
                "dtype": None,
                "latency_ms": None,
                "flops": None,
                "timing": None,
                "stat": None,
                "kernels": None,
                "error": error,
            }
        )

    def flush(self, error: str | None = None) -> None:
        if error is not None:
            if not self.records:
                self._synthesize_error_record(error)
            else:
                for record in self.records:
                    if record.get("error") is None:
                        record["error"] = error
        if len(self.records) == 1:
            self.records[0]["params"].pop("call", None)
        for record in self.records:
            emit_row(
                record["operator"],
                record["backend"],
                params=record["params"],
                latency_ms=record["latency_ms"],
                dtype=record["dtype"],
                flops=record["flops"],
                timing=record["timing"],
                stat=record.get("stat"),
                kernels=record.get("kernels"),
                error=record.get("error"),
                raw_name=self.raw_name,
            )

    def install_timing_patches(self) -> None:
        """Wrap the timing functions every backend exposes (triton + tilelang)."""
        _install_kernel_launch_hooks()
        import triton

        triton_testing = triton.testing
        for name, timing in (
            ("do_bench", "event"),
            ("do_bench_cudagraph", "cudagraph"),
        ):
            self._patches[(triton_testing, name)] = getattr(
                triton_testing, name
            )
            setattr(
                triton_testing, name, self._observed(
                    self._patches[(triton_testing, name)], timing
                )
            )

        try:
            import tilelang.profiler as tilelang_profiler
        except ImportError:
            tilelang_profiler = None
        if tilelang_profiler is not None:
            for name, timing in (
                ("do_bench", "event"),
                ("do_bench_cudagraph", "cudagraph"),
            ):
                original = getattr(tilelang_profiler, name, None)
                if original is not None:
                    self._patches[(tilelang_profiler, name)] = original
                    setattr(
                        tilelang_profiler, name,
                        self._observed(original, timing),
                    )

        # A test that did ``from tilelang.profiler import do_bench``
        # captured the original function at import time; rebind that name
        # in the test module's globals for the duration of the test so
        # the call stays observable without touching the test file.
        module_dict = getattr(self.item.module, "__dict__", None)
        if module_dict:
            for name, timing in (
                ("do_bench", "event"),
                ("do_bench_cudagraph", "cudagraph"),
            ):
                value = module_dict.get(name)
                if (
                    isinstance(value, types.FunctionType)
                    and (value.__module__ or "").startswith("tilelang")
                ):
                    self._module_patches[name] = value
                    module_dict[name] = self._observed(value, timing)

    def remove_timing_patches(self) -> None:
        for (module, attr), original in self._patches.items():
            setattr(module, attr, original)
        self._patches = {}
        module_dict = getattr(self.item.module, "__dict__", None)
        if module_dict:
            for name, original in self._module_patches.items():
                module_dict[name] = original
        self._module_patches = {}

    def _observed(self, original: Callable, timing: str) -> Callable:
        context = self

        def wrapper(*args, **kwargs):
            explicit = context.pending
            context.pending = None
            start = len(_LAUNCHED_KERNELS)

            def window_kernels() -> list[str] | None:
                seen: set[str] = set()
                unique: list[str] = []
                for name in _LAUNCHED_KERNELS[start:]:
                    if name not in seen:
                        seen.add(name)
                        unique.append(name)
                return unique or None

            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                context.record(
                    explicit=explicit,
                    timing=timing,
                    error=_error_summary(exc),
                    kernels=window_kernels(),
                )
                raise
            try:
                latency = float(result)
            except (TypeError, ValueError):
                return result
            context.record(
                latency_ms=latency, timing=timing, explicit=explicit,
                kernels=window_kernels(),
            )
            return result

        return wrapper


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Observe timing calls of bench tests while the perf entry collects."""
    global _ACTIVE_CTX
    if not os.environ.get(RAW_DIR_ENV) or "bench" not in item.keywords:
        yield
        return
    context = _PerfCollectContext(item)
    _ACTIVE_CTX = context
    context.install_timing_patches()
    try:
        outcome = yield
    finally:
        context.remove_timing_patches()
        _ACTIVE_CTX = None
        if outcome.excinfo is None:
            context.flush()
        elif not isinstance(
            outcome.excinfo[1], pytest.skip.Exception
        ):
            context.flush(error=_error_summary(outcome.excinfo[1]))
