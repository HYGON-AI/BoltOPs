#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Run the BoltOps kernel performance benchmarks (auto-discovered).

This script is the optest ``per`` entry, so it stays at this fixed path.

The script only collects and formats: every measurement lives in the
operator's own bench tests under ``test/`` (plain ``@pytest.mark.bench``
unit tests).  Ops are **auto-discovered** -- there is no registry:
every ``test_*.py`` file carrying a ``@pytest.mark.bench`` test is an op
bench.  The op name is the file name minus its ``test_`` / ``generic_``
prefixes, automatically: a generic stem (``bench`` / ``benchmark`` /
``perf`` / ``test``) or a stem shared by two files gets its parent
directory prefixed (``test/step4_dsa/test_bench.py`` ->
``step4_dsa_bench``; ``test/ihc/test_post.py`` / ``test/mhc/test_post.py``
-> ``ihc_post`` / ``mhc_post``); a module-level ``PERF_OP`` constant
pins the name manually when auto-derivation cannot disambiguate (e.g.
files directly under ``test/``).  A file's backends are
auto-detected from its module-level
``pytest.importorskip("<toolchain>")`` calls (``PERF_BACKEND`` / 
``PERF_BACKENDS`` constants can override; default ``triton``);
``PERF_SKIP = "<reason>"`` excludes it from the entry (plain
``pytest --bench`` still runs it).
Adding a perf case is just writing a normal bench test.

While ``$OPTEST_BENCH_RAW_DIR`` is set, the collection layer in
``test/conftest.py`` observes every timing call the tests make
(``time_ms`` fixture / ``triton.testing.do_bench(_cudagraph)`` /
``tilelang.profiler.do_bench(_cudagraph)``) and emits one raw result
row per call.  The entry groups the rows by op and writes one unified
``<op>.csv`` per op; an op whose benches emit no observed timing (e.g.
manual event sampling) runs but produces no CSV.

CSVs are written into ``<repo>/optest_results/`` (or ``-o``) *inside the
source tree*: optest detects changed ``*.csv`` files in the repo between
run start and end and copies them into its own result directory, so the
files must not be written straight into ``$OPTEST_OUTPUT_DIR``.  The raw
jsonl rows land in ``$OPTEST_OUTPUT_DIR/raw`` under optest, else next to
the CSVs.

Backends: an op is one (set of) bench test file(s) written against the
op's unified interface -- the tests never name a backend.  Each file
runs in one pytest pass per backend it declares, with
``$OPTEST_BACKEND`` set, so the op's dispatch layer picks the
implementation; all backends' rows merge into the single ``<op>.csv``,
distinguished by the ``backend`` column.  Each row's ``version``
column carries that backend's dist version, whose string embeds the
toolchain's own build commit.

GPU scheduling: --gpus 1,2,3,4 distributes file/backend processes across
the selected cards, one process per card. Without --gpus, inherit the
optest allocation or visibility mask; an unmasked local run stays serial.
For stock optest on Linux, recover its explicit --gpus pool from the
immediate parent process argv, since its child masks may name only one
leased card. Additional cards are reserved using optest GPU leases.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from boltops.utility.arch_info import get_arch  # noqa: E402
from boltops.utility.bench import (  # noqa: E402
    BACKEND_ENV,
    auto_op_name_for_file,
    backend_version,
    bench_files,
    derived_op_counts,
    detect_backends_from_file,
    read_rows,
    tflops,
)

RESULTS_DIR = REPO_ROOT / "optest_results"
TEST_DIR = REPO_ROOT / "test"
PERF_CONSTANTS = ("PERF_OP", "PERF_BACKEND", "PERF_BACKENDS", "PERF_SKIP")


@dataclass(frozen=True)
class FileSpec:
    """One discovered bench file and how the entry must run it."""

    file: str
    op: str
    backends: tuple[str, ...]


@dataclass(frozen=True)
class OpSpec:
    """One discovered op: its bench files and the union of their backends."""

    op: str
    files: tuple[str, ...]
    backends: tuple[str, ...]


def _parse_perf_constants(path: Path) -> dict[str, object]:
    """Module-level ``PERF_*`` constants, parsed via AST (no import:
    bench files may need optional dependencies that are absent locally)."""
    constants: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in PERF_CONSTANTS
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return constants


def discover_ops() -> tuple[dict[str, OpSpec], list[FileSpec], list[tuple[str, str]]]:
    """Auto-discover the op benches (no registry).

    Every ``test_*.py`` under ``test/`` carrying a ``@pytest.mark.bench``
    line is a bench file.  op name = ``PERF_OP`` or the file-derived
    name (auto-qualified with the parent directory when generic or
    colliding, see :func:`_auto_op_name`); backends =
    ``PERF_BACKENDS`` / ``PERF_BACKEND`` (explicit) else auto-detected
    from module-level ``pytest.importorskip("<toolchain>")`` calls else
    ``("triton",)``; ``PERF_SKIP`` excludes the file (with reason).
    Returns ``(ops by name, file specs, skipped (file, reason))``.
    """
    # Phase 1: parse every bench file (constants).  The op-name
    # qualification counts cover ALL bench files (PERF_SKIP included) so
    # the collection layer in test/conftest.py -- which does not know
    # about PERF_SKIP -- derives identical names.
    candidates: list[tuple[Path, str, dict[str, object]]] = []
    skipped: list[tuple[str, str]] = []
    for path in bench_files(TEST_DIR):
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            constants = _parse_perf_constants(path)
        except SyntaxError as exc:
            raise RuntimeError(f"cannot parse {rel}: {exc}") from exc
        skip_reason = constants.get("PERF_SKIP")
        if skip_reason:
            reason = "PERF_SKIP" if skip_reason is True else str(skip_reason)
            skipped.append((rel, reason))
            continue
        candidates.append((path, rel, constants))

    derived_counts = derived_op_counts(bench_files(TEST_DIR))

    by_op: dict[str, list[str]] = {}
    op_backends: dict[str, set[str]] = {}
    file_specs: list[FileSpec] = []
    for path, rel, constants in candidates:
        op = (
            constants.get("PERF_OP")
            or auto_op_name_for_file(path, derived_counts)
        )
        explicit = constants.get("PERF_BACKENDS") or constants.get(
            "PERF_BACKEND"
        )
        if explicit is not None:
            if isinstance(explicit, str):
                explicit = [explicit]
            if (
                not isinstance(explicit, list)
                or not explicit
                or not all(isinstance(b, str) and b for b in explicit)
            ):
                raise RuntimeError(
                    f"{rel}: PERF_BACKEND(S) must be a non-empty string "
                    "or list of strings"
                )
            backends = tuple(dict.fromkeys(explicit))
        else:
            # Auto-detect from module-level pytest.importorskip calls
            backends = detect_backends_from_file(path) or ("triton",)
        by_op.setdefault(op, []).append(rel)
        op_backends.setdefault(op, set()).update(backends)
        file_specs.append(FileSpec(rel, op, backends))
    ops = {
        op: OpSpec(op, tuple(by_op[op]), tuple(sorted(op_backends[op])))
        for op in sorted(by_op)
    }
    if not ops:
        raise RuntimeError(
            f"no bench test files under {TEST_DIR} "
            "(expected files with @pytest.mark.bench tests)"
        )
    return ops, file_specs, skipped


def plan_runs(
    file_specs: list[FileSpec],
    op_names: list[str],
    backend_filter: list[str] | None,
) -> tuple[list[tuple[str, list[str], list[str]]], list[str]]:
    """Group selected ops into one pytest pass per backend value.

    Each file runs in one pass per backend it declares (a multi-backend
    op therefore runs different files under different backends).
    Returns ``[(backend, files, ops), ...]`` in deterministic order
    (backends sorted; files keep discovery order) plus the ops skipped
    because ``--backends`` matched none of their backends.
    """
    selected_ops = set(op_names)
    by_backend: dict[str, dict[str, list[str]]] = {}
    for spec in file_specs:
        if spec.op not in selected_ops:
            continue
        for backend in spec.backends:
            if backend_filter is not None and backend not in backend_filter:
                continue
            bucket = by_backend.setdefault(backend, {"files": [], "ops": []})
            if spec.file not in bucket["files"]:
                bucket["files"].append(spec.file)
            if spec.op not in bucket["ops"]:
                bucket["ops"].append(spec.op)
    skipped = [
        op for op in op_names
        if not any(op in by_backend[b]["ops"] for b in by_backend)
    ]
    return [
        (backend, by_backend[backend]["files"], by_backend[backend]["ops"])
        for backend in sorted(by_backend)
    ], skipped


def optest_gpu_pool(
    env: dict[str, str], proc_root: Path = Path("/proc"),
) -> str | None:
    """Recover explicit --gpus from our immediate optest parent on Linux.

    Stock optest overwrites its child's masks with one leased GPU. Read only
    that parent's argv, only in an optest run; never walk unrelated ancestors
    or infer a pool from the host's visible hardware.
    """
    if not env.get("OPTEST_OUTPUT_DIR") or not env.get("OPTEST_GPUS"):
        return None
    try:
        argv = [os.fsdecode(arg) for arg in
                (proc_root / str(os.getppid()) / "cmdline").read_bytes().split(b"\0") if arg]
    except OSError:
        return None
    if not argv:
        return None
    names = {"optest", "optest.bak"}
    if Path(argv[0]).name in names:
        cli_args = argv[1:]
    elif Path(argv[0]).name.startswith("python"):
        if len(argv) > 1 and Path(argv[1]).name in names:
            cli_args = argv[2:]
        elif argv[1:3] == ["-m", "optest.cli"]:
            cli_args = argv[3:]
        else:
            return None
    else:
        return None
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for option in ("--op_name", "--test_mode", "--gpus"):
        parser.add_argument(option)
    try:
        args, _ = parser.parse_known_args(cli_args)
    except (argparse.ArgumentError, SystemExit):
        return None
    if args.op_name == "boltops" and args.test_mode in ("per", "all"):
        return args.gpus if args.gpus not in (None, "auto") else None
    return None


def resolve_gpu_pool(
    requested: str | None, env: dict[str, str], parent_pool: str | None = None,
) -> list[str | None]:
    """Use only explicitly selected or inherited GPU IDs; never scan the host."""
    def parse(value: str) -> list[str]:
        ids = [part.strip() for part in value.split(",")]
        if not all(part.isascii() and part.isdecimal() for part in ids):
            raise ValueError(f"expected comma-separated nonnegative GPU IDs, got {value!r}")
        ids = [str(int(part)) for part in ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate GPU IDs: {value!r}")
        return ids

    limits = []
    allocation = env.get("OPTEST_GPUS", "auto")
    if allocation != "auto":
        limits.append(parse(allocation))
    # HIP takes precedence over CUDA on ROCm, but neither mask may be widened.
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if name in env:
            limits.append(parse(env[name]))
    if parent_pool is not None:
        allowed = parse(parent_pool)
        if any(not set(limit).issubset(allowed) for limit in limits):
            raise ValueError("optest allocation/mask is outside its original --gpus pool")
        limits = [allowed]
    pool = parse(requested) if requested is not None else (limits[0] if limits else None)
    if pool is None:
        return [None]  # Legacy local invocation: inherit its device environment.
    if any(not set(pool).issubset(limit) for limit in limits):
        raise ValueError(f"GPU pool {pool} exceeds the inherited allocation/mask {limits}")
    if "ROCR_VISIBLE_DEVICES" in env:
        raise ValueError(
            "GPU scheduling requires HIP_VISIBLE_DEVICES instead of "
            "ROCR_VISIBLE_DEVICES (different ID namespaces)"
        )
    return pool


@contextmanager
def lease_gpu_pool(devices: list[str | None], parent_allocation: str | None):
    """Borrow the parent's leased device(s), reserve additional ones ourselves."""
    if parent_allocation is None:
        yield devices
        return
    inherited = set(parent_allocation.split(","))
    extra = [gpu for gpu in devices if gpu not in inherited]
    if not extra:
        yield devices
        return
    from optest.device import DeviceAllocator

    allocator = DeviceAllocator(
        requested=",".join(extra), usage_threshold=0,
        wait_seconds=0, stable_samples=1, sample_interval=0,
    )
    owned = []
    try:
        for _ in extra:
            selected = allocator.try_acquire(1)
            if not selected:
                break
            owned.extend(selected)
        available = inherited | {str(gpu) for gpu in owned}
        assigned = [gpu for gpu in devices if gpu in available]
        if not assigned:
            raise RuntimeError("all requested GPUs are leased by other optest tasks")
        skipped = [gpu for gpu in devices if gpu not in available]
        if skipped:
            print(f"[schedule] GPUs leased elsewhere, omitted: {skipped}", flush=True)
        yield assigned
    finally:
        allocator.release(owned)


def _pytest_command(file: str) -> list[str]:
    return [
        sys.executable, "-m", "pytest", file,
        "--bench", "-m", "bench",
        "--continue-on-collection-errors",
        "--durations=10",
        "-q", "-s", "-ra", "--no-header", "-p", "no:cacheprovider",
    ]


def run_benchmarks(
    runs: list[tuple[str, list[str], list[str]]],
    raw_dir: Path,
    devices: list[str | None],
) -> list[int]:
    """One process per file/backend, at most one process on each assigned GPU.

    Task-local raw files are merged in plan order after execution. Finished
    logs are replayed as contiguous blocks, including failed/crashed tasks.
    """
    tasks = [(backend, file) for backend, files, _ in runs for file in files]
    codes = [1] * len(tasks)
    task_dirs = [raw_dir / "tasks" / f"{i + 1:04d}" for i in range(len(tasks))]
    running = {}
    available = list(devices)
    next_task = 0
    raw_dir.mkdir(parents=True, exist_ok=True)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        while next_task < len(tasks) or running:
            while available and next_task < len(tasks):
                index = next_task
                next_task += 1
                gpu = available.pop(0)
                backend, file = tasks[index]
                directory = task_dirs[index]
                directory.mkdir(parents=True, exist_ok=True)
                env = dict(
                    os.environ, OPTEST_BENCH_RAW_DIR=str(directory),
                    **{BACKEND_ENV: backend},
                )
                if gpu is not None:
                    env.update(
                        HIP_VISIBLE_DEVICES=gpu, CUDA_VISIBLE_DEVICES=gpu,
                        OPTEST_GPUS=gpu,
                    )
                log_path = directory / "pytest.log"
                print(
                    f"[{index + 1}/{len(tasks)}] backend={backend} file={file} "
                    f"gpu={gpu or 'inherited'} log={log_path}", flush=True,
                )
                handle = log_path.open("wb")
                try:
                    process = subprocess.Popen(
                        _pytest_command(file), cwd=REPO_ROOT, env=env,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except Exception:
                    handle.close()
                    available.append(gpu)
                    traceback.print_exc()
                    print(f"[summary] backend={backend} file={file}: launch FAILED", flush=True)
                    continue
                running[index] = (process, handle, gpu)

            for index, (process, handle, gpu) in list(running.items()):
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                codes[index] = rc
                backend, file = tasks[index]
                print(
                    f"[{index + 1}/{len(tasks)}] backend={backend} file={file} "
                    f"gpu={gpu or 'inherited'} completed", flush=True,
                )
                with (task_dirs[index] / "pytest.log").open(errors="replace") as log:
                    shutil.copyfileobj(log, sys.stdout)
                print(f"\n[summary] backend={backend} file={file}: pytest rc={rc}", flush=True)
                del running[index]
                available.append(gpu)
            if running:
                time.sleep(0.1)
    finally:
        try:
            # Include compiler/profiler descendants on Ctrl-C or CI cancellation.
            for process, _handle, _gpu in running.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for process, handle, _gpu in running.values():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The leader may already have exited while a compiler child
                # ignored SIGTERM. Escalate for the group, not just its leader.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                handle.close()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

    for directory in task_dirs:
        for path in sorted(directory.glob("*.raw.jsonl")):
            with path.open("rb") as source, (raw_dir / path.name).open("ab") as target:
                shutil.copyfileobj(source, target)
    return codes


def collect_op_rows(
    op: str,
    files: list[str],
    rows_by_file_op: dict[str, list[dict]],
    derived_counts: dict[str, int],
) -> list[dict]:
    """Rows of one declared op, merged over its test files.

    Raw rows are written per test file (``<derived op>.raw.jsonl``);
    several files can map onto one op (e.g. via a shared ``PERF_OP``).
    The file lookup uses the same auto-qualified name the collection
    layer stamps on the raw file.
    """
    rows: list[dict] = []
    for file in files:
        rows.extend(
            rows_by_file_op.get(auto_op_name_for_file(file, derived_counts), [])
        )
    return rows


def print_benchmark_failures(
    runs: list[tuple[str, list[str], list[str]]],
    run_rcs: list[int],
    raw_dir: Path,
) -> None:
    """Repeat failed case details at the end, after later successful tasks."""
    from ci.optest_to_junit import _read_error_context

    tasks = [(backend, file) for backend, files, _ in runs for file in files]
    for index, ((backend, file), rc) in enumerate(zip(tasks, run_rcs), start=1):
        if rc == 0:
            continue
        log_path = raw_dir / "tasks" / f"{index:04d}" / "pytest.log"
        print(
            f"[FAILED] backend={backend} file={file}: pytest rc={rc}; "
            f"log={log_path}", flush=True,
        )
        print(_read_error_context(log_path.parent, log_path.name), flush=True)


def format_op_csv(op: str, rows: list[dict], output_dir: Path) -> Path:
    """Write the unified ``<op>.csv`` from the op's raw rows.

    Column order: ``operator``, ``backend``, ``kernels`` (kernel names
    launched inside the timing window; appears only when any row
    carries it), the case ``params`` in first-seen order, then the
    framework tail.  Framework columns:
    ``version`` (the dist version of the row's own toolchain -- the
    string embeds that toolchain's build commit, e.g.
    ``3.6.0+git46766868.staging``; ``""`` for unknown backends),
    ``arch``, ``timing`` (event / cudagraph, appears when any row
    carries it), ``stat`` (the statistic over iterations, appears only
    when at least one row used a non-default mode; default rows read
    ``median``), ``latency_ms``.
    ``dtype`` / ``tflops`` appear only when at least one row carries the
    value (``time_ms(dtype=...)`` / ``time_ms(flops=...)`` only).
    Failed benches leave ``latency_ms`` empty and fill the trailing
    ``error`` column (appears only when at least one row carries an
    error; a test that never measured anything contributes one such
    triage row so the op still gets its CSV).
    """
    param_cols: list[str] = []
    for row in rows:
        for key in row["params"]:
            if key not in param_cols:
                param_cols.append(key)
    fieldnames = ["operator", "backend"]
    if any(row.get("kernels") for row in rows):
        fieldnames.append("kernels")
    fieldnames.extend(param_cols)
    if any(row.get("dtype") for row in rows):
        fieldnames.append("dtype")
    fieldnames.append("version")
    fieldnames.append("arch")
    if any(row.get("timing") for row in rows):
        fieldnames.append("timing")
    if any(row.get("stat") for row in rows):
        fieldnames.append("stat")
    fieldnames.append("latency_ms")
    if any(row.get("flops") is not None for row in rows):
        fieldnames.append("tflops")
    if any(row.get("error") for row in rows):
        fieldnames.append("error")
    output_file = output_dir / f"{op}.csv"
    arch = get_arch()
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            backend = row.get("backend", "")
            latency = row.get("latency_ms")
            out = {
                "operator": row["operator"],
                "backend": backend,
                **row["params"],
                "arch": arch,
                "latency_ms": "" if latency is None else f"{latency:.6f}",
            }
            out["version"] = backend_version(backend)
            if "timing" in fieldnames:
                out["timing"] = row.get("timing") or ""
            if "stat" in fieldnames:
                out["stat"] = row.get("stat") or "median"
            if "kernels" in fieldnames:
                out["kernels"] = row.get("kernels") or ""
            if "dtype" in fieldnames:
                out["dtype"] = row.get("dtype") or ""
            if (
                "tflops" in fieldnames
                and row.get("flops") is not None
                and latency is not None
            ):
                out["tflops"] = f"{tflops(row['flops'], latency):.2f}"
            if "error" in fieldnames:
                out["error"] = row.get("error") or ""
            writer.writerow(out)
    return output_file


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        registry, file_specs, skipped_files = discover_ops()
    except RuntimeError as exc:
        print(f"[discover] {exc}", file=sys.stderr, flush=True)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default=None,
        help="comma-separated GPU IDs; default: parent optest --gpus, OPTEST_GPUS or visibility mask, else one inherited device",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="CSV output directory (default: <repo>/optest_results; must stay "
             "inside the source tree so optest can collect the changed CSVs)",
    )
    parser.add_argument(
        "--ops", nargs="+", choices=sorted(registry), default=None,
        help="ops to run (omit to run everything)",
    )
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="backends to run (omit to run every backend each selected op "
             "declares)",
    )
    args = parser.parse_args(argv)
    parent_pool = optest_gpu_pool(dict(os.environ))
    try:
        devices = resolve_gpu_pool(args.gpus, dict(os.environ), parent_pool)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = (args.output_dir or RESULTS_DIR).resolve()

    op_names = args.ops if args.ops is not None else sorted(registry)
    known_backends = sorted(
        {b for op in op_names for b in registry[op].backends}
    )
    unknown = [b for b in (args.backends or []) if b not in known_backends]
    if unknown:
        print(
            f"[backends] unknown: {unknown} "
            f"(available for selected ops: {known_backends})",
            file=sys.stderr,
            flush=True,
        )
        return 1

    runs, skipped = plan_runs(file_specs, op_names, args.backends)
    for op in skipped:
        print(f"[op] {op}: skipped (no backend selected)", file=sys.stderr, flush=True)
    for file, reason in skipped_files:
        print(f"[op] {file}: skipped (PERF_SKIP: {reason})", file=sys.stderr, flush=True)
    if not runs:
        print("nothing to run: every selected op was filtered out", file=sys.stderr, flush=True)
        return 1
    failures = 0
    csv_count = 0

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    optest_output = os.environ.get("OPTEST_OUTPUT_DIR")
    raw_dir = Path(optest_output) / "raw" if optest_output else output_dir / "raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if optest_output:
        print(
            f"[stage] CSVs staged at {output_dir}; optest copies the changed "
            f"ones to {optest_output}",
            flush=True,
        )

    # One pytest per file: a process-level crash (device VMFault abort,
    # segfault in a runtime handler) kills only that file's tests; the
    # remaining files still run and their ops still get CSVs.
    if parent_pool is not None:
        print(f"[schedule] recovered parent optest --gpus={parent_pool}", flush=True)
    with lease_gpu_pool(
        devices, os.environ.get("OPTEST_GPUS") if parent_pool is not None else None,
    ) as assigned:
        print(f"[schedule] GPU pool={assigned}; one pytest process per GPU", flush=True)
        run_rcs = run_benchmarks(runs, raw_dir, assigned)
    failures = sum(rc != 0 for rc in run_rcs)

    if runs:
        rows_by_file_op = read_rows(raw_dir)
        derived_counts = derived_op_counts(bench_files(TEST_DIR))
        any_rc_zero = 0 in run_rcs
        for op in op_names:
            if op in skipped:
                continue
            spec = registry[op]
            rows = collect_op_rows(
                op, list(spec.files), rows_by_file_op, derived_counts
            )
            if rows:
                output_file = format_op_csv(op, rows, output_dir)
                csv_count += 1
                print(
                    f"Wrote {output_file} ({len(rows)} rows, "
                    f"{time.monotonic() - started:.1f}s)",
                    flush=True,
                )
            else:
                why = (
                    "bench failed, or no observed timing calls"
                    if not any_rc_zero
                    else "bench tests skipped, or no observed timing calls"
                )
                print(
                    f"[op] {op}: no rows emitted ({why}; e.g. manual event "
                    "sampling) -- no CSV",
                    file=sys.stderr,
                    flush=True,
                )

    if failures:
        print_benchmark_failures(runs, run_rcs, raw_dir)
        print(f"{failures} benchmark failure(s)", file=sys.stderr)
        return 1
    print(f"Wrote {csv_count} CSV file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
