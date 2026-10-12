#!/usr/bin/env python3
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
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import shutil
import subprocess
import sys
import time
import traceback
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


def run_op_benches(files: list[str], raw_dir: Path, backend: str) -> int:
    """Run one bench file's tests; they emit raw rows into ``raw_dir``.

    ``$OPTEST_BACKEND`` tells the op dispatch layers (and the collection
    hook's ``backend`` column) which backend this pass exercises.
    Returns the pytest exit code; ``-s`` keeps the bench's progress lines
    visible in the run log.
    """
    cmd = [
        sys.executable, "-m", "pytest", *files,
        "--bench", "-m", "bench",
        "--continue-on-collection-errors",
        "-q", "-s", "--no-header", "-p", "no:cacheprovider",
    ]
    env = dict(
        os.environ,
        OPTEST_BENCH_RAW_DIR=str(raw_dir),
        **{BACKEND_ENV: backend},
    )
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


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
    try:
        registry, file_specs, skipped_files = discover_ops()
    except RuntimeError as exc:
        print(f"[discover] {exc}", file=sys.stderr, flush=True)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
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
    run_rcs: list[int] = []
    total_runs = sum(len(files) for _, files, _ in runs)
    index = 0
    for backend, files, _run_ops in runs:
        for file in files:
            index += 1
            print(
                f"[{index}/{total_runs}] backend={backend} file={file}",
                flush=True,
            )
            try:
                rc = run_op_benches([file], raw_dir, backend)
            except Exception:
                failures += 1
                print(
                    f"[{index}/{total_runs}] backend={backend} "
                    f"file={file} FAILED",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                continue
            run_rcs.append(rc)
            if rc != 0:
                failures += 1
                print(
                    f"[summary] backend={backend} file={file}: "
                    f"pytest rc={rc}",
                    file=sys.stderr,
                    flush=True,
                )

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
        print(f"{failures} benchmark failure(s)", file=sys.stderr)
        return 1
    print(f"Wrote {csv_count} CSV file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
