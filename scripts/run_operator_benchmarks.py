#!/usr/bin/env python3
"""Run the registered BoltOps kernel performance benchmarks.

This script is the optest ``per`` entry, so it stays at this fixed path.
Benchmarks live in ``scripts/perf/<backend>/<kernel>.py``; add one by
dropping a module that exposes a module-level ``SPEC``
(``perf.bench.KernelSpec``) -- nothing to register.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf import discover_specs  # noqa: E402
from perf.bench import run_kernel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    specs, import_errors = discover_specs()
    for error in import_errors:
        print(f"[discover] {error}", file=sys.stderr, flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("OPTEST_OUTPUT_DIR") or Path.cwd()
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path(default_output),
        help="CSV output directory (default: $OPTEST_OUTPUT_DIR, else current directory)",
    )
    parser.add_argument(
        "--kernels", nargs="+", choices=sorted(specs), default=None,
        help="kernels to run as 'backend/kernel' (default: all)",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()

    names = args.kernels or sorted(specs)
    kernel_failures: dict[str, int] = {}
    kernel_errors = 0
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        try:
            kernel_failures[name] = run_kernel(specs[name], output_dir)
        except Exception:
            kernel_errors += 1
            print(f"[{index}/{len(names)}] {name} FAILED", file=sys.stderr, flush=True)
            traceback.print_exc()

    case_failures = sum(kernel_failures.values())
    failures = len(import_errors) + kernel_errors + case_failures
    for name, count in kernel_failures.items():
        if count:
            print(f"[summary] {name}: {count} case(s) failed", file=sys.stderr, flush=True)
    if failures:
        print(f"{failures} benchmark failure(s)", file=sys.stderr)
        return 1
    print(f"Wrote {len(names)} CSV file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
