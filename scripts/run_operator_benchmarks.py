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
    failures = len(import_errors)
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        try:
            run_kernel(specs[name], output_dir)
        except Exception:
            failures += 1
            print(f"[{index}/{len(names)}] {name} FAILED", file=sys.stderr, flush=True)
            traceback.print_exc()

    if failures:
        print(f"{failures} benchmark target(s) failed", file=sys.stderr)
        return 1
    print(f"Wrote {len(names)} CSV file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
