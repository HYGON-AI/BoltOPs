#!/usr/bin/env python3
"""Run configured BoltOps kernel benchmarks and collect their CSV files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "scripts" / "benchmarks"

# Use ``backend/kernel`` so identically named Triton and TileLang kernels remain
# distinct. The entry resolves to ``<backend>/benchmark_<kernel>.py``.
BENCHMARK_KERNELS = (
    "triton/batched_gemm_bf16",
    "triton/flash_mla_sparse",
)


def find_benchmark_script(kernel_spec: str) -> Path:
    try:
        backend, kernel = kernel_spec.split("/", 1)
    except ValueError as exc:
        raise ValueError(
            f"invalid kernel {kernel_spec!r}; expected 'backend/kernel'"
        ) from exc
    if not backend or not kernel or "/" in kernel:
        raise ValueError(f"invalid kernel {kernel_spec!r}; expected 'backend/kernel'")
    expected_name = f"benchmark_{kernel}.py"
    benchmark_script = BENCHMARK_ROOT / backend / expected_name
    if not benchmark_script.is_file():
        raise FileNotFoundError(
            f"no benchmark script found for {kernel_spec!r}; "
            f"expected {benchmark_script}"
        )
    return benchmark_script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path.cwd(),
        help="CSV output directory (default: current directory)",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not BENCHMARK_KERNELS:
        parser.error("BENCHMARK_KERNELS is empty")

    failures = 0
    for index, kernel in enumerate(BENCHMARK_KERNELS, 1):
        try:
            benchmark_script = find_benchmark_script(kernel)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{index}/{len(BENCHMARK_KERNELS)}] {exc}", file=sys.stderr)
            failures += 1
            continue
        print(
            f"[{index}/{len(BENCHMARK_KERNELS)}] {kernel} -> "
            f"{benchmark_script.relative_to(ROOT)}",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--output-dir", str(output_dir)],
            cwd=ROOT,
            check=False,
        )
        failures += int(result.returncode != 0)

    if failures:
        print(f"{failures} kernel benchmark(s) failed", file=sys.stderr)
        return 1
    print(f"Wrote {len(BENCHMARK_KERNELS)} CSV file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
