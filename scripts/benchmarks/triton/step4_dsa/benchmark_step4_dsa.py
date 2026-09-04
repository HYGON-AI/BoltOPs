#!/usr/bin/env python3
"""Benchmark Step-4 DSA production shapes and write a CSV.

Shapes are the handoff ``tp8-production`` registry. Known width-513 / split-16
decode cases are included on purpose.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

# This file is one directory deeper than the other triton benches.
_TEST_DSA = Path(__file__).resolve().parents[4] / "test" / "step4_dsa"
if str(_TEST_DSA) not in sys.path:
    sys.path.insert(0, str(_TEST_DSA))

from bench_inputs import prepare_case  # noqa: E402
from benchmark_cases import list_cases  # noqa: E402

BACKEND = "triton"
KERNEL = "step4_dsa"
WARMUP = 10
REPEAT = 50


def _time_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        default="tp8-production",
        choices=("tp8-smoke", "tp8-production", "tp1-validation"),
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    device = torch.device("cuda")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{BACKEND}_{KERNEL}.csv"
    cases = list_cases(args.profile)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "backend",
                "operator",
                "case_id",
                "profile",
                "params",
                "latency_ms",
            ),
        )
        writer.writeheader()
        for case in cases:
            fn = prepare_case(case, device=device)
            fn()
            torch.cuda.synchronize()
            latency_ms = _time_ms(fn, args.warmup, args.repeat)
            writer.writerow(
                {
                    "backend": BACKEND,
                    "operator": case.operator,
                    "case_id": case.case_id,
                    "profile": case.profile,
                    "params": repr(case.params),
                    "latency_ms": f"{latency_ms:.4f}",
                }
            )
            print(f"{case.case_id}: {latency_ms:.3f} ms", flush=True)
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
