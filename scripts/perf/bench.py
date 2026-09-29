"""Minimal performance-benchmark framework for BoltOps.

A kernel benchmark is one module per backend directory
(``<backend>/<kernel>.py``) exposing a module-level ``SPEC``
(``KernelSpec``): CSV columns plus a generator of timed cases.  This
module owns the shared plumbing -- CUDA-graph timing (event timing as
fallback), progress output, and CSV writing -- so adding a kernel is
just dropping a new module, nothing to register.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import triton


@dataclass(frozen=True)
class Case:
    """One timed measurement.

    ``params`` must provide every column of the owning spec's
    ``fieldnames`` except ``latency_ms`` / ``tflops`` (filled in here).
    ``run`` is called once to warm up, then repeatedly by the timer;
    build inputs once per case and capture them in the closure.
    """

    label: str
    params: Mapping[str, object]
    run: Callable[[], None]
    flops: float


@dataclass(frozen=True)
class KernelSpec:
    backend: str
    kernel: str
    fieldnames: Sequence[str]
    cases: Callable[[], Iterator[Case]]
    warmup_ms: float = 10.0
    repeat_ms: float = 30.0

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.kernel}"

    @property
    def csv_name(self) -> str:
        return f"{self.backend}_{self.kernel}.csv"


def _time_cudagraph_ms(run: Callable[[], None], repeat_ms: float) -> float:
    """Median latency in ms via CUDA-graph replay (no host launch
    overhead).  Triton handles warmup, graph capture, and the timed
    replays; ``repeat_ms`` is the replay time budget."""
    return float(
        triton.testing.do_bench_cudagraph(
            run, rep=repeat_ms, return_mode="median",
        )
    )


def _time_event_ms(
    run: Callable[[], None], warmup_ms: float, repeat_ms: float
) -> float:
    return float(
        triton.testing.do_bench(
            run, warmup=warmup_ms, rep=repeat_ms, return_mode="median",
        )
    )


def _time_ms(
    run: Callable[[], None], warmup_ms: float, repeat_ms: float
) -> tuple[float, str]:
    """Median latency in ms plus the method used ("cudagraph"/"event").

    CUDA-graph capture is preferred; ops that cannot be captured fall
    back to plain event timing so a non-capturable kernel still runs.
    Errors unrelated to capture (e.g. a broken kernel) are re-raised
    so they fail loudly instead of being timed with the fallback path.
    """
    try:
        return _time_cudagraph_ms(run, repeat_ms), "cudagraph"
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
        return _time_event_ms(run, warmup_ms, repeat_ms), "event"


def _check_columns(spec: KernelSpec, case: Case) -> None:
    expected = set(spec.fieldnames) - {"latency_ms", "tflops"}
    actual = set(case.params)
    if expected != actual:
        raise ValueError(
            f"{spec.name}: case {case.label!r} params {sorted(actual)} "
            f"do not match fieldnames {sorted(expected)}"
        )


def run_case(spec: KernelSpec, case: Case) -> Mapping[str, object]:
    _check_columns(spec, case)
    case.run()
    latency_ms, method = _time_ms(case.run, spec.warmup_ms, spec.repeat_ms)
    tflops = case.flops / (latency_ms * 1e-3) / 1e12
    row = dict(case.params)
    row["latency_ms"] = f"{latency_ms:.6f}"
    row["tflops"] = f"{tflops:.2f}"
    print(
        f"[bench] {spec.kernel} {case.label} ({method}): "
        f"{latency_ms:.6f} ms, {tflops:.2f} TFLOPS",
        flush=True,
    )
    return row


def run_kernel(spec: KernelSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / spec.csv_name
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec.fieldnames)
        writer.writeheader()
        for case in spec.cases():
            writer.writerow(run_case(spec, case))
            handle.flush()
    print(f"Wrote {output_file}", flush=True)
    return output_file
