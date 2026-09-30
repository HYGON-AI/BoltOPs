"""Minimal performance-benchmark framework for BoltOps.

A kernel benchmark is one module per backend directory
(``<backend>/<kernel>.py``) exposing a module-level ``SPEC``
(``KernelSpec``): CSV columns plus a generator of timed cases.  This
module owns the shared plumbing -- CUDA-graph timing (event timing as
fallback), per-case failure isolation, progress output, and CSV
writing -- so adding a kernel is just dropping a new module, nothing
to register.  Columns in ``FRAMEWORK_COLS`` are filled in by the
framework automatically; a spec only declares them in ``fieldnames``.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Sequence

import triton

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Case:
    """One timed measurement.

    ``params`` must provide every column of the owning spec's
    ``fieldnames`` except the framework-filled ``FRAMEWORK_COLS``.
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


#: Columns filled in by the framework for every spec that declares
#: them: toolchain / device identity plus the timed results.  Specs
#: list the ones they want in ``fieldnames`` (fixed tail) but never
#: provide them in case ``params``.  ``dtype`` is kernel-specific and
#: stays spec-provided.
FRAMEWORK_COLS = ("arch", "triton_version", "latency_ms", "tflops")


def dtype_str(dtype: "torch.dtype") -> str:
    """CSV spelling of a torch dtype: ``torch.bfloat16`` -> ``bfloat16``."""
    return str(dtype).removeprefix("torch.")


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
    expected = set(spec.fieldnames) - set(FRAMEWORK_COLS)
    actual = set(case.params)
    if expected != actual:
        auto = sorted(set(spec.fieldnames) & set(FRAMEWORK_COLS))
        raise ValueError(
            f"{spec.name}: case {case.label!r} params {sorted(actual)} "
            f"do not match spec-provided fieldnames {sorted(expected)} "
            f"(framework-filled: {auto})"
        )


def _framework_values(latency_ms: float, tflops: float) -> dict[str, str]:
    from boltops.utility.arch_info import get_arch

    return {
        "arch": get_arch(),
        "triton_version": triton.__version__,
        "latency_ms": f"{latency_ms:.6f}",
        "tflops": f"{tflops:.2f}",
    }


def run_case(spec: KernelSpec, case: Case) -> Mapping[str, object]:
    _check_columns(spec, case)
    case.run()
    latency_ms, method = _time_ms(case.run, spec.warmup_ms, spec.repeat_ms)
    tflops = case.flops / (latency_ms * 1e-3) / 1e12
    values = _framework_values(latency_ms, tflops)
    row = dict(case.params)
    row.update({col: values[col] for col in spec.fieldnames if col in values})
    print(
        f"[bench] {spec.kernel} {case.label} ({method}): "
        f"{latency_ms:.6f} ms, {tflops:.2f} TFLOPS",
        flush=True,
    )
    return row


def run_kernel(spec: KernelSpec, output_dir: Path) -> int:
    """Time every case of ``spec`` and write its CSV.

    Returns the number of failed cases.  A failing case is reported
    and skipped, so one bad shape cannot discard the rest of the
    kernel's data; the caller folds the count into the exit code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / spec.csv_name
    failed = 0
    started = time.monotonic()
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec.fieldnames)
        writer.writeheader()
        for case in spec.cases():
            try:
                writer.writerow(run_case(spec, case))
            except Exception:
                failed += 1
                print(
                    f"[bench] {spec.name} case {case.label!r} FAILED",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
            handle.flush()
    elapsed = time.monotonic() - started
    if failed:
        print(
            f"Wrote {output_file} ({failed} case(s) failed, {elapsed:.1f}s)",
            flush=True,
        )
    else:
        print(f"Wrote {output_file} ({elapsed:.1f}s)", flush=True)
    return failed
