"""Shared pytest hooks for correctness vs performance suites.

Default: run functional/correctness tests only.
With ``--runslow``: also collect ``@pytest.mark.slow`` GPU grids (TileLang MQA
shape sweeps, extra kernels). CI omits these.
With ``--bench`` / ``--benchmark``: also collect ``@pytest.mark.bench`` cases
(real-model sizes, timing). Bench and correctness use different shape sets on
purpose.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch


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
        help="CUDA warmup iters for --bench/--benchmark (default: 10)",
    )
    group.addoption(
        "--bench-repeat",
        type=int,
        default=50,
        help="CUDA timed iters for --bench/--benchmark (default: 50)",
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
    for item in items:
        if not run_bench and "bench" in item.keywords:
            item.add_marker(skip_bench)
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)


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
def time_ms() -> Callable[[Callable[[], None], int, int], float]:
    def _time_ms(fn: Callable[[], None], warmup: int, repeat: int) -> float:
        if warmup < 0 or repeat <= 0:
            raise ValueError("warmup must be >= 0 and repeat must be > 0")
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

    return _time_ms
