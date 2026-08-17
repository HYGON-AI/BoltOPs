"""Shared pytest hooks for correctness vs performance suites.

Default: run functional/correctness tests only.
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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bench: performance benchmark on real-model sizes (requires --bench/--benchmark)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--bench"):
        return
    skip_bench = pytest.mark.skip(
        reason="pass --bench or --benchmark to run performance benchmarks"
    )
    for item in items:
        if "bench" in item.keywords:
            item.add_marker(skip_bench)


@pytest.fixture
def bench_warmup(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bench-warmup"))


@pytest.fixture
def bench_repeat(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bench-repeat"))


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
