"""Production-shape performance benches for Step-4 DSA.

Shapes come from ``benchmark_cases.py`` (handoff ``dsa_benchmark_cases.py``).
Run with ``pytest test/step4_dsa --bench -s``.
"""

from __future__ import annotations

import pytest
import torch

from . import _common as C
from .bench_inputs import prepare_case
from .benchmark_cases import list_cases

PRODUCTION_CASES = tuple(list_cases("tp8-production"))
PRODUCTION_IDS = tuple(case.case_id for case in PRODUCTION_CASES)


@C.GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("case", PRODUCTION_CASES, ids=PRODUCTION_IDS)
def test_bench_tp8_production(case, time_ms, bench_warmup: int, bench_repeat: int):
    device = torch.device(C.DEVICE)
    fn = prepare_case(case, device=device)
    fn()
    torch.cuda.synchronize()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    print(f"[bench] {case.case_id} {case.operator} {case.params}: {ms:.3f} ms")
