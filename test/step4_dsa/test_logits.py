# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for weighted-ReLU indexer logits (stage 2)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_indexer_logits
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .compare import compare
from .gen import make_indexer_inputs
from .ref import ref_indexer_logits

RTOL = 1e-3
ATOL = 5e-2


@C.GPU_REQUIRED
@pytest.mark.parametrize("seq_q,seq_k", [(64, 256), (128, 512)])
def test_indexer_logits_matches_ref(seq_q: int, seq_k: int):
    device = torch.device(C.DEVICE)
    inputs = make_indexer_inputs(
        STEP4,
        seq_q=seq_q,
        seq_k=seq_k,
        device=device,
        seed=20260811 + seq_q * 31 + seq_k,
    )
    actual = triton_indexer_logits(*inputs)
    reference = ref_indexer_logits(*inputs)
    torch.cuda.synchronize()
    assert tuple(actual.shape) == tuple(reference.shape)
    check = compare(actual, reference, rtol=RTOL, atol=ATOL)
    assert check.passed, check.format()
