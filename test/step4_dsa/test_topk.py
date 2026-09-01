"""Correctness tests for signed region top-k (stage 3)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_region_topk_ids, triton_topk_select_and_pack
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .gen import make_banded_tie_inputs, make_topk_inputs
from .ref import PREFILL_REGION_ID_MASK, ref_topk_select


def _sets(ids: torch.Tensor) -> list[set[int]]:
    return [{int(v) for v in row if int(v) >= 0} for row in ids.detach().cpu().tolist()]


@C.GPU_REQUIRED
@pytest.mark.parametrize("total_q,seq_regions,topk", [(64, 256, 16), (32, 128, 32)])
def test_region_topk_ids_matches_ref_set(total_q: int, seq_regions: int, topk: int):
    device = torch.device(C.DEVICE)
    geo = STEP4
    inputs = make_topk_inputs(
        geo, total_q=total_q, seq_regions=seq_regions, device=device, seed=20260813
    )
    actual = triton_region_topk_ids(inputs.logits, inputs.lengths, topk=topk)
    reference = ref_topk_select(inputs.logits, inputs.lengths, topk=topk)
    torch.cuda.synchronize()
    assert _sets(actual) == _sets(reference)
    live = actual >= 0
    assert int((live.sum(dim=1) == torch.minimum(inputs.lengths, torch.tensor(topk, device=device))).sum()) == actual.shape[0]


@C.GPU_REQUIRED
def test_topk_pack_physical_set_and_no_duplicates():
    device = torch.device(C.DEVICE)
    geo = STEP4
    inputs = make_topk_inputs(
        geo, total_q=32, seq_regions=128, device=device, seed=7
    )
    topk = 16
    packed = triton_topk_select_and_pack(
        inputs.logits,
        inputs.lengths,
        inputs.q_positions,
        inputs.block_table,
        topk=topk,
        region_size=geo.region_block_size,
        regions_per_page=geo.regions_per_page,
    )
    logical = triton_region_topk_ids(inputs.logits, inputs.lengths, topk=topk)
    torch.cuda.synchronize()
    phys = torch.where(packed < 0, packed, packed & PREFILL_REGION_ID_MASK)
    for row in range(packed.shape[0]):
        live_phys = [int(v) for v in phys[row].tolist() if int(v) >= 0]
        live_log = [int(v) for v in logical[row].tolist() if int(v) >= 0]
        assert len(live_phys) == len(set(live_phys))
        assert len(live_log) == len(set(live_log))
        assert len(live_phys) == min(topk, int(inputs.lengths[row]))


@C.GPU_REQUIRED
def test_topk_tie_band_takes_prefix():
    device = torch.device(C.DEVICE)
    geo = STEP4
    inputs = make_banded_tie_inputs(
        geo,
        total_q=16,
        seq_regions=64,
        device=device,
        seed=11,
        high_count=4,
        tied_count=12,
        constant_visible=32,
    )
    topk = 8
    ids = triton_region_topk_ids(inputs.logits, inputs.lengths, topk=topk)
    torch.cuda.synchronize()
    for row in range(ids.shape[0]):
        chosen = [int(v) for v in ids[row].tolist() if int(v) >= 0]
        assert len(chosen) == topk
        assert set(range(4)).issubset(chosen)
        assert all(v < 16 for v in chosen)
