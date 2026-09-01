"""Correctness tests for CSA region layout + compression (stage 1)."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa import triton_csa_prefill
from boltops.step4_dsa.geometry import STEP4

from . import _common as C
from .gen import make_csa_prefill_inputs
from .ref import quantize_summary_e4m3, ref_csa_region_summary

ATOL = 1.0e-5
RTOL = 1.0e-5


@C.GPU_REQUIRED
@pytest.mark.parametrize(
    "seq_lens",
    [
        (64,),
        (1024,),
        (1020, 1017),
        (64, 8, 512),
    ],
)
def test_csa_prefill_matches_ref(seq_lens: tuple[int, ...]):
    device = torch.device(C.DEVICE)
    geo = STEP4
    regions_per_request = max(-(-length // geo.region_block_size) for length in seq_lens)
    metadata = make_csa_prefill_inputs(
        geo,
        seq_lens=seq_lens,
        regions_per_request=regions_per_request,
        device=device,
        seed=20260812 + sum(seq_lens),
    )
    summary, mean = triton_csa_prefill(
        metadata.index_k,
        metadata.index_z,
        metadata.seq_lens,
        regions_per_request=regions_per_request,
        region_size=geo.region_block_size,
    )
    torch.cuda.synchronize()

    total_regions = len(seq_lens) * regions_per_request
    assert summary.shape == (total_regions, geo.indexer_num_k_heads, geo.proxy_dim)
    assert mean.dtype == torch.float8_e4m3fn

    valid = metadata.token_valid & (metadata.flat_slot >= 0)
    slots = metadata.flat_slot[valid]
    rows = valid.nonzero(as_tuple=False).flatten()
    grouped: dict[int, list[int]] = {}
    for slot, row in zip(slots.cpu().tolist(), rows.cpu().tolist()):
        grouped.setdefault(int(slot), []).append(int(row))

    for region, token_rows in grouped.items():
        idx = torch.tensor(token_rows, device=device, dtype=torch.long)
        ref = ref_csa_region_summary(metadata.index_k[idx, 0], metadata.index_z[idx, 0])
        torch.testing.assert_close(
            summary[region, 0].cpu(), ref.cpu(), atol=ATOL, rtol=RTOL
        )
        actual_bytes = mean[region, 0].view(torch.uint8)
        expected_bytes = quantize_summary_e4m3(ref)
        assert torch.equal(actual_bytes.cpu(), expected_bytes.cpu())
