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


def _classify_byte_diffs(
    triton_fp32: torch.Tensor,
    actual_bits: torch.Tensor,
    expected_bits: torch.Tensor,
) -> tuple[bool, bool]:
    """Split an FP8 byte difference into "our rounding is wrong" vs "the input moved".

    Returns ``(faithful, adjacent)``.

    ``faithful`` re-quantizes the kernel's *own* fp32 summary through the same
    fp32 -> bf16 -> e4m3 path and checks it reproduces the byte the kernel stored. If it
    does, the kernel's rounding is exactly the reference's and the byte difference is
    inherited from a different fp32 input, not produced by the store.

    ``adjacent`` checks every differing byte is one e4m3 code away. Together they say:
    the reduction landed a hair from an e4m3 decision boundary and the double rounding
    tipped it one step. A genuine rounding-path or reduction-order bug fails one of the
    two -- it lands on the wrong code by construction, or moves by more than one step.
    """
    requantized = quantize_summary_e4m3(triton_fp32)
    faithful = bool(torch.equal(requantized, actual_bits))
    differing = actual_bits != expected_bits
    codes = _E4M3_ORDER.to(actual_bits.device)
    steps = (
        codes[actual_bits[differing].int()] - codes[expected_bits[differing].int()]
    ).abs()
    adjacent = bool((steps <= 1).all())
    return faithful, adjacent


def _build_e4m3_order() -> torch.Tensor:
    """Rank of each e4m3 byte in ascending numeric order, for "is it one code away".

    e4m3 is sign-magnitude, so byte arithmetic is not value arithmetic (0x80 is -0, and
    the negative half counts downward). Ranking by value makes "adjacent code" mean what
    it says. NaN bytes are pinned outside the finite range so nothing reads as adjacent
    to them.
    """
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(values)
    order = torch.full((256,), 1 << 20, dtype=torch.int64)
    ranks = values[finite].argsort().argsort()
    order[finite] = ranks
    return order


_E4M3_ORDER = _build_e4m3_order()


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
        if torch.equal(actual_bytes, expected_bytes):
            continue
        faithful, adjacent = _classify_byte_diffs(
            summary[region, 0], actual_bytes, expected_bytes
        )
        assert faithful and adjacent, (
            f"region {region}: CSA e4m3 store is not a tie-flip "
            f"(faithful={faithful}, adjacent={adjacent})"
        )
