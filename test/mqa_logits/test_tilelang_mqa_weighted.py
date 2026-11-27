# SPDX-License-Identifier: MIT
"""Correctness tests for the weighted (non-QSA) TileLang MQA logits prefill.

Covers both backends behind ``tilelang_mqa_logits``: the gfx938 WASP kernel
(``@GFX938_REQUIRED``) and the architecture-independent generic kernel.

Weights are deliberately signed: a negative weight cannot be folded into the
ReLU, so a kernel that clamps after the multiply fails these cases.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mqa_logits import tilelang_mqa_logits
from boltops.mqa_logits.tilelang.configs.mqa_prefill import (
    resolve_get_config_weighted as resolve_get_weighted_prefill_config,
)
from boltops.mqa_logits.tilelang.kernels.common.mqa_logits import (
    weighted_mqa_prefill as generic_weighted_mqa_prefill,
    torch_mqa_logits,
)
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa_prefill_wasp_weighted import (
    _tilelang_mqa_prefill_wasp_weighted_kernel,
)
from boltops.utility.arch_info import get_device_info

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _is_gfx938_device() -> bool:
    if not torch.cuda.is_available():
        return False
    return str(get_device_info(DEVICE).arch).startswith("gfx938")


GFX938_REQUIRED = pytest.mark.skipif(
    not _is_gfx938_device(),
    reason="the weighted MQA logits WASP kernel requires gfx938",
)
SLOW = pytest.mark.slow


def _make_inputs(rows: int, keys: int, heads: int, *, seed: int = 0):
    """Ragged ranges plus signed weights, head_dim fixed at the WASP 128."""
    torch.manual_seed(seed)
    head_dim = 128
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    # Signed, O(1) magnitude: both ReLU branches and both weight signs stay
    # represented after the bf16 GEMM.
    weights = (
        torch.randn(rows, heads, device=DEVICE, dtype=torch.float32) * 0.5
    )
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    if rows >= 30 and keys >= 96:
        starts[10:20] = 13
        ends[10:20] = 79
        ends[20:30] = 48
    return q, kv, weights, starts, ends


def _assert_matches_torch(out, q, kv, weights, starts, ends):
    ref = torch_mqa_logits(q, kv, weights, starts, ends)
    assert out.shape == ref.shape
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=2e-3, rtol=2e-3)
    assert torch.equal(torch.isfinite(out), finite)


def _wasp_key_range_boundaries(keys: int, block_n: int = 128):
    """Keep the range matrix aligned with the QSA prefill boundary tests."""
    candidates = [
        (0, 1),
        (0, block_n - 1),
        (0, block_n),
        (0, block_n + 1),
        (0, 31 * block_n + 1),
        (0, 32 * block_n + 1),
        (0, 128 * block_n + 1),
        (0, 128 * block_n + 17),
        (0, 128 * block_n + 63),
        (0, 128 * block_n + 127),
        (13, 128 * block_n + 63),
        (11, 267),
        (11, 300),
        (0, keys - 1),
        (0, keys),
        (13, keys),
        (13, max(keys - 29, 13)),
    ]
    seen: set[tuple[int, int]] = set()
    ranges: list[tuple[int, int]] = []
    for start, end in candidates:
        start = max(0, min(start, keys))
        end = max(start, min(end, keys))
        pair = (start, end)
        if pair not in seen:
            seen.add(pair)
            ranges.append(pair)
    return ranges


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [2048, 8192, 32768, 65536])
@pytest.mark.parametrize("rows", [64, 65])
def test_weighted_matches_torch_key_length_boundaries(keys: int, rows: int):
    """Match the QSA public-entry grid, using signed head weights."""
    q, kv, weights, _, _ = _make_inputs(rows, keys, 4, seed=keys + rows)
    idx = torch.arange(rows, device=DEVICE, dtype=torch.int32)
    range_sets = [
        (
            torch.zeros(rows, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys, device=DEVICE, dtype=torch.int32),
        ),
        (
            idx % 17,
            torch.full((rows,), keys, device=DEVICE, dtype=torch.int32) - (idx % 29),
        ),
    ]
    for start, end in _wasp_key_range_boundaries(keys)[:8]:
        range_sets.append(
            (
                torch.full((rows,), start, device=DEVICE, dtype=torch.int32),
                torch.full((rows,), end, device=DEVICE, dtype=torch.int32),
            )
        )
    for starts, ends in range_sets:
        ends = torch.maximum(starts, ends)
        out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
        _assert_matches_torch(out, q, kv, weights, starts, ends)
        columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
        valid = (columns >= starts.unsqueeze(1)) & (columns < ends.unsqueeze(1))
        assert torch.all(torch.isneginf(out[~valid]))


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [2048, 8192, 65536])
def test_wasp_weighted_key_length_boundaries(keys: int):
    """Match the QSA direct-WASP ranges and phase settings on 2K/8K/64K."""
    block_q, rows, heads, head_dim = 64, 128, 4, 128
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=41)
    full_starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    full_ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    ref = torch_mqa_logits(q, kv, weights, full_starts, full_ends)
    sentinel = -1234567.0
    n_tile_phases = 32 if keys >= 32 * 1024 else 8
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=True,
        n_tile_phases=n_tile_phases,
        split_n=1,
        apply_scale=False,
    )
    for range_start, range_end in _wasp_key_range_boundaries(keys):
        starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
        ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
        out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)
        kernel(q.reshape(-1, head_dim), kv, weights, out, starts, ends, 1.0)
        torch.testing.assert_close(
            out[:, range_start:range_end],
            ref[:, range_start:range_end],
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.all(out[:, :range_start] == sentinel)
        assert torch.all(out[:, range_end:] == sentinel)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("block_q", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "range_start,range_len", [(0, 37), (13, 128), (7, 160), (0, 256), (11, 289)]
)
def test_wasp_weighted_tile_drains_matches_torch(
    range_start: int, range_len: int, block_q: int
):
    """Match QSA single-tile, even-pair, odd-tail and partial-tail drains."""
    rows, keys, heads, head_dim = block_q, 512, 4, 128
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=3)
    range_end = range_start + range_len
    starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
    out = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    # QSA uses the default block_n=64, not the public dispatch's block_n=128.
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        block_q=block_q,
        block_n=64,
        rows_divisible=True,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), kv, weights, out, starts, ends, 1.0)

    ref = torch_mqa_logits(q, kv, weights, starts, ends)
    torch.testing.assert_close(
        out[:, range_start:range_end],
        ref[:, range_start:range_end],
        atol=2e-3,
        rtol=2e-3,
    )


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("range_end", [267, 300])
def test_wasp_weighted_phase_preserves_invalid_columns(range_end: int):
    """Match QSA phased full/partial traversals, preserving invalid columns."""
    block_q, rows, keys, heads, head_dim = 64, 128, 512, 4, 128
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=31)
    range_start = 11
    sentinel = -1234567.0
    starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
    out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)

    # Direct compute only: the public entry's mask would replace sentinel.
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        block_q=block_q,
        block_n=64,
        rows_divisible=True,
        n_tile_phases=4,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), kv, weights, out, starts, ends, 1.0)

    ref = torch_mqa_logits(q, kv, weights, starts, ends)
    torch.testing.assert_close(
        out[:, range_start:range_end],
        ref[:, range_start:range_end],
        atol=2e-3,
        rtol=2e-3,
    )
    assert torch.all(out[:, :range_start] == sentinel)
    assert torch.all(out[:, range_end:] == sentinel)


@GPU_REQUIRED
@pytest.mark.parametrize("rows", [41, pytest.param(513, marks=SLOW)])
def test_weighted_prefill_public_matches_torch(rows: int):
    """Mirror QSA's public BF16 smoke/slow shapes and ragged windows."""
    q, kv, weights, starts, ends = _make_inputs(rows, 96, 4)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GPU_REQUIRED
@SLOW
def test_common_weighted_prefill_handles_partial_rows_without_host_padding():
    """Mirror QSA's common-entry 33-row/83-key tail case."""
    rows, keys = 33, 83
    q, kv, weights, _, _ = _make_inputs(rows, keys, 4, seed=1)
    starts = torch.arange(rows, device=DEVICE, dtype=torch.int32) % 9
    ends = torch.full((rows,), keys - 3, device=DEVICE, dtype=torch.int32)
    out = generic_weighted_mqa_prefill(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GPU_REQUIRED
@pytest.mark.parametrize("rows", [41, pytest.param(513, marks=SLOW)])
def test_generic_weighted_matches_torch(rows: int):
    """Generic (non-WASP) kernel: tail rows and ragged ranges."""
    q, kv, weights, starts, ends = _make_inputs(rows, 96, 4)
    out = generic_weighted_mqa_prefill(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GFX938_REQUIRED
@pytest.mark.parametrize("heads", [4, 8, 32, 64])
def test_wasp_weighted_matches_torch(heads: int):
    """Every WASP head count, with a row count that exercises the tail CTA."""
    q, kv, weights, starts, ends = _make_inputs(41, 96, heads)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("heads", [4, 8, 32, 64])
@pytest.mark.parametrize("rows", [64, 128, 256, 512, 1024])
def test_wasp_weighted_small_query_dispatch_matches_torch(rows: int, heads: int):
    """Small-Q dispatch, signed weights, ragged/empty windows and output masking."""
    keys = 4096
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=53)
    row_ids = torch.arange(rows, device=DEVICE, dtype=torch.int32)
    starts = (row_ids * 37) % 2049
    lengths = (row_ids * 131 + 17) % 2049
    lengths[row_ids % 19 == 0] = 0
    ends = starts + lengths
    destination = torch.full(
        (rows, keys), 1234567.0, device=DEVICE, dtype=torch.float32
    )
    out = tilelang_mqa_logits(
        q, kv, None, weights, starts, ends, D_out=destination
    )
    assert out.data_ptr() == destination.data_ptr()
    for begin in range(0, rows, 32):
        end = min(begin + 32, rows)
        _assert_matches_torch(
            out[begin:end], q[begin:end], kv, weights[begin:end],
            starts[begin:end], ends[begin:end],
        )
    columns = torch.arange(keys, device=DEVICE)[None, :]
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    assert torch.isneginf(out[~valid]).all()


@GFX938_REQUIRED
@pytest.mark.parametrize("heads", [4, 8, 32, 64])
def test_wasp_weighted_split_n128_matches_torch(heads: int):
    """Long keys and many rows reach logical-M=256, i.e. the splitN consumers."""
    rows, keys = 4608, 8192
    q, kv, weights, starts, ends = _make_inputs(rows, keys, heads, seed=2)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    # H64's full [rows, keys, heads] FP32 reference is large. Check all
    # rows in bounded-memory chunks, retaining the same tolerance.
    for begin in range(0, rows, 64):
        end = min(begin + 64, rows)
        _assert_matches_torch(
            out[begin:end], q[begin:end], kv, weights[begin:end],
            starts[begin:end], ends[begin:end],
        )


@GFX938_REQUIRED
def test_wasp_weighted_h8_split_n128_tile_and_row_tails():
    """Exercise H8 lo/hi drains, phase wrapping and a partial query CTA."""
    block_q, rows, keys, heads = 32, 193, 641, 8
    q, kv, weights, starts, ends = _make_inputs(rows, keys, heads, seed=19)
    ranges = [
        (0, 37), (13, 128), (7, 256), (0, 384),
        (11, 289), (0, 0), (13, 529),
    ]
    for block, (start, length) in enumerate(ranges):
        starts[block * block_q : (block + 1) * block_q] = start
        ends[block * block_q : (block + 1) * block_q] = start + length
    out = torch.full((rows, keys), -float("inf"), device=DEVICE)
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=False,
        n_tile_phases=4,
        split_n=1,
        apply_scale=False,
    )
    kernel(q.reshape(-1, 128), kv, weights, out, starts, ends, 1.0)
    _assert_matches_torch(out, q, kv, weights, starts, ends)
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    invalid = (columns < starts[:, None]) | (columns >= ends[:, None])
    assert torch.isneginf(out[invalid]).all()


@GFX938_REQUIRED
@pytest.mark.parametrize("n_tile_phases", [1, 4])
def test_wasp_weighted_h64_split_n128_tile_and_row_tails(n_tile_phases: int):
    """H64 lo/hi drains, phase wrapping and a last CTA with one valid row."""
    block_q, rows, keys, heads = 4, 25, 641, 64
    q, kv, weights, starts, ends = _make_inputs(rows, keys, heads, seed=19)
    ranges = [
        (0, 37), (13, 128), (7, 256), (0, 384),
        (11, 289), (0, 0), (13, 529),
    ]
    for block, (start, length) in enumerate(ranges):
        starts[block * block_q : (block + 1) * block_q] = start
        ends[block * block_q : (block + 1) * block_q] = start + length
    out = torch.full((rows, keys), -float("inf"), device=DEVICE)
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=False,
        n_tile_phases=n_tile_phases,
        split_n=1,
        apply_scale=False,
    )
    kernel(q.reshape(-1, 128), kv, weights, out, starts, ends, 1.0)
    _assert_matches_torch(out, q, kv, weights, starts, ends)
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    invalid = (columns < starts[:, None]) | (columns >= ends[:, None])
    assert torch.isneginf(out[invalid]).all()


@GFX938_REQUIRED
def test_wasp_weighted_rows_divisible_matches_torch():
    """rows % block_q == 0 compiles the predicate-free specialization."""
    q, kv, weights, starts, ends = _make_inputs(512, 1024, 4, seed=3)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GFX938_REQUIRED
def test_wasp_weighted_negative_weights_are_not_clamped():
    """All-negative weights: output must be <= 0 wherever any score is > 0."""
    q, kv, _, starts, ends = _make_inputs(64, 256, 4, seed=4)
    weights = torch.full((64, 4), -1.0, device=DEVICE, dtype=torch.float32)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    _assert_matches_torch(out, q, kv, weights, starts, ends)
    assert (out <= 0).all()


@GFX938_REQUIRED
def test_wasp_weighted_matches_generic_kernel():
    """The WASP and generic kernels must agree with each other, not just torch."""
    q, kv, weights, starts, ends = _make_inputs(41, 96, 4, seed=5)
    wasp = tilelang_mqa_logits(q, kv, None, weights, starts, ends)
    generic = generic_weighted_mqa_prefill(q, kv, None, weights, starts, ends)
    finite = torch.isfinite(generic)
    torch.testing.assert_close(wasp[finite], generic[finite], atol=2e-3, rtol=2e-3)
    assert torch.equal(torch.isfinite(wasp), finite)


@GPU_REQUIRED
def test_weighted_writes_into_d_out():
    q, kv, weights, starts, ends = _make_inputs(41, 96, 4, seed=6)
    d_out = torch.empty((41, 96), dtype=torch.float32, device=DEVICE)
    out = tilelang_mqa_logits(q, kv, None, weights, starts, ends, D_out=d_out)
    assert out.data_ptr() == d_out.data_ptr()
    _assert_matches_torch(out, q, kv, weights, starts, ends)


@GPU_REQUIRED
def test_weighted_rejects_bad_dtypes_and_shapes():
    q, kv, weights, starts, ends = _make_inputs(41, 96, 4, seed=7)
    scales = torch.ones(96, device=DEVICE, dtype=torch.float32)
    with pytest.raises((ValueError, NotImplementedError)):
        tilelang_mqa_logits(q, kv, scales, weights, starts, ends)
    with pytest.raises(ValueError):
        tilelang_mqa_logits(q, kv, None, weights[:, :2], starts, ends)
    with pytest.raises(ValueError):
        tilelang_mqa_logits(q.float(), kv.float(), None, weights, starts, ends)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("rows", [9152, 9280])
def test_wasp_weighted_persistent_partial_rounds(rows: int):
    """Uneven second rounds and a single-CTA third round.

    ``rows=9152/9280`` give 143/145 CTAs at ``block_q=64``, i.e. 1.99/2.01
    grid rounds on 72 CUs.  Every other weighted case fits in a single
    round, so this is the only coverage of CTA turnover.
    """
    torch.manual_seed(4)
    keys, heads, head_dim, block_q = 256, 4, 128, 64
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device=DEVICE, dtype=torch.float32) * 0.5
    cta = torch.arange(rows, device=DEVICE, dtype=torch.int32) // block_q
    starts = (cta % 2) * 128
    ends = starts + 128
    sentinel = -1234567.0
    out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        block_q=block_q, rows_divisible=True, apply_scale=False
    )
    kernel(q.reshape(-1, head_dim), k, weights, out, starts, ends, 1.0)

    # Alternating CTAs cover [0,128) and [128,256); the complement is left to
    # the separate mask kernel, which is not invoked on this direct call.
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    covered = (columns >= starts.unsqueeze(1)) & (columns < ends.unsqueeze(1))
    ref = torch_mqa_logits(q, k, weights, starts, ends)
    torch.testing.assert_close(out[covered], ref[covered], atol=2e-3, rtol=2e-3)
    assert torch.all(out[~covered] == sentinel)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("end_offset", [0, 1])
@pytest.mark.parametrize(
    "heads,block_q,n_tile_phases,split_n",
    [(4, 64, 4, 1), (4, 64, 32, 1), (32, 8, 16, 2)],
)
def test_wasp_weighted_partial_tail_keeps_phase_spacing(
    heads: int, block_q: int, n_tile_phases: int, split_n: int, end_offset: int
):
    """A peeled boundary tile must not perturb steady phase spacing.

    Include both QSA configurations and the existing H4 four-phase case.
    ``end_offset=0`` reproduces QSA's ragged ends; ``end_offset=1`` also
    gives the second query block a partial tail at a nonzero tile phase.
    """
    torch.manual_seed(37)
    rows, keys, head_dim = block_q * 2, 32 * 1024, 128
    sentinel = -1234567.0
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device=DEVICE, dtype=torch.float32) * 0.5
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    # QSA's last row ends exactly at keys.  Shifting all ends by one keeps
    # that block's boundary tile partial as well, exercising tail + wrap.
    positions = torch.arange(
        keys * 4 - rows, keys * 4, device=DEVICE, dtype=torch.int64
    )
    ends = (
        torch.div(positions + 1, 4, rounding_mode="floor").to(torch.int32)
        - end_offset
    )
    out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=True,
        n_tile_phases=n_tile_phases,
        split_n=split_n,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), k, weights, out, starts, ends, 1.0)

    # The compute kernel covers the CTA's union range; the separate mask
    # kernel clears each row's [end, block_end_max) region.  No mask kernel
    # runs here, so only the valid columns are checked and the uncovered
    # ones must still hold the sentinel.
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    valid = columns < ends.unsqueeze(1)
    ref = torch_mqa_logits(q, k, weights, starts, ends)
    torch.testing.assert_close(out[valid], ref[valid], atol=2e-3, rtol=2e-3)
    block_end_max = ends.reshape(-1, block_q).amax(dim=1)
    covered = columns < block_end_max.repeat_interleave(block_q).unsqueeze(1)
    assert torch.all(out[~covered] == sentinel)


@GFX938_REQUIRED
@SLOW
def test_wasp_weighted_phase_spacing_intermediate_boundaries():
    """Match QSA phase-count crossings and boundary remainders."""
    block_q, rows, keys, heads, head_dim = 64, 256, 32 * 1024, 4, 128
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=41)
    full_starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    full_ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    ref = torch_mqa_logits(q, kv, weights, full_starts, full_ends)
    sentinel = -1234567.0
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=True,
        n_tile_phases=32,
        split_n=1,
        apply_scale=False,
    )

    # Same seven ranges as QSA: around n_tiles == phases, long ranges with
    # different block_n remainders, and a nonzero start.
    for range_start, range_end in (
        (0, 31 * 128 + 1),
        (0, 32 * 128 + 1),
        (0, 128 * 128 + 1),
        (0, 128 * 128 + 17),
        (0, 128 * 128 + 63),
        (0, 128 * 128 + 127),
        (13, 128 * 128 + 63),
    ):
        starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
        ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
        out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)
        kernel(q.reshape(-1, head_dim), kv, weights, out, starts, ends, 1.0)
        torch.testing.assert_close(
            out[:, range_start:range_end],
            ref[:, range_start:range_end],
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.all(out[:, :range_start] == sentinel)
        assert torch.all(out[:, range_end:] == sentinel)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize(
    "heads,block_q",
    [(8, 4), (8, 32), (32, 2), (32, 8), (64, 2), (64, 4)],
)
@pytest.mark.parametrize(
    "block_n,row_blocks,range_start,range_end",
    [(64, 1, 13, 302), (128, 2, 0, 512)],
    ids=["qsa-range", "n128-full-range"],
)
def test_wasp_weighted_query_head_counts(
    heads: int,
    block_q: int,
    block_n: int,
    row_blocks: int,
    range_start: int,
    range_end: int,
):
    """Match QSA head layouts and ranges; retain the full-range N=128 cases."""
    rows, keys, head_dim = block_q * row_blocks, 512, 128
    seed = 4 if block_n == 64 else 11
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=seed)
    starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
    out = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=heads,
        block_q=block_q,
        block_n=block_n,
        rows_divisible=True,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), kv, weights, out, starts, ends, 1.0)

    ref = torch_mqa_logits(q, kv, weights, starts, ends)
    torch.testing.assert_close(
        out[:, range_start:range_end],
        ref[:, range_start:range_end],
        atol=2e-3,
        rtol=2e-3,
    )


@GFX938_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("rows", [
    pytest.param(64, marks=SLOW),
    pytest.param(128, marks=SLOW),
    256, 1024, 4096, 8192, 16384,
])
def test_bench_weighted_mqa_prefill_kernel(
    rows: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Profile the weighted WASP kernel with the same workload as QSA."""
    heads, head_dim, keys = 4, 128, 4096
    keys_per_block = 2048
    device_info = get_device_info(DEVICE)
    config = resolve_get_weighted_prefill_config(device_info.arch)(
        rows, keys, device_info.cu_count, num_heads=heads
    )
    q, kv, weights, _, _ = _make_inputs(rows, keys, heads, seed=2)
    logits = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    # Match QSA's workload: alternate the visited half of K per CTA.
    group = torch.arange(rows, device=DEVICE, dtype=torch.int32) // config.block_q
    starts = (group % 2) * keys_per_block
    ends = starts + keys_per_block
    kernel = _tilelang_mqa_prefill_wasp_weighted_kernel(
        heads=config.heads,
        head_dim=head_dim,
        block_n=config.block_n,
        block_q=config.block_q,
        threads=config.threads,
        rows_divisible=config.rows_divisible,
        n_tile_phases=config.n_tile_phases,
        split_n=config.split_n,
        apply_scale=False,
    )

    def fn():
        kernel(q.reshape(-1, head_dim), kv, weights, logits, starts, ends, 1.0)

    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    # Match QSA's GEMM FLOP accounting; excludes weighted reduction work.
    tflops = 2 * rows * heads * head_dim * keys_per_block / (ms * 1e9)
    print(
        f"[bench] weighted_mqa_prefill_kernel rows={rows} keys={keys} heads={heads} "
        f"head_dim={head_dim} range={keys_per_block} BLOCK_Q={config.block_q} "
        f"BLOCK_N={config.block_n} split_n={config.split_n} "
        f"n_tile_phases={config.n_tile_phases}: {ms:.4f} ms, "
        f"{tflops:.2f} TFLOP/s"
    )
