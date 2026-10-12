# SPDX-License-Identifier: MIT
"""Correctness tests for TileLang MQA logits prefill / decode.

Requires ``tilelang`` (``pytest.importorskip``). FP8 cases need gfx938
(``@GFX938_REQUIRED``); on other arches they skip.

CI default (no ``--runslow``) FP8 smoke:

- ``test_qsa_mqa_prefill_fp8_matches_torch``
- ``test_qsa_mqa_decode_fp8_packed_matches_torch``
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mqa_logits import tilelang_qsa_mqa_decode, tilelang_qsa_mqa_prefill
from boltops.mqa_logits.tilelang.kernels.common.mqa import (
    _get_decode_suffix_splits,
    _tilelang_qsa_mqa_mask_kernel,
    _tilelang_qsa_mqa_decode_mask_kernel,
    qsa_mqa_decode as common_qsa_mqa_decode,
    tilelang_qsa_mqa_prefill as common_tilelang_qsa_mqa_prefill,
    torch_qsa_mqa_decode,
    torch_qsa_mqa_prefill,
)
from boltops.mqa_logits.tilelang.kernels.common.mqa_decode import (
    _tilelang_qsa_mqa_decode_kernel,
    _tilelang_qsa_mqa_decode_persistent_kernel,
)
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa_prefill_wasp import (
    _tilelang_qsa_mqa_prefill_wasp_kernel,
)
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa import (
    prefill_mask_launch_kwargs as _landed_prefill_mask_kwargs,
)
from boltops.mqa_logits.tilelang.configs.mqa_prefill import (
    resolve_get_config as resolve_get_prefill_config,
)
from boltops.mqa_logits.tilelang.configs.mqa_decode import resolve_get_config
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
    reason="TileLang FP8 QSA MQA requires gfx938",
)
SLOW = pytest.mark.slow


@pytest.mark.parametrize(
    "batch,keys,expected",
    [
        (16, 8 * 1024, 1),
        (16, 16 * 1024, 4),
        (32, 64 * 1024, 2),
        (48, 64 * 1024, 2),
        (64, 64 * 1024, 1),
        (4, 64 * 1024, 16),
    ],
)
def test_qsa_mqa_decode_suffix_split_policy(
    batch: int, keys: int, expected: int
):
    assert _get_decode_suffix_splits(batch, keys) == expected


def test_qsa_mqa_decode_rejects_host_tensors():
    q = torch.empty(1, 4, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(1, 16, 1, 128, dtype=torch.bfloat16)
    page_table = torch.zeros(1, 1, dtype=torch.int32)
    context_lens = torch.ones(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="requires device tensors"):
        common_qsa_mqa_decode(q, k_cache, page_table, context_lens, 16)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("suffix_splits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("reverse_suffix", [False, True])
def test_qsa_mqa_decode_suffix_mask_boundaries(
    suffix_splits: int, reverse_suffix: bool
):
    batch, keys = 5, 4097
    logits = torch.full((batch, keys), 7.0, device=DEVICE, dtype=torch.float32)
    context_lens = torch.tensor(
        [0, 1, 1023, 2048, keys], device=DEVICE, dtype=torch.int32
    )
    _tilelang_qsa_mqa_decode_mask_kernel(
        suffix_splits=suffix_splits,
        reverse_suffix=reverse_suffix,
    )(logits, context_lens)
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    valid = columns < context_lens.unsqueeze(1)
    assert torch.all(logits[valid] == 7.0)
    assert torch.all(torch.isneginf(logits[~valid]))


def _assert_prefill_mask(
    logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, fill: float = 7.0
):
    keys = logits.shape[1]
    columns = torch.arange(keys, device=logits.device).unsqueeze(0)
    start = starts.clamp(0, keys)
    end = torch.maximum(start, ends.clamp(0, keys)).unsqueeze(1)
    start = start.unsqueeze(1)
    valid = (columns >= start) & (columns < end)
    assert torch.all(logits[valid] == fill)
    assert torch.all(torch.isneginf(logits[~valid]))


def _prefill_mask_range_cases(rows: int, keys: int):
    idx = torch.arange(rows, device=DEVICE, dtype=torch.int32)
    round_elems = 1024
    cases = {
        "ragged": (
            idx % 17,
            keys - (idx % 29),
        ),
        "full": (
            torch.zeros(rows, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys, device=DEVICE, dtype=torch.int32),
        ),
        "empty": (
            torch.full((rows,), keys // 2, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys // 2, device=DEVICE, dtype=torch.int32),
        ),
        "prefix_only": (
            torch.full((rows,), 13, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys, device=DEVICE, dtype=torch.int32),
        ),
        "suffix_only": (
            torch.zeros(rows, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), max(keys - 13, 0), device=DEVICE, dtype=torch.int32),
        ),
        "clamped": (
            torch.full((rows,), keys + 8, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys + 16, device=DEVICE, dtype=torch.int32),
        ),
        "start_gt_end": (
            torch.full((rows,), min(100, keys), device=DEVICE, dtype=torch.int32),
            torch.full((rows,), min(50, keys), device=DEVICE, dtype=torch.int32),
        ),
    }
    if keys > round_elems:
        cases["round_minus"] = (
            torch.full((rows,), round_elems - 1, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys - (round_elems - 1), device=DEVICE, dtype=torch.int32),
        )
        cases["round_exact"] = (
            torch.full((rows,), round_elems, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys - round_elems, device=DEVICE, dtype=torch.int32),
        )
        cases["round_plus"] = (
            torch.full((rows,), round_elems + 1, device=DEVICE, dtype=torch.int32),
            torch.full((rows,), keys - (round_elems + 1), device=DEVICE, dtype=torch.int32),
        )
    mixed_starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    mixed_ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    mixed_starts[0] = 0
    mixed_ends[0] = 0
    mixed_starts[-1] = min(13, keys)
    mixed_ends[-1] = keys
    if rows > 2:
        mixed_starts[1] = min(round_elems + 1, keys)
        mixed_ends[1] = max(mixed_starts[1], keys - 3)
    cases["mixed_rows"] = (mixed_starts, mixed_ends)
    return cases


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("rows", [5, 33])
@pytest.mark.parametrize("reverse_suffix", [False, True])
def test_qsa_mqa_prefill_mask_row_phase_boundaries(
    rows: int, reverse_suffix: bool
):
    keys = 4097
    logits = torch.full((rows, keys), 7.0, device=DEVICE, dtype=torch.float32)
    starts = torch.arange(rows, device=DEVICE, dtype=torch.int32) % 17
    ends = keys - (torch.arange(rows, device=DEVICE, dtype=torch.int32) % 29)
    _tilelang_qsa_mqa_mask_kernel(
        row_phases=4,
        reverse_suffix=reverse_suffix,
    )(logits, starts, ends)
    _assert_prefill_mask(logits, starts, ends)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [1024, 8192, 32768, 65536])
@pytest.mark.parametrize("reverse_suffix", [False, True])
@pytest.mark.parametrize("steady_phases,row_phases", [(1, 1), (16, 4)])
def test_qsa_mqa_prefill_mask_key_and_range_boundaries(
    keys: int, reverse_suffix: bool, steady_phases: int, row_phases: int
):
    """Reverse and phase are correctness paths on every key length, not 64K-only."""
    rows = 33
    kernel = _tilelang_qsa_mqa_mask_kernel(
        steady_phases=steady_phases,
        row_phases=row_phases,
        reverse_suffix=reverse_suffix,
    )
    for starts, ends in _prefill_mask_range_cases(rows, keys).values():
        logits = torch.full((rows, keys), 7.0, device=DEVICE, dtype=torch.float32)
        kernel(logits, starts, ends)
        _assert_prefill_mask(logits, starts, ends)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [8192, 65536])
@pytest.mark.parametrize("rows", [4097, 4608])
@pytest.mark.parametrize("reverse_suffix", [False, True])
def test_qsa_mqa_prefill_mask_large_row_reverse(
    keys: int, rows: int, reverse_suffix: bool
):
    kernel = _tilelang_qsa_mqa_mask_kernel(
        steady_phases=16,
        row_phases=4,
        reverse_suffix=reverse_suffix,
    )
    for name in ("ragged", "suffix_only", "round_plus", "mixed_rows"):
        starts, ends = _prefill_mask_range_cases(rows, keys)[name]
        logits = torch.full((rows, keys), 7.0, device=DEVICE, dtype=torch.float32)
        kernel(logits, starts, ends)
        _assert_prefill_mask(logits, starts, ends)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [8192, 32768, 65536])
@pytest.mark.parametrize("rows", [33, 4097, 4608])
def test_qsa_mqa_prefill_mask_landed_dispatch(keys: int, rows: int):
    kwargs = _landed_prefill_mask_kwargs(rows, keys)
    kernel = _tilelang_qsa_mqa_mask_kernel(**kwargs)
    starts, ends = _prefill_mask_range_cases(rows, keys)["ragged"]
    logits = torch.full((rows, keys), 7.0, device=DEVICE, dtype=torch.float32)
    kernel(logits, starts, ends)
    _assert_prefill_mask(logits, starts, ends)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "rows",
    [
        41,
        pytest.param(513, marks=SLOW),
    ],
)
def test_qsa_mqa_prefill_matches_torch(rows: int):
    torch.manual_seed(0)
    keys, heads, head_dim = 96, 4, 128
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    starts[10:20] = 13
    ends[10:20] = 79
    ends[20:30] = 48
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_prefill(q, k, starts, ends, scale)
    out = tilelang_qsa_mqa_prefill(q, k, starts, ends, scale)
    assert out.shape == (rows, keys)
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)
    assert torch.equal(torch.isfinite(out), finite)


def _assert_fp8_packed_decode_matches_torch(heads: int, *, seed: int = 3) -> None:
    from boltops.mqa_logits.tilelang.kernels.common.mqa import pack_fp8_paged_k_cache

    torch.manual_seed(seed)
    batch, head_dim, page_size = 3, 128, 16
    max_pages = 16
    pages = 20
    max_model_len = max_pages * page_size
    q_f = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.float32)
    k_f = torch.randn(pages, page_size, 1, head_dim, device=DEVICE, dtype=torch.float32)
    scales = k_f.abs().amax(dim=-1).clamp(min=1e-4) / 240.0
    k_fp8 = (k_f / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    q = q_f.to(torch.float8_e4m3fn)
    packed = pack_fp8_paged_k_cache(k_fp8, scales.squeeze(2))
    page_table = torch.zeros(batch, max_pages, device=DEVICE, dtype=torch.int32)
    for b in range(batch):
        page_table[b] = torch.arange(max_pages, device=DEVICE) % pages
    context_lens = torch.tensor([17, 63, 95], device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_decode(
        q, packed, page_table, context_lens, max_model_len, scale
    )
    out = tilelang_qsa_mqa_decode(
        q, packed, page_table, context_lens, max_model_len, scale
    )
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)
    assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@GFX938_REQUIRED
def test_qsa_mqa_prefill_fp8_matches_torch():
    """CI default: small shape + ragged row windows (gfx938 only)."""
    torch.manual_seed(0)
    rows, keys, heads, head_dim = 41, 96, 4, 128
    fp8 = torch.float8_e4m3fn
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.float32).to(fp8)
    k = torch.randn(keys, 1, head_dim, device=DEVICE, dtype=torch.float32).to(fp8)
    kv_scales = torch.rand(keys, device=DEVICE, dtype=torch.float32) + 0.25
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    starts[10:20] = 13
    ends[10:20] = 79
    ends[20:30] = 48
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_prefill(q, k, starts, ends, scale, kv_scales)
    out = tilelang_qsa_mqa_prefill(q, k, starts, ends, scale, kv_scales)
    assert out.shape == (rows, keys)
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)
    assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("heads", [32, 64])
def test_qsa_mqa_prefill_fp8_heuristic_shape_matches_torch(heads: int):
    """rows=4608 keys=8192; extra heads compile distinct WASP configs."""
    torch.manual_seed(11)
    rows, keys, head_dim = 4608, 8192, 128
    fp8 = torch.float8_e4m3fn
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.float32).to(fp8)
    k = torch.randn(keys, 1, head_dim, device=DEVICE, dtype=torch.float32).to(fp8)
    kv_scales = torch.rand(keys, device=DEVICE, dtype=torch.float32) + 0.25
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), keys, device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_prefill(q, k, starts, ends, scale, kv_scales)
    out = tilelang_qsa_mqa_prefill(q, k, starts, ends, scale, kv_scales)
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)
    assert torch.equal(torch.isfinite(out), finite)


def _wasp_key_range_boundaries(keys: int, block_n: int = 128):
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
def test_qsa_mqa_prefill_matches_torch_key_length_boundaries(keys: int, rows: int):
    """Public prefill (WASP + landed mask) across key lengths and range tails."""
    torch.manual_seed(keys + rows)
    heads, head_dim = 4, 128
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, head_dim, device=DEVICE, dtype=torch.bfloat16)
    scale = math.sqrt(head_dim)
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
        ref = torch_qsa_mqa_prefill(q, k, starts, ends, scale)
        out = tilelang_qsa_mqa_prefill(q, k, starts, ends, scale)
        finite = torch.isfinite(ref)
        torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)
        assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("keys", [2048, 8192, 65536])
def test_qsa_mqa_prefill_wasp_key_length_boundaries(keys: int):
    """Reuse WASP remainder/phase cases on 2K/8K/64K tensors."""
    torch.manual_seed(41)
    block_q, rows, heads, head_dim = 64, 128, 4, 128
    sentinel = -1234567.0
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    n_tile_phases = 32 if keys >= 32 * 1024 else 8
    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
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
        kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)
        torch.testing.assert_close(
            out[:, range_start:range_end],
            ref[:, range_start:range_end],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(out[:, :range_start] == sentinel)
        assert torch.all(out[:, range_end:] == sentinel)


@GPU_REQUIRED
@SLOW
def test_common_mqa_prefill_handles_partial_rows_without_host_padding():
    torch.manual_seed(1)
    rows, keys, heads, head_dim = 33, 83, 4, 128
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.arange(rows, device=DEVICE, dtype=torch.int32) % 9
    ends = torch.full((rows,), keys - 3, device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)

    ref = torch_qsa_mqa_prefill(q, k, starts, ends, scale)
    out = common_tilelang_qsa_mqa_prefill(q, k, starts, ends, scale)

    assert out.shape == (rows, keys)
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)
    assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("block_q", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "range_start,range_len", [(0, 37), (13, 128), (7, 160), (0, 256), (11, 289)]
)
def test_qsa_mqa_prefill_wasp_matches_torch(
    range_start: int, range_len: int, block_q: int
):
    """Cover the WASP single-tile, even-pair, odd-tail and partial-tail drains."""
    torch.manual_seed(3)
    rows, keys, heads, head_dim = block_q, 512, 4, 128
    range_end = range_start + range_len
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
    out = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        block_q=block_q, rows_divisible=True, apply_scale=False
    )
    kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)

    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    torch.testing.assert_close(
        out[:, range_start:range_end],
        ref[:, range_start:range_end],
        atol=1e-4,
        rtol=1e-4,
    )


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("range_end", [267, 300])
def test_qsa_mqa_prefill_wasp_phase_preserves_invalid_columns(range_end: int):
    """Phased full/partial N traversals must preserve columns outside the range."""
    torch.manual_seed(31)
    block_q, rows, keys, heads, head_dim = 64, 128, 512, 4, 128
    range_start = 11
    sentinel = -1234567.0
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.full((rows,), range_start, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
    out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        block_q=block_q,
        rows_divisible=True,
        n_tile_phases=4,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)

    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    torch.testing.assert_close(
        out[:, range_start:range_end],
        ref[:, range_start:range_end],
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(out[:, :range_start] == sentinel)
    assert torch.all(out[:, range_end:] == sentinel)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize(
    "heads,block_q,n_tile_phases,split_n",
    [(4, 64, 32, 1), (32, 8, 16, 2)],
)
def test_qsa_mqa_prefill_wasp_partial_tail_keeps_full_range_phase_spacing(
    heads: int, block_q: int, n_tile_phases: int, split_n: int
):
    """A peeled boundary tile must not perturb steady phase spacing."""
    torch.manual_seed(37)
    rows, keys, head_dim = block_q * 2, 32 * 1024, 128
    sentinel = -1234567.0
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    positions = torch.arange(
        keys * 4 - rows, keys * 4, device=DEVICE, dtype=torch.int64
    )
    ends = torch.div(positions + 1, 4, rounding_mode="floor").to(torch.int32)
    out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=True,
        n_tile_phases=n_tile_phases,
        split_n=split_n,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)

    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    columns = torch.arange(keys, device=DEVICE).unsqueeze(0)
    valid = columns < ends.unsqueeze(1)
    torch.testing.assert_close(out[valid], ref[valid], atol=1e-4, rtol=1e-4)
    # The compute kernel processes the union range of every block_q group;
    # the separate mask kernel clears each row's [end, block_end_max) region.
    block_end_max = ends.reshape(-1, block_q).amax(dim=1)
    covered = columns < block_end_max.repeat_interleave(block_q).unsqueeze(1)
    assert torch.all(out[~covered] == sentinel)


@GPU_REQUIRED
@SLOW
def test_qsa_mqa_prefill_wasp_phase_spacing_intermediate_boundaries():
    """Cover phase-count crossings and representative boundary remainders."""
    torch.manual_seed(41)
    block_q, rows, keys, heads, head_dim = 64, 256, 32 * 1024, 4, 128
    sentinel = -1234567.0
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        heads=heads,
        block_q=block_q,
        block_n=128,
        rows_divisible=True,
        n_tile_phases=32,
        split_n=1,
        apply_scale=False,
    )

    # Around n_tiles == phases, well inside a long range, every important
    # block_n remainder, and a nonzero start supported by the public contract.
    for range_start, range_end in (
        (0, 31 * 128 + 1),
        (0, 32 * 128 + 1),
        (0, 128 * 128 + 1),
        (0, 128 * 128 + 17),
        (0, 128 * 128 + 63),
        (0, 128 * 128 + 127),
        (13, 128 * 128 + 63),
    ):
        starts = torch.full(
            (rows,), range_start, device=DEVICE, dtype=torch.int32
        )
        ends = torch.full((rows,), range_end, device=DEVICE, dtype=torch.int32)
        out = torch.full((rows, keys), sentinel, device=DEVICE, dtype=torch.float32)
        kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)
        torch.testing.assert_close(
            out[:, range_start:range_end],
            ref[:, range_start:range_end],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(out[:, :range_start] == sentinel)
        assert torch.all(out[:, range_end:] == sentinel)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize(
    "heads,block_q",
    [(8, 4), (8, 32), (32, 2), (32, 8), (64, 2), (64, 4)],
)
def test_qsa_mqa_prefill_wasp_supports_query_head_counts(heads: int, block_q: int):
    torch.manual_seed(4)
    rows, keys, head_dim = block_q, 512, 128
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    starts = torch.full((rows,), 13, device=DEVICE, dtype=torch.int32)
    ends = torch.full((rows,), 302, device=DEVICE, dtype=torch.int32)
    out = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        heads=heads,
        block_q=block_q,
        rows_divisible=True,
        apply_scale=False,
    )
    kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)

    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    torch.testing.assert_close(out[:, 13:302], ref[:, 13:302], atol=1e-4, rtol=1e-4)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("rows", [9152, 9280])
def test_qsa_mqa_prefill_wasp_persistent_partial_rounds(rows: int):
    """Exercise uneven second rounds and a single-CTA third round."""
    torch.manual_seed(4)
    keys, heads, head_dim, block_q = 256, 4, 128, 64
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    cta = torch.arange(rows, device=DEVICE, dtype=torch.int32) // block_q
    starts = (cta % 2) * 128
    ends = starts + 128
    out = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
        block_q=block_q, rows_divisible=True, apply_scale=False
    )
    kernel(q.reshape(-1, head_dim), k, out, starts, ends, 1.0)

    ref = torch.relu(torch.einsum("mhd,nd->mnh", q.float(), k.float())).sum(-1)
    columns = torch.arange(keys, device=DEVICE)[None, :]
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    torch.testing.assert_close(out[valid], ref[valid], atol=1e-4, rtol=1e-4)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "page_size,heads",
    [
        (16, 4),
        pytest.param(8, 4, marks=SLOW),
        pytest.param(32, 4, marks=SLOW),
        pytest.param(64, 4, marks=SLOW),
        pytest.param(16, 8, marks=SLOW),
        pytest.param(16, 16, marks=SLOW),
        pytest.param(16, 32, marks=SLOW),
        pytest.param(16, 64, marks=SLOW),
    ],
)
def test_qsa_mqa_decode_matches_torch(page_size: int, heads: int):
    torch.manual_seed(1)
    batch, head_dim = 3, 128
    max_pages = 16
    pages = 20
    max_model_len = max_pages * page_size
    q = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn(
        pages, page_size, 1, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    page_table = torch.zeros(batch, max_pages, device=DEVICE, dtype=torch.int32)
    for b in range(batch):
        page_table[b] = torch.arange(max_pages, device=DEVICE) % pages
    # Deliberately cross both physical-page and 64-token GEMM-group tails.
    context_lens = torch.tensor([17, 63, 95], device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, scale
    )
    out = tilelang_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, scale
    )
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)
    assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@GFX938_REQUIRED
def test_qsa_mqa_decode_fp8_packed_matches_torch():
    """CI default: batch=3 H=4 packed cache (gfx938 only)."""
    _assert_fp8_packed_decode_matches_torch(4)


@GPU_REQUIRED
@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("heads", [16, 32, 64])
def test_qsa_mqa_decode_fp8_packed_extra_heads(heads: int):
    _assert_fp8_packed_decode_matches_torch(heads)


@GPU_REQUIRED
@SLOW
def test_qsa_mqa_decode_group128_matches_torch():
    torch.manual_seed(31)
    batch, heads, head_dim, page_size = 3, 16, 128, 16
    max_model_len = 256
    max_pages = max_model_len // page_size
    pages = batch * max_pages
    q = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn(
        pages, page_size, 1, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        pages, device=DEVICE, dtype=torch.int32
    ).reshape(batch, max_pages)
    context_lens = torch.tensor([17, 127, 191], device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, scale
    )
    out = torch.full_like(ref, -float("inf"))
    _tilelang_qsa_mqa_decode_kernel(
        heads=heads,
        head_dim=head_dim,
        page_size=page_size,
        group_size=128,
        groups_per_cta=4,
        num_stages=1,
        use_async=True,
    )(
        q.unsqueeze(1),
        k_cache,
        page_table,
        context_lens,
        out,
        scale,
    )
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)


@GPU_REQUIRED
@SLOW
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("persistent_splits", [2, 4, 8, 16, 32])
def test_qsa_mqa_decode_group128_persistent_matches_torch(
    persistent_splits: int, heads: int,
):
    torch.manual_seed(32)
    batch, head_dim, page_size = 3, 128, 16
    max_model_len = 512
    max_pages = max_model_len // page_size
    pages = batch * max_pages
    q = torch.randn(batch, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn(
        pages, page_size, 1, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    page_table = torch.randperm(
        pages, device=DEVICE, dtype=torch.int32
    ).reshape(batch, max_pages)
    context_lens = torch.tensor([17, 256, 447], device=DEVICE, dtype=torch.int32)
    scale = math.sqrt(head_dim)
    ref = torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, scale
    )
    out = torch.full_like(ref, -float("inf"))
    _tilelang_qsa_mqa_decode_persistent_kernel(
        heads=heads,
        head_dim=head_dim,
        page_size=page_size,
        group_size=128,
        persistent_splits=persistent_splits,
        num_stages=1,
        use_async=True,
    )(
        q.unsqueeze(1),
        k_cache,
        page_table,
        context_lens,
        out,
        scale,
    )
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)
    assert torch.equal(torch.isfinite(out), finite)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("rows", [4096, 8192, 16384])
def test_bench_qsa_mqa_prefill_kernel(
    rows: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Time the prefill WASP kernel with the gfx938 default get_config."""
    heads, head_dim, keys = 4, 128, 4096
    keys_per_block = 2048
    device_info = get_device_info(DEVICE)
    config = resolve_get_prefill_config(device_info.arch)(
        rows, keys, device_info.cu_count, num_heads=heads
    )

    torch.manual_seed(2)
    q = torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(keys, head_dim, device=DEVICE, dtype=torch.bfloat16)
    logits = torch.empty((rows, keys), device=DEVICE, dtype=torch.float32)

    # Alternate the half of K visited by adjacent CTAs.  Every BLOCK_Q-row CTA
    # sees one exact 2048-key interval, exercising realistic range reduction
    # without making different rows in a CTA expand the union of their ranges.
    cta = torch.arange(rows, device=DEVICE, dtype=torch.int32) // config.block_q
    starts = (cta % 2) * keys_per_block
    ends = starts + keys_per_block

    kernel = _tilelang_qsa_mqa_prefill_wasp_kernel(
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
        kernel(q.reshape(-1, head_dim), k, logits, starts, ends, 1.0)

    fn()
    ms = time_ms(fn, bench_warmup, bench_repeat)
    tflops = 2 * rows * heads * head_dim * keys_per_block / (ms * 1e9)
    print(
        f"[bench] qsa_mqa_prefill_kernel rows={rows} keys={keys} heads={heads} "
        f"head_dim={head_dim} range={keys_per_block} BLOCK_Q={config.block_q} "
        f"BLOCK_N={config.block_n} split_n={config.split_n} "
        f"n_tile_phases={config.n_tile_phases}: {ms:.4f} ms, "
        f"{tflops:.2f} TFLOP/s"
    )


def _decode_kernel_from_config(query_heads: int, page_size: int, head_dim: int, config):
    kwargs = dict(
        heads=query_heads,
        head_dim=head_dim,
        page_size=page_size,
        group_size=config.group_size,
        num_stages=config.num_stages,
        use_async=config.use_async,
    )
    if config.use_persistent:
        return _tilelang_qsa_mqa_decode_persistent_kernel(
            persistent_splits=config.persistent_splits, **kwargs
        )
    return _tilelang_qsa_mqa_decode_kernel(
        groups_per_cta=config.groups_per_cta, **kwargs
    )


@GPU_REQUIRED
@pytest.mark.bench
def test_bench_qsa_mqa_decode_cooperative_load(
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Measure the gfx938 default decode get_config across batch sizes."""
    # Skipped: the cooperative_load kernel faults with an invalid address
    # access (XID81 VMFault) and aborts the whole test process. Re-enable
    # once the kernel OOB is fixed.
    pytest.skip(
        "cooperative_load kernel VMFaults (invalid address access); "
        "re-enable after the kernel OOB fix"
    )
    page_size, head_dim = 16, 128
    scale = math.sqrt(head_dim)
    device_info = get_device_info(DEVICE)
    get_config = resolve_get_config(device_info.arch)

    for context_len in (8192, 32768, 65536):
        max_pages = context_len // page_size
        for batch in (16, 32, 64, 128):
            pages = batch * max_pages
            k_cache = torch.randn(
                pages,
                page_size,
                1,
                head_dim,
                device=DEVICE,
                dtype=torch.bfloat16,
            )
            page_table = torch.arange(pages, device=DEVICE, dtype=torch.int32).reshape(
                batch, max_pages
            )
            context_lens = torch.full(
                (batch,), context_len, device=DEVICE, dtype=torch.int32
            )
            logits = torch.empty(batch, context_len, device=DEVICE, dtype=torch.float32)
            for query_heads in (4, 32, 64):
                config = get_config(
                    batch, context_len, device_info.cu_count, num_heads=query_heads
                )
                kernel = _decode_kernel_from_config(
                    query_heads, page_size, head_dim, config
                )
                q = torch.empty(
                    batch,
                    1,
                    query_heads,
                    head_dim,
                    device=DEVICE,
                    dtype=torch.bfloat16,
                )
                q.normal_()

                def fn(run_kernel=kernel, run_q=q):
                    run_kernel(
                        run_q, k_cache, page_table, context_lens, logits, scale
                    )

                fn()
                ms = time_ms(fn, bench_warmup, bench_repeat)
                k_bytes = batch * context_len * head_dim * 2
                k_gbps = k_bytes / (ms * 1e6)
                gemm_tflops = (
                    2 * batch * context_len * query_heads * head_dim / (ms * 1e9)
                )
                schedule = (
                    f"S{config.persistent_splits}"
                    if config.use_persistent
                    else f"G{config.groups_per_cta}"
                )
                print(
                    f"[bench] qsa_mqa_decode B={batch} H={query_heads} "
                    f"context={context_len} {schedule}: {ms:.4f} ms, "
                    f"K={k_gbps:.1f} GB/s, GEMM={gemm_tflops:.2f} TFLOP/s"
                )
