# SPDX-License-Identifier: MIT
"""Independent weighted FP8 WASP tests; reference uses the quantized inputs.

Positive per-Key scales are independent of signed per-Query/head weights.
The reference includes no implicit 1/sqrt(D). Quantization error relative to
the original unquantized tensors is deliberately outside kernel correctness.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mqa_logits import tilelang_mqa_logits
from boltops.mqa_logits.tilelang.kernels.common.mqa_logits import (
    weighted_mqa_prefill as generic_weighted_mqa_prefill,
)
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa import (
    weighted_mqa_prefill as gfx938_weighted_mqa_prefill,
)
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa_prefill_wasp_weighted_fp8 import (
    _tilelang_mqa_prefill_wasp_weighted_fp8_kernel,
)
from boltops.utility.arch_info import get_device_info


def _is_gfx938():
    return torch.cuda.is_available() and get_device_info("cuda").arch.startswith("gfx938")


GFX938_REQUIRED = pytest.mark.skipif(not _is_gfx938(), reason="requires gfx938 GPU")
SLOW = pytest.mark.slow


def _inputs(rows, keys, heads, *, seed=7, device="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(rows, heads, 128, device=device).to(torch.float8_e4m3fn)
    kv = torch.randn(keys, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, heads, device=device) * 0.5
    scales = torch.linspace(0.125, 1.75, keys, device=device)
    # Zero scale must yield zero for valid outputs, yet preserve -inf masks.
    scales[::17] = 0
    rid = torch.arange(rows, device=device, dtype=torch.int32)
    starts = rid * 13 % max(1, keys // 3)
    ends = starts + (keys - starts) * (rid % 5 + 1) // 5
    ends[rid % 11 == 0] = starts[rid % 11 == 0]
    return q, kv, scales, weights, starts, ends


def _assert_reference(out, q, kv, scales, weights, starts, ends):
    assert out.dtype == torch.float32
    columns = torch.arange(kv.shape[0], device=q.device)[None, :]
    for begin in range(0, q.shape[0], 32):
        end = min(begin + 32, q.shape[0])
        # FP32 dot, then ReLU, signed weighting, head reduction, Key scale.
        scores = torch.einsum("mhd,nd->mnh", q[begin:end].float(), kv.float())
        expected = (scores.relu() * weights[begin:end, None, :].float()).sum(-1)
        expected = expected * scales.float()[None, :]
        valid = (columns >= starts[begin:end, None]) & (columns < ends[begin:end, None])
        expected = expected.masked_fill(~valid, -float("inf"))
        torch.testing.assert_close(out[begin:end], expected, atol=2e-2, rtol=2e-2)
        assert torch.isneginf(out[begin:end][~valid]).all()


@GFX938_REQUIRED
@pytest.mark.parametrize("heads", [4, 8, 32, 64])
def test_weighted_fp8_public_matches_torch(heads):
    """All supported heads, partial Query/Key tiles, signed weights and empty rows."""
    args = _inputs(41, 96, heads)
    q, kv, scales, weights, starts, ends = args
    destination = torch.full((41, 96), 1234567.0, device="cuda")
    actual = tilelang_mqa_logits(
        q, kv, scales, weights, starts, ends, D_out=destination,
    )
    assert actual.data_ptr() == destination.data_ptr()
    _assert_reference(actual, *args)


@GFX938_REQUIRED
def test_weighted_fp8_signed_weights_and_noncontiguous_inputs():
    """No post-reduction ReLU, no implicit attention scale, and column scales."""
    rows, keys, heads = 41, 96, 4
    q = torch.ones(rows, heads, 256, device="cuda").to(torch.float8_e4m3fn)[..., ::2]
    kv = torch.ones(keys, 256, device="cuda").to(torch.float8_e4m3fn)[:, ::2]
    weights = torch.full((rows, heads * 2), -0.25, device="cuda")[:, ::2]
    scales = torch.linspace(0, 2, keys * 2, device="cuda")[::2]
    starts = torch.zeros(rows * 2, device="cuda", dtype=torch.int64)[::2]
    ends = torch.full((rows * 2,), keys, device="cuda", dtype=torch.int64)[::2]
    actual = tilelang_mqa_logits(q, kv, scales, weights, starts, ends)
    # dot=128; sum(weights)=-1. This also detects hidden 1/sqrt(128).
    torch.testing.assert_close(actual, (-128 * scales)[None, :].expand(rows, -1))
    assert (actual[:, 1:] < 0).all()


@GFX938_REQUIRED
def test_weighted_fp8_effective_range(monkeypatch):
    """The public hint reaches the FP8 policy; invalid hints fail before launch."""
    from boltops.mqa_logits.tilelang.kernels.gfx938 import mqa

    resolve = mqa.resolve_get_prefill_config_fp8
    hints = []

    def recording_resolver(arch):
        policy = resolve(arch)

        def recording_policy(*args, **kwargs):
            hints.append(kwargs["effective_range"])
            return policy(*args, **kwargs)

        return recording_policy

    monkeypatch.setattr(mqa, "resolve_get_prefill_config_fp8", recording_resolver)
    args = _inputs(41, 96, 4)
    actual = tilelang_mqa_logits(*args, effective_range=96)
    assert hints == [96]
    _assert_reference(actual, *args)
    for hint in (0, 97):
        with pytest.raises(ValueError, match="effective_range"):
            tilelang_mqa_logits(*args, effective_range=hint)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("rows,heads", [(128, 4), (1024, 8), (256, 32), (256, 64)])
def test_weighted_fp8_public_configurations(rows, heads):
    """Actual dispatch: split-N, phase, and logical-M128/256; full reference."""
    args = _inputs(rows, 4096, heads, seed=13)
    actual = tilelang_mqa_logits(*args)
    _assert_reference(actual, *args)


@GFX938_REQUIRED
@SLOW
@pytest.mark.parametrize("heads", [4, 8, 32, 64])
def test_weighted_fp8_overlay_phase_tile_drains(heads):
    """M256 overlays Q/K LDS, all tile drains, phase wrap and a one-row CTA."""
    block_q = 256 // heads
    rows, keys = 6 * block_q + 1, 641
    q, kv, scales, weights, starts, ends = _inputs(rows, keys, heads, seed=19)
    ranges = [(0, 37), (13, 128), (7, 256), (0, 384),
              (11, 529), (0, 0), (13, 529)]
    for block, (start, length) in enumerate(ranges):
        starts[block * block_q:(block + 1) * block_q] = start
        ends[block * block_q:(block + 1) * block_q] = start + length
    sentinel = -1234567.0
    output = torch.full((rows, keys), sentinel, device="cuda")
    kernel = _tilelang_mqa_prefill_wasp_weighted_fp8_kernel(
        heads=heads, block_q=block_q, block_n=128,
        rows_divisible=False, n_tile_phases=4, split_n=1,
    )
    kernel(q.reshape(-1, 128), kv, scales, weights, output, starts, ends, 1.0)
    columns = torch.arange(keys, device="cuda")[None, :]
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    assert torch.all(output[~valid] == sentinel)
    # The direct kernel preserves the sentinel; public mask is a separate op.
    masked = output.masked_fill(~valid, -float("inf"))
    _assert_reference(masked, q, kv, scales, weights, starts, ends)


@GFX938_REQUIRED
@SLOW
def test_weighted_fp8_phase_remainders_and_split_n():
    """Reuse one specialization across phase critical points and partial tiles."""
    heads, block_q, rows, keys = 4, 32, 256, 8192
    q, kv, scales, weights, starts, ends = _inputs(rows, keys, heads, seed=23)
    kernel = _tilelang_mqa_prefill_wasp_weighted_fp8_kernel(
        heads=heads, block_q=block_q, block_n=128,
        rows_divisible=True, n_tile_phases=4, split_n=4,
    )
    for length in (0, 1, 127, 128, 129, 267, 300, 4096, 4097, 4113, 4159, 4223):
        starts.fill_(13)
        ends.fill_(13 + length)
        output = torch.full((rows, keys), -1234567.0, device="cuda")
        kernel(q.reshape(-1, 128), kv, scales, weights, output, starts, ends, 1.0)
        assert (output[:, :13] == -1234567.0).all()
        assert (output[:, 13 + length:] == -1234567.0).all()
        columns = torch.arange(keys, device="cuda")[None, :]
        valid = (columns >= starts[:, None]) & (columns < ends[:, None])
        _assert_reference(output.masked_fill(~valid, -float("inf")),
                          q, kv, scales, weights, starts, ends)


@GFX938_REQUIRED
@pytest.mark.parametrize("rows,keys", [(0, 96), (41, 0), (0, 0)])
def test_weighted_fp8_empty_tensors(rows, keys):
    args = _inputs(rows, keys, 4)
    destination = torch.empty(rows, keys, device="cuda")
    actual = tilelang_mqa_logits(*args, D_out=destination)
    assert actual is destination
    assert actual.shape == (rows, keys)
    assert actual.dtype == torch.float32


def test_weighted_fp8_rejects_invalid_contracts():
    """Validation fails before any launch; fnuz/e5m2 are not reinterpreted as fn."""
    args = list(_inputs(3, 7, 4, device="cpu"))
    bad_cases = [
        (0, args[0].float(), ValueError, "float8_e4m3fn"),
        (1, args[1].to(torch.float8_e5m2), ValueError, "float8_e4m3fn"),
        (1, args[1].to(torch.float8_e4m3fnuz), ValueError, "float8_e4m3fn"),
        (2, torch.ones(7, dtype=torch.float16), ValueError, "kv_scales"),
        (2, torch.ones(7, 1), ValueError, "kv_scales"),
        (2, torch.ones(6), ValueError, "kv_scales"),
        (3, torch.ones(3, 2), ValueError, "weights"),
        (4, torch.zeros(2, dtype=torch.int32), ValueError, "cu_starts"),
        (5, torch.ones(3, 1, dtype=torch.int32), ValueError, "cu_starts"),
        (0, torch.ones(3, 16, 128).to(torch.float8_e4m3fn), NotImplementedError, "H in"),
        (1, torch.ones(7, 64).to(torch.float8_e4m3fn), ValueError, "matching D"),
    ]
    for index, value, error, match in bad_cases:
        bad = args.copy()
        bad[index] = value
        with pytest.raises(error, match=match):
            gfx938_weighted_mqa_prefill(*bad)
    with pytest.raises(ValueError, match="same GPU"):
        gfx938_weighted_mqa_prefill(*args)
    with pytest.raises(NotImplementedError, match="gfx938 WASP"):
        generic_weighted_mqa_prefill(*args)


@GFX938_REQUIRED
def test_weighted_fp8_rejects_invalid_output():
    args = _inputs(41, 96, 4)
    for destination in (
        torch.empty(41, 96, device="cuda", dtype=torch.float16),
        torch.empty(41, 95, device="cuda"),
        torch.empty(41, 96),
        torch.empty(96, 41, device="cuda").T,
    ):
        with pytest.raises(ValueError, match="D_out"):
            tilelang_mqa_logits(*args, D_out=destination)
