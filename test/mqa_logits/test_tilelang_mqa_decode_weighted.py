# SPDX-License-Identifier: MIT
"""Weighted paged decode: QSA coverage parity and independent FP32 reference."""

import math

import pytest
import torch

pytest.importorskip("tilelang")

from boltops.mqa_logits import tilelang_mqa_decode
from boltops.mqa_logits.tilelang._dispatch import _resolve_for_arch
from boltops.mqa_logits.tilelang.kernels.gfx938.mqa_decode_weighted import (
    _tilelang_weighted_mqa_decode_kernel,
    _tilelang_weighted_mqa_decode_persistent_kernel,
    _tilelang_weighted_mqa_decode_fp8_kernel,
    _tilelang_weighted_mqa_decode_fp8_persistent_kernel,
    weighted_mqa_decode,
)
from boltops.utility.arch_info import get_device_info


GFX938_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or not get_device_info("cuda").arch.startswith("gfx938"),
    reason="requires gfx938 GPU",
)


def _inputs(lengths, capacity, *, seed=41, device="cuda", heads=4,
            page_size=64, fp8=False):
    torch.manual_seed(seed)
    batch, max_pages = len(lengths), (capacity + page_size - 1) // page_size
    pages = batch * max_pages + 7
    q = torch.randn(batch, heads, 128, device=device, dtype=torch.bfloat16)
    cache = torch.randn(pages, page_size, 1, 128, device=device, dtype=torch.bfloat16)
    weights = torch.randn(batch, heads, device=device, dtype=torch.float32) * 0.5
    if fp8:
        from boltops.mqa_logits.tilelang.kernels.common.mqa import pack_fp8_paged_k_cache
        q = q.to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page_size, device=device) * 1.75 + .125
        scales[:, ::17] = 0
        cache = pack_fp8_paged_k_cache(cache.to(torch.float8_e4m3fn), scales)
    table = torch.full((batch, max_pages), -1, device=device, dtype=torch.int32)
    shuffled = torch.randperm(pages, device=device)
    for b, length in enumerate(lengths):
        used = (length + page_size - 1) // page_size
        table[b, :used] = shuffled[b * max_pages:b * max_pages + used].to(torch.int32)
    return q, cache, weights, table, torch.tensor(lengths, device=device, dtype=torch.int32)


def _reference(q, cache, weights, table, lengths, capacity, score_scale=None):
    score_scale = math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    page_size = cache.shape[1]
    scales = None
    if q.dtype == torch.float8_e4m3fn:
        # Decode the byte contract independently of the production unpacker.
        raw = cache.contiguous().view(torch.uint8).reshape(cache.shape[0], -1)
        scales = raw[:, page_size * 128:].view(torch.float32).reshape(-1, page_size)
        cache = raw[:, :page_size * 128].view(torch.float8_e4m3fn).reshape(-1, page_size, 1, 128)
    expected = torch.full((q.shape[0], capacity), -float("inf"), device=q.device)
    for b, length in enumerate(lengths.tolist()):
        if length:
            positions = torch.arange(length, device=q.device)
            physical_pages = table[b, positions // page_size].long()
            keys = cache[physical_pages, positions % page_size, 0, :].float()
            dots = torch.einsum("hd,nd->nh", q[b].float(), keys)
            expected[b, :length] = (dots.relu() * weights[b]).sum(-1)
            if scales is not None:
                expected[b, :length] *= scales[physical_pages, positions % page_size]
            expected[b, :length] /= score_scale
    return expected


def _check(actual, args, capacity, score_scale=None):
    expected = _reference(*args, capacity, score_scale)
    assert actual.dtype == torch.float32
    tolerance = 2e-2 if args[0].dtype == torch.float8_e4m3fn else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def test_weighted_decode_dispatch_is_independent():
    assert callable(tilelang_mqa_decode)
    assert _resolve_for_arch("weighted_mqa_decode", "gfx938") is weighted_mqa_decode
    assert _resolve_for_arch("qsa_mqa_decode", "gfx938").__module__.endswith("gfx938.mqa")
    with pytest.raises(RuntimeError, match="does not support"):
        _resolve_for_arch("weighted_mqa_decode", "gfx936")


@GFX938_REQUIRED
def test_weighted_decode_typical_matches_torch():
    """B16/H4/D128/page64/K8192: full output, loops, ragged and empty contexts."""
    args = _inputs([8192, 8191, 4096, 4097, 150, 90, 129, 128,
                    127, 65, 64, 63, 1, 0, 1024, 3073], 8192)
    destination = torch.full((16, 8192), 1234567.0, device="cuda")
    actual = tilelang_mqa_decode(*args, 8192, D_out=destination)
    assert actual is destination
    _check(actual, args, 8192)


@GFX938_REQUIRED
@pytest.mark.parametrize("group_size,groups_per_cta", [(64, 1), (64, 8), (128, 8)])
def test_weighted_decode_direct_group_and_tail_boundaries(group_size, groups_per_cta):
    """Partial final physical page/table, no writes to invalid sentinel columns."""
    capacity = 1101  # 18 physical page slots; group128 crosses page boundaries.
    args = _inputs([0, 1, 63, 64, 65, 127, 128, 129, 150, 1024, 1025, 1101], capacity)
    output = torch.full((12, capacity), -1234567.0, device="cuda")
    _tilelang_weighted_mqa_decode_kernel(
        group_size=group_size, groups_per_cta=groups_per_cta,
    )(*args, output, math.sqrt(128))
    valid = torch.arange(capacity, device="cuda")[None, :] < args[-1][:, None]
    assert (output[~valid] == -1234567.0).all()
    _check(output.masked_fill(~valid, -float("inf")), args, capacity)


@GFX938_REQUIRED
def test_weighted_decode_signed_weights_and_noncontiguous_inputs():
    """Detect weighting before ReLU, hidden attention scale and stale suffixes."""
    q = torch.ones(2, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    q[:, 1] = -1
    cache = torch.ones(8, 64, 1, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    weights = torch.tensor([[-2., 0., 100., 0., .5, 0., 0., 0.]], device="cuda").expand(2, -1)[:, ::2]
    table = torch.tensor([[7, -1, 2, -1, 5, -1], [4, -1, 6, -1, -1, -1]], device="cuda", dtype=torch.int64)[:, ::2]
    lengths = torch.tensor([150, -1, 90, -1], device="cuda", dtype=torch.int64)[::2]
    args = q, cache, weights, table, lengths
    out = tilelang_mqa_decode(*args, 192, score_scale=1.0)
    _check(out, args, 192, score_scale=1.0)
    assert (out[0, :150] == -192).all()


@GFX938_REQUIRED
@pytest.mark.parametrize("batch,capacity", [(0, 192), (2, 0)])
def test_weighted_decode_empty_outputs(batch, capacity):
    args = _inputs([0] * batch, capacity)
    out = tilelang_mqa_decode(*args, capacity)
    assert out.shape == (batch, capacity)
    assert out.dtype == torch.float32


def test_weighted_decode_rejects_invalid_contracts():
    args = list(_inputs([3, 2], 64, device="cpu"))
    for index, value, match in [
        (0, torch.ones(2, 3, 128, dtype=torch.bfloat16), "Q"),
        (0, args[0].float(), "BF16"),
        (1, torch.ones(9, 12, 1, 128, dtype=torch.bfloat16), "k_cache"),
        (2, torch.ones(2, 4, dtype=torch.float16), "weights"),
        (2, torch.ones(2, 3), "weights"),
        (3, torch.ones(1, 1, dtype=torch.int32), "page_table"),
        (3, args[3].float(), "int32/int64"),
        (4, torch.ones(2, 1, dtype=torch.int32), "context_lens"),
        (4, args[4].float(), "int32/int64"),
    ]:
        bad = args.copy()
        bad[index] = value
        with pytest.raises(ValueError, match=match):
            weighted_mqa_decode(*bad, 64)
    with pytest.raises(ValueError, match="max_model_len"):
        weighted_mqa_decode(*args, -1)
    with pytest.raises(ValueError, match="same GPU"):
        weighted_mqa_decode(*args, 64)


@GFX938_REQUIRED
def test_weighted_decode_rejects_invalid_output():
    args = _inputs([150, 90], 192)
    for dest in (torch.empty(2, 192), torch.empty(2, 192, device="cuda", dtype=torch.float16),
                 torch.empty(2, 191, device="cuda"), torch.empty(192, 2, device="cuda").T):
        with pytest.raises(ValueError, match="D_out"):
            tilelang_mqa_decode(*args, 192, D_out=dest)


HEAD_PAGES = [
    pytest.param(page, heads, marks=() if (page, heads) == (16, 4) else pytest.mark.slow)
    for page in (8, 16, 32, 64) for heads in (4, 8, 16, 32, 64)
]


@GFX938_REQUIRED
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("page_size,heads", HEAD_PAGES)
def test_weighted_decode_public_heads_and_pages(page_size, heads, fp8):
    """Superset of QSA public page/head tests, including packed FP8 H8."""
    capacity = page_size * 16
    args = _inputs([17, 63, 95], capacity, heads=heads, page_size=page_size, fp8=fp8)
    out = tilelang_mqa_decode(*args, capacity)
    _check(out, args, capacity)
    # All-one head weights recover QSA with the same default sqrt(D) scale.
    from boltops.mqa_logits import tilelang_qsa_mqa_decode
    unit = (*args[:2], torch.ones_like(args[2]), *args[3:])
    actual = tilelang_mqa_decode(*unit, capacity)
    qsa = tilelang_qsa_mqa_decode(args[0], args[1], args[3], args[4], capacity)
    tol = 2e-2 if fp8 else 2e-3
    torch.testing.assert_close(actual, qsa, atol=tol, rtol=tol)


def _direct(args, output, *, group_size=128, groups_per_cta=4,
            persistent_splits=0, num_stages=1, use_async=True, scale_async=False,
            score_scale=None):
    q, cache, weights, table, lens = args
    fp8 = q.dtype == torch.float8_e4m3fn
    kwargs = dict(heads=q.shape[1], head_dim=128, page_size=cache.shape[1],
                  group_size=group_size, num_stages=num_stages, use_async=use_async)
    if persistent_splits:
        factory = (_tilelang_weighted_mqa_decode_fp8_persistent_kernel if fp8
                   else _tilelang_weighted_mqa_decode_persistent_kernel)
        kwargs["persistent_splits"] = persistent_splits
    else:
        factory = _tilelang_weighted_mqa_decode_fp8_kernel if fp8 else _tilelang_weighted_mqa_decode_kernel
        kwargs["groups_per_cta"] = groups_per_cta
    if fp8:
        kwargs["scale_async"] = scale_async
        shape = ((cache.shape[0], 1, cache.shape[1] * 132) if q.shape[1] > 16
                 else (cache.shape[0], cache.shape[1] * 132))
        cache = cache.view(torch.float8_e4m3fn).reshape(shape)
    kernel = factory(**kwargs)
    kernel(q, cache, weights, table, lens, output,
           math.sqrt(128) if score_scale is None else score_scale)
    return kernel


def _check_direct(args, capacity, **kwargs):
    sentinel = -1234567.0
    output = torch.full((args[0].shape[0], capacity), sentinel, device="cuda")
    kernel = _direct(args, output, **kwargs)
    valid = torch.arange(capacity, device="cuda")[None, :] < args[-1][:, None]
    assert (output[~valid] == sentinel).all()
    _check(output.masked_fill(~valid, -float("inf")), args, capacity,
           score_scale=kwargs.get("score_scale"))
    return kernel


@GFX938_REQUIRED
@pytest.mark.slow
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("persistent_splits", [2, 4, 8, 16, 32])
def test_weighted_decode_group128_persistent_matches_torch(persistent_splits, heads, fp8):
    """QSA's H16/32/64 x S2/4/8/16/32 grid plus H4/H8 and FP8.

    The short vector mirrors QSA [17,256,447]. Longer vectors additionally
    force multiple loop iterations per split, including S32, and exact tails.
    """
    for capacity, lengths in [
        (512, [17, 256, 447]),
        (128 * (persistent_splits * 2 + 1) + 1,
         [0, 1, 127, 128, 129, 128 * (persistent_splits * 2 + 1) + 1]),
    ]:
        args = _inputs(lengths, capacity, heads=heads, page_size=16, fp8=fp8)
        _check_direct(args, capacity, persistent_splits=persistent_splits)


@GFX938_REQUIRED
@pytest.mark.slow
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
def test_weighted_decode_group128_ordinary_matches_torch(heads, fp8):
    """QSA group128/GPC4 plus full loops, odd page capacity and CTA boundaries."""
    args = _inputs([17, 127, 191], 256, heads=heads, page_size=16, fp8=fp8)
    _check_direct(args, 256)
    capacity = 1101
    args = _inputs([0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1024, 1101],
                   capacity, heads=heads, page_size=16, fp8=fp8)
    _check_direct(args, capacity)


@GFX938_REQUIRED
@pytest.mark.slow
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
def test_weighted_decode_sync_async_and_groups(fp8, use_async, group_size):
    """Includes the documented BF16 group64 small-page synchronous fallback."""
    args = _inputs([0, 17, 63, 64, 65, 127, 128, 129, 255, 447, 1101],
                   1101, heads=8, page_size=8, fp8=fp8)
    _check_direct(args, 1101, group_size=group_size, num_stages=int(use_async),
                  use_async=use_async)


@GFX938_REQUIRED
@pytest.mark.slow
@pytest.mark.parametrize("heads,persistent_splits", [(4, 0), (32, 4), (4, 16), (32, 32)])
def test_weighted_decode_fp8_scale_async(heads, persistent_splits):
    args = _inputs([0, 17, 128, 129, 1101], 1101, heads=heads, page_size=32, fp8=True)
    _check_direct(args, 1101, persistent_splits=persistent_splits, scale_async=True)


@GFX938_REQUIRED
@pytest.mark.slow
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("batch,capacity,heads", [
    (1, 8192, 4), (8, 8192, 8), (16, 8192, 4), (16, 8192, 64),
    (32, 16384, 32), (64, 65536, 16), (1, 131072, 64),
])
def test_weighted_decode_public_schedule_buckets(batch, capacity, heads, fp8):
    """Exercise real selector branches, long suffixes and full long contexts."""
    lengths = ([capacity, capacity - 1, 4097, 129, 1, 0] * ((batch + 5) // 6))[:batch]
    args = _inputs(lengths, capacity, heads=heads, fp8=fp8)
    dest = torch.full((batch, capacity), 1234567., device="cuda")
    out = tilelang_mqa_decode(*args, capacity, D_out=dest)
    assert out is dest
    _check(out, args, capacity)


@GFX938_REQUIRED
def test_weighted_decode_fp8_signed_weights_scale_and_storage():
    """Scale belongs outside the head sum; packed storage may be byte or FP8."""
    from boltops.mqa_logits.tilelang.kernels.common.mqa import pack_fp8_paged_k_cache
    q = torch.ones(2, 8, 128, device="cuda").to(torch.float8_e4m3fn)
    q[:, 1] = -1
    keys = torch.ones(6, 16, 1, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.arange(96, device="cuda", dtype=torch.float32).reshape(6, 16) / 32
    scales[:, ::3] = 0
    scales[:, 1::5] *= -1  # Detect moving the per-Key scale inside ReLU.
    cache = pack_fp8_paged_k_cache(keys, scales)
    weights = torch.tensor([[-2., 100., .5, 0, 0, 0, 0, 0]], device="cuda").expand(2, -1)
    table = torch.tensor([[5, 1, 3], [4, -1, -1]], device="cuda", dtype=torch.int64)
    lens = torch.tensor([47, 1], device="cuda", dtype=torch.int64)
    for storage in (cache, cache.view(torch.float8_e4m3fn)):
        args = q, storage, weights, table, lens
        _check(tilelang_mqa_decode(*args, 49), args, 49)


@GFX938_REQUIRED
def test_weighted_decode_fp8_noncontiguous_inputs():
    args = list(_inputs([0, 17, 129], 192, heads=8, page_size=16, fp8=True))
    for i in range(len(args)):
        source = args[i]
        shape = list(source.shape)
        shape[-1] *= 2
        storage = torch.empty(shape, dtype=source.dtype, device=source.device)
        storage[..., ::2] = source
        args[i] = storage[..., ::2]
    # Also exercise both index conversions and strides in the same call.
    args[3] = args[3].long()
    args[4] = args[4].long()
    out = tilelang_mqa_decode(*args, 192)
    _check(out, args, 192)


@GFX938_REQUIRED
def test_weighted_decode_fp8_rejects_unpacked_and_wrong_storage():
    q, cache, weights, table, lens = _inputs([17], 128, fp8=True)
    for bad in (torch.empty(9, 64, 1, 128, device="cuda", dtype=torch.float8_e4m3fn),
                cache.float()):
        with pytest.raises(ValueError, match="k_cache"):
            tilelang_mqa_decode(q, bad, weights, table, lens, 128)


@GFX938_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
def test_bench_weighted_mqa_decode(heads, fp8, time_ms, bench_warmup, bench_repeat):
    """Compute-only benchmark with the weighted default policy, B16/K8192."""
    from boltops.mqa_logits.tilelang.configs.mqa_decode import weighted as policy
    args = _inputs([8192] * 16, 8192, heads=heads, fp8=fp8)
    cu_count = get_device_info("cuda").cu_count
    config = (policy.get_config_fp8 if fp8 else policy.get_config)(16, 8192, cu_count, heads)
    kwargs = dict(group_size=config.group_size, num_stages=config.num_stages,
                  use_async=config.use_async, persistent_splits=config.persistent_splits,
                  groups_per_cta=config.groups_per_cta)
    output = torch.empty(16, 8192, device="cuda")
    kernel = _direct(args, output, **kwargs)
    _check(output, args, 8192)
    q, cache, weights, table, lens = args
    if fp8:
        shape = (cache.shape[0], 1, -1) if heads > 16 else (cache.shape[0], -1)
        cache = cache.view(torch.float8_e4m3fn).reshape(shape)
    elapsed = time_ms(lambda: kernel(q, cache, weights, table, lens, output, math.sqrt(128)),
                      warmup=bench_warmup, repeat=bench_repeat)
    print(f"[bench] weighted_mqa_decode H={heads} fp8={fp8} {config}: {elapsed * 1000:.3f} us")


@GFX938_REQUIRED
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("persistent_splits", [0, 16])
def test_weighted_decode_runtime_score_scale(heads, fp8, persistent_splits):
    """Reuse one compiled kernel with changing runtime scale and signed weights."""
    capacity = 1101
    args = _inputs([0, 1, 127, 128, 129, 1101], capacity, heads=heads, fp8=fp8)
    original_weights = args[2].clone()
    kernels = []
    for scale in (1.0, 2.5, math.sqrt(128)):
        kernels.append(_check_direct(args, capacity, persistent_splits=persistent_splits,
                                     score_scale=scale))
        out = tilelang_mqa_decode(*args, capacity, score_scale=scale)
        _check(out, args, capacity, score_scale=scale)
    assert all(kernel is kernels[0] for kernel in kernels)
    torch.testing.assert_close(args[2], original_weights, atol=0, rtol=0)
    default = tilelang_mqa_decode(*args, capacity)
    torch.testing.assert_close(default, out, atol=0, rtol=0)
    from boltops.mqa_logits import tilelang_qsa_mqa_decode
    unit = (*args[:2], torch.ones_like(args[2]), *args[3:])
    actual = tilelang_mqa_decode(*unit, capacity, score_scale=2.5)
    qsa = tilelang_qsa_mqa_decode(args[0], args[1], args[3], args[4], capacity, score_scale=2.5)
    tol = 2e-2 if fp8 else 2e-3
    torch.testing.assert_close(actual, qsa, atol=tol, rtol=tol)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_weighted_decode_rejects_invalid_score_scale(scale):
    args = _inputs([1], 64, device="cpu")
    with pytest.raises(ValueError, match="score_scale"):
        weighted_mqa_decode(*args, 64, score_scale=scale)
