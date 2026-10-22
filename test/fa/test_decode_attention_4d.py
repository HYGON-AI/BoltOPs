"""Exports, FP8-cache correctness and benchmarks for four-dimensional Decode."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from boltops.fa import decode_attention_fwd_4d

from ._fa_bench_utils import case_parameters, paged_fixture, prepared_case, useful_flops

GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
PERF_BACKEND = "triton"
PERF_OP = "fa_decode_4d"


def test_decode_public_exports_and_signature():
    from boltops.fa.decode_attention import decode_attention_fwd_4d as canonical
    from boltops.fa.decode_attention.triton.decode_attention import decode_attention_fwd_4d as backend
    assert canonical is decode_attention_fwd_4d
    assert inspect.signature(canonical) == inspect.signature(backend)


def _decode_reference(q, k, v, sm_scale, ks, vs):
    scores = (q.float() @ (k.float() * ks).transpose(-1, -2)) * sm_scale
    return torch.softmax(scores, -1) @ (v.float() * vs)


@GPU_REQUIRED
@pytest.mark.parametrize("length,splits", [(1, 1), (7, 16), (65, 3), (257, 8)])
@pytest.mark.parametrize("ks,vs", [(1.0, 1.0), (0.37, 1.7)])
def test_decode_logical_matches_reference(length, splits, ks, vs):
    torch.manual_seed(length)
    q = torch.randn(2, 3, 1, 128, device="cuda", dtype=torch.bfloat16)
    # Build on CPU to avoid relying on a separate GPU quantization kernel.
    k = torch.randn(2, 3, length, 128).to(torch.float8_e4m3fn).cuda()
    v = torch.randn(2, 3, length, 128).to(torch.float8_e4m3fn).cuda()
    expected = _decode_reference(q, k, v, 0.125, ks, vs)
    out = decode_attention_fwd_4d(q, k, v, sm_scale=0.125, k_scale=ks, v_scale=vs, num_kv_splits=splits)
    assert out.shape == q.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


@GPU_REQUIRED
@pytest.mark.parametrize("strided", [False, True])
def test_decode_logical_dynamic_lengths_reuse_kernel_within_bucket(monkeypatch, strided):
    from boltops.mla.triton.decode_attention import _fwd_kernel_stage1

    torch.manual_seed(1201)
    original_run = _fwd_kernel_stage1.run
    kernels, page_sizes = [], []

    def record_kernel(*args, **kwargs):
        kernel = original_run(*args, **kwargs)
        kernels.append(kernel)
        page_sizes.append(kwargs["PAGE_SIZE"])
        return kernel

    monkeypatch.setattr(_fwd_kernel_stage1, "run", record_kernel)
    q = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    for length in (129, 191, 256, 257):
        width = 256 if strided else 128
        k = torch.randn(1, 2, length, width).to(torch.float8_e4m3fn).cuda()[..., :128]
        v = torch.randn(1, 2, length, width).to(torch.float8_e4m3fn).cuda()[..., :128]
        expected = _decode_reference(q, k, v, 0.125, 0.37, 1.7)
        out = decode_attention_fwd_4d(
            q, k, v, sm_scale=0.125, k_scale=0.37, v_scale=1.7, num_kv_splits=3,
        )
        torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)

    assert page_sizes == [256, 256, 256, 512]
    assert kernels[0] is kernels[1] is kernels[2]
    assert kernels[3] is not kernels[2]


@GPU_REQUIRED
@pytest.mark.parametrize("page_size", [1, 16, 32])
def test_decode_paged_reordered_shared_pages_ragged_lengths(page_size):
    torch.manual_seed(page_size)
    batch, heads, pages = 3, 2, 8
    q = torch.randn(batch, heads, 1, 128, device="cuda", dtype=torch.bfloat16)
    # Retain padding between heads/pages to exercise physical strides.
    storage = torch.randn(pages, page_size, heads, 256).to(torch.float8_e4m3fn).cuda()
    k, v = storage[..., :128], storage[..., 128:]
    table = torch.tensor([[5, 1, -1], [5, 3, 0], [7, -1, -1]], device="cuda", dtype=torch.int64)
    lengths = torch.tensor([page_size + 1, page_size * 3 - 1, 1], device="cuda", dtype=torch.int32)
    ks = torch.tensor(0.37, device="cuda", dtype=torch.float32)
    vs = torch.tensor(1.7, device="cuda", dtype=torch.float32)
    expected = []
    for b, length in enumerate(lengths.tolist()):
        ids = table[b, : (length + page_size - 1) // page_size].long()
        # Index byte views because some PyTorch builds lack FP8 advanced indexing.
        keys = k.view(torch.uint8)[ids].view(torch.float8_e4m3fn).flatten(0, 1)[:length].transpose(0, 1)
        vals = v.view(torch.uint8)[ids].view(torch.float8_e4m3fn).flatten(0, 1)[:length].transpose(0, 1)
        expected.append(_decode_reference(q[b], keys, vals, 0.125, ks, vs))
    out = decode_attention_fwd_4d(
        q, k, v, block_table=table, seq_lens=lengths,
        sm_scale=0.125, k_scale=ks, v_scale=vs, num_kv_splits=4,
    )
    torch.testing.assert_close(out.float(), torch.stack(expected), atol=0.02, rtol=0.02)


@GPU_REQUIRED
@pytest.mark.parametrize("invalid", ["dtype", "scale", "query_length", "bad_page", "zero_length", "missing_lengths"])
def test_decode_rejects_unsupported_or_unsafe_inputs(invalid):
    q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 1, 128).to(torch.float8_e4m3fn).cuda()
    kwargs = dict(block_table=torch.zeros(1, 1, device="cuda", dtype=torch.int32),
                  seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
                  sm_scale=0.125, k_scale=0.37, v_scale=1.7)
    if invalid == "dtype":
        k = k.to(torch.bfloat16)
    elif invalid == "scale":
        kwargs["k_scale"] = 0.0
    elif invalid == "query_length":
        q = q.expand(1, 1, 2, 128)
    elif invalid == "bad_page":
        kwargs["block_table"].fill_(1)
    elif invalid == "zero_length":
        kwargs["seq_lens"].zero_()
    elif invalid == "missing_lengths":
        kwargs["seq_lens"] = None
    with pytest.raises((ValueError, TypeError)):
        decode_attention_fwd_4d(q, k, k, **kwargs)


@GPU_REQUIRED
@pytest.mark.parametrize("field", ["sm_scale", "k_scale", "v_scale"])
@pytest.mark.parametrize("value", [None, True, 0.0, -1.0, float("inf"), float("nan")])
def test_decode_rejects_invalid_scalar_scales(field, value):
    q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    kv = q.to(torch.float8_e4m3fn)
    kwargs = dict(sm_scale=0.125, k_scale=0.37, v_scale=1.7, num_kv_splits=1)
    kwargs[field] = value
    with pytest.raises((ValueError, TypeError), match=field):
        decode_attention_fwd_4d(q, kv, kv, **kwargs)


@GPU_REQUIRED
@pytest.mark.parametrize("missing", ["sm_scale", "k_scale", "v_scale"])
def test_decode_requires_each_scale(missing):
    q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    kv = q.to(torch.float8_e4m3fn)
    kwargs = dict(sm_scale=0.125, k_scale=0.37, v_scale=1.7)
    del kwargs[missing]
    with pytest.raises(TypeError, match=missing):
        decode_attention_fwd_4d(q, kv, kv, **kwargs)


@GPU_REQUIRED
def test_decode_rejects_nonunit_feature_stride():
    q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    storage = torch.zeros(1, 1, 7, 256, device="cuda").to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="unit last-dimension stride"):
        decode_attention_fwd_4d(q, storage[..., ::2], storage[..., ::2],
                                sm_scale=0.125, k_scale=0.37, v_scale=1.7)


@GPU_REQUIRED
@pytest.mark.parametrize("page", [1, 16, 64])
def test_fp8_page_bytes_follow_independent_cpu_oracle(page):
    # Distinct batch/head/token bytes, nontrivial whole-page permutation.
    shape = (2, 3, 2 * page, 128)
    logical = (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 251).byte()
    logical = logical.view(torch.float8_e4m3fn).cuda()
    perm = torch.tensor([2, 0, 3, 1], device="cuda")
    physical = paged_fixture(logical, page, perm)
    truth = logical.view(torch.uint8).cpu()
    for b in range(2):
        for token in range(2 * page):
            page_id = int(perm.cpu()[b * 2 + token // page])
            got = physical[page_id, token % page].view(torch.uint8).cpu()
            assert torch.equal(got, truth[b, :, token])


@GPU_REQUIRED
@pytest.mark.parametrize("prepared_case", case_parameters({"decode"}, full=True),
                         indirect=True, scope="module", ids=lambda c: c["case_id"])
def test_original_shape_decode(prepared_case):
    case, _ = prepared_case  # Input preparation executes and checks the reference.
    print(f"[correctness] {case['case_id']}: passed")


@pytest.mark.bench
@pytest.mark.parametrize("prepared_case", case_parameters({"decode"}),
                         indirect=True, scope="module", ids=lambda c: c["case_id"])
def test_bench_fa_decode(prepared_case, time_ms, bench_warmup, bench_repeat):
    case, run = prepared_case
    ms = time_ms(run, bench_warmup, bench_repeat,
                 flops=useful_flops(case), dtype=torch.bfloat16)
    assert math.isfinite(ms) and ms > 0
    print(f"[bench] {case['case_id']}: {ms:.6f} ms")
