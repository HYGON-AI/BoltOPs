# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared inputs, FP32 references and capacity checks for FA operator tests."""

from __future__ import annotations

import contextlib
import gc
import math
import os

import pytest
import torch
from ._fa_bench_cases import CASES


def case_parameters(paths, *, full=False):
    wanted = set(filter(None, os.environ.get("FA_BENCH_CASES", "").split(",")))
    full = full or os.environ.get("FA_BENCH_SET", "representative") == "full"
    # CP8 cases are B=32 (73-164 GiB of Q/K/V): beyond CI runners, so they
    # need --runslow for both correctness and benchmark collection.
    return [
        pytest.param(c, marks=pytest.mark.slow) if c["path"] == "segmented" else c
        for c in CASES
        if c["path"] in paths
        and (not wanted or c["case_id"] in wanted)
        and (full or c["representative"] or c["case_id"] in wanted)
    ]


def _sample_rows(length, segmented=False):
    if segmented:
        fixed = {i for i in (0, 1, 63, 64, 127, 128, length - 2, length - 1) if 0 <= i < length}
        gen = torch.Generator().manual_seed(length)
        for i in torch.randperm(length, generator=gen).tolist():
            if len(fixed) >= min(16, length):
                break
            fixed.add(i)
        return torch.tensor(sorted(fixed), device="cuda")
    if length <= 256:
        return torch.arange(length, device="cuda")
    gen = torch.Generator().manual_seed(length)
    middle = torch.randperm(length - 128, generator=gen)[:128] + 64
    return torch.cat((torch.arange(64), middle, torch.arange(length - 64, length))).cuda()


def metrics(actual, expected):
    got, ref = actual.double().flatten(), expected.double().flatten()
    err2, ref2, got2 = (
        (got - ref).square().sum().item(),
        ref.square().sum().item(),
        got.square().sum().item(),
    )
    dot = (got * ref).sum().item()
    return {"err2": err2, "ref2": ref2, "got2": got2, "dot": dot,
            "max_abs_err": (got - ref).abs().max().item()}


def combine_metrics(items, finite):
    sums = {k: sum(r[k] for r in items) for k in ("err2", "ref2", "got2", "dot")}
    rel = math.sqrt(sums["err2"] / max(sums["ref2"], 1e-300))
    cos = sums["dot"] / max(math.sqrt(sums["got2"] * sums["ref2"]), 1e-300)
    return {"finite": finite, "max_abs_err": max(r["max_abs_err"] for r in items),
            "rel_l2": rel, "cosine": cos, "max_rel_l2": 0.01,
            "min_cosine": 0.9999, "pass": finite and rel <= 0.01 and cos >= 0.9999}


def prefill_reference(q, k, v, actual, scale, segments=None):
    """Independent global-token mask; full keys for each sampled query."""
    batch, heads, sq = q.shape[:3]
    segmented = segments is not None
    segments = segments or [(0, sq, k.shape[2] - sq)]
    items, count = [], 0
    batches = list(range(batch)) if segmented else sorted({0, batch - 1})
    for b in batches:
        for h in sorted({0, heads // 2, heads - 1}):
            for begin, length, global_start in segments:
                rows = _sample_rows(length, segmented)
                scores = (q[b, h, begin + rows].float() @ k[b, h].float().T) * scale
                visible = torch.arange(k.shape[2], device=q.device)[None, :] <= (
                    global_start + rows[:, None]
                )
                scores.masked_fill_(~visible, -torch.inf)
                expected = torch.softmax(scores, -1) @ v[b, h].float()
                items.append(metrics(actual[b, h, begin + rows].float(), expected))
                count += rows.numel()
    result = combine_metrics(items, bool(torch.isfinite(actual).all()))
    result.update(reference="FP32 sampled queries, full KV, explicit global mask",
                  sample_query_rows=count, reference_elements=count * v.shape[-1],
                  sample_batch_indices=batches)
    return result


def decode_reference(q, k, v, actual, scale, ks, vs):
    items = []
    for b in range(q.shape[0]):
        for h in range(0, q.shape[1], 4):
            keys = k[b, h:h + 4].float() * ks
            vals = v[b, h:h + 4].float() * vs
            scores = (q[b, h:h + 4].float() @ keys.transpose(-1, -2)) * scale
            expected = torch.softmax(scores, -1) @ vals
            items.append(metrics(actual[b, h:h + 4].float(), expected))
    result = combine_metrics(items, bool(torch.isfinite(actual).all()))
    result.update(reference="FP32 full output, logical FP8 values times scalar scales",
                  reference_elements=actual.numel(), metadata_origin="synthetic fixture")
    return result


def make_strided(t):
    b, h, s, d = t.shape
    storage = torch.empty((b, 2 * s, h, d), dtype=t.dtype, device=t.device)
    view = storage[:, ::2].transpose(1, 2)
    view.copy_(t)
    return view


def paged_fixture(logical, page_size, permutation):
    """Byte-copy FP8 pages; verify exact inverse mapping before timing."""
    b, h, length, d = logical.shape
    assert length % page_size == 0
    source = logical.transpose(1, 2).contiguous().view(-1, page_size, h, d)
    physical = torch.empty_like(source)
    physical.view(torch.uint8).index_copy_(0, permutation, source.view(torch.uint8))
    restored = physical.view(torch.uint8).index_select(0, permutation)
    assert torch.equal(restored, source.view(torch.uint8)), "page mapping changed FP8 bytes"
    return physical


def cp8_geometry(total, prefix, rank):
    """Return two zigzag intervals for one CP8 rank, including uneven splits."""
    if total < 0 or prefix < 0 or not 0 <= rank < 8:
        raise ValueError("invalid token count, prefix or CP8 rank")
    sizes = [total // 16 + int(i < total % 16) for i in range(16)]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    intervals = [(prefix + offsets[i], prefix + offsets[i + 1]) for i in (rank, 15 - rank)]
    return {
        "block_lengths": sizes,
        "intervals": intervals,
        "q_segment_lens": [sizes[i] for i in (rank, 15 - rank)],
        "kv_prefix_lens": [end for start, end in intervals],
    }


def cp_geometry(case):
    index = int(case["source_op"].removeprefix("mha_slice"))
    history = (0, 53248, 106496)[index]
    meta = cp8_geometry(case["Sk"] - history, history, case["rank"])
    starts = tuple(start for start, end in meta["intervals"])
    return meta["q_segment_lens"][0], starts, tuple(meta["kv_prefix_lens"])


def _build_prefill(case):
    from boltops.fa import dense_prefill_attention_fwd_4d, dense_prefill_attention_fwd_segmented
    b, h, sq, sk, dq, dv = (case[k] for k in ("B", "H", "Sq", "Sk", "Dqk", "Dv"))
    q = torch.randn(b, h, sq, dq, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, sk, dq, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, sk, dv, device="cuda", dtype=torch.bfloat16)
    if case["layout"] == "strided":
        q, k, v = (make_strided(t) for t in (q, k, v))
    scale = case["sm_scale"]
    segments = None
    if case["path"] == "segmented":
        block, starts, ends = cp_geometry(case)
        lengths, prefixes = [[block, block]] * b, [list(ends)] * b
        run = lambda: dense_prefill_attention_fwd_segmented(
            q, k, v, q_segment_lens=lengths, kv_prefix_lens=prefixes, sm_scale=scale)
        segments = [(0, block, starts[0]), (block, block, starts[1])]
    else:
        run = lambda: dense_prefill_attention_fwd_4d(q, k, v, sm_scale=scale)
    actual = run()
    return run, prefill_reference(q, k, v, actual, scale, segments)


def _build_decode(case):
    from boltops.fa import decode_attention_fwd_4d

    b, h, length, d = (case[k] for k in ("B", "H", "Sk", "Dqk"))
    q = torch.randn(b, h, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.empty((b, h, length, d), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    for bi in range(b):
        for cache in (k, v):
            for hi in range(0, h, 8):
                cache[bi, hi:hi + 8].copy_(torch.randn(
                    min(8, h - hi), length, d, device="cuda", dtype=torch.bfloat16).to(cache.dtype))
    ks, vs, scale = (case[k] for k in ("k_scale", "v_scale", "sm_scale"))
    page = case["page_size"]
    if page:
        gen = torch.Generator().manual_seed(20261021 + page)
        if case.get("page_mapping") == "reverse_requests":
            assert page == length, "reverse_requests requires one page per request"
            permutation = torch.arange(b - 1, -1, -1, device="cuda")
        else:
            permutation = torch.randperm(b * length // page, generator=gen).cuda()
        kp, vp = (paged_fixture(t, page, permutation) for t in (k, v))
        table = permutation.reshape(b, length // page).to(torch.int32)
        lens = torch.full((b,), length, device="cuda", dtype=torch.int32)
        run = lambda: decode_attention_fwd_4d(
            q, kp, vp, sm_scale=scale, k_scale=ks, v_scale=vs,
            block_table=table, seq_lens=lens)
    else:
        run = lambda: decode_attention_fwd_4d(q, k, v, sm_scale=scale, k_scale=ks, v_scale=vs)
    actual = run()
    return run, decode_reference(q, k, v, actual, scale, ks, vs)


@contextlib.contextmanager
def strict_fp32_matmul():
    """Keep FP32 references free of TF32, then restore the caller's settings."""
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        # The precision setter also rewrites allow_tf32; restore that last.
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def tensor_net_bytes(case):
    b, h, sq, sk, dq, dv = (case[k] for k in ("B", "H", "Sq", "Sk", "Dqk", "Dv"))
    kv_bytes = 1 if case["path"] == "decode" else 2
    return b * h * (2 * sq * (dq + dv) + kv_bytes * sk * (dq + dv))


def prepare(case):
    """Build inputs and check correctness; timing belongs to the shared fixture."""
    print(f"[fa] prepare {case['case_id']}", flush=True)
    # Pytest can release the previous fixture's cached callable only after its
    # finalizer has run. Reclaim those now-inactive blocks before the next case.
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    net = tensor_net_bytes(case)
    capacity = (f"{case['case_id']}: original B={case['B']} needs >= {net/2**30:.3f} GiB; "
                f"device total={total/2**30:.3f} GiB")
    if net > total:
        pytest.skip(f"{capacity}; minimum exceeds device total")
    torch.manual_seed(case["seed"])
    oom = None
    try:
        with strict_fp32_matmul():
            run, result = _build_decode(case) if case["path"] == "decode" else _build_prefill(case)
    except torch.cuda.OutOfMemoryError as exc:
        oom = f"{capacity}; OOM during input/operator/reference preparation: {exc}"
    if oom is not None:
        # Leave the except block first: its traceback can retain partial inputs.
        gc.collect()
        torch.cuda.empty_cache()
        pytest.skip(oom)
    print(f"[reference] {case['case_id']}: {result}", flush=True)
    assert result["pass"], result
    return run


@pytest.fixture(scope="module")
def prepared_case(request):
    case = request.param
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")
    prepared = None
    try:
        prepared = prepare(case)
        yield case, prepared
    finally:
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


def useful_flops(case):
    sq, sk = case["Sq"], case["Sk"]
    if case["path"] == "segmented":
        block, starts, _ = cp_geometry(case)
        pairs = sum(block * (start + 1) + block * (block - 1) // 2 for start in starts)
    else:
        pairs = sq * (sk - sq + 1) + sq * (sq - 1) // 2
    return 2.0 * case["B"] * case["H"] * pairs * (case["Dqk"] + case["Dv"])
