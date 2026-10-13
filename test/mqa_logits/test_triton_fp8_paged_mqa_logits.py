"""Correctness and optional GPU-latency benches for paged FP8 MQA logits.

Default pytest run: correctness / contract tests (``kv_block_size`` 1 and 64).

Perf (needs ``--bench``; CUDA-graph pure GPU). Production path timed is
``kv_block_size=64`` SplitKV + ``fuse_output_init=True``::

    pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s
    pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc

Standalone CLI (same benches)::

    python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py
    python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --vs-hipc
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Optional

import pytest
import torch

pytest.importorskip("triton")

from boltops.mqa_logits import (  # noqa: E402
    triton_fp8_paged_mqa_logits,
    triton_fp8_paged_mqa_logits_build_bucket_schedule,
    triton_fp8_paged_mqa_logits_prepare,
    triton_fp8_paged_mqa_logits_stage1,
)
from boltops.utility.arch_info import get_fp8_e4m3_dtype  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

e4m3_type = get_fp8_e4m3_dtype()
fp8_max = torch.finfo(e4m3_type).max


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Pack ``[num_blocks, block_size, 1, D]`` bf16/fp32 keys into ``[..., D+4]``."""
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    x_scaled = (x * (1.0 / sf)).to(e4m3_type)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def _decode_packed_kv_cache(
    kv_cache_fp8: torch.Tensor, hidden_dim: int, block_size: int
) -> torch.Tensor:
    num_blocks = kv_cache_fp8.size(0)
    flat = kv_cache_fp8.view(num_blocks, block_size * (hidden_dim + 4))
    kv_fp8 = flat[..., : block_size * hidden_dim].view(num_blocks, block_size, hidden_dim)
    kv_scale = flat[..., block_size * hidden_dim :].view(torch.float32).view(
        num_blocks, block_size, 1
    )
    return kv_fp8.view(e4m3_type).float() * kv_scale


def ref_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    block_size = kv_cache_fp8.size(1)
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    q = q_fp8.float()
    keys_by_block = _decode_packed_kv_cache(kv_cache_fp8, hidden_dim, block_size)

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        for n in range(next_n):
            row = b * next_n + n
            q_offset = ctx_len - next_n + n
            for t in range(ctx_len):
                phys = int(kv_indices[b, t].item())
                kx = keys_by_block[phys, t % block_size]
                scores = (q[b, n] * kx).sum(dim=1)
                weighted = torch.relu(scores) * weights[row]
                logits[row, t] = weighted.sum()
                if t > q_offset:
                    logits[row, t] = float("-inf")
    return logits


def ref_fp8_paged_mqa_logits_block64(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Reference for ``block_size=64`` + ``block_table`` (hipc layout)."""
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    block_size = kv_cache_fp8.size(1)
    assert block_size == 64
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    q = q_fp8.float()
    keys_by_block = _decode_packed_kv_cache(kv_cache_fp8, hidden_dim, block_size)

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        for n in range(next_n):
            row = b * next_n + n
            q_offset = ctx_len - next_n + n
            for t in range(ctx_len):
                block_rk = t // block_size
                phys = int(block_table[b, block_rk].item())
                kx = keys_by_block[phys, t % block_size]
                scores = (q[b, n] * kx).sum(dim=1)
                weighted = torch.relu(scores) * weights[row]
                logits[row, t] = weighted.sum()
                if t > q_offset:
                    logits[row, t] = float("-inf")
    return logits


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,seed",
    [
        (1, 1, 64, 128, 128, [59], 0),
        (2, 1, 64, 128, 256, [73, 121], 1),
        (1, 2, 64, 128, 512, [487], 2),
        (2, 2, 64, 128, 1024, [153, 987], 3),
        (4, 1, 32, 128, 2048, [401, 888, 1333, 1999], 4),
    ],
)
def test_triton_fp8_paged_mqa_logits_block64_correctness(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    seed,
):
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    block_size = 64
    num_blocks = max_model_len * 2

    q = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    q_fp8 = q.to(e4m3_type)
    kv = torch.randn(
        (num_blocks, block_size, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE, dtype=torch.float32)
    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)
    assert torch.all(context_lens >= next_n)

    max_blocks = _cdiv(max(context_lens_list), block_size)
    block_table = torch.zeros((batch_size, max_blocks), device=DEVICE, dtype=torch.int32)
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    counter = 0
    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        for j in range(_cdiv(ctx_len, block_size)):
            block_table[b, j] = block_idx_pool[counter]
            counter += 1

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        out,
        context_lens,
        block_table,
        max_model_len,
        kv_block_size=64,
    )
    ref = ref_fp8_paged_mqa_logits_block64(
        q_fp8, kv_fp8, weights, context_lens, block_table, max_model_len
    )

    positions = torch.arange(max_model_len, device=DEVICE).unsqueeze(0)
    row_to_batch = torch.arange(batch_size * next_n, device=DEVICE) // next_n
    row_to_n = torch.arange(batch_size * next_n, device=DEVICE) % next_n
    valid_mask = positions <= (
        context_lens[row_to_batch] - next_n + row_to_n
    ).unsqueeze(1)

    assert torch.isfinite(out[valid_mask]).all()
    # Outside the causal window must be exact -inf (not merely very negative).
    assert torch.isneginf(out[~valid_mask]).all()
    torch.testing.assert_close(out[valid_mask], ref[valid_mask], rtol=5e-2, atol=5e-2)


def _build_block64_inputs(
    batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list, seed
):
    """Shared input builder for the block-table path.

    The block table deliberately scatters physical pages so a bad page lookup
    shows up as a numeric mismatch rather than silently reading the right row.
    """
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    block_size = 64
    num_blocks = max_model_len * 2

    q = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    q_fp8 = q.to(e4m3_type)
    kv = torch.randn(
        (num_blocks, block_size, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn(
        (batch_size * next_n, heads), device=DEVICE, dtype=torch.float32
    )
    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)
    assert torch.all(context_lens >= next_n)

    max_blocks = _cdiv(max(context_lens_list), block_size)
    block_table = torch.zeros((batch_size, max_blocks), device=DEVICE, dtype=torch.int32)
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    counter = 0
    for b in range(batch_size):
        for j in range(_cdiv(int(context_lens[b].item()), block_size)):
            block_table[b, j] = block_idx_pool[counter]
            counter += 1

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    return q_fp8, kv_fp8, weights, context_lens, block_table, out


def _valid_logits_mask(context_lens, batch_size, next_n, max_model_len):
    positions = torch.arange(max_model_len, device=DEVICE).unsqueeze(0)
    rows = torch.arange(batch_size * next_n, device=DEVICE)
    return positions <= (context_lens[rows // next_n] - next_n + rows % next_n).unsqueeze(1)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,"
    "bucket_tokens,seed",
    [
        (1, 1, 64, 128, 4096, [59], 2048, 0),          # one bucket
        (2, 1, 64, 128, 8192, [2049, 7000], 2048, 2),  # imbalanced
        (1, 1, 1, 128, 512, [59], 256, 6),             # non-MMA heads=1
    ],
)
def test_triton_fp8_paged_mqa_logits_block64_bucket_correctness(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    bucket_tokens,
    seed,
):
    """Bucketed scheduling must match the reference exactly.

    Bucketing only changes *which program walks which KV range*, so the numeric
    result must be bit-identical to the block kernel; any drift means a bucket
    was dropped, double counted, or attributed to the wrong row.
    """
    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list, seed
    )
    triton_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        out,
        context_lens,
        block_table,
        max_model_len,
        kv_block_size=64,
        use_bucket=True,
        bucket_tokens=bucket_tokens,
    )
    ref = ref_fp8_paged_mqa_logits_block64(
        q_fp8, kv_fp8, weights, context_lens, block_table, max_model_len
    )

    valid_mask = _valid_logits_mask(context_lens, batch_size, next_n, max_model_len)
    assert torch.isfinite(out[valid_mask]).all()
    torch.testing.assert_close(out[valid_mask], ref[valid_mask], rtol=5e-2, atol=5e-2)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,seed",
    [
        (2, 1, 64, 128, 8192, [2049, 7000], 11),
        (4, 2, 64, 128, 4096, [211, 777, 943, 3991], 12),
    ],
)
def test_triton_fp8_paged_mqa_logits_bucket_matches_block_kernel(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    seed,
):
    """The two schedulers must agree bit-for-bit on the same inputs.

    This is the tighter check of the pair: comparing both against the reference
    could hide a shared error, whereas a direct comparison pins the bucket path
    to the already-validated block kernel.
    """
    bucket_tokens = 512
    q_fp8, kv_fp8, weights, context_lens, block_table, out_block = _build_block64_inputs(
        batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list, seed
    )
    out_bucket = out_block.clone()

    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out_block, context_lens, block_table, max_model_len,
        kv_block_size=64,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out_bucket, context_lens, block_table, max_model_len,
        kv_block_size=64, use_bucket=True, bucket_tokens=bucket_tokens,
    )

    valid_mask = _valid_logits_mask(context_lens, batch_size, next_n, max_model_len)
    torch.testing.assert_close(
        out_block[valid_mask], out_bucket[valid_mask], rtol=0.0, atol=0.0
    )


def test_triton_resolve_split_kv_pins_the_measured_regimes():
    """The heuristic is host arithmetic; pin the cases the A/B matrix used.

    Numbers are for CU=72, waves_per_eu=2, chunk_k=64 -- the machine the
    short-row term was measured on. The occupancy term still depends on CU
    count, so these are exact only at that CU; the short-row / long-row
    *gating* (which term binds) is independent of CU as long as occupancy
    stays below the short-row bound, which it does at 72.
    """
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import _resolve_split_kv

    cu, waves, ck = 72, 2, 64

    def sk(batch, next_n, max_ctx):
        return _resolve_split_kv(
            None,
            tile_q_count=batch * next_n,
            total_cu_count=cu,
            waves_per_eu=waves,
            max_ctx=max_ctx,
            chunk_k=ck,
        )

    # Caller override is absolute.
    assert _resolve_split_kv(
        7, tile_q_count=64, total_cu_count=cu, waves_per_eu=waves,
        max_ctx=4504, chunk_k=ck,
    ) == 7
    # No max_ctx: occupancy only (batch=64 -> 10).
    assert _resolve_split_kv(
        None, tile_q_count=64, total_cu_count=cu, waves_per_eu=waves,
        max_ctx=None, chunk_k=ck,
    ) == 10

    # Long rows: short-row term must not fire. batch=1 uses the 8-wave floor.
    assert sk(1, 1, 64844) == 576
    assert sk(2, 1, 76239) == 288
    assert sk(64, 1, 78697) == 52
    assert sk(128, 1, 79168) == 52
    # Just past the short-row gate (8192/64=128 tiles).
    assert sk(64, 1, 8192) == 10

    # Short rows (~71 tiles): 3 tiles/program, capped at 2048 programs.
    assert sk(32, 1, 4504) == 24
    assert sk(64, 1, 4504) == 24
    assert sk(128, 1, 4504) == 16
    assert sk(256, 1, 4504) == 10
    assert sk(64, 2, 4504) == 16
    assert sk(128, 2, 4504) == 10


def test_triton_build_paged_bucket_schedule_layout():
    """Schedule shape/contents are pure host logic, so pin them without a GPU."""
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import (
        build_paged_bucket_schedule,
    )

    ctx = torch.tensor([1, 2048, 2049, 7000], dtype=torch.int32)
    sched = build_paged_bucket_schedule(ctx, bucket_tokens=2048, max_model_len=8192)

    # ceil(1/2048)=1, ceil(2048/2048)=1, ceil(2049/2048)=2, ceil(7000/2048)=4.
    assert sched.tolist() == [
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 2048],
        [3, 0],
        [3, 2048],
        [3, 4096],
        [3, 6144],
    ]
    assert sched.dtype == torch.int32

    # Every row must contribute a contiguous, gap-free run of buckets.
    for row in range(ctx.numel()):
        starts = [k for r, k in sched.tolist() if r == row]
        assert starts == [i * 2048 for i in range(len(starts))]

    # context_lens longer than max_model_len is clamped, matching the kernel's
    # store mask so no bucket walks past the output width.
    clamped = build_paged_bucket_schedule(
        torch.tensor([99999], dtype=torch.int32), bucket_tokens=2048, max_model_len=4096
    )
    assert clamped.tolist() == [[0, 0], [0, 2048]]

    # Zero rows must not crash on the empty reduce.
    empty = build_paged_bucket_schedule(torch.zeros((0,), dtype=torch.int32))
    assert empty.shape == (0, 2)

    # next_n expands to one entry per *query row*, since the kernel indexes the
    # schedule by ``batch * next_n + n_idx`` rather than by batch.
    sched_n2 = build_paged_bucket_schedule(
        torch.tensor([100, 3000], dtype=torch.int32),
        bucket_tokens=2048,
        max_model_len=8192,
        next_n=2,
    )
    assert sched_n2.tolist() == [
        [0, 0],          # batch 0, n=0, ctx=100  -> 1 bucket
        [1, 0],          # batch 0, n=1, ctx=100  -> 1 bucket
        [2, 0], [2, 2048],  # batch 1, n=0, ctx=3000 -> 2 buckets
        [3, 0], [3, 2048],  # batch 1, n=1, ctx=3000 -> 2 buckets
    ]
    assert (
        build_paged_bucket_schedule(
            torch.zeros((0,), dtype=torch.int32), next_n=2
        ).shape
        == torch.Size([0, 2])
    )


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,context_lens_list,bucket_tokens",
    [
        (1, 1, [59], 128),
        (1, 1, [3072], 128),
        (2, 1, [2049, 7000], 128),
        (2, 1, [2049, 7000], 1024),
        (4, 2, [211, 777, 943, 3991], 128),
        (4, 2, [211, 777, 943, 3991], 256),
        (5, 1, [1, 128, 129, 4096, 4097], 128),
        # One row, one bucket: the binary search's degenerate case.
        (1, 1, [1], 128),
        # Every row a single bucket, so the partition is trivial but the
        # row-advance path never runs.
        (3, 1, [100, 100, 100], 128),
    ],
)
def test_triton_build_bucket_schedule_device_matches_host(
    batch_size, next_n, context_lens_list, bucket_tokens
):
    """The device partitioner must reproduce the host one exactly.

    ``build_paged_bucket_schedule`` is the definition the kernel was validated
    against, so the device prefix sum is only correct if it yields the same
    per-row bucket counts and the same total. Any drift is an off-by-one that
    would silently drop or double-count a KV range.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import (
        build_paged_bucket_schedule,
    )

    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)
    max_model_len = max(context_lens_list) + 128
    num_rows = batch_size * next_n

    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n,
        bucket_tokens=bucket_tokens,
    )
    host = build_paged_bucket_schedule(
        context_lens, bucket_tokens=bucket_tokens,
        max_model_len=max_model_len, next_n=next_n,
    )

    host_counts = torch.bincount(
        host[:, 0].to(torch.int64), minlength=num_rows
    ).to(torch.int32)
    dev_counts = sch.base[1 : num_rows + 1] - sch.base[:num_rows]

    assert sch.num_rows == num_rows
    assert sch.bucket_tokens == bucket_tokens
    assert torch.equal(dev_counts, host_counts)
    assert int(sch.base[num_rows].item()) == int(host.size(0))
    # ``base`` is a monotone prefix sum, which is what makes the binary search
    # in the compute kernel valid.
    assert torch.all(sch.base[: num_rows + 1][1:] >= sch.base[: num_rows + 1][:-1])


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,bucket_tokens,seed",
    [
        (2, 1, 64, 128, 8192, [2049, 7000], 128, 32),
        (4, 2, 64, 128, 4096, [211, 777, 943, 3991], 128, 33),
        (1, 1, 1, 128, 512, [59], 128, 35),
    ],
)
def test_triton_persistent_bucket_matches_block_kernel(
    batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list,
    bucket_tokens, seed,
):
    """The persistent bucket kernel must be bit-identical to the block kernel.

    The S curves of the two schedulers differ (fixed SplitKV grid vs contiguous
    slices of a device-built bucket list), but which program walks which KV range
    must not change a single logit. Bitwise equality is the right bar here: a
    mismatch means a range was dropped, double counted, or given to the wrong row.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list, seed
    )

    # ``out`` is pre-filled with -inf, which the bucket kernel requires: it only
    # writes whole buckets, so slots past a row's coverage keep the fill.
    ref = out.clone()
    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, ref, context_lens, block_table, max_model_len,
        kv_block_size=64,
    )

    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n,
        bucket_tokens=bucket_tokens,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out, context_lens, block_table, max_model_len,
        kv_block_size=64, bucket_schedule=sch,
    )

    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


@GPU_REQUIRED
def test_triton_persistent_bucket_matches_host_schedule_bucket():
    """The device schedule must drive the same partition as the host one.

    Bitwise equality between the two bucket paths is a stronger statement than
    each matching the block kernel: it pins the device prefix sum to the host
    list element for element, since the two kernels walk identical ranges.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    bucket_tokens = 256
    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        4, 2, 64, 128, 4096, [211, 777, 943, 3991], 37
    )
    out_host = out.clone()
    out_dev = out.clone()

    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out_host, context_lens, block_table, 4096,
        kv_block_size=64, use_bucket=True, bucket_tokens=bucket_tokens,
    )
    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=4, next_n=2, bucket_tokens=bucket_tokens,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out_dev, context_lens, block_table, 4096,
        kv_block_size=64, bucket_schedule=sch,
    )

    torch.testing.assert_close(out_dev, out_host, rtol=0.0, atol=0.0)


@GPU_REQUIRED
def test_triton_persistent_bucket_reuses_scratch_buffer():
    """A caller-supplied ``base`` must be overwritten in place, not appended to.

    This is the property that makes the op graph-capturable: the serving loop
    allocates the scratch once and every step rewrites it, so no allocation (and
    no address change) happens inside the capture.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    batch_size, next_n = 4, 1
    max_rows = 16  # over-sized on purpose: mirrors a buffer shared across graphs
    scratch = torch.full((max_rows + 1,), 12345, dtype=torch.int32, device=DEVICE)

    first_lens = [211, 777, 943, 3991]
    second_lens = [1, 128, 129, 4097]

    context_lens = torch.tensor(first_lens, device=DEVICE, dtype=torch.int32)
    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n, bucket_tokens=128,
        base=scratch,
    )
    fresh = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n, bucket_tokens=128,
    )
    assert torch.equal(sch.base[: batch_size + 1], fresh.base)

    # Rewriting with a different batch must fully replace the old contents, and
    # must not have touched the tail of the over-sized buffer.
    context_lens = torch.tensor(second_lens, device=DEVICE, dtype=torch.int32)
    sch2 = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n, bucket_tokens=128,
        base=scratch,
    )
    fresh2 = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n, bucket_tokens=128,
    )
    assert torch.equal(sch2.base[: batch_size + 1], fresh2.base)
    assert torch.all(scratch[batch_size + 1 :] == 12345)


@GPU_REQUIRED
def test_triton_persistent_bucket_tolerates_empty_context_rows():
    """A zero-length context must not stall the row walk.

    The op trusts the caller for numeric ``context_lens`` bounds. A row with
    ``ctx == 0`` owns no buckets, which makes ``base`` flat and can hand the
    binary search a row that owns nothing at all. The row walk has to skip it.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        3, 1, 64, 128, 4096, [1024, 2048, 3072], 41
    )
    context_lens = context_lens.clone()
    context_lens[1] = 0

    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=3, bucket_tokens=128,
    )
    # The empty row owns no bucket, so its ``base`` entry repeats its neighbour's.
    assert int(sch.base[2].item()) == int(sch.base[1].item())

    triton_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, out, context_lens, block_table, 4096,
        kv_block_size=64, bucket_schedule=sch,
    )
    torch.cuda.synchronize()

    # The empty row keeps the -inf fill; the rows around it are unaffected.
    assert torch.all(out[1] == float("-inf"))
    assert torch.isfinite(out[0, :1000]).any()
    assert torch.isfinite(out[2, :2000]).any()


@GPU_REQUIRED
def test_triton_persistent_bucket_enables_graph_capture():
    """Capture and replay must reproduce the eager result.

    The grid is a host constant and the partition lives in a device buffer, so
    nothing in the launch path reads ``context_lens`` back. That is what lets the
    capture succeed where the host-schedule path cannot.
    """
    from boltops.mqa_logits import (
        triton_fp8_paged_mqa_logits_build_bucket_schedule,
        triton_fp8_paged_mqa_logits_prepare,
    )

    batch_size, next_n, max_model_len = 2, 1, 8192
    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        batch_size, next_n, 64, 128, max_model_len, [2049, 7000], 38
    )
    # Both prepare-time artifacts: host max_ctx (for SplitKV) and the device
    # partition. Missing max_ctx only weakens the heuristic; a host read of
    # context_lens for scheduling (e.g. use_bucket) is what capture rejects.
    info = triton_fp8_paged_mqa_logits_prepare(
        q_fp8, kv_fp8, context_lens, block_table
    )
    scratch = torch.empty(batch_size + 1, dtype=torch.int32, device=DEVICE)
    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=batch_size, next_n=next_n, bucket_tokens=128,
        base=scratch, max_ctx=info.max_ctx,
    )

    eager = out.clone()

    def call(buf):
        triton_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, weights, buf, context_lens, block_table, max_model_len,
            kv_block_size=64, bucket_schedule=sch, batch_info=info,
        )

    call(eager)

    graph = torch.cuda.CUDAGraph()
    captured = torch.full_like(out, float("-inf"))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            call(captured)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        call(captured)
    captured.fill_(float("-inf"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, rtol=0.0, atol=0.0)


@GPU_REQUIRED
def test_triton_persistent_bucket_rejects_row_count_mismatch():
    """A schedule built for another batch shape must be rejected, not mis-indexed.

    The kernel trusts ``num_rows`` to bound its binary search, so a stale
    schedule from a different graph bucket would read the wrong row silently.
    """
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        2, 1, 64, 128, 8192, [2049, 7000], 39
    )
    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=2, next_n=2, bucket_tokens=128,
    )
    with pytest.raises(ValueError, match="query rows"):
        triton_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, weights, out, context_lens, block_table, 8192,
            kv_block_size=64, bucket_schedule=sch,
        )


@GPU_REQUIRED
def test_triton_persistent_bucket_rejects_use_bucket_combo():
    """``bucket_schedule`` and ``use_bucket`` must not be combined."""
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        2, 1, 64, 128, 8192, [2049, 7000], 40
    )
    sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, batch_size=2, bucket_tokens=128,
    )
    with pytest.raises(ValueError, match="supersedes"):
        triton_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, weights, out, context_lens, block_table, 8192,
            kv_block_size=64, bucket_schedule=sch, use_bucket=True,
        )


@GPU_REQUIRED
def test_triton_build_bucket_schedule_validates_arguments():
    """Bad inputs must raise rather than launch a kernel on garbage."""
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    context_lens = torch.tensor([2049, 7000], device=DEVICE, dtype=torch.int32)

    with pytest.raises(ValueError, match="bucket_tokens"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(context_lens, bucket_tokens=0)
    with pytest.raises(ValueError, match="next_n"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(context_lens, next_n=0)
    with pytest.raises(ValueError, match="num_programs"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(
            context_lens, num_programs=0
        )
    with pytest.raises(ValueError, match="at least one query row"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(
            torch.zeros((0,), device=DEVICE, dtype=torch.int32), batch_size=0
        )
    with pytest.raises(ValueError, match="int32"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(
            context_lens, base=torch.zeros(8, dtype=torch.float32, device=DEVICE)
        )
    with pytest.raises(ValueError, match="at least"):
        triton_fp8_paged_mqa_logits_build_bucket_schedule(
            context_lens, base=torch.zeros(2, dtype=torch.int32, device=DEVICE)
        )


@GPU_REQUIRED
def test_triton_default_bucket_num_programs_clamps_around_estimate():
    """The default grid is the bucket-count estimate, clamped at both ends.

    Exercised directly because the clamp is the only thing keeping a wide, skewed
    batch from launching an order of magnitude more programs than it has work
    for, and that is not visible from any single end-to-end number.
    """
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import (
        DEFAULT_BUCKET_MAX_PROGRAMS_PER_CU,
        DEFAULT_BUCKET_MIN_PROGRAMS_PER_CU,
        DEFAULT_BUCKET_WAVES_PER_EU,
        _default_bucket_num_programs,
        _default_total_cu_count,
    )

    cu = _default_total_cu_count()
    floor = cu * DEFAULT_BUCKET_MIN_PROGRAMS_PER_CU
    ceiling = cu * DEFAULT_BUCKET_MAX_PROGRAMS_PER_CU

    # No max_ctx: fall back to the flat waves default.
    assert _default_bucket_num_programs(64, 1, None, 128) == cu * DEFAULT_BUCKET_WAVES_PER_EU

    # Genuinely small work -> floor, not the estimate.
    assert _default_bucket_num_programs(1, 1, 128, 128) == floor
    # Huge batch -> ceiling.
    assert _default_bucket_num_programs(100_000, 1, 110_000, 128) == ceiling

    # In between the estimate is used verbatim: batch * cdiv(max_ctx, bucket).
    batch = max(4, (cu * 8) // 32)
    assert (
        _default_bucket_num_programs(batch, 1, 4096, 128)
        == batch * _cdiv(4096, 128)
    )
    # next_n scales the row count, hence the estimate.
    assert _default_bucket_num_programs(batch, 2, 4096, 128) == min(
        ceiling, batch * 2 * _cdiv(4096, 128)
    )


@GPU_REQUIRED
def test_triton_paged_prepare_returns_batch_info():
    """``prepare`` records shapes and resolves ``max_ctx`` (no shape asserts)."""
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_prepare

    batch_size, next_n, heads, hidden_dim, max_model_len = 4, 2, 64, 128, 4096
    lens = [2049, 777, 3991, 211]
    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        batch_size, next_n, heads, hidden_dim, max_model_len, lens, 21
    )
    info = triton_fp8_paged_mqa_logits_prepare(
        q_fp8, kv_fp8, context_lens, block_table
    )
    assert info.batch_size == batch_size
    assert info.next_n == next_n
    assert info.heads == heads
    assert info.hidden_dim == hidden_dim
    assert info.block_size == 64
    # max_ctx is the real max, which is what lets the SplitKV heuristic stay
    # context-aware on the prepared (sync-free) path.
    assert info.max_ctx == max(lens)


@GPU_REQUIRED
def test_triton_paged_prepare_output_matches_unprepared():
    """Passing ``batch_info`` must not change the result, only the host path.

    The split_kv heuristic may legitimately pick a different value than the
    unprepared call (only the prepared path supplies ``max_ctx``), and SplitKV
    is numerically exact -- each program owns a disjoint KV band. So the two
    must agree exactly.
    """
    from boltops.mqa_logits import (
        triton_fp8_paged_mqa_logits_prepare,
        triton_fp8_paged_mqa_logits as _op,
    )

    batch_size, next_n, heads, hidden_dim = 2, 1, 64, 128
    max_model_len, context_lens_list, seed = 8192, [2049, 7000], 22
    q_fp8, kv_fp8, weights, context_lens, block_table, out_ref = _build_block64_inputs(
        batch_size, next_n, heads, hidden_dim, max_model_len, context_lens_list, seed
    )
    out_prep = out_ref.clone()

    _op(q_fp8, kv_fp8, weights, out_ref, context_lens, block_table, max_model_len,
        kv_block_size=64)
    info = triton_fp8_paged_mqa_logits_prepare(
        q_fp8, kv_fp8, context_lens, block_table
    )
    _op(q_fp8, kv_fp8, weights, out_prep, context_lens, block_table, max_model_len,
        kv_block_size=64, batch_info=info)

    valid = _valid_logits_mask(context_lens, batch_size, next_n, max_model_len)
    torch.testing.assert_close(
        out_ref[valid], out_prep[valid], rtol=0.0, atol=0.0
    )


@GPU_REQUIRED
def test_triton_paged_batch_info_enables_graph_capture():
    """Prepared and unprepared paths are both graph-capturable.

    The launch path never reads ``context_lens`` on the host. ``prepare`` still
    matters for context-aware SplitKV (``max_ctx``) without a per-step sync.
    """
    from boltops.mqa_logits import (
        triton_fp8_paged_mqa_logits_prepare,
        triton_fp8_paged_mqa_logits as _op,
    )

    max_model_len = 2048
    q_fp8, kv_fp8, weights, context_lens, block_table, out = _build_block64_inputs(
        4, 1, 64, 128, max_model_len, [512, 900, 1300, 1999], 26
    )

    def capture(**kwargs):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _op(q_fp8, kv_fp8, weights, out, context_lens, block_table,
                    max_model_len, kv_block_size=64, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _op(q_fp8, kv_fp8, weights, out, context_lens, block_table,
                max_model_len, kv_block_size=64, **kwargs)
        g.replay()
        torch.cuda.synchronize()
        return g

    info = triton_fp8_paged_mqa_logits_prepare(
        q_fp8, kv_fp8, context_lens, block_table
    )

    graph = capture(batch_info=info)
    assert graph is not None

    # Capturing must not have cost correctness: a replayed graph has to produce
    # the same logits as a direct call. Overwrite the buffer with NaN first so a
    # position the graph fails to write shows up as NaN instead of a stale value
    # masking the bug.
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    ref = ref_fp8_paged_mqa_logits_block64(
        q_fp8, kv_fp8, weights, context_lens, block_table, max_model_len
    )
    valid = _valid_logits_mask(context_lens, 4, 1, max_model_len)
    assert torch.isfinite(out[valid]).all(), "graph replay left NaN in valid outputs"
    torch.testing.assert_close(out[valid], ref[valid], rtol=5e-2, atol=5e-2)

    assert capture() is not None


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,seed",
    [
        (1, 1, 64, 128, 64, [59], 0),
        (2, 2, 64, 128, 128, [73, 121], 1),
        (1, 2, 64, 128, 1024, [987], 2),
        (2, 2, 64, 128, 2048, [1536, 1987], 3),
        (4, 2, 64, 128, 1024, [211, 777, 943, 1001], 4),
        (1, 1, 64, 128, 4096, [3072], 5),
        (2, 1, 64, 128, 4096, [2049, 3991], 6),
        (4, 1, 64, 128, 2048, [401, 888, 1333, 1999], 7),
        # Small head counts exercise the non-MMA lowering path, which emits an
        # explicit fp8 -> fp32 cast (rounding = rtne) instead of feeding fp8
        # straight into the MMA. Regression guard for the HCU compiler map gap.
        # max_model_len must be >= max(context_lens).
        (1, 1, 1, 128, 512, [59], 8),
        (2, 1, 2, 128, 512, [73, 121], 9),
        (2, 2, 4, 128, 512, [153, 487], 10),
        (4, 1, 8, 128, 1024, [401, 888, 999, 1013], 11),
    ],
)
def test_triton_fp8_paged_mqa_logits_correctness(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    seed,
):
    torch.manual_seed(seed)
    block_size = 1
    num_blocks = max_model_len

    q = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    q_fp8 = q.to(e4m3_type)
    kv = torch.randn(
        (num_blocks, block_size, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE, dtype=torch.float32)
    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)
    assert context_lens.numel() == batch_size
    assert torch.all(context_lens >= next_n)

    kv_indices = torch.zeros((batch_size, max_model_len), device=DEVICE, dtype=torch.int32)
    for b in range(batch_size):
        kv_indices[b, : context_lens[b]] = torch.randperm(max_model_len, device=DEVICE)[
            : context_lens[b]
        ]

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )

    triton_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        out,
        context_lens,
        kv_indices,
        max_model_len,
    )
    ref = ref_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, context_lens, kv_indices, max_model_len
    )

    positions = torch.arange(max_model_len, device=DEVICE).unsqueeze(0)
    row_to_batch = torch.arange(batch_size * next_n, device=DEVICE) // next_n
    row_to_n = torch.arange(batch_size * next_n, device=DEVICE) % next_n
    valid_mask = positions <= (
        context_lens[row_to_batch] - next_n + row_to_n
    ).unsqueeze(1)

    assert torch.isfinite(out[valid_mask]).all()
    assert torch.isneginf(out[~valid_mask]).all()
    torch.testing.assert_close(out[valid_mask], ref[valid_mask], rtol=5e-2, atol=5e-2)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,seed",
    [
        (1, 1, 64, 128, 64, [59], 0),
        (2, 2, 64, 128, 128, [73, 121], 1),
        (1, 2, 64, 128, 1024, [987], 2),
        (4, 1, 64, 128, 2048, [401, 888, 1333, 1999], 7),
    ],
)
def test_triton_fp8_paged_mqa_logits_stage1_matches_full(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    seed,
):
    torch.manual_seed(seed)
    block_size = 1
    num_blocks = max_model_len

    q = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    q_fp8 = q.to(e4m3_type)
    kv = torch.randn(
        (num_blocks, block_size, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE, dtype=torch.float32)
    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)

    kv_indices = torch.zeros((batch_size, max_model_len), device=DEVICE, dtype=torch.int32)
    for b in range(batch_size):
        kv_indices[b, : context_lens[b]] = torch.randperm(max_model_len, device=DEVICE)[
            : context_lens[b]
        ]

    out_full = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        out_full,
        context_lens,
        kv_indices,
        max_model_len,
    )

    out_qk = torch.full(
        (heads, batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits_stage1(
        q_fp8,
        kv_fp8,
        weights,
        out_qk,
        context_lens,
        kv_indices,
        max_model_len,
        chunk_q=heads,
    )
    out_stage1 = out_qk.sum(dim=0)

    positions = torch.arange(max_model_len, device=DEVICE).unsqueeze(0)
    row_to_batch = torch.arange(batch_size * next_n, device=DEVICE) // next_n
    row_to_n = torch.arange(batch_size * next_n, device=DEVICE) % next_n
    valid_mask = positions <= (
        context_lens[row_to_batch] - next_n + row_to_n
    ).unsqueeze(1)

    torch.testing.assert_close(
        out_stage1[valid_mask], out_full[valid_mask], rtol=5e-2, atol=5e-2
    )


@GPU_REQUIRED
@pytest.mark.parametrize(
    "batch_size,next_n,heads,hidden_dim,max_model_len,context_lens_list,seed",
    [
        (1, 1, 64, 128, 64, [59], 0),
        (2, 2, 64, 128, 128, [73, 121], 1),
        (1, 2, 64, 128, 1024, [987], 2),
        (4, 1, 64, 128, 2048, [401, 888, 1333, 1999], 7),
    ],
)
def test_triton_fp8_paged_mqa_logits_stage1_block64_matches_full(
    batch_size,
    next_n,
    heads,
    hidden_dim,
    max_model_len,
    context_lens_list,
    seed,
):
    """stage1 with kv_block_size=64 must sum to the full (dense) kernel output."""
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    block_size = 64
    num_blocks = max_model_len * 2

    q = torch.randn(
        (batch_size, next_n, heads, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    q_fp8 = q.to(e4m3_type)
    kv = torch.randn(
        (num_blocks, block_size, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE, dtype=torch.float32)
    context_lens = torch.tensor(context_lens_list, device=DEVICE, dtype=torch.int32)

    max_blocks = _cdiv(max(context_lens_list), block_size)
    block_table = torch.zeros((batch_size, max_blocks), device=DEVICE, dtype=torch.int32)
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    counter = 0
    for b in range(batch_size):
        for j in range(_cdiv(int(context_lens[b].item()), block_size)):
            block_table[b, j] = block_idx_pool[counter]
            counter += 1

    out_full = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        out_full,
        context_lens,
        block_table,
        max_model_len,
        kv_block_size=64,
    )

    out_qk = torch.full(
        (heads, batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits_stage1(
        q_fp8,
        kv_fp8,
        weights,
        out_qk,
        context_lens,
        block_table,
        max_model_len,
        kv_block_size=64,
        chunk_q=heads,
    )
    out_stage1 = out_qk.sum(dim=0)

    positions = torch.arange(max_model_len, device=DEVICE).unsqueeze(0)
    row_to_batch = torch.arange(batch_size * next_n, device=DEVICE) // next_n
    row_to_n = torch.arange(batch_size * next_n, device=DEVICE) % next_n
    valid_mask = positions <= (
        context_lens[row_to_batch] - next_n + row_to_n
    ).unsqueeze(1)

    # Compare only where BOTH are finite (sum(dim=0) over -inf gives -inf, and
    # -inf - (-inf) = nan would pollute assert_close).
    both = valid_mask & torch.isfinite(out_stage1) & torch.isfinite(out_full)
    assert both.any(), "no comparable positions"
    torch.testing.assert_close(
        out_stage1[both], out_full[both], rtol=5e-2, atol=5e-2
    )


@GPU_REQUIRED
@pytest.mark.parametrize("heads", [1, 8, 64])
def test_triton_fp8_paged_mqa_logits_stage1_default_chunk_q(heads):
    """Default ``chunk_q`` must work for small heads and the heads=64 path."""
    batch_size, next_n, hidden_dim = 2, 1, 128
    max_model_len, ctx_len = 256, 200
    q_fp8 = torch.randn(
        batch_size, next_n, heads, hidden_dim, device=DEVICE, dtype=torch.bfloat16
    ).to(e4m3_type)
    kv = torch.randn((max_model_len, 1, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16)
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE)
    context_lens = torch.tensor([ctx_len] * batch_size, device=DEVICE, dtype=torch.int32)
    kv_indices = torch.zeros((batch_size, max_model_len), device=DEVICE, dtype=torch.int32)
    for b in range(batch_size):
        kv_indices[b, :ctx_len] = torch.arange(ctx_len, device=DEVICE, dtype=torch.int32)

    out_qk = torch.full(
        (heads, batch_size * next_n, max_model_len),
        float("-inf"),
        device=DEVICE,
        dtype=torch.float32,
    )
    triton_fp8_paged_mqa_logits_stage1(
        q_fp8, kv_fp8, weights, out_qk, context_lens, kv_indices, max_model_len,
    )
    ref = ref_fp8_paged_mqa_logits(
        q_fp8, kv_fp8, weights, context_lens, kv_indices, max_model_len
    )
    out = out_qk.sum(dim=0)
    valid = torch.arange(max_model_len, device=DEVICE) <= (ctx_len - next_n)
    torch.testing.assert_close(out[0, valid], ref[0, valid], rtol=5e-2, atol=5e-2)


@GPU_REQUIRED
def test_triton_fp8_paged_mqa_logits_rejects_context_lens_batch_mismatch():
    """``context_lens`` must have one entry per batch row."""
    batch_size, next_n, heads, hidden_dim = 2, 1, 8, 128
    max_model_len = 256
    q_fp8 = torch.randn(
        batch_size, next_n, heads, hidden_dim, device=DEVICE, dtype=torch.bfloat16
    ).to(e4m3_type)
    kv = torch.randn((max_model_len, 1, 1, hidden_dim), device=DEVICE, dtype=torch.bfloat16)
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE)
    context_lens = torch.tensor([100], device=DEVICE, dtype=torch.int32)  # 1 != 2
    kv_indices = torch.zeros((batch_size, max_model_len), device=DEVICE, dtype=torch.int32)
    out_logits = torch.full(
        (batch_size * next_n, max_model_len), float("-inf"), device=DEVICE,
        dtype=torch.float32,
    )

    with pytest.raises(AssertionError, match="context_lens must have batch size"):
        triton_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, weights, out_logits, context_lens, kv_indices,
            max_model_len, kv_block_size=1,
        )


@GPU_REQUIRED
@pytest.mark.parametrize("kv_block_size", [1, 64])
def test_triton_prepare_paged_kv_views_layout_and_strides(kv_block_size):
    """The packed-cache split must expose the values and scales with correct strides.

    The cache row is ``block_size * D`` fp8 bytes followed by ``block_size`` fp32
    scales. A wrong in-page stride silently reads scale bytes as KV values, so pin
    the layout: values at a page pitch of ``block_size * (D+4)`` bytes, scales as a
    contiguous run per page.
    """
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import _prepare_paged_kv_views

    num_blocks, hidden_dim = 4, 128
    index_dim = hidden_dim + 4
    kv = torch.zeros((num_blocks, kv_block_size, 1, index_dim), dtype=torch.uint8)

    kv_fp8, kv_scale = _prepare_paged_kv_views(
        kv, kv_block_size, hidden_dim, e4m3_type
    )

    row_bytes = kv_block_size * index_dim
    assert kv_fp8.shape[-1] == hidden_dim
    assert kv_fp8.stride(-1) == 1
    assert kv_fp8.stride(0) == row_bytes

    assert kv_scale.shape == (num_blocks, kv_block_size)
    assert kv_scale.stride(0) == row_bytes // 4
    assert kv_scale.stride(1) == 1

    if kv_block_size == 1:
        # Per-token path indexes with stride(0) only, so both stay 2D.
        assert kv_fp8.dim() == 2 and kv_scale.dim() == 2
    else:
        # Block-table path also uses stride(1) to walk tokens inside a page.
        assert kv_fp8.stride(-2) == hidden_dim

    # Values and scales must not alias: writing a scale must not touch a value.
    before = kv_fp8[0].clone()
    kv_scale[0, 0] = 3.5
    assert kv_scale[0, 0].item() == pytest.approx(3.5)
    assert torch.equal(kv_fp8[0], before), "scale write leaked into the value view"

    # And the reverse: a value write must not disturb the scales.
    scale_before = kv_scale.clone()
    kv_fp8[0, 0] = kv_fp8[0, 0]
    assert torch.equal(kv_scale, scale_before)


@GPU_REQUIRED
def test_triton_prepare_paged_kv_views_rejects_non_byte_cache():
    """A wider element type can't be byte-addressed; reject instead of mis-viewing."""
    from boltops.mqa_logits.triton.fp8_paged_mqa_logits import _prepare_paged_kv_views

    hidden_dim = 128
    kv = torch.zeros((2, 64, 1, hidden_dim + 4), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="byte-addressed"):
        _prepare_paged_kv_views(kv, 64, hidden_dim, e4m3_type)


@pytest.mark.parametrize("bad_block_size", [32])
@GPU_REQUIRED
def test_triton_fp8_paged_mqa_logits_rejects_unsupported_kv_block_size(bad_block_size):
    """Only 1 (per-token) and 64 (block table) are supported page sizes."""
    batch_size, next_n, heads, hidden_dim = 2, 1, 8, 128
    max_model_len = 256
    q_fp8 = torch.randn(
        batch_size, next_n, heads, hidden_dim, device=DEVICE, dtype=torch.bfloat16
    ).to(e4m3_type)
    # Page size follows ``bad_block_size`` so the guard, not the shape check, fires.
    kv = torch.randn(
        (_cdiv(max_model_len, bad_block_size), bad_block_size, 1, hidden_dim),
        device=DEVICE, dtype=torch.bfloat16,
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE)
    context_lens = torch.tensor([100, 100], device=DEVICE, dtype=torch.int32)
    kv_indices = torch.zeros(
        (batch_size, _cdiv(max_model_len, bad_block_size)), device=DEVICE, dtype=torch.int32
    )
    out_logits = torch.full(
        (batch_size * next_n, max_model_len), float("-inf"), device=DEVICE,
        dtype=torch.float32,
    )

    with pytest.raises(AssertionError, match="supports KVBlockSize 1 and 64"):
        triton_fp8_paged_mqa_logits(
            q_fp8, kv_fp8, weights, out_logits, context_lens, kv_indices,
            max_model_len, kv_block_size=bad_block_size,
        )


@pytest.mark.parametrize("bad_block_size", [32])
@GPU_REQUIRED
def test_triton_fp8_paged_mqa_logits_stage1_rejects_unsupported_kv_block_size(bad_block_size):
    """stage1 must reject unsupported page sizes up front, like the main entry."""
    batch_size, next_n, heads, hidden_dim = 2, 1, 8, 128
    max_model_len = 256
    q_fp8 = torch.randn(
        batch_size, next_n, heads, hidden_dim, device=DEVICE, dtype=torch.bfloat16
    ).to(e4m3_type)
    kv = torch.randn(
        (_cdiv(max_model_len, bad_block_size), bad_block_size, 1, hidden_dim),
        device=DEVICE, dtype=torch.bfloat16,
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn((batch_size * next_n, heads), device=DEVICE)
    context_lens = torch.tensor([100, 100], device=DEVICE, dtype=torch.int32)
    kv_indices = torch.zeros(
        (batch_size, _cdiv(max_model_len, bad_block_size)), device=DEVICE, dtype=torch.int32
    )
    out_qk = torch.full(
        (heads, batch_size * next_n, max_model_len), float("-inf"), device=DEVICE,
        dtype=torch.float32,
    )

    with pytest.raises(AssertionError, match="supports KVBlockSize 1 and 64"):
        triton_fp8_paged_mqa_logits_stage1(
            q_fp8, kv_fp8, weights, out_qk, context_lens, kv_indices,
            max_model_len, kv_block_size=bad_block_size,
        )


# ---------------------------------------------------------------------------
# Output contract (minimal): default path does not write the tail; fuse fills
# the whole row; unsupported fuse paths must raise (not silently ignore).
# Outside-window -inf is checked in the main correctness tests above.
# ---------------------------------------------------------------------------

NEG_INF = float("-inf")
SENTINEL = 12345.0  # never a legitimate logit; survives only if the kernel skips
KV_BLOCK_SIZE = 64

# Non-multiple of chunk_k + mixed per-row ctx (catches max_ctx boundary bugs).
CONTRACT_SHAPES = [
    (2, 1, 64, 128, 4096, [2049, 4095]),
    (4, 2, 64, 128, 8192, [211, 777, 3991, 8191]),
]
CONTRACT_SHAPE_IDS = [
    f"b{b}_n{n}_h{h}_ctx{'-'.join(map(str, ctxs))}"
    for b, n, h, _, _, ctxs in CONTRACT_SHAPES
]


def _contract_run(
    q, kv, weights, context_lens, block_table, max_model_len, fill, *,
    split_kv=None, fuse=False,
):
    rows = q.size(0) * q.size(1)
    out = torch.full((rows, max_model_len), fill, device=DEVICE, dtype=torch.float32)
    info = triton_fp8_paged_mqa_logits_prepare(q, kv, context_lens, block_table)
    triton_fp8_paged_mqa_logits(
        q, kv, weights, out, context_lens, block_table, max_model_len,
        kv_block_size=KV_BLOCK_SIZE,
        batch_info=info,
        split_kv=split_kv,
        fuse_output_init=fuse,
    )
    return out


def _row_context_lens(context_lens_list, next_n):
    return [int(ctx) for ctx in context_lens_list for _ in range(next_n)]


def _written_end(ctx_len, max_model_len):
    end = -(-int(ctx_len) // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    return min(end, max_model_len)


@GPU_REQUIRED
@pytest.mark.parametrize("shape", CONTRACT_SHAPES, ids=CONTRACT_SHAPE_IDS)
def test_triton_paged_past_the_last_tile_is_left_untouched(shape):
    """Default SplitKV must not touch past ceil(ctx/chunk_k)*chunk_k."""
    batch, next_n, heads, hidden, max_model_len, ctxs = shape
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=1
    )

    untouched = _contract_run(
        q, kv, weights, context_lens, block_table, max_model_len, SENTINEL
    )
    cleaned = _contract_run(
        q, kv, weights, context_lens, block_table, max_model_len, NEG_INF
    )

    for row, ctx_len in enumerate(_row_context_lens(ctxs, next_n)):
        end = _written_end(ctx_len, max_model_len)
        assert torch.all(untouched[row, end:] == SENTINEL), (
            f"row {row} (ctx={ctx_len}): wrote past {end}"
        )
        if end:
            assert torch.equal(untouched[row, :end], cleaned[row, :end])


@GPU_REQUIRED
@pytest.mark.parametrize("split_kv", [1, 8])
def test_triton_paged_fuse_output_init_writes_the_whole_row_from_a_poisoned_buffer(split_kv):
    """fuse_output_init must fill every column (NaN poison; SplitKV=8 stresses stride)."""
    batch, next_n, heads, hidden, max_model_len, ctxs = (
        2, 1, 64, 128, 8192, [2049, 4095],
    )
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=3
    )

    fused = _contract_run(
        q, kv, weights, context_lens, block_table, max_model_len, float("nan"),
        split_kv=split_kv, fuse=True,
    )
    assert not torch.isnan(fused).any()
    valid = _valid_logits_mask(context_lens, batch, next_n, max_model_len)
    assert torch.isneginf(fused[~valid]).all()
    ref = ref_fp8_paged_mqa_logits_block64(
        q, kv, weights, context_lens, block_table, max_model_len
    )
    torch.testing.assert_close(fused[valid], ref[valid], rtol=5e-2, atol=5e-2)

    primed = _contract_run(
        q, kv, weights, context_lens, block_table, max_model_len, NEG_INF,
        split_kv=split_kv, fuse=False,
    )
    assert torch.equal(fused, primed)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "mode",
    ["kv_block_size_1", "use_bucket", "bucket_schedule"],
    ids=["block1", "use_bucket", "bucket_schedule"],
)
def test_triton_paged_fuse_output_init_is_rejected_where_the_tail_cannot_be_filled(mode):
    """Paths without a tail fill must raise, not ignore fuse_output_init."""
    batch, next_n, heads, hidden, max_model_len, ctxs = (
        2, 1, 64, 128, 8192, [2048, 4096],
    )
    q, kv, weights, context_lens, block_table, out = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=4
    )
    info = triton_fp8_paged_mqa_logits_prepare(q, kv, context_lens, block_table)
    kwargs = dict(
        context_lens=context_lens,
        kv_indices=block_table,
        max_model_len=max_model_len,
        batch_info=info,
        fuse_output_init=True,
    )
    if mode == "kv_block_size_1":
        kwargs["kv_block_size"] = 1
    elif mode == "use_bucket":
        kwargs["kv_block_size"] = KV_BLOCK_SIZE
        kwargs["use_bucket"] = True
    else:
        kwargs["kv_block_size"] = KV_BLOCK_SIZE
        kwargs["bucket_schedule"] = triton_fp8_paged_mqa_logits_build_bucket_schedule(
            context_lens, next_n=next_n, bucket_tokens=2048
        )

    with pytest.raises(NotImplementedError, match="fuse_output_init"):
        triton_fp8_paged_mqa_logits(q, kv, weights, out, **kwargs)


@GPU_REQUIRED
def test_triton_paged_kv_block_size_1_leaves_the_tail_untouched():
    """Per-token path has no tail loop; large max_model_len - ctx must stay poison."""
    batch, next_n, heads, hidden, max_model_len, ctxs = (
        1, 1, 64, 128, 40960, [8192],
    )
    torch.manual_seed(5)
    q = torch.randn(
        (batch, next_n, heads, hidden), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(
        torch.randn((max_model_len, 1, 1, hidden), device=DEVICE, dtype=torch.bfloat16)
    )
    weights = torch.randn(
        (batch * next_n, heads), device=DEVICE, dtype=torch.float32
    )
    context_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    kv_indices = torch.zeros(
        (batch, max_model_len), device=DEVICE, dtype=torch.int32
    )
    kv_indices[0, :ctxs[0]] = torch.randperm(max_model_len, device=DEVICE)[:ctxs[0]]
    out = torch.full(
        (batch * next_n, max_model_len), float("nan"), device=DEVICE, dtype=torch.float32
    )

    triton_fp8_paged_mqa_logits(
        q, kv_fp8, weights, out, context_lens, kv_indices, max_model_len,
        kv_block_size=1,
    )
    end = _written_end(ctxs[0], max_model_len)
    assert torch.isnan(out[0, end:]).all()
    assert not torch.isnan(out[0, :end]).any()
    assert int(torch.isnan(out).sum()) == max_model_len - end


# ---------------------------------------------------------------------------
# Perf: CUDA-graph pure-GPU latency (pytest --bench or __main__ CLI)
# ---------------------------------------------------------------------------

BENCH_MAX_MODEL_LEN = 111_000
BENCH_BLOCK = 64
BENCH_HEAD_DIM = 128

# (batch, next_n, heads, avg_kv) -- hipc paged test grid subset
BENCH_SHAPES = [
    (1, 1, 64, 4096),
    (64, 1, 64, 4096),
    (64, 1, 64, 72000),
    (128, 1, 64, 4096),
    (64, 2, 64, 4096),
    (64, 1, 32, 4096),
    (128, 1, 32, 4096),
    (64, 1, 32, 72000),
]

BENCH_PATHS = ("splitkv_fuse", "splitkv_nofuse", "bucket_persistent")


def _bench_build_inputs(batch: int, next_n: int, heads: int, avg_kv: int, seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    num_blocks = max(batch * _cdiv(int(1.1 * avg_kv) + BENCH_BLOCK, BENCH_BLOCK) + 8, 256)
    q = torch.randn(
        (batch, next_n, heads, BENCH_HEAD_DIM), device=DEVICE, dtype=torch.bfloat16
    ).to(e4m3_type)
    kv = kv_cache_cast_to_fp8(
        torch.randn(
            (num_blocks, BENCH_BLOCK, 1, BENCH_HEAD_DIM),
            device=DEVICE,
            dtype=torch.bfloat16,
        )
    )
    weights = torch.randn(
        (batch * next_n, heads), device=DEVICE, dtype=torch.float32
    )
    lo = max(int(0.9 * avg_kv), next_n)
    hi = int(1.1 * avg_kv)
    context_lens = torch.randint(
        lo, hi + 1, (batch,), device=DEVICE, dtype=torch.int32
    )
    max_blk = _cdiv(int(context_lens.max().item()), BENCH_BLOCK)
    # hipc pads the second dim to a token-aligned width; only the first
    # ceil(ctx/64) entries are used as block indices.
    block_table = torch.zeros(
        (batch, max_blk * BENCH_BLOCK), device=DEVICE, dtype=torch.int32
    )
    pool = list(range(num_blocks))
    random.shuffle(pool)
    c = 0
    for i in range(batch):
        for j in range(_cdiv(int(context_lens[i]), BENCH_BLOCK)):
            block_table[i, j] = pool[c]
            c += 1
    return q, kv, weights, context_lens, block_table


def _bench_graph_us(fn: Callable[[], None], *, warm: int = 5, reps: int = 30) -> float:
    """Best-of CUDA-graph replay time in microseconds (pure GPU)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warm):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[0]


def _bench_make_bolt_fn(
    path: str,
    q,
    kv,
    weights,
    out,
    context_lens,
    block_table,
    info,
    bucket_schedule=None,
):
    if path == "splitkv_fuse":

        def fn():
            triton_fp8_paged_mqa_logits(
                q,
                kv,
                weights,
                out,
                context_lens,
                block_table,
                BENCH_MAX_MODEL_LEN,
                kv_block_size=64,
                batch_info=info,
                fuse_output_init=True,
            )

        return fn

    if path == "splitkv_nofuse":
        # Prefill -inf outside the timed region for compute-only graph.
        out.fill_(float("-inf"))
        torch.cuda.synchronize()

        def fn():
            triton_fp8_paged_mqa_logits(
                q,
                kv,
                weights,
                out,
                context_lens,
                block_table,
                BENCH_MAX_MODEL_LEN,
                kv_block_size=64,
                batch_info=info,
                fuse_output_init=False,
            )

        return fn

    if path == "bucket_persistent":
        assert bucket_schedule is not None

        def fn():
            triton_fp8_paged_mqa_logits(
                q,
                kv,
                weights,
                out,
                context_lens,
                block_table,
                BENCH_MAX_MODEL_LEN,
                kv_block_size=64,
                batch_info=info,
                bucket_schedule=bucket_schedule,
            )

        return fn

    raise ValueError(f"unknown path {path!r}; expected one of {BENCH_PATHS}")


def _try_import_hipc():
    try:
        import lightop as hipc  # package install name

        return hipc.gemmopt
    except Exception:
        return None


def run_paged_bench(
    *,
    shapes=None,
    paths=("splitkv_fuse",),
    vs_hipc: bool = False,
    warm: int = 5,
    reps: int = 30,
) -> list:
    shapes = list(BENCH_SHAPES if shapes is None else shapes)
    gemmopt = _try_import_hipc() if vs_hipc else None
    if vs_hipc and gemmopt is None:
        raise RuntimeError("--vs-hipc requested but hipc is not importable")

    rows = []
    hdr = f"{'shape':36s} {'path':18s} {'bolt_us':>10}"
    if vs_hipc:
        hdr += f" {'hipc_us':>10} {'hipc/bolt':>9}"
    print(
        f"device={torch.cuda.get_device_name(0)}  max_model_len={BENCH_MAX_MODEL_LEN}  "
        f"timing=CUDA-graph replay (pure GPU, best of {reps})",
        flush=True,
    )
    print(hdr, flush=True)

    for batch, next_n, heads, avg_kv in shapes:
        tag = f"B={batch} n={next_n} H={heads} L={avg_kv}"
        q, kv, weights, context_lens, block_table = _bench_build_inputs(
            batch, next_n, heads, avg_kv
        )
        info = triton_fp8_paged_mqa_logits_prepare(q, kv, context_lens, block_table)
        out = torch.empty(
            (batch * next_n, BENCH_MAX_MODEL_LEN), device=DEVICE, dtype=torch.float32
        )

        sch = None
        if "bucket_persistent" in paths:
            base = torch.empty(batch * next_n + 1, dtype=torch.int32, device=DEVICE)
            sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
                context_lens,
                batch_size=batch,
                next_n=next_n,
                base=base,
                max_ctx=info.max_ctx,
            )

        hipc_us = None
        if vs_hipc:

            def hipc_fn():
                gemmopt.paged_mqa_logits(
                    q,
                    kv,
                    weights,
                    context_lens,
                    block_table,
                    None,
                    BENCH_MAX_MODEL_LEN,
                    clean_logits=True,
                )

            hipc_us = _bench_graph_us(hipc_fn, warm=warm, reps=reps)

        for path in paths:
            try:
                bolt_fn = _bench_make_bolt_fn(
                    path,
                    q,
                    kv,
                    weights,
                    out,
                    context_lens,
                    block_table,
                    info,
                    bucket_schedule=sch,
                )
                bolt_us = _bench_graph_us(bolt_fn, warm=warm, reps=reps)
                row = {
                    "shape": tag,
                    "path": path,
                    "bolt_us": bolt_us,
                    "hipc_us": hipc_us,
                }
                if vs_hipc and hipc_us is not None:
                    ratio = hipc_us / bolt_us if bolt_us > 0 else float("inf")
                    row["ratio"] = ratio
                    print(
                        f"{tag:36s} {path:18s} {bolt_us:10.1f} "
                        f"{hipc_us:10.1f} {ratio:8.2f}x",
                        flush=True,
                    )
                else:
                    print(f"{tag:36s} {path:18s} {bolt_us:10.1f}", flush=True)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{tag:36s} {path:18s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

        del q, kv, weights, context_lens, block_table, out, info, sch
        torch.cuda.empty_cache()

    return rows


def _bench_shape_params():
    for shape in BENCH_SHAPES:
        batch, next_n, heads, avg_kv = shape
        yield pytest.param(
            shape,
            id=f"B{batch}-n{next_n}-H{heads}-L{avg_kv}",
        )


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape", list(_bench_shape_params()))
@torch.inference_mode()
def test_bench_triton_fp8_paged_mqa_logits_gpu(shape):
    """Pure-GPU (graph) latency of the default SplitKV+fuse path."""
    rows = run_paged_bench(shapes=[shape], paths=("splitkv_fuse",), vs_hipc=False)
    assert rows and rows[0]["bolt_us"] > 0


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape", list(_bench_shape_params()))
@torch.inference_mode()
def test_bench_triton_fp8_paged_mqa_logits_vs_hipc(shape):
    """Pure-GPU BoltOPs vs hipc; skipped when hipc is missing."""
    if _try_import_hipc() is None:
        pytest.skip("hipc not installed")
    rows = run_paged_bench(shapes=[shape], paths=("splitkv_fuse",), vs_hipc=True)
    assert rows and rows[0]["bolt_us"] > 0 and rows[0]["hipc_us"] > 0


def _bench_main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="FP8 paged MQA logits CUDA-graph GPU latency"
    )
    ap.add_argument(
        "--vs-hipc",
        action="store_true",
        help="also time hipc paged_mqa_logits (same graph-replay method)",
    )
    ap.add_argument(
        "--path",
        nargs="+",
        choices=BENCH_PATHS,
        default=["splitkv_fuse"],
        help="BoltOPs branches to time (default: splitkv_fuse only)",
    )
    ap.add_argument(
        "--shapes",
        type=int,
        nargs="*",
        default=None,
        help="indices into BENCH_SHAPES (default: all)",
    )
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args(argv)

    idx = args.shapes if args.shapes is not None else list(range(len(BENCH_SHAPES)))
    shapes = [BENCH_SHAPES[i] for i in idx]
    run_paged_bench(
        shapes=shapes,
        paths=tuple(args.path),
        vs_hipc=args.vs_hipc,
        warm=args.warm,
        reps=args.reps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_bench_main())
