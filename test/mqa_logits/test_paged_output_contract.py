"""The output contract of the paged FP8 MQA logits op.

Two properties of ``out_logits`` that the signature does not express, and that
any drop-in replacement has to honour:

1. **Where it writes.** Each row is only written over
   ``[0, ceil(ctx / chunk_k) * chunk_k)`` clamped to ``max_model_len``. Callers
   size the buffer once for the whole decode, so ``max_model_len`` is normally
   far above the context a step actually attends to: the rest of the row is left
   *untouched* -- not zeroed, not filled. A caller that reuses its output buffer
   must therefore pre-fill ``-inf`` itself. aiter's own
   ``test_pa_mqa_logits_offset.py`` pins the same property from the other side
   (it primes the buffer with a sentinel and asserts the sentinel survives),
   which is why the two ops can stand in for each other.

2. **What is inside it.** Within the written region, positions outside the causal
   window are exactly ``-inf``, not merely very negative.

The sentinel test is the one that actually constrains the boundary: asserting
"the tail contains -inf" would also pass if the op cleaned the whole row, i.e.
if it paid for the memset this contract says it does not pay for.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.mqa_logits import (  # noqa: E402
    triton_fp8_paged_mqa_logits,
    triton_fp8_paged_mqa_logits_prepare,
)
from test_fp8_paged_mqa_logits import (  # noqa: E402
    _build_block64_inputs,
    _valid_logits_mask,
    kv_cache_cast_to_fp8,
    ref_fp8_paged_mqa_logits_block64,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
NEG_INF = float("-inf")
# Deliberately not -inf / 0 / nan: a tail left at the sentinel is only
# detectable if the value is one the kernel would never compute by itself.
SENTINEL = 12345.0
# ``_build_block64_inputs`` builds the block-table path, whose KV pages are 64
# tokens wide.
KV_BLOCK_SIZE = 64
# ``_build_block64_inputs`` always makes batch*next_n rows in batch-major order.
SHAPES = [
    # ctx == max_model_len: the write region is the whole row, so the boundary
    # logic has to reduce to "no tail".
    (1, 1, 64, 128, 4096, [4096]),
    # ctx is an exact multiple of chunk_k: the last tile lands exactly on the
    # boundary, clamped to nothing.
    (1, 1, 64, 128, 4096, [3072]),
    # ctx is not a multiple of chunk_k: the final tile is written only in part,
    # so the boundary sits past anything the row can legitimately produce.
    (2, 1, 64, 128, 4096, [2049, 4095]),
    # Mixed rows: per-row boundaries differ by ~40x, which is what catches a
    # boundary computed from max_ctx instead of each row's own ctx.
    (4, 2, 64, 128, 8192, [211, 777, 3991, 8191]),
    # heads=1 takes the non-MMA lowering path, with a large tail.
    (1, 2, 1, 128, 8192, [4095]),
]
SHAPE_IDS = [f"b{b}_n{n}_h{h}_ctx{'-'.join(map(str, ctxs))}" for b, n, h, _, _, ctxs in SHAPES]


def _run(q, kv, weights, context_lens, block_table, max_model_len, fill, *,
         split_kv=None, fuse=False):
    """Call the op on a buffer primed with ``fill`` and return that buffer."""
    rows = q.size(0) * q.size(1)
    out = torch.full((rows, max_model_len), fill, device=DEVICE, dtype=torch.float32)
    info = triton_fp8_paged_mqa_logits_prepare(
        q, kv, context_lens, block_table
    )
    triton_fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        out,
        context_lens,
        block_table,
        max_model_len,
        kv_block_size=KV_BLOCK_SIZE,
        batch_info=info,
        split_kv=split_kv,
        fuse_output_init=fuse,
    )
    return out


def _row_context_lens(context_lens_list, next_n):
    """Flatten ``context_lens`` into one entry per output row (batch-major)."""
    return [int(ctx) for ctx in context_lens_list for _ in range(next_n)]


def _written_end(ctx_len, max_model_len):
    """First column the kernel never writes, for a row attending to ``ctx_len``."""
    end = -(-int(ctx_len) // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    return min(end, max_model_len)


@GPU_REQUIRED
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_matches_reference_and_is_exactly_neg_inf_outside_the_window(shape):
    batch, next_n, heads, hidden, max_model_len, ctxs = shape
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=0
    )

    out = _run(q, kv, weights, context_lens, block_table, max_model_len, NEG_INF)
    ref = ref_fp8_paged_mqa_logits_block64(
        q, kv, weights, context_lens, block_table, max_model_len
    )

    valid = _valid_logits_mask(context_lens, batch, next_n, max_model_len)
    # Downstream softmax relies on padded positions contributing exactly zero, so
    # they must be -inf rather than merely very negative.
    assert torch.isneginf(out[~valid]).all()
    torch.testing.assert_close(out[valid], ref[valid], rtol=5e-2, atol=5e-2)


@GPU_REQUIRED
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_past_the_last_tile_is_left_untouched(shape):
    batch, next_n, heads, hidden, max_model_len, ctxs = shape
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=1
    )

    untouched = _run(q, kv, weights, context_lens, block_table, max_model_len, SENTINEL)
    cleaned = _run(q, kv, weights, context_lens, block_table, max_model_len, NEG_INF)

    row_ctxs = _row_context_lens(ctxs, next_n)
    assert len(row_ctxs) == untouched.size(0)

    for row, ctx_len in enumerate(row_ctxs):
        end = _written_end(ctx_len, max_model_len)
        assert torch.all(untouched[row, end:] == SENTINEL), (
            f"row {row} (ctx={ctx_len}): the kernel wrote past "
            f"ceil(ctx/chunk_k)*chunk_k={end}, so it is doing work the caller "
            "should not have to pay for"
        )
        if end == 0:
            continue
        # Everything up to the boundary is overwritten, and overwritten with the
        # same values as when the whole row was primed with -inf: the result must
        # not depend on what the buffer held.
        assert torch.equal(untouched[row, :end], cleaned[row, :end])


# ``fuse_output_init=True`` is the opt-in that lifts the pre-fill requirement:
# the kernel writes the whole row, tail included, so the caller can drop its
# ``torch.full``. Shapes here pick up the boundary cases the tail loop can get
# wrong: an empty tail (``ctx == max_model_len``), a ``ctx`` that is neither a
# multiple of ``chunk_k`` nor of the loop's ``TailBlock`` stride, and a batch
# whose rows differ by ~2x so a fill keyed off the wrong row shows up.
FUSE_SHAPES = [
    (2, 1, 64, 128, 8192, [2048, 4096]),
    (1, 1, 64, 128, 4096, [4096]),
    (2, 1, 64, 128, 8192, [2049, 4095]),
]
FUSE_SHAPE_IDS = [
    f"b{b}_n{n}_h{h}_ctx{'-'.join(map(str, ctxs))}"
    for b, n, h, _, _, ctxs in FUSE_SHAPES
]


@GPU_REQUIRED
@pytest.mark.parametrize("split_kv", [1, 8])
@pytest.mark.parametrize("shape", FUSE_SHAPES, ids=FUSE_SHAPE_IDS)
def test_fuse_output_init_writes_the_whole_row_from_a_poisoned_buffer(shape, split_kv):
    """The fused fill has to own every column the caller stopped priming.

    The probe is a NaN poison, not a ``-inf`` prime: -inf is the value the
    kernel is *supposed* to store, so priming with it would let a missing store
    pass as a correct one. With NaN, any position the fill fails to reach is
    still NaN at the end. SplitKV 8 spreads the tail across grid programs
    (``pid_split_kv * TailBlock`` strided by ``SplitKV * TailBlock``), which is
    where an off-by-one in the stride would drop a whole block of the tail.
    """
    batch, next_n, heads, hidden, max_model_len, ctxs = shape
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=3
    )

    fused = _run(
        q, kv, weights, context_lens, block_table, max_model_len, float("nan"),
        split_kv=split_kv, fuse=True,
    )

    assert not torch.isnan(fused).any(), (
        "fuse_output_init=True left NaN in the row, so at least one column it "
        "owns was never written; the caller dropped its torch.full for nothing"
    )
    # Fusing does not relax the contract: outside the causal window it is still
    # exactly -inf, and inside it the values still match the reference.
    valid = _valid_logits_mask(context_lens, batch, next_n, max_model_len)
    assert torch.isneginf(fused[~valid]).all()

    ref = ref_fp8_paged_mqa_logits_block64(
        q, kv, weights, context_lens, block_table, max_model_len
    )
    torch.testing.assert_close(fused[valid], ref[valid], rtol=5e-2, atol=5e-2)

    # And opting in must not perturb the numbers: same split, same result as
    # priming the buffer by hand, bit for bit. A separate ``torch.full`` costs
    # time, not accuracy, so the two have to agree exactly.
    primed = _run(
        q, kv, weights, context_lens, block_table, max_model_len, NEG_INF,
        split_kv=split_kv, fuse=False,
    )
    assert torch.equal(fused, primed), (
        "fused fill and a hand-primed buffer disagree; fusing is only a way to "
        "skip the caller's torch.full, it must not change the logits"
    )


@GPU_REQUIRED
def test_fuse_output_init_is_rejected_where_the_tail_cannot_be_filled():
    """A path that cannot fill the tail must refuse, not ignore the request.

    Only the SplitKV block path implements the fill. The other paths would
    otherwise return a row whose tail still holds whatever the caller left
    there, and the consumer reads any non--inf value as a real top-k index -- so
    silence here is a wrong answer, not a missed optimisation.
    """
    batch, next_n, heads, hidden, max_model_len, ctxs = (
        2, 1, 64, 128, 8192, [2048, 4096],
    )
    q, kv, weights, context_lens, block_table, out = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=4
    )
    info = triton_fp8_paged_mqa_logits_prepare(
        q, kv, context_lens, block_table
    )
    call = dict(
        context_lens=context_lens,
        kv_indices=block_table,
        max_model_len=max_model_len,
        batch_info=info,
        fuse_output_init=True,
    )

    # kv_block_size=1: the per-token kernel writes only [0, context_lens), so it
    # has no tail loop and the request is unimplementable there.
    with pytest.raises(NotImplementedError, match="fuse_output_init"):
        triton_fp8_paged_mqa_logits(
            q, kv, weights, out, kv_block_size=1, **call
        )

    # Eager bucket scheduler.
    with pytest.raises(NotImplementedError, match="fuse_output_init"):
        triton_fp8_paged_mqa_logits(
            q, kv, weights, out, kv_block_size=KV_BLOCK_SIZE, use_bucket=True, **call
        )

    # Device-resident bucket schedule. This one is a ``PagedBucketSchedule``
    # object (the host ``build_paged_bucket_schedule`` above is only the raw
    # tensor, which is what the bucket *kernel* consumes, not this entry point).
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

    schedule = triton_fp8_paged_mqa_logits_build_bucket_schedule(
        context_lens, next_n=next_n, bucket_tokens=2048
    )
    with pytest.raises(NotImplementedError, match="fuse_output_init"):
        triton_fp8_paged_mqa_logits(
            q, kv, weights, out, kv_block_size=KV_BLOCK_SIZE,
            bucket_schedule=schedule, **call
        )


def _build_block1_inputs(batch_size, next_n, heads, hidden, max_model_len, ctxs, seed):
    """Per-token-index (``kv_block_size=1``) inputs, mirroring the block-64 builder.

    One physical cache row per logical token, so ``num_blocks == max_model_len``
    and ``kv_indices[b, t]`` is scattered the same way the block table is.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        (batch_size, next_n, heads, hidden), device=DEVICE, dtype=torch.bfloat16
    )
    kv = torch.randn(
        (max_model_len, 1, 1, hidden), device=DEVICE, dtype=torch.bfloat16
    )
    kv_fp8 = kv_cache_cast_to_fp8(kv)
    weights = torch.randn(
        (batch_size * next_n, heads), device=DEVICE, dtype=torch.float32
    )
    context_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    assert torch.all(context_lens >= next_n)

    kv_indices = torch.zeros(
        (batch_size, max_model_len), device=DEVICE, dtype=torch.int32
    )
    for b in range(batch_size):
        n_tok = int(context_lens[b].item())
        kv_indices[b, :n_tok] = torch.randperm(max_model_len, device=DEVICE)[:n_tok]

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("nan"),
        device=DEVICE,
        dtype=torch.float32,
    )
    return q, kv_fp8, weights, context_lens, kv_indices, out


# The per-token path has **no tail loop at all** -- it writes only
# ``[0, ceil(ctx/chunk_k)*chunk_k)`` -- which is *why* ``fuse_output_init``
# raises there: the op cannot fill the tail, rather than forgetting to. Poisoning
# the buffer pins that difference: everything past the last tile has to come back
# untouched, so the caller still owns the pre-fill. The first shape reproduces
# the ``32768/32768`` tail count the commit message cites: ``max_model_len - ctx``
# is exactly 32768 tail positions, all of which must survive.
BLOCK1_SHAPES = [
    (1, 1, 64, 128, 40960, [8192]),
    (2, 1, 64, 128, 8192, [2049, 4095]),
]
BLOCK1_SHAPE_IDS = [
    f"b{b}_n{n}_h{h}_ctx{'-'.join(map(str, ctxs))}"
    for b, n, h, _, _, ctxs in BLOCK1_SHAPES
]


@GPU_REQUIRED
@pytest.mark.parametrize("shape", BLOCK1_SHAPES, ids=BLOCK1_SHAPE_IDS)
def test_kv_block_size_1_leaves_the_tail_untouched(shape):
    batch, next_n, heads, hidden, max_model_len, ctxs = shape
    q, kv, weights, context_lens, kv_indices, out = _build_block1_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=5
    )

    triton_fp8_paged_mqa_logits(
        q, kv, weights, out, context_lens, kv_indices, max_model_len,
        kv_block_size=1,
    )

    row_ctxs = _row_context_lens(ctxs, next_n)
    for row, ctx_len in enumerate(row_ctxs):
        end = _written_end(ctx_len, max_model_len)
        # Past the last tile: never written, so the NaN survives.
        assert torch.isnan(out[row, end:]).all(), (
            f"row {row} (ctx={ctx_len}): the per-token kernel wrote past "
            f"ceil(ctx/chunk_k)*chunk_k={end}, a region no path here can fill"
        )
        # Up to the boundary: written, so the poison is gone (a value there may
        # be -inf by contract, but it must not still be the caller's NaN).
        assert not torch.isnan(out[row, :end]).any(), (
            f"row {row} (ctx={ctx_len}): the valid window was not fully written"
        )

    # The poison double-counts nothing: every tail position, and only those,
    # survive. Asserting the total keeps the accounting honest rather than
    # checking it one row at a time.
    surviving = int(torch.isnan(out).sum())
    expected = sum(max_model_len - _written_end(c, max_model_len) for c in row_ctxs)
    assert surviving == expected, (
        f"{surviving} poisoned positions survived, expected {expected} "
        "tail positions"
    )

    # And it still refuses to claim it filled the tail.
    with pytest.raises(NotImplementedError, match="fuse_output_init"):
        triton_fp8_paged_mqa_logits(
            q, kv, weights, out, context_lens, kv_indices, max_model_len,
            kv_block_size=1, fuse_output_init=True,
        )


def _reference_or_skip():
    """Return the reference implementation (imported as ``hipc``) or skip.

    The reference package still installs under its old module name; ``hipc`` is
    the display name used throughout this repo, so the old name appears only on
    the ``import`` line below.
    """
    try:
        import lightop as hipc
    except ImportError:  # pragma: no cover - reference implementation absent
        pytest.skip("hipc reference implementation is not installed")
    return hipc


@GPU_REQUIRED
def test_agrees_with_hipc():
    """Same inputs, same logits -- the two ops are drop-in for each other.

    Skipped when the reference implementation (hipc) is not installed.
    ``clean_logits=True`` is what a production caller uses, and it is also the
    only mode comparable with the reference above: with ``clean_logits=False``
    the caller owns the pre-fill, exactly as ``_run`` does here.
    """
    gemmopt = _reference_or_skip().gemmopt

    batch, next_n, heads, hidden, max_model_len, ctxs = (2, 1, 64, 128, 8192, [2048, 4096])
    q, kv, weights, context_lens, block_table, _ = _build_block64_inputs(
        batch, next_n, heads, hidden, max_model_len, ctxs, seed=2
    )

    ours = _run(q, kv, weights, context_lens, block_table, max_model_len, NEG_INF)
    schedule_meta = gemmopt.get_paged_mqa_logits_metadata(
        context_lens,
        KV_BLOCK_SIZE,
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    theirs = gemmopt.paged_mqa_logits(
        q,
        kv,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
        clean_logits=True,
    )

    assert ours.shape == theirs.shape
    # The padded regions must agree exactly: if only one op writes -inf there,
    # then one of them is handing the softmax a score the other is not.
    assert torch.equal(ours == NEG_INF, theirs == NEG_INF)

    valid = _valid_logits_mask(context_lens, batch, next_n, max_model_len)
    a = ours.masked_fill(~valid, 0.0).double()
    b = theirs.masked_fill(~valid, 0.0).double()
    scale = max(b.abs().max().item(), 1e-9)
    diff = (a - b).abs().max().item() / scale
    assert diff < 1e-3, f"relative diff vs hipc = {diff}"
