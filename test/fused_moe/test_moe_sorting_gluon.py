# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU-reference tests for stable Gluon MoE routing and packed padding."""

import pytest
import torch

pytest.importorskip("triton.experimental.gluon")
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from boltops.fused_moe.triton.moe_sorting_gluon import (
    _ready_counts,
    moe_sorting_fwd,
)


def make_case(tokens, experts, topk, unit, masked=False, skewed=False):
    rng = torch.Generator(device="cpu").manual_seed(2401)
    ids = torch.rand((tokens, experts), generator=rng).topk(topk, dim=1).indices.int()
    if skewed:
        ids = (
            torch.arange(topk, dtype=torch.int32)[None, :]
            .expand(tokens, topk)
            .contiguous()
        )
    weights = torch.rand((tokens, topk), generator=rng)
    mask = torch.ones(experts, dtype=torch.int32) if masked else None
    if mask is not None:
        mask[1::3] = 0
    capacity = tokens * topk + experts * (unit - 1)
    capacity = max(capacity, 1)
    outputs = (
        torch.full((capacity,), -1, dtype=torch.int32, device="cpu").cuda(),
        torch.full((capacity,), -1.0, dtype=torch.float32, device="cpu").cuda(),
        torch.full(
            ((capacity + unit - 1) // unit,), -1, dtype=torch.int32, device="cpu"
        ).cuda(),
        torch.full((2 * experts,), -1, dtype=torch.int32, device="cpu").cuda(),
        torch.full((1,), -1, dtype=torch.int32, device="cpu").cuda(),
    )
    args = (
        ids.cuda(),
        weights.cuda(),
        *outputs,
        experts,
        unit,
        mask.cuda() if mask is not None else None,
    )
    return args, (ids, weights, mask)


def check_case(args, inputs, strict_positions=True):
    ids, weights, mask = inputs
    _, _, out, outw, oute, pos, valid, experts, unit, _ = args
    tokens, topk = ids.shape
    expected_ids, expected_weights, expected_experts = [], [], []
    counts, starts = [], []
    local = 0
    for e in range(experts):
        routes = (ids.flatten() == e).nonzero().flatten()
        count = routes.numel()
        counts.append(count)
        enabled = mask is None or bool(mask[e])
        start = len(expected_ids) if enabled and count else -1
        starts.append(start)
        if enabled and count:
            pad = (-count) % unit
            expected_ids.extend(
                ((routes // topk) | ((routes % topk) << 24)).int().tolist()
            )
            packed_pad = (
                torch.tensor(tokens | (topk << 24), dtype=torch.int64).int().item()
            )
            expected_ids.extend([packed_pad] * pad)
            expected_weights.extend(weights.flatten()[routes].tolist())
            expected_weights.extend([0.0] * pad)
            expected_experts.extend(
                [local if mask is not None else e] * ((count + pad) // unit)
            )
        local += enabled
    assert valid.cpu().item() == len(expected_ids)
    size = len(expected_ids)
    assert out[:size].cpu().tolist() == expected_ids
    assert outw[:size].cpu().tolist() == expected_weights
    assert oute[: len(expected_experts)].cpu().tolist() == expected_experts
    actual = pos.cpu().tolist()
    for e in range(experts):
        if strict_positions or mask is None or bool(mask[e]):
            assert actual[e] == counts[e]
        if strict_positions or starts[e] >= 0:
            assert actual[experts + e] == starts[e]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize(
    "tokens,experts,topk,unit",
    [
        (0, 8, 2, 16),
        (1, 8, 2, 16),
        (17, 8, 2, 32),
        (128, 64, 8, 32),
        (512, 64, 8, 32),
        (2048, 64, 8, 32),
        (32, 256, 8, 128),
        (4096, 256, 8, 64),
        (9, 7, 3, 1),
        (513, 13, 3, 17),
        (65, 256, 128, 16),
        (1, 256, 255, 16),
        (257, 17, 9, 32),
        (1025, 13, 3, 17),
        (2049, 8, 1, 128),
        # The 16-case performance matrix adds these shapes to the boundary suite.
        (1, 8, 1, 16),
        (3, 7, 6, 17),
        (16, 64, 8, 32),
        (17, 13, 6, 17),
        (32, 256, 8, 32),
        (129, 63, 1, 17),
        (257, 127, 6, 63),
        (512, 256, 8, 64),
        (1025, 255, 6, 63),
        (2049, 127, 1, 63),
        (4096, 256, 8, 128),
        (4097, 255, 6, 127),
        (8192, 256, 8, 128),
        (8193, 255, 1, 127),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_sorting(tokens, experts, topk, unit, masked):
    args, inputs = make_case(tokens, experts, topk, unit, masked)
    moe_sorting_fwd(*args)
    check_case(args, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize("masked", [False, True])
def test_skewed_experts(masked):
    args, inputs = make_case(513, 64, 4, 17, masked, skewed=True)
    moe_sorting_fwd(*args)
    check_case(args, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize("tokens", [17, 512])
def test_all_experts_disabled(tokens):
    args, inputs = make_case(tokens, 13, 8, 32, masked=True)
    ids, weights, _ = inputs
    mask = torch.zeros(13, dtype=torch.int32, device="cpu")
    args[-1].copy_(mask)
    moe_sorting_fwd(*args)
    check_case(args, (ids, weights, mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize("tokens", [17, 512])
def test_fused_moe_sorting_entry(tokens):
    from boltops.fused_moe.triton.fused_moe import moe_sorting_gluon

    args, inputs = make_case(tokens, 13, 8, 32, masked=True)
    out, outw, oute, valid, positions = moe_sorting_gluon(
        inputs[0].long().cuda(),
        args[1],
        13,
        block_size=32,
        expert_mask=args[-1],
    )
    check_case(
        (args[0], args[1], out, outw, oute, positions, valid, 13, 32, args[-1]),
        inputs,
    )


@g.jit
def _count_prefix_probe(Parts, Pids, Output, NB: gl.constexpr):
    block = gl.program_id(0)
    pid = gl.load(Pids + block)
    counts, prefix = _ready_counts(Parts, pid, 13, 16, NB)
    e = gl.arange(0, 16, gl.BlockedLayout([1], [64], [4], [0]))
    gl.store(Output + block * 32 + e, counts)
    gl.store(Output + block * 32 + 16 + e, prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize("chunks", [63, 64])
def test_completed_count_prefix_with_tails(chunks):
    tokens, experts, topk, block = chunks * 32 - 1, 13, 8, 256
    ids = (torch.arange(tokens * topk, dtype=torch.int32) % experts).reshape(
        tokens, topk
    )
    table = torch.stack(
        [
            torch.bincount(
                ids.flatten()[c * block : (c + 1) * block].long(), minlength=experts
            ).int()
            for c in range(chunks)
        ]
    )
    pids = torch.tensor([0, 31, 63, 64], dtype=torch.int32)
    output = torch.empty((4, 32), dtype=torch.int32, device="cuda")
    _count_prefix_probe[(4,)](table.cuda(), pids.cuda(), output, chunks)
    actual = output.cpu()
    count = torch.bincount(ids.flatten().long(), minlength=experts).int()
    for row, pid in enumerate(pids.tolist()):
        prefix = torch.bincount(
            ids.flatten()[: pid * block].long(), minlength=experts
        ).int()
        assert actual[row, :experts].tolist() == count.tolist()
        assert actual[row, 16 : 16 + experts].tolist() == prefix.tolist()
        assert torch.count_nonzero(actual[row, experts:16]).item() == 0
        assert torch.count_nonzero(actual[row, 16 + experts :]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
def test_sorting_handles_unaligned_pointers():
    # Native Gluon specialization must also handle contiguous offset views.
    for tokens in (17, 513):
        args, inputs = make_case(tokens, 13, 8, 17, masked=True)
        moe_sorting_fwd(*args)
        check_case(args, inputs)
        shifted = list(args)
        for i, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                storage = torch.empty(
                    value.numel() + 1, dtype=value.dtype, device=value.device
                )
                shifted[i] = storage[1:].view(value.shape)
                shifted[i].copy_(value)
        for call in (args, tuple(shifted), args, tuple(shifted)):
            moe_sorting_fwd(*call)
            check_case(call, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU or PMD")
@pytest.mark.parametrize("tokens", [17, 513])
@pytest.mark.parametrize("masked", [False, True])
def test_sorting_preserves_weight_bits(tokens, masked):
    args, _ = make_case(tokens, 13, 6, 17, masked=masked)
    # Include signed zero, subnormals, infinities and a quiet NaN payload.
    patterns = torch.tensor(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
        ],
        dtype=torch.int64,
    ).int()
    bits = patterns.repeat((tokens * 6 + patterns.numel() - 1) // patterns.numel())
    bits = bits[: tokens * 6].reshape(tokens, 6)
    args[1].copy_(bits.view(torch.float32))
    moe_sorting_fwd(*args)
    size = args[6].cpu().item()
    packed = args[2][:size].cpu().long()
    token, slot = packed & 0x00FFFFFF, (packed >> 24) & 255
    real = (token < tokens) & (slot < 6)
    expected = torch.zeros(size, dtype=torch.int32)
    expected[real] = bits[token[real], slot[real]]
    assert torch.equal(args[3][:size].cpu().view(torch.int32), expected)


def _change_routing(args, original, step):
    ids, weights, mask = original
    ids = (ids + step) % args[7]
    weights = weights * (0.25 + step / 8)
    if mask is not None:
        mask = torch.roll(mask, step)
        if step % 3 == 0:
            mask = torch.zeros_like(mask)
        elif step % 3 == 1:
            mask = mask * -2
    args[0].copy_(ids)
    args[1].copy_(weights)
    if mask is not None:
        args[-1].copy_(mask)
    return ids, weights, mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "shape",
    [
        (17, 13, 3, 17),
        (257, 17, 9, 32),
        (512, 64, 8, 32),
        (4096, 256, 8, 128),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_real_graph_replay_with_changed_inputs(shape, masked):
    args, original = make_case(*shape, masked=masked)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe_sorting_fwd(*args)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # Each captured sorter call must own independent count scratch.
        moe_sorting_fwd(*args)
        moe_sorting_fwd(*args)
    for step in range(1, 7):
        with torch.cuda.stream(stream):
            expected = _change_routing(args, original, step)
            graph.replay()
        stream.synchronize()
        check_case(args, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_multistream_independent_workspaces():
    cases = [make_case(513, 13, 8, 17, masked=(i % 2 == 0)) for i in range(12)]
    streams = [torch.cuda.Stream() for _ in cases]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for step in range(1, 5):
        expected = []
        for (args, original), stream in zip(cases, streams):
            with torch.cuda.stream(stream):
                expected.append(_change_routing(args, original, step))
                moe_sorting_fwd(*args)
        for stream in streams:
            stream.synchronize()
        for (args, _), golden in zip(cases, expected):
            check_case(args, golden)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("tokens", [17, 513])
def test_real_wrong_current_device(tokens):
    with torch.cuda.device(1):
        args, golden = make_case(tokens, 13, 8, 32, masked=True)
    with torch.cuda.device(0):
        moe_sorting_fwd(*args)
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(1)
    check_case(args, golden)
