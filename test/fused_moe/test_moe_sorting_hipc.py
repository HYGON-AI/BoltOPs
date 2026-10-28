# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for the private HIPC MoE sorter.

Default ``pytest``: functional correctness / dependency checks.
``pytest --bench``: production-ish token/expert timing.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import statistics

import pytest
import torch


_CSRC = (
    Path(__file__).parents[2]
    / "boltops"
    / "fused_moe"
    / "triton"
    / "moe_sorting_hipc"
    / "csrc"
)


def test_moe_sorting_sources_are_standalone():
    # Upstream provenance in header comments is not a build dependency.
    sources = "\n".join(
        line
        for path in _CSRC.iterdir()
        if path.is_file()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("//")
    )
    assert "ck_tile" not in sources
    assert "composable_kernel" not in sources
    assert "#include <aiter" not in sources
    assert "#include \"aiter" not in sources


@pytest.mark.parametrize(
    "source",
    [
        "#include <ck_tile/core.hpp>",
        '#include "composable_kernel.hpp"',
        "#include <aiter/ops.hpp>",
        '#include "aiter/ops.hpp"',
    ],
)
def test_standalone_source_check_rejects_dependencies(tmp_path, monkeypatch, source):
    (tmp_path / "kernel.h").write_text(
        "// Upstream: composable_kernel/include/ck_tile/core.hpp\n" + source,
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "_CSRC", tmp_path)
    with pytest.raises(AssertionError):
        test_moe_sorting_sources_are_standalone()


def _hipc_available() -> bool:
    return torch.cuda.is_available() and (
        shutil.which("aicc") is not None or Path("/opt/dtk/bin/aicc").is_file()
    )


def _reference(topk_ids, topk_weights, num_experts, unit_size, mask):
    tokens, topk = topk_ids.shape
    packed_pad = tokens | (topk << 24)
    ids = []
    weights = []
    experts = []
    counts = []
    starts = [-1] * num_experts
    local_expert = 0
    for expert in range(num_experts):
        selected = []
        for token in range(tokens):
            for slot in range(topk):
                if int(topk_ids[token, slot]) == expert:
                    selected.append((token | (slot << 24), float(topk_weights[token, slot])))
        counts.append(len(selected))
        enabled = mask is None or int(mask[expert]) != 0
        if not enabled or not selected:
            if enabled:
                local_expert += 1
            continue
        starts[expert] = len(ids)
        padded = ((len(selected) + unit_size - 1) // unit_size) * unit_size
        ids.extend(item[0] for item in selected)
        weights.extend(item[1] for item in selected)
        ids.extend([packed_pad] * (padded - len(selected)))
        weights.extend([0.0] * (padded - len(selected)))
        experts.extend([local_expert if mask is not None else expert] * (padded // unit_size))
        local_expert += 1
    return ids, weights, experts, counts, starts


@pytest.mark.skipif(not _hipc_available(), reason="requires a Hygon GPU and aicc")
@pytest.mark.parametrize(
    "tokens,num_experts,topk,unit_size,use_mask",
    [
        (1, 8, 2, 16, False),
        (17, 8, 2, 32, True),
        (128, 64, 8, 32, False),
        (512, 64, 8, 32, True),
    ],
)
def test_moe_sorting_matches_reference(
    tokens, num_experts, topk, unit_size, use_mask
):
    from boltops.fused_moe.triton.moe_sorting_hipc import moe_sorting_fwd

    torch.manual_seed(20260720)
    topk_ids = (
        torch.rand((tokens, num_experts), device="cuda")
        .topk(topk, dim=-1)
        .indices.to(torch.int32)
    )
    topk_weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    mask = torch.ones(num_experts, device="cuda", dtype=torch.int32) if use_mask else None
    if mask is not None:
        mask[1::4] = 0

    capacity = tokens * topk + num_experts * unit_size - topk
    blocks = (capacity + unit_size - 1) // unit_size
    sorted_ids = torch.full((capacity,), -1, device="cuda", dtype=torch.int32)
    sorted_weights = torch.full((capacity,), -1.0, device="cuda", dtype=torch.float32)
    sorted_experts = torch.full((blocks,), -1, device="cuda", dtype=torch.int32)
    positions = torch.full((num_experts * 2,), -1, device="cuda", dtype=torch.int32)
    valid = torch.empty(1, device="cuda", dtype=torch.int32)

    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        positions,
        valid,
        None,
        num_experts,
        unit_size,
        mask,
    )
    torch.cuda.synchronize()

    expected = _reference(
        topk_ids.cpu(),
        topk_weights.cpu(),
        num_experts,
        unit_size,
        None if mask is None else mask.cpu(),
    )
    expected_ids, expected_weights, expected_experts, counts, starts = expected
    size = int(valid.item())
    assert size == len(expected_ids)
    assert sorted_ids[:size].cpu().tolist() == expected_ids
    assert sorted_weights[:size].cpu().tolist() == expected_weights
    assert sorted_experts[: len(expected_experts)].cpu().tolist() == expected_experts
    actual_counts = positions[:num_experts].cpu().tolist()
    for expert, count in enumerate(counts):
        if mask is None or int(mask[expert]) != 0:
            assert actual_counts[expert] == count
    actual_starts = positions[num_experts:].cpu().tolist()
    for expert, start in enumerate(starts):
        if start >= 0:
            assert actual_starts[expert] == start


# ---------------------------------------------------------------------------
# Performance (production-ish token/expert sizes). Skipped unless --bench.
# ---------------------------------------------------------------------------


def _bench_allocate(tokens, experts, topk, block, use_mask):
    ids = torch.rand((tokens, experts), device="cuda").topk(topk, dim=-1).indices.int()
    weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    capacity = tokens * topk + experts * block - topk
    outputs = (
        torch.empty(capacity, device="cuda", dtype=torch.int32),
        torch.empty(capacity, device="cuda", dtype=torch.float32),
        torch.empty((capacity + block - 1) // block, device="cuda", dtype=torch.int32),
        torch.empty(experts * 2, device="cuda", dtype=torch.int32),
        torch.empty(1, device="cuda", dtype=torch.int32),
    )
    mask = torch.ones(experts, device="cuda", dtype=torch.int32) if use_mask else None
    if mask is not None:
        mask[1::4] = 0
    return (ids, weights, *outputs, None, experts, block, mask)


@pytest.mark.skipif(not _hipc_available(), reason="requires a Hygon GPU and aicc")
@pytest.mark.bench
@pytest.mark.parametrize(
    "tokens,experts,topk,block,use_mask",
    [
        (2048, 64, 8, 32, False),
        (2048, 64, 8, 32, True),
    ],
)
def test_bench_moe_sorting(tokens, experts, topk, block, use_mask, bench_warmup, bench_repeat):
    from boltops.fused_moe.triton.moe_sorting_hipc import moe_sorting_fwd

    torch.manual_seed(20260720)
    call_args = _bench_allocate(tokens, experts, topk, block, use_mask)
    # Prefer more samples for microbenchmarks of short kernels.
    warmup = max(bench_warmup, 60)
    repeat = max(bench_repeat, 100)

    def _once():
        moe_sorting_fwd(*call_args)

    # Per-iter samples for median/p10/p90 (short kernel).
    for _ in range(warmup):
        _once()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _once()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # us
    samples.sort()
    median_us = statistics.median(samples)
    p10_us = samples[round((len(samples) - 1) * 0.1)]
    p90_us = samples[round((len(samples) - 1) * 0.9)]
    print(
        f"[bench] moe_sorting tokens={tokens} experts={experts} "
        f"topk={topk} mask={use_mask}: "
        f"median={median_us:.1f}us p10={p10_us:.1f}us p90={p90_us:.1f}us"
    )
