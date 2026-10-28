# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness and optional performance tests for Mamba state scatter.

Default ``pytest``: functional / edge-case correctness.
``pytest --bench``: real-model size timing (different shape set).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("triton")

from boltops.generic import (  # noqa: E402
    fused_mamba_state_scatter_with_mask,
)
from boltops.generic.triton.mamba_state_scatter import (  # noqa: E402
    _select_mamba_state_scatter_config,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _ref_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    """PyTorch reference matching the kernel's bounds and masking semantics."""
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return
    src_indices = torch.arange(total_requests, device=step_indices_raw.device)
    valid = (
        (step_indices_raw >= 0)
        & (dst_indices_raw >= 0)
        & (dst_indices_raw < dst.shape[1])
        & (src_indices < src.shape[1])
        & (step_indices_raw < src.shape[2])
    )
    if valid.any():
        dst_indices = dst_indices_raw[valid].long()
        src_indices = src_indices[valid].long()
        step_indices = step_indices_raw[valid].long()
        dst[:, dst_indices] = src[:, src_indices, step_indices]


@GPU_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("steps", ["all_valid", "part_invalid", "all_invalid"])
def test_fused_matches_reference_small(dtype, index_dtype, steps):
    torch.manual_seed(42)
    layers, requests, cache, draft_steps = 3, 8, 13, 4
    trailing = (2, 17)
    src = torch.randn(
        (layers, requests, draft_steps, *trailing), device=DEVICE, dtype=dtype
    )
    initial = torch.randn((layers, cache, *trailing), device=DEVICE, dtype=dtype)
    dst_indices = torch.randperm(cache, device=DEVICE)[:requests].to(index_dtype)
    step_indices = torch.arange(requests, device=DEVICE, dtype=torch.int32) % draft_steps
    if steps == "part_invalid":
        step_indices[::3] = -1
    elif steps == "all_invalid":
        step_indices.fill_(-1)

    expected = initial.clone()
    actual = initial.clone()
    _ref_scatter_with_mask(expected, src, dst_indices, step_indices)
    fused_mamba_state_scatter_with_mask(actual, src, dst_indices, step_indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU_REQUIRED
def test_empty_indices_is_noop():
    dst = torch.randn((2, 5, 7), device=DEVICE)
    src = torch.randn((2, 3, 4, 7), device=DEVICE)
    before = dst.clone()
    empty = torch.empty(0, device=DEVICE, dtype=torch.int32)
    fused_mamba_state_scatter_with_mask(dst, src, empty, empty)
    torch.testing.assert_close(dst, before, rtol=0, atol=0)


@GPU_REQUIRED
def test_out_of_bounds_indices_are_ignored():
    dst = torch.zeros((2, 5, 9), device=DEVICE)
    src = torch.randn((2, 4, 3, 9), device=DEVICE)
    dst_indices = torch.tensor([-1, 1, 5, 3, 2], device=DEVICE)
    steps = torch.tensor([0, 3, 1, 2, 0], device=DEVICE)
    expected = dst.clone()
    _ref_scatter_with_mask(expected, src, dst_indices, steps)
    fused_mamba_state_scatter_with_mask(dst, src, dst_indices, steps)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


def test_nonempty_cpu_tensors_are_rejected():
    with pytest.raises(ValueError, match="only supports CUDA"):
        fused_mamba_state_scatter_with_mask(
            torch.empty(2, 3, 4),
            torch.empty(2, 2, 2, 4),
            torch.tensor([0]),
            torch.tensor([0]),
        )


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("dst", "src", "dst_indices", "steps", "message"),
    [
        (
            lambda: torch.empty(2, 3, device=DEVICE),
            lambda: torch.empty(3, 2, 2, device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            "Layer dimension mismatch",
        ),
        (
            lambda: torch.empty(2, 3, 4, device=DEVICE),
            lambda: torch.empty(2, 2, 2, 5, device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            "Trailing dims mismatch",
        ),
        (
            lambda: torch.empty(2, 3, 4, device=DEVICE),
            lambda: torch.empty(2, 2, 2, 4, device=DEVICE),
            lambda: torch.tensor([[0]], device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            "indices must be 1D",
        ),
        (
            lambda: torch.empty(2, 3, 4, device=DEVICE),
            lambda: torch.empty(2, 2, 2, 4, device=DEVICE),
            lambda: torch.tensor([0, 1], device=DEVICE),
            lambda: torch.tensor([0], device=DEVICE),
            "indices length mismatch",
        ),
    ],
)
def test_input_validation(dst, src, dst_indices, steps, message):
    with pytest.raises(ValueError, match=message):
        fused_mamba_state_scatter_with_mask(
            dst(), src(), dst_indices(), steps()
        )


@GPU_REQUIRED
@pytest.mark.parametrize("which", ["dst", "src"])
def test_noncontiguous_tensors_are_rejected(which):
    dst = torch.empty((2, 3, 4, 5), device=DEVICE)
    src = torch.empty((2, 3, 2, 4, 5), device=DEVICE)
    if which == "dst":
        dst = dst.transpose(-1, -2)
        src = src.transpose(-1, -2).contiguous()
    else:
        src = src.transpose(-1, -2)
        dst = dst.transpose(-1, -2).contiguous()
    indices = torch.tensor([0], device=DEVICE)
    with pytest.raises(ValueError, match=f"{which} tensor must be contiguous"):
        fused_mamba_state_scatter_with_mask(dst, src, indices, indices)


REAL_SHAPES = [
    # High-weight shapes captured from the target network at TP=2.
    pytest.param((16, 128, 128), torch.float32, id="observed-ssm-fp32"),
    pytest.param((4096, 3), torch.bfloat16, id="observed-conv-bf16"),
    # Lower-weight TP=4 variants derived from the same Qwen3-Next state layout.
    pytest.param((8, 128, 128), torch.float32, id="common-tp4-ssm-fp32"),
    pytest.param((2048, 3), torch.bfloat16, id="common-tp4-conv-bf16"),
]


@GPU_REQUIRED
@pytest.mark.parametrize(("trailing", "dtype"), REAL_SHAPES)
def test_real_network_shapes(trailing, dtype):
    """Exercise representative request counts from the captured runtime shapes."""
    layers, cache, spec_size, draft_steps = 30, 287, 49, 4
    src = torch.empty(
        (layers, spec_size, draft_steps, *trailing), device=DEVICE, dtype=dtype
    )
    # Give each request/step a distinct exactly representable value.
    for request in range(32):
        for step in range(draft_steps):
            src[:, request, step].fill_(request * draft_steps + step)

    expected = torch.zeros((layers, cache, *trailing), device=DEVICE, dtype=dtype)
    actual = torch.zeros_like(expected)

    for request_count in (1, 4, 16, 31, 32):
        base_indices = torch.arange(request_count, device=DEVICE) + 17
        step_indices = torch.arange(request_count, device=DEVICE) % draft_steps
        invalid_count = request_count // 10
        if invalid_count:
            step_indices[:invalid_count] = -1
        touched = base_indices[step_indices >= 0].long()

        for index_dtype in (torch.int32, torch.int64):
            expected[:, base_indices.long()] = 0
            actual[:, base_indices.long()] = 0
            dst_indices = base_indices.to(index_dtype)
            _ref_scatter_with_mask(expected, src, dst_indices, step_indices)
            fused_mamba_state_scatter_with_mask(
                actual, src, dst_indices, step_indices
            )
            torch.testing.assert_close(
                actual[:, touched], expected[:, touched], rtol=0, atol=0
            )


# ---------------------------------------------------------------------------
# Performance (captured / derived network sizes). Different from the small
# functional cases above. Skipped unless ``pytest --bench``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchShape:
    name: str
    trailing: tuple[int, ...]
    dtype: torch.dtype


_BENCH_SHAPES = {
    "observed-ssm": _BenchShape("observed-ssm", (16, 128, 128), torch.float32),
    "observed-conv": _BenchShape("observed-conv", (4096, 3), torch.bfloat16),
    "common-tp4-ssm": _BenchShape("common-tp4-ssm", (8, 128, 128), torch.float32),
    "common-tp4-conv": _BenchShape("common-tp4-conv", (2048, 3), torch.bfloat16),
}
_BENCH_REQUESTS = (1, 4, 16, 31, 32)


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape_name", list(_BENCH_SHAPES))
@pytest.mark.parametrize("request_count", _BENCH_REQUESTS)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mask_mode", ["all-valid", "ten-percent-invalid"])
def test_bench_mamba_state_scatter(
    shape_name,
    request_count,
    index_dtype,
    mask_mode,
    time_ms,
    bench_warmup,
    bench_repeat,
):
    """Time ship API on observed / TP-derived network state shapes."""
    case = _BENCH_SHAPES[shape_name]
    layers, cache, spec_size, draft_steps = 30, 287, 49, 4
    src = torch.empty(
        (layers, spec_size, draft_steps, *case.trailing),
        device=DEVICE,
        dtype=case.dtype,
    )
    dst = torch.empty(
        (layers, cache, *case.trailing), device=DEVICE, dtype=case.dtype
    )
    base_indices = torch.arange(request_count, device=DEVICE) + 17
    dst_indices = base_indices.to(index_dtype)
    steps = torch.arange(request_count, device=DEVICE, dtype=torch.int32)
    steps.remainder_(draft_steps)
    invalid_count = request_count // 10
    if mask_mode == "ten-percent-invalid" and invalid_count:
        steps[:invalid_count] = -1
    valid_count = int((steps >= 0).sum().item())
    bytes_moved = (
        2 * valid_count * layers * dst[0, 0].numel() * dst.element_size()
    )

    # Use reduced defaults for this op's larger payload matrix.
    warmup = min(bench_warmup, 5)
    repeat = min(bench_repeat, 20)
    ms = time_ms(
        lambda: fused_mamba_state_scatter_with_mask(dst, src, dst_indices, steps),
        warmup,
        repeat,
    )
    gbps = bytes_moved / (ms * 1e6) if ms else float("inf")
    cfg = _select_mamba_state_scatter_config(dst[0, 0].numel(), request_count)
    print(
        f"[bench] mamba_state_scatter {shape_name} req={request_count} "
        f"{str(index_dtype).removeprefix('torch.')} {mask_mode}: "
        f"{ms:.4f} ms, {gbps:.2f} GB/s, "
        f"B{cfg['BLOCK_SIZE']}/W{cfg['num_warps']}, valid={valid_count}/{request_count}"
    )
    del src, dst
    gc.collect()
    torch.cuda.empty_cache()
