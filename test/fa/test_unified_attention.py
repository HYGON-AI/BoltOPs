# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""ABI, routing, and correctness coverage for UnifiedAttention."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from functools import partial

import pytest
import torch

import yaml

from boltops.fa.unified_attention import (
    KVQuantMode,
    reshape_and_cache_int4,
    unified_attention,
)
from boltops.fa.unified_attention.triton import config
from boltops.tools.unified_attention_triton_tune import tune_unified_attention as tuner
from boltops.fa.unified_attention.triton.unified_attention import (
    _normalize_kv_quant_mode,
    _select_backend,
)


_PARAMETERS = (
    "q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k",
    "max_seqlen_k", "softmax_scale", "causal", "window_size", "block_table",
    "softcap", "q_descale", "k_descale", "v_descale", "seq_threshold_3D",
    "num_par_softmax_segments", "softmax_segm_output", "softmax_segm_max",
    "softmax_segm_expsum", "alibi_slopes", "output_scale", "qq_bias", "sinks",
    "mm_prefix_range", "rswa_prefix_lens", "rswa_window", "use_alibi_sqrt",
    "kv_quant_mode", "k_scale_cache", "v_scale_cache", "chunk_lookback", "use_td",
    "mm_prefix_clamp_sliding_window",
)


@dataclass(frozen=True)
class UnifiedAttentionBenchmarkCase:
    name: str
    seq_lens: tuple[tuple[int, int], ...]
    query_heads: int
    kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    sliding_window: int | None
    use_sinks: bool
    dtype: str
    soft_cap: float | None
    num_segments: int = 16
    seq_threshold_3d: int = 128

    @property
    def kernel(self) -> str:
        return "3D decode" if self.kind == config.KIND_3D else "2D prefill"

    @property
    def kind(self) -> str:
        use_3d = (
            max(q_len for q_len, _ in self.seq_lens) == 1
            and len(self.seq_lens) <= self.seq_threshold_3d
        )
        return config.KIND_3D if use_3d else config.KIND_2D

    @property
    def num_queries_per_kv(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16}[self.dtype]


def _load_unified_attention_benchmark_cases() -> tuple[UnifiedAttentionBenchmarkCase, ...]:
    case_file = (
        Path(__file__).parents[2]
        / "boltops/tools/unified_attention_triton_tune/benchmark_cases.yaml"
    )
    rows = yaml.safe_load(case_file.read_text(encoding="utf-8"))["cases"]
    return tuple(
        UnifiedAttentionBenchmarkCase(
            name=row["name"],
            seq_lens=tuple(tuple(pair) for pair in row["seq_lens"]),
            query_heads=row["query_heads"],
            kv_heads=row["kv_heads"],
            head_size=row["head_size"],
            block_size=row["block_size"],
            num_blocks=row["num_blocks"],
            sliding_window=row["sliding_window"],
            use_sinks=row["use_sinks"],
            dtype=row["dtype"],
            soft_cap=row["soft_cap"],
            num_segments=row["num_segments"],
            seq_threshold_3d=row["seq_threshold_3d"],
        )
        for row in rows
        if row["enabled"]
    )


# Single shape source, migrated from aiter's tune_unified_attention.py.
UNIFIED_ATTENTION_BENCHMARK_CASES = _load_unified_attention_benchmark_cases()


def test_unified_attention_has_the_35_argument_abi() -> None:
    assert tuple(inspect.signature(unified_attention).parameters) == _PARAMETERS
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    assert tuple(inspect.signature(backend.unified_attention).parameters) == _PARAMETERS


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "core"),
        ({"causal": False}, "general"),
        ({"q_descale": object()}, "general"),
        ({"rswa_prefix_lens": object(), "rswa_window": 64}, "general"),
        ({"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR}, "general"),
        ({"k_scale_cache": object()}, "general"),
        ({"v_scale_cache": object()}, "general"),
        ({"chunk_lookback": 0}, "general"),
        ({"use_td": True}, "general"),
        ({"mm_prefix_clamp_sliding_window": True}, "general"),
    ],
)
def test_feature_routing_prefers_core_only_for_its_coverage(kwargs, expected) -> None:
    values = dict(
        causal=True, q_descale=None, rswa_prefix_lens=None, rswa_window=None,
        kv_quant_mode=None, k_scale_cache=None, v_scale_cache=None,
        chunk_lookback=-1, use_td=False, mm_prefix_clamp_sliding_window=False,
    )
    values.update(kwargs)
    assert _select_backend(**values) == expected


@pytest.mark.parametrize("mode", [None, 0, KVQuantMode.NONE])
def test_kv_quant_mode_normalization(mode) -> None:
    assert _normalize_kv_quant_mode(mode) is KVQuantMode.NONE


def test_unknown_kv_quant_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported kv_quant_mode"):
        _normalize_kv_quant_mode(999)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sinks": torch.zeros(3)}, AssertionError, "Sinks must be"),
        ({"mm_prefix_range": torch.zeros(1, 2)}, ValueError, "Unsupported mm_prefix_range"),
        (
            {"kv_quant_mode": KVQuantMode.INT8_PER_TOKEN_HEAD},
            AssertionError,
            "requires k_scale_cache / v_scale_cache",
        ),
        (
            {"causal": False, "kv_quant_mode": KVQuantMode.INT4_PER_TOKEN_HEAD},
            AssertionError,
            "only supports causal attention",
        ),
    ],
)
def test_invalid_feature_contracts_fail_before_kernel_launch(kwargs, error, message) -> None:
    q = torch.zeros(1, 2, 32)
    k = torch.zeros(1, 16, 1, 32)
    v = torch.zeros_like(k)
    values = {"causal": True}
    values.update(kwargs)
    with pytest.raises(error, match=message):
        unified_attention(
            q, k, v, torch.empty_like(q), torch.tensor([0, 1], dtype=torch.int32),
            1, torch.tensor([1], dtype=torch.int64), 1, 32 ** -0.5,
            values.pop("causal"), (-1, -1), torch.tensor([[0]], dtype=torch.int32),
            0.0, None, None, None, **values,
        )


def test_public_facade_forwards_full_abi_positionally(monkeypatch) -> None:
    captured = {}
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )

    def fake(*args):
        captured["args"] = args

    monkeypatch.setattr(backend, "unified_attention", fake)
    entrypoint = importlib.import_module("boltops.fa.unified_attention")
    args = list(range(26))
    args[9] = True
    args[13] = None
    args = tuple(args)
    entrypoint.unified_attention(*args, use_alibi_sqrt=True)
    assert captured["args"][:26] == args
    assert captured["args"][26:29] == (None, None, True)
    assert captured["args"][29:] == (None, None, None, -1, False, False)


def test_common_prefix_and_core_tail_are_forwarded_positionally(monkeypatch) -> None:
    captured = {}
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )

    def fake(*args):
        captured["args"] = args

    monkeypatch.setattr(backend, "_unified_attention_core", fake)
    args = list(range(26))
    args[9] = True
    args[13] = None
    args = tuple(args)
    backend.unified_attention(*args, use_alibi_sqrt=True)
    assert captured["args"] == args + (True,)


def test_general_path_is_forwarded_positionally(monkeypatch) -> None:
    captured = {}
    module_name = "boltops.fa.unified_attention.triton.general_attention"

    def fake(*args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(unified_attention=fake))
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    args = tuple(range(26))
    backend.unified_attention(*args, chunk_lookback=0)
    assert captured["args"][:26] == args
    assert captured["args"][32:] == (0, False, False)
    assert captured["args"][29] is KVQuantMode.NONE


def _reference_paged_attention(q, k_cache, v_cache, q_lens, kv_lens, table, scale, causal):
    out = torch.empty_like(q)
    cursor = 0
    block_size = k_cache.shape[1]
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[seq, : math.ceil(kv_len / block_size)].long()
        k = k_cache[blocks].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:kv_len]
        v = v_cache[blocks].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:kv_len]
        q_seq = q[cursor : cursor + q_len]
        group = q_seq.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_seq.float(), k.float()) * scale
        seq_causal = bool(causal[seq]) if isinstance(causal, (list, tuple, torch.Tensor)) else causal
        if seq_causal:
            positions = torch.arange(q_len, device=q.device) + kv_len - q_len
            scores.masked_fill_(
                torch.arange(kv_len, device=q.device)[None, None, :] > positions[None, :, None],
                float("-inf"),
            )
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out[cursor : cursor + q_len] = torch.einsum("hqk,khd->qhd", probs, v)
        cursor += q_len
    return out


def _reference_single_masked(
    q,
    k,
    v,
    kv_len,
    scale,
    keep,
    *,
    softcap=0.0,
    sinks=None,
    alibi_slopes=None,
    use_alibi_sqrt=False,
    qq_bias=None,
):
    """Small independent reference for a one-sequence paged-cache feature case."""
    q_len, hq = q.shape[:2]
    k = k.reshape(-1, k.shape[2], k.shape[3])[:kv_len]
    v = v.reshape(-1, v.shape[2], v.shape[3])[:kv_len]
    group = hq // k.shape[1]
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    keys = torch.arange(kv_len, device=q.device)
    context_len = kv_len - q_len
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    if alibi_slopes is not None:
        q_pos = context_len + torch.arange(q_len, device=q.device)
        if use_alibi_sqrt:
            relative = keys[None, :] - q_pos[:, None]
            offset = torch.where(
                relative <= 0,
                -torch.sqrt((-relative).float()),
                torch.zeros_like(relative, dtype=torch.float32),
            )
        else:
            offset = (keys - context_len).float().expand(q_len, -1)
        scores += alibi_slopes.float()[:, None, None] * offset[None, :, :]
    if qq_bias is not None:
        key_rel = keys - context_len
        valid = (key_rel >= 0) & (key_rel < q_len)
        scores[:, :, valid] += qq_bias.float()[None, :, key_rel[valid]]
    for row in range(q_len):
        scores[:, row, :].masked_fill_(~keep(kv_len - q_len + row, keys), float("-inf"))
    if sinks is not None:
        sink_logits = sinks.float()[:, None, None].expand(-1, q_len, -1)
        scores = torch.cat((scores, sink_logits), dim=-1)
    probs = torch.softmax(scores, -1)[..., :kv_len]
    return torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)


def _paged_inputs(device: str):
    torch.manual_seed(0)
    q_lens, kv_lens = [1, 3], [13, 15]
    heads_q, heads_kv, dim, page = 4, 2, 32, 16
    q = torch.randn(sum(q_lens), heads_q, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, page, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    indptr = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=device, dtype=torch.int32)
    lens = torch.tensor(kv_lens, device=device, dtype=torch.int64)
    table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    return q_lens, kv_lens, q, k, v, out, indptr, lens, table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize("causal", [True, False])
def test_unified_attention_matches_independent_paged_reference(causal: bool) -> None:
    q_lens, kv_lens, q, k, v, out, indptr, lens, table = _paged_inputs("cuda")
    unified_attention(
        q, k, v, out, indptr, max(q_lens), lens, 16, q.shape[-1] ** -0.5,
        causal, (-1, -1), table, 0.0, None, None, None,
    )
    reference = _reference_paged_attention(
        q, k, v, q_lens, kv_lens, table, q.shape[-1] ** -0.5, causal
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_per_sequence_causal_tensor_matches_reference() -> None:
    q_lens, kv_lens, q, k, v, out, indptr, lens, table = _paged_inputs("cuda")
    causal = torch.tensor([True, False], device="cuda", dtype=torch.bool)
    unified_attention(
        q, k, v, out, indptr, max(q_lens), lens, 16, q.shape[-1] ** -0.5,
        causal, (-1, -1), table, 0.0, None, None, None,
    )
    reference = _reference_paged_attention(
        q, k, v, q_lens, kv_lens, table, q.shape[-1] ** -0.5, causal
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize(("heads_q", "heads_kv"), [(2, 2), (4, 1)])
def test_mha_and_mqa_cross_page_block_mapping(heads_q: int, heads_kv: int) -> None:
    torch.manual_seed(2)
    device, page, dim, q_len, kv_len = "cuda", 16, 32, 2, 23
    q = torch.randn(q_len, heads_q, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(3, page, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    # Logical pages deliberately map to non-adjacent physical pages.
    table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
    unified_attention(
        q, k, v, out,
        torch.tensor([0, q_len], device=device, dtype=torch.int32), q_len,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        dim ** -0.5, True, (-1, -1), table, 0.0, None, None, None,
    )
    reference = _reference_paged_attention(
        q, k, v, [q_len], [kv_len], table, dim ** -0.5, True
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize(
    "feature",
    ["sliding_window", "softcap", "sinks", "alibi", "alibi_sqrt", "qq_bias", "mm_prefix"],
)
def test_core_optional_features_match_reference(feature: str) -> None:
    torch.manual_seed(3)
    device, page, hq, hkv, dim, q_len, kv_len = "cuda", 16, 4, 2, 32, 3, 10
    q = torch.randn(q_len, hq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, page, hkv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    kwargs = {}
    ref_kwargs = {}
    keep = lambda qpos, keys: keys <= qpos
    if feature == "sliding_window":
        kwargs["window_size"] = (3, 0)
        keep = lambda qpos, keys: (keys <= qpos) & ((qpos - keys) < 4)
    elif feature == "softcap":
        kwargs["softcap"] = 1.5
        ref_kwargs["softcap"] = 1.5
    elif feature == "sinks":
        sinks = torch.tensor([0.3, -0.2, 0.1, 0.5], device=device)
        kwargs["sinks"] = sinks
        ref_kwargs["sinks"] = sinks
    elif feature in ("alibi", "alibi_sqrt"):
        slopes = torch.tensor([0.05, 0.1, 0.2, 0.4], device=device)
        kwargs["alibi_slopes"] = slopes
        ref_kwargs["alibi_slopes"] = slopes
        if feature == "alibi_sqrt":
            kwargs["use_alibi_sqrt"] = True
            ref_kwargs["use_alibi_sqrt"] = True
    elif feature == "qq_bias":
        bias = torch.randn(q_len, q_len, device=device) * 0.25
        kwargs["qq_bias"] = bias
        ref_kwargs["qq_bias"] = bias
    else:
        prefix = torch.tensor([[[6, 9]]], device=device, dtype=torch.int32)
        kwargs["mm_prefix_range"] = prefix
        keep = lambda qpos, keys: (keys <= qpos) | (
            ((qpos >= 6) & (qpos <= 9)) & (keys >= 6) & (keys <= 9)
        )

    unified_attention(
        q, k, v, out,
        torch.tensor([0, q_len], device=device, dtype=torch.int32), q_len,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        dim ** -0.5, True, kwargs.pop("window_size", (-1, -1)),
        torch.tensor([[0]], device=device, dtype=torch.int32),
        kwargs.pop("softcap", 0.0), None, None, None, **kwargs,
    )
    reference = _reference_single_masked(
        q, k, v, kv_len, dim ** -0.5, keep, **ref_kwargs
    )
    torch.testing.assert_close(out, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_3d_split_kv_matches_2d_reference() -> None:
    _, _, q, k, v, out, _, _, _ = _paged_inputs("cuda")
    q, out = q[:1], out[:1]
    q_lens, kv_lens = [1], [13]
    indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int64)
    table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    segments, padded = 2, 32
    partial = torch.empty((1, q.shape[1], segments, padded), device="cuda", dtype=torch.float32)
    partial_max = torch.empty((1, q.shape[1], segments), device="cuda", dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    unified_attention(
        q, k, v, out, indptr, 1, lens, 16, q.shape[-1] ** -0.5, True,
        (-1, -1), table, 0.0, None, None, None, 2, segments, partial,
        partial_max, partial_sum,
    )
    reference = _reference_paged_attention(q, k, v, q_lens, kv_lens, table, q.shape[-1] ** -0.5, True)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


def _recorded_kernel(kernel, hits):
    class _Recorder:
        def __getitem__(self, grid):
            launcher = kernel[grid]

            def wrapped(*args, **kwargs):
                hits.append(grid)
                return launcher(*args, **kwargs)

            return wrapped

    return _Recorder()


def _prefill_paged_inputs(device: str):
    torch.manual_seed(1)
    q_len, kv_len = 3, 13
    heads_q, heads_kv, dim, page = 4, 2, 32, 16
    q = torch.randn(q_len, heads_q, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, page, heads_kv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    lens = torch.tensor([kv_len], device=device, dtype=torch.int64)
    table = torch.tensor([[0]], device=device, dtype=torch.int32)
    return q, k, v, out, indptr, lens, table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_3d_split_kv_prefill_matches_reference(monkeypatch) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    q, k, v, out, indptr, lens, table = _prefill_paged_inputs("cuda")
    hits_2d: list = []
    hits_3d: list = []
    monkeypatch.setattr(
        backend, "kernel_unified_attention_2d",
        _recorded_kernel(backend.kernel_unified_attention_2d, hits_2d),
    )
    monkeypatch.setattr(
        backend, "kernel_unified_attention_3d",
        _recorded_kernel(backend.kernel_unified_attention_3d, hits_3d),
    )
    monkeypatch.setattr(
        backend,
        "_get_unified_attention_3d_config",
        lambda **kwargs: {
            "BLOCK_M": 16,
            "TILE_SIZE": 4,
            "num_warps": 1,
            "num_stages": 1,
            "reduce_num_warps": 1,
            "reduce_num_stages": 1,
        },
    )
    partial = torch.empty((q.shape[0], q.shape[1], 2, 32), device="cuda", dtype=torch.float32)
    partial_max = torch.empty((q.shape[0], q.shape[1], 2), device="cuda", dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    unified_attention(
        q, k, v, out, indptr, q.shape[0], lens, lens.item(), q.shape[-1] ** -0.5, True,
        (-1, -1), table, 0.0, None, None, None, 128, 2, partial,
        partial_max, partial_sum,
    )
    assert len(hits_3d) == 1 and hits_2d == []
    reference = _reference_paged_attention(
        q, k, v, [q.shape[0]], [lens.item()], table, q.shape[-1] ** -0.5, True
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_prefill_3d_dispatch_gates(monkeypatch) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    q, k, v, out, indptr, lens, table = _prefill_paged_inputs("cuda")
    reference = _reference_paged_attention(
        q, k, v, [q.shape[0]], [lens.item()], table, q.shape[-1] ** -0.5, True
    )
    hits_2d: list = []
    hits_3d: list = []
    monkeypatch.setattr(
        backend, "kernel_unified_attention_2d",
        _recorded_kernel(backend.kernel_unified_attention_2d, hits_2d),
    )
    monkeypatch.setattr(
        backend, "kernel_unified_attention_3d",
        _recorded_kernel(backend.kernel_unified_attention_3d, hits_3d),
    )

    # Decode-sized scratch (one row) never opens the 3D path for prefill.
    partial = torch.empty((1, q.shape[1], 2, 32), device="cuda", dtype=torch.float32)
    partial_max = torch.empty((1, q.shape[1], 2), device="cuda", dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    out.zero_()
    unified_attention(
        q, k, v, out, indptr, q.shape[0], lens, lens.item(),
        q.shape[-1] ** -0.5, True, (-1, -1), table, 0.0,
        None, None, None, 128, 2, partial, partial_max, partial_sum,
    )
    assert hits_3d == [] and len(hits_2d) == 1
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    # Sized scratch but no tuned 3D config for the shape keeps 2D.
    partial = torch.empty((q.shape[0], q.shape[1], 2, 32), device="cuda", dtype=torch.float32)
    partial_max = torch.empty((q.shape[0], q.shape[1], 2), device="cuda", dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    hits_2d.clear()
    out.zero_()
    unified_attention(
        q, k, v, out, indptr, q.shape[0], lens, lens.item(),
        q.shape[-1] ** -0.5, True, (-1, -1), table, 0.0,
        None, None, None, 128, 2, partial, partial_max, partial_sum,
    )
    assert hits_3d == [] and len(hits_2d) == 1
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    # Sized scratch plus a tuned 3D config switches prefill to split-KV.
    original = backend._get_unified_attention_3d_config
    monkeypatch.setattr(
        backend,
        "_get_unified_attention_3d_config",
        lambda **kwargs: {
            "BLOCK_M": 16,
            "TILE_SIZE": 4,
            "num_warps": 1,
            "num_stages": 1,
            "reduce_num_warps": 1,
            "reduce_num_stages": 1,
        },
    )
    hits_2d.clear()
    out.zero_()
    unified_attention(
        q, k, v, out, indptr, q.shape[0], lens, lens.item(),
        q.shape[-1] ** -0.5, True, (-1, -1), table, 0.0,
        None, None, None, 128, 2, partial, partial_max, partial_sum,
    )
    assert len(hits_3d) == 1 and hits_2d == []
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize("feature", ["chunk", "rswa", "td", "int8_per_token_head"])
def test_general_path_matches_reference(feature: str) -> None:
    torch.manual_seed(1)
    device, page, hq, hkv, dim, kv_len = "cuda", 16, 4, 2, 32, 15
    q = torch.randn(1, hq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, page, hkv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = {}
    keep = lambda qpos, keys: keys <= qpos
    if feature == "chunk":
        kwargs.update(window_size=(15, 0), chunk_lookback=0)
        # sliding_window_val is 16 here, hence lookback=0 retains the query's
        # whole first 16-token chunk. This still compiles the chunk-only path.
        keep = lambda qpos, keys: (keys <= qpos) & ((keys // 16) == (qpos // 16))
    elif feature == "rswa":
        kwargs.update(
            rswa_prefix_lens=torch.tensor([4], device=device, dtype=torch.int32),
            rswa_window=4,
        )
        keep = lambda qpos, keys: (keys <= qpos) & ((keys < 4) | ((qpos - keys) < 4))
    elif feature == "td":
        kwargs.update(use_td=True)
    else:
        k = torch.randint(-8, 8, k.shape, device=device, dtype=torch.int8)
        v = torch.randint(-8, 8, v.shape, device=device, dtype=torch.int8)
        kwargs.update(
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            k_scale_cache=torch.ones((1, page, hkv), device=device),
            v_scale_cache=torch.ones((1, page, hkv), device=device),
        )
    out = torch.empty_like(q)
    unified_attention(
        q, k, v, out, torch.tensor([0, 1], device=device, dtype=torch.int32), 1,
        torch.tensor([kv_len], device=device, dtype=torch.int64), page,
        dim ** -0.5, True, kwargs.pop("window_size", (-1, -1)),
        torch.tensor([[0]], device=device, dtype=torch.int32), 0.0, None, None, None,
        **kwargs,
    )
    reference = _reference_single_masked(q, k, v, kv_len, dim ** -0.5, keep)
    torch.testing.assert_close(out, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_general_path_noncausal_d72_matches_reference() -> None:
    # ViT bi-attention family (shaobob0 manifest unified_attention-141):
    # non-power-of-2 head size 72 (padded to 128) with non-causal masking,
    # routed to the general backend. The reference is computed on CPU copies
    # because GPU-side rocBLAS has no gfx946 TensileLibrary on the PMD build
    # (see docs/ops/gfx946_gdn_unified_attention.md); kernel inputs stay on
    # device.
    torch.manual_seed(1)
    device, page, hq, hkv, dim, kv_len = "cuda", 16, 4, 4, 72, 15
    q_cpu = torch.randn(1, hq, dim, dtype=torch.bfloat16)
    k_cpu = torch.randn(1, page, hkv, dim, dtype=torch.bfloat16)
    v_cpu = torch.randn_like(k_cpu)
    q, k, v = q_cpu.to(device), k_cpu.to(device), v_cpu.to(device)
    out = torch.empty_like(q)
    unified_attention(
        q, k, v, out, torch.tensor([0, 1], device=device, dtype=torch.int32), 1,
        torch.tensor([kv_len], device=device, dtype=torch.int64), page,
        dim ** -0.5, False, (-1, -1),
        torch.tensor([[0]], device=device, dtype=torch.int32), 0.0, None, None, None,
    )
    reference = _reference_single_masked(
        q_cpu, k_cpu, v_cpu, kv_len, dim ** -0.5,
        lambda qpos, keys: torch.ones_like(keys, dtype=torch.bool)
    )
    torch.testing.assert_close(out.cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize("mode", [KVQuantMode.FP8_PER_TENSOR, KVQuantMode.FP8_PER_TOKEN_HEAD])
def test_fp8_kv_cache_modes_match_dequantized_reference(mode: KVQuantMode) -> None:
    torch.manual_seed(5)
    device, page, hq, hkv, dim, kv_len = "cuda", 16, 4, 2, 32, 13
    q = torch.randn(1, hq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, page, hkv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fp8_dtype = torch.float8_e4m3fn
    kwargs = {"kv_quant_mode": mode}
    if mode is KVQuantMode.FP8_PER_TENSOR:
        k_scale = torch.tensor(0.25, device=device)
        v_scale = torch.tensor(0.5, device=device)
        k_q = (k.float() / k_scale).to(fp8_dtype)
        v_q = (v.float() / v_scale).to(fp8_dtype)
        k_ref = k_q.float() * k_scale
        v_ref = v_q.float() * v_scale
        descales = (k_scale, v_scale)
    else:
        token = torch.arange(page, device=device, dtype=torch.float32)
        k_scale = (0.2 + token[:, None] * 0.005).expand(page, hkv)[None].contiguous()
        v_scale = (0.3 + token[:, None] * 0.004).expand(page, hkv)[None].contiguous()
        k_q = (k.float() / k_scale[..., None]).to(fp8_dtype)
        v_q = (v.float() / v_scale[..., None]).to(fp8_dtype)
        k_ref = k_q.float() * k_scale[..., None]
        v_ref = v_q.float() * v_scale[..., None]
        kwargs.update(k_scale_cache=k_scale, v_scale_cache=v_scale)
        descales = (None, None)

    out = torch.empty_like(q)
    unified_attention(
        q, k_q, v_q, out,
        torch.tensor([0, 1], device=device, dtype=torch.int32), 1,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        dim ** -0.5, True, (-1, -1),
        torch.tensor([[0]], device=device, dtype=torch.int32),
        0.0, None, descales[0], descales[1], **kwargs,
    )
    keep = lambda qpos, keys: keys <= qpos
    reference = _reference_single_masked(
        q, k_ref, v_ref, kv_len, dim ** -0.5, keep
    )
    torch.testing.assert_close(out, reference, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_fp8_output_scale_matches_reference() -> None:
    torch.manual_seed(7)
    device, page, hq, hkv, dim, kv_len = "cuda", 16, 4, 2, 32, 13
    q = torch.randn(1, hq, dim, device=device, dtype=torch.float16)
    k = torch.randn(1, page, hkv, dim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    output_scale = torch.tensor(0.25, device=device, dtype=torch.float32)
    out = torch.empty(q.shape, device=device, dtype=torch.float8_e4m3fn)
    unified_attention(
        q, k, v, out,
        torch.tensor([0, 1], device=device, dtype=torch.int32), 1,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        dim ** -0.5, True, (-1, -1),
        torch.tensor([[0]], device=device, dtype=torch.int32),
        0.0, None, None, None, output_scale=output_scale,
    )
    keep = lambda qpos, keys: keys <= qpos
    reference = _reference_single_masked(q, k, v, kv_len, dim ** -0.5, keep)
    dequantized = out.float() * output_scale
    torch.testing.assert_close(dequantized, reference.float(), atol=2e-1, rtol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
def test_int4_cache_write_and_attention_match_reference() -> None:
    torch.manual_seed(8)
    device, page, hq, hkv, dim, kv_len = "cuda", 16, 4, 2, 32, 12
    q = torch.randn(1, hq, dim, device=device, dtype=torch.bfloat16)
    k_tokens = torch.randn(kv_len, hkv, dim, device=device, dtype=torch.bfloat16)
    v_tokens = torch.randn_like(k_tokens)
    k_cache = torch.empty(1, page, hkv, dim // 2, device=device, dtype=torch.uint8)
    v_cache = torch.empty_like(k_cache)
    k_scales = torch.empty(1, page, hkv, device=device, dtype=torch.float32)
    v_scales = torch.empty_like(k_scales)
    reshape_and_cache_int4(
        k_tokens, v_tokens, k_cache, v_cache,
        torch.arange(kv_len, device=device, dtype=torch.int64),
        k_scale_cache=k_scales, v_scale_cache=v_scales,
    )
    out = torch.empty_like(q)
    unified_attention(
        q, k_cache, v_cache, out,
        torch.tensor([0, 1], device=device, dtype=torch.int32), 1,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        dim ** -0.5, True, (-1, -1),
        torch.tensor([[0]], device=device, dtype=torch.int32),
        0.0, None, None, None, kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        k_scale_cache=k_scales, v_scale_cache=v_scales,
    )
    keep = lambda qpos, keys: keys <= qpos
    reference = _reference_single_masked(
        q, k_tokens[None], v_tokens[None], kv_len, dim ** -0.5, keep
    )
    torch.testing.assert_close(out, reference, atol=3e-1, rtol=3e-1)


def _make_unified_attention_benchmark_inputs(
    case: UnifiedAttentionBenchmarkCase,
) -> dict[str, torch.Tensor | int | float | tuple[int, int] | None]:
    torch.manual_seed(0)
    device = "cuda"
    query_lens = [q_len for q_len, _ in case.seq_lens]
    kv_lens = [kv_len for _, kv_len in case.seq_lens]
    max_blocks_per_seq = math.ceil(max(kv_lens) / case.block_size)
    assert case.num_blocks >= len(case.seq_lens) * max_blocks_per_seq

    q = torch.randn(
        sum(query_lens), case.query_heads, case.head_size,
        device=device, dtype=torch.bfloat16,
    )
    k = torch.randn(
        case.num_blocks, case.block_size, case.kv_heads, case.head_size,
        device=device, dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    block_table = torch.randperm(case.num_blocks, device=device, dtype=torch.int64)
    block_table = block_table[: len(case.seq_lens) * max_blocks_per_seq]
    block_table = block_table.reshape(len(case.seq_lens), max_blocks_per_seq).int()
    sinks = (
        torch.randn(case.query_heads, device=device, dtype=torch.bfloat16)
        if case.use_sinks
        else None
    )
    padded_head = config.head_size_layout(case.head_size)[0]
    # The 3D scratch is indexed by global query-token offset, so it must
    # cover every query token (prefill) in addition to the decode row
    # count (seq_threshold_3d).
    segm_rows = max(case.seq_threshold_3d, sum(q_len for q_len, _ in case.seq_lens))
    partial = torch.empty(
        segm_rows,
        case.query_heads,
        case.num_segments,
        padded_head,
        device=device,
        dtype=torch.float32,
    )
    partial_max = torch.empty(
        segm_rows,
        case.query_heads,
        case.num_segments,
        device=device,
        dtype=torch.float32,
    )
    partial_sum = torch.empty_like(partial_max)
    window_size = (
        (case.sliding_window - 1, 0)
        if case.sliding_window is not None
        else (-1, -1)
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": seqused_k,
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": case.head_size ** -0.5,
        "window_size": window_size,
        "block_table": block_table,
        "sinks": sinks,
        "softmax_segm_output": partial,
        "softmax_segm_max": partial_max,
        "softmax_segm_expsum": partial_sum,
    }


def _unified_attention_benchmark_work(case: UnifiedAttentionBenchmarkCase) -> tuple[float, int]:
    attended_tokens = 0
    for query_len, kv_len in case.seq_lens:
        context_len = kv_len - query_len
        for query_idx in range(query_len):
            attended = context_len + query_idx + 1
            if case.sliding_window is not None:
                attended = min(attended, case.sliding_window)
            attended_tokens += attended

    flops = 4.0 * case.query_heads * case.head_size * attended_tokens
    dtype_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    query_tokens = sum(q_len for q_len, _ in case.seq_lens)
    query_output_bytes = (
        2 * query_tokens * case.query_heads * case.head_size * dtype_bytes
    )
    kv_bytes = (
        2 * attended_tokens * case.kv_heads * case.head_size * dtype_bytes
    )
    return flops, query_output_bytes + kv_bytes


@pytest.mark.bench
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA device")
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in UNIFIED_ATTENTION_BENCHMARK_CASES],
)
@torch.inference_mode()
def test_unified_attention_benchmark(
    case: UnifiedAttentionBenchmarkCase,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
) -> None:
    """Benchmark the six production cases carried by aiter's tune script."""
    data = _make_unified_attention_benchmark_inputs(case)

    def run() -> None:
        unified_attention(
            data["q"], data["k"], data["v"], data["out"],
            data["cu_seqlens_q"], data["max_seqlen_q"], data["seqused_k"],
            data["max_seqlen_k"], data["softmax_scale"], True,
            data["window_size"], data["block_table"], 0.0, None, None, None,
            case.seq_threshold_3d, case.num_segments,
            data["softmax_segm_output"], data["softmax_segm_max"],
            data["softmax_segm_expsum"], sinks=data["sinks"],
        )

    run()
    assert torch.isfinite(data["out"]).all()
    # CUDA-graph-first timing (time_ms default): production runs the
    # operator in graph-replay mode, so the bench measures the same
    # steady state (graph keeps KV L2-warm; do_bench would flush L2
    # per iteration and time decode ~1.3-1.4x slower, verified
    # 2026-10-15).  Capture failure falls back to event timing with a
    # warning; the perf collector records the method actually used in
    # the CSV timing column.
    elapsed_ms = time_ms(run, bench_warmup, bench_repeat)
    flops, io_bytes = _unified_attention_benchmark_work(case)
    tflops = flops / (elapsed_ms * 1e9)
    bandwidth_gbs = io_bytes / (elapsed_ms * 1e6)
    seqs = ",".join(f"{q}/{kv}" for q, kv in case.seq_lens)
    print(
        f"Unified Attention {case.kernel}: case={case.name}, q/kv={seqs}, "
        f"Hq/Hkv={case.query_heads}/{case.kv_heads}, D={case.head_size}, "
        f"block={case.block_size}, SW={case.sliding_window}, "
        f"sinks={case.use_sinks}: {elapsed_ms:.4f} ms, "
        f"{bandwidth_gbs:.2f} GB/s, {tflops:.2f} TFLOPS"
    )


@pytest.fixture
def clear_unified_attention_config_cache():
    config.clear_config_cache()
    yield
    config.clear_config_cache()


def _lookup_tuned_config(case: UnifiedAttentionBenchmarkCase):
    kwargs = dict(
        cache_block_size=case.block_size,
        head_size=case.head_size,
        num_seqs=len(case.seq_lens),
        num_query_tokens=sum(q_len for q_len, _ in case.seq_lens),
        max_seqlen_q=max(q_len for q_len, _ in case.seq_lens),
        max_seqlen_k=max(kv_len for _, kv_len in case.seq_lens),
        num_queries_per_kv=case.num_queries_per_kv,
        sliding_window=case.sliding_window or 0,
        use_alibi_slopes=False,
        use_qq_bias=False,
        use_softcap=bool(case.soft_cap and case.soft_cap > 0),
        use_sinks=case.use_sinks,
        use_mm_prefix=False,
        kv_dtype=case.torch_dtype,
    )
    if case.kind == config.KIND_2D:
        return config.get_kernel_config(case.kind, use_fp8_output=False, **kwargs)
    return config.get_kernel_config(
        case.kind,
        num_segments_per_seq=case.num_segments,
        **kwargs,
    )


def test_yaml_covers_aiter_cases_and_kernel_paths() -> None:
    cases = UNIFIED_ATTENTION_BENCHMARK_CASES
    assert tuple(case.name for case in cases) == (
        "prefill-gqa-sink-swa",
        "prefill-mqa",
        "prefill-gqa-sink-swa-multibatch",
        "prefill-mqa-multibatch",
        "decode-gqa-sink-swa-3d",
        "decode-mqa-3d",
    )
    assert [case.kind for case in cases].count(config.KIND_2D) == 4
    assert [case.kind for case in cases].count(config.KIND_3D) == 2
    keys = [
        config.make_config_key(
            num_seqs=len(case.seq_lens),
            num_query_tokens=sum(q_len for q_len, _ in case.seq_lens),
            max_seqlen_q=max(q_len for q_len, _ in case.seq_lens),
            max_seqlen_k=max(kv_len for _, kv_len in case.seq_lens),
            num_queries_per_kv=case.num_queries_per_kv,
        )
        for case in cases
    ]
    assert len(keys) == len(set(keys))


def test_3d_reduce_search_uses_full_cartesian_product(monkeypatch) -> None:
    case = next(case for case in tuner.load_cases() if case.kind == config.KIND_3D)
    attention_configs = tuner.candidate_configs(case)
    captured = []

    def fake_select_best(case, data, reference, candidates, **kwargs):
        captured.extend(candidates)
        return candidates[0], 1.0

    monkeypatch.setattr(tuner, "_select_best", fake_select_best)
    best, best_ms = tuner._tune_reduce(
        case, {}, torch.empty(0), attention_configs,
        quick=False, warmup_ms=1, rep_ms=1,
    )

    assert len(attention_configs) == 24
    assert len(captured) == 24 * 8
    assert len({tuple(candidate.items()) for candidate in captured}) == 24 * 8
    assert best == captured[0]
    assert best_ms == 1.0


def test_prefill_uses_expanded_search_space_without_changing_decode_space() -> None:
    cases = {case.name: case for case in tuner.load_cases()}

    assert len(tuner.candidate_configs(cases["prefill-gqa-sink-swa"])) == 108
    assert len(tuner.candidate_configs(cases["prefill-mqa"])) == 72
    assert (
        len(tuner.candidate_configs(cases["prefill-gqa-sink-swa-multibatch"]))
        == 108
    )
    assert len(tuner.candidate_configs(cases["prefill-mqa-multibatch"])) == 72
    assert len(tuner.candidate_configs(cases["decode-gqa-sink-swa-3d"])) == 24
    assert len(tuner.candidate_configs(cases["decode-mqa-3d"])) == 24


def test_runtime_reads_exact_versioned_json(
    tmp_path, monkeypatch, clear_unified_attention_config_cache
) -> None:
    case = UNIFIED_ATTENTION_BENCHMARK_CASES[0]
    expected = {
        "BLOCK_M": 32,
        "TILE_SIZE": 16,
        "num_warps": 4,
        "num_stages": 1,
    }
    multibatch_expected = {
        "BLOCK_M": 16,
        "TILE_SIZE": 64,
        "num_warps": 4,
        "num_stages": 2,
    }
    monkeypatch.setattr(config, "CONFIG_ROOT", tmp_path)
    monkeypatch.setattr(
        config,
        "get_device_info",
        lambda device=None: SimpleNamespace(arch="gfx938", cu_count=64),
    )
    monkeypatch.setattr(
        config,
        "get_compatible_triton_config_dirs",
        lambda root, capabilities: [root / "triton_3_6"],
    )
    directory = tmp_path / "triton_3_6"
    directory.mkdir()
    filename = config.make_config_filename(
        case.kind,
        device_label="gfx938_cu64",
        cache_block_size=case.block_size,
        head_size=case.head_size,
        sliding_window=case.sliding_window or 0,
        use_alibi_slopes=False,
        use_qq_bias=False,
        use_softcap=False,
        use_sinks=case.use_sinks,
        use_mm_prefix=False,
        kv_dtype=case.torch_dtype,
        use_fp8_output=False,
    )
    config_key = config.make_config_key(
        num_seqs=len(case.seq_lens),
        num_query_tokens=sum(q_len for q_len, _ in case.seq_lens),
        max_seqlen_q=max(q_len for q_len, _ in case.seq_lens),
        max_seqlen_k=max(kv_len for _, kv_len in case.seq_lens),
        num_queries_per_kv=case.num_queries_per_kv,
    )
    multibatch_case = replace(
        case,
        name="prefill-gqa-sink-swa-multibatch",
        seq_lens=((3, 4096), (49, 4096), (511, 4096)),
    )
    multibatch_key = config.make_config_key(
        num_seqs=len(multibatch_case.seq_lens),
        num_query_tokens=sum(q_len for q_len, _ in multibatch_case.seq_lens),
        max_seqlen_q=max(q_len for q_len, _ in multibatch_case.seq_lens),
        max_seqlen_k=max(kv_len for _, kv_len in multibatch_case.seq_lens),
        num_queries_per_kv=multibatch_case.num_queries_per_kv,
    )
    (directory / filename).write_text(
        json.dumps(
            {
                "key": list(config.CONFIG_KEY_FIELDS),
                "config": {
                    str(config_key): expected,
                    str(multibatch_key): multibatch_expected,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _lookup_tuned_config(case) == expected
    assert _lookup_tuned_config(multibatch_case) == multibatch_expected


def test_default_config_warning_is_emitted_once(monkeypatch, caplog) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    backend._warn_default_config_once.cache_clear()
    monkeypatch.setattr(config, "get_kernel_config", lambda *args, **kwargs: None)
    kwargs = dict(
        cache_block_size=7,
        head_size=48,
        num_seqs=2,
        num_query_tokens=5,
        max_seqlen_q=3,
        max_seqlen_k=17,
        num_queries_per_kv=2,
        sliding_window=0,
        use_alibi_slopes=False,
        use_qq_bias=False,
        use_softcap=False,
        use_sinks=False,
        use_mm_prefix=False,
        use_fp8=False,
        kv_dtype=torch.bfloat16,
    )
    with caplog.at_level(logging.WARNING, logger=backend.__name__):
        backend._get_unified_attention_2d_config(**kwargs)
        backend._get_unified_attention_2d_config(**kwargs)
    messages = [
        record.message for record in caplog.records
        if "No tuned UnifiedAttention 2D config" in record.message
    ]
    assert len(messages) == 1
    backend._warn_default_config_once.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA device")
def test_all_packaged_aiter_cases_hit_runtime_configs(
    clear_unified_attention_config_cache,
) -> None:
    if not any(config.CONFIG_ROOT.glob("triton_*/*.json")):
        pytest.skip("no packaged UnifiedAttention configs")
    info = config.get_device_info()
    if (info.arch, info.cu_count) not in {
        ("gfx936", 80),
        ("gfx938", 64),
        ("gfx938", 72),
    }:
        pytest.skip(f"no packaged configs for {info.arch}_cu{info.cu_count}")
    for case in UNIFIED_ATTENTION_BENCHMARK_CASES:
        assert _lookup_tuned_config(case) is not None, case.name


def test_gfx946_fp8_prefill_fallback_uses_wide_tile(monkeypatch) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    monkeypatch.setattr(backend, "get_arch", lambda device=None: "gfx946")

    fp8_dtypes = [torch.float8_e4m3fn]
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    for dtype in fp8_dtypes:
        for head_size, queries_per_kv in ((256, 8), (256, 16), (128, 8)):
            result = backend._default_2d_config(
                "cuda",
                16,
                32,
                cache_block_size=16,
                head_size=head_size,
                num_seqs=1,
                num_queries_per_kv=queries_per_kv,
                max_seqlen_q=256,
                max_seqlen_k=4096,
                kv_dtype=dtype,
                plain_attention=True,
            )

            assert result == {
                "BLOCK_M": 64,
                "TILE_SIZE": 64,
                "num_warps": 4,
                "num_stages": 1,
            }


def test_gfx946_fp8_fallback_does_not_change_featured_attention(monkeypatch) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    monkeypatch.setattr(backend, "get_arch", lambda device=None: "gfx946")

    result = backend._default_2d_config(
        "cuda",
        16,
        32,
        cache_block_size=16,
        head_size=256,
        num_seqs=1,
        num_queries_per_kv=8,
        max_seqlen_q=256,
        max_seqlen_k=4096,
        kv_dtype=torch.float8_e4m3fn,
        plain_attention=False,
    )

    assert result == {
        "BLOCK_M": 16,
        "TILE_SIZE": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


# ---------------------------------------------------------------------------
# shaobob0_1102 workload shape coverage.
#
# _GFX946_UA_CASES holds all 95 ready unified_attention shapes of the
# shaobob0_1102 manifest, inlined so the suite has no generated-manifest
# dependency.  Fields:
#   (case_id, batch, query_heads, kv_heads, query_length, kv_length,
#    head_size, value_head_size, fp8_kv)
#
# Shapes are tiered by size so the default run stays cheap on any arch
# (including the PMD perf model, whose GPU rocBLAS has no gfx946
# TensileLibrary and therefore cannot compute an on-device reference):
#   * small  (score_count <= 2e8): always run; reference on CPU
#   * large  : run with BOLTOPS_UA_RUN_LARGE=1; reference on device
#   * huge   (paged-cache footprint > 96GB, the B32 Kimi prefill shapes
#     that need ~140-190GB of HBM): additionally BOLTOPS_UA_RUN_HUGE=1
# The kernel itself is arch-agnostic Triton, so every tier runs on gfx938
# CI and gfx946 alike; only the reference location differs.
# ---------------------------------------------------------------------------


class _UaWorkloadShape(NamedTuple):
    case_id: str
    batch: int
    query_heads: int
    kv_heads: int
    query_length: int
    kv_length: int
    head_size: int
    value_head_size: int
    fp8_kv: bool

    @property
    def score_count(self) -> int:
        return self.query_heads * self.query_length * self.kv_length

    @property
    def cache_bytes(self) -> int:
        kv_bytes = 1 if self.fp8_kv else 2
        head_bytes = self.head_size + self.value_head_size
        return (
            self.batch * self.kv_length * self.kv_heads * head_bytes * kv_bytes
            + self.batch * self.query_length * self.query_heads * head_bytes * 2
        )


_GFX946_UA_CASES = [
    _UaWorkloadShape("unified_attention-012", 1, 16, 2, 2560, 4096, 256, 256, True),
    _UaWorkloadShape("unified_attention-017", 64, 16, 2, 1, 4608, 256, 256, True),
    _UaWorkloadShape("unified_attention-018", 1, 16, 2, 6656, 6656, 256, 256, True),
    _UaWorkloadShape("unified_attention-019", 1, 16, 2, 6656, 13312, 256, 256, True),
    _UaWorkloadShape("unified_attention-020", 1, 16, 2, 6656, 19968, 256, 256, True),
    _UaWorkloadShape("unified_attention-021", 1, 16, 2, 6656, 26624, 256, 256, True),
    _UaWorkloadShape("unified_attention-022", 1, 16, 2, 6656, 33280, 256, 256, True),
    _UaWorkloadShape("unified_attention-023", 1, 16, 2, 6656, 39936, 256, 256, True),
    _UaWorkloadShape("unified_attention-024", 1, 16, 2, 6656, 46592, 256, 256, True),
    _UaWorkloadShape("unified_attention-025", 1, 16, 2, 6656, 53248, 256, 256, True),
    _UaWorkloadShape("unified_attention-026", 1, 16, 2, 6656, 59904, 256, 256, True),
    _UaWorkloadShape("unified_attention-027", 1, 16, 2, 6656, 66560, 256, 256, True),
    _UaWorkloadShape("unified_attention-028", 1, 16, 2, 6656, 73216, 256, 256, True),
    _UaWorkloadShape("unified_attention-029", 1, 16, 2, 6656, 79872, 256, 256, True),
    _UaWorkloadShape("unified_attention-030", 1, 16, 2, 6656, 86528, 256, 256, True),
    _UaWorkloadShape("unified_attention-031", 1, 16, 2, 6656, 93184, 256, 256, True),
    _UaWorkloadShape("unified_attention-032", 1, 16, 2, 6656, 99840, 256, 256, True),
    _UaWorkloadShape("unified_attention-033", 1, 16, 2, 6656, 106496, 256, 256, True),
    _UaWorkloadShape("unified_attention-034", 1, 16, 2, 6656, 113152, 256, 256, True),
    _UaWorkloadShape("unified_attention-035", 1, 16, 2, 6656, 119808, 256, 256, True),
    _UaWorkloadShape("unified_attention-036", 1, 16, 2, 6656, 126464, 256, 256, True),
    _UaWorkloadShape("unified_attention-037", 1, 16, 2, 6656, 131072, 256, 256, True),
    _UaWorkloadShape("unified_attention-043", 1, 16, 2, 2048, 131072, 256, 256, True),
    _UaWorkloadShape("unified_attention-048", 12, 16, 2, 1, 131584, 256, 256, True),
    _UaWorkloadShape("unified_attention-049", 4, 64, 64, 6656, 6656, 192, 128, False),
    _UaWorkloadShape("unified_attention-050", 4, 64, 64, 6656, 13312, 192, 128, False),
    _UaWorkloadShape("unified_attention-051", 4, 64, 64, 6656, 19968, 192, 128, False),
    _UaWorkloadShape("unified_attention-052", 4, 64, 64, 6656, 26624, 192, 128, False),
    _UaWorkloadShape("unified_attention-053", 4, 64, 64, 6656, 33280, 192, 128, False),
    _UaWorkloadShape("unified_attention-054", 4, 64, 64, 6656, 39936, 192, 128, False),
    _UaWorkloadShape("unified_attention-055", 4, 64, 64, 6656, 46592, 192, 128, False),
    _UaWorkloadShape("unified_attention-056", 4, 64, 64, 6656, 53248, 192, 128, False),
    _UaWorkloadShape("unified_attention-057", 4, 64, 64, 6656, 59904, 192, 128, False),
    _UaWorkloadShape("unified_attention-058", 4, 64, 64, 6656, 66560, 192, 128, False),
    _UaWorkloadShape("unified_attention-059", 4, 64, 64, 6656, 73216, 192, 128, False),
    _UaWorkloadShape("unified_attention-060", 4, 64, 64, 6656, 79872, 192, 128, False),
    _UaWorkloadShape("unified_attention-061", 4, 64, 64, 6656, 86528, 192, 128, False),
    _UaWorkloadShape("unified_attention-062", 4, 64, 64, 6656, 93184, 192, 128, False),
    _UaWorkloadShape("unified_attention-063", 4, 64, 64, 6656, 99840, 192, 128, False),
    _UaWorkloadShape("unified_attention-064", 4, 64, 64, 6656, 106496, 192, 128, False),
    _UaWorkloadShape("unified_attention-065", 4, 64, 64, 6656, 113152, 192, 128, False),
    _UaWorkloadShape("unified_attention-066", 4, 64, 64, 6656, 119808, 192, 128, False),
    _UaWorkloadShape("unified_attention-067", 4, 64, 64, 6656, 126464, 192, 128, False),
    _UaWorkloadShape("unified_attention-068", 4, 64, 64, 4608, 131072, 192, 128, False),
    _UaWorkloadShape("unified_attention-069", 32, 64, 64, 6656, 53248, 192, 128, False),
    _UaWorkloadShape("unified_attention-070", 32, 64, 64, 6656, 106496, 192, 128, False),
    _UaWorkloadShape("unified_attention-071", 32, 64, 64, 3072, 131072, 192, 128, False),
    _UaWorkloadShape("unified_attention-072", 4, 64, 64, 1, 131072, 128, 128, True),
    _UaWorkloadShape("unified_attention-073", 1, 64, 8, 6656, 6656, 128, 128, True),
    _UaWorkloadShape("unified_attention-074", 1, 64, 8, 6656, 13312, 128, 128, True),
    _UaWorkloadShape("unified_attention-075", 1, 64, 8, 6656, 19968, 128, 128, True),
    _UaWorkloadShape("unified_attention-076", 1, 64, 8, 6656, 26624, 128, 128, True),
    _UaWorkloadShape("unified_attention-077", 1, 64, 8, 6656, 33280, 128, 128, True),
    _UaWorkloadShape("unified_attention-078", 1, 64, 8, 6656, 39936, 128, 128, True),
    _UaWorkloadShape("unified_attention-079", 1, 64, 8, 6656, 46592, 128, 128, True),
    _UaWorkloadShape("unified_attention-080", 1, 64, 8, 6656, 53248, 128, 128, True),
    _UaWorkloadShape("unified_attention-081", 1, 64, 8, 6656, 59904, 128, 128, True),
    _UaWorkloadShape("unified_attention-082", 1, 64, 8, 6656, 66560, 128, 128, True),
    _UaWorkloadShape("unified_attention-083", 1, 64, 8, 6656, 73216, 128, 128, True),
    _UaWorkloadShape("unified_attention-084", 1, 64, 8, 6656, 79872, 128, 128, True),
    _UaWorkloadShape("unified_attention-085", 1, 64, 8, 6656, 86528, 128, 128, True),
    _UaWorkloadShape("unified_attention-086", 1, 64, 8, 6656, 93184, 128, 128, True),
    _UaWorkloadShape("unified_attention-087", 1, 64, 8, 6656, 99840, 128, 128, True),
    _UaWorkloadShape("unified_attention-088", 1, 64, 8, 6656, 106496, 128, 128, True),
    _UaWorkloadShape("unified_attention-089", 1, 64, 8, 6656, 113152, 128, 128, True),
    _UaWorkloadShape("unified_attention-090", 1, 64, 8, 6656, 119808, 128, 128, True),
    _UaWorkloadShape("unified_attention-091", 1, 64, 8, 6656, 126464, 128, 128, True),
    _UaWorkloadShape("unified_attention-092", 1, 64, 8, 6656, 131072, 128, 128, True),
    _UaWorkloadShape("unified_attention-093", 1, 64, 8, 512, 131072, 128, 128, True),
    _UaWorkloadShape("unified_attention-094", 4, 64, 8, 4, 131584, 128, 128, True),
    _UaWorkloadShape("unified_attention-095", 4, 64, 8, 1, 131584, 128, 128, True),
    _UaWorkloadShape("unified_attention-101", 1, 32, 2, 6656, 6656, 256, 256, True),
    _UaWorkloadShape("unified_attention-102", 1, 32, 2, 6656, 13312, 256, 256, True),
    _UaWorkloadShape("unified_attention-103", 1, 32, 2, 6656, 19968, 256, 256, True),
    _UaWorkloadShape("unified_attention-104", 1, 32, 2, 6656, 26624, 256, 256, True),
    _UaWorkloadShape("unified_attention-105", 1, 32, 2, 6656, 33280, 256, 256, True),
    _UaWorkloadShape("unified_attention-106", 1, 32, 2, 6656, 39936, 256, 256, True),
    _UaWorkloadShape("unified_attention-107", 1, 32, 2, 6656, 46592, 256, 256, True),
    _UaWorkloadShape("unified_attention-108", 1, 32, 2, 6656, 53248, 256, 256, True),
    _UaWorkloadShape("unified_attention-109", 1, 32, 2, 6656, 59904, 256, 256, True),
    _UaWorkloadShape("unified_attention-110", 1, 32, 2, 6656, 66560, 256, 256, True),
    _UaWorkloadShape("unified_attention-111", 1, 32, 2, 6656, 73216, 256, 256, True),
    _UaWorkloadShape("unified_attention-112", 1, 32, 2, 6656, 79872, 256, 256, True),
    _UaWorkloadShape("unified_attention-113", 1, 32, 2, 6656, 86528, 256, 256, True),
    _UaWorkloadShape("unified_attention-114", 1, 32, 2, 6656, 93184, 256, 256, True),
    _UaWorkloadShape("unified_attention-115", 1, 32, 2, 6656, 99840, 256, 256, True),
    _UaWorkloadShape("unified_attention-116", 1, 32, 2, 6656, 106496, 256, 256, True),
    _UaWorkloadShape("unified_attention-117", 1, 32, 2, 6656, 113152, 256, 256, True),
    _UaWorkloadShape("unified_attention-118", 1, 32, 2, 6656, 119808, 256, 256, True),
    _UaWorkloadShape("unified_attention-119", 1, 32, 2, 6656, 126464, 256, 256, True),
    _UaWorkloadShape("unified_attention-120", 1, 32, 2, 6656, 131072, 256, 256, True),
    _UaWorkloadShape("unified_attention-126", 1, 32, 2, 3584, 131072, 256, 256, True),
    _UaWorkloadShape("unified_attention-139", 48, 16, 1, 4, 131584, 256, 256, True),
    _UaWorkloadShape("unified_attention-140", 48, 16, 1, 1, 131584, 256, 256, True),
    _UaWorkloadShape("unified_attention-144", 512, 32, 8, 1, 376, 128, 128, True),
]


@pytest.mark.parametrize("case", _GFX946_UA_CASES, ids=lambda case: case.case_id)
def test_gfx946_workload_fallback_dispatch(case: _UaWorkloadShape, monkeypatch) -> None:
    backend = importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )
    monkeypatch.setattr(backend, "get_arch", lambda device=None: "gfx946")
    result = backend._default_2d_config(
        "cuda", 16, 32,
        cache_block_size=16,
        head_size=case.head_size,
        num_seqs=case.batch,
        num_queries_per_kv=case.query_heads // case.kv_heads,
        max_seqlen_q=case.query_length,
        max_seqlen_k=case.kv_length,
        kv_dtype=torch.float8_e4m3fn,
        plain_attention=True,
    )
    expected_wide_ids = {
        f"unified_attention-{index:03d}"
        for index in (
            [12] + list(range(18, 38)) + [43] + list(range(73, 94))
            + list(range(101, 121)) + [126]
        )
    }
    expected = (
        dict(BLOCK_M=64, TILE_SIZE=64, num_warps=4, num_stages=1)
        if case.case_id in expected_wide_ids
        else dict(BLOCK_M=16, TILE_SIZE=32, num_warps=4, num_stages=2)
    )
    assert result == expected


_SMALL_UA_SHAPE_SCORES = 2 * 10**8
_HUGE_UA_SHAPE_BYTES = 96 * 1024**3
_UA_LARGE_ENABLED = os.environ.get("BOLTOPS_UA_RUN_LARGE", "") not in ("", "0")
_UA_HUGE_ENABLED = os.environ.get("BOLTOPS_UA_RUN_HUGE", "") not in ("", "0")
_ON_PMD = bool(os.environ.get("PMD_PATH"))


def _reference_paged_online(
    q, k_cache, v_cache, q_lens, kv_lens, table, scale, *, tile=512,
):
    """Chunked online-softmax paged attention reference.

    Runs on the device of the inputs (CPU for small shapes, GPU for the
    large ones); the KV dimension is processed in ``tile`` chunks so the
    score matrix is never fully materialized.  Caches must already be
    dequantized (fp8 is not gatherable by aten, so callers pass the
    float-dequantized values).
    """
    total_q, num_q_heads = q.shape[0], q.shape[1]
    v_head = v_cache.shape[-1]
    out = torch.empty(total_q, num_q_heads, v_head, dtype=q.dtype, device=q.device)
    block_size = k_cache.shape[1]
    floor = torch.finfo(torch.float32).min
    cursor = 0
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[seq, : math.ceil(kv_len / block_size)].long()
        k = k_cache[blocks].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:kv_len]
        v = v_cache[blocks].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:kv_len]
        q_seq = q[cursor : cursor + q_len].float()
        group = num_q_heads // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        m = torch.full((num_q_heads, q_len), floor, device=q.device, dtype=torch.float32)
        l = torch.zeros((num_q_heads, q_len), device=q.device, dtype=torch.float32)
        acc = torch.zeros(num_q_heads, q_len, v_head, device=q.device, dtype=torch.float32)
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device, dtype=torch.int64)
        for start in range(0, kv_len, tile):
            stop = min(start + tile, kv_len)
            keys = torch.arange(start, stop, device=q.device, dtype=torch.int64)
            k_tile = k[start:stop].float()
            v_tile = v[start:stop].float()
            s = torch.einsum("qhd,khd->hqk", q_seq, k_tile) * scale
            # Clamp (instead of -inf) so fully masked tiles cannot produce
            # NaN; the garbage they accumulate is zeroed by alpha once the
            # row's real tile arrives.
            s = s.masked_fill(keys[None, None, :] > q_pos[None, :, None], floor)
            m_new = torch.maximum(m, s.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new[:, :, None])
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, :, None] + torch.einsum("hqk,khd->hqd", p, v_tile)
            m = m_new
        out[cursor : cursor + q_len] = (acc / l[:, :, None]).transpose(0, 1).to(q.dtype)
        cursor += q_len
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize("case", _GFX946_UA_CASES, ids=lambda case: case.case_id)
def test_workload_shape_matches_reference(case: _UaWorkloadShape) -> None:
    """Every ready workload shape, causal, matches a chunked reference."""
    if case.cache_bytes > _HUGE_UA_SHAPE_BYTES and not _UA_HUGE_ENABLED:
        pytest.skip(
            f"paged-cache footprint {case.cache_bytes / 1024**3:.0f}GB > 96GB; "
            "set BOLTOPS_UA_RUN_HUGE=1 to run"
        )
    if case.score_count > _SMALL_UA_SHAPE_SCORES and not _UA_LARGE_ENABLED:
        pytest.skip("large shape; set BOLTOPS_UA_RUN_LARGE=1 to run")
    if case.score_count > _SMALL_UA_SHAPE_SCORES and _ON_PMD:
        pytest.skip("perf model cannot compute the on-device reference")

    torch.manual_seed(1)
    device = "cuda"
    q_len, kv_len = case.query_length, case.kv_length
    page = 16
    pages_per_seq = math.ceil(kv_len / page)
    total_pages = case.batch * pages_per_seq
    total_q = case.batch * q_len
    scale = case.head_size ** -0.5

    q = torch.randn(total_q, case.query_heads, case.head_size,
                    device=device, dtype=torch.bfloat16)
    k = torch.randn(total_pages, page, case.kv_heads, case.head_size,
                    device=device, dtype=torch.bfloat16)
    v = torch.randn(total_pages, page, case.kv_heads, case.value_head_size,
                    device=device, dtype=torch.bfloat16)

    k_descale = v_descale = None
    kwargs = {}
    if case.fp8_kv:
        k_descale = torch.tensor(448.0 / k.abs().max().item(), device=device)
        v_descale = torch.tensor(448.0 / v.abs().max().item(), device=device)
        k = (k.float() / k_descale).to(torch.float8_e4m3fn)
        v = (v.float() / v_descale).to(torch.float8_e4m3fn)
        kwargs["kv_quant_mode"] = KVQuantMode.FP8_PER_TENSOR

    out = torch.empty(total_q, case.query_heads, case.value_head_size,
                      device=device, dtype=torch.bfloat16)
    table = torch.arange(total_pages, device=device, dtype=torch.int32).reshape(
        case.batch, pages_per_seq
    )
    unified_attention(
        q, k, v, out,
        torch.arange(0, total_q + 1, q_len, device=device, dtype=torch.int32),
        q_len,
        torch.full((case.batch,), kv_len, device=device, dtype=torch.int64),
        kv_len,
        scale, True, (-1, -1),
        table, 0.0, None, k_descale, v_descale, **kwargs,
    )

    on_device = case.score_count > _SMALL_UA_SHAPE_SCORES
    q_lens = [q_len] * case.batch
    kv_lens = [kv_len] * case.batch
    if on_device:
        # fp8 caches are not gatherable by aten, so the reference consumes
        # the float-dequantized cache; bf16 stays bf16 (cast per tile).
        k_ref = k.float() * k_descale if k_descale is not None else k
        v_ref = v.float() * v_descale if v_descale is not None else v
        reference = _reference_paged_online(
            q, k_ref, v_ref, q_lens, kv_lens, table, scale,
        ).cpu()
    else:
        # Small shapes: reference on CPU copies (works on any arch, incl. the
        # PMD perf model whose GPU rocBLAS has no gfx946 TensileLibrary).
        # Dequantize on device first so the CPU reference sees the exact
        # values the kernel consumes.
        k_ref = k.float().cpu()
        v_ref = v.float().cpu()
        if k_descale is not None:
            k_ref = k_ref * k_descale.cpu()
            v_ref = v_ref * v_descale.cpu()
        reference = _reference_paged_online(
            q.float().cpu(), k_ref, v_ref, q_lens, kv_lens,
            table.cpu(), scale,
        )
    tolerance = 8e-2 if case.fp8_kv else 3e-2
    torch.testing.assert_close(
        out.float().cpu(), reference.float(), atol=tolerance, rtol=tolerance
    )


_BENCH_1102_DTYPES = {'BF16': torch.bfloat16, 'FP16': torch.float16,
                      'FP32': torch.float32, 'FP8E4M3': torch.float8_e4m3fn}


_BENCH_1102_SHAPES = [
    {'id': 's005', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 4096, 256], [1, 2, 4096, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 43}]},
    {'id': 's011', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Remainder', 'inputs': [[1, 16, 2560, 256], [1, 2, 4096, 256], [1, 2, 4096, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 2560, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 105}]},
    {'id': 's016', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'Decode', 'inputs': [[64, 16, 1, 256], [64, 2, 4608, 256], [64, 2, 4608, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[64, 16, 1, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'decode_shape', 'row': 42}]},
    {'id': 's017', 'op': 'gqa_attention_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 6656, 256], [1, 2, 6656, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 39}]},
    {'id': 's018', 'op': 'gqa_attention_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 13312, 256], [1, 2, 13312, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 40}]},
    {'id': 's019', 'op': 'gqa_attention_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 19968, 256], [1, 2, 19968, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 41}]},
    {'id': 's020', 'op': 'gqa_attention_slice3', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 26624, 256], [1, 2, 26624, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 42}]},
    {'id': 's021', 'op': 'gqa_attention_slice4', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 33280, 256], [1, 2, 33280, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 43}]},
    {'id': 's022', 'op': 'gqa_attention_slice5', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 39936, 256], [1, 2, 39936, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 44}]},
    {'id': 's023', 'op': 'gqa_attention_slice6', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 46592, 256], [1, 2, 46592, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 45}]},
    {'id': 's024', 'op': 'gqa_attention_slice7', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 53248, 256], [1, 2, 53248, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 46}]},
    {'id': 's025', 'op': 'gqa_attention_slice8', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 59904, 256], [1, 2, 59904, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 47}]},
    {'id': 's026', 'op': 'gqa_attention_slice9', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 66560, 256], [1, 2, 66560, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 48}]},
    {'id': 's027', 'op': 'gqa_attention_slice10', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 73216, 256], [1, 2, 73216, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 49}]},
    {'id': 's028', 'op': 'gqa_attention_slice11', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 79872, 256], [1, 2, 79872, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 50}]},
    {'id': 's029', 'op': 'gqa_attention_slice12', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 86528, 256], [1, 2, 86528, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 51}]},
    {'id': 's030', 'op': 'gqa_attention_slice13', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 93184, 256], [1, 2, 93184, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 52}]},
    {'id': 's031', 'op': 'gqa_attention_slice14', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 99840, 256], [1, 2, 99840, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 53}]},
    {'id': 's032', 'op': 'gqa_attention_slice15', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 106496, 256], [1, 2, 106496, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 54}]},
    {'id': 's033', 'op': 'gqa_attention_slice16', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 113152, 256], [1, 2, 113152, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 55}]},
    {'id': 's034', 'op': 'gqa_attention_slice17', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 119808, 256], [1, 2, 119808, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 56}]},
    {'id': 's035', 'op': 'gqa_attention_slice18', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 126464, 256], [1, 2, 126464, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 57}]},
    {'id': 's036', 'op': 'gqa_attention_slice19', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 16, 6656, 256], [1, 2, 131072, 256], [1, 2, 131072, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 58}]},
    {'id': 's042', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Remainder', 'inputs': [[1, 16, 2048, 256], [1, 2, 131072, 256], [1, 2, 131072, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 16, 2048, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 114}]},
    {'id': 's047', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'Decode', 'inputs': [[12, 16, 1, 256], [12, 2, 131584, 256], [12, 2, 131584, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[12, 16, 1, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'decode_shape', 'row': 38}]},
    {'id': 's048', 'op': 'mha_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 6656, 192], [4, 64, 6656, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 16}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 53}]},
    {'id': 's049', 'op': 'mha_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 13312, 192], [4, 64, 13312, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 17}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 54}]},
    {'id': 's050', 'op': 'mha_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 19968, 192], [4, 64, 19968, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 18}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 55}]},
    {'id': 's051', 'op': 'mha_slice3', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 26624, 192], [4, 64, 26624, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 19}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 56}]},
    {'id': 's052', 'op': 'mha_slice4', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 33280, 192], [4, 64, 33280, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 20}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 57}]},
    {'id': 's053', 'op': 'mha_slice5', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 39936, 192], [4, 64, 39936, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 21}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 58}]},
    {'id': 's054', 'op': 'mha_slice6', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 46592, 192], [4, 64, 46592, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 22}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 59}]},
    {'id': 's055', 'op': 'mha_slice7', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 53248, 192], [4, 64, 53248, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 23}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 60}]},
    {'id': 's056', 'op': 'mha_slice8', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 59904, 192], [4, 64, 59904, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 24}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 61}]},
    {'id': 's057', 'op': 'mha_slice9', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 66560, 192], [4, 64, 66560, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 25}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 62}]},
    {'id': 's058', 'op': 'mha_slice10', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 73216, 192], [4, 64, 73216, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 26}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 63}]},
    {'id': 's059', 'op': 'mha_slice11', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 79872, 192], [4, 64, 79872, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 27}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 64}]},
    {'id': 's060', 'op': 'mha_slice12', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 86528, 192], [4, 64, 86528, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 28}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 65}]},
    {'id': 's061', 'op': 'mha_slice13', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 93184, 192], [4, 64, 93184, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 29}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 66}]},
    {'id': 's062', 'op': 'mha_slice14', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 99840, 192], [4, 64, 99840, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 30}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 67}]},
    {'id': 's063', 'op': 'mha_slice15', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 106496, 192], [4, 64, 106496, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 31}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 68}]},
    {'id': 's064', 'op': 'mha_slice16', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 113152, 192], [4, 64, 113152, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 32}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 69}]},
    {'id': 's065', 'op': 'mha_slice17', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 119808, 192], [4, 64, 119808, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 33}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 70}]},
    {'id': 's066', 'op': 'mha_slice18', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 6656, 192], [4, 64, 126464, 192], [4, 64, 126464, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 34}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 71}]},
    {'id': 's067', 'op': 'mha_slice19', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[4, 64, 4608, 192], [4, 64, 131072, 192], [4, 64, 131072, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[4, 64, 4608, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 35}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 72}]},
    {'id': 's068', 'op': 'mha_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[32, 64, 6656, 192], [32, 64, 53248, 192], [32, 64, 53248, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[32, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 16}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 36}]},
    {'id': 's069', 'op': 'mha_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[32, 64, 6656, 192], [32, 64, 106496, 192], [32, 64, 106496, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[32, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 17}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 37}]},
    {'id': 's070', 'op': 'mha_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[32, 64, 3072, 192], [32, 64, 131072, 192], [32, 64, 131072, 128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[32, 64, 3072, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 18}, {'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 38}]},
    {'id': 's071', 'op': 'eagle3_paged_attn', 'type': 'mha', 'stage': 'NextN_Decode', 'inputs': [[4, 64, 1, 128], [4, 64, 131072, 128], [4, 64, 131072, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[4, 64, 1, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_09_kimi26_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 66}]},
    {'id': 's072', 'op': 'gqa_attention_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 6656, 128], [1, 8, 6656, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 15}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 53}]},
    {'id': 's073', 'op': 'gqa_attention_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 13312, 128], [1, 8, 13312, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 16}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 54}]},
    {'id': 's074', 'op': 'gqa_attention_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 19968, 128], [1, 8, 19968, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 17}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 55}]},
    {'id': 's075', 'op': 'gqa_attention_slice3', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 26624, 128], [1, 8, 26624, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 18}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 56}]},
    {'id': 's076', 'op': 'gqa_attention_slice4', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 33280, 128], [1, 8, 33280, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 19}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 57}]},
    {'id': 's077', 'op': 'gqa_attention_slice5', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 39936, 128], [1, 8, 39936, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 20}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 58}]},
    {'id': 's078', 'op': 'gqa_attention_slice6', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 46592, 128], [1, 8, 46592, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 21}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 59}]},
    {'id': 's079', 'op': 'gqa_attention_slice7', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 53248, 128], [1, 8, 53248, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 22}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 60}]},
    {'id': 's080', 'op': 'gqa_attention_slice8', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 59904, 128], [1, 8, 59904, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 23}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 61}]},
    {'id': 's081', 'op': 'gqa_attention_slice9', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 66560, 128], [1, 8, 66560, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 24}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 62}]},
    {'id': 's082', 'op': 'gqa_attention_slice10', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 73216, 128], [1, 8, 73216, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 25}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 63}]},
    {'id': 's083', 'op': 'gqa_attention_slice11', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 79872, 128], [1, 8, 79872, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 26}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 64}]},
    {'id': 's084', 'op': 'gqa_attention_slice12', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 86528, 128], [1, 8, 86528, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 27}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 65}]},
    {'id': 's085', 'op': 'gqa_attention_slice13', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 93184, 128], [1, 8, 93184, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 28}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 66}]},
    {'id': 's086', 'op': 'gqa_attention_slice14', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 99840, 128], [1, 8, 99840, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 29}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 67}]},
    {'id': 's087', 'op': 'gqa_attention_slice15', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 106496, 128], [1, 8, 106496, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 30}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 68}]},
    {'id': 's088', 'op': 'gqa_attention_slice16', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 113152, 128], [1, 8, 113152, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 31}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 69}]},
    {'id': 's089', 'op': 'gqa_attention_slice17', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 119808, 128], [1, 8, 119808, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 32}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 70}]},
    {'id': 's090', 'op': 'gqa_attention_slice18', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 126464, 128], [1, 8, 126464, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 33}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 71}]},
    {'id': 's091', 'op': 'gqa_attention_slice19', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 131072, 128], [1, 8, 131072, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 34}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 72}]},
    {'id': 's092', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Remainder', 'inputs': [[1, 64, 512, 128], [1, 8, 131072, 128], [1, 8, 131072, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 512, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 107}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 126}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 75}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 94}]},
    {'id': 's093', 'op': 'gqa_attention_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 53248, 128], [1, 8, 53248, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 16}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 37}]},
    {'id': 's094', 'op': 'gqa_attention_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 106496, 128], [1, 8, 106496, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 17}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 38}]},
    {'id': 's095', 'op': 'gqa_attention_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 64, 6656, 128], [1, 8, 131072, 128], [1, 8, 131072, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 64, 6656, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 18}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 39}]},
    {'id': 's096', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'Decode', 'inputs': [[4, 64, 4, 128], [4, 8, 131584, 128], [4, 8, 131584, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[4, 64, 4, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 15}, {'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 33}]},
    {'id': 's097', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'NextN_Decode', 'inputs': [[4, 64, 1, 128], [4, 8, 131584, 128], [4, 8, 131584, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[4, 64, 1, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_10_hy3_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 70}]},
    {'id': 's103', 'op': 'gqa_attention_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 6656, 256], [1, 2, 6656, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 46}]},
    {'id': 's104', 'op': 'gqa_attention_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 13312, 256], [1, 2, 13312, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 47}]},
    {'id': 's105', 'op': 'gqa_attention_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 19968, 256], [1, 2, 19968, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 48}]},
    {'id': 's106', 'op': 'gqa_attention_slice3', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 26624, 256], [1, 2, 26624, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 49}]},
    {'id': 's107', 'op': 'gqa_attention_slice4', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 33280, 256], [1, 2, 33280, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 50}]},
    {'id': 's108', 'op': 'gqa_attention_slice5', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 39936, 256], [1, 2, 39936, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 51}]},
    {'id': 's109', 'op': 'gqa_attention_slice6', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 46592, 256], [1, 2, 46592, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 52}]},
    {'id': 's110', 'op': 'gqa_attention_slice7', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 53248, 256], [1, 2, 53248, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 53}]},
    {'id': 's111', 'op': 'gqa_attention_slice8', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 59904, 256], [1, 2, 59904, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 54}]},
    {'id': 's112', 'op': 'gqa_attention_slice9', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 66560, 256], [1, 2, 66560, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 55}]},
    {'id': 's113', 'op': 'gqa_attention_slice10', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 73216, 256], [1, 2, 73216, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 56}]},
    {'id': 's114', 'op': 'gqa_attention_slice11', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 79872, 256], [1, 2, 79872, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 57}]},
    {'id': 's115', 'op': 'gqa_attention_slice12', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 86528, 256], [1, 2, 86528, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 58}]},
    {'id': 's116', 'op': 'gqa_attention_slice13', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 93184, 256], [1, 2, 93184, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 59}]},
    {'id': 's117', 'op': 'gqa_attention_slice14', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 99840, 256], [1, 2, 99840, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 60}]},
    {'id': 's118', 'op': 'gqa_attention_slice15', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 106496, 256], [1, 2, 106496, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 61}]},
    {'id': 's119', 'op': 'gqa_attention_slice16', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 113152, 256], [1, 2, 113152, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 62}]},
    {'id': 's120', 'op': 'gqa_attention_slice17', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 119808, 256], [1, 2, 119808, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 63}]},
    {'id': 's121', 'op': 'gqa_attention_slice18', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 126464, 256], [1, 2, 126464, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 64}]},
    {'id': 's122', 'op': 'gqa_attention_slice19', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 131072, 256], [1, 2, 131072, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 65}]},
    {'id': 's128', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Remainder', 'inputs': [[1, 32, 3584, 256], [1, 2, 131072, 256], [1, 2, 131072, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 3584, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 133}, {'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 122}]},
    {'id': 's133', 'op': 'gqa_attention_slice0', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 53248, 256], [1, 2, 53248, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 49}]},
    {'id': 's134', 'op': 'gqa_attention_slice1', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 106496, 256], [1, 2, 106496, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 50}]},
    {'id': 's135', 'op': 'gqa_attention_slice2', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[1, 32, 6656, 256], [1, 2, 131072, 256], [1, 2, 131072, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[1, 32, 6656, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 51}]},
    {'id': 's144', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'Decode', 'inputs': [[48, 16, 4, 256], [48, 1, 131584, 256], [48, 1, 131584, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[48, 16, 4, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 45}]},
    {'id': 's145', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'NextN_Decode', 'inputs': [[48, 16, 1, 256], [48, 1, 131584, 256], [48, 1, 131584, 256]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[48, 16, 1, 256]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 86}]},
    {'id': 's146', 'op': 'vit_bi_attention', 'type': 'mha', 'stage': 'Prefill_Once', 'inputs': [[29, 16, 784, 72], [29, 16, 784, 72], [29, 16, 784, 72]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[29, 16, 784, 72]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_12_qwen3vl8b_226_s150.xlsx', 'sheet': 'prefill_shape', 'row': 10}]},
    {'id': 's147', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Chunk', 'inputs': [[29, 32, 229, 128], [29, 8, 226, 128], [29, 8, 226, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[29, 32, 229, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_12_qwen3vl8b_226_s150.xlsx', 'sheet': 'prefill_shape', 'row': 33}]},
    {'id': 's148', 'op': 'gqa_attention', 'type': 'mha', 'stage': 'Prefill_Remainder', 'inputs': [[11, 32, 232, 128], [11, 8, 226, 128], [11, 8, 226, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[11, 32, 232, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_12_qwen3vl8b_226_s150.xlsx', 'sheet': 'prefill_shape', 'row': 50}]},
    {'id': 's149', 'op': 'gqa_paged_attn', 'type': 'mha', 'stage': 'Decode', 'inputs': [[512, 32, 1, 128], [512, 8, 376, 128], [512, 8, 376, 128]], 'dtypes': ['BF16', 'FP8E4M3', 'FP8E4M3'], 'outputs': [[512, 32, 1, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_12_qwen3vl8b_226_s150.xlsx', 'sheet': 'decode_shape', 'row': 14}]},
]


def _bench_1102_cases():
    cases = []
    for shape in _BENCH_1102_SHAPES:
        routes = ('no_scratch', 'scratch')
        for route in routes:
            cases.append({
                'shape_id': shape['id'], 'source_op': shape['op'], 'route': route,
                'measurement': 'component' if shape['op'] == 'gdn_chunk_delta' else 'operator',
                'input_shapes': json.dumps(shape['inputs'], separators=(',', ':')),
                'input_dtypes': ','.join(shape['dtypes']),
                'output_shapes': json.dumps(shape['outputs'], separators=(',', ':')),
                'output_dtypes': ','.join(shape['output_dtypes']),
                'sources': json.dumps(shape['locations'], separators=(',', ':')),
                'shape': shape,
            })
    return cases


def _bench_1102_case_id(case):
    return f"{case['source_op']}-{case['route']}-{case['shape_id']}"


def _build_bench_1102_runner(case, device='cuda'):
    """Allocate inputs once; return an operator/component invocation for timing."""
    shape = case['shape']
    inputs = shape['inputs']
    dtypes = [_BENCH_1102_DTYPES[name] for name in shape['dtypes']]
    route = case['route']

    def tensor(*dims, dtype=torch.bfloat16, value=None):
        tensor = torch.empty(dims, dtype=dtype, device=device)
        if tensor.device.type != 'meta':
            if value is not None:
                tensor.fill_(value)
            elif dtype == torch.float8_e4m3fn:
                tensor.view(torch.uint8).random_(24, 40)
            else:
                tensor.uniform_(-0.02, 0.02)
        return tensor

    def indices(count):
        return torch.arange(count, dtype=torch.int32, device=device)
    target = importlib.import_module('boltops.fa.unified_attention.triton.unified_attention')
    (batch, heads, query_len, width) = inputs[0]
    (_, groups, key_len, _) = inputs[1]
    value_width = inputs[2][-1]
    page_size = 16
    pages = (key_len + page_size - 1) // page_size
    options = {}
    if route == 'scratch':
        rows = batch * query_len if query_len <= 512 and width == value_width else 1
        padded = target.config_module.head_size_layout(width)[0]
        options.update(seq_threshold_3D=batch, num_par_softmax_segments=16, softmax_segm_output=tensor(rows, heads, 16, padded, dtype=torch.float32), softmax_segm_max=tensor(rows, heads, 16, dtype=torch.float32), softmax_segm_expsum=tensor(rows, heads, 16, dtype=torch.float32))
    return partial(target.unified_attention, q=tensor(batch * query_len, heads, width, dtype=dtypes[0]), k=tensor(batch * pages, page_size, groups, width, dtype=dtypes[1]), v=tensor(batch * pages, page_size, groups, value_width, dtype=dtypes[2]), out=tensor(batch * query_len, heads, value_width), cu_seqlens_q=indices(batch + 1) * query_len, max_seqlen_q=query_len, seqused_k=torch.full((batch,), key_len, dtype=torch.int32, device=device), max_seqlen_k=key_len, softmax_scale=width ** (-0.5), causal=shape['op'] != 'vit_bi_attention', window_size=(-1, -1), softcap=0.0, block_table=indices(batch * pages).reshape(batch, pages), q_descale=None, k_descale=tensor(1, dtype=torch.float32, value=1), v_descale=tensor(1, dtype=torch.float32, value=1), **options)


@pytest.mark.bench
@pytest.mark.parametrize('case', _bench_1102_cases(), ids=_bench_1102_case_id)
def test_bench_1102_unified_attention_triton(case, bench_warmup, bench_repeat, time_ms):
    if not torch.cuda.is_available():
        pytest.skip('CUDA/HIP device required for benchmarks')
    run = _build_bench_1102_runner(case)
    run()
    latency = time_ms(run, bench_warmup, bench_repeat, dtype=case['input_dtypes'])
    print(f'{_bench_1102_case_id(case)} ({case["measurement"]}): {latency * 1000:.3f} us')
