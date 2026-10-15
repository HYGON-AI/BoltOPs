"""ABI, routing, and correctness coverage for UnifiedAttention."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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
