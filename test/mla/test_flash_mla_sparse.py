"""Correctness and API tests for FlashMLA sparse forward."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

pytest.importorskip("triton")

import boltops.mla.triton.flash_mla_sparse as sparse_module

from boltops.mla import flash_mla_sparse_fwd
from boltops.mla.triton.flash_mla_sparse import (
    _config_match_score,
    _device_model,
    _supports_matrix_load,
)


GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    d_v: int,
    topk_length: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, _ = q.shape
    h_kv = kv.shape[1]
    group_size = h_q // h_kv
    out = torch.zeros(s_q, h_q, d_v, device=q.device, dtype=torch.float32)
    max_logits = torch.full(
        (s_q, h_q), -torch.inf, device=q.device, dtype=torch.float32
    )
    lse = torch.full_like(max_logits, -torch.inf)
    for row in range(s_q):
        length = indices.shape[-1] if topk_length is None else int(topk_length[row])
        for head in range(h_q):
            kv_head = head // group_size
            selected = indices[row, kv_head, : max(0, length)].long()
            selected = selected[(selected >= 0) & (selected < kv.shape[0])]
            if selected.numel() == 0:
                continue
            keys = kv[selected, kv_head].float()
            scores = (q[row, head].float() @ keys.T) * scale
            max_logits[row, head] = scores.max()
            lse[row, head] = torch.logsumexp(scores, dim=0)
            out[row, head] = torch.softmax(scores, dim=0) @ keys[:, :d_v]
    return out, max_logits, lse


@GPU_REQUIRED
@pytest.mark.parametrize("use_lengths", [False, True])
def test_flash_mla_sparse_fwd_matches_reference(use_lengths: bool):
    torch.manual_seed(0)
    s_q, s_kv, h_q, h_kv, d_qk, d_v, topk = 4, 80, 16, 1, 576, 512, 33
    q = torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_kv, h_kv, d_qk, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    indices[0, 0, 0] = -1
    indices[1, 0, 1] = s_kv + 1
    lengths = (
        torch.tensor([0, 7, topk, 19], device="cuda", dtype=torch.int32)
        if use_lengths
        else None
    )
    scale = 1.0 / math.sqrt(d_qk)

    out, max_logits, lse = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        softmax_scale=scale,
        d_v=d_v,
        topk_length=lengths,
    )
    ref_out, ref_max, ref_lse = _reference(q, kv, indices, scale, d_v, lengths)

    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(max_logits, ref_max, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-2)


def test_flash_mla_sparse_fwd_validates_shapes_before_launch():
    q = torch.empty(2, 16, 576, dtype=torch.bfloat16)
    kv = torch.empty(8, 1, 576, dtype=torch.bfloat16)
    indices = torch.empty(3, 1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="indices must have shape"):
        flash_mla_sparse_fwd(q, kv, indices, 1.0)


def test_flash_mla_sparse_fwd_is_public():
    from boltops.mla import __all__

    assert "flash_mla_sparse_fwd" in __all__


def test_flash_mla_sparse_tuned_config_schema():
    config_root = (
        Path(__file__).parents[2]
        / "boltops"
        / "mla"
        / "triton"
        / "configs"
    )
    paths = list(config_root.glob("triton_*_*/flash_mla_sparse_*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data["metadata"]
        assert metadata["arch"] in path.name
        device_model = metadata["device_name"].split(",", 1)[0]
        assert device_model in path.name
        matches = [tuple(sorted(entry["match"].items())) for entry in data["configs"]]
        assert len(matches) == len(set(matches))
        for entry in data["configs"]:
            assert "dtype" in entry["match"]
            if "latency_ms" in entry or "speedup" in entry:
                assert entry["latency_ms"] > 0
                assert entry["speedup"] > 0
            else:
                assert entry["evidence"]["kind"] == "static_range_materialization"
            if metadata["device_name"] == "K100_AI":
                assert "SPLIT_KV" in entry["config"]
                assert "SORT_INDICES" in entry["config"]
                assert "SPLIT_KV_POLICY" not in entry["config"]
                assert "ONE_TILE_BLOCK_DMODEL" not in entry["config"]


def test_tuned_config_range_matching_and_specificity():
    shape = {"s_q": 4, "topk": 2048, "dtype": "bfloat16"}
    assert _config_match_score({"s_q_max": 8, "topk": 2048}, shape) is not None
    assert _config_match_score({"s_q_min": 9, "topk": 2048}, shape) is None
    range_score = _config_match_score({"s_q_max": 8, "topk": 2048}, shape)
    exact_score = _config_match_score({"s_q": 4, "topk": 2048}, shape)
    assert exact_score is not None and range_score is not None
    assert exact_score > range_score


@pytest.mark.parametrize(
    ("s_q", "h_q", "topk", "split_kv", "block_dmodel"),
    [
        (1, 16, 128, 8, 512),
        (29, 16, 128, 8, 512),
        (30, 16, 128, 4, 256),
        (1, 16, 512, 32, 512),
        (7, 16, 512, 32, 512),
        (8, 16, 512, 16, 256),
        (24, 16, 16384, 8, 256),
        (48, 16, 16384, 4, 256),
        (49, 16, 16384, 2, 256),
        (90, 16, 16384, 2, 256),
        (91, 16, 16384, 1, 256),
        (16, 48, 16384, 4, 256),
        (96, 16, 16384, 1, 256),
    ],
)
def test_k100_static_tuned_configs(
    s_q: int, h_q: int, topk: int, split_kv: int, block_dmodel: int
):
    path = (
        Path(__file__).parents[2]
        / "boltops"
        / "mla"
        / "triton"
        / "configs"
        / "triton_3_6"
        / "flash_mla_sparse_gfx928_K100_AI.json"
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["configs"]
    shape = {
        "s_q": s_q,
        "s_kv": 131072,
        "h_q": h_q,
        "h_kv": 1,
        "topk": topk,
        "d_qk": 576,
        "d_v": 512,
        "dtype": "bfloat16",
    }
    matches = [
        (_config_match_score(entry["match"], shape), entry["config"])
        for entry in entries
    ]
    selected = max(
        (item for item in matches if item[0] is not None), key=lambda item: item[0]
    )
    assert selected[1]["SPLIT_KV"] == split_kv
    assert selected[1]["BLOCK_DMODEL"] == block_dmodel


@pytest.mark.parametrize(
    ("s_q", "h_q", "topk", "sort_indices"),
    [
        (119, 16, 512, False),
        (120, 16, 512, True),
        (71, 16, 1024, False),
        (72, 16, 1024, True),
        (90, 16, 1024, True),
        (91, 16, 1024, False),
        (103, 16, 1024, False),
        (104, 16, 1024, True),
        (29, 16, 2048, False),
        (30, 16, 2048, True),
        (14, 16, 16384, False),
        (15, 16, 16384, True),
        (16, 48, 16384, False),
    ],
)
def test_k100_static_sort_indices_buckets(
    s_q: int, h_q: int, topk: int, sort_indices: bool
):
    path = (
        Path(__file__).parents[2]
        / "boltops"
        / "mla"
        / "triton"
        / "configs"
        / "triton_3_6"
        / "flash_mla_sparse_gfx928_K100_AI.json"
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["configs"]
    shape = {
        "s_q": s_q,
        "s_kv": 131072,
        "h_q": h_q,
        "h_kv": 1,
        "topk": topk,
        "d_qk": 576,
        "d_v": 512,
        "dtype": "bfloat16",
    }
    matches = [
        (_config_match_score(entry["match"], shape), entry["config"])
        for entry in entries
    ]
    selected = max(
        (item for item in matches if item[0] is not None), key=lambda item: item[0]
    )
    assert selected[1]["SORT_INDICES"] is sort_indices


def test_device_model_excludes_platform_description():
    assert _device_model("BW200, UBB BW1000") == "BW200"
    assert _device_model("Product Name") == "Product_Name"


def test_missing_tuned_config_warns_once(monkeypatch, tmp_path):
    sparse_module._load_device_configs.cache_clear()
    sparse_module._warn_config_once.cache_clear()
    monkeypatch.setattr(sparse_module, "_CONFIG_ROOT", tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sparse_module._load_device_configs("triton_3_6", "gfx928", "Missing")
        sparse_module._load_device_configs("triton_3_6", "gfx928", "Missing")
    assert len(caught) == 1
    assert "file is missing" in str(caught[0].message)


def test_unmatched_tuned_config_warns_once(monkeypatch):
    sparse_module._warn_config_once.cache_clear()
    monkeypatch.setattr(
        sparse_module,
        "_device_identity",
        lambda _device: ("gfx928", "K100_AI"),
    )
    monkeypatch.setattr(
        sparse_module,
        "_load_device_configs",
        lambda *_args: {
            "configs": [
                {"match": {"topk": 1, "dtype": "bfloat16"}, "config": {}}
            ]
        },
    )
    q = SimpleNamespace(shape=(2, 16, 576), dtype="bfloat16", device=None)
    kv = SimpleNamespace(shape=(32, 1, 576))
    indices = SimpleNamespace(shape=(2, 1, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sparse_module._get_tuned_config(q, kv, indices, 512)
        sparse_module._get_tuned_config(q, kv, indices, 512)
    assert len(caught) == 1
    assert "does not match the current shape" in str(caught[0].message)


@GPU_REQUIRED
def test_flash_mla_sparse_matrix_load_q_matches_reference():
    if not _supports_matrix_load(torch.device("cuda")):
        pytest.skip("matrix_load is not supported on this device")
    torch.manual_seed(7)
    s_q, s_kv, h_q, h_kv, d_qk, d_v, topk = 2, 80, 16, 1, 576, 512, 33
    q = torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_kv, h_kv, d_qk, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(d_qk)
    config = {
        "BLOCK_N": 16,
        "BLOCK_H": 16,
        "BLOCK_DMODEL": 64,
        "num_warps": 4,
        "num_stages": 1,
        "USE_MATRIX_LOAD_Q": True,
    }
    actual = flash_mla_sparse_fwd(
        q, kv, indices, scale, d_v=d_v, config=config
    )
    expected = _reference(q, kv, indices, scale, d_v, None)
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs.float(), rhs.float(), atol=2e-2, rtol=2e-2)


@GPU_REQUIRED
def test_flash_mla_sparse_sorted_indices_preserve_active_prefix():
    torch.manual_seed(11)
    s_q, s_kv, h_q, h_kv, d_qk, d_v, topk = 4, 80, 16, 1, 576, 512, 33
    q = torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_kv, h_kv, d_qk, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    indices[1, 0, 1] = -1
    indices[2, 0, 2] = s_kv + 1
    lengths = torch.tensor([0, 7, topk, 19], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d_qk)
    config = {
        "BLOCK_N": 16,
        "BLOCK_H": 16,
        "BLOCK_DMODEL": 256,
        "num_warps": 4,
        "num_stages": 2,
        "SORT_INDICES": True,
    }
    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        scale,
        d_v=d_v,
        topk_length=lengths,
        config=config,
    )
    expected = _reference(q, kv, indices, scale, d_v, lengths)
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs.float(), rhs.float(), atol=2e-2, rtol=2e-2)
