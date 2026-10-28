# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import json

import pytest
import torch

from boltops.fla.kda.triton import config_utils
from boltops.fla.kda.triton.chunk import (
    get_chunk_gla_fwd_o_config,
    get_kda_gate_chunk_cumsum_vector_config,
    get_kda_gate_fwd_config,
    get_recompute_w_u_fwd_config,
)


def _write_config(root, kernel, arch, payload):
    path = root / kernel / f"{kernel}-{arch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_exact_key_is_merged_over_fallback(tmp_path, monkeypatch):
    _write_config(tmp_path, "demo", "gfx936", {
        "config": {"B=1,dtype=bf16,T=128": {"BV": 64, "num_warps": 8}},
    })
    monkeypatch.setattr(config_utils, "_CONFIG_ROOT", tmp_path)
    config_utils._load_config.cache_clear()
    cfg = config_utils.get_kda_tuned_config(
        "demo", "B=1,dtype=bf16,T=128",
        fallback={"BV": 32, "num_warps": 4, "num_stages": 2},
        arch="gfx936",
    )
    assert cfg == {"BV": 64, "num_warps": 8, "num_stages": 2}


def test_missing_key_or_file_uses_fallback(tmp_path, monkeypatch):
    _write_config(tmp_path, "demo", "gfx936", {"config": {}})
    monkeypatch.setattr(config_utils, "_CONFIG_ROOT", tmp_path)
    config_utils._load_config.cache_clear()
    fallback = {"BV": 32, "num_warps": 4}
    assert config_utils.get_kda_tuned_config(
        "demo", "missing", fallback=fallback, arch="gfx936"
    ) is fallback
    assert config_utils.get_kda_tuned_config(
        "absent", "missing", fallback=fallback, arch="gfx936"
    ) is fallback


def test_malformed_json_uses_fallback(tmp_path, monkeypatch):
    path = tmp_path / "demo" / "demo-gfx936.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    monkeypatch.setattr(config_utils, "_CONFIG_ROOT", tmp_path)
    config_utils._load_config.cache_clear()
    fallback = {"BV": 32}
    assert config_utils.get_kda_tuned_config(
        "demo", "any", fallback=fallback, arch="gfx936"
    ) is fallback


def test_unknown_key_rejects_whole_entry(tmp_path, monkeypatch):
    _write_config(tmp_path, "demo", "gfx936", {
        "config": {"key": {"BV": 64, "num_warp": 8}},
    })
    monkeypatch.setattr(config_utils, "_CONFIG_ROOT", tmp_path)
    config_utils._load_config.cache_clear()
    fallback = {"BV": 32, "num_warps": 4}
    assert config_utils.get_kda_tuned_config(
        "demo", "key", fallback=fallback, arch="gfx936"
    ) is fallback


@pytest.mark.parametrize(("function", "expected"), [
    (
        lambda get: get_recompute_w_u_fwd_config(
            128, 8, 128, 256, False, b=2, dtype=torch.bfloat16
        ),
        "recompute_w_u_fwd",
    ),
    (
        lambda get: get_chunk_gla_fwd_o_config(
            128, 8, 128, 256, False, b=2, dtype=torch.bfloat16
        ),
        "chunk_gla_fwd_o_gk",
    ),
    (
        lambda get: get_kda_gate_chunk_cumsum_vector_config(
            128, 8, 128, False, b=2, dtype=torch.bfloat16
        ),
        "kda_gate_chunk_cumsum",
    ),
    (
        lambda get: get_kda_gate_fwd_config(
            128, 8, 128, b=2, dtype=torch.bfloat16
        ),
        "kda_gate_fwd",
    ),
])
def test_prefill_selectors_use_b_and_dtype_in_exact_key(
    monkeypatch, function, expected
):
    seen = []

    def fake_get(kernel, key, *, fallback, arch=None):
        seen.append((kernel, key))
        return dict(fallback, tuned=True)

    monkeypatch.setattr(
        "boltops.fla.kda.triton.chunk.get_kda_tuned_config", fake_get
    )
    cfg = function(fake_get)
    assert cfg["tuned"] is True
    assert seen == [(
        expected,
        "B=2,dtype=bf16,T=128,H=8,K=128,V=256,varlen=False"
        if expected in {"recompute_w_u_fwd", "chunk_gla_fwd_o_gk"}
        else
        "B=2,dtype=bf16,T=128,H=8,S=128,varlen=False"
        if expected == "kda_gate_chunk_cumsum"
        else "B=2,dtype=bf16,T=128,H=8"
    )]
