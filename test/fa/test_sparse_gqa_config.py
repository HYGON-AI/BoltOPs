"""Tests for Sparse-GQA capability-versioned runtime configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boltops.fa.sparse_gqa.triton import config as sparse_gqa_config
from boltops.utility.triton_capability import TritonCapabilities


def _caps(version: str) -> TritonCapabilities:
    return TritonCapabilities(version, frozenset(), "hcu", version)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    sparse_gqa_config.load_config.cache_clear()
    yield
    sparse_gqa_config.load_config.cache_clear()


def _use_config_root(monkeypatch, tmp_path: Path, version: str) -> Path:
    root = tmp_path / "configs"
    monkeypatch.setattr(sparse_gqa_config, "CONFIG_ROOT", root)
    monkeypatch.setattr(
        sparse_gqa_config,
        "get_triton_capabilities",
        lambda: _caps(version),
    )
    monkeypatch.setattr(sparse_gqa_config, "get_arch", lambda: "gfx938")
    monkeypatch.setattr(sparse_gqa_config, "get_num_sms", lambda: 64)
    return root


def _write_configs(
    root: Path, version: str, filename: str, entries: list[dict]
) -> None:
    directory = root / f"triton_{version.replace('.', '_')}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps({"configs": entries}),
        encoding="utf-8",
    )


def _entry(kind: str, batch: int, hq: int, dtype: str, config: dict) -> dict:
    return {
        "match": sparse_gqa_config.config_match(
            kind, batch, 2051, hq, 1, 256, dtype
        ),
        "config": config,
    }


def test_config_filenames_identify_kernel_and_arch() -> None:
    assert sparse_gqa_config.config_filename(
        "compact", arch="gfx938", cu_count=64
    ) == "qwen_sparse_kv_compact-gfx938-cu64.json"
    assert sparse_gqa_config.config_filename(
        "decode", arch="gfx938", cu_count=72
    ) == "sparse_gqa_packed_decode-gfx938-cu72.json"
    assert sparse_gqa_config.legacy_config_filename(
        "compact", arch="gfx938"
    ) == "qwen_sparse_kv_compact-gfx938.json"
    with pytest.raises(ValueError, match="cu_count must be positive"):
        sparse_gqa_config.config_filename("decode", arch="gfx938", cu_count=0)


def test_config_prefers_current_capability_then_falls_back(
    monkeypatch, tmp_path: Path
) -> None:
    root = _use_config_root(monkeypatch, tmp_path, "3.6")
    filename = sparse_gqa_config.config_filename("decode")
    lower = {"BLOCK_N": 16, "num_warps": 4, "num_stages": 1}
    current = {"BLOCK_N": 32, "num_warps": 8, "num_stages": 2}
    _write_configs(root, "3.2", filename, [_entry("decode", 8, 3, "bfloat16", lower)])

    assert sparse_gqa_config.load_config(
        "decode", 8, 2051, 3, 1, 256, "bfloat16"
    ) == lower

    _write_configs(
        root,
        "3.6",
        filename,
        [_entry("decode", 8, 3, "bfloat16", current)],
    )
    sparse_gqa_config.load_config.cache_clear()
    assert sparse_gqa_config.load_config(
        "decode", 8, 2051, 3, 1, 256, "bfloat16"
    ) == current


def test_cu_specific_config_precedes_legacy_file(
    monkeypatch, tmp_path: Path
) -> None:
    root = _use_config_root(monkeypatch, tmp_path, "3.6")
    exact_filename = sparse_gqa_config.config_filename("decode")
    legacy_filename = sparse_gqa_config.legacy_config_filename("decode")
    exact = {"BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
    legacy = {"BLOCK_N": 32, "num_warps": 8, "num_stages": 2}
    entries = [_entry("decode", 8, 3, "bfloat16", exact)]
    _write_configs(root, "3.6", exact_filename, entries)
    _write_configs(
        root,
        "3.6",
        legacy_filename,
        [_entry("decode", 8, 3, "bfloat16", legacy)],
    )

    assert sparse_gqa_config.load_config(
        "decode", 8, 2051, 3, 1, 256, "bfloat16"
    ) == exact


def test_legacy_arch_file_is_used_when_cu_file_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    root = _use_config_root(monkeypatch, tmp_path, "3.6")
    legacy_filename = sparse_gqa_config.legacy_config_filename("compact")
    legacy = {"BLOCK_TOPK": 16, "num_warps": 8, "num_stages": 1}
    _write_configs(
        root,
        "3.6",
        legacy_filename,
        [_entry("compact", 8, 0, "float8_e4m3fn_to_bfloat16", legacy)],
    )

    assert sparse_gqa_config.load_config(
        "compact",
        8,
        2051,
        0,
        1,
        256,
        "float8_e4m3fn_to_bfloat16",
    ) == legacy


def test_invalid_current_config_falls_back_to_lower_capability(
    monkeypatch, tmp_path: Path
) -> None:
    root = _use_config_root(monkeypatch, tmp_path, "3.6")
    filename = sparse_gqa_config.config_filename("compact")
    valid = {"BLOCK_TOPK": 8, "num_warps": 2, "num_stages": 1}
    _write_configs(
        root,
        "3.6",
        filename,
        [
            _entry(
                "compact",
                8,
                0,
                "float8_e4m3fn_to_bfloat16",
                {"BLOCK_TOPK": 0},
            )
        ],
    )
    _write_configs(
        root,
        "3.2",
        filename,
        [_entry("compact", 8, 0, "float8_e4m3fn_to_bfloat16", valid)],
    )

    assert sparse_gqa_config.load_config(
        "compact",
        8,
        2051,
        0,
        1,
        256,
        "float8_e4m3fn_to_bfloat16",
    ) == valid


def test_config_requires_an_exact_shape_match(monkeypatch, tmp_path: Path) -> None:
    root = _use_config_root(monkeypatch, tmp_path, "3.6")
    filename = sparse_gqa_config.config_filename("decode")
    config = {"BLOCK_N": 32, "num_warps": 8, "num_stages": 2}
    _write_configs(
        root,
        "3.6",
        filename,
        [_entry("decode", 8, 3, "bfloat16", config)],
    )

    assert sparse_gqa_config.load_config(
        "decode", 7, 2051, 3, 1, 256, "bfloat16"
    ) == {}
