# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Persistent tuning configuration helpers for Sparse-GQA."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import torch

from boltops.utility import logger
from boltops.utility.arch_info import get_arch, get_num_sms
from boltops.utility.triton_capability import (
    get_compatible_triton_config_dirs,
    get_triton_capabilities,
    get_triton_config_dir,
)


CONFIG_ROOT = Path(__file__).with_name("configs")
CONFIG_DIR = get_triton_config_dir(CONFIG_ROOT, get_triton_capabilities())
_KIND_REQUIRED_KEYS = {
    "compact": ("BLOCK_TOPK", "num_warps", "num_stages"),
    "decode": ("BLOCK_N", "num_warps", "num_stages"),
}
_KIND_CONFIG_STEMS = {
    "compact": "qwen_sparse_kv_compact",
    "decode": "sparse_gqa_packed_decode",
}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


_DECODE_OPTIONAL_KEYS = {
    "SPLITS": _is_positive_int,
    "MERGE_BLOCK_D": _is_positive_int,
    "MERGE_NUM_WARPS": _is_positive_int,
}

_COMPACT_OPTIONAL_KEYS = {
    "GROUPED_V": lambda value: isinstance(value, bool),
    "TOKEN_MAJOR": lambda value: isinstance(value, bool),
    "PACKED_V": lambda value: isinstance(value, bool),
}


def _selector(value: object) -> str:
    """Return a stable filename/key selector value."""
    normalized = str(value).replace("torch.", "")
    return re.sub(r"[^a-zA-Z0-9_.+-]+", "_", normalized).strip("_").lower()


def config_filename(
    kind: str,
    *,
    arch: Optional[str] = None,
    cu_count: Optional[int] = None,
) -> str:
    """Return the per-kernel, per-architecture-and-CU config filename."""
    if kind not in _KIND_REQUIRED_KEYS:
        raise ValueError(f"unknown Sparse-GQA config kind: {kind!r}")
    resolved_cu_count = get_num_sms() if cu_count is None else int(cu_count)
    if resolved_cu_count <= 0:
        raise ValueError(f"cu_count must be positive, got {resolved_cu_count}")
    return (
        f"{_KIND_CONFIG_STEMS[kind]}-{_selector(arch or get_arch())}"
        f"-cu{resolved_cu_count}.json"
    )


def legacy_config_filename(kind: str, *, arch: Optional[str] = None) -> str:
    """Return the pre-CU filename retained as a compatibility fallback."""
    if kind not in _KIND_REQUIRED_KEYS:
        raise ValueError(f"unknown Sparse-GQA config kind: {kind!r}")
    return f"{_KIND_CONFIG_STEMS[kind]}-{_selector(arch or get_arch())}.json"


def config_path(
    kind: str,
    *,
    config_dir: Optional[Path] = None,
) -> Path:
    """Return the current-capability path for a Sparse-GQA config family."""
    directory = CONFIG_DIR if config_dir is None else Path(config_dir)
    return directory / config_filename(kind)


def config_match(
    kind: str,
    batch: int,
    topk: int,
    hq: int,
    hkv: int,
    dim: int,
    dtype: Union[torch.dtype, str],
    *,
    page_size: int = 64,
) -> dict:
    """Build the exact runtime selectors stored inside the short JSON file."""
    if kind not in _KIND_REQUIRED_KEYS:
        raise ValueError(f"unknown Sparse-GQA config kind: {kind!r}")
    match = {
        "batch": batch,
        "topk": topk,
        "hkv": hkv,
        "head_dim": dim,
        "dtype": _selector(dtype),
    }
    if kind == "compact":
        match["page_size"] = page_size
    else:
        match["hq"] = hq
    return match


def _validate_config(kind: str, config: object, path: Path, batch: int) -> dict:
    if not isinstance(config, dict):
        logger.warning(
            "Invalid Sparse-GQA config for batch %d at %s: expected object.",
            batch,
            path,
        )
        return {}
    required = _KIND_REQUIRED_KEYS[kind]
    for key in required:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            logger.warning(
                "Invalid Sparse-GQA config for batch %d at %s: "
                "%s must be a positive integer.",
                batch,
                path,
                key,
            )
            return {}
    if kind in ("decode", "compact"):
        optional = (
            _DECODE_OPTIONAL_KEYS if kind == "decode" else _COMPACT_OPTIONAL_KEYS
        )
        for key, validator in optional.items():
            if key in config and not validator(config[key]):
                logger.warning(
                    "Invalid Sparse-GQA config for batch %d at %s: "
                    "%s has invalid value %r.",
                    batch,
                    path,
                    key,
                    config[key],
                )
                return {}
    return config


def _load_matching_config(
    path: Path,
    kind: str,
    match: dict,
) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Invalid Sparse-GQA config JSON at %s (%s); trying fallback.",
            path,
            exc,
        )
        return None
    if not isinstance(payload, dict) or not isinstance(
        payload.get("configs"), list
    ):
        logger.warning(
            "Invalid Sparse-GQA config schema at %s: expected configs list; "
            "trying fallback.",
            path,
        )
        return None
    for entry in payload["configs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("match"), dict):
            logger.warning(
                "Ignoring invalid Sparse-GQA config entry at %s.",
                path,
            )
            continue
        if entry["match"] == match:
            return _validate_config(
                kind,
                entry.get("config"),
                path,
                int(match["batch"]),
            )
    return {}


@lru_cache(maxsize=None)
def load_config(
    kind: str,
    batch: int,
    topk: int,
    hq: int,
    hkv: int,
    dim: int,
    dtype: Union[torch.dtype, str],
    *,
    page_size: int = 64,
) -> dict:
    """Load one exact-shape config with Triton capability fallback."""
    filename = config_filename(kind)
    legacy_filename = legacy_config_filename(kind)
    match = config_match(
        kind,
        batch,
        topk,
        hq,
        hkv,
        dim,
        dtype,
        page_size=page_size,
    )
    config_dirs = get_compatible_triton_config_dirs(
        CONFIG_ROOT, get_triton_capabilities()
    )
    paths = [directory / filename for directory in config_dirs]
    paths.extend(directory / legacy_filename for directory in config_dirs)
    for path in paths:
        if not path.is_file():
            continue
        config = _load_matching_config(path, kind, match)
        if config:
            return config
        if config is None:
            continue
        logger.warning(
            "Sparse-GQA %s config at %s has no valid exact match for %s; "
            "trying fallback.",
            kind,
            path,
            match,
        )
    logger.warning(
        "Using default Sparse-GQA %s config; no valid exact match for %s at %s.",
        kind,
        match,
        config_dirs[0] / filename,
    )
    return {}
