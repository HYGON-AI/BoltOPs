"""Runtime configuration lookup for TriangleAttention."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import triton

from boltops.utility.arch_info import get_device_info
from boltops.utility.triton_capability import (
    get_compatible_triton_config_dirs,
    get_triton_capabilities,
    get_triton_config_dir,
)

CONFIG_ROOT = Path(__file__).with_name("configs")
CONFIG_KEY_FIELDS = ("batch", "heads", "seq_len", "head_dim", "mask_layout")
DEFAULT_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}


def make_config_key(
    *, batch: int, heads: int, seq_len: int, head_dim: int, mask_layout: str
):
    dimensions = (int(batch), int(heads), int(seq_len), int(head_dim))
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"TriangleAttention config key must be positive: {dimensions}")
    if mask_layout not in ("contiguous", "strided"):
        raise ValueError(f"Unsupported mask layout: {mask_layout}")
    key = (*dimensions, mask_layout)
    return key


def make_config_filename(*, device_label: str) -> str:
    """Build the single per-device config filename."""
    return f"triangle_attention-device={device_label}.json"


def _validate(config: object) -> dict | None:
    if not isinstance(config, dict):
        return None
    required = set(DEFAULT_CONFIG)
    if set(config) != required:
        return None
    try:
        result = {key: int(value) for key, value in config.items()}
    except (TypeError, ValueError):
        return None
    if (any(type(config[key]) is not int for key in required)
            or result["BLOCK_M"] < 16 or result["BLOCK_N"] < 16
            or result["num_warps"] not in (1, 2, 4, 8)
            or result["num_stages"] not in (1, 2, 3, 4)) :
        return None
    if result["BLOCK_M"] & (result["BLOCK_M"] - 1) or result["BLOCK_N"] & (result["BLOCK_N"] - 1):
        return None
    return result


def get_write_path(filename: str) -> Path:
    return get_triton_config_dir(CONFIG_ROOT, get_triton_capabilities()) / filename


@lru_cache(maxsize=256)
def get_kernel_config(
    *, batch: int, heads: int, seq_len: int, head_dim: int,
    mask_layout: str, device=None,
) -> dict:
    info = get_device_info(device)
    key = make_config_key(
        batch=batch, heads=heads, seq_len=seq_len,
        head_dim=head_dim, mask_layout=mask_layout,
    )
    filename = make_config_filename(device_label=f"{info.arch}_cu{info.cu_count}")
    for directory in get_compatible_triton_config_dirs(CONFIG_ROOT, get_triton_capabilities()):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            if tuple(payload.get("key", ())) != CONFIG_KEY_FIELDS:
                continue
            configs = payload.get("config", {})
            if not isinstance(configs, dict):
                continue
            candidate = configs.get(str(key))
            validated = _validate(candidate)
            if validated is not None:
                return validated
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return dict(DEFAULT_CONFIG)


def clear_config_cache() -> None:
    get_kernel_config.cache_clear()


__all__ = ["CONFIG_ROOT", "CONFIG_KEY_FIELDS", "DEFAULT_CONFIG", "clear_config_cache", "get_kernel_config", "get_write_path", "make_config_filename", "make_config_key"]
