# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Configuration lookup for the optimized Extend Attention kernels."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from boltops.utility import logger
from boltops.utility.arch_info import get_device_info
from boltops.utility.triton_capability import (
    get_compatible_triton_config_dirs,
    get_triton_capabilities,
    get_triton_config_dir,
)


CONFIG_ROOT = Path(__file__).with_name("configs")
CONFIG_DIR = get_triton_config_dir(CONFIG_ROOT, get_triton_capabilities())

LONG_QUERY = "long_query"
SHORT_QUERY = "short_query"

_CONFIG_LAYOUT = {
    LONG_QUERY: "long_query",
    SHORT_QUERY: "short_query",
}

_DEFAULT_CONFIGS = {
    LONG_QUERY: {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "waves_per_eu": 1,
        "schedule_hint": "attention",
        "matrix_instr_nonkdim": 16,
        "sched_latency": "mmac5-ds10",
        "kpack": 2,
        "USE_MLS": False,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    },
    SHORT_QUERY: {
        "BLOCK_M": 16,
        "BLOCK_N": 32,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "num_warps": 4,
        "num_stages": 1,
        "USE_MLS": False,
    },
}


def _parse_key(value: str) -> tuple:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple):
        raise ValueError(f"Extend Attention config key must be a tuple: {value!r}")
    return parsed


def _file_selectors(path: Path) -> tuple[str, ...]:
    selectors = []
    for component in path.stem.split("-")[2:]:
        name, separator, value = component.partition("=")
        if not separator:
            raise ValueError(f"Invalid Extend Attention config filename: {path.name}")
        selectors.append(f"{name}={value}")
    return tuple(selectors)


@lru_cache(maxsize=None)
def _load_configs(kind: str, device_label: str) -> dict:
    try:
        filename_stem = _CONFIG_LAYOUT[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown Extend Attention config kind: {kind!r}") from exc

    pattern = f"{filename_stem}-device={device_label}*.json"
    tables = {}
    config_dirs = get_compatible_triton_config_dirs(
        CONFIG_ROOT, get_triton_capabilities()
    )
    for config_dir in config_dirs:
        for path in sorted(config_dir.glob(pattern)):
            selectors = _file_selectors(path)
            if selectors in tables:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                configs = payload["config"]
                if not isinstance(configs, dict):
                    raise TypeError("config must be an object")
                grouped = defaultdict(list)
                for key_string in configs:
                    key = _parse_key(key_string)
                    if len(key) < 2:
                        raise ValueError(
                            f"config key has fewer than two grid fields: {key}"
                        )
                    grouped[key[2:]].append((key[:2], key_string))
                tables[selectors] = {
                    "path": path,
                    "configs": configs,
                    "grouped": grouped,
                }
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                SyntaxError,
                json.JSONDecodeError,
            ) as exc:
                logger.warning(
                    "Ignoring invalid Extend Attention config %s: %s", path, exc
                )
    return tables


def _closest_grid(candidates: Iterable[tuple], target: tuple[int, int]) -> str:
    def score(item):
        grid, _ = item
        distance = sum(abs(grid[index] - target[index]) for index in range(2))
        smaller_dimensions = sum(grid[index] < target[index] for index in range(2))
        return distance, smaller_dimensions

    return min(candidates, key=score)[1]


@lru_cache(maxsize=1024)
def get_kernel_config(
    kind: str,
    file_key: tuple[str, ...],
    config_key: tuple,
    *,
    device=None,
) -> dict:
    """Return an exact or nearest-grid config for one optimized kernel."""
    info = get_device_info(device)
    device_label = f"{info.arch}_cu{info.cu_count}"
    table = _load_configs(kind, device_label).get(file_key)
    default = dict(_DEFAULT_CONFIGS[kind])
    if table is None:
        logger.warning(
            "No %s Extend Attention config for %s and %s; using defaults.",
            kind,
            device_label,
            file_key,
        )
        return default

    configs = table["configs"]
    exact_key = str(config_key)
    if exact_key in configs:
        return dict(configs[exact_key])

    grid = config_key[:2]
    shape_key = config_key[2:]
    candidates = table["grouped"].get(shape_key)
    if not candidates:
        logger.warning(
            "No %s Extend Attention config for key %s in %s; using defaults.",
            kind,
            shape_key,
            table["path"],
        )
        return default

    selected = _closest_grid(candidates, grid)
    logger.warning(
        "No exact %s Extend Attention config for grid %s in %s; using %s.",
        kind,
        grid,
        table["path"],
        selected,
    )
    return dict(configs[selected])


def make_file_key(q_extend, k_buffer) -> tuple[str, str]:
    return (
        f"Q_Extend={q_extend.dtype}",
        f"K_Buffer={k_buffer.dtype}",
    )


def make_config_key(
    kind: str,
    *,
    batch_size: int,
    max_len_extend: int,
    kv_group_num: int,
    Lq: int,
    Lv: int,
    use_custom_mask: bool,
    is_causal: bool,
    skip_prefix_custom_mask: bool,
    has_sink: bool,
    sliding_window_size: int,
    xai_temperature_len: int,
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    sinks,
    window_kv_offsets,
) -> tuple:
    """Build a tuning key shared by both optimized kernel layouts."""
    values = [
        batch_size,
        max_len_extend,
        kv_group_num,
        Lq,
        Lv,
        use_custom_mask,
        is_causal,
        skip_prefix_custom_mask,
        has_sink,
        sliding_window_size,
        xai_temperature_len,
    ]
    tensors = [
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
    ]
    if kind == SHORT_QUERY:
        tensors.extend((custom_mask, mask_indptr))
    elif kind != LONG_QUERY:
        raise ValueError(f"Unknown Extend Attention config kind: {kind!r}")
    tensors.extend((sinks, window_kv_offsets))
    values.extend(str(tensor.dtype) for tensor in tensors if hasattr(tensor, "dtype"))
    return tuple(values)


def sanitize_launch_config(config: dict) -> dict:
    """Drop backend compiler options unsupported by the installed Triton."""
    kernel_meta = {"BLOCK_M", "BLOCK_N", "USE_MLS"}
    standard_options = {"num_warps", "num_stages", "num_ctas"}
    supported = get_triton_capabilities().compiler_options
    return {
        key: value
        for key, value in config.items()
        if key in kernel_meta or key in standard_options or key in supported
    }


__all__ = [
    "LONG_QUERY",
    "SHORT_QUERY",
    "get_kernel_config",
    "make_config_key",
    "make_file_key",
    "sanitize_launch_config",
]
