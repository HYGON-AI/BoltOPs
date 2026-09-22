"""Per-arch, per-shape JSON config support for KDA Triton kernels."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import torch

from boltops.utility import arch_info, logger

_CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def dtype_name(dtype: torch.dtype) -> str:
    """Stable short names used in JSON config keys."""
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype).replace("torch.", ""))


@functools.lru_cache(maxsize=None)
def _load_config(kernel: str, arch: str) -> dict[str, Any]:
    """Load one cached tuning file; malformed or absent files are fallbacks."""
    path = _CONFIG_ROOT / kernel / f"{kernel}-{arch}.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"invalid KDA tuning config {path}: {exc}")
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        logger.warning(f"KDA tuning config {path} must contain a 'config' object")
        return {}
    return payload["config"]


def get_kda_tuned_config(
    kernel: str,
    key: str,
    *,
    fallback: dict[str, Any],
    arch: str | None = None,
) -> dict[str, Any]:
    """Return an exact per-shape tuning entry merged over ``fallback``.

    KDA intentionally does not use nearest-key matching: an unrelated nearby
    shape can produce an illegal or much slower tile. Missing files and keys
    fall back to the current static selector.
    """
    config = _load_config(kernel, arch or arch_info.get_arch()).get(key)
    if not isinstance(config, dict):
        return fallback
    unknown = set(config) - set(fallback)
    if unknown:
        logger.warning(
            f"rejecting KDA tuning config {kernel}/{key}; unknown keys {sorted(unknown)}"
        )
        return fallback
    merged = dict(fallback)
    merged.update(config)
    return merged
