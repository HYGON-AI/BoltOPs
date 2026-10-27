"""Shared Python helpers for Step-4 DSA Triton kernels.

Vendored from the Step-4 DSA handoff ``inference/kernel.py`` (dsa_kernels /
qknorm fp-fusion launch flag). Triton ``@jit`` helpers stay next to the kernels
that call them; this module is Python-only.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from typing import Any

import torch

# Occupancy JSON is keyed by a stable product family, not the marketing string
# ``torch.cuda.get_device_name()`` returns. Board renames that share one SKU
# (BW1101 and BW1100) keep a single ``*_BW1100.json``; do not copy the file.
# Longer tokens are matched first so BW1100 does not collapse into BW200.
_CONFIG_SKU_ALIASES: dict[str, str] = {
    "bw1101": "BW1100",
    "bw1100": "BW1100",
    "bw200": "BW200",
    "k100": "K100_AI",
}
_CONFIG_SKU_TOKENS = tuple(sorted(_CONFIG_SKU_ALIASES, key=len, reverse=True))


def _canonical_config_sku(device_name: str) -> str:
    """Map a marketing / board name onto the SKU used in config filenames."""
    compact = (
        device_name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(",", "")
    )
    for token in _CONFIG_SKU_TOKENS:
        if token in compact:
            return _CONFIG_SKU_ALIASES[token]
    return device_name.replace(" ", "_")


def _config_sku_candidates(device_name: str) -> tuple[str, ...]:
    """Filenames to try: canonical family first, then the raw board slug."""
    canonical = _canonical_config_sku(device_name)
    raw = device_name.replace(" ", "_")
    if raw == canonical:
        return (canonical,)
    return (canonical, raw)


def _config_path_for_device(
    config_dir: str,
    filename_for: Callable[[str], str],
    device: str,
) -> str | None:
    """Return the first existing occupancy JSON for ``device``, or ``None``."""
    for sku in _config_sku_candidates(device):
        path = os.path.join(config_dir, filename_for(sku))
        if os.path.isfile(path):
            return path
    return None


def _device_name_for_config() -> str:
    """Canonical product SKU used when reading or writing occupancy JSON."""
    if not torch.cuda.is_available():
        return "unknown"
    return _canonical_config_sku(torch.cuda.get_device_name())


def _load_device_configs(
    config_dir: str, filename_for: Callable[[str], str], device: str
) -> dict[str, Any]:
    """Read a device config, falling back to defaults for missing/invalid JSON."""
    path = _config_path_for_device(config_dir, filename_for, device)
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_match_score(
    match: dict[str, Any], shape: dict[str, Any]
) -> tuple[int, int, float] | None:
    """Rank exact constraints, then range bounds, then narrower ranges.

    Missing shape fields are mismatches, including legacy sequence-length keys.
    Equal scores preserve JSON entry order in :func:`_select_config`.
    """
    exact = 0
    bounds = 0
    widths: dict[str, dict[str, float]] = {}
    for key, expected in match.items():
        if key.endswith(("_min", "_max")):
            field, kind = key.rsplit("_", 1)
            actual = shape.get(field)
            if actual is None:
                return None
            if kind == "min" and actual < expected:
                return None
            if kind == "max" and actual > expected:
                return None
            widths.setdefault(field, {})[kind] = expected
            bounds += 1
        elif key not in shape or shape[key] != expected:
            return None
        else:
            exact += 1
    total_width = 0.0
    for limits in widths.values():
        if "min" in limits and "max" in limits:
            total_width += limits["max"] - limits["min"]
        else:
            total_width += float("inf")
    return exact, bounds, -total_width


def _select_config(data: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any] | None:
    """Copy the most specific matched config; ignore malformed entries."""
    entries = data.get("configs", ())
    if not isinstance(entries, (list, tuple)):
        return None
    selected = None
    selected_score = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        match = entry.get("match", {})
        if not isinstance(match, dict):
            continue
        score = _config_match_score(match, shape)
        if score is not None and (selected_score is None or score > selected_score):
            selected, selected_score = entry, score
    if selected is None:
        return None
    config = selected.get("config", {})
    return dict(config) if isinstance(config, dict) else None


def _portable_launch_options(
    config: dict[str, Any],
    *,
    consumed_keys: frozenset[str],
    supports: Callable[[str], bool] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate launch options from host/kernel fields and filter compiler options."""
    options = dict(defaults or {})
    standard_keys = frozenset({"num_warps", "num_stages"})
    for name in standard_keys:
        if name in config:
            options[name] = int(config[name])
    if supports is not None:
        consumed = consumed_keys | standard_keys
        for name, value in config.items():
            if name not in consumed and supports(name):
                options[name] = value
    return options


def _cache_is_fp8(dtype: torch.dtype) -> bool:
    """The KV cache dtype, rather than the query dtype, selects the FP8 path."""
    return bool(dtype.is_floating_point and dtype.itemsize == 1)


def _clamp_auto_kv_splits(value: int, packed_width: int) -> int:
    return min(max(int(value), 1), max(int(packed_width), 1))


def _choose_kv_splits(
    num_queries: int,
    num_kv_groups: int,
    packed_width: int,
    *,
    target_programs: int | None = None,
    min_slots_per_split: int = 8,
    device: torch.device | None = None,
) -> int:
    """Fill compute units while retaining a minimum packed-slot window per split."""
    packed_width = max(int(packed_width), 1)
    programs_base = max(int(num_queries) * int(num_kv_groups), 1)
    if target_programs is None:
        try:
            index = (
                device.index
                if device is not None and device.index is not None
                else torch.cuda.current_device()
            )
            target_programs = int(torch.cuda.get_device_properties(index).multi_processor_count)
        except Exception:  # noqa: BLE001 — best-effort device query; CPU fallback.
            target_programs = 1
    target_programs = max(int(target_programs or 1), 1)
    splits = -(-target_programs // programs_base)
    min_slots = max(int(min_slots_per_split), 1)
    # ceil(K/S) >= M  <=>  S <= (K - 1) // (M - 1), for M > 1.
    max_splits = (
        packed_width if min_slots <= 1 else max(1, (packed_width - 1) // (min_slots - 1))
    )
    return min(max(splits, 1), max_splits, packed_width)


def _occupancy_dtype(dtype: torch.dtype) -> str:
    """Occupancy JSON ``dtype`` token: torch dtype name without ``torch.``.

    Lookup keys this off the KV cache, not the query. Query stays bfloat16 on
    the FP8 serving path; ``float8_e4m3fn`` is what distinguishes that path.
    """
    return str(dtype).replace("torch.", "")


# Step-4 compresses every 8 consecutive tokens into one summary vector. The CuTe
# kernels hard-specialize on this; we keep it a parameter but never exercise
# anything else.
REGION_BLOCK_SIZE = 8


def _shared_memory_limit() -> int:
    """Bytes of shared memory one block may use, or 0 when that is unknowable.

    The sparse attention kernels stage a whole region-block of keys *and* values in
    shared memory, so their tile size is bounded by the device rather than by the
    geometry.  Hopper allows 227 KiB per block and the ``block_regions=16`` default is
    sized for it; a 64 KiB device (CDNA/HCU) cannot launch that tile at all and Triton
    reports OutOfResources at first launch rather than at import.  Ask the driver
    instead of hardcoding either number.
    """
    try:
        from triton.runtime.driver import driver

        return int(
            driver.active.utils.get_device_properties(
                torch.cuda.current_device()
            )["max_shared_mem"]
        )
    except Exception:  # noqa: BLE001 — best-effort driver capability probe.
        return 0


def _fit_block_regions(
    block_regions: int,
    *,
    region_size: int,
    block_d: int,
    element_size: int,
) -> int:
    """Shrink ``block_regions`` until one key+value tile fits in shared memory.

    Halved rather than reduced by one so the region count stays a power of two, which
    keeps the ``tl.dot`` operand shapes the kernels are tuned around.  Returns at least
    one region-block: a device that cannot stage even one is a real failure, and the
    OutOfResources error names the shape better than anything raised here would.
    """
    limit = _shared_memory_limit()
    if limit <= 0:
        return block_regions
    while block_regions > 1:
        if 2 * block_regions * region_size * block_d * element_size <= limit:
            break
        block_regions //= 2
    return block_regions


def _require_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    """Sparse attention kernels address the last axis as ``base + d``, not ``d * stride``.

    Token and head strides are kernel arguments; the innermost dimension is not.
    A legal shape with ``stride(-1) != 1`` (permute, stepped slice) would otherwise
    be silently mis-addressed. Callers may still pad token/head strides.
    """
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in the last dimension")


def shared_table_stride(block_table: torch.Tensor) -> int:
    """Row stride to pass for a block table, zero when the table is shared.

    A single-row table means "every request uses this mapping". Expressing that as a zero
    row stride rather than a branch inside the kernel keeps the two cases on one code path,
    and -- more importantly -- makes the shared case safe: the alternative, indexing a
    one-row table by a nonzero request id, reads past the tensor and returns plausible
    garbage instead of failing.
    """
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )
    return 0 if block_table.shape[0] == 1 else block_table.stride(0)


def _detect_rounded_intrinsics() -> bool:
    """True when the active Triton backend exposes CUDA-style ``*_rn`` libdevice ops.

    CUDA exposes the explicitly rounded scalar intrinsics; HIP/HCU does not.  On HIP
    the plain fp32 operators are already round-to-nearest-even, so fusion is disabled
    at launch instead (see ``_NO_FP_FUSION``).
    """
    try:
        from triton.runtime.driver import driver

        backend = driver.active.get_current_target().backend
        module = importlib.import_module(f"triton.language.extra.{backend}.libdevice")
    except Exception:  # noqa: BLE001 — optional backend support probe.
        return False
    return all(
        hasattr(module, name)
        for name in ("add_rn", "sub_rn", "mul_rn", "div_rn", "fma_rn")
    )


_HAS_ROUNDED_INTRINSICS = _detect_rounded_intrinsics()
# Passing ``enable_fp_fusion=False`` keeps ``a * b`` and ``+ c`` as two separately
# rounded operations. CSA compression needs this on backends without ``*_rn``.
_NO_FP_FUSION = {} if _HAS_ROUNDED_INTRINSICS else {"enable_fp_fusion": False}
