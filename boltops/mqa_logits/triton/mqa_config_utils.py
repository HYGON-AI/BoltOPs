# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Load shipped Triton launch configs for ``fp8_mqa_logits``.

Layout mirrors fused_moe (arch in filename)::

    triton/configs/triton_<capability>/fp8_mqa_logits,arch=gfx938.json
    triton/configs/triton_<capability>/fp8_mqa_logits.json   # optional fallback
    triton/configs/triton_<capability>/fp8_mqa_logits_tiled,arch=gfx938.json
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any, Optional

import boltops.utility.arch_info as arch_info
from boltops.utility.triton_capability import (
    get_compatible_triton_config_dirs,
    get_triton_capabilities,
)

_TRITON_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_CONFIG_STEM = "fp8_mqa_logits"
_TILED_CONFIG_STEM = "fp8_mqa_logits_tiled"
_GLUON_CONFIG_STEM = "fp8_mqa_logits_gluon"
_HALF_CONFIG_STEM = "mqa_logits_half"
_HALF_ROW_CONFIG_STEM = "mqa_logits_half_row_short_kv"

# Code fallback when no JSON is found.
_FALLBACK_LAUNCH = {
    "block_kv": 64,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
    "schedule_hint": "none",
    "use_block_pingpong": False,
    "clear_block_kv": 512,
    "clear_num_warps": 8,
    "clear_num_stages": 1,
    "clear_waves_per_eu": 2,
    "clear_kpack": 2,
    "clear_schedule_hint": "none",
}

# Tile geometry: 16|32 x 128.
_TILED_BLOCK_N = 128
_SMALL_TILE_BLOCK_N = 64


def tiled_block_m(seq_len: int) -> int:
    """``BLOCK_M`` for tiled path: 32 when ``M >= 256`` else 16."""
    return 32 if seq_len >= 256 else 16


def tiled_block_n() -> int:
    return _TILED_BLOCK_N


def is_large_prefill(seq_len: int, seq_len_kv: int) -> bool:
    """Large-prefill gate for BN=128 tiled geometry."""
    return (seq_len >= 64 and seq_len_kv >= 1024) or seq_len_kv > 8192


def _gluon_grey_zone_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    """``head_split_k`` for former BN grey-zone shapes on the Gluon tiled path."""
    # Grey-zone split-k heuristic: sk=4 ≈ BN on 16×{4k,8k}/32×4k H=64.
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


def _dispatch_row_or_gluon(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Gluon-capable backends: only ``row`` and ``gluon`` (no BN)."""
    # Tiny KV — classic row with launch knobs for N<=128.
    if seq_len_kv <= 128:
        if num_heads >= 32:
            # 16/32×66 H=64: BLOCK_KV=32, warps=1, stages=2.
            return {
                "path": "row",
                "head_split_k": 1,
                "block_kv": 32,
                "num_warps": 1,
                "num_stages": 2,
            }
        # H=8 tiny: BLOCK_KV=128, warps=2, stages=1.
        return {
            "path": "row",
            "head_split_k": 1,
            "block_kv": 128,
            "num_warps": 2,
            "num_stages": 1,
        }

    # Former BN grey zone (high-H, small-M, mid-N): Gluon mode3 + split-k.
    # Covers 16×{4k,8k} / 32×4k H=64; 32×8192 stays sk=1 below.
    if (
        num_heads >= 32
        and seq_len <= 32
        and seq_len_kv <= 8192
        and not (seq_len >= 32 and seq_len_kv >= 8192)
    ):
        return {
            "path": "gluon",
            "head_split_k": _gluon_grey_zone_split_k(
                seq_len, seq_len_kv, num_heads
            ),
        }

    return {
        "path": "gluon",
        "head_split_k": 1,
    }


def _dispatch_row_or_bn(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Triton-only backends: only ``row`` and ``bn`` (no Gluon)."""
    if seq_len_kv <= 128 and (num_heads < 32 or seq_len >= 64):
        # Same tiny-row knobs as Gluon mode (H=8 / larger-M).
        if num_heads < 32:
            return {
                "path": "row",
                "head_split_k": 1,
                "block_kv": 128,
                "num_warps": 2,
                "num_stages": 1,
            }
        return {
            "path": "row",
            "head_split_k": 1,
            "block_kv": 32,
            "num_warps": 1,
            "num_stages": 2,
        }
    if seq_len_kv <= 128:
        # Small-M high-H tiny without Gluon: still row with H=64 knobs
        # (BN unavailable preference when caller asked row|bn only — use BN).
        bn = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
        return {"path": "bn", **bn}
    bn = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
    return {"path": "bn", **bn}


def get_fp8_mqa_shape_dispatch(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    *,
    gluon_available: bool = False,
) -> dict[str, Any]:
    """Pick backend path for prefill FP8 MQA logits.

    * ``gluon_available=True``: only ``row`` / ``gluon`` (BN replaced by Gluon
      tiled + optional ``head_split_k`` on the former BN grey zone).
    * ``gluon_available=False``: only ``row`` / ``bn``.

    Shape heuristics for common prefill grids.
    Auto Gluon path uses the tiled head-loop kernel only.
    """
    if gluon_available:
        return _dispatch_row_or_gluon(seq_len, seq_len_kv, num_heads)
    return _dispatch_row_or_bn(seq_len, seq_len_kv, num_heads)


def tiled_grid_n(seq_len_kv: int, block_n: int) -> int:
    """N-tile count covering the full KV axis (no host ``cu_ends`` sync)."""
    return max(1, (int(seq_len_kv) + int(block_n) - 1) // int(block_n))


def shape_head_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    """Small/medium ``split_k`` (grid z-dim), kept divisible into ``H``."""
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


def get_bn_tile_dispatch(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Shape-based BN-tile launch replacing the old Triton tiled kernel.

    * Large (``is_large_prefill``): ``BN=128``, ``BM=16|32``, ``split_k=1``,
      HIP fields from shipped tiled JSON.
    * Small/medium: ``BN=64``, ``BM=16``, shape ``split_k``, ``warps=2``
      when ``split_k >= 4``.
    """
    if is_large_prefill(seq_len, seq_len_kv):
        tiled = get_fp8_mqa_tiled_launch_config(seq_len, seq_len_kv)
        return {
            "bn_tile": int(tiled["block_n"]),
            "block_m_tile": int(tiled["block_m"]),
            "head_split_k": 1,
            "num_warps": int(tiled["num_warps"]),
            "num_stages": int(tiled["num_stages"]),
            "waves_per_eu": int(tiled["waves_per_eu"]),
            "matrix_instr_nonkdim": int(tiled["matrix_instr_nonkdim"]),
            "kpack": int(tiled["kpack"]),
            "schedule_hint": str(tiled["schedule_hint"]),
            "use_block_pingpong": bool(tiled["use_block_pingpong"]),
        }

    split_k = shape_head_split_k(seq_len, seq_len_kv, num_heads)
    return {
        "bn_tile": _SMALL_TILE_BLOCK_N,
        "block_m_tile": 16,
        "head_split_k": split_k,
        "num_warps": 2 if split_k >= 4 else 4,
        # Q-prefetch BN path benefits from stages>=2.
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "schedule_hint": "none",
        "use_block_pingpong": False,
    }


_TILED_FALLBACK_LAUNCH = {
    "block_m": 32,
    "block_n": _TILED_BLOCK_N,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 4,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
    "schedule_hint": "none",
    "use_block_pingpong": False,
}

# Gluon large-prefill knobs (HIP fields still come from tiled JSON when present).
_GLUON_FALLBACK_LAUNCH = {
    "w_group": 16,
    "kv_tiles_n": 2,
}

# Half-precision BN launch knobs (geometry still from ``get_half_bn_dispatch``).
_HALF_FALLBACK_LAUNCH = {
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "w_group": 8,
}


def clamp_gluon_w_group(preferred: int, num_heads: int) -> int:
    """Pick an even ``w_group`` that divides ``num_heads``, preferring ``preferred``."""
    pref = int(preferred)
    if pref >= 2 and pref % 2 == 0 and num_heads % pref == 0:
        return pref
    for g in (16, 8, 4, 2):
        if num_heads % g == 0:
            return g
    return 2



def get_config_file_name(arch: Optional[str] = None) -> str:
    """MoE-style name: ``fp8_mqa_logits,arch=gfx938.json``."""
    arch = arch or arch_info.get_arch()
    return f"{_CONFIG_STEM},arch={arch}.json"


def get_tiled_config_file_name(arch: Optional[str] = None) -> str:
    """MoE-style name: ``fp8_mqa_logits_tiled,arch=gfx938.json``."""
    arch = arch or arch_info.get_arch()
    return f"{_TILED_CONFIG_STEM},arch={arch}.json"


def get_gluon_config_file_name(arch: Optional[str] = None) -> str:
    """MoE-style name: ``fp8_mqa_logits_gluon,arch=gfx938.json``."""
    arch = arch or arch_info.get_arch()
    return f"{_GLUON_CONFIG_STEM},arch={arch}.json"


def get_half_config_file_name(arch: Optional[str] = None) -> str:
    """MoE-style name: ``mqa_logits_half,arch=gfx938.json``."""
    arch = arch or arch_info.get_arch()
    return f"{_HALF_CONFIG_STEM},arch={arch}.json"


def get_half_row_config_file_name(arch: Optional[str] = None) -> str:
    """MoE-style name: ``mqa_logits_half_row_short_kv,arch=gfx938.json``."""
    arch = arch or arch_info.get_arch()
    return f"{_HALF_ROW_CONFIG_STEM},arch={arch}.json"


def _flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or {}
    clear = entry.get("clear") or {}
    out = {**_FALLBACK_LAUNCH}
    out.update(main)
    out.update(clear)
    return out


def _flatten_tiled_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or entry
    out = {**_TILED_FALLBACK_LAUNCH}
    out.update(main)
    return out


@functools.lru_cache(maxsize=8)
def _load_config_file(arch: str) -> Optional[dict[str, Any]]:
    """Prefer arch-specific JSON, then generic ``fp8_mqa_logits.json``."""
    caps = get_triton_capabilities()
    candidates = (
        get_config_file_name(arch),
        f"{_CONFIG_STEM}.json",
    )
    for config_dir in get_compatible_triton_config_dirs(_TRITON_CONFIGS_DIR, caps):
        for name in candidates:
            path = config_dir / name
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


@functools.lru_cache(maxsize=8)
def _load_tiled_config_file(arch: str) -> Optional[dict[str, Any]]:
    caps = get_triton_capabilities()
    candidates = (
        get_tiled_config_file_name(arch),
        f"{_TILED_CONFIG_STEM}.json",
    )
    for config_dir in get_compatible_triton_config_dirs(_TRITON_CONFIGS_DIR, caps):
        for name in candidates:
            path = config_dir / name
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


@functools.lru_cache(maxsize=8)
def _load_gluon_config_file(arch: str) -> Optional[dict[str, Any]]:
    caps = get_triton_capabilities()
    candidates = (
        get_gluon_config_file_name(arch),
        f"{_GLUON_CONFIG_STEM}.json",
    )
    for config_dir in get_compatible_triton_config_dirs(_TRITON_CONFIGS_DIR, caps):
        for name in candidates:
            path = config_dir / name
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


@functools.lru_cache(maxsize=8)
def _load_half_config_file(arch: str) -> Optional[dict[str, Any]]:
    caps = get_triton_capabilities()
    candidates = (
        get_half_config_file_name(arch),
        f"{_HALF_CONFIG_STEM}.json",
    )
    for config_dir in get_compatible_triton_config_dirs(_TRITON_CONFIGS_DIR, caps):
        for name in candidates:
            path = config_dir / name
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


@functools.lru_cache(maxsize=8)
def _load_half_row_config_file(arch: str) -> Optional[dict[str, Any]]:
    caps = get_triton_capabilities()
    candidates = (
        get_half_row_config_file_name(arch),
        f"{_HALF_ROW_CONFIG_STEM}.json",
    )
    for config_dir in get_compatible_triton_config_dirs(_TRITON_CONFIGS_DIR, caps):
        for name in candidates:
            path = config_dir / name
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


def _flatten_gluon_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or entry
    out = {**_GLUON_FALLBACK_LAUNCH}
    out.update(main)
    return out


def _flatten_half_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or entry
    out: dict[str, Any] = {}
    for key in (
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "w_group",
        "block_m",
        "block_n",
        "head_split_k",
        # Optional per-shape overrides for the kernel-level knobs.  Absent from
        # every shipped entry today, so this changes nothing by default; it just
        # lets a tune record them without another code change.
        "q_depth",
        "enable_v_mmac_cluster",
        "matrix_instr_nonkdim",
        "kpack",
    ):
        if key in main:
            out[key] = main[key]
    return out


def _flatten_half_row_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or entry
    out: dict[str, Any] = {}
    for key in (
        "block_kv",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "head_split_k",
        "clear_block_kv",
        "clear_num_warps",
        "clear_num_stages",
    ):
        if key in main:
            out[key] = main[key]
    return out


def get_fp8_mqa_launch_config(seq_len_kv: int) -> dict[str, Any]:
    """Pick launch kwargs for ``fp8_mqa_logits`` from shipped JSON.

    File selection (like MoE):
      1. ``fp8_mqa_logits,arch=<get_arch()>.json`` under compatible triton_* dirs
      2. ``fp8_mqa_logits.json``
      3. code fallback

    Within a file: smallest ``by_seq_kv[].seq_kv_leq >= seq_len_kv``, else
    ``default``.
    """
    arch = arch_info.get_arch()
    data = _load_config_file(arch)
    if not data:
        return copy.deepcopy(_FALLBACK_LAUNCH)

    buckets = sorted(
        data.get("by_seq_kv") or [],
        key=lambda b: int(b["seq_kv_leq"]),
    )
    for bucket in buckets:
        if seq_len_kv <= int(bucket["seq_kv_leq"]):
            return _flatten_entry(bucket)

    if data.get("default"):
        return _flatten_entry(data["default"])
    return copy.deepcopy(_FALLBACK_LAUNCH)


def get_fp8_mqa_tiled_launch_config(seq_len: int, seq_len_kv: int) -> dict[str, Any]:
    """Pick launch kwargs for ``fp8_mqa_logits_tiled`` from shipped JSON.

    Tile geometry uses ``block_n=128`` and
    ``block_m=32`` when ``seq_len >= 256`` else ``16``. Other launch fields
    (warps, stages, kpack, …) come from JSON buckets keyed by ``seq_len_kv``.
    """
    arch = arch_info.get_arch()
    data = _load_tiled_config_file(arch)
    if not data:
        out = copy.deepcopy(_TILED_FALLBACK_LAUNCH)
        out["block_m"] = tiled_block_m(seq_len)
        out["block_n"] = tiled_block_n()
        return out

    buckets = sorted(
        data.get("by_seq_kv") or [],
        key=lambda b: int(b["seq_kv_leq"]),
    )
    entry = None
    for bucket in buckets:
        if seq_len_kv <= int(bucket["seq_kv_leq"]):
            entry = bucket
            break
    if entry is None:
        entry = data.get("default") or {}

    out = _flatten_tiled_entry(entry)
    out["block_m"] = tiled_block_m(seq_len)
    out["block_n"] = tiled_block_n()
    return out


def _find_gluon_by_shape_entry(
    data: dict[str, Any], seq_len: int, seq_len_kv: int, num_heads: int
) -> Optional[dict[str, Any]]:
    """Exact ``(seq_len, seq_kv, num_heads)`` hit in shipped ``by_shape``."""
    for entry in data.get("by_shape") or []:
        if (
            int(entry.get("seq_len", -1)) == int(seq_len)
            and int(entry.get("seq_kv", -1)) == int(seq_len_kv)
            and int(entry.get("num_heads", -1)) == int(num_heads)
        ):
            return entry
    return None


def _find_gluon_by_region_entry(
    data: dict[str, Any], seq_len: int, seq_len_kv: int, num_heads: int
) -> Optional[dict[str, Any]]:
    """Tightest ``by_region`` hit with ``M/N/H <= *_leq`` upper bounds.

    Among matching regions, prefer smaller ``num_heads_leq``, then
    ``seq_kv_leq``, then ``seq_len_leq``.
    """
    regions = data.get("by_region") or []
    if not regions:
        return None
    best: Optional[dict[str, Any]] = None
    best_key: Optional[tuple[int, int, int]] = None
    for entry in regions:
        m_leq = int(entry.get("seq_len_leq", -1))
        n_leq = int(entry.get("seq_kv_leq", -1))
        h_leq = int(entry.get("num_heads_leq", -1))
        if m_leq < 0 or n_leq < 0 or h_leq < 0:
            continue
        if seq_len > m_leq or seq_len_kv > n_leq or num_heads > h_leq:
            continue
        key = (h_leq, n_leq, m_leq)
        if best is None or key < best_key:  # type: ignore[operator]
            best = entry
            best_key = key
    return best


def _clamp_gluon_head_split_k(split_k: int, num_heads: int) -> int:
    """Ensure ``head_split_k`` divides ``num_heads`` (halve while needed)."""
    sk = max(1, int(split_k))
    while sk > 1 and int(num_heads) % sk != 0:
        sk //= 2
    return sk


def get_fp8_mqa_gluon_launch_config(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Pick Gluon launch knobs from ``fp8_mqa_logits_gluon,arch=*.json``.

    Preference order:
      1. tightest ``by_region`` with ``seq_len/seq_kv/num_heads <= *_leq``
      2. legacy exact ``by_shape`` match on ``(seq_len, seq_kv, num_heads)``
      3. legacy smallest ``by_seq_kv[].seq_kv_leq >= seq_len_kv``
      4. ``default`` / code fallback (``w_group=16``, ``kv_tiles_n=2``)

    ``w_group`` is clamped so it divides ``num_heads``; ``head_split_k`` is
    clamped so it divides ``num_heads``.
    """
    arch = arch_info.get_arch()
    data = _load_gluon_config_file(arch)
    if not data:
        out = copy.deepcopy(_GLUON_FALLBACK_LAUNCH)
    else:
        entry = _find_gluon_by_region_entry(data, seq_len, seq_len_kv, num_heads)
        if entry is None:
            entry = _find_gluon_by_shape_entry(
                data, seq_len, seq_len_kv, num_heads
            )
        if entry is None:
            buckets = sorted(
                data.get("by_seq_kv") or [],
                key=lambda b: int(b["seq_kv_leq"]),
            )
            for bucket in buckets:
                if seq_len_kv <= int(bucket["seq_kv_leq"]):
                    entry = bucket
                    break
        if entry is None:
            entry = data.get("default") or {}
        out = _flatten_gluon_entry(entry)

    out["w_group"] = clamp_gluon_w_group(int(out["w_group"]), int(num_heads))
    if "head_split_k" in out:
        out["head_split_k"] = _clamp_gluon_head_split_k(
            int(out["head_split_k"]), int(num_heads)
        )
    return out


def get_mqa_logits_half_launch_config(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Pick half-BN launch knobs from shipped ``mqa_logits_half`` JSON.

    Preference order:
      1. exact ``by_shape`` match on ``(seq_len, seq_kv, num_heads)``
      2. tightest ``by_region`` with ``seq_len/seq_kv/num_heads <= *_leq``
      3. smallest ``by_seq_kv[].seq_kv_leq >= seq_len_kv``
      4. ``default`` / ``{}`` so callers keep code geometry defaults
    """
    arch = arch_info.get_arch()
    data = _load_half_config_file(arch)
    if not data:
        return {}

    entry = _find_gluon_by_shape_entry(data, seq_len, seq_len_kv, num_heads)
    if entry is None:
        entry = _find_gluon_by_region_entry(data, seq_len, seq_len_kv, num_heads)
    if entry is None:
        buckets = sorted(
            data.get("by_seq_kv") or [],
            key=lambda b: int(b["seq_kv_leq"]),
        )
        for bucket in buckets:
            if seq_len_kv <= int(bucket["seq_kv_leq"]):
                entry = bucket
                break
    if entry is None:
        entry = data.get("default") or {}
    return _flatten_half_entry(entry)


def get_mqa_logits_half_row_launch_config(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Pick half-row knobs from ``mqa_logits_half_row_short_kv`` JSON.

    Preference: exact ``by_shape``, else tightest ``by_region``.
    """
    arch = arch_info.get_arch()
    data = _load_half_row_config_file(arch)
    if not data:
        return {}
    entry = _find_gluon_by_shape_entry(data, seq_len, seq_len_kv, num_heads)
    if entry is None:
        entry = _find_gluon_by_region_entry(data, seq_len, seq_len_kv, num_heads)
    if entry is None:
        return {}
    return _flatten_half_row_entry(entry)


def get_half_mqa_shape_dispatch(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Half auto path: ``row`` or ``bn`` from shipped GPU compare (both Triton versions).

    Reads ``mqa_logits_half_row_short_kv`` ``compare`` / ``dispatch`` tables written by
    the short-KV (or all-sizes) row|BN retune. Prefer exact ``(M,N,H)``; optionally
    a coarse ``by_region`` path pick. No match → ``bn`` (safe default).

    Gluon half remains opt-in via ``backend='gluon_bn_align'``.
    """
    arch = arch_info.get_arch()
    row_data = _load_half_row_config_file(arch)
    use_row = False
    row_kw: dict[str, Any] = {}
    if row_data:
        # Explicit path table (exact shapes), if present.
        for ent in row_data.get("dispatch") or []:
            sq = int(ent.get("s_q", ent.get("seq_len", -1)))
            sk = int(ent.get("s_k", ent.get("seq_kv", -1)))
            nh = int(ent.get("num_heads", -1))
            if sq != int(seq_len) or sk != int(seq_len_kv) or nh != int(num_heads):
                continue
            path = str(ent.get("path", "bn")).lower()
            use_row = path == "row"
            if use_row:
                row_kw = dict(ent.get("row_cfg") or {})
            break
        else:
            for cmp in row_data.get("compare") or []:
                sq = int(cmp.get("s_q", cmp.get("seq_len", -1)))
                sk = int(cmp.get("s_k", cmp.get("seq_kv", -1)))
                nh = int(cmp.get("num_heads", -1))
                if (
                    sq != int(seq_len)
                    or sk != int(seq_len_kv)
                    or nh != int(num_heads)
                ):
                    continue
                path = str(cmp.get("path", "")).lower()
                if path == "row" or (
                    not path and float(cmp.get("row_over_bn", 2.0)) < 1.0
                ):
                    use_row = True
                    row_kw = dict(cmp.get("row_cfg") or {})
                break
            else:
                # Coarse region path pick (FP8-style Q/KV/H leq), if shipped.
                region = _find_gluon_by_region_entry(
                    {"by_region": row_data.get("by_region_path") or []},
                    seq_len,
                    seq_len_kv,
                    num_heads,
                )
                if region is not None and str(region.get("path", "")).lower() == "row":
                    use_row = True
                    row_kw = dict(region.get("row_cfg") or {})

        if use_row and not row_kw:
            row_kw = get_mqa_logits_half_row_launch_config(
                seq_len, seq_len_kv, num_heads
            )
    if use_row:
        return {"path": "row", **row_kw}
    return {"path": "bn"}
