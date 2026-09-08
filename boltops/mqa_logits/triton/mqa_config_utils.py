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


def _flatten_gluon_entry(entry: dict[str, Any]) -> dict[str, Any]:
    main = entry.get("main") or entry
    out = {**_GLUON_FALLBACK_LAUNCH}
    out.update(main)
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


def get_fp8_mqa_gluon_launch_config(
    seq_len: int, seq_len_kv: int, num_heads: int
) -> dict[str, Any]:
    """Pick Gluon-specific launch knobs (``w_group``, ``kv_tiles_n``).

    Loads ``fp8_mqa_logits_gluon,arch=*.json`` by ``seq_len_kv`` bucket when
    present; otherwise uses code fallback (prefer ``w_group=16``,
    ``kv_tiles_n=2``). ``w_group`` is always clamped so it divides ``num_heads``.
    """
    del seq_len  # reserved for future M-bucketing
    arch = arch_info.get_arch()
    data = _load_gluon_config_file(arch)
    if not data:
        out = copy.deepcopy(_GLUON_FALLBACK_LAUNCH)
    else:
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
        out = _flatten_gluon_entry(entry)

    out["w_group"] = clamp_gluon_w_group(int(out["w_group"]), int(num_heads))
    return out
