"""Runtime configuration protocol shared with UnifiedAttention tuning."""

from __future__ import annotations

import ast
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


KIND_2D = "unified_attention_2d"
KIND_3D = "unified_attention_3d"
CONFIG_ROOT = Path(__file__).with_name("configs")
CONFIG_KEY_FIELDS = (
    "num_seqs",
    "num_query_tokens",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_queries_per_kv",
)

DEFAULT_CONFIGS = {
    KIND_2D: {
        "BLOCK_M": 16,
        "TILE_SIZE": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
    KIND_3D: {
        "BLOCK_M": 16,
        "TILE_SIZE": 16,
        "num_warps": 4,
        "num_stages": 2,
        "reduce_num_warps": 1,
        "reduce_num_stages": 1,
    },
}


def _kv_type(dtype) -> str:
    return "fp8" if "float8" in str(dtype) else "auto"


def make_config_filename(
    kind: str,
    *,
    device_label: str,
    cache_block_size: int,
    head_size: int,
    sliding_window: int,
    use_alibi_slopes: bool,
    use_qq_bias: bool,
    use_softcap: bool,
    use_sinks: bool,
    use_mm_prefix: bool,
    kv_dtype,
    use_fp8_output: bool = False,
    num_segments_per_seq: int | None = None,
) -> str:
    """Build the exact JSON filename used by tuning and runtime."""
    if kind not in (KIND_2D, KIND_3D):
        raise ValueError(f"Unknown UnifiedAttention config kind: {kind!r}")
    head_size_padded = triton.next_power_of_2(head_size)
    fields = [
        kind,
        f"device={device_label}",
        f"bs={cache_block_size}",
        f"hs={head_size}",
        f"sw={sliding_window}",
        f"alibi={int(use_alibi_slopes)}",
        f"qq={int(use_qq_bias)}",
        f"softcap={int(use_softcap)}",
        f"sinks={int(use_sinks)}",
        f"mm={int(use_mm_prefix)}",
    ]
    if kind == KIND_2D:
        fields.append(f"fp8={int(use_fp8_output)}")
    else:
        if num_segments_per_seq is None or num_segments_per_seq <= 0:
            raise ValueError("3D configs require num_segments_per_seq > 0")
        fields.append(f"seg={num_segments_per_seq}")
    fields.extend(
        (
            f"hsp={head_size_padded}",
            f"pad={int(head_size != head_size_padded)}",
            f"kv={_kv_type(kv_dtype)}",
        )
    )
    return "-".join(fields) + ".json"


def get_write_path(filename: str) -> Path:
    """Return the current Triton-capability output path for a tuned JSON."""
    return get_triton_config_dir(CONFIG_ROOT, get_triton_capabilities()) / filename


def _validate_config(kind: str, config: object, num_queries_per_kv: int) -> dict | None:
    if not isinstance(config, dict):
        return None
    required = {"BLOCK_M", "TILE_SIZE", "num_warps", "num_stages"}
    allowed = set(required)
    if kind == KIND_3D:
        allowed.update(("reduce_num_warps", "reduce_num_stages"))
    if not required.issubset(config) or not set(config).issubset(allowed):
        return None
    try:
        result = {key: int(value) for key, value in config.items()}
    except (TypeError, ValueError):
        return None
    if any(result[key] <= 0 for key in required):
        return None
    if result["BLOCK_M"] < num_queries_per_kv:
        return None
    if kind == KIND_3D:
        result.setdefault("reduce_num_warps", 1)
        result.setdefault("reduce_num_stages", 1)
        if result["reduce_num_warps"] <= 0 or result["reduce_num_stages"] <= 0:
            return None
    return result


def make_config_key(
    *,
    num_seqs: int,
    num_query_tokens: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_queries_per_kv: int,
) -> tuple[int, int, int, int, int]:
    """Build the dynamic shape key shared by tuning and runtime."""
    key = (
        int(num_seqs),
        int(num_query_tokens),
        int(max_seqlen_q),
        int(max_seqlen_k),
        int(num_queries_per_kv),
    )
    if any(value <= 0 for value in key):
        raise ValueError(f"UnifiedAttention config key must be positive: {key}")
    return key


def _parse_config_key(value: str) -> tuple[int, int, int, int, int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != len(CONFIG_KEY_FIELDS):
        raise ValueError(f"Invalid UnifiedAttention config key: {value!r}")
    return make_config_key(**dict(zip(CONFIG_KEY_FIELDS, parsed)))


def _closest_shape(
    candidates: list[tuple[tuple[int, ...], dict]],
    target: tuple[int, ...],
) -> dict:
    def score(item):
        key, _ = item
        relative_distance = sum(
            abs(key[index] - target[index]) / target[index]
            for index in range(4)
        )
        smaller_dimensions = sum(
            key[index] < target[index] for index in range(4)
        )
        return relative_distance, smaller_dimensions

    return min(candidates, key=score)[1]


def _load_shape_config(
    path: Path,
    *,
    kind: str,
    config_key: tuple[int, int, int, int, int],
) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload["key"]) != CONFIG_KEY_FIELDS:
            return None
        configs = payload["config"]
        if not isinstance(configs, dict) or not configs:
            return None
        parsed = [(_parse_config_key(key), value) for key, value in configs.items()]
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        SyntaxError,
        json.JSONDecodeError,
    ):
        return None
    table = {key: value for key, value in parsed}
    candidate = table.get(config_key)
    if candidate is None:
        same_layout = [item for item in parsed if item[0][-1] == config_key[-1]]
        if not same_layout:
            return None
        candidate = _closest_shape(same_layout, config_key)
    return _validate_config(kind, candidate, config_key[-1])


@lru_cache(maxsize=512)
def get_kernel_config(
    kind: str,
    *,
    cache_block_size: int,
    head_size: int,
    num_seqs: int,
    num_query_tokens: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_queries_per_kv: int,
    sliding_window: int,
    use_alibi_slopes: bool,
    use_qq_bias: bool,
    use_softcap: bool,
    use_sinks: bool,
    use_mm_prefix: bool,
    kv_dtype,
    use_fp8_output: bool = False,
    num_segments_per_seq: int | None = None,
    device=None,
) -> dict | None:
    """Load an exact shape or the closest shape with the same head layout."""
    info = get_device_info(device)
    config_key = make_config_key(
        num_seqs=num_seqs,
        num_query_tokens=num_query_tokens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_queries_per_kv=num_queries_per_kv,
    )
    filename = make_config_filename(
        kind,
        device_label=f"{info.arch}_cu{info.cu_count}",
        cache_block_size=cache_block_size,
        head_size=head_size,
        sliding_window=sliding_window,
        use_alibi_slopes=use_alibi_slopes,
        use_qq_bias=use_qq_bias,
        use_softcap=use_softcap,
        use_sinks=use_sinks,
        use_mm_prefix=use_mm_prefix,
        kv_dtype=kv_dtype,
        use_fp8_output=use_fp8_output,
        num_segments_per_seq=num_segments_per_seq,
    )
    for directory in get_compatible_triton_config_dirs(
        CONFIG_ROOT, get_triton_capabilities()
    ):
        path = directory / filename
        if path.is_file():
            loaded = _load_shape_config(path, kind=kind, config_key=config_key)
            if loaded is not None:
                return loaded
    return None


def clear_config_cache() -> None:
    get_kernel_config.cache_clear()


__all__ = [
    "CONFIG_ROOT",
    "CONFIG_KEY_FIELDS",
    "DEFAULT_CONFIGS",
    "KIND_2D",
    "KIND_3D",
    "clear_config_cache",
    "get_kernel_config",
    "get_write_path",
    "make_config_key",
    "make_config_filename",
]
