# SPDX-License-Identifier: MIT
 
import torch
from typing import Any, Dict, Optional, List
import json
import functools
from pathlib import Path
import boltops.utility.arch_info as arch_info
from boltops.utility import logger
from boltops.utility.triton_capability import (
    get_compatible_triton_config_dirs,
    get_triton_capabilities,
)

# Launch JSON configs shipped beside this module:
# triton/configs/triton_<capability>/*.json
_TRITON_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[list[int]] = None,
                         is_bottom: bool = False,
                         K: Optional[int] = None,
                         top_k: Optional[int] = None) -> str:
    if (K is None) != (top_k is None):
        raise ValueError("K and top_k must be specified together")
    arch = arch_info.get_arch()
    shape_selector = "" if K is None else f",K={K},topk={top_k}"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    is_bottom_selector = ("" if is_bottom == False else ",is_bottom=True")
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N}{shape_selector},arch={arch}{dtype_selector}{is_bottom_selector}{block_shape_selector}.json"  # noqa: E501


def _config_file_suffix(
    dtype: Optional[str],
    block_shape: Optional[list[int]],
    is_bottom: bool,
) -> str:
    arch = arch_info.get_arch()
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    is_bottom_selector = "" if not is_bottom else ",is_bottom=True"
    block_shape_selector = (
        ""
        if not block_shape or not all(block_shape)
        else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f",arch={arch}{dtype_selector}{is_bottom_selector}{block_shape_selector}.json"


def _parse_selector_set(value: str) -> Optional[tuple[int, ...]]:
    try:
        values = tuple(sorted({int(item) for item in value.split("-")}))
    except ValueError:
        return None
    return values or None


def _parse_dimension_selector(
    fields: dict[str, str],
    key: str,
) -> Optional[tuple[str, tuple[int, ...]]]:
    exact_key = key
    set_key = f"{key}Set"
    min_key = f"{key}Min"
    max_key = f"{key}Max"
    mod_key = f"{key}Mod"

    modes = sum(
        (
            exact_key in fields,
            set_key in fields,
            any(item in fields for item in (min_key, max_key, mod_key)),
        )
    )
    if modes != 1:
        return None

    try:
        if exact_key in fields:
            value = int(fields[exact_key])
            return ("exact", (value,)) if value > 0 else None
        if set_key in fields:
            values = _parse_selector_set(fields[set_key])
            return ("set", values) if values and all(v > 0 for v in values) else None

        if min_key not in fields:
            return None
        minimum = int(fields[min_key])
        modulus = int(fields.get(mod_key, "1"))
        if minimum <= 0 or modulus <= 0:
            return None

        # K fuzzy matching intentionally has no upper bound: KMin/KMod names
        # a reusable arithmetic family rather than one closed interval.
        if key == "K":
            if max_key in fields or mod_key not in fields:
                return None
            return "lower_mod", (minimum, modulus)

        if max_key not in fields:
            return None
        maximum = int(fields[max_key])
        if maximum < minimum:
            return None
        first_multiple = ((minimum + modulus - 1) // modulus) * modulus
        if first_multiple > maximum:
            return None
        return "range", (minimum, maximum, modulus)
    except ValueError:
        return None


def _parse_config_selector(fragment: str) -> Optional[dict[str, Any]]:
    fields: dict[str, str] = {}
    allowed = {
        "K", "KSet", "KMin", "KMod",
        "topk", "topkSet", "topkMin", "topkMax", "topkMod",
    }
    for item in fragment.split(","):
        if "=" not in item:
            return None
        key, value = item.split("=", 1)
        if key not in allowed or key in fields or not value:
            return None
        fields[key] = value

    k_selector = _parse_dimension_selector(fields, "K")
    topk_selector = _parse_dimension_selector(fields, "topk")
    if k_selector is None or topk_selector is None:
        return None
    return {"K": k_selector, "topk": topk_selector}


def _selector_matches(selector: tuple[str, tuple[int, ...]], value: int) -> bool:
    kind, values = selector
    if kind == "exact":
        return value == values[0]
    if kind == "set":
        return value in values
    if kind == "lower_mod":
        minimum, modulus = values
        return value >= minimum and value % modulus == 0
    minimum, maximum, modulus = values
    return minimum <= value <= maximum and value % modulus == 0


def _selector_specificity(selector: dict[str, Any]) -> tuple[int, ...]:
    kind_rank = {"exact": 4, "set": 3, "range": 2, "lower_mod": 1}

    def dimension_score(item: tuple[str, tuple[int, ...]]) -> tuple[int, int, int]:
        kind, values = item
        if kind == "exact":
            cardinality = 1
            lower_bound = values[0]
        elif kind == "set":
            cardinality = len(values)
            lower_bound = min(values)
        elif kind == "lower_mod":
            minimum, modulus = values
            # Prefer the more restrictive modulus and then the higher lower
            # bound if custom selector families overlap.
            return kind_rank[kind], modulus, minimum
        else:
            minimum, maximum, modulus = values
            first_multiple = ((minimum + modulus - 1) // modulus) * modulus
            cardinality = (maximum - first_multiple) // modulus + 1
            lower_bound = minimum
        return kind_rank[kind], -cardinality, lower_bound

    k_rank, k_cardinality, k_lower_bound = dimension_score(selector["K"])
    topk_rank, topk_cardinality, topk_lower_bound = dimension_score(
        selector["topk"]
    )
    # K changes both GEMM shapes, so use it as the tie-breaker when two
    # overlapping selectors have the same total specificity.
    return (
        k_rank + topk_rank,
        k_rank,
        topk_rank,
        k_cardinality,
        topk_cardinality,
        k_lower_bound,
        topk_lower_bound,
    )

def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        use_int8_w8a16: Optional[bool] = False,
        use_int8_w8a8: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False,
        use_int4_w4a8: Optional[bool] = False,
        use_mxfp4_w4a4: Optional[bool] = False) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int4_w4a8:
        return "int4_w4a8"
    elif use_mxfp4_w4a4:
        return "mxfp4_w4a4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None

def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[list[int]] = None,
    is_bottom: bool = False,
) -> dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "COMBINE_SCALE_LOAD": False,
            "num_warps": 4,
            "num_stages": 2,
        }
    elif dtype == "int4_w4a8" and block_shape is not None:
        # The W4A8 kernel requires BLOCK_SIZE_K == quantization group size.
        # Some legacy tuning files violate this compile-time constraint, so
        # the safe fallback must derive K directly from block_shape.
        if M <= E:
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 1,
                "COMBINE_SCALE_LOAD": False,
            }
        else:
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 8,
                "COMBINE_SCALE_LOAD": False,
            }
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        if M <= 20:
            config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_shape[1],
                      "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_shape[1],
                      "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
        else:
            config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_shape[1],
                      "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False}
    elif is_marlin:
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break
        return {"BLOCK_SIZE_M": block_size_m}
    elif M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "COMBINE_SCALE_LOAD": False,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "COMBINE_SCALE_LOAD": False,
        }
    return config

def closest_power_of_two(a, dtype):
    MIN_CONF_E = 2
    MAX_CONF_E = 16 if dtype == "int8_w8a8" else 32
    if a <= MIN_CONF_E:
        return MIN_CONF_E
    if a >= MAX_CONF_E:
        return MAX_CONF_E
    upper = 1
    while upper < a:
        upper <<= 1
    lower = upper >> 1
    if a - lower <= upper - a:
        return lower
    else:
        return upper

@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_bottom: bool = False,
    K: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    if (K is None) != (top_k is None):
        raise ValueError("K and top_k must be specified together")

    # Prefer an exact K/topk match, then a validated fuzzy selector, and only
    # then a legacy file without either selector.  Within each match tier the
    # current Triton capability remains preferred over lower capabilities.
    block_shape = [block_n, block_k] if block_n and block_k else None
    legacy_json_file_name = get_config_file_name(
        E, N, dtype, block_shape, is_bottom
    )
    config_dirs = get_compatible_triton_config_dirs(
        _TRITON_CONFIGS_DIR, get_triton_capabilities()
    )
    requested_json_file_name = (
        get_config_file_name(
            E, N, dtype, block_shape, is_bottom, K=K, top_k=top_k
        )
        if K is not None and top_k is not None
        else legacy_json_file_name
    )
    config_file_path = config_dirs[0] / requested_json_file_name

    def _load_config(config_path: Path) -> Optional[dict[int, Any]]:
        try:
            with open(config_path) as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid MoE config JSON at %s (%s). Fallback to other configs/default.",
                config_path,
                e,
            )
            return None

        if not isinstance(config_data, dict):
            logger.warning(
                "Invalid MoE config format at %s (expect dict). Fallback to other configs/default.",
                config_path,
            )
            return None
        return {int(key): val for key, val in config_data.items()}

    def _load_selector_pair(
        config_dir: Path,
        candidate_e: int,
        selector_fragment: str,
    ) -> Optional[dict[int, Any]]:
        prefix = f"E={candidate_e},N={N},{selector_fragment}"
        top_path = config_dir / (
            prefix + _config_file_suffix(dtype, block_shape, False)
        )
        # New selector families are anchored by the Gemm1 file.  Ignoring a
        # bottom-only family prevents Gemm1/Gemm2 from selecting different
        # K/topk families and then violating their shared BLOCK_SIZE_M rule.
        if not top_path.exists():
            return None
        paths = [top_path]
        if is_bottom:
            bottom_path = config_dir / (
                prefix + _config_file_suffix(dtype, block_shape, True)
            )
            paths = [bottom_path, top_path]
        for candidate_path in paths:
            if not candidate_path.exists():
                continue
            logger.info("Using configuration from %s for MoE layer.", candidate_path)
            loaded_cfg = _load_config(candidate_path)
            if loaded_cfg is not None:
                return loaded_cfg
        return None

    def _fuzzy_selectors(
        config_dir: Path,
        candidate_e: int,
    ) -> list[tuple[tuple[int, ...], str]]:
        prefix = f"E={candidate_e},N={N},"
        suffix = _config_file_suffix(dtype, block_shape, False)
        matches: list[tuple[tuple[int, ...], str]] = []
        for candidate_path in config_dir.glob(f"E={candidate_e},N={N},*.json"):
            name = candidate_path.name
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            fragment = name[len(prefix):-len(suffix)]
            selector = _parse_config_selector(fragment)
            if selector is None:
                continue
            if not _selector_matches(selector["K"], K):
                continue
            if not _selector_matches(selector["topk"], top_k):
                continue
            # The exact/exact spelling has already been checked directly.
            if selector["K"][0] == "exact" and selector["topk"][0] == "exact":
                continue
            matches.append((_selector_specificity(selector), fragment))
        matches.sort(reverse=True)
        return matches

    # For EP mode, local experts may not match a config file. Try the nearest
    # supported power-of-two expert count after all exact-E choices.
    nearestE = closest_power_of_two(E, dtype)
    expert_counts = [E] + ([nearestE] if nearestE != E else [])

    for candidate_e in expert_counts:
        if K is not None and top_k is not None:
            exact_fragment = f"K={K},topk={top_k}"
            for config_dir in config_dirs:
                loaded_cfg = _load_selector_pair(
                    config_dir, candidate_e, exact_fragment
                )
                if loaded_cfg is not None:
                    return loaded_cfg

            fuzzy_candidates: list[
                tuple[tuple[int, ...], int, Path, str]
            ] = []
            for capability_index, config_dir in enumerate(config_dirs):
                for specificity, fragment in _fuzzy_selectors(
                    config_dir, candidate_e
                ):
                    fuzzy_candidates.append(
                        (specificity, -capability_index, config_dir, fragment)
                    )
            fuzzy_candidates.sort(
                key=lambda item: (item[1], item[0], item[3]), reverse=True
            )
            for _, _, config_dir, fragment in fuzzy_candidates:
                loaded_cfg = _load_selector_pair(
                    config_dir, candidate_e, fragment
                )
                if loaded_cfg is not None:
                    return loaded_cfg

        # Legacy files remain a compatibility fallback.  They carry no claim
        # that the original tune covered the caller's K or topk.
        legacy_names = [
            get_config_file_name(
                candidate_e, N, dtype, block_shape, is_bottom
            )
        ]
        if is_bottom:
            legacy_names.append(
                get_config_file_name(candidate_e, N, dtype, block_shape)
            )
        for config_dir in config_dirs:
            for candidate_name in legacy_names:
                candidate_path = config_dir / candidate_name
                if not candidate_path.exists():
                    continue
                logger.info(
                    "Using legacy configuration without K/topk selector from %s "
                    "for MoE layer.",
                    candidate_path,
                )
                loaded_cfg = _load_config(candidate_path)
                if loaded_cfg is not None:
                    return loaded_cfg

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None

def try_get_optimal_moe_config(
    w_shape: tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[list[int]] = None,
    is_bottom: bool = False,
    is_gated: bool = True,
):

    # First try to load optimal config from the file
    # we use is_bottom to judge whethor is w1 or w2
    if is_bottom:
        # w2_weight
        E, K, N = w_shape
        if dtype == "int4_w4a16" or dtype == "int4_w4a8":
            # for int4_w4a16, N(intermediate_size) for w2_weight will be packed, N * 2 for get intermediate_size before packed
            N = N * 2
    else:
        # w1_weight
        E, N, K = w_shape
        # Gated MoE stores gate/up together as [E, 2N, K]. Non-gated
        # activations such as relu2 store [E, N, K], so keep N unchanged.
        if is_gated:
            N = N // 2

    # K in the config selector is the model hidden size used by the tune CLI.
    # Gemm1 INT4 weights pack two logical K values into each storage element;
    # Gemm2 keeps the model hidden size in w2.shape[1].
    config_k = K
    if not is_bottom and dtype in ("int4_w4a16", "int4_w4a8"):
        config_k *= 2

    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    configs = get_moe_configs(
        E,
        N,
        dtype,
        block_n,
        block_k,
        is_bottom,
        K=config_k,
        top_k=top_k,
    )

    if configs and dtype == "int4_w4a8" and block_k > 0:
        configs = {
            m: cfg
            for m, cfg in configs.items()
            if cfg.get("BLOCK_SIZE_K") == block_k
        }

    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]

        if is_bottom:
            # Note: this is a hcu optimize to save memory.
            # max_block_m = max([cfg["BLOCK_SIZE_M"] for key, cfg in configs.items()])
            max_block_m = max([cfg["BLOCK_SIZE_M"] for key, cfg in configs.items() if key <= M])
            max_block_m = max(max_block_m, config["BLOCK_SIZE_M"])
            return config, max_block_m
        else:
            # When is_bottom=False, return config only
            return config
    else:
        # Else use the default config
        config = get_default_config(M, E, N, K, top_k, dtype,
                                    is_marlin, block_shape, is_bottom)
        if is_bottom:
            max_block_m = config["BLOCK_SIZE_M"]
            return config, max_block_m
        else:
            return config

def get_optimal_moe_config_func(
    A: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_int4_w4a8: Optional[bool] = False,
    use_mxfp4_w4a4: Optional[bool] = False,
    block_shape: Optional[List[int]] = None,
    is_bottom: bool = False,
    is_gated: bool = True,
):
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a8=use_int8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        use_int4_w4a8=use_int4_w4a8,
                                        use_mxfp4_w4a4=use_mxfp4_w4a4,
                                        dtype=A.dtype)

    return functools.partial(
        try_get_optimal_moe_config,
        W.size(),
        topk_ids.size(1),
        config_dtype,
        block_shape=block_shape,
        is_bottom=is_bottom,
        is_gated=is_gated,
    )
