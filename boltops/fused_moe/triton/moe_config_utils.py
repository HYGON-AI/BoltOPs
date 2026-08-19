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
# triton/configs/moe/triton_<capability>/*.json
_TRITON_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[list[int]] = None,
			 is_bottom: bool = False) -> str:
    arch = arch_info.get_arch()
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    is_bottom_selector = ("" if is_bottom == False else ",is_bottom=True")
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N},arch={arch}{dtype_selector}{is_bottom_selector}{block_shape_selector}.json"  # noqa: E501

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
) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the current
    # Triton capability directory, then fall back to lower capabilities.
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape, is_bottom)
    config_dirs = get_compatible_triton_config_dirs(
        _TRITON_CONFIGS_DIR / "moe", get_triton_capabilities()
    )
    config_file_path = config_dirs[0] / json_file_name

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

    candidate_names = [json_file_name]
    if is_bottom:
        # Preserve the existing bottom -> top config fallback within each
        # capability directory.
        candidate_names.append(get_config_file_name(E, N, dtype, block_shape))

    # For EP mode, local experts may not match a config file. Try the nearest
    # supported power-of-two expert count within each capability directory.
    nearestE = closest_power_of_two(E, dtype)
    if nearestE != E:
        candidate_names.append(
            get_config_file_name(nearestE, N, dtype, block_shape, is_bottom)
        )

    for config_dir in config_dirs:
        for candidate_name in candidate_names:
            candidate_path = config_dir / candidate_name
            if not candidate_path.exists():
                continue
            logger.info("Using configuration from %s for MoE layer.", candidate_path)
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

    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    configs = get_moe_configs(E, N, dtype, block_n, block_k, is_bottom)

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
