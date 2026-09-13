import os
from contextlib import contextmanager
from itertools import product
from typing import Optional, List, Dict, Any
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.runtime.errors import HSACOError

import boltops.fused_moe.triton.moe_op as moe_op
import boltops.utility.arch_info as arch_info
from boltops.fused_moe.triton.moe_compat import per_token_quant_triton, per_block_quant_wrapper
from boltops.fused_moe.triton import (
    triton_moe_sum,
    moe_align_block_size,
)
from boltops.fused_moe.triton.fused_moe import (
    _should_use_naive_block_assignment,
)
from boltops.fused_moe.triton.moe_activation import (
    _normalize_activation_and_gate,
    adjust_N_for_activation,
    _apply_activation,
)
from boltops.utility.triton_capability import get_triton_capabilities


capMLS = arch_info.get_arch() in ("gfx938", "gfx92a")
triton_capabilities = get_triton_capabilities()

training_mode = int(os.environ.get("TRAINING_MODE", "0"))
test_type= str(os.environ.get("AUTOTUNE_TEST_TYPE", "int4"))
compile_only  = int(os.environ.get("TRITON_COMPILE_ONLY", "0"))
num_groups_compile = int(os.environ.get("NUM_GROUPS_COMPILE", "1"))
cur_group = int(os.environ.get("CUR_GROUP", "0"))
ep_size = int(os.environ.get("EP_SIZE", "1"))
splitk_size = int(os.environ.get("SPLITK_SIZE", "0"))

mode_autotune = True if training_mode == 1 else False
print(f"{mode_autotune=} {test_type=}")
mode_best_config_tune = False

_ALLOWED_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SCHEDULE_OPTION_NAMES = ("instruction_sched_variant", "schedule_hint")
_LARGE_MOE_TOKEN_THRESHOLD = 2048
_SKIPPABLE_HIP_LOAD_ERRORS = ("named symbol not found",)


def _max_autotune_total_tokens(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    allow_sorted_bridge: bool,
    max_block_m: int = 256,
) -> int:
    """Size tuning buffers for the largest legal sorted-bridge padding."""
    max_padded_tokens = (
        min(num_tokens * topk, num_experts + 1) * (max_block_m - 1)
        if allow_sorted_bridge
        else 0
    )
    return num_tokens * topk + max_padded_tokens


def _skip_invalid_hip_loads(do_bench):
    """Make a broken HSACO one invalid candidate instead of aborting a tune."""

    def robust_do_bench(kernel_call, quantiles):
        try:
            return do_bench(kernel_call, quantiles)
        except RuntimeError as exc:
            message = str(exc)
            if not any(marker in message for marker in _SKIPPABLE_HIP_LOAD_ERRORS):
                raise
            print(f"[autotune] skip unloadable HIP candidate: {message}")
            # Triton's Autotuner already maps HSACOError to an infinite timing.
            # The HCU loader reports the same invalid-code-object condition as
            # a generic RuntimeError, which would otherwise abort every later
            # config (including smaller BLOCK_SIZE_M candidates).
            raise HSACOError(message) from exc

    return robust_do_bench


def _compiler_option_values(name, values):
    """Return search values, or one omission marker for unsupported options."""

    return values if triton_capabilities.supports(name) else (None,)


def _add_compiler_options(config, **options):
    """Add capability-filtered compiler options, omitting unsupported ones."""

    for name, value in options.items():
        if value is not None:
            config[name] = value


def _schedule_option_name():
    """Return the schedule option spelling accepted by the active Triton."""

    for name in _SCHEDULE_OPTION_NAMES:
        if triton_capabilities.supports(name):
            return name
    return None


def _add_schedule_option(config, option_name, value):
    """Add the version-specific schedule option when Triton exposes one."""

    if option_name is not None:
        config[option_name] = value


def _keep_autotune_config(
    config,
    *,
    block_m,
    n,
    k,
    num_tokens,
    is_gemm1,
    per_channel_quant,
    num_stages,
    num_warps,
    schedule,
):
    """Apply shape, kernel and compiler-option pruning in one place."""

    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    group_m = config["GROUP_SIZE_M"]
    combine_scale_load = config["COMBINE_SCALE_LOAD"]
    use_mls_load = config["USE_MLS_LOAD"]
    bottom_a_use_mls_load = config["bottom_a_use_mls_load"]
    kpack = config.get("kpack")
    uses_matrix_load_mls = use_mls_load or bottom_a_use_mls_load

    # Cache swizzle needs enough runtime workgroups to amortize its setup.
    if (
        config.get("buffer_cache_swizzle", False)
        and num_tokens < _LARGE_MOE_TOKEN_THRESHOLD
    ):
        return False

    # BlockPingpong already forces buffer cache swizzle in the HCU backend;
    # tuning both flags would benchmark the same generated program twice.
    if config.get("use_block_pingpong", False) and config.get(
        "buffer_cache_swizzle", False
    ):
        return False

    if uses_matrix_load_mls:
        if num_stages >= 3 or n % block_n != 0 or k % block_k != 0:
            return False
        # The HCU MLS lowering fixes kWidth to the MLS base width and ignores
        # kpack.  Keeping kpack=2 would compile the same program as kpack=1
        # and could save a misleading configuration.
        if kpack == 2:
            return False

    if per_channel_quant:
        if combine_scale_load:
            return False
        if test_type in ("int4", "int4int8_channel") and block_k < 32:
            return False

    if kpack == 2 and block_k <= 32:
        return False
    if test_type in ("int4", "int4int8_channel") and block_k < 32:
        return False
    if block_k > k or block_n > n or group_m > block_m:
        return False

    if combine_scale_load:
        if k % (block_k * 2) != 0:
            return False
        if block_k != mode_autotune_max_block_size_k or not is_gemm1:
            return False
    # A one-stage pipeline ignores non-default scheduling hints, so those
    # candidates would only duplicate the default schedule.
    if num_stages == 1 and schedule not in (None, "none"):
        return False

    byte_per_elem = (
        1
        if test_type
        in (
            "int8",
            "int8_channel",
            "int4int8",
            "int4int8_channel",
            "fp8",
            "fp8_channel",
        )
        else 2
    )
    if config.get("use_block_pingpong", False):
        # BlockPingpong materializes two A/B buffer generations regardless of
        # the source-level pipeline stage count.
        shared_mem_size = 2 * block_k * (block_m + block_n) * byte_per_elem
    elif num_stages == 1:
        shared_mem_size = max(
            block_m * block_k * byte_per_elem,
            block_k * block_n * byte_per_elem,
        )
    else:
        stage_factor = 1 + (num_stages - 2)
        shared_mem_size = (
            block_m * block_k * byte_per_elem * stage_factor
            + block_k * block_n * byte_per_elem * stage_factor
        )
    if shared_mem_size > 64 * 1024:
        return False

    if block_m * block_n / (16 * 16) <= num_warps / 8:
        return False
    if block_n == 16 and num_warps > 4:
        return False
    tile_per_wave = max(1, block_m * block_n / (16 * 16) / num_warps)
    if tile_per_wave > 32:
        return False

    if config.get("use_block_pingpong", False):
        # BlockPingpong deliberately does not match MLS matrix-load ops, so
        # this combination cannot generate a pingpong loop.
        if uses_matrix_load_mls:
            return False
        # Blockwise FP8 scale loads prevent the current HCU PP matcher from
        # rematching the reduction loop.
        if test_type == "fp8":
            return False
        return (
            num_warps == 8
            and block_m >= 128
            and block_n >= 128
            and block_k in (32, 64)
            # GEMM2 with a short total reduction cannot amortize PP setup.
            # This is the runtime GEMM K, not BLOCK_SIZE_K.
            and (is_gemm1 or k > 256)
        )
    return True


def _default_output_dtype(hidden_dtype: torch.dtype) -> torch.dtype:
    return torch.bfloat16 if hidden_dtype == torch.bfloat16 else torch.float16


def _compute_type_from_output_dtype(output_dtype: torch.dtype) -> tl.dtype:
    if output_dtype == torch.bfloat16:
        return tl.bfloat16
    if output_dtype == torch.float16:
        return tl.float16
    if output_dtype == torch.float32:
        return tl.float32
    raise ValueError(
        f"Unsupported output_dtype: {output_dtype}, expected one of {_ALLOWED_OUTPUT_DTYPES}"
    )


if test_type == "int4int8":
    mode_autotune_min_block_size_k1  = 64
    mode_autotune_max_block_size_k1 = 64
    mode_autotune_max_block_size_k = 64
    mode_autotune_min_block_size_k2  = 64
    mode_autotune_max_block_size_k2 = 64
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 128
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512
elif test_type == "int4int8_channel":
    mode_autotune_min_block_size_k  = 32
    mode_autotune_max_block_size_k  = 512
    mode_autotune_min_block_size_n1 = 32
    mode_autotune_max_block_size_n1 = 256
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512
elif test_type == "int4":
    mode_autotune_min_block_size_k  = 16
    mode_autotune_max_block_size_k  = 512
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 256
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512
elif test_type == "int8_channel" or test_type == "fp8_channel":
    mode_autotune_min_block_size_k1 = 32
    mode_autotune_max_block_size_k1 = 512
    mode_autotune_max_block_size_k  = 128
    mode_autotune_min_block_size_k2 = 32
    mode_autotune_max_block_size_k2 = 256
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 256
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512
elif test_type == "int8" or test_type == "fp8":
    mode_autotune_min_block_size_k  = 32
    mode_autotune_max_block_size_k  = 128
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 256
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512
elif  test_type == "bf16":
    mode_autotune_min_block_size_k  = 32
    mode_autotune_max_block_size_k  = 512
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 256
    mode_autotune_min_block_size_n2 = 16
    mode_autotune_max_block_size_n2 = 512

def _keep_n_tile_reuse(config, runtime_args, num_warps):
    args = {**runtime_args, **config, "num_warps": num_warps}
    # A partial second tile is correct, but not a useful default search point.
    # Do not hard-code an M cutoff: occupancy/profitability needs measurement.
    return (
        args["N"] >= 2 * args["BLOCK_SIZE_N"]
        and moe_op._n_tile_reuse_error(args) is None
    )


def generate_config2_lists(
    BLOCK_SIZE_M=16,
    N=16,
    K=16,
    MUL_ROUTED_WEIGHT=False,
    per_channel_quant=False,
    bottom_a_use_mls_load=False,
    num_valid_tokens=None,
    top_k=1,
    num_input_tokens=None,
    supports_n_tiles=None,
    runtime_args=None,
):
    is_gemm1 = MUL_ROUTED_WEIGHT == False
    if supports_n_tiles is None:
        supports_n_tiles = test_type in ("int8", "int8_channel", "bf16", "fp8", "fp8_channel")
    # Shape-only callers can generate candidates without GPU tensors. The real
    # launcher supplies actual quantization groups, dtype and kernel flags.
    reuse_args = {
        "N": N, "K": K, "MUL_ROUTED_WEIGHT": MUL_ROUTED_WEIGHT,
        "per_channel_quant": per_channel_quant,
        "use_int8_w8a8": test_type in ("int8", "int8_channel"),
        "use_fp8_w8a8": test_type in ("fp8", "fp8_channel"),
        "group_n": 128 if test_type == "int8" else 0,
        "group_k": 128 if test_type == "int8" else 0,
        "compute_type": tl.bfloat16 if test_type == "bf16" else tl.float16,
        **(runtime_args or {}),
    }

    # BLOCK_SIZE_M :
    block_m = BLOCK_SIZE_M

    # BLOCK_SIZE_N : [16, min(512, N)]
    if is_gemm1:
        block_size_n_options = [2**i for i in range(
                                                mode_autotune_min_block_size_n1.bit_length() -1,
                                                min(mode_autotune_max_block_size_n1.bit_length(), (N.bit_length())))]
    else:
        block_size_n_options = [2**i for i in range(
                                                mode_autotune_min_block_size_n2.bit_length() -1,
                                                min(mode_autotune_max_block_size_n2.bit_length(), (N.bit_length())))]

    # BLOCK_SIZE_K : [32, min(128, K))]
    if test_type in ["int8_channel", "int4int8", "fp8_channel"]:
        if is_gemm1:
            block_size_k_options = [2**i for i in range(mode_autotune_min_block_size_k1.bit_length() - 1,
                                                    min(mode_autotune_max_block_size_k1.bit_length(), (K.bit_length())))]
        else:
            block_size_k_options = [2**i for i in range(mode_autotune_min_block_size_k2.bit_length() - 1,
                                                    min(mode_autotune_max_block_size_k2.bit_length(), (K.bit_length())))]
    else:
        block_size_k_options = [2**i for i in range(mode_autotune_min_block_size_k.bit_length() - 1,
                                                min(mode_autotune_max_block_size_k.bit_length(), (K.bit_length())))]

    if torch.is_tensor(num_input_tokens):
        num_input_tokens = int(num_input_tokens.item())
    if torch.is_tensor(num_valid_tokens):
        num_valid_tokens = int(num_valid_tokens.item())
    if torch.is_tensor(top_k):
        top_k = int(top_k.item())
    top_k = max(int(top_k), 1)
    # Gemm2 is launched with top_k=1 even though num_valid_tokens still counts
    # the original M*MoE-topk routed entries. Prefer the explicit input M so
    # large-M-only options are not accidentally enabled for small Gemm2 cases.
    if num_input_tokens is not None:
        num_tokens = int(num_input_tokens)
    elif num_valid_tokens is not None:
        num_tokens = int(num_valid_tokens) // top_k
    else:
        num_tokens = 0
    group_size_m_options = (
        [1, 4, 8] if num_tokens >= _LARGE_MOE_TOKEN_THRESHOLD else [1]
    )
    num_stage_options = [1, 2]
    combine_scale_load_options = [False, True] if test_type in ["int8", "int4int8", "fp8"] else [False]

    use_mls_load_options = [False, True] if test_type in ["fp8", "fp8_channel", "int8", "int8_channel", "bf16"] and capMLS == True else [False]
    # The outer fused-MoE search selects the bridge layout because it changes
    # Gemm1 output, padding and Gemm2 addressing. The inner Gemm2 search
    # can use ordinary buffer loads or matrix_load on MLS-capable targets.
    # A sorted bridge remains legal when K has a tail, even though MLS does not.
    bottom_a_sorted = bool(bottom_a_use_mls_load)
    bottom_a_use_mls_load_options = (
        [False, True] if bottom_a_sorted and capMLS and not is_gemm1 else [False]
    )

    # warp 1 no selected, so no need to tune
    num_warp_options = [2, 4, 8, 16]
    schedule_option_name = _schedule_option_name()
    schedule_options = ("none", "local-prefetch") if schedule_option_name else (None,)
    sched_latency = "none" if triton_capabilities.supports("sched_latency") else None

    kpack_options = _compiler_option_values(
        "kpack", (1, 2)
        if test_type in (
            "bf16", "fp8", "fp8_channel", "int8", "int8_channel"
        )
        else (1,),
    )
    use_block_pingpong_options = _compiler_option_values(
        "use_block_pingpong", (False, True)
    )
    buffer_cache_swizzle_options = _compiler_option_values(
        "buffer_cache_swizzle",
        (False, True)
        if num_tokens >= _LARGE_MOE_TOKEN_THRESHOLD
        else (False,),
    )
    # Mode 2 includes mode 1's barriers and cross-region A/B-load analysis.
    # It matched or beat mode 1 in the profitable small FP8-channel cases, so
    # keep only the default and mode 2 in this targeted search.
    enable_v_mmac_cluster_options = _compiler_option_values(
        "enable_v_mmac_cluster",
        (0, 2)
        if test_type == "fp8_channel"
        and num_tokens < _LARGE_MOE_TOKEN_THRESHOLD
        else (0,),
    )
    # PrepareBlockPingpong owns the actual three-stage/two-buffer pipeline.
    # Keep one canonical frontend stage value for PP to avoid tuning duplicate
    # configurations; do not make PP legality depend on this external value.
    pipeline_options = tuple(
        (num_stage, use_block_pingpong)
        for use_block_pingpong in use_block_pingpong_options
        for num_stage in ([1] if use_block_pingpong else num_stage_options)
    )

    print(f"BLOCK_SIZE_M: {BLOCK_SIZE_M} is_gemm1: {is_gemm1} Autotune Config limit range: \
                block_size_n_options: {block_size_n_options}, \
                block_size_k_options: {block_size_k_options}, \
                group_size_m_options: {group_size_m_options}, \
                num_stage_options: {num_stage_options}, \
                combine_scale_load_options: {combine_scale_load_options}, \
                use_mls_load_options: {use_mls_load_options}, \
                bottom_a_use_mls_load_options: {bottom_a_use_mls_load_options}, \
                num_warp_options: {num_warp_options},   \
                schedule_option_name: {schedule_option_name}, \
                schedule_options: {schedule_options}, \
                sched_latency: {sched_latency}, \
                kpack_options: {kpack_options}, \
                use_block_pingpong_options: {use_block_pingpong_options}, \
                buffer_cache_swizzle_options: {buffer_cache_swizzle_options}, \
                enable_v_mmac_cluster_options: {enable_v_mmac_cluster_options}")

    configs = []
    config_count = 0
    cur_group_config_count=0
    if mode_best_config_tune == False:
        option_combinations = product(
            block_size_n_options,
            block_size_k_options,
            group_size_m_options,
            combine_scale_load_options,
            pipeline_options,
            buffer_cache_swizzle_options,
            enable_v_mmac_cluster_options,
            num_warp_options,
            schedule_options,
            kpack_options,
            use_mls_load_options,
            bottom_a_use_mls_load_options,
        )
        for (
            block_n,
            block_k,
            group_m,
            combine_scale_load,
            (num_stage, use_block_pingpong),
            buffer_cache_swizzle,
            enable_v_mmac_cluster,
            num_warp,
            schedule,
            kpack,
            use_mls_load,
            bottom_a_use_mls_load,
        ) in option_combinations:
            config = {
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": group_m,
                "COMBINE_SCALE_LOAD": combine_scale_load,
                "USE_MLS_LOAD": use_mls_load,
                "a_sorted": bottom_a_sorted,
                "bottom_a_use_mls_load": bottom_a_use_mls_load,
            }
            _add_compiler_options(
                config,
                sched_latency=sched_latency,
                kpack=kpack,
                use_block_pingpong=use_block_pingpong,
                buffer_cache_swizzle=buffer_cache_swizzle,
                enable_v_mmac_cluster=enable_v_mmac_cluster,
            )
            _add_schedule_option(config, schedule_option_name, schedule)

            if not _keep_autotune_config(
                config,
                block_m=block_m,
                n=N,
                k=K,
                num_tokens=num_tokens,
                is_gemm1=is_gemm1,
                per_channel_quant=per_channel_quant,
                num_stages=num_stage,
                num_warps=num_warp,
                schedule=schedule,
            ):
                continue

            n_tile_options = (None,)
            if supports_n_tiles:
                n_tile_options = (
                    (1, 2) if _keep_n_tile_reuse(config, reuse_args, num_warp) else (1,)
                )
            for n_tiles in n_tile_options:
                candidate = dict(config)
                if n_tiles is not None:
                    candidate["N_TILES_PER_CTA"] = n_tiles
                config_count += 1
                if compile_only and (config_count - 1) % num_groups_compile != cur_group:
                    continue
                cur_group_config_count += 1
                print(
                    f"Autotune Config for compile_group={cur_group} "
                    f"config_idx:{config_count}: BLOCK_SIZE_M:{BLOCK_SIZE_M} "
                    f"{candidate} {num_stage} {num_warp}"
                )
                configs.append(
                    triton.Config(candidate, num_stages=num_stage, num_warps=num_warp)
                )
    else:
        # # # tune best config for gemm1 & bs=32
        if test_type == "fp8":
            if is_gemm1:
                config = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                          "USE_MLS_LOAD": True}
                _add_schedule_option(config, schedule_option_name, "none")
                triton_cfg = triton.Config(config, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)

            else:
                # # tune best config for gemm2 & bs=32:
                config = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                          "USE_MLS_LOAD": True}
                _add_schedule_option(config, schedule_option_name, "none")
                triton_cfg = triton.Config(config, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)

                config = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                          "USE_MLS_LOAD": True}
                _add_schedule_option(config, schedule_option_name, "none")
                triton_cfg = triton.Config(config, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)
        else:
            config = {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                      "USE_MLS_LOAD": True}
            _add_schedule_option(config, schedule_option_name, "local-prefetch")
            triton_cfg = triton.Config(config, num_stages=2, num_warps=4)
            config_count += 1
            cur_group_config_count += 1
            configs.append(triton_cfg)

            # # # tune best config for gemm2 & bs=32:
            config = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                      "USE_MLS_LOAD": True}
            _add_schedule_option(config, schedule_option_name, "local-prefetch")
            triton_cfg = triton.Config(config, num_stages=2, num_warps=4)
            config_count += 1
            cur_group_config_count += 1
            configs.append(triton_cfg)

    if mode_best_config_tune and supports_n_tiles:
        for config in configs:
            config.kwargs["N_TILES_PER_CTA"] = 1
    print(f"\nBLOCK_SIZE_M: {BLOCK_SIZE_M}, is_gemm1: {is_gemm1}, Autotune Total: {config_count}, cur compile group Total: {cur_group_config_count}")
    return configs

# 重写MyHeuristics类以支持warmup函数
class MyHeuristics(triton.KernelInterface):


    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.warmup(*args, **kwargs)

def apply_my_heuristics(ori_fn):
    new_fn = ori_fn
    if isinstance(ori_fn, triton.runtime.Heuristics):
        if hasattr(ori_fn, 'fn') and hasattr(ori_fn, 'values') and hasattr(ori_fn, 'arg_names'):
            new_fn = MyHeuristics(ori_fn.fn, ori_fn.arg_names, ori_fn.values)
    return new_fn

# 修改装饰器函数，使其支持动态配置生成
def dynamic_autotune(fn):
    # 创建一个可以使用下标语法的类
    class SubscriptableKernel:
        def __init__(self, kernel_fn):
            self.kernel_fn = kernel_fn

        def __getitem__(self, grid):
            # 返回一个函数，该函数在调用时会使用给定的grid调用内核
            def launcher(*args, **kwargs):
                kernel_name = getattr(self.kernel_fn, "__name__", "")
                # 优先按参数名绑定，避免依赖固定位置导致签名变更即错位
                arg_names = getattr(self.kernel_fn, "arg_names", None)
                if arg_names is None:
                    raise RuntimeError(
                        f"dynamic_autotune requires kernel_fn.arg_names, got None for kernel {getattr(self.kernel_fn, '__name__', '<unknown>')}"
                    )
                named_args = dict(zip(arg_names, args))
                N = kwargs.get("N", named_args.get("N"))
                K = kwargs.get("K", named_args.get("K"))
                if N is None or K is None:
                    raise RuntimeError(
                        f"dynamic_autotune failed to resolve N/K by name; available args: {list(named_args.keys())}"
                    )
                if torch.is_tensor(N):
                    if N.numel() != 1:
                        raise RuntimeError(f"dynamic_autotune expected scalar N, got tensor shape={tuple(N.shape)}")
                    N = int(N.item())
                if torch.is_tensor(K):
                    if K.numel() != 1:
                        raise RuntimeError(f"dynamic_autotune expected scalar K, got tensor shape={tuple(K.shape)}")
                    K = int(K.item())

                # BLOCK_SIZE_M 获取
                BLOCK_SIZE_M = kwargs.get('BLOCK_SIZE_M', 16)

                # MUL_ROUTED_WEIGHT 是第 35 个参数
                MUL_ROUTED_WEIGHT = kwargs.get('MUL_ROUTED_WEIGHT', False)
                # per_channel_quant 是第 36 个参数
                per_channel_quant = kwargs.get('per_channel_quant', False) or (
                    "channelwise" in kernel_name
                )

                bottom_a_sorted = bool(
                    kwargs.get('a_sorted', False)
                    or kwargs.get('bottom_a_use_mls_load', False)
                )

                # 确保所有参数都不是None
                BLOCK_SIZE_M = 16 if BLOCK_SIZE_M is None else BLOCK_SIZE_M
                N = 16 if N is None else N
                K = 16 if K is None else K

                # 动态生成配置列表
                topk_weights = kwargs.get(
                    "topk_weights_ptr", named_args.get("topk_weights_ptr")
                )
                num_input_tokens = (
                    int(topk_weights.shape[0])
                    if torch.is_tensor(topk_weights) and topk_weights.ndim > 0
                    else None
                )
                dynamic_configs = generate_config2_lists(
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    N=N,
                    K=K,
                    MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT,
                    per_channel_quant=per_channel_quant,
                    bottom_a_use_mls_load=bottom_a_sorted,
                    num_valid_tokens=kwargs.get(
                        "num_valid_tokens", named_args.get("num_valid_tokens")
                    ),
                    top_k=kwargs.get("top_k", named_args.get("top_k", 1)),
                    num_input_tokens=num_input_tokens,
                    supports_n_tiles="N_TILES_PER_CTA" in arg_names,
                    runtime_args={**named_args, **kwargs, "N": N, "K": K},
                )
                if "int4" in test_type:
                    for cfg in dynamic_configs:
                        cfg.kwargs.pop("bottom_a_use_mls_load", None)
                        cfg.kwargs.pop("a_sorted", None)
                if not dynamic_configs:
                    if compile_only:
                        # A compile shard can legitimately receive no candidates.
                        return []
                    raise RuntimeError(
                        f"No legal autotune candidates for {kernel_name}: "
                        f"N={N}, K={K}, BLOCK_SIZE_M={BLOCK_SIZE_M}, "
                        f"a_sorted={bottom_a_sorted}"
                    )
                def _prune_configs_by_runtime_constraints(configs, named_args, **_kwargs):
                    pruned = []
                    for cfg in configs:
                        cfg_kwargs = cfg.kwargs
                        if cfg_kwargs.get("N_TILES_PER_CTA") == 2 and not _keep_n_tile_reuse(
                            cfg_kwargs, {**named_args, **_kwargs}, cfg.num_warps
                        ):
                            continue
                        cfg_use_mls = bool(cfg_kwargs.get("USE_MLS_LOAD", False))
                        cfg_bottom_mls = bool(cfg_kwargs.get("bottom_a_use_mls_load", False))
                        cfg_block_k = int(cfg_kwargs.get("BLOCK_SIZE_K", 0))
                        cfg_uses_matrix_load_mls = cfg_use_mls or cfg_bottom_mls
                        if cfg_uses_matrix_load_mls:
                            if test_type == "bf16" and cfg_block_k < 32:
                                continue
                            if test_type in ["fp8", "fp8_channel", "int8", "int8_channel"] and cfg_block_k < 64:
                                continue
                        pruned.append(cfg)
                    return pruned

                launch_kwargs = dict(kwargs)
                launch_kwargs.pop("a_sorted", None)
                launch_kwargs.pop("bottom_a_use_mls_load", None)
                # This field belongs to each candidate, not the outer BM config.
                launch_kwargs.pop("N_TILES_PER_CTA", None)
                # 应用autotune装饰器
                autotuned_fn = triton.autotune(
                    configs=dynamic_configs,
                    key=['EM', 'N', 'K', 'num_valid_tokens', 'MUL_ROUTED_WEIGHT',
                        'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'USE_MLS_LOAD', 'GROUP_SIZE_M', 'per_channel_quant'],
                    perf_debug=True,
                    perf_profiling=False,
                    prune_configs_by={"early_config_prune": _prune_configs_by_runtime_constraints},
                    rep=20
                )(self.kernel_fn)
                autotuned_fn.__dict__["do_bench"] = _skip_invalid_hip_loads(
                    autotuned_fn.do_bench
                )

                # 使用grid调用autotuned内核
                print(f"DEBUG - {compile_only=}, kwargs keys being passed to kernel:{list(launch_kwargs.keys())}")
                if compile_only:
                    return autotuned_fn.warmup(*args, grid=grid, **launch_kwargs)
                else:
                    return autotuned_fn[grid](*args, **launch_kwargs)

            return launcher

    # 返回可下标的内核对象
    return SubscriptableKernel(apply_my_heuristics(fn))


# @triton.autotune
# 改写这段代码，使用多层 for循环 实现 config list的生成，其中：
# 1> BLOCK_SIZE_M 的 范围 从 32 到 128， 只能是 2的幂次。
# 2> BLOCK_SIZE_N 的 范围 从 64 到 1024  只能是 2的幂次。
# 3> BLOCK_SIZE_K 的 范围 从 32 到 256， 只能是 2的幂次。
# 4> GROUP_SIZE_M 的范围 从 1 到 8， 只能是2的幂次。
def generate_config_lists():
    configs = []
    config_count = 0

    if mode_best_config_tune == False:
        block_size_m_options = [2**i for i in range(4, 9)]       # 16 to 256

        bottom_a_mls_options = (
            (False, True)
            if test_type in ("bf16", "fp8", "fp8_channel", "int8", "int8_channel")
            else (False,)
        )

        for block_m in block_size_m_options:
            for bottom_a_use_mls_load in bottom_a_mls_options:
                config = {
                    "BLOCK_SIZE_M": block_m,
                    "bottom_a_use_mls_load": bottom_a_use_mls_load,
                }
                config_count += 1
                print(f"Config {config_count}: {config}")
                configs.append(config)
    else:
        # bs=16 & gemm1 best config
        config = { "BLOCK_SIZE_M": 16 }
        config_count += 1
        print(f"Config {config_count}: {config}")
        configs.append(config)

    print(f"\n总共生成了 {config_count} 个配置")

    return configs

def next_power_of_2(n):
    if isinstance(n, torch.Tensor):
        return torch.pow(2, torch.ceil(torch.log2(n.float()))).int()
    else:
        # 使用位操作，效率最高
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32  # 处理大整数
        return n + 1

k100_ai_config_lists = generate_config_lists() if mode_autotune else [ ]

def fused_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int4_w4a8: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    c_sorted: bool = False,
    bottom_a_use_mls_load: bool = False,
    naive_block_assignment: bool = False,
    scale_bias_with_routed_weight: bool = False,
    B_bias: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert (block_shape is None
                or triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2))
        assert (block_shape is None
                or triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1))

    elif use_int8_w8a16 or use_int4_w4a16 or use_int4_w4a8:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    total_tokens = A.size(0)
    num_tokens = topk_ids.numel()
    sorted_weights = None
    a_sorted = bool(bottom_a_use_mls_load)
    bottom_a_use_mls_load = a_sorted and capMLS
    if naive_block_assignment:
        assert not a_sorted and not c_sorted

    if config is None:
       assert "BLOCK_SIZE_M need be set when autotune for moe!"

    if naive_block_assignment:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    else:
        EM = sorted_token_ids.size(0)
    if not naive_block_assignment and A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N'] * META.get('N_TILES_PER_CTA', 1)), )

    if (use_int8_w8a16 or use_int4_w4a16 or use_int4_w4a8) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        offset_max = 2**32 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_b = B.numel() * B.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max

        if use_int4_w4a8:
            moe_op.fused_moe_kernel_gptq_awq_w4a8[grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                sorted_weights,
                expert_ids,
                num_tokens_post_padded,
                B.size(1),
                A.size(1),
                EM,
                num_tokens,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0),
                B_scale.stride(2),
                B_scale.stride(1),
                B_zp.stride(0) if B_zp is not None else 0,
                B_zp.stride(2) if B_zp is not None else 0,
                B_zp.stride(1) if B_zp is not None else 0,
                group_k=block_shape[1],
                group_size=block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                has_zp=B_zp is not None,
                use_int4_w4a16=use_int4_w4a16,
                use_int4_w4a8=use_int4_w4a8,
                use_int8_w8a16=use_int8_w8a16,
                ck_sorting=False,
                ck_topk=8,
                NUM_XCDS=1,
                **config
            )
            return

        moe_op.fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            expert_ids,
            num_tokens_post_padded,
            B.size(1),
            A.size(1),
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
            USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
            USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            ck_sorting=False,
            ck_topk=8,
            NUM_XCDS=1,
            **config,
        )
    else:
        offset_max = 2**32 - 1
        use_addr_offset_int64_a = A.numel() * A.element_size() >= offset_max
        use_addr_offset_int64_c = C.numel() * C.element_size() >= offset_max
        use_addr_offset_int64_b = B.numel() * B.element_size() >= offset_max

        if use_int4_w4a8 and per_channel_quant:
            assert B_scale is not None and B_scale.ndim in (2, 3)
            assert B_zp is None
            channelwise_config = config.copy()
            block_size_k = channelwise_config.pop("BLOCK_SIZE_K", None)
            channelwise_config.pop("USE_MLS_LOAD", None)
            channelwise_config.pop("COMBINE_SCALE_LOAD", None)
            w4a8_grid = lambda META: (
                triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
            )
            channelwise_kwargs = dict(
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_int4_w4a8=use_int4_w4a8,
                ck_sorting=False,
                ck_topk=8,
                NUM_XCDS=1,
                **channelwise_config,
            )
            if block_size_k is not None:
                channelwise_kwargs["BLOCK_SIZE_K"] = block_size_k
            return moe_op.fused_moe_kernel_gptq_awq_w4a8_channelwise[w4a8_grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                sorted_weights,
                expert_ids,
                num_tokens_post_padded,
                B.size(1),
                A.size(1),
                EM,
                num_tokens,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0),
                B_scale.stride(1),
                **channelwise_kwargs,
            )

        SPLIT_K = splitk_size # 0
        if mul_routed_weight or A.size(0) > 32:
            SPLIT_K = 0

        if SPLIT_K != 0 and not naive_block_assignment:
            grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
                    B.shape[1], META['BLOCK_SIZE_N']), SPLIT_K)
            assert B.size(2) % (256 * SPLIT_K) == 0, "B.size(2) must be divisible by BLOCK_SIZE_K * SPLIT_K"

            splitk_cache = torch.zeros((SPLIT_K,) + C.shape, device=C.device,dtype=C.dtype)

            use_addr_offset_int64_c = C.numel() * C.element_size() * SPLIT_K >= offset_max

            moe_op.fused_moe_splitk_kernel[grid](
                A,
                B,
                splitk_cache,
                topk_weights,
                num_tokens_post_padded,
                expert_ids,
                sorted_token_ids,
                sorted_weights,
                A_scale,
                B_scale,
                B.size(1),
                B.size(2),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                splitk_cache.stride(-3),
                splitk_cache.stride(-2),
                splitk_cache.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2)
                if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                A.size(0),
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                SPLIT_K=SPLIT_K,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                c_sorted=c_sorted,
                a_sorted=a_sorted,
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=False,
                ck_topk=8,
                NUM_XCDS=1,
                **config,
            )
            C.copy_(torch.sum(splitk_cache.to(torch.float32), dim=0).to(C.dtype))
        else:
            moe_op.fused_moe_kernel[grid](
                A,
                B,
                C,
                topk_weights,
                num_tokens_post_padded,
                expert_ids,
                sorted_token_ids,
                sorted_weights,
                A_scale,
                B_scale,
                (B_bias if B_bias is not None else B),
                B.size(1),
                B.size(2),
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1)
                if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2)
                if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1)
                if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_bias.stride(0) if B_bias is not None else B.stride(0),
                B_bias.stride(1) if B_bias is not None else B.stride(1),
                A.size(0),
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                USE_ADDR_OFFSET_INT64_A=use_addr_offset_int64_a,
                USE_ADDR_OFFSET_INT64_B=use_addr_offset_int64_b,
                USE_ADDR_OFFSET_INT64_C=use_addr_offset_int64_c,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                c_sorted=c_sorted,
                a_sorted=a_sorted,
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=False,
                ck_topk=8,
                naive_block_assignment=naive_block_assignment,
                NUM_XCDS=1,
                SCALE_BIAS_WITH_ROUTED_WEIGHT=scale_bias_with_routed_weight,
                ADD_BIAS=B_bias is not None,
                **config,
            )

def fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       output_dtype: Optional[torch.dtype] = None,
                       inplace: bool = False,
                       activation: str = "silu",
                       is_gated: Optional[bool] = None,
                       b1: Optional[torch.Tensor] = None,
                       b2: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       use_int4_w4a8: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       no_combine: bool = False,
                       routed_scaling_factor: Optional[float] = 1.0,
                       gemm1_alpha: Optional[float] = None,
                       gemm1_limit: Optional[float] = None,
                       activation_situ_beta: Optional[float] = None,
                       activation_situ_linear_beta: Optional[float] = None):
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    activation, is_gated = _normalize_activation_and_gate(activation, is_gated)
    activation_out_dim = adjust_N_for_activation(w1.shape[1], is_gated)

    # Check constraints.
    if use_int4_w4a16 or use_int4_w4a8:
        if is_gated:
            assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
        else:
            assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16, torch.int8,  torch.float8_e4m3fn
    ]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 65536
    M = min(num_tokens, CHUNK_SIZE)


    # bottom_a_use_mls_load selects the sorted GEMM1-to-GEMM2 bridge even on
    # devices without MLS, where GEMM2 reads it with ordinary buffer loads.
    # Reserve the sorted padding independently of the hardware MLS capability.
    can_use_bottom_a_sorted = not use_int4_w4a8


    topk = top_k_num
    max_total_tokens = _max_autotune_total_tokens(
        M,
        topk,
        E,
        allow_sorted_bridge=can_use_bottom_a_sorted,
    )


    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    # cache13 needs to be large enough for intermediate_cache1 which can be up to
    # max_total_tokens * N due to padding, and also for intermediate_cache3
    if output_dtype is None:
        output_dtype = _default_output_dtype(hidden_states.dtype)
    if output_dtype not in _ALLOWED_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output_dtype: {output_dtype}, expected one of {_ALLOWED_OUTPUT_DTYPES}"
        )
    compute_type = _compute_type_from_output_dtype(output_dtype)
    intermediate_dtype = output_dtype

    cache13 = torch.zeros(max_total_tokens * max(N, w2.shape[1]),
                          device=hidden_states.device,
                          dtype=intermediate_dtype)
    # intermediate_cache1 = cache13[:M * top_k_num * N].view(
    #     (M, topk_ids.shape[1], N))
    intermediate_cache3 = cache13[:M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]))

    if no_combine:
        assert not inplace, "no_combine + inplace is not supported"
        out_hidden_states = torch.zeros(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=intermediate_dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.zeros(hidden_states.shape, device=hidden_states.device, dtype=intermediate_dtype)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        # autotune
        config_count = 0
        num_tokens = curr_hidden_states.shape[0]
        num_tokens_aligned = next_power_of_2(num_tokens)

        # 当 num_tokens >= 32 时，以相反顺序遍历配置列表, 以加速training
        config_list = list(reversed(k100_ai_config_lists)) if num_tokens >= 32 else k100_ai_config_lists
        for config in config_list:
            config_count += 1
            print(f"Config {config_count}: {config}")
            bottom_a_use_mls_load = bool(
                config.get("bottom_a_use_mls_load", False)
            )
            kernel_config = {"BLOCK_SIZE_M": config["BLOCK_SIZE_M"]}

            # when compile_only=True, num_tokens == 1 we also check and prune configs
            if not compile_only:
                if (config['BLOCK_SIZE_M'] > max(num_tokens_aligned, 16)):
                    print(f"{config} skiped, {num_tokens_aligned=}")
                    continue
                # eg. top_k_num = 8, E = 256
                # 16: 16*8*1/256 = 1
                # 32: 32*8*2/256 = 2
                # 64: 64*8*2/256 = 4
                # 128: 128*8*2/256 = 8
                # 256: 256*8*2/256 = 16
                # 512: 512*8*2/256 = 32
                # 1024: 1024*8*2/256 = 64
                # 2048: 2048*8*2/256 = 128
                # 4096: 4096*8*2/256 = 256
                # 8192: 8192*8*2/256 = 512
                # 16384: 16384*8*2/256 = 1024
                # For top_k=1 with many experts, keep both the small M tiles
                # needed by sparse per-expert routing and BLOCK_SIZE_M=128,
                # the smallest PP-capable M tile.  The route-density estimate
                # alone can otherwise remove every PP candidate.
                min_block_m_ceiling = (
                    128
                    if top_k_num == 1 and num_tokens >= 2048
                    else 32
                )
                tune_block_size_m_max = max(
                    num_tokens_aligned * top_k_num * 2 / E,
                    min_block_m_ceiling,
                )
                if (config['BLOCK_SIZE_M'] > tune_block_size_m_max):
                    print(f"{config} skiped, {tune_block_size_m_max=}")
                    continue
                if (
                    num_tokens >= 2048
                    and top_k_num > 1
                    and config['BLOCK_SIZE_M'] <= 32
                ):
                    print(f"{config} skiped, {num_tokens=}")
                    continue

            padded_tokens = (
                min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
                if bottom_a_use_mls_load
                else 0
            )

            total_tokens = tokens_in_chunk * topk + padded_tokens
            intermediate_cache1 = cache13[: total_tokens * N].view(
                (total_tokens, N),
            )
            intermediate_cache2 = torch.zeros(
                    (total_tokens, activation_out_dim),
                    device=hidden_states.device,
                    dtype=intermediate_dtype,
                )

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            num_routes = tokens_in_chunk * topk
            naive_block_assignment = _should_use_naive_block_assignment(
                num_routes=num_routes,
                global_num_experts=global_num_experts,
                has_expert_map=expert_map is not None,
                bottom_a_sorted=bottom_a_use_mls_load,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                use_int4_w4a8=use_int4_w4a8,
                use_block_quant=block_shape is not None,
            )
            if naive_block_assignment:
                assert curr_topk_weights.is_contiguous(), (
                    "Top-k weights must be contiguous for naive block assignment"
                )
                route_ids = curr_topk_ids.contiguous().view(-1)
                sorted_token_ids = route_ids
                expert_ids = route_ids
                num_tokens_post_padded = route_ids
            else:
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(
                        curr_topk_ids,
                        config["BLOCK_SIZE_M"],
                        global_num_experts,
                        expert_map,
                    )
                )

            if (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8) and per_channel_quant:
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype == torch.bfloat16:
                    input_q, input_scale = per_token_quant_triton(curr_hidden_states, quant_dtype=quant_dtype)
                else:
                    input_q, input_scale = curr_hidden_states, a1_scale
                fused_moe(input_q,
                            w1,
                            intermediate_cache1,
                            input_scale,
                            w1_scale,
                            w1_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            apply_router_weight_on_input,
                            top_k_num,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            c_sorted=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            B_bias=b1,
                            config=kernel_config)
            elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype == torch.bfloat16:
                    input_q, input_scale = per_block_quant_wrapper((1, block_shape[1]))(per_token_quant_triton)(curr_hidden_states, quant_dtype=quant_dtype)
                else:
                    input_q, input_scale = curr_hidden_states, a1_scale
                fused_moe(input_q,
                            w1,
                            intermediate_cache1,
                            input_scale,
                            w1_scale,
                            w1_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            apply_router_weight_on_input,
                            top_k_num,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            c_sorted=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            B_bias=b1,
                            config=kernel_config)
            else:
                fused_moe(curr_hidden_states,
                            w1,
                            intermediate_cache1,
                            a1_scale,
                            w1_scale,
                            w1_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            apply_router_weight_on_input,
                            top_k_num,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            c_sorted=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            B_bias=b1,
                            config=kernel_config)
            _apply_activation(
                activation=activation,
                is_gated=is_gated,
                activated_out=intermediate_cache2,
                ffn1_out_2d=intermediate_cache1.view(-1, N),
                gemm1_alpha=gemm1_alpha,
                gemm1_limit=gemm1_limit,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
            # intermediate_cache1 and intermediate_cache3 share cache13.
            intermediate_cache1.fill_(0)
            if (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8) and per_channel_quant:
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if intermediate_cache2.dtype == torch.float16 or intermediate_cache2.dtype == torch.bfloat16:
                    bridge_q, bridge_scale = per_token_quant_triton(intermediate_cache2, quant_dtype=quant_dtype)
                else:
                    bridge_q, bridge_scale = intermediate_cache2, a2_scale
                fused_moe(bridge_q,
                            w2,
                            intermediate_cache3,
                            bridge_scale,
                            w2_scale,
                            w2_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            (not apply_router_weight_on_input) and (not no_combine),
                            1,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            bottom_a_use_mls_load=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=kernel_config)
            elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if intermediate_cache2.dtype == torch.float16 or intermediate_cache2.dtype == torch.bfloat16:
                    bridge_q, bridge_scale = per_block_quant_wrapper((1, block_shape[1]))(per_token_quant_triton)(intermediate_cache2, quant_dtype=quant_dtype)
                else:
                    bridge_q, bridge_scale = intermediate_cache2, a2_scale
                fused_moe(bridge_q,
                            w2,
                            intermediate_cache3,
                            bridge_scale,
                            w2_scale,
                            w2_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            (not apply_router_weight_on_input) and (not no_combine),
                            1,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            bottom_a_use_mls_load=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=kernel_config)
            else:
                fused_moe(intermediate_cache2,
                            w2,
                            intermediate_cache3,
                            a2_scale,
                            w2_scale,
                            w2_zp,
                            curr_topk_weights,
                            curr_topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            (not apply_router_weight_on_input) and (not no_combine),
                            1,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            use_int4_w4a8=use_int4_w4a8,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape,
                            bottom_a_use_mls_load=bottom_a_use_mls_load,
                            naive_block_assignment=naive_block_assignment,
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=kernel_config)


            if no_combine:
                out_hidden_states[begin_chunk_idx:end_chunk_idx].copy_(intermediate_cache3)
            else:
                triton_moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor=routed_scaling_factor,
                )
            if end_chunk_idx < hidden_states.shape[0]:
                intermediate_cache3.fill_(0)
    return out_hidden_states


@contextmanager
def patched_environment():
    patchers = []
    try:
        if mode_autotune:
            patchers.append(patch(
                'boltops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq)
            ))
            patchers.append(patch(
                'boltops.fused_moe.triton.moe_op.fused_moe_kernel',
                new=dynamic_autotune(moe_op.fused_moe_kernel)
            ))
            patchers.append(patch(
                'boltops.fused_moe.triton.moe_op.fused_moe_splitk_kernel',
                new=dynamic_autotune(moe_op.fused_moe_splitk_kernel)
            ))
            patchers.append(patch(
                'boltops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq_w4a8',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq_w4a8)
            ))
            patchers.append(patch(
                'boltops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq_w4a8_channelwise',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq_w4a8_channelwise)
            ))
            patchers.append(patch(
                'boltops.fused_moe.triton.fused_experts_impl',
                new=fused_experts_impl
            ))
            for patcher in patchers:
                patcher.start()
            print("All patches applied successfully")
        yield
    finally:
        for patcher in reversed(patchers):
            try:
                patcher.stop()
            except RuntimeError:
                pass
        if patchers:
            print("All patches reverted successfully")
