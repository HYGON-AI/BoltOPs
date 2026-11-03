# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import os
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import functools
import importlib
import inspect
from itertools import product
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from boltops.tools.fused_moe_triton_tune.common import autotune_outer_configs

import boltops.fused_moe.triton.moe_op as moe_op
from boltops.fused_moe.triton import mxfp4, mxfp8
import boltops.utility.arch_info as arch_info
from boltops.utility.triton_capability import get_triton_capabilities


capMLS = moe_op.support_mls()
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

_SCHEDULE_OPTION_NAMES = ("instruction_sched_variant", "schedule_hint")
_LARGE_MOE_TOKEN_THRESHOLD = 2048


def _check_hip_loads(do_bench):
    """Never publish a partial ranking after process-wide HIP resource failure."""

    def robust_do_bench(kernel_call, quantiles):
        try:
            return do_bench(kernel_call, quantiles)
        except RuntimeError as exc:
            message = str(exc)
            if "named symbol not found" not in message:
                raise
            # This also occurs when HIP exhausts this process's VMA budget,
            # not just for an invalid HSACO. Mapping it to HSACOError/infinity
            # silently excludes every later candidate and biases the winner.
            raise RuntimeError(
                "HIP module lookup failed; restart this autotune shard in a fresh "
                f"process instead of accepting a partial ranking: {message}"
            ) from exc

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
        # Wide FP8-channel Gemm1 exposes a gfx938 MLS runtime fault at decode
        # sizes (first reproduced by E384/N768/K7168, whose Gemm1 N is 1536).
        # Ordinary loads remain legal, and narrower TP8 Gemm1 keeps MLS.
        if test_type == "fp8_channel" and num_tokens < 128 and n >= 1536:
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

    # Wide-N tiles with 8/16 warps also VMFault for small FP8-channel inputs,
    # while their 2/4-warp variants remain usable.
    if (
        test_type == "fp8_channel"
        and num_tokens < 128
        and block_m < 64
        and block_n >= 256
        and num_warps >= 8
    ):
        return False
    # On gfx938, small-M FP8-channel BM16/BN256/BK256 VMFaults across several
    # compiler-option variants. Device faults are reported asynchronously, so
    # the individual triggering variant cannot be isolated safely; drop the
    # oversized reduction tile for this small-M regime.
    if (
        test_type == "fp8_channel"
        and num_tokens < 128
        and block_m < 64
        and block_n >= 256
        and block_k >= 256
    ):
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
    if (runtime_args or {}).get("use_mxfp8", False):
        # Only the regular non-combined loop has native FP8 scaled dot.
        return [triton.Config(dict(BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                                  GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                                  USE_MLS_LOAD=False), num_warps=4, num_stages=1)
                for index, bn in enumerate((64, 128))
                if not compile_only or index % num_groups_compile == cur_group]
    phase = (runtime_args or {}).get("_autotune_is_bottom")
    is_gemm1 = not phase if phase is not None else not MUL_ROUTED_WEIGHT
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

    if reuse_args.get("use_int4_w4a8") and reuse_args.get("group_k", 0) > 0:
        # This kernel dequantizes one complete weight/activation group per
        # iteration; smaller tiles are not legal alternatives to the group.
        block_size_k_options = [reuse_args["group_k"]]

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
                buffer_cache_swizzle_options: {buffer_cache_swizzle_options}")

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
        self.__name__ = getattr(fn, "__name__", "")
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


def _prune_configs_by_runtime_constraints(configs, runtime):
    """Filter candidates using the same tile/layout constraints as the GEMM."""
    N, K = runtime["N"], runtime["K"]
    group_k = runtime.get("group_k", 0)
    split = int(runtime.get("SPLIT_K", 1))
    min_mls_block_k = {
        "bf16": 32, "fp8": 64, "fp8_channel": 64,
        "int8": 64, "int8_channel": 64,
    }.get(test_type, 0)
    pruned = []
    for cfg in configs:
        cfg_kwargs = cfg.kwargs
        bk = cfg_kwargs["BLOCK_SIZE_K"]
        bn = cfg_kwargs["BLOCK_SIZE_N"]
        if not runtime.get("use_mxfp8", False) and group_k > 0 and (bk > group_k or group_k % bk):
            continue
        if runtime.get("use_int4_w4a8") and group_k > 0:
            if bk != runtime.get("group_size") or K % bk:
                continue
        if runtime.get("use_mxfp8", False) and (
                cfg_kwargs.get("COMBINE_SCALE_LOAD", False) or bk % 32
                or cfg_kwargs.get("N_TILES_PER_CTA", 1) != 1):
            continue
        if cfg_kwargs.get("COMBINE_SCALE_LOAD", False):
            if (bk != group_k or K % (2 * bk)
                    or runtime.get("stride_ask") != 1
                    or runtime.get("MUL_ROUTED_WEIGHT", False)
                    or not (runtime.get("use_fp8_w8a8", False)
                            or runtime.get("use_int8_w8a8", False))):
                continue
        if cfg_kwargs.get("USE_MLS_LOAD", False) and (K % bk or N % bn):
            continue
        if cfg_kwargs.get("bottom_a_use_mls_load", False) and (
                not cfg_kwargs.get("a_sorted", runtime.get("a_sorted", False))
                or runtime.get("c_sorted", False)):
            continue
        if split > 1 and K % (cfg_kwargs["BLOCK_SIZE_K"] * split):
            continue
        if cfg_kwargs.get("N_TILES_PER_CTA") == 2 and not _keep_n_tile_reuse(
            cfg_kwargs, runtime, cfg.num_warps
        ):
            continue
        uses_mls = (cfg_kwargs.get("USE_MLS_LOAD", False)
                    or cfg_kwargs.get("bottom_a_use_mls_load", False))
        if uses_mls and bk < min_mls_block_k:
            continue
        pruned.append(cfg)
    return pruned


def dynamic_autotune(fn):
    class SubscriptableKernel:
        def __init__(self, kernel_fn):
            self.kernel_fn = kernel_fn

        def __getitem__(self, grid):
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

                BLOCK_SIZE_M = kwargs.get('BLOCK_SIZE_M', 16)

                MUL_ROUTED_WEIGHT = kwargs.get('MUL_ROUTED_WEIGHT', False)
                per_channel_quant = kwargs.get('per_channel_quant', False) or (
                    "channelwise" in kernel_name
                )

                bottom_a_sorted = bool(
                    kwargs.get('a_sorted', False)
                    or kwargs.get('bottom_a_use_mls_load', False)
                )

                BLOCK_SIZE_M = 16 if BLOCK_SIZE_M is None else BLOCK_SIZE_M
                N = 16 if N is None else N
                K = 16 if K is None else K

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
                    runtime_args={**named_args, **kwargs, "N": N, "K": K,
                                  "_autotune_is_bottom": _tuning_phase.get()},
                )
                for cfg in dynamic_configs:
                    # Layout flags are host decisions. Only expose constexprs
                    # implemented by this kernel (WNA16 has no sorted-A ABI).
                    for name in ("bottom_a_use_mls_load", "a_sorted", "USE_MLS_LOAD",
                                 "COMBINE_SCALE_LOAD"):
                        if name not in arg_names:
                            cfg.kwargs.pop(name, None)
                # Filter before constructing Autotuner: its single-config run
                # bypasses early_config_prune, while warmup does not.
                dynamic_configs = _prune_configs_by_runtime_constraints(
                    dynamic_configs, {**named_args, **kwargs, "N": N, "K": K})
                if not dynamic_configs:
                    if compile_only:
                        # A compile shard can legitimately receive no candidates.
                        return []
                    raise RuntimeError(
                        f"No legal autotune candidates for {kernel_name}: "
                        f"N={N}, K={K}, BLOCK_SIZE_M={BLOCK_SIZE_M}, "
                        f"a_sorted={bottom_a_sorted}"
                    )
                launch_kwargs = dict(kwargs)
                # The production launcher supplies a complete baseline config.
                # Candidate-owned fields must have a single source of truth.
                for cfg in dynamic_configs:
                    for name in cfg.all_kwargs():
                        launch_kwargs.pop(name, None)
                autotuned_fn = triton.autotune(
                    configs=dynamic_configs,
                    key=[name for name in ('EM', 'N', 'K', 'num_valid_tokens',
                         'MUL_ROUTED_WEIGHT', 'BLOCK_SIZE_M', 'per_channel_quant',
                         'use_mxfp8', 'group_n', 'group_k')
                         if name in arg_names],
                    perf_debug=True,
                    perf_profiling=False,
                    rep=20
                )(self.kernel_fn)
                autotuned_fn.__dict__["do_bench"] = _check_hip_loads(
                    autotuned_fn.do_bench
                )

                print(f"DEBUG - {compile_only=}, kwargs keys being passed to kernel:{list(launch_kwargs.keys())}")
                if compile_only:
                    return autotuned_fn.warmup(*args, grid=grid, **launch_kwargs)
                else:
                    result = autotuned_fn[grid](*args, **launch_kwargs)
                    if len(dynamic_configs) == 1:
                        _ensure_single_config_timing(autotuned_fn, lambda: autotuned_fn[grid](*args, **launch_kwargs))
                        _print_best_config(autotuned_fn)
                    return result

            return launcher

    return SubscriptableKernel(apply_my_heuristics(fn))


def generate_config_lists():
    configs = []
    config_count = 0

    if mode_best_config_tune == False:
        for config in autotune_outer_configs(
            test_type, os.environ.get("AUTOTUNE_OUTER_CONFIG")
        ):
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


# Keep orchestration in the production module: quantization, activation, router
# folding, chunking and Split-K must not acquire a second implementation here.
_production = importlib.import_module("boltops.fused_moe.triton.fused_moe")
_production_experts = _production.fused_experts_impl
# The custom-op wrapper hides its Python signature. Bind against its actual
# registered schema, so new public parameters/defaults stay in sync.
_experts_signature = inspect.Signature([
    inspect.Parameter(argument.name,
                      inspect.Parameter.KEYWORD_ONLY if argument.kwarg_only
                      else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      default=argument.default_value if argument.has_default_value()
                      else inspect.Parameter.empty)
    for argument in torch.ops.boltops.fused_experts_impl.default._schema.arguments
])
_production_gemm = moe_op.fused_moe
_tuning_phase = ContextVar("moe_tuning_is_bottom", default=None)


def _ensure_single_config_timing(tuner, launch):
    # Triton bypasses benchmarking when only one candidate survives a shard.
    if not hasattr(tuner, "configs_timings"):
        tuner.configs_timings = {
            tuner.best_config: tuner.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        }


def _print_best_config(autotuner, *, native_mxfp4=False):
    """Emit the historical parser ABI, independently of Triton's debug format."""
    config = autotuner.best_config
    values = dict(config.kwargs)
    if native_mxfp4:
        values = dict(BLOCK_SIZE_N=values["BN"], BLOCK_SIZE_K=128,
                      GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False)
    values.update(num_warps=config.num_warps, num_stages=config.num_stages)
    timing = autotuner.configs_timings[config]
    milliseconds = timing[0] if isinstance(timing, (list, tuple)) else timing
    print("fused_moe_kernel:")
    print("* 0: " + ", ".join(f"{key}: {value}" for key, value in values.items()))
    print(f"  : times(best): {milliseconds} ms")


def _native_mxfp4_autotune(kernel):
    class NativeKernel:
        def __getitem__(self, _grid):
            def launch(*args, **kwargs):
                named = dict(zip(kernel.arg_names, args))
                named.update(kwargs)
                named.pop("BN", None)
                named.pop("num_warps", None)
                named.pop("num_stages", None)
                configs = [triton.Config({"BN": bn}, num_warps=4, num_stages=1)
                           for index, bn in enumerate((64, 128))
                           if not compile_only or index % num_groups_compile == cur_group]
                if not configs:
                    if compile_only:
                        return []
                    raise RuntimeError("MXFP4 tuning shard has no candidates")
                tuner = triton.autotune(
                    configs=configs,
                    key=["N", "K", "BM", "A_GROUP_SIZE", "A_FORMAT", "B_FORMAT",
                         "MUL_ROUTED_WEIGHT", "STATIC_A", "FUSE_SWIGLU", "ALPHA", "LIMIT"],
                    rep=20,
                )(kernel)
                tuner.__dict__["do_bench"] = _check_hip_loads(tuner.do_bench)
                output_n = named["N"] // 2 if named.get("FUSE_SWIGLU", False) else named["N"]
                grid = lambda meta: (_grid[0], triton.cdiv(output_n, meta["BN"]))
                if compile_only:
                    return tuner.warmup(grid=grid, **named)
                result = tuner[grid](**named)
                _ensure_single_config_timing(tuner, lambda: tuner[grid](**named))
                _print_best_config(tuner, native_mxfp4=True)
                return result
            return launch
    return NativeKernel()


def _tuning_config(base, outer, *, is_bottom, supports_splitk=False, k=None):
    config = base.copy()
    config.update(BLOCK_SIZE_M=outer["BLOCK_SIZE_M"], N_TILES_PER_CTA=1,
                  _autotune_is_bottom=is_bottom)
    # Sorted bridge changes both GEMMs, so it is chosen by the outer search.
    sorted_bridge = bool(outer.get("bottom_a_use_mls_load", False))
    config.pop("a_sorted", None)
    config.pop("bottom_a_use_mls_load", None)
    if is_bottom:
        config.update(a_sorted=sorted_bridge, bottom_a_use_mls_load=sorted_bridge)
    # Candidate BK belongs to the kernel search; allocate Split-K partials
    # through the production launcher, with a baseline dividing K.
    config.pop("SPLIT_K", None)
    if supports_splitk:
        config["SPLIT_K"] = splitk_size if not is_bottom else 0
        if config["SPLIT_K"] > 1:
            if k is None or k % splitk_size:
                raise ValueError("MoE K must be divisible by SPLITK_SIZE")
            while k % (config["BLOCK_SIZE_K"] * splitk_size):
                config["BLOCK_SIZE_K"] //= 2
    return config


@functools.wraps(_production_experts)
def fused_experts_impl(*args, **kwargs):
    if not mode_autotune:
        return _production_experts(*args, **kwargs)
    bound = _experts_signature.bind(*args, **kwargs)
    bound.apply_defaults()
    hidden = bound.arguments["hidden_states"]
    native = bound.arguments["use_mxfp4_w4a8"] or bound.arguments["use_mxfp8"]
    supports_splitk = not any(bound.arguments[name] for name in (
        "use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16",
        "use_int4_w4a16", "use_int4_w4a8", "use_mxfp4_w4a8", "use_mxfp8"))
    original_query = _production.get_optimal_moe_config_func
    native_module = mxfp8 if bound.arguments["use_mxfp8"] else mxfp4
    native_name = "get_mxfp8_moe_config" if bound.arguments["use_mxfp8"] else "get_mxfp4_moe_config"
    original_native_query = getattr(native_module, native_name)
    configs = k100_ai_config_lists
    if native:
        configs = autotune_outer_configs(test_type, os.environ.get("AUTOTUNE_OUTER_CONFIG"))
    if hidden.shape[0] >= 32:
        configs = list(reversed(configs))
    # Repeated candidates must receive the same original input for inplace.
    saved_input = hidden.clone() if bound.arguments["inplace"] else None
    output = None
    index = 0
    for outer in configs:
        if not compile_only and outer["BLOCK_SIZE_M"] > max(next_power_of_2(hidden.shape[0]), 16):
            continue
        index += 1
        print(f"Config {index}: {outer}")

        def query(*query_args, **query_kwargs):
            base_query = original_query(*query_args, **query_kwargs)
            bottom = query_kwargs.get("is_bottom", False)
            def select(tokens):
                selected = base_query(tokens)
                base = selected[0] if bottom else selected
                config = _tuning_config(
                    base, outer, is_bottom=bottom, supports_splitk=supports_splitk,
                    k=bound.arguments["w1"].shape[-1])
                return (config, outer["BLOCK_SIZE_M"]) if bottom else config
            return select

        def native_query(tokens, *, is_bottom=False, **selectors):
            selected = original_native_query(tokens, is_bottom=is_bottom, **selectors)
            base = selected[0] if is_bottom else selected
            config = _tuning_config(base, outer, is_bottom=is_bottom)
            return (config, outer["BLOCK_SIZE_M"]) if is_bottom else config

        with ExitStack() as stack:
            stack.enter_context(patch.object(_production, "get_optimal_moe_config_func", query))
            stack.enter_context(patch.object(native_module, native_name, native_query))
            if saved_input is not None:
                hidden.copy_(saved_input)
            output = _production_experts(**bound.arguments)
    if output is None:
        raise RuntimeError("No legal outer MoE autotune configurations")
    return output


@functools.wraps(_production_gemm)
def fused_moe(*args, **kwargs):
    """Attach explicit GEMM phase without changing the production kernel ABI."""
    config = kwargs.get("config")
    phase = None
    if config is not None:
        config = config.copy()
        phase = config.pop("_autotune_is_bottom", None)
        kwargs["config"] = config
    token = _tuning_phase.set(phase)
    try:
        return _production_gemm(*args, **kwargs)
    finally:
        _tuning_phase.reset(token)


@contextmanager
def patched_environment():
    """Patch both public and legacy package entries; always restore on exit."""
    with ExitStack() as stack:
        if mode_autotune:
            for name in ("fused_moe_kernel_gptq_awq", "fused_moe_kernel",
                         "fused_moe_splitk_kernel", "fused_moe_persistent_kernel",
                         "fused_moe_kernel_gptq_awq_w4a8",
                         "fused_moe_kernel_gptq_awq_w4a8_channelwise"):
                stack.enter_context(patch.object(moe_op, name, dynamic_autotune(getattr(moe_op, name))))
            stack.enter_context(patch.object(
                mxfp4, "_moe_gemm_mxfp4_kernel",
                _native_mxfp4_autotune(mxfp4._moe_gemm_mxfp4_kernel),
            ))
            stack.enter_context(patch.object(moe_op, "fused_moe", fused_moe))
            stack.enter_context(patch.object(_production, "invoke_fused_moe_kernel", fused_moe))
            stack.enter_context(patch.object(_production, "fused_experts_impl", fused_experts_impl))
            stack.enter_context(patch("boltops.fused_moe.triton.fused_experts_impl", fused_experts_impl))
        yield
