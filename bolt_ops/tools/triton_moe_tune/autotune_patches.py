import os
import torch
from typing import Optional, List, Dict, Any
from unittest.mock import patch
import triton
import triton.language as tl
import bolt_ops.fused_moe.triton.moe_op as moe_op
import bolt_ops.utility.arch_info as arch_info
from bolt_ops.fused_moe.triton.moe_compat import per_token_quant_hip, per_block_quant_wrapper
from bolt_ops.fused_moe.triton import (
    triton_moe_sum,
    moe_align_block_size,
)
from bolt_ops.fused_moe.triton.moe_activation import (
    _normalize_activation_and_gate,
    adjust_N_for_activation,
    _apply_activation,
)
from contextlib import contextmanager


capMLS = arch_info.get_arch() in ("gfx938", "gfx92a")
support_sched_latency = arch_info.get_arch() in ("gfx928", "gfx936", "gfx92a")

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
    mode_autotune_max_block_size_k2 = 128
    mode_autotune_min_block_size_n1 = 16
    mode_autotune_max_block_size_n1 = 128
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

def generate_config2_lists(BLOCK_SIZE_M=16, N=16, K=16, MUL_ROUTED_WEIGHT=False, per_channel_quant=False, bottom_a_use_mls_load=False):
    is_gemm1 = MUL_ROUTED_WEIGHT == False

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

    group_size_m_options = [1]
    num_stage_options = [1, 2]
    combine_scale_load_options = [False, True] if test_type in ["int8", "int4int8", "fp8"] else [False]

    use_mls_load_options = [False, True] if test_type in ["fp8", "fp8_channel", "int8", "int8_channel", "bf16"] and capMLS == True else [False]
    bottom_a_use_mls_load_options = [False, True] if bottom_a_use_mls_load else [False]

    # warp 1 no selected, so no need to tune
    num_warp_options = [2, 4, 8, 16]
    waves_per_eu = 1

    instruction_sched_variant_options = ["none", "local-prefetch"]
    sched_latency_options = ["none", "mmac5-ds10"] if support_sched_latency else ["none"]

    kpack_options = [1, 2] if test_type == "bf16" else [1]

    print(f"BLOCK_SIZE_M: {BLOCK_SIZE_M} is_gemm1: {is_gemm1} Autotune Config limit range: \
                block_size_n_options: {block_size_n_options}, \
                block_size_k_options: {block_size_k_options}, \
                group_size_m_options: {group_size_m_options}, \
                num_stage_options: {num_stage_options}, \
                combine_scale_load_options: {combine_scale_load_options}, \
                use_mls_load_options: {use_mls_load_options}, \
                bottom_a_use_mls_load_options: {bottom_a_use_mls_load_options}, \
                num_warp_options: {num_warp_options},   \
                waves_per_eu: {waves_per_eu}, \
                instruction_sched_variant_options: {instruction_sched_variant_options}, \
                sched_latency_options: {sched_latency_options}, \
                kpack_options: {kpack_options}")

    configs = []
    config_count = 0
    cur_group_config_count=0
    if mode_best_config_tune == False:

        for block_n in block_size_n_options:
            for block_k in block_size_k_options:
                for group_m in group_size_m_options:
                    for combine_scale_load in combine_scale_load_options:
                        for num_stage in num_stage_options:
                            for num_warp in num_warp_options:
                                for instruction_sched_variant in instruction_sched_variant_options:
                                    for sched_latency in sched_latency_options:
                                        for kpack in kpack_options:
                                            for use_mls_load in use_mls_load_options:
                                                for bottom_a_use_mls_load in bottom_a_use_mls_load_options:
                                                    if use_mls_load == True or bottom_a_use_mls_load == True:
                                                        if (num_stage >= 3):
                                                            continue
                                                        if N % block_n != 0 or K % block_k != 0:
                                                            continue

                                                    if per_channel_quant == True:
                                                        if combine_scale_load == True:
                                                            continue
                                                        if test_type in ["int4", "int4int8_channel"] and block_k < 32:
                                                            continue
                                                    if kpack == 2 and block_k <= 32:  # minBlockSizeK = waveSize/(waveSize/16) * kpack
                                                        continue
                                                    if test_type in ["int4", "int4int8_channel"] and block_k < 32:
                                                        continue
                                                    if block_k > K or block_n > N:
                                                        continue

                                                    if group_m > block_m:
                                                        continue

                                                    if combine_scale_load == True:
                                                        if K % (block_k * 2) != 0:
                                                            continue
                                                        if block_k != mode_autotune_max_block_size_k:
                                                            continue
                                                        if is_gemm1 == False:
                                                            continue

                                                        # not write the code support.
                                                        if bottom_a_use_mls_load == True:
                                                            continue

                                                    if num_stage == 1:
                                                        if instruction_sched_variant != "none":
                                                            # 1 stage时，instruction_sched_variant 在 lower_instruction_sched_hints pass 都不会被处理
                                                            # 会 被直接设置为 none，所以 这里不用添加这个 options，没有意义。
                                                            continue

                                                    # occupy 限制
                                                    byte_per_elem = 1 if test_type in ["int8", "int8_channel", "int4int8", "int4int8_channel", "fp8"] else 2
                                                    if num_stage == 1:
                                                        share_mem_size =  max(block_m * block_k * byte_per_elem, block_k * block_n * byte_per_elem)
                                                    else:
                                                        share_mem_size =  block_m * block_k * byte_per_elem * (1 + (num_stage-2)) + block_k * block_n * byte_per_elem * (1 + (num_stage-2))
                                                    if share_mem_size > 64 * 1024:
                                                        continue

                                                    # wave 重复工作限制
                                                    if block_m * block_n / (16 * 16) <=  num_warp/8:
                                                        continue
                                                    if block_n == 16 and num_warp > 4:
                                                        continue
                                                    # reg spill 限制
                                                    tile_per_wave = max(1, block_m * block_n / (16 * 16) / num_warp)
                                                    if (tile_per_wave > 32):
                                                        continue

                                                    config = {
                                                        "BLOCK_SIZE_N": block_n,
                                                        "BLOCK_SIZE_K": block_k,
                                                        "GROUP_SIZE_M": group_m,
                                                        "COMBINE_SCALE_LOAD": combine_scale_load,
                                                        "USE_MLS_LOAD": use_mls_load,
                                                        "bottom_a_use_mls_load": bottom_a_use_mls_load,
                                                        "waves_per_eu": waves_per_eu,
                                                        "instruction_sched_variant": instruction_sched_variant,
                                                        "sched_latency": sched_latency,
                                                        "kpack": kpack,
                                                    }
                                                    config_count += 1
                                                    if compile_only:
                                                        # groups in each batch need cover all configs
                                                        if (config_count - 1) % num_groups_compile != cur_group:
                                                            continue
                                                    cur_group_config_count += 1
                                                    print(f"Autotune Config for compile_group={cur_group} config_idx:{config_count}: BLOCK_SIZE_M:{BLOCK_SIZE_M} {config} {num_stage} {num_warp}")
                                                    triton_cfg = triton.Config(config, num_stages=num_stage, num_warps=num_warp)
                                                    configs.append(triton_cfg)
    else:
        # # # tune best config for gemm1 & bs=32
        if test_type == "fp8":
            if is_gemm1:
                triton_cfg = triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                                            "USE_MLS_LOAD": True,
                                            "instruction_sched_variant": "none"
                                           }, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)

            else:
                # # tune best config for gemm2 & bs=32:
                triton_cfg = triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                                            "USE_MLS_LOAD": True,
                                            "instruction_sched_variant": "none"
                                           }, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)

                triton_cfg = triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                                            "USE_MLS_LOAD": True,
                                            "instruction_sched_variant": "none"
                                           }, num_stages=2, num_warps=8)
                config_count += 1
                cur_group_config_count += 1
                configs.append(triton_cfg)
        else:
            triton_cfg = triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                                        "USE_MLS_LOAD": True,
                                        "instruction_sched_variant": "local-prefetch"}, num_stages=2, num_warps=4)
            config_count += 1
            cur_group_config_count += 1
            configs.append(triton_cfg)

            # # # tune best config for gemm2 & bs=32:
            triton_cfg = triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "COMBINE_SCALE_LOAD": False,
                                        "USE_MLS_LOAD": True,
                                        "instruction_sched_variant": "local-prefetch"}, num_stages=2, num_warps=4)
            config_count += 1
            cur_group_config_count += 1
            configs.append(triton_cfg)

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

                bottom_a_use_mls_load = kwargs.get('bottom_a_use_mls_load', False)

                # 确保所有参数都不是None
                BLOCK_SIZE_M = 16 if BLOCK_SIZE_M is None else BLOCK_SIZE_M
                N = 16 if N is None else N
                K = 16 if K is None else K

                # 动态生成配置列表
                dynamic_configs = generate_config2_lists(BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                            N=N,
                                                            K=K,
                                                            MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT,
                                                            per_channel_quant=per_channel_quant,
                                                            bottom_a_use_mls_load=bottom_a_use_mls_load)
                if "int4" in test_type:
                    for cfg in dynamic_configs:
                        cfg.kwargs.pop("bottom_a_use_mls_load", None)
                def _prune_configs_by_runtime_constraints(configs, named_args, **_kwargs):
                    pruned = []
                    for cfg in configs:
                        cfg_kwargs = cfg.kwargs
                        cfg_use_mls = bool(cfg_kwargs.get("USE_MLS_LOAD", False))
                        cfg_bottom_mls = bool(cfg_kwargs.get("bottom_a_use_mls_load", False))
                        cfg_block_k = int(cfg_kwargs.get("BLOCK_SIZE_K", 0))
                        # Any MLS path (kernel MLS load or bottom-A MLS load) must
                        # satisfy BLOCK_SIZE_K lower-bound constraints.
                        if cfg_use_mls or cfg_bottom_mls:
                            if test_type == "bf16" and cfg_block_k < 32:
                                continue
                            if test_type in ["fp8", "fp8_channel", "int8", "int8_channel"] and cfg_block_k < 64:
                                continue
                        pruned.append(cfg)
                    return pruned

                launch_kwargs = dict(kwargs)
                launch_kwargs.pop("bottom_a_use_mls_load", None)
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

        for block_m in block_size_m_options:
            config = {
                "BLOCK_SIZE_M": block_m,
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

    if config is None:
       assert "BLOCK_SIZE_M need be set when autotune for moe!"

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )

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

        if SPLIT_K != 0:
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
                bottom_a_use_mls_load=bottom_a_use_mls_load,
                ck_sorting=False,
                ck_topk=8,
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


    # Note: we always assume that use bottom_a_use_mls_load can gen good performance than not using it.
    # TODO: need split it but need solved gemm1 & gemm2 bind problem.
    bottom_a_use_mls_load = (capMLS and not use_int4_w4a8)


    max_block_m = 256
    topk = top_k_num
    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if bottom_a_use_mls_load else 0
    )
    max_total_tokens = M * topk + max_padded_tokens


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
                tune_block_size_m_max = max(num_tokens_aligned * top_k_num * 2 / E, 32)
                if (config['BLOCK_SIZE_M'] > tune_block_size_m_max):
                    print(f"{config} skiped, {tune_block_size_m_max=}")
                    continue
                if num_tokens >= 2048 and config['BLOCK_SIZE_M'] <= 32:
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

            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))

            if (use_int8_w8a8 or use_fp8_w8a8 or use_int4_w4a8) and per_channel_quant:
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype == torch.bfloat16:
                    input_q, input_scale = per_token_quant_hip(curr_hidden_states, quant_dtype=quant_dtype)
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
                            B_bias=b1,
                            config=config)
            elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if curr_hidden_states.dtype == torch.float16 or curr_hidden_states.dtype == torch.bfloat16:
                    input_q, input_scale = per_block_quant_wrapper((1, block_shape[1]))(per_token_quant_hip)(curr_hidden_states, quant_dtype=quant_dtype)
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
                            B_bias=b1,
                            config=config)
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
                            B_bias=b1,
                            config=config)
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
                    bridge_q, bridge_scale = per_token_quant_hip(intermediate_cache2, quant_dtype=quant_dtype)
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
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=config)
            elif block_shape is not None and (use_int8_w8a8 or use_int4_w4a8 or use_fp8_w8a8):
                quant_dtype = torch.float8_e4m3fn if use_fp8_w8a8 else torch.int8
                if intermediate_cache2.dtype == torch.float16 or intermediate_cache2.dtype == torch.bfloat16:
                    bridge_q, bridge_scale = per_block_quant_wrapper((1, block_shape[1]))(per_token_quant_hip)(intermediate_cache2, quant_dtype=quant_dtype)
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
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=config)
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
                            scale_bias_with_routed_weight=(not apply_router_weight_on_input) and (not no_combine),
                            B_bias=b2,
                            config=config)


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
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_kernel',
                new=dynamic_autotune(moe_op.fused_moe_kernel)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_splitk_kernel',
                new=dynamic_autotune(moe_op.fused_moe_splitk_kernel)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq_w4a8',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq_w4a8)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.moe_op.fused_moe_kernel_gptq_awq_w4a8_channelwise',
                new=dynamic_autotune(moe_op.fused_moe_kernel_gptq_awq_w4a8_channelwise)
            ))
            patchers.append(patch(
                'bolt_ops.fused_moe.triton.fused_experts_impl',
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
