# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch

import triton
import triton.language as tl
import time
import gc
import os
import sys

from typing import Optional, List  # Add this import at the top

from boltops.tools.fused_moe_triton_tune.routing import fused_topk
from boltops.utility import dtypes
import boltops.fused_moe.triton as fused_moe_module
try:
    from boltops.tools.fused_moe_triton_tune.common import (
        get_env_int,
        parse_batch_sizes_from_env,
        resolve_device_settings,
        build_compile_group_ids,
        build_run_group_ids,
        resolve_moe_activation_and_gate,
        resolve_situ_parameters,
        apply_activation_ref,
        resolve_patched_environment,
        enforce_training_patch_contract,
    )
except ImportError:
    try:
        from boltops.tools.fused_moe_triton_tune.common import (
            get_env_int,
            parse_batch_sizes_from_env,
            resolve_device_settings,
            build_compile_group_ids,
            build_run_group_ids,
            resolve_moe_activation_and_gate,
            resolve_situ_parameters,
            apply_activation_ref,
            resolve_patched_environment,
            enforce_training_patch_contract,
        )
    except ImportError:
        from boltops.tools.fused_moe_triton_tune.common import (
            get_env_int,
            parse_batch_sizes_from_env,
            resolve_device_settings,
            build_compile_group_ids,
            build_run_group_ids,
            resolve_moe_activation_and_gate,
            resolve_situ_parameters,
            apply_activation_ref,
            resolve_patched_environment,
            enforce_training_patch_contract,
        )

from boltops.utility.perf_utils import checkAllclose, perftest,benchmark
# os.environ["VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
# os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

os.environ["TRITON_FUSED_MOE_CHUNK_SIZE"] = "16384"
# 从环境变量获取配置，如果不存在则使用默认值
# os.environ["TRITON_COMPILE_ONLY"] = "1"
compile_only  = int(os.environ.get("TRITON_COMPILE_ONLY", "0"))
num_groups_compile = int(os.environ.get("NUM_GROUPS_COMPILE", "1"))
ep_size = get_env_int("EP_SIZE", 1)
training_mode = int(os.environ.get("TRAINING_MODE", "0"))
benchmark_mode = int(os.environ.get("BENCHMARK_MODE", "0"))
cur_group = int(os.environ.get("CUR_GROUP", "0"))
patched_environment, patch_source, patch_is_noop, patch_import_errors = resolve_patched_environment()
enforce_training_patch_contract(
    training_mode=training_mode,
    benchmark_mode=benchmark_mode,
    patch_is_noop=patch_is_noop,
    patch_source=patch_source,
    import_errors=patch_import_errors,
)
print(f"[bf16 patch] source={patch_source}, noop={patch_is_noop}")
total_cuda_devices = torch.cuda.device_count()
device_start_id, device_count, num_groups_run = resolve_device_settings(total_cuda_devices)

topk_ids_dir = os.environ.get("TOPK_IDS_DIR", None)
# topk_ids_dir = os.environ.get("TOPK_IDS_DIR", "/xin.zhen/data_ai/glm4.5v_topk_49716")

DEFAULT_FULL_BATCH_SIZES = [1,2,4,8,16,24,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
FULL_BATCH_SIZES = parse_batch_sizes_from_env(DEFAULT_FULL_BATCH_SIZES)

if compile_only == 1:
    assert num_groups_compile <= 56
    group_ids = build_compile_group_ids(
        num_groups_compile=num_groups_compile,
        device_start_id=device_start_id,
        device_count=device_count,
        compile_batches=[1, 64, 2048],
    )
    print(f"{group_ids=}")
else:
    group_ids = build_run_group_ids(
        device_start_id=device_start_id,
        device_count=device_count,
        num_groups_run=num_groups_run,
        full_batches=FULL_BATCH_SIZES,
    )
    print(
        f"[bf16 group config] start={device_start_id}, count={device_count}, run_groups={num_groups_run}"
    )
    print(f"{group_ids=}")

# 定义全局配置变量
training_config = {
    "training_mode"     : training_mode,        # 0: 运行模式, 1: 多卡训练模式
    "cur_group"         : cur_group,            # 0: 第一组, 1: 第二组, 2: 第三组
    "default_device"    : device_start_id,      # 默认设备ID, or None
    "group_id"          : group_ids,
}

NUM_SHARED_EXPERTS = [get_env_int("MOE_NUM_SHARED_EXPERTS", 0)]

## deepseek model
# NUM_EXPERTS = [256]
# HIDDEN_SIZES = [256]
# FFN_HIDDEN_SIZES = [7168]

## glm4.5v model
NUM_EXPERTS = [get_env_int("MOE_NUM_EXPERTS", 128)]
HIDDEN_SIZES = [get_env_int("MOE_HIDDEN_SIZE", 352)]
FFN_HIDDEN_SIZES = [get_env_int("MOE_FFN_HIDDEN_SIZE", 4096)]

EP_SIZE = [ep_size]
TOP_KS = [get_env_int("MOE_TOP_K", 8)]
DTYPES = [torch.bfloat16]
INPLACE = [True]
BATCH_SIZES = FULL_BATCH_SIZES.copy()
# BATCH_SIZES = [4]


ACTIVATION_NAME, IS_GATED = resolve_moe_activation_and_gate()
ACTIVATION_SITU_BETA, ACTIVATION_SITU_LINEAR_BETA = resolve_situ_parameters(
    ACTIVATION_NAME
)


print(f"[test_moe_bf16.py] Running with TRAINING_MODE={training_mode}, CUR_GROUP={cur_group}!!!")
print(f"[test_moe_bf16.py] activation={ACTIVATION_NAME}, is_gated={IS_GATED}")

# 根据训练模式选择不同的批次大小参数
if training_config["training_mode"] == 0:
    # 常规运行模式，使用所有批次大小
    pass
else:
    # 多卡训练模式，根据当前分组选择批次大小
    if training_config["cur_group"] in training_config["group_id"]:
        BATCH_SIZES = training_config["group_id"][training_config["cur_group"]][1]
        print(f"!!!BATCH_SIZES: {BATCH_SIZES}")
    else:
        raise ValueError(f"Invalid training group: {training_config['cur_group']}")

def input_helper(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    num_shared_experts: int = 0,
    seed: int = 0,
    ep_id: int = None,
):
    torch.manual_seed(seed)
    e_sum = e + num_shared_experts
    inter_dim = 2 * n if IS_GATED else n
    # topk 是原生 topk（不包含共享专家），计算最终 topk（包含共享专家）
    final_topk = topk + num_shared_experts
    a = torch.randn((m, k), device=device, dtype=dtype) / 10

    # ========================================================================
    # 情况1: ep_size = 1, num_shared_experts = 0
    # 单GPU，无共享专家，最简单的场景
    # ========================================================================
    if ep_size == 1 and num_shared_experts == 0:
        w1 = torch.randn((e, inter_dim, k), device=device, dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
        score_all = torch.randn((m, e), device=device, dtype=dtype)
        e_map = None

        # 使用 fused_topk 选择专家
        topk_weights, topk_ids = fused_topk(a, score_all, topk, renormalize=False)
        if topk_ids_dir and m <= 49716:
            loaded_topk_ids = torch.load(f"{topk_ids_dir}/topk_ids_layer{seed%45+3}_idx{seed%2}.pt", map_location="cpu")
            if loaded_topk_ids.device.type != device:
                loaded_topk_ids = loaded_topk_ids.to(device)
            tokens, topk_size = topk_ids.shape
            if loaded_topk_ids.shape[0] < tokens or loaded_topk_ids.shape[1] < topk_size:
                raise ValueError(
                    f"loaded topk_ids shape {loaded_topk_ids.shape} smaller than "
                    f"required {(tokens, topk_size)} from fused_topk"
                )
            topk_ids = loaded_topk_ids[:tokens, :topk_size].contiguous()

    # ========================================================================
    # 情况2: ep_size > 1, num_shared_experts = 0
    # 多GPU并行，无共享专家，需要 expert parallel 支持
    # ========================================================================
    elif ep_size > 1 and num_shared_experts == 0:
        if ep_id is None:
            ep_id = ep_size - 1  # 默认使用最后一个 GPU

        # 创建 expert_mask（不需要 fake expert，因为没有共享专家）
        expert_mask = torch.zeros((e,), dtype=dtypes.i32, device=device)
        # 设置当前 GPU 负责的非共享专家范围
        expert_mask[ep_id * (e // ep_size) : (ep_id + 1) * e // ep_size] = 1

        # 计算本地专家数量
        local_E = torch.sum(expert_mask).item()

        # 创建全局权重（包含所有非共享专家）
        w1_global = torch.randn((e, inter_dim, k), device=device, dtype=dtype) / 10
        w2_global = torch.randn((e, k, n), device=device, dtype=dtype) / 10

        # 选择当前 GPU 负责的非共享专家权重
        e_local_range = torch.arange(
            ep_id * (e // ep_size),
            (ep_id + 1) * e // ep_size,
            device=device,
            dtype=torch.int64
        )
        w1 = w1_global[e_local_range]
        w2 = w2_global[e_local_range]

        # 构建 e_map：将全局专家ID映射到本地专家ID
        indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        e_map = torch.where(
            expert_mask == 0,
            torch.full_like(expert_mask, -1, dtype=dtypes.i32),
            expert_mask,
        )
        e_map = torch.where(e_map == 1, indices, e_map)

        # score 包含所有非共享专家
        score_all = torch.randn((m, e), device=device, dtype=dtype)

        # 使用 fused_topk 选择专家
        topk_weights, topk_ids = fused_topk(a, score_all, topk, renormalize=False)

    # ========================================================================
    # 情况3:（ep_size = 1, num_shared_experts != 0）
    # 单GPU但有共享专家，使用 moe_fused_gate
    # ========================================================================
    elif ep_size == 1 and num_shared_experts != 0:
        w1 = torch.randn((e_sum, inter_dim, k), device=device, dtype=dtype) / 10
        w2 = torch.randn((e_sum, k, n), device=device, dtype=dtype) / 10
        score_all = torch.randn((m, e_sum), device=device, dtype=dtype)
        score = score_all[:, :e]
        e_map = None

        # 使用 moe_fused_gate 选择专家（包含共享专家处理）
        correction_bias = torch.randn((e,), device=device, dtype=dtype)
        w_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.fp32)
        id_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.i32)

        scale_factor = 1.2
        num_expert_group = 8
        topk_group = 4
        from boltops.tools.fused_moe_triton_tune import compat
        # final_topk 参数包含了共享专家的数量（final_topk = topk + num_shared_experts）
        _ = compat.moe_fused_gate(score, correction_bias, w_sglang, id_sglang, num_expert_group, topk_group,
                                 final_topk, num_shared_experts, scale_factor,)
        topk_weights = _[0]
        topk_ids = _[1]
        topk_ids, _sglang = torch.sort(topk_ids)
        topk_weights = topk_weights.gather(1, _sglang)
    # ========================================================================
    # 情况3: ep_size > 1, num_shared_experts != 0
    # 多GPU并行，有共享专家，最复杂的场景
    # ========================================================================
    elif ep_size > 1 and num_shared_experts != 0:
        if ep_id is None:
            ep_id = ep_size - 1  # 默认使用最后一个 GPU

        # 创建 expert_mask，参考 test_moe_w8a8_ep.py
        # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
        expert_mask = torch.zeros((e + num_shared_experts + 1,), dtype=dtypes.i32, device=device)
        # 设置当前 GPU 负责的非共享专家范围
        expert_mask[ep_id * (e // ep_size) : (ep_id + 1) * e // ep_size] = 1
        # 确保共享专家在所有 GPU 上都可用
        expert_mask[e:e_sum] = 1
        # Fake expert 用于掩码，设为 0（不处理）
        expert_mask[-1] = 0

        # 计算本地专家数量（当前 GPU 的非共享专家 + 所有共享专家）
        local_E = torch.sum(expert_mask).item()

        # 创建全局权重（包含所有专家：非共享 + 共享）
        w1_global = torch.randn((e_sum, inter_dim, k), device=device, dtype=dtype) / 10
        w2_global = torch.randn((e_sum, k, n), device=device, dtype=dtype) / 10

        # 选择当前 GPU 负责的非共享专家权重
        e_local_range = torch.arange(
            ep_id * (e // ep_size),
            (ep_id + 1) * e // ep_size,
            device=device,
            dtype=torch.int64
        )
        w1_local = w1_global[e_local_range]
        w2_local = w2_global[e_local_range]

        # 添加共享专家的权重（所有 GPU 都需要共享专家）
        w1_shared = w1_global[e:e_sum]
        w2_shared = w2_global[e:e_sum]

        # 合并本地专家和共享专家权重
        w1 = torch.cat([w1_local, w1_shared], dim=0)
        w2 = torch.cat([w2_local, w2_shared], dim=0)

        # 构建 e_map：将全局专家ID映射到本地专家ID
        # e_map 的大小是全局专家数（e + num_shared_experts + 1）
        indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        e_map = torch.where(
            expert_mask == 0,
            torch.full_like(expert_mask, -1, dtype=dtypes.i32),
            expert_mask,
        )
        e_map = torch.where(e_map == 1, indices, e_map)  # 有效专家映射到本地专家id，无效专家映射到-1

        # score 只包含非共享专家，用于 moe_fused_gate
        score = torch.randn((m, e), device=device, dtype=dtype)
        # score_all 包含所有专家（非共享 + 共享），用于后续计算
        score_all = torch.randn((m, e_sum), device=device, dtype=dtype)

        # 使用 moe_fused_gate 选择专家（包含共享专家处理）
        assert num_shared_experts == 1, "Currently only support num_shared_experts == 1"
        correction_bias = torch.randn((e,), device=device, dtype=dtype)
        w_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.fp32)
        id_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.i32)

        scale_factor = 1.2
        num_expert_group = 8
        topk_group = 4
        from boltops.tools.fused_moe_triton_tune import compat
        # moe_fused_gate 的 input 应该是非共享专家的门控输出
        # final_topk 参数包含了共享专家的数量（final_topk = topk + num_shared_experts）
        _ = compat.moe_fused_gate(score, correction_bias, w_sglang, id_sglang, num_expert_group, topk_group,
            final_topk, num_shared_experts, scale_factor,)
        topk_weights = _[0]
        topk_ids = _[1]
        # moe_fused_gate 返回的 topk_ids 最后 num_shared_experts 个位置是共享专家ID
        # 排序以便比较
        topk_ids, _sglang = torch.sort(topk_ids)
        topk_weights = topk_weights.gather(1, _sglang)
    else:
        assert False, "Invalid input"

    # 返回的 score 应该是包含所有专家的，用于后续计算
    return a, w1, w2, score_all, e_map, topk_weights, topk_ids

@perftest(num_warmup=1, num_iters=11,testGraph=True)
def fused_experts_impl_benchmark(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       output_dtype:torch.dtype, #compute or output type for i8& f8
                       inplace: bool = False,
                       activation: str = "silu",
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
                       activation_situ_beta: Optional[float] = None,
                       activation_situ_linear_beta: Optional[float] = None)-> torch.Tensor:

    return fused_moe_module.fused_experts_impl(hidden_states,
                                w1,
                                w2,
                                topk_weights,
                                topk_ids,
                                output_dtype=output_dtype,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                inplace=inplace,
                                activation=activation,
                                is_gated=IS_GATED,
                                activation_situ_beta=activation_situ_beta,
                                activation_situ_linear_beta=activation_situ_linear_beta)


@pytest.mark.parametrize("m", BATCH_SIZES)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
@pytest.mark.parametrize("k", FFN_HIDDEN_SIZES)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("num_shared_experts", NUM_SHARED_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", INPLACE)
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    num_shared_experts: int,
    dtype: torch.dtype,
    inplace: bool,
):
    # 设置CUDA设备
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # 根据配置设置设备
    if training_config["training_mode"] == 0:
        device_id = training_config["default_device"]
        if device_id is not None:
            torch.cuda.set_device(device_id)
            print(f"!!!Set device to {device_id}")
    else:
        device_id = training_config["group_id"][training_config["cur_group"]][0]
        if device_id is not None:
            torch.cuda.set_device(device_id)

    # 获取当前设备
    device = "cuda"

    ENABLE_TORCH_REF = False if m > 256 or training_config["training_mode"] == 1 else True

    # 计算最终 topk（包含共享专家）
    final_topk = topk + num_shared_experts

    # 计算专家数口径：global_e 用于全局路由，final_e 用于当前 rank 的本地执行专家数
    final_e = e // ep_size + num_shared_experts if ep_size > 1 else e + num_shared_experts
    global_e = e + num_shared_experts

    # 打印当前测试用例信息
    print(
        f"\nTest: m={m}, n={n}, k={k}, e={e}(global_e={global_e}, final_e={final_e}), "
        f"topk={topk} (final_topk={final_topk}), device={torch.cuda.current_device()}, "
        f"dtype={dtype}, ENABLE_TORCH_REF={ENABLE_TORCH_REF}!!!\n"
    )
    a, w1, w2, score, e_map, topk_weights, topk_ids = input_helper(
        m=m,
        n=n,
        k=k,
        e=e,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        num_shared_experts=num_shared_experts,
    )

    if ENABLE_TORCH_REF:
        # ENote: origin code:
        # def torch_moe(a, w1, w2, score, topk, expert_map):
        #     assert topk_ids_dir is None, "topk_ids_dir is not None"
        #     B, D = a.shape
        #     a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        #     out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        #     score = torch.softmax(score, dim=-1, dtype=torch.float32)
        #     topk_weight, topk_ids = torch.topk(score, topk)
        #     topk_weight = topk_weight.view(-1)
        #     topk_ids = topk_ids.view(-1)
        #     if expert_map is not None:
        #         topk_ids = expert_map[topk_ids]
        #     for i in range(w1.shape[0]):
        #         mask = topk_ids == i
        #         if mask.sum():
        #             out[mask] = _silu_and_mul_ref(
        #                 a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
        #     return (out.view(B, -1, w2.shape[1]) *
        #             topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)
        # torch_output = torch_moe(a, w1, w2, score, topk, e_map)

        def torch_moe(a, w1, w2, topk_weight, topk_ids, topk, expert_map):
            assert topk_ids_dir is None, "topk_ids_dir is not None"
            B, D = a.shape
            a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
            out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
            topk_ids = topk_ids.view(-1)
            if expert_map is not None:
                topk_ids = expert_map[topk_ids]
            for i in range(w1.shape[0]):
                mask = topk_ids == i
                if mask.sum():
                    out[mask] = apply_activation_ref(
                        a[mask] @ w1[i].transpose(0, 1),
                        activation_name=ACTIVATION_NAME,
                        is_gated=IS_GATED,
                        activation_situ_beta=ACTIVATION_SITU_BETA,
                        activation_situ_linear_beta=ACTIVATION_SITU_LINEAR_BETA,
                    ) @ w2[i].transpose(0, 1)
            return (out.view(B, -1, w2.shape[1]) *
                    topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)
        # 使用 final_topk（包含共享专家）
        torch_output = torch_moe(a, w1, w2, topk_weights, topk_ids, final_topk, e_map)


        # from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (fused_moe as iterative_moe)
        # iterative_output = iterative_moe(a,
        #                                 w1,
        #                                 w2,
        #                                 score,
        #                                 topk,
        #                                 global_num_experts=e,
        #                                 expert_map=e_map,
        #                                 renormalize=False)
        # torch.testing.assert_close(iterative_output, torch_output, atol=2e-2, rtol=0)

    with patched_environment():
        triton_output = fused_moe_module.fused_experts_impl(a,
                                w1,
                                w2,
                                topk_weights,
                                topk_ids,
                                output_dtype=a.dtype,
                                global_num_experts=e + num_shared_experts,
                                expert_map=e_map,
                                inplace=inplace,
                                activation=ACTIVATION_NAME,
                                is_gated=IS_GATED,
                                activation_situ_beta=ACTIVATION_SITU_BETA,
                                activation_situ_linear_beta=ACTIVATION_SITU_LINEAR_BETA)
    if ENABLE_TORCH_REF:
        # torch.set_printoptions(profile="full")
        # print(f"triton_output: {triton_output}, {triton_output.dtype}")
        # print(f"torch_output: {torch_output}, {torch_output.dtype}")
        torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)

    # triton_output, avg_triton = fused_experts_impl_benchmark(a,
    #                                 w1,
    #                                 w2,
    #                                 topk_weights,
    #                                 topk_ids,
    #                                 a.dtype,
    #                                 global_num_experts=e,
    #                                 expert_map=e_map,
    #                                 inplace=inplace)
    # msg = f"[TRITON_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, triton_avg: {avg_triton:>8.2f} us"
    # print(msg)


# 定义MoE测试用例列表
#  BATCH_SIZE: 批次大小 (m)
#  HIDDEN_SIZE: 隐藏层大小 (n)
#  FFN_HIDDEN_SIZE: FFN中间层大小 (k)
#  NUM_EXPERTS: 专家数量 (e)
#  TOP_K: 每个token选择的专家数量
moe_perf_model_cases_list = [
    (m, n, k, e, topk, num_shared_experts, dtype)
    for m in BATCH_SIZES
    for n in HIDDEN_SIZES
    for k in FFN_HIDDEN_SIZES
    for e in NUM_EXPERTS
    for topk in TOP_KS
    for num_shared_experts in NUM_SHARED_EXPERTS
    for dtype in DTYPES
]

# 配置MoE基准测试参数
moe_configs = [
    triton.testing.Benchmark(
        x_names=['BATCH_SIZE', 'HIDDEN_SIZE', 'FFN_HIDDEN_SIZE', 'NUM_EXPERTS', 'TOP_K', 'NUM_SHARED_EXPERTS', 'dtype'],
        x_vals=moe_perf_model_cases_list,
        line_arg='provider',  # 用于区分不同实现

        # line_vals=['triton', 'torch'],  # 测试的实现
        # line_names=['Triton', 'Torch'],  # 图例名称
        # styles=[('red', '-'), ('blue', '--')],
        line_vals=['triton'],  # 测试的实现
        line_names=['Triton'],  # 图例名称
        styles=[('red', '-')],

        ylabel='Time(ms)',
        xlabel='Batch Size',
        plot_name='MoE Performance（Times ms）',  # 图表标题
        args={
            'device': 'cuda',
            'inplace': True,
        }
    )
]


case_counter = 1

@triton.testing.perf_report(moe_configs)
def bench_fused_moe_bf16(BATCH_SIZE, HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_EXPERTS, TOP_K, NUM_SHARED_EXPERTS,
                   provider, dtype=torch.bfloat16, device="cuda", inplace=False):
    """
    BATCH_SIZE: 批次大小 (m)
    HIDDEN_SIZE: 隐藏层大小 (n)
    FFN_HIDDEN_SIZE: FFN中间层大小 (k)
    NUM_EXPERTS: 专家数量 (e)
    TOP_K: 每个token选择的专家数量
    """
    warmup = 25
    rep = 100

    # 根据配置设置设备
    assert training_config["training_mode"] == 0, "需要设置device_id"
    device_id = training_config["default_device"]
    if device_id is not None:
        torch.cuda.set_device(device_id)

    # 获取当前设备
    device = "cuda"

    # 打印当前测试用例
    global case_counter  # 添加全局声明
    print("\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, NUM_EXPERTS={}, TOP_K={}, NUM_SHARED_EXPERTS={}, dtype={}\n".format(
        case_counter,
        BATCH_SIZE,
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NUM_EXPERTS,
        TOP_K,
        NUM_SHARED_EXPERTS,
        dtype
    ))
    case_counter += 1  # 递增计数器

    ep_size = EP_SIZE[0]
    e = NUM_EXPERTS
    a, w1, w2, score, e_map, topk_weights, topk_ids = input_helper(
        m=BATCH_SIZE,
        n=HIDDEN_SIZE,
        k=FFN_HIDDEN_SIZE,
        e=e,
        topk=TOP_K,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        num_shared_experts=NUM_SHARED_EXPERTS,
    )


    ms = None
    if provider == "triton":
        with patched_environment():
            _, avg_us = fused_experts_impl_benchmark(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                a.dtype,
                global_num_experts=e + NUM_SHARED_EXPERTS,
                expert_map=e_map,
                inplace=inplace,
                activation=ACTIVATION_NAME,
                activation_situ_beta=ACTIVATION_SITU_BETA,
                activation_situ_linear_beta=ACTIVATION_SITU_LINEAR_BETA,
            )
            ms = avg_us / 1000.0
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return ms

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
    # test_fused_moe(32767, 352, 4096, 128, 8, 1, torch.bfloat16, False)
    pass

# 运行基准测试
if benchmark_mode == 1:
    bench_fused_moe_bf16.run(print_data=True)
