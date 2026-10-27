# Modified by Hygon in 2026 for BoltOPs HCU integration.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools
import os
import sys

import pytest
import torch
import triton
import triton.language as tl

import torch.nn.functional as F
from boltops.tools.fused_moe_triton_tune.routing import fused_topk
from boltops.fused_moe.triton.moe_compat import per_token_quant_triton, pertoken_quant
import boltops.fused_moe.triton as fused_moe_module

from boltops.utility import dtypes
from boltops.utility.perf_utils import checkAllclose, perftest
from typing import List, Dict, Optional
try:
    from boltops.tools.fused_moe_triton_tune.common import (
        get_env_int,
        parse_batch_sizes_from_env,
        resolve_device_settings,
        build_compile_group_ids,
        build_run_group_ids,
        resolve_moe_activation_and_gate,
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
            apply_activation_ref,
            resolve_patched_environment,
            enforce_training_patch_contract,
        )

os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

# 从环境变量获取配置
compile_only = int(os.environ.get("TRITON_COMPILE_ONLY", "0"))
num_groups_compile = int(os.environ.get("NUM_GROUPS_COMPILE", "1"))
ep_size = int(os.environ.get("EP_SIZE", "1"))
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
print(f"[fp8_channel patch] source={patch_source}, noop={patch_is_noop}")
total_cuda_devices = torch.cuda.device_count()
device_start_id, device_count, num_groups_run = resolve_device_settings(total_cuda_devices)
DEFAULT_FULL_RUN_BATCH_SIZES = [1,2,4,8,16,24,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
FULL_RUN_BATCH_SIZES = parse_batch_sizes_from_env(DEFAULT_FULL_RUN_BATCH_SIZES)

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
        full_batches=FULL_RUN_BATCH_SIZES,
    )
    print(
        f"[fp8_channel group config] start={device_start_id}, count={device_count}, run_groups={num_groups_run}"
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

#deepseek model tp8
NUM_EXPERTS = [get_env_int("MOE_NUM_EXPERTS", 256)]
HIDDEN_SIZES = [get_env_int("MOE_HIDDEN_SIZE", 2048 if ep_size > 1 else 128)]
FFN_HIDDEN_SIZES = [get_env_int("MOE_FFN_HIDDEN_SIZE", 7168)]
EP_SIZE = [ep_size]
TOP_KS = [get_env_int("MOE_TOP_K", 8)]
DTYPES = [torch.float16]
INPLACE = [True]

print(f"[tune_moe_fp8_channel.py] Running with TRAINING_MODE={training_mode}, CUR_GROUP={cur_group}, EP_SIZE={ep_size}")

# 根据训练模式选择不同的批次大小参数
if training_config["training_mode"] == 0:
    # 常规运行模式，使用所有批次大小
    BATCH_SIZES = FULL_RUN_BATCH_SIZES.copy()
else:
    # 多卡训练模式，根据当前分组选择批次大小
    if training_config["cur_group"] in training_config["group_id"]:
        BATCH_SIZES = training_config["group_id"][training_config["cur_group"]][1]
        print(f"!!!BATCH_SIZES: {BATCH_SIZES}")
    else:
        raise ValueError(f"Invalid training group: {training_config['cur_group']}")


ACTIVATION_NAME, IS_GATED = resolve_moe_activation_and_gate()
print(f"[tune_moe_fp8_channel.py] activation={ACTIVATION_NAME}, is_gated={IS_GATED}")


def setup_device_from_training_config() -> int:
    if training_config["training_mode"] == 0:
        device_id = training_config["default_device"]
    else:
        device_id = training_config["group_id"][training_config["cur_group"]][0]
    if device_id is not None:
        torch.cuda.set_device(device_id)
        print(f"!!!Set device to {device_id}")
    return device_id if device_id is not None else torch.cuda.current_device()


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Per-token input quantization + per-channel weight quantization matmul."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    M = A.numel() // A.shape[-1]
    B = B.t()
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    C = torch.matmul(A, B)
    C = As * C * Bs.view(1, -1)
    return C.reshape(origin_C_shape).to(output_dtype)


def torch_w8a8_per_channel_fp8_moe_expert_mask(
    a,
    w1,
    w2,
    w1_s,
    w2_s,
    topk_weight,
    topk_ids,
    expert_map=None,
):
    """FP8 per-channel MoE torch reference with scale-aware dequant path."""
    B, D = a.shape
    topk = topk_weight.shape[1]

    # Match triton path: quantize activations per token before each GEMM.
    a_q, a_s = per_token_quant_triton(a, quant_dtype=dtypes.fp8)
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).to(torch.float32)
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=a.dtype
            )
            act_out = apply_activation_ref(
                inter_out,
                activation_name=ACTIVATION_NAME,
                is_gated=IS_GATED,
            )
            act_q, act_s = per_token_quant_triton(act_out, quant_dtype=dtypes.fp8)
            out[mask] = native_w8a8_per_token_matmul(
                act_q, w2[i], act_s, w2_s[i], output_dtype=a.dtype
            )
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


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
    ep_id: Optional[int] = None,
):
    """Helper function to generate input data for FP8 per-channel quantization MoE tests."""
    assert e % ep_size == 0, f"num_experts({e}) must be divisible by ep_size({ep_size})"

    torch.manual_seed(seed)
    e_sum = e + num_shared_experts
    inter_dim = 2 * n if IS_GATED else n
    final_topk = topk + num_shared_experts

    a = torch.randn((m, k), dtype=dtype, device=device) / 10
    e_per_rank = e // ep_size

    # 生成全局权重（bf16 格式，后续量化为 fp8）
    w1_global = torch.randn((e_sum, inter_dim, k), dtype=dtype, device=device) / 10
    w2_global = torch.randn((e_sum, k, n), dtype=dtype, device=device) / 10

    # 情况1: ep_size == 1, 无共享专家
    if ep_size == 1 and num_shared_experts == 0:
        score_all = torch.randn((m, e), dtype=dtype, device=device)
        topk_weights, topk_ids = fused_topk(a, score_all, topk, renormalize=False)
        e_map = None
        w1, w2 = w1_global[:e], w2_global[:e]

    # 情况2: ep_size > 1, 无共享专家
    elif ep_size > 1 and num_shared_experts == 0:
        if ep_id is None:
            ep_id = ep_size - 1
        score_all = torch.randn((m, e), dtype=dtype, device=device)
        topk_weights, topk_ids = fused_topk(a, score_all, topk, renormalize=False)

        expert_mask = torch.zeros((e,), dtype=dtypes.i32, device=device)
        start = ep_id * e_per_rank
        end = (ep_id + 1) * e_per_rank
        expert_mask[start:end] = 1
        indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        e_map = torch.where(
            expert_mask == 0,
            torch.full_like(expert_mask, -1, dtype=dtypes.i32),
            indices,
        )
        w1 = w1_global[start:end]
        w2 = w2_global[start:end]

    # 情况3: ep_size == 1, 有共享专家
    elif ep_size == 1 and num_shared_experts != 0:
        from boltops.tools.fused_moe_triton_tune import compat

        score_all = torch.randn((m, e_sum), dtype=dtype, device=device)
        score = score_all[:, :e]
        correction_bias = torch.randn((e,), device=device, dtype=dtype)
        w_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.fp32)
        id_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.i32)
        scale_factor = 1.2
        num_expert_group = 8
        topk_group = 4
        gate_out = compat.moe_fused_gate(
            score,
            correction_bias,
            w_sglang,
            id_sglang,
            num_expert_group,
            topk_group,
            final_topk,
            num_shared_experts,
            scale_factor,
        )
        topk_weights = gate_out[0]
        topk_ids = gate_out[1]
        topk_ids, sorted_idx = torch.sort(topk_ids)
        topk_weights = topk_weights.gather(1, sorted_idx)

        e_map = None
        w1, w2 = w1_global, w2_global

    # 情况4: ep_size > 1, 有共享专家
    elif ep_size > 1 and num_shared_experts != 0:
        from boltops.tools.fused_moe_triton_tune import compat

        if ep_id is None:
            ep_id = ep_size - 1

        # shared experts 路径沿用 bf16 逻辑：构造 fake expert 作为 mask 占位
        expert_mask = torch.zeros((e + num_shared_experts + 1,), dtype=dtypes.i32, device=device)
        start = ep_id * e_per_rank
        end = (ep_id + 1) * e_per_rank
        expert_mask[start:end] = 1
        expert_mask[e:e_sum] = 1
        expert_mask[-1] = 0

        indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        e_map = torch.where(
            expert_mask == 0,
            torch.full_like(expert_mask, -1, dtype=dtypes.i32),
            indices,
        )

        # 本地非共享专家 + 全局共享专家
        w1_local = w1_global[start:end]
        w2_local = w2_global[start:end]
        w1_shared = w1_global[e:e_sum]
        w2_shared = w2_global[e:e_sum]
        w1 = torch.cat([w1_local, w1_shared], dim=0)
        w2 = torch.cat([w2_local, w2_shared], dim=0)

        score = torch.randn((m, e), dtype=dtype, device=device)
        score_all = torch.randn((m, e_sum), dtype=dtype, device=device)
        correction_bias = torch.randn((e,), device=device, dtype=dtype)
        w_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.fp32)
        id_sglang = torch.empty_strided((m, final_topk), (final_topk + 10, 1), device=device, dtype=dtypes.i32)
        scale_factor = 1.2
        num_expert_group = 8
        topk_group = 4
        gate_out = compat.moe_fused_gate(
            score,
            correction_bias,
            w_sglang,
            id_sglang,
            num_expert_group,
            topk_group,
            final_topk,
            num_shared_experts,
            scale_factor,
        )
        topk_weights = gate_out[0]
        topk_ids = gate_out[1]
        topk_ids, sorted_idx = torch.sort(topk_ids)
        topk_weights = topk_weights.gather(1, sorted_idx)

    else:
        raise AssertionError("Invalid input combination")

    # Per-channel quantization for weights
    w1_q, w1_s = pertoken_quant(w1, quant_dtype=dtypes.fp8)
    w2_q, w2_s = pertoken_quant(w2, quant_dtype=dtypes.fp8)

    return a, w1_q, w2_q, w1_s, w2_s, score_all, e_map, topk_weights, topk_ids


case_counter_x = 1


@perftest(num_warmup=1, num_iters=11, testGraph=True)
def fused_experts_impl_benchmark(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output_dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    activation: str = "silu",
):
    return fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        output_dtype=output_dtype,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        activation=activation,
        is_gated=IS_GATED,
    )

@pytest.mark.parametrize("m", BATCH_SIZES)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
@pytest.mark.parametrize("k", FFN_HIDDEN_SIZES)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("num_shared_experts", NUM_SHARED_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_w8a8_channel_fp8_fused_moe(m, n, k, e, topk, ep_size, num_shared_experts, dtype):
    # 设置CUDA设备
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    setup_device_from_training_config()

    # 获取当前设备
    device = "cuda"

    # torch results for ep_size != 1 may be incorrect.
    ENABLE_TORCH_REF = False if m > 256 or (training_config["training_mode"] == 1 or ep_size != 1) else True

    final_e = e // ep_size + num_shared_experts if ep_size > 1 else e + num_shared_experts
    global_e = e + num_shared_experts

    # 打印当前测试用例
    global case_counter_x
    print(
        f"\nTest Count {case_counter_x}: m={m}, n={n}, k={k}, e={e}"
        f"(global_e={global_e}, final_e={final_e}), topk={topk}, "
        f"num_shared_experts={num_shared_experts}, ep_size={ep_size}, "
        f"device={torch.cuda.current_device()}, dtype={dtype}, ENABLE_TORCH_REF={ENABLE_TORCH_REF}!!!\n"
    )
    case_counter_x += 1

    M, N, K, E = m, n, k, e
    a, w1_q, w2_q, w1_s, w2_s, _score_all, e_map, topk_weights, topk_ids = input_helper(
        m=M,
        n=N,
        k=K,
        e=E,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        num_shared_experts=num_shared_experts,
        seed=0,
    )

    with patched_environment():
        out = fused_moe_module.fused_experts_impl(
            a,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            output_dtype=a.dtype,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=E + num_shared_experts,
            expert_map=e_map,
            w1_scale=w1_s,
            w2_scale=w2_s,
            activation=ACTIVATION_NAME,
            is_gated=IS_GATED,
        )

    if ENABLE_TORCH_REF:
        ref_out = torch_w8a8_per_channel_fp8_moe_expert_mask(
                        a, w1_q, w2_q, w1_s, w2_s, topk_weights, topk_ids, expert_map=e_map)
        msg = (
            f"[TRITON_check] {M=}, {K=}, {N=}, {E=}, {topk=}, "
            f"{num_shared_experts=}, {ep_size=}, dtype={dtype}"
        )
        checkAllclose(ref_out, out, rtol=0.01, atol=100, msg=msg)


# 定义MoE测试用例列表
moe_perf_model_cases_list = [
    (m, n, k, e, topk, num_shared_experts)
    for m in BATCH_SIZES
    for n in HIDDEN_SIZES
    for k in FFN_HIDDEN_SIZES
    for e in NUM_EXPERTS
    for topk in TOP_KS
    for num_shared_experts in NUM_SHARED_EXPERTS
]

# 配置MoE基准测试参数
moe_configs = [
    triton.testing.Benchmark(
        x_names=['BATCH_SIZE', 'HIDDEN_SIZE', 'FFN_HIDDEN_SIZE', 'NUM_EXPERTS', 'TOP_K', 'NUM_SHARED_EXPERTS'],
        x_vals=moe_perf_model_cases_list,
        line_arg='provider',  # 用于区分不同实现

        line_vals=['triton'],  # 测试的实现
        line_names=['Triton'],  # 图例名称
        styles=[('red', '-')],

        ylabel='Time(ms)',
        xlabel='Batch Size',
        plot_name='MoE FP8 Channel Performance (Times ms)',  # 图表标题
        args={
            'dtype': torch.float16,
            'device': 'cuda',
        }
    )
]


case_counter = 1
perf_profling = False

@triton.testing.perf_report(moe_configs)
def bench_fused_moe_fp8(BATCH_SIZE, HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_EXPERTS, TOP_K, NUM_SHARED_EXPERTS,
                   provider, dtype=torch.float16, device="cuda"):
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
    setup_device_from_training_config()

    # 获取当前设备
    device = "cuda"

    # 打印当前测试用例
    global case_counter
    print("\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, NUM_EXPERTS={}, TOP_K={}, EP_SIZE={}\n".format(
        case_counter,
        BATCH_SIZE,
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NUM_EXPERTS,
        TOP_K,
        ep_size
    ))
    case_counter += 1

    # 准备输入数据
    e = NUM_EXPERTS
    m = BATCH_SIZE
    n = HIDDEN_SIZE
    k = FFN_HIDDEN_SIZE

    M, N, K, E = m, n, k, e
    a, w1_q, w2_q, w1_s, w2_s, _score_all, e_map, topk_weights, topk_ids = input_helper(
        m=M,
        n=N,
        k=K,
        e=E,
        topk=TOP_K,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        num_shared_experts=NUM_SHARED_EXPERTS,
        seed=0,
    )

    ms = None
    if provider == "triton":
        with patched_environment():
            _, avg_us = fused_experts_impl_benchmark(
                a,
                w1_q,
                w2_q,
                topk_weights,
                topk_ids,
                a.dtype,
                use_fp8_w8a8=True,
                per_channel_quant=True,
                global_num_experts=E + NUM_SHARED_EXPERTS,
                expert_map=e_map,
                w1_scale=w1_s,
                w2_scale=w2_s,
                activation=ACTIVATION_NAME,
            )
            ms = avg_us / 1000.0
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return ms

# 运行基准测试
if benchmark_mode == 1:
    bench_fused_moe_fp8.run(print_data=True)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
