# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/vllm-project/vllm/blob/aeeb36b1f17145975c6713242f2447bb8b98782b/tests/kernels/moe/test_moe.py
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

import boltops.fused_moe.triton as fused_moe_module
from boltops.tools.fused_moe_triton_tune.routing import fused_topk
from boltops.fused_moe.triton.moe_compat import per_token_quant_triton, per_block_quant_wrapper
from boltops.utility.perf_utils import perftest
try:
    from boltops.tools.fused_moe_triton_tune.common import (
        get_env_int,
        parse_batch_sizes_from_env,
        resolve_device_settings,
        build_compile_group_ids,
        build_run_group_ids,
        ScalarType,
        scalar_types,
        quantize_weights,
        resolve_moe_activation_and_gate,
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
            ScalarType,
            scalar_types,
            quantize_weights,
            resolve_moe_activation_and_gate,
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
            ScalarType,
            scalar_types,
            quantize_weights,
            resolve_moe_activation_and_gate,
            resolve_patched_environment,
            enforce_training_patch_contract,
        )

# os.environ["VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

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
print(f"[int4int8 patch] source={patch_source}, noop={patch_is_noop}")
total_cuda_devices = torch.cuda.device_count()
device_start_id, device_count, num_groups_run = resolve_device_settings(total_cuda_devices)
DEFAULT_FULL_RUN_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
FULL_RUN_BATCH_SIZES = parse_batch_sizes_from_env(DEFAULT_FULL_RUN_BATCH_SIZES)

if compile_only == 1:
    assert num_groups_compile <= 56
    group_ids = build_compile_group_ids(
        num_groups_compile=num_groups_compile,
        device_start_id=device_start_id,
        device_count=device_count,
        compile_batches=[1, 64],
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
        f"[int4int8 group config] start={device_start_id}, count={device_count}, run_groups={num_groups_run}"
    )
    print(f"{group_ids=}")

# 定义全局配置变量
training_config = {
    "training_mode"     : training_mode,        # 0: 运行模式, 1: 多卡训练模式
    "cur_group"         : cur_group,            # 0: 第一组, 1: 第二组, 2: 第三组
    "default_device"    : device_start_id,      # 默认设备ID, or None
    "group_id"          : group_ids,
}

NUM_EXPERTS = [get_env_int("MOE_NUM_EXPERTS", 256)]
HIDDEN_SIZES = [get_env_int("MOE_HIDDEN_SIZE", 256)]
FFN_HIDDEN_SIZES = [get_env_int("MOE_FFN_HIDDEN_SIZE", 6144)]
EP_SIZE = [ep_size]
TOP_KS = [get_env_int("MOE_TOP_K", 8)]
DTYPES = [torch.float16]
GROUP_SIZES = [64]  # 固定常量：当前脚本按 group_size=64 调优/测试
HAS_ZPS = [True]
WEIGHT_BITS = [4]

print(f"[tune_moe_int4int8.py] Running with TRAINING_MODE={training_mode}, CUR_GROUP={cur_group}")


# 根据训练模式选择不同的批次大小参数
if training_config["training_mode"] == 0:
    # Preserve the original small-batch correctness sweep.
    # Larger batches may hit invalid default int4_w4a8 kernel configs when
    # no tuned config file is present for the current device.
    BATCH_SIZES = parse_batch_sizes_from_env([4, 8])
else:
    # 多卡训练模式，根据当前分组选择批次大小
    if training_config["cur_group"] in training_config["group_id"]:
        BATCH_SIZES = training_config["group_id"][training_config["cur_group"]][1]
        print(f"!!!BATCH_SIZES: {BATCH_SIZES}")
    else:
        raise ValueError(f"Invalid training group: {training_config['cur_group']}")


ACTIVATION_NAME, IS_GATED = resolve_moe_activation_and_gate()
print(f"[tune_moe_int4int8.py] activation={ACTIVATION_NAME}, is_gated={IS_GATED}")


case_counter_x = 1


def setup_device_from_training_config(skip_if_unavailable: bool = False) -> int:
    if not torch.cuda.is_available():
        if skip_if_unavailable:
            pytest.skip("CUDA is not available")
        raise RuntimeError("CUDA is not available")

    if training_config["training_mode"] == 0:
        device_id = training_config["default_device"]
    else:
        device_id = training_config["group_id"][training_config["cur_group"]][0]

    if device_id is not None:
        torch.cuda.set_device(device_id)
        print(f"!!!Set device to {device_id}")
    return device_id if device_id is not None else torch.cuda.current_device()


def get_quant_config(weight_bits: int, has_zp: bool) -> tuple[int, ScalarType]:
    if weight_bits == 4:
        return 2, scalar_types.uint4 if has_zp else scalar_types.uint4b8
    if weight_bits == 8:
        return 1, scalar_types.uint8 if has_zp else scalar_types.uint8b128
    raise ValueError(f"Unsupported weight_bits: {weight_bits}")


def allocate_quantized_weights(
    e: int,
    inter_dim: int,
    n: int,
    k: int,
    pack_factor: int,
    group_size: int,
    dtype: torch.dtype,
    device: str,
):
    w1_qweight = torch.empty((e, inter_dim, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_scales = torch.empty((e, inter_dim, k // group_size), device=device, dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device=device, dtype=dtype)
    w1_qzeros = torch.empty((e, inter_dim // pack_factor, k // group_size), device=device, dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k // pack_factor, n // group_size), device=device, dtype=torch.uint8)
    return w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros


def quantize_weight_tensor(
    weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
):
    _, qweight, scales, qzeros = quantize_weights(weight.T, quant_type, group_size, has_zp, False)
    qweight = qweight.T.contiguous().to(torch.uint8)
    scales = scales.T
    if has_zp:
        qzeros = qzeros.T.contiguous().to(torch.uint8)
    if weight_bits == 4:
        qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
        if has_zp:
            qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]
    return qweight, scales, qzeros if has_zp else None


def quantize_dense_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
    dtype: torch.dtype,
    device: str,
):
    e, _, k = w1.shape
    n = w2.shape[-1]
    pack_factor, _ = get_quant_config(weight_bits, has_zp)
    w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros = allocate_quantized_weights(
        e, w1.shape[1], n, k, pack_factor, group_size, dtype, device
    )

    for expert_id in range(e):
        qweight, scales, qzeros = quantize_weight_tensor(
            w1[expert_id], quant_type, group_size, has_zp, weight_bits
        )
        w1_qweight[expert_id] = qweight
        w1_scales[expert_id] = scales
        if has_zp:
            w1_qzeros[expert_id] = qzeros

        qweight, scales, qzeros = quantize_weight_tensor(
            w2[expert_id], quant_type, group_size, has_zp, weight_bits
        )
        w2_qweight[expert_id] = qweight
        w2_scales[expert_id] = scales
        if has_zp:
            w2_qzeros[expert_id] = qzeros

    return w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros


def quantize_streaming_weights(
    e: int,
    n: int,
    k: int,
    inter_dim: int,
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
    dtype: torch.dtype,
    device: str,
):
    pack_factor, _ = get_quant_config(weight_bits, has_zp)
    w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros = allocate_quantized_weights(
        e, inter_dim, n, k, pack_factor, group_size, dtype, device
    )

    for expert_id in range(e):
        w1_expert = torch.randn((inter_dim, k), device=device, dtype=dtype) / 10
        qweight, scales, qzeros = quantize_weight_tensor(
            w1_expert, quant_type, group_size, has_zp, weight_bits
        )
        w1_qweight[expert_id] = qweight
        w1_scales[expert_id] = scales
        if has_zp:
            w1_qzeros[expert_id] = qzeros
        del w1_expert, qweight, scales, qzeros

        w2_expert = torch.randn((k, n), device=device, dtype=dtype) / 10
        qweight, scales, qzeros = quantize_weight_tensor(
            w2_expert, quant_type, group_size, has_zp, weight_bits
        )
        w2_qweight[expert_id] = qweight
        w2_scales[expert_id] = scales
        if has_zp:
            w2_qzeros[expert_id] = qzeros
        del w2_expert, qweight, scales, qzeros

        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    return w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros


def build_expert_map(
    e: int,
    ep_size: int,
    device: str,
    *expert_tensors: torch.Tensor,
):
    if ep_size <= 1:
        return None, expert_tensors

    local_e = e // ep_size
    e_ids = torch.randint(0, e, (local_e,), device=device, dtype=torch.int32)
    e_map = torch.full((e,), -1, device=device, dtype=torch.int32)
    e_map[e_ids] = torch.arange(local_e, device=device, dtype=torch.int32)
    return e_map, tuple(tensor[e_ids] for tensor in expert_tensors)


def input_helper(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
    device: str = "cuda",
    seed: int = 0,
    streaming_weights: bool = False,
    renormalize: bool = False,
):
    assert ep_size >= 1
    assert e % ep_size == 0, f"num_experts({e}) must be divisible by ep_size({ep_size})"

    torch.manual_seed(seed)
    inter_dim = 2 * n if IS_GATED else n
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    score_all = torch.randn((m, e), device=device, dtype=dtype)

    _, quant_type = get_quant_config(weight_bits, has_zp)

    if streaming_weights:
        quantized_weights = quantize_streaming_weights(
            e, n, k, inter_dim, quant_type, group_size, has_zp, weight_bits, dtype, device
        )
    else:
        w1 = torch.randn((e, inter_dim, k), device=device, dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
        quantized_weights = quantize_dense_weights(
            w1, w2, quant_type, group_size, has_zp, weight_bits, dtype, device
        )

    e_map, quantized_weights = build_expert_map(e, ep_size, device, *quantized_weights)
    topk_weights, topk_ids = fused_topk(a, score_all, topk, renormalize=renormalize)
    a_q, a_scale = per_block_quant_wrapper((1, group_size))(per_token_quant_triton)(a)
    w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros = quantized_weights
    return (
        a_q,
        w1_qweight,
        w2_qweight,
        w1_scales,
        w2_scales,
        score_all,
        e_map,
        topk_weights,
        topk_ids,
        a_scale,
        w1_qzeros,
        w2_qzeros,
    )


@perftest(num_warmup=1, num_iters=11, testGraph=True)
def fused_experts_impl_benchmark(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    output_dtype,
    use_int4_w4a16=False,
    use_int8_w8a16=False,
    use_int4_w4a8=False,
    per_channel_quant=False,
    global_num_experts=-1,
    expert_map=None,
    w1_scale=None,
    w2_scale=None,
    w1_zp=None,
    w2_zp=None,
    a1_scale=None,
    block_shape=None,
    activation="silu",
):
    return fused_moe_module.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        output_dtype=output_dtype,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=IS_GATED,
    )

@pytest.mark.parametrize("m", BATCH_SIZES)
@pytest.mark.parametrize("n", HIDDEN_SIZES)
@pytest.mark.parametrize("k", FFN_HIDDEN_SIZES)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZPS)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
@torch.inference_mode()
def test_fused_moe_int4int8(m: int, n: int, k: int, e: int, topk: int,
                            ep_size: int, dtype: torch.dtype, group_size: int,
                            has_zp: bool, weight_bits: int):
    assert ep_size >= 1
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    setup_device_from_training_config(skip_if_unavailable=True)
    device = "cuda"

    global case_counter_x
    print(
        f"\nTest Count {case_counter_x}: m={m}, n={n}, k={k}, e={e}, topk={topk}, "
        f"ep_size={ep_size}, device={torch.cuda.current_device()}, dtype={dtype}, "
        f"group_size={group_size}, has_zp={has_zp}, weight_bits={weight_bits}!!!\n"
    )
    case_counter_x += 1

    M, N, K, E = m, n, k, e
    a_q, w1_qweight, w2_qweight, w1_scales, w2_scales, _score_all, e_map, topk_weights, topk_ids, a_scale, w1_qzeros, w2_qzeros = input_helper(
        m=M,
        n=N,
        k=K,
        e=E,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        group_size=group_size,
        has_zp=has_zp,
        weight_bits=weight_bits,
        device=device,
        seed=0,
        renormalize=False,
    )

    with patched_environment():
        fused_moe_module.fused_experts_impl(
            a_q,
            w1_qweight,
            w2_qweight,
            topk_weights,
            topk_ids,
            output_dtype=dtype,
            use_int4_w4a8=True,
            global_num_experts=E,
            expert_map=e_map,
            w1_scale=w1_scales,
            w2_scale=w2_scales,
            w1_zp=w1_qzeros if has_zp else None,
            w2_zp=w2_qzeros if has_zp else None,
            block_shape=[0, group_size],
            a1_scale=a_scale,
            activation=ACTIVATION_NAME,
            is_gated=IS_GATED,
        )

# 定义MoE测试用例列表
moe_perf_model_cases_list = [
    (m, n, k, e, topk, ep_size, dtype, group_size, has_zp, weight_bits)
    for m in BATCH_SIZES
    for n in HIDDEN_SIZES
    for k in FFN_HIDDEN_SIZES
    for e in NUM_EXPERTS
    for topk in TOP_KS
    for ep_size in EP_SIZE
    for dtype in DTYPES
    for group_size in GROUP_SIZES
    for has_zp in HAS_ZPS
    for weight_bits in WEIGHT_BITS
]

# 配置MoE基准测试参数
moe_configs = [
    triton.testing.Benchmark(
        x_names=['BATCH_SIZE', 'HIDDEN_SIZE', 'FFN_HIDDEN_SIZE', 'NUM_EXPERTS', 'TOP_K', 'EP_SIZE', 'dtype', 'group_size', 'has_zp', 'weight_bits'],
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
        plot_name='MoE INT4INT8 Performance (Times ms)',  # 图表标题
        args={
            'device': 'cuda',
        }
    )
]


case_counter = 1
perf_profling = False

@triton.testing.perf_report(moe_configs)
def bench_fused_moe_int4int8(BATCH_SIZE, HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_EXPERTS, TOP_K, EP_SIZE,
                             provider, dtype=torch.float16, device="cuda",
                             group_size=64, has_zp=True, weight_bits=4):
    """
    BATCH_SIZE: 批次大小 (m)
    HIDDEN_SIZE: 隐藏层大小 (n)
    FFN_HIDDEN_SIZE: FFN中间层大小 (k)
    NUM_EXPERTS: 专家数量 (e)
    TOP_K: 每个token选择的专家数量
    """
    warmup = 25
    rep = 100

    setup_device_from_training_config()
    device = "cuda"

    # 打印当前测试用例
    global case_counter  # 添加全局声明
    print("\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, NUM_EXPERTS={}, TOP_K={}\n".format(
        case_counter,
        BATCH_SIZE,
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NUM_EXPERTS,
        TOP_K
    ))
    case_counter += 1
    ep_size = EP_SIZE
    e = NUM_EXPERTS

    if perf_profling:
        assert provider == "triton"
        print("Before quantization:")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

    M, N, K, E = BATCH_SIZE, HIDDEN_SIZE, FFN_HIDDEN_SIZE, e
    a_q, w1_qweight, w2_qweight, w1_scales, w2_scales, _score_all, e_map, topk_weights, topk_ids, a_scale, w1_qzeros, w2_qzeros = input_helper(
        m=M,
        n=N,
        k=K,
        e=E,
        topk=TOP_K,
        ep_size=ep_size,
        dtype=dtype,
        group_size=group_size,
        has_zp=has_zp,
        weight_bits=weight_bits,
        device=device,
        seed=0,
        streaming_weights=perf_profling,
        renormalize=True,
    )

    if perf_profling:
        torch.cuda.empty_cache()
        gc.collect()
        print("After quantization:")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

    if provider == "triton":
        with patched_environment():
            if perf_profling:
                fn = lambda: fused_moe_module.fused_experts_impl(
                    a_q,
                    w1_qweight,
                    w2_qweight,
                    topk_weights,
                    topk_ids,
                    output_dtype=dtype,
                    use_int4_w4a8=True,
                    global_num_experts=E,
                    expert_map=e_map,
                    w1_scale=w1_scales,
                    w2_scale=w2_scales,
                    w1_zp=w1_qzeros if has_zp else None,
                    w2_zp=w2_qzeros if has_zp else None,
                    block_shape=[0, group_size],
                    a1_scale=a_scale,
                    activation=ACTIVATION_NAME,
                    is_gated=IS_GATED,
                )
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, perf_profiling=True)
            else:
                _, avg_us = fused_experts_impl_benchmark(
                    a_q,
                    w1_qweight,
                    w2_qweight,
                    topk_weights,
                    topk_ids,
                    dtype,
                    use_int4_w4a8=True,
                    global_num_experts=E,
                    expert_map=e_map,
                    w1_scale=w1_scales,
                    w2_scale=w2_scales,
                    w1_zp=w1_qzeros if has_zp else None,
                    w2_zp=w2_qzeros if has_zp else None,
                    block_shape=[0, group_size],
                    a1_scale=a_scale,
                    activation=ACTIVATION_NAME,
                )
                ms = avg_us / 1000.0
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return ms

# # # 运行基准测试
if benchmark_mode == 1:
    bench_fused_moe_int4int8.run(print_data=True)
