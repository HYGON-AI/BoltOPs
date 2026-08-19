import triton
import triton.language as tl
import time
import gc
import os
import sys

import pytest
import torch
import boltops.fused_moe.triton as fused_moe_module
from boltops.tools.fused_moe_triton_tune.routing import fused_topk
from boltops.fused_moe.triton.moe_compat import per_token_quant_hip, per_block_quant_wrapper
from boltops.utility.perf_utils import checkAllclose, perftest
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
            ScalarType,
            scalar_types,
            quantize_weights,
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
            ScalarType,
            scalar_types,
            quantize_weights,
            resolve_moe_activation_and_gate,
            apply_activation_ref,
            resolve_patched_environment,
            enforce_training_patch_contract,
        )

import os

# os.environ["VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

compile_only  = int(os.environ.get("TRITON_COMPILE_ONLY", "0"))
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
print(f"[int4 patch] source={patch_source}, noop={patch_is_noop}")
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
        f"[int4 group config] start={device_start_id}, count={device_count}, run_groups={num_groups_run}"
    )
    print(f"{group_ids=}")

# 定义全局配置变量
training_config = {
    "training_mode"     : training_mode,        # 0: 运行模式, 1: 多卡训练模式
    "cur_group"         : cur_group,            # 0: 第一组, 1: 第二组, 2: 第三组
    "default_device"    : device_start_id,      # 默认设备ID, or None
    "group_id"          : group_ids,
}

NUM_EXPERTS = [get_env_int("MOE_NUM_EXPERTS", 384)]
HIDDEN_SIZES = [get_env_int("MOE_HIDDEN_SIZE", 2048)]  # E=xxx, N=HIDDEN_SIZE
FFN_HIDDEN_SIZES = [get_env_int("MOE_FFN_HIDDEN_SIZE", 7168)]
EP_SIZE = [ep_size]
TOP_KS = [get_env_int("MOE_TOP_K", 8)]
DTYPES = [torch.float16]
GROUP_SIZES = [64]
HAS_ZPS = [True]
WEIGHT_BITS = [4]

print(f"[test_moe_int4.py] Running with TRAINING_MODE={training_mode}, CUR_GROUP={cur_group}")


# 根据训练模式选择不同的批次大小参数
if training_config["training_mode"] == 0:
    # 常规运行模式，使用所有批次大小
    BATCH_SIZES = FULL_RUN_BATCH_SIZES.copy()
    # BATCH_SIZES = [4096]
else:
    # 多卡训练模式，根据当前分组选择批次大小
    if training_config["cur_group"] in training_config["group_id"]:
        BATCH_SIZES = training_config["group_id"][training_config["cur_group"]][1]
        print(f"!!!BATCH_SIZES: {BATCH_SIZES}")
    else:
        raise ValueError(f"Invalid training group: {training_config['cur_group']}")


ACTIVATION_NAME, IS_GATED = resolve_moe_activation_and_gate()
print(f"[test_moe_int4.py] activation={ACTIVATION_NAME}, is_gated={IS_GATED}")


def setup_device_from_training_config() -> int:
    if training_config["training_mode"] == 0:
        device_id = training_config["default_device"]
    else:
        device_id = training_config["group_id"][training_config["cur_group"]][0]
    if device_id is not None:
        torch.cuda.set_device(device_id)
        print(f"!!!Set device to {device_id}")
    return device_id if device_id is not None else torch.cuda.current_device()


def torch_int4_moe_reference(a, w1, w2, topk_weight, topk_ids, topk, expert_map):
    batch, dim = a.shape
    a = a.view(batch, -1, dim).repeat(1, topk, 1).reshape(-1, dim)
    out = torch.zeros(batch * topk, w2.shape[1], dtype=a.dtype, device=a.device)
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
            ) @ w2[i].transpose(0, 1)
    return (out.view(batch, -1, w2.shape[1]) * topk_weight.view(batch, -1, 1).to(out.dtype)).sum(dim=1)


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
):
    torch.manual_seed(seed)
    inter_dim = 2 * n if IS_GATED else n
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    score = torch.randn((m, e), device=device, dtype=dtype)
    e_ep = e // ep_size

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128
    else:
        raise ValueError(f"Unsupported {weight_bits=}")

    w1_qweight = torch.empty((e_ep, inter_dim, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qweight = torch.empty((e_ep, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_scales = torch.empty((e_ep, inter_dim, k // group_size), device=device, dtype=dtype)
    w2_scales = torch.empty((e_ep, k, n // group_size), device=device, dtype=dtype)
    w1_qzeros = (
        torch.empty((e_ep, inter_dim // pack_factor, k // group_size), device=device, dtype=torch.uint8)
        if has_zp
        else None
    )
    w2_qzeros = (
        torch.empty((e_ep, k // pack_factor, n // group_size), device=device, dtype=torch.uint8)
        if has_zp
        else None
    )

    # golden 参考路径使用量化后反量化的权重，保证与 triton 输入一致
    w1_ref = torch.empty((e_ep, inter_dim, k), device=device, dtype=dtype)
    w2_ref = torch.empty((e_ep, k, n), device=device, dtype=dtype)

    for expert_id in range(e_ep):
        w1_expert = torch.randn((inter_dim, k), device=device, dtype=dtype) / 10
        w1_weight, w1_qw, w1_s, w1_qz = quantize_weights(w1_expert.T, quant_type, group_size, has_zp, False)
        del w1_expert

        w1_weight = w1_weight.T
        w1_qw = w1_qw.T.contiguous().to(torch.uint8)
        w1_s = w1_s.T
        if has_zp:
            w1_qz = w1_qz.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            w1_qw = w1_qw[:, 1::2] * 16 + w1_qw[:, ::2]
            if has_zp:
                w1_qz = w1_qz[1::2, :] * 16 + w1_qz[::2, :]

        w1_qweight[expert_id] = w1_qw
        w1_scales[expert_id] = w1_s
        w1_ref[expert_id] = w1_weight
        if has_zp:
            w1_qzeros[expert_id] = w1_qz

        del w1_weight, w1_qw, w1_s, w1_qz
        torch.cuda.empty_cache()

        w2_expert = torch.randn((k, n), device=device, dtype=dtype) / 10
        w2_weight, w2_qw, w2_s, w2_qz = quantize_weights(w2_expert.T, quant_type, group_size, has_zp, False)
        del w2_expert

        w2_weight = w2_weight.T
        w2_qw = w2_qw.T.contiguous().to(torch.uint8)
        w2_s = w2_s.T
        if has_zp:
            w2_qz = w2_qz.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            w2_qw = w2_qw[:, 1::2] * 16 + w2_qw[:, ::2]
            if has_zp:
                w2_qz = w2_qz[1::2, :] * 16 + w2_qz[::2, :]

        w2_qweight[expert_id] = w2_qw
        w2_scales[expert_id] = w2_s
        w2_ref[expert_id] = w2_weight
        if has_zp:
            w2_qzeros[expert_id] = w2_qz

        del w2_weight, w2_qw, w2_s, w2_qz
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with patched_environment():
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=True)
        if ep_size > 1:
            local_e = e // ep_size
            begin_e_id = (topk_ids[0, 2] // local_e * local_e).item()
            e_ids = torch.arange(begin_e_id, begin_e_id + local_e, device="cuda", dtype=torch.int32)
            e_map = torch.full((e,), -1, device=device, dtype=torch.int32)
            e_map[e_ids] = torch.arange(local_e, device=device, dtype=torch.int32)
        else:
            e_map = None

    return a, topk_weights, topk_ids, e_map, w1_qweight, w2_qweight, w1_scales, w2_scales, w1_qzeros, w2_qzeros, w1_ref, w2_ref


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


case_counter_x = 1

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
def test_fused_moe_wn16(m: int, n: int, k: int, e: int, topk: int,
                        ep_size: int, dtype: torch.dtype, group_size: int,
                        has_zp: bool, weight_bits: int):
    assert ep_size >= 1
    print(m, n, k, e, ep_size, topk, dtype, group_size, has_zp, weight_bits)

    # 设置CUDA设备
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    setup_device_from_training_config()

    # 获取当前设备
    device = "cuda"
    enable_torch_ref = False if m > 256 or training_config["training_mode"] == 1 else True
    # 打印当前测试用例
    global case_counter_x  # 添加全局声明
    print("\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, NUM_EXPERTS={}, TOP_K={}, EP_SIZE={}\n".format(
        case_counter_x,
        m,
        n,
        k,
        e,
        topk,
        ep_size
    ))
    case_counter_x += 1  # 递增计数器

    (
        a,
        topk_weights,
        topk_ids,
        e_map,
        w1_qweight,
        w2_qweight,
        w1_scales,
        w2_scales,
        w1_qzeros,
        w2_qzeros,
        w1_ref,
        w2_ref,
    ) = input_helper(
        m=m,
        n=n,
        k=k,
        e=e,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        group_size=group_size,
        has_zp=has_zp,
        weight_bits=weight_bits,
        device=device,
        seed=0,
    )

    with patched_environment():
        triton_output = fused_moe_module.fused_experts_impl(
            a,
            w1_qweight,
            w2_qweight,
            topk_weights,
            topk_ids,
            output_dtype=a.dtype,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            global_num_experts=e,
            expert_map=e_map,
            w1_scale=w1_scales,
            w2_scale=w2_scales,
            w1_zp=w1_qzeros if has_zp else None,
            w2_zp=w2_qzeros if has_zp else None,
            block_shape=[0, group_size],
            activation=ACTIVATION_NAME,
            is_gated=IS_GATED,
        )

    if enable_torch_ref:
        torch_output = torch_int4_moe_reference(
            a, w1_ref, w2_ref, topk_weights, topk_ids, topk, e_map
        )
        msg = (
            f"[INT4_golden_check] m={m}, n={n}, k={k}, e={e}, topk={topk}, ep_size={ep_size}, "
            f"dtype={dtype}, group_size={group_size}, has_zp={has_zp}, weight_bits={weight_bits}"
        )
        checkAllclose(torch_output, triton_output, rtol=0.02, atol=0.02, msg=msg)


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
        plot_name='MoE Performance（Times ms）',  # 图表标题
        args={
            'device': 'cuda',
        }
    )
]


case_counter = 1
perf_profling = False

@triton.testing.perf_report(moe_configs)
def bench_fused_moe_wn16(BATCH_SIZE, HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_EXPERTS, TOP_K, EP_SIZE,
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

    # 获取当前设备
    device = "cuda"

    # 打印当前测试用例
    global case_counter  # 添加全局声明
    print("\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, NUM_EXPERTS={}, TOP_K={}, EP_SIZE={}\n".format(
        case_counter,
        BATCH_SIZE,
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NUM_EXPERTS,
        TOP_K,
        EP_SIZE
    ))
    case_counter += 1  # 递增计数器

    # 准备输入数据
    e = NUM_EXPERTS
    (
        a,
        topk_weights,
        topk_ids,
        e_map,
        w1_qweight,
        w2_qweight,
        w1_scales,
        w2_scales,
        w1_qzeros,
        w2_qzeros,
        _w1_ref,
        _w2_ref,
    ) = input_helper(
        m=BATCH_SIZE,
        n=HIDDEN_SIZE,
        k=FFN_HIDDEN_SIZE,
        e=e,
        topk=TOP_K,
        ep_size=EP_SIZE,
        dtype=dtype,
        group_size=group_size,
        has_zp=has_zp,
        weight_bits=weight_bits,
        device=device,
        seed=0,
    )

    if provider == "triton":
        with patched_environment():
            if perf_profling:
                fn = lambda: fused_moe_module.fused_experts_impl(a,
                                      w1_qweight,
                                      w2_qweight,
                                      topk_weights,
                                      topk_ids,
                                      output_dtype=a.dtype,
                                      use_int4_w4a16=weight_bits == 4,
                                      use_int8_w8a16=weight_bits == 8,
                                      global_num_experts=e,
                                      expert_map=e_map,
                                      w1_scale=w1_scales,
                                      w2_scale=w2_scales,
                                      w1_zp=w1_qzeros if has_zp else None,
                                      w2_zp=w2_qzeros if has_zp else None,
                                      block_shape=[0, group_size],
                                      activation=ACTIVATION_NAME,
                                      is_gated=IS_GATED)
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, perf_profiling=True)
            else:
                _, avg_us = fused_experts_impl_benchmark(
                    a,
                    w1_qweight,
                    w2_qweight,
                    topk_weights,
                    topk_ids,
                    a.dtype,
                    use_int4_w4a16=weight_bits == 4,
                    use_int8_w8a16=weight_bits == 8,
                    global_num_experts=e,
                    expert_map=e_map,
                    w1_scale=w1_scales,
                    w2_scale=w2_scales,
                    w1_zp=w1_qzeros if has_zp else None,
                    w2_zp=w2_qzeros if has_zp else None,
                    block_shape=[0, group_size],
                    activation=ACTIVATION_NAME,
                )
                ms = avg_us / 1000.0

    return ms

# # # 运行基准测试
if benchmark_mode == 1:
    bench_fused_moe_wn16.run(print_data=True)
