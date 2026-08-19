# SPDX-License-Identifier: Apache-2.0
"""Tests for the channel-wise W4A8 MoE path."""

import gc
import os
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
import triton

import boltops.fused_moe.triton as fused_moe_module
from boltops.fused_moe.triton.moe_compat import per_token_quant_hip
from boltops.utility.perf_utils import perftest
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

compile_only = int(os.environ.get("TRITON_COMPILE_ONLY", "0"))
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
print(f"[int4int8_channel patch] source={patch_source}, noop={patch_is_noop}")
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
        compile_batches=[1, 16],
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
        f"[int4int8_channel group config] start={device_start_id}, count={device_count}, run_groups={num_groups_run}"
    )
    print(f"{group_ids=}")

# 定义全局配置变量
training_config = {
    "training_mode": training_mode,
    "cur_group": cur_group,
    "default_device": device_start_id,
    "group_id": group_ids,
}

NUM_EXPERTS = [get_env_int("MOE_NUM_EXPERTS", 256)]
HIDDEN_SIZES = [get_env_int("MOE_HIDDEN_SIZE", 256)]
FFN_HIDDEN_SIZES = [get_env_int("MOE_FFN_HIDDEN_SIZE", 6144)]
EP_SIZE = [ep_size]
TOP_KS = [get_env_int("MOE_TOP_K", 8)]
DTYPES = [torch.float16]

print(f"[tune_moe_int4int8_channel.py] Running with TRAINING_MODE={training_mode}, CUR_GROUP={cur_group}")
ACTIVATION_NAME, IS_GATED = resolve_moe_activation_and_gate()
print(f"[tune_moe_int4int8_channel.py] activation={ACTIVATION_NAME}, is_gated={IS_GATED}")

if training_config["training_mode"] == 0:
    BATCH_SIZES = parse_batch_sizes_from_env([1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
else:
    if training_config["cur_group"] in training_config["group_id"]:
        BATCH_SIZES = training_config["group_id"][training_config["cur_group"]][1]
        print(f"!!!BATCH_SIZES: {BATCH_SIZES}")
    else:
        raise ValueError(f"Invalid training group: {training_config['cur_group']}")

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


def allocate_quantized_weights(
    e: int,
    inter_dim: int,
    n: int,
    k: int,
    device: str,
):
    w1_qweight = torch.empty((e, inter_dim, k // 2), device=device, dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
    w1_scales = torch.empty((e, inter_dim, 1), device=device, dtype=torch.float32)
    w2_scales = torch.empty((e, k, 1), device=device, dtype=torch.float32)
    return w1_qweight, w2_qweight, w1_scales, w2_scales


def generate_channelwise_w4a8_weight(
    n: int,
    k: int,
    device: str,
):
    weight_u4 = torch.randint(0, 15, (n, k), device=device, dtype=torch.int8)
    weight_ref = torch.where(weight_u4 > 7, weight_u4 - 16, weight_u4).contiguous()
    # Match HIPC contiguous K-pack (no shuffle / no marlin repack):
    # even-k nibble in high 4 bits, odd-k nibble in low 4 bits.
    # Same as pack_weight_k_contiguous / pack_weight_k_for_aiter_triton.
    qweight = (
        weight_u4[:, ::2].to(torch.uint8) * 16
        + weight_u4[:, 1::2].to(torch.uint8)
    ).contiguous()
    scales = torch.randn((n, 1), device=device, dtype=torch.float32)
    return weight_ref, qweight, scales


def quantize_dense_weights(
    e: int,
    n: int,
    k: int,
    inter_dim: int,
    device: str,
    return_reference: bool = True,
):
    w1_qweight, w2_qweight, w1_scales, w2_scales = allocate_quantized_weights(
        e, inter_dim, n, k, device
    )

    w1_ref = torch.empty((e, inter_dim, k), device=device, dtype=torch.int8) if return_reference else None
    w2_ref = torch.empty((e, k, n), device=device, dtype=torch.int8) if return_reference else None

    for expert_id in range(e):
        ref, qweight, scales = generate_channelwise_w4a8_weight(inter_dim, k, device)
        if return_reference:
            w1_ref[expert_id] = ref
        w1_qweight[expert_id] = qweight
        w1_scales[expert_id] = scales

        ref, qweight, scales = generate_channelwise_w4a8_weight(k, n, device)
        if return_reference:
            w2_ref[expert_id] = ref
        w2_qweight[expert_id] = qweight
        w2_scales[expert_id] = scales

    return w1_ref, w2_ref, w1_qweight, w2_qweight, w1_scales, w2_scales


def quantize_streaming_weights(
    e: int,
    n: int,
    k: int,
    inter_dim: int,
    device: str,
):
    w1_qweight, w2_qweight, w1_scales, w2_scales = allocate_quantized_weights(
        e, inter_dim, n, k, device
    )

    for expert_id in range(e):
        _ref, qweight, scales = generate_channelwise_w4a8_weight(inter_dim, k, device)
        w1_qweight[expert_id] = qweight
        w1_scales[expert_id] = scales
        del qweight, scales, _ref

        _ref, qweight, scales = generate_channelwise_w4a8_weight(k, n, device)
        w2_qweight[expert_id] = qweight
        w2_scales[expert_id] = scales
        del qweight, scales, _ref

        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    return None, None, w1_qweight, w2_qweight, w1_scales, w2_scales


def build_expert_map(
    e: int,
    ep_size: int,
    device: str,
    *expert_tensors,
):
    if ep_size <= 1:
        return None, expert_tensors

    local_e = e // ep_size
    local_ids = torch.arange(local_e, device=device, dtype=torch.int32)
    e_map = torch.full((e,), -1, device=device, dtype=torch.int32)
    e_map[local_ids] = torch.arange(local_e, device=device, dtype=torch.int32)

    local_tensors = []
    for tensor in expert_tensors:
        local_tensors.append(None if tensor is None else tensor[local_ids])
    return e_map, tuple(local_tensors)


def generate_unique_int_tensor(
    m: int,
    topk: int,
    high: int,
    device: str,
):
    if topk > high:
        raise ValueError(f"topk ({topk}) cannot exceed num_experts ({high})")

    topk_ids = torch.empty((m, topk), device=device, dtype=torch.int32)
    for row in range(m):
        topk_ids[row] = torch.randperm(high, device=device, dtype=torch.int32)[:topk]
    return topk_ids


def input_helper(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
    streaming_weights: bool = False,
    renormalize: bool = False,
    return_reference: bool = True,
):
    assert ep_size >= 1
    assert e % ep_size == 0, f"num_experts({e}) must be divisible by ep_size({ep_size})"

    torch.manual_seed(seed)
    a = torch.rand((m, k), device=device, dtype=dtype) / 10000
    topk_logits = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = F.softmax(topk_logits, dim=1)
    topk_ids = generate_unique_int_tensor(m, topk, e, device)
    score_all = None
    inter_dim = 2 * n if IS_GATED else n

    if streaming_weights:
        quantized_weights = quantize_streaming_weights(e, n, k, inter_dim, device)
    else:
        quantized_weights = quantize_dense_weights(
            e,
            n,
            k,
            inter_dim,
            device,
            return_reference=return_reference,
        )

    e_map, quantized_weights = build_expert_map(e, ep_size, device, *quantized_weights)
    w1_ref, w2_ref, w1_qweight, w2_qweight, w1_scales, w2_scales = quantized_weights
    return (
        a,
        w1_ref,
        w2_ref,
        w1_qweight,
        w2_qweight,
        w1_scales,
        w2_scales,
        score_all,
        e_map,
        topk_weights,
        topk_ids,
    )


def prepare_case(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str,
):
    a, w1_ref, w2_ref, w1_qweight, w2_qweight, w1_scale, w2_scale, _score_all, e_map, topk_weights, topk_ids = input_helper(
        m=m,
        n=n,
        k=k,
        e=e,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        seed=0,
        renormalize=False,
        return_reference=True,
    )
    return (
        a,
        w1_ref,
        w2_ref,
        w1_qweight,
        w2_qweight,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        e_map,
    )


def torch_w4a8_channelwise_reference_fallback(
    a: torch.Tensor,
    w1_ref: torch.Tensor,
    w2_ref: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
):
    num_tokens, hidden = a.shape
    topk = topk_weights.shape[1]
    a_q, a_scale = per_token_quant_hip(a)

    a_q = a_q.view(num_tokens, 1, hidden).repeat(1, topk, 1).reshape(-1, hidden)
    a_scale = a_scale.view(num_tokens, 1, 1).repeat(1, topk, 1).reshape(-1, 1)

    routed_ids = topk_ids.reshape(-1)
    routed_weights = topk_weights.reshape(-1)
    if expert_map is not None:
        routed_ids = expert_map[routed_ids]

    out = torch.zeros((num_tokens * topk, w2_ref.shape[1]), device=a.device, dtype=torch.float32)

    for expert_id in range(w1_ref.shape[0]):
        mask = routed_ids == expert_id
        if not mask.any():
            continue

        inter = torch.matmul(
            a_q[mask].to(torch.float32),
            w1_ref[expert_id].to(torch.float32).T,
        ) * a_scale[mask].to(torch.float32) * w1_scale[expert_id].view(1, -1)
        act = apply_activation_ref(
            inter,
            activation_name=ACTIVATION_NAME,
            is_gated=IS_GATED,
        ).to(a.dtype)

        act_q, act_scale = per_token_quant_hip(act)
        out[mask] = torch.matmul(
            act_q.to(torch.float32),
            w2_ref[expert_id].to(torch.float32).T,
        ) * act_scale.to(torch.float32) * w2_scale[expert_id].view(1, -1)

    out = out.view(num_tokens, topk, -1)
    out = out * routed_weights.view(num_tokens, topk, 1).to(out.dtype)
    return out.sum(dim=1).to(a.dtype)


def torch_w4a8_channelwise_reference(
    a: torch.Tensor,
    w1_ref: torch.Tensor,
    w2_ref: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
):
    """Reference for Triton W4A8 channelwise MoE.

    Triton now consumes HIPC contiguous K-pack (even-k high nibble, odd-k low)
    without moe_c/marlin shuffle. The oracle takes unpacked signed int4 refs
    (same as generate_channelwise_w4a8_weight). Prefer CI golden when available.
    """
    if expert_map is None:
        try:
            from aiter.moe_c_golden import run_w4a8_perchannel_golden

            return run_w4a8_perchannel_golden(
                a,
                w1_ref,
                w2_ref,
                topk_weights,
                topk_ids,
                a.dtype,
                w1_scale,
                w2_scale,
            )
        except BaseException:
            pass

    return torch_w4a8_channelwise_reference_fallback(
        a,
        w1_ref,
        w2_ref,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        expert_map,
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
@torch.inference_mode()
def test_int4_w4a8_channelwise_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
):
    setup_device_from_training_config(skip_if_unavailable=True)
    device = "cuda"

    ENABLE_TORCH_REF = training_config["training_mode"] == 0 and m <= 256

    global case_counter_x
    print(
        f"\nTest Count {case_counter_x}: m={m}, n={n}, k={k}, e={e}, topk={topk}, "
        f"ep_size={ep_size}, device={torch.cuda.current_device()}, dtype={dtype}, "
        f"ENABLE_TORCH_REF={ENABLE_TORCH_REF}!!!\n"
    )
    case_counter_x += 1

    a, w1_ref, w2_ref, w1_qweight, w2_qweight, w1_scale, w2_scale, _score_all, e_map, topk_weights, topk_ids = input_helper(
        m=m,
        n=n,
        k=k,
        e=e,
        topk=topk,
        ep_size=ep_size,
        dtype=dtype,
        device=device,
        seed=0,
        renormalize=False,
        return_reference=ENABLE_TORCH_REF,
    )

    with patched_environment():
        out = fused_moe_module.fused_experts_impl(
            a,
            w1_qweight,
            w2_qweight,
            topk_weights,
            topk_ids,
            output_dtype=a.dtype,
            use_int4_w4a8=True,
            per_channel_quant=True,
            global_num_experts=e,
            expert_map=e_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            activation=ACTIVATION_NAME,
            is_gated=IS_GATED,
        )

    if ENABLE_TORCH_REF:
        ref = torch_w4a8_channelwise_reference(
            a,
            w1_ref,
            w2_ref,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            e_map,
        )
        rel_diff = (
            torch.mean(torch.abs(out.to(torch.float32) - ref.to(torch.float32)))
            / torch.clamp(torch.mean(torch.abs(ref.to(torch.float32))), min=1e-6)
        )
        assert rel_diff < 0.05, (
            f"{m=}, {n=}, {k=}, {e=}, {topk=}, {ep_size=}, {dtype=}, {rel_diff=}"
        )


# 定义MoE测试用例列表
moe_perf_model_cases_list = [
    (m, n, k, e, topk, ep_size)
    for m in BATCH_SIZES
    for n in HIDDEN_SIZES
    for k in FFN_HIDDEN_SIZES
    for e in NUM_EXPERTS
    for topk in TOP_KS
    for ep_size in EP_SIZE
]

# 配置MoE基准测试参数
moe_configs = [
    triton.testing.Benchmark(
        x_names=["BATCH_SIZE", "HIDDEN_SIZE", "FFN_HIDDEN_SIZE", "NUM_EXPERTS", "TOP_K", "EP_SIZE"],
        x_vals=moe_perf_model_cases_list,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("red", "-")],
        ylabel="Time(ms)",
        xlabel="Batch Size",
        plot_name="MoE INT4INT8 Channelwise Performance (Times ms)",
        args={
            "dtype": torch.float16,
            "device": "cuda",
        },
    )
]


case_counter = 1
perf_profling = False


@triton.testing.perf_report(moe_configs)
def bench_fused_moe_int4int8_channelwise(
    BATCH_SIZE,
    HIDDEN_SIZE,
    FFN_HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    EP_SIZE,
    provider,
    dtype=torch.float16,
    device="cuda",
):
    warmup = 25
    rep = 100

    setup_device_from_training_config()
    device = "cuda"

    global case_counter
    print(
        "\nTest Count {}: BATCH_SIZE={}, HIDDEN_SIZE={}, FFN_HIDDEN_SIZE={}, "
        "NUM_EXPERTS={}, TOP_K={}, EP_SIZE={}\n".format(
            case_counter,
            BATCH_SIZE,
            HIDDEN_SIZE,
            FFN_HIDDEN_SIZE,
            NUM_EXPERTS,
            TOP_K,
            EP_SIZE,
        )
    )
    case_counter += 1

    if perf_profling:
        assert provider == "triton"
        print("Before quantization:")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

    a, _w1_ref, _w2_ref, w1_qweight, w2_qweight, w1_scale, w2_scale, _score_all, e_map, topk_weights, topk_ids = input_helper(
        m=BATCH_SIZE,
        n=HIDDEN_SIZE,
        k=FFN_HIDDEN_SIZE,
        e=NUM_EXPERTS,
        topk=TOP_K,
        ep_size=EP_SIZE,
        dtype=dtype,
        device=device,
        seed=0,
        streaming_weights=perf_profling,
        renormalize=True,
        return_reference=False,
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
                    a,
                    w1_qweight,
                    w2_qweight,
                    topk_weights,
                    topk_ids,
                    output_dtype=dtype,
                    use_int4_w4a8=True,
                    per_channel_quant=True,
                    global_num_experts=NUM_EXPERTS,
                    expert_map=e_map,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    activation=ACTIVATION_NAME,
                    is_gated=IS_GATED,
                )
                ms = triton.testing.do_bench(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    perf_profiling=True,
                )
            else:
                _, avg_us = fused_experts_impl_benchmark(
                    a,
                    w1_qweight,
                    w2_qweight,
                    topk_weights,
                    topk_ids,
                    dtype,
                    use_int4_w4a8=True,
                    per_channel_quant=True,
                    global_num_experts=NUM_EXPERTS,
                    expert_map=e_map,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    activation=ACTIVATION_NAME,
                )
                ms = avg_us / 1000.0
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return ms


if benchmark_mode == 1:
    bench_fused_moe_int4int8_channelwise.run(print_data=True)
