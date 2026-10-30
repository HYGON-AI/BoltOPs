# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Single-K specialization: legality, rounding tolerance and unchanged fallbacks."""
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from boltops.fused_moe.triton import moe_op
from boltops.fused_moe.triton.moe_op import fused_moe
from boltops.utility import arch_info


GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
MXFP_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None
    or arch_info.get_arch() != "gfx946",
    reason="native MXFP tests require gfx946",
)


def _args(**changes):
    return dict({
        "K": 128, "BLOCK_SIZE_K": 128, "COMBINE_SCALE_LOAD": False,
        "MUL_ROUTED_WEIGHT": False, "ADD_BIAS": True,
        "use_int8_w8a8": False, "use_fp8_w8a8": False,
        "use_int8_w8a16": False,
        "a_ptr": SimpleNamespace(dtype=torch.bfloat16),
        "b_ptr": SimpleNamespace(dtype=torch.bfloat16),
    }, **changes)


@pytest.mark.parametrize("k,expected", [(0, False), (1, True), (127, True),
                                         (128, True), (129, False), (7168, False)])
@pytest.mark.parametrize("mode", ["int8", "fp8", "bf16"])
def test_single_k_extent(k, expected, mode):
    args = _args(K=k, use_int8_w8a8=mode == "int8", use_fp8_w8a8=mode == "fp8")
    assert moe_op.fused_moe_kernel.values["single_k_tile"](args) == expected


@pytest.mark.parametrize("changes,expected", [
    ({"COMBINE_SCALE_LOAD": True}, False),
    ({"BLOCK_SIZE_K": 64}, False),
    ({"MUL_ROUTED_WEIGHT": True}, True),
    ({"ADD_BIAS": False}, True),
    ({"use_int8_w8a16": True}, True),
    ({"a_ptr": SimpleNamespace(dtype=torch.float16)}, True),
    ({"b_ptr": SimpleNamespace(dtype=torch.float16)}, True),
    ({"a_ptr": SimpleNamespace(dtype=torch.float32)}, True),
])
def test_single_k_mode_guards(changes, expected):
    assert moe_op.fused_moe_kernel.values["single_k_tile"](_args(**changes)) == expected


def _check_close(actual, expected, *, special=False):
    # NaN payloads are not part of this numerical contract; classification is.
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    rtol, atol = ((.02, .002) if actual.dtype == torch.bfloat16 else
                  (.003, .002) if actual.dtype == torch.float16 else (.0001, .00005))
    torch.testing.assert_close(actual[finite], expected[finite], rtol=rtol, atol=atol)
    if special:
        zeros = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


def _run_pair(monkeypatch, mode, routed=False, output="fp16", k=63, n=129,
              bk=128, bn=128, sorted_io=False, bias=True, special=False,
              group_n=128, group_k=128, waves=4, stages=1, fusion=True,
              scale_bias=True, fp8_dtype=torch.float8_e4m3fn):
    if mode.startswith("fp8") and not arch_info.is_fp8_avail():
        pytest.skip("device does not support FP8 dot")
    if mode == "fp32":
        # The independent reference uses full FP32 inputs, not default TF32.
        monkeypatch.setattr(triton.knobs.language, "fp32_default", "ieee")
    torch.manual_seed(177)
    m, experts, topk, bm = 19, 3, 2, 32
    routes, tk = m * topk, 1 if routed else topk
    quant = mode.startswith(("int8_", "fp8_"))
    w8a16 = mode == "w8a16"
    block, tensor, fp8 = mode.endswith("block"), mode.endswith("tensor"), mode.startswith("fp8")
    dtype, compute = {"fp16": (torch.float16, tl.float16),
                      "bf16": (torch.bfloat16, tl.bfloat16),
                      "fp32": (torch.float32, tl.float32)}[output]
    in_dtype = (fp8_dtype if fp8 else torch.int8 if quant else dtype if w8a16 else
                {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[mode])
    rows = routes if routed else m
    if quant and not fp8:
        a = torch.randint(-127, 128, (rows, k), dtype=in_dtype)
        b = torch.randint(-127, 128, (experts, n, k), dtype=in_dtype)
    else:
        a = (torch.randn((rows, k)) * 2).to(in_dtype)
        b = (torch.randn((experts, n, k)) * 2).to(in_dtype)
        if w8a16:
            b = torch.randint(-127, 128, (experts, n, k), dtype=torch.int8)
    sa = sb = None
    if quant:
        sa = torch.rand((rows, triton.cdiv(k, group_k) if block else 1)) * .01
        sb = torch.rand((experts, triton.cdiv(n, group_n) if block else n,
                         triton.cdiv(k, group_k) if block else 1)) * .01
        if tensor:
            sa, sb = sa[0, 0].clone(), sb[:, 0, 0].clone()
    if w8a16:
        sb = torch.rand((experts, n)) * .01
    ids = torch.randint(experts, (m, topk), dtype=torch.int32)
    weights = torch.rand((m, topk))
    bias_value = torch.randn((experts, n)) * .05 if bias else None
    if special:
        values = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"),
                               2.**-126, 2.**-133, 2.**-149, 65504., 65520.])
        if quant and not tensor:
            sa[:len(values), 0] = values
        elif not quant:
            a[:len(values), 0] = values.to(in_dtype)
        if bias:
            bias_value[:, :len(values)] = values
        weights.flatten()[:4] = torch.tensor([0., -0., 1., -1.])
    # CPU golden: materialize FP32 products/scales, then the public
    # epilogue order. Do not reuse the specialized Triton implementation.
    route_ids = ids.flatten().long()
    sources = torch.arange(routes) // tk
    effective_bk = min(bk, group_n, group_k) if block else bk
    ref = torch.zeros((routes, n))
    if block:
        ni = torch.arange(n) // group_n
        for start in range(0, k, effective_bk):
            dot = (a.float()[sources, start:start + effective_bk][:, None, :]
                   * b.float()[route_ids, :, start:start + effective_bk]).sum(-1)
            aa = sa[sources, start // group_k, None]
            bb = sb[route_ids[:, None], ni[None, :], start // group_k]
            ref += dot * aa * bb if bn > group_n else dot * (aa * bb)
    else:
        ref = (a.float()[sources, None, :] * b.float()[route_ids]).sum(-1)
    if quant and not block:
        ref = ref * (sa if tensor else sa[sources]) * (sb[route_ids, None] if tensor else sb[route_ids, :, 0])
    if w8a16:
        ref *= sb[route_ids]
    # vLLM epilogue: dequantize -> bias -> router weight -> output cast.
    # An explicit bias multiplier is needed only when routing is folded
    # elsewhere; otherwise the final row multiplier also weights the bias.
    if bias:
        bias_ref = bias_value[route_ids]
        if scale_bias and not routed:
            bias_ref = bias_ref * weights.flatten()[:, None]
        ref = ref + bias_ref.float()
    if routed:
        ref *= weights.flatten()[:, None]
    ref = ref.to(dtype)
    sid_list, eid_list = [], []
    flat = ids.cpu().flatten()
    for expert in range(experts):
        selected = (flat == expert).nonzero().flatten().tolist()
        padded = triton.cdiv(len(selected), bm) * bm
        sid_list.extend(selected + [routes] * (padded - len(selected)))
        eid_list.extend([expert] * (padded // bm))
    sid = torch.tensor(sid_list, dtype=torch.int32)
    eid = torch.tensor(eid_list, dtype=torch.int32)
    count = torch.tensor([sid.numel()], dtype=torch.int32)
    valid = sid < routes
    if sorted_io:
        af = torch.zeros((sid.numel(), k))
        af[valid] = a.float()[sid[valid].long() // tk]
        a = af.to(in_dtype)
        if quant and not tensor:
            sf = torch.zeros((sid.numel(), sa.shape[1]))
            sf[valid] = sa[sid[valid].long() // tk]
            sa = sf
    device_a, device_b = a.cuda(), b.cuda()
    device_sa = sa.cuda() if sa is not None else None
    device_sb = sb.cuda() if sb is not None else None
    device_weights, device_ids = weights.cuda(), ids.cuda()
    device_sid, device_eid, device_count = sid.cuda(), eid.cuda(), count.cuda()
    device_bias = bias_value.cuda() if bias_value is not None else None
    cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
               num_warps=waves, num_stages=stages, SPLIT_K=0, COMBINE_SCALE_LOAD=False,
               N_TILES_PER_CTA=1, enable_fp_fusion=fusion)
    predicate = moe_op.fused_moe_kernel.values["single_k_tile"]
    compiled, outputs, flags = [], [], []
    original_run = triton.runtime.JITFunction.run

    def record(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        if self.__name__ == "fused_moe_kernel":
            compiled.append(result)
            flags.append(kwargs["single_k_tile"])
        return result

    monkeypatch.setattr(triton.runtime.JITFunction, "run", record)
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", False)
    for enabled in (False, True):
        monkeypatch.setitem(moe_op.fused_moe_kernel.values, "single_k_tile", predicate if enabled else lambda _: False)
        storage = torch.full(((sid.numel() if sorted_io else routes) + 1, n + 1),
                             -123, device="cuda", dtype=dtype)
        c = storage[:-1, :-1]
        saved = dict(cfg)
        moe_op.fused_moe(device_a, device_b, c, device_sa, device_sb, None,
                         device_weights, device_ids, device_sid, None, device_eid, device_count,
                         routed, tk, compute, use_int8_w8a8=quant and not fp8,
                         use_fp8_w8a8=fp8, use_int8_w8a16=w8a16,
                         per_channel_quant=quant and not block and not tensor,
                         block_shape=[group_n, group_k] if block else None,
                         a_sorted=sorted_io, c_sorted=sorted_io, B_bias=device_bias,
                         scale_bias_with_routed_weight=scale_bias, config=cfg)
        torch.cuda.synchronize()
        assert cfg == saved
        storage_cpu = storage.cpu()
        assert (storage_cpu[-1] == -123).all() and (storage_cpu[:, -1] == -123).all()
        c_cpu = storage_cpu[:-1, :-1]
        if sorted_io:
            assert (c_cpu[~valid] == -123).all()
            out = torch.empty((routes, n), dtype=dtype)
            out[sid[valid].long()] = c_cpu[valid]
        else:
            out = c_cpu.clone()
        outputs.append(out)
    assert flags == [False, 0 < k <= effective_bk]
    if flags[1]:
        assert "scf.for" not in compiled[1].asm["ttir"]
    else:
        raw = torch.int32 if dtype == torch.float32 else torch.int16
        assert torch.equal(outputs[0].view(raw), outputs[1].view(raw))
    _check_close(outputs[1], outputs[0], special=special)
    for out in outputs:
        _check_close(out, ref)


MODES = ["int8_block", "int8_channel", "int8_tensor", "fp8_block",
         "fp8_channel", "fp8_tensor", "bf16"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("output", ["fp16", "bf16", "fp32"])
def test_single_k_bias_and_routing(monkeypatch, mode, routed, output):
    _run_pair(monkeypatch, mode, routed=routed, output=output,
              sorted_io=True, scale_bias=routed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8_block", "int8_channel", "fp8_block", "bf16"])
@pytest.mark.parametrize("case", [
    dict(k=1, n=65, bk=64), dict(k=64, bk=64, waves=8),
    dict(k=64, n=257, bn=256, bk=64, stages=2),
    dict(k=129), dict(k=63, group_n=64, group_k=128),
    dict(fusion=False, output="fp32", bias=True),
])
def test_single_k_boundaries_and_fallback(monkeypatch, mode, case):
    _run_pair(monkeypatch, mode, **case)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8_block", "int8_channel", "fp8_block", "bf16"])
@pytest.mark.parametrize("routed", [False, True])
def test_single_k_special_values(monkeypatch, mode, routed):
    _run_pair(monkeypatch, mode, special=True, bias=False, routed=routed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_single_k_fp8_e5m2(monkeypatch):
    _run_pair(monkeypatch, "fp8_block", fp8_dtype=torch.float8_e5m2,
              output="fp32", scale_bias=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8_block", "fp8_block"])
def test_single_k_uses_clamped_block_k(monkeypatch, mode):
    # Host launch clamps requested BK=256 to the quantization group's 128.
    _run_pair(monkeypatch, mode, bk=256, k=129)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8_block", "int8_channel", "fp8_block", "bf16"])
@pytest.mark.parametrize("routed", [False, True])
def test_single_k_bias_scaling_is_independent_of_routing(monkeypatch, mode, routed):
    _run_pair(monkeypatch, mode, routed=routed, scale_bias=not routed, output="fp32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8_block", "bf16"])
@pytest.mark.parametrize("routed", [False, True])
def test_single_k_special_bias(monkeypatch, mode, routed):
    _run_pair(monkeypatch, mode, special=True, bias=True, output="fp32",
              routed=routed, scale_bias=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode,output", [
    ("fp16", "fp16"), ("fp16", "bf16"), ("fp16", "fp32"),
    ("fp32", "fp32"), ("w8a16", "fp16"), ("w8a16", "bf16"),
])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("case", [
    dict(k=63, sorted_io=True), dict(k=64, bk=64, waves=8),
    dict(k=129), dict(k=63, special=True),
])
def test_single_k_unquantized_and_weight_only(monkeypatch, mode, output, routed, case):
    _run_pair(monkeypatch, mode, output=output, routed=routed, **case)


def _cpu_moe_routing(ids, experts, block_m):
    routes = ids.flatten().tolist()
    sorted_ids, expert_ids = [], []
    for expert in range(experts):
        selected = [i for i, e in enumerate(routes) if e == expert]
        padded = ((len(selected) + block_m - 1) // block_m) * block_m
        sorted_ids.extend(selected + [len(routes)] * (padded - len(selected)))
        expert_ids.extend([expert] * (padded // block_m))
    return (torch.tensor(sorted_ids, dtype=torch.int32),
            torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int32))


@GPU_REQUIRED
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('expanded_rows', [False, True])
def test_splitk_fp32_partials_and_output_boundary(monkeypatch, dtype, expanded_rows):
    torch.manual_seed(731)
    m, topk, experts, k, n = 3, 2, 3, 256, 70
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    router = torch.tensor([[.2, .8], [.7, .3], [.4, .6]])
    rows = m * topk if expanded_rows else m
    a = (torch.randn(rows, k) * .1).to(dtype)
    b = (torch.randn(experts, n, k) * .1).to(dtype)
    expected = []
    for route, expert in enumerate(ids.flatten().tolist()):
        row = route if expanded_rows else route // topk
        acc = a[row].float() @ b[expert].float().T
        expected.append(acc.to(dtype))
    expected = torch.stack(expected)
    c_shape = (m, topk, n) if expanded_rows else (m*topk, n)
    result = torch.empty(c_shape, dtype=dtype, device='cuda')
    sorted_ids, expert_ids, padded = _cpu_moe_routing(ids, experts, 16)
    # Assert the actual dispatch: weighted GEMM2 disables Split-K, so it
    # must not accidentally masquerade as Split-K coverage here.
    op = moe_op
    kernel = op.fused_moe_splitk_kernel
    dispatched = []

    class SplitKSpy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                partials = args[2]
                assert partials.dtype == torch.float32
                assert partials.shape == (2, m * topk, n)
                dispatched.append(True)
                return kernel[grid](*args, **kwargs)
            return launch

    monkeypatch.setattr(op, 'fused_moe_splitk_kernel', SplitKSpy())
    fused_moe(
        a.cuda(), b.cuda(), result, None, None, None,
        router.cuda(), ids.cuda(), sorted_ids.cuda(), None, expert_ids.cuda(), padded.cuda(),
        False, 1 if expanded_rows else topk,
        compute_type=tl.bfloat16 if dtype == torch.bfloat16 else tl.float16,
        config=dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                    num_warps=4, num_stages=1, SPLIT_K=2),
    )
    assert dispatched == [True]
    torch.testing.assert_close(result.cpu().reshape_as(expected), expected, rtol=.02, atol=.002)


def _fp8_groups(x, group):
    grouped = x.float().reshape(*x.shape[:-1], -1, group)
    scale = grouped.abs().amax(-1).clamp_min(1.e-10) / 448
    q = (grouped / scale[..., None]).to(torch.float8_e4m3fn).reshape(x.shape)
    return q, scale


def _mxfp4_values(shape, group):
    # CPU expansion is exact: no weights/scales are recomputed on device.
    packed = torch.randint(0, 256, (*shape[:-1], shape[-1]//2), dtype=torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    values = lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.)
    exponents = torch.randint(121, 126, (*shape[:-1], shape[-1]//group))
    scales = torch.exp2(exponents.float() - 127)
    return values.to(torch.float8_e4m3fn), scales


@GPU_REQUIRED
@pytest.mark.parametrize('bottom', [False, True], ids=['gemm1', 'gemm2'])
@pytest.mark.parametrize('kind', ['bf16', 'fp8_w8a8', 'fp8_block128', 'mxfp4_w4a8', 'int4_w4a16'])
def test_moe_gemm_cpu_prepared(kind, bottom):
    torch.manual_seed(21)
    m, topk, experts = 3, 2, 3
    hidden, intermediate = 128, 128
    width = intermediate if bottom else hidden
    out_width = hidden if bottom else 2 * intermediate
    a_rows = m * topk if bottom else m
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)  # expert 1 empty
    router = torch.tensor([[.25, .75], [.4, .6], [.8, .2]])
    a = (torch.randn(a_rows, width) * .1).bfloat16()
    a_scale = b_scale = None
    flags = {}
    if kind == 'bf16':
        b = (torch.randn(experts, out_width, width) * .1).bfloat16()
        a_ref, b_ref = a.float(), b.float()
    elif kind == 'int4_w4a16':
        codes = torch.randint(0, 16, (experts, out_width, width), dtype=torch.uint8)
        b = codes[..., 0::2] | (codes[..., 1::2] << 4)
        b_scale = torch.full((experts, out_width, width//32), .02, dtype=torch.bfloat16)
        # GPTQ/AWQ symmetric INT4 uses unsigned nibble minus 8.
        b_ref = ((codes.float() - 8) * b_scale.float().repeat_interleave(32, -1)).bfloat16().float()
        a_ref = a.float()
        flags = dict(use_int4_w4a16=True, block_shape=[0, 32])
    else:
        a, a_scale128 = _fp8_groups(a, 128)
        a_ref = a.float() * a_scale128.repeat_interleave(128, -1)
        if kind == 'fp8_block128':
            raw = torch.randn(experts, out_width, width) * .1
            blocks = raw.reshape(experts, out_width//128, 128, width//128, 128)
            b_scale = blocks.abs().amax((2, 4)).clamp_min(1.e-10) / 448
            expanded_scale = b_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
            b = (raw / expanded_scale).to(torch.float8_e4m3fn)
            a_scale = a_scale128
            b_ref = b.float() * expanded_scale
            flags = dict(use_fp8_w8a8=True, block_shape=[128, 128])
        elif kind == 'mxfp4_w4a8':
            b, b_scale = _mxfp4_values((experts, out_width, width), 32)
            a_scale = a_scale128.repeat_interleave(4, -1)
            b_ref = b.float() * b_scale.repeat_interleave(32, -1)
            flags = dict(use_fp8_w8a8=True, block_shape=[1, 32])
        else:
            b, b_scale = _fp8_groups(torch.randn(experts, out_width, width) * .1, 32)
            a_scale = a_scale128.repeat_interleave(4, -1)
            b_ref = b.float() * b_scale.repeat_interleave(32, -1)
            flags = dict(use_fp8_w8a8=True, block_shape=[1, 32])
    expected = torch.empty(m * topk, out_width, dtype=torch.bfloat16)
    for route, expert in enumerate(ids.flatten().tolist()):
        value = a_ref[route if bottom else route // topk] @ b_ref[expert].T
        if bottom:
            value *= router.flatten()[route]
        expected[route] = value.bfloat16()
    sorted_ids, expert_ids, padded = _cpu_moe_routing(ids, experts, 16)
    result = torch.empty((m*topk, out_width), dtype=torch.bfloat16, device='cuda')
    fused_moe(a.cuda(), b.cuda(), result,
              None if a_scale is None else a_scale.cuda(),
              None if b_scale is None else b_scale.cuda(), None,
              router.cuda(), ids.cuda(), sorted_ids.cuda(), None,
              expert_ids.cuda(), padded.cuda(), bottom, 1 if bottom else topk,
              compute_type=tl.bfloat16,
              config=dict(BLOCK_SIZE_M=16,
                          BLOCK_SIZE_N=128 if kind == "fp8_block128" else 16,
                          BLOCK_SIZE_K=128 if kind == "fp8_block128" else 32,
                          GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                          num_warps=4, num_stages=1), **flags)
    torch.testing.assert_close(result.cpu(), expected, rtol=.02, atol=.003)


@MXFP_REQUIRED
@pytest.mark.parametrize('bottom', [False, True], ids=['gemm1', 'gemm2'])
@pytest.mark.parametrize('a_group,width,scale_format',
                         [(128, 256, fmt) for fmt in ('e8m0', 'packed_e8m0', 'fp32_e8m0')]
                         + [(32, width, fmt) for width in (32, 160, 256, 2048)
                            for fmt in ('e8m0', 'fp32_e8m0')])
def test_moe_packed_mxfp4_recipes(bottom, a_group, width, scale_format):
    """No expansion on device: direct FP8 x packed E2M1, A g128/g32 and B g32 SFs.

    CPU oracle follows the upstream K32 scaled-product accumulation. Router
    weighting is deliberately absent: vLLM does it after GEMM output casting.
    """
    torch.manual_seed(67)
    m, topk, experts = 3, 2, 3
    out_width = 70 if bottom else 130  # masked N tails, independent of gating
    router = torch.tensor([[.3, .7, 1.], [.6, .4, 1.], [.1, .9, 1.]])[:, :topk]
    mul_router = bottom and a_group == 32
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    rows = m * topk if bottom else m
    # Independent CPU quantizer with already chosen power-of-two scales.
    raw = torch.randn(rows, width) * .1
    grouped = raw.reshape(rows, width//a_group, a_group)
    sa_float = torch.exp2(torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1.e-10) / 448)))
    a = (grouped / sa_float[..., None]).to(torch.float8_e4m3fn).reshape(rows, width)
    sa = (((sa_float.view(torch.int32) >> 23) & 255).to(torch.uint8))
    b = torch.randint(0, 256, (experts, out_width, width//2), dtype=torch.uint8)
    sb = torch.randint(121, 126, (experts, out_width, width//32), dtype=torch.uint8)
    # CPU reference decodes raw FP4 directly, never via the production expander.
    codes = torch.stack((b & 15, b >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    b_values = lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.)
    b_values *= torch.exp2(sb.float() - 127).repeat_interleave(32, -1)
    a_values = a.float() * sa_float.repeat_interleave(a_group, -1)
    expected = torch.empty(m*topk, out_width, dtype=torch.bfloat16)
    for route, expert in enumerate(ids.flatten().tolist()):
        row = route if bottom else route // topk
        acc = torch.zeros(out_width, dtype=torch.float32)
        for k32 in range(0, width, 32):
            acc += a_values[row, k32:k32+32] @ b_values[expert, :, k32:k32+32].T
        if mul_router:
            acc *= router.flatten()[route]
        expected[route] = acc.bfloat16()
    def format_scale(x):
        if scale_format == 'fp32_e8m0':
            return torch.exp2(x.float() - 127)
        if scale_format == 'packed_e8m0':
            padded = torch.full((*x.shape[:-1], ((x.shape[-1]+3)//4)*4), 127, dtype=torch.uint8)
            padded[..., :x.shape[-1]] = x
            v = padded.reshape(*x.shape[:-1], -1, 4).to(torch.int64)
            packed = sum(v[..., i] << (i*8) for i in range(4)).to(torch.int32)
            # Upstream transformed scales are MN-major, not row-major.
            return packed.transpose(-1, -2).contiguous().transpose(-1, -2)
        return x
    sorted_ids, expert_ids, padded = _cpu_moe_routing(ids, experts, 16)
    c = torch.empty(m*topk, out_width, dtype=torch.bfloat16, device='cuda')
    fused_moe(a.cuda(), b.cuda(), c, format_scale(sa).cuda(), format_scale(sb).cuda(), None,
              router.cuda(), ids.cuda(), sorted_ids.cuda(), None,
              expert_ids.cuda(), padded.cuda(), mul_router, 1 if bottom else topk,
              tl.bfloat16, use_mxfp4_w4a8=True)
    torch.testing.assert_close(c.cpu(), expected, rtol=.01, atol=.002)


@MXFP_REQUIRED
@pytest.mark.parametrize('topk', [6, 8, 10])
def test_packed_mxfp4_ck_routing_target_topk(topk):
    """CPU constructs CK-encoded routes; only the GEMM runs on device."""
    torch.manual_seed(92)
    m, experts, k, n = 1, 12, 128, 17
    ids = torch.tensor([[0] + list(range(2, topk+1))], dtype=torch.int32)
    a = torch.randn(m, k).to(torch.float8_e4m3fn)
    b = torch.randint(0, 256, (experts, n, k//2), dtype=torch.uint8)
    sa = torch.full((m, 1), 124, dtype=torch.uint8)
    sb = torch.full((experts, n, k//32), 123, dtype=torch.uint8)
    codes = torch.stack((b & 15, b >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    b_ref = lut[codes & 7] * torch.where((codes & 8) != 0, -1., 1.) / 16
    a_ref = a.float() / 8
    expected = torch.stack([a_ref[0] @ b_ref[e].T for e in ids[0].tolist()]).bfloat16()
    sorted_ids, eids, padded = _cpu_moe_routing(ids, experts, 16)
    encoded = torch.where(sorted_ids < m*topk,
                          (sorted_ids % topk) * (1 << 24) + sorted_ids // topk,
                          torch.full_like(sorted_ids, 0x00ffffff))
    result = torch.empty(topk, n, dtype=torch.bfloat16, device='cuda')
    fused_moe(a.cuda(), b.cuda(), result, sa.cuda(), sb.cuda(), None,
              torch.ones(m, topk).cuda(), ids.cuda(), encoded.cuda(), None,
              eids.cuda(), padded.cuda(), False, topk, tl.bfloat16,
              ck_sorting=True, ck_topk=topk, use_mxfp4_w4a8=True)
    torch.testing.assert_close(result.cpu(), expected, rtol=.01, atol=.002)


@pytest.mark.parametrize('flag', ['use_int8_w8a16', 'use_int4_w4a16', 'use_int4_w4a8'])
def test_grouped_quantized_bias_is_rejected(flag):
    # A CPU-only contract check: these grouped kernel ABIs have no bias input.
    a = torch.zeros(2, 128, dtype=torch.bfloat16)
    b = torch.zeros(1, 32, 128 if flag == 'use_int8_w8a16' else 64, dtype=torch.uint8)
    c = torch.empty(2, 32, dtype=torch.bfloat16)
    ids = torch.zeros(2, 1, dtype=torch.int32)
    sorted_ids, expert_ids, padded = _cpu_moe_routing(ids, 1, 16)
    with pytest.raises(ValueError, match='B_bias is not supported'):
        fused_moe(
            a, b, c, None, torch.ones(1, 32, 4), None,
            torch.ones(2, 1), ids, sorted_ids, None, expert_ids, padded,
            False, 1, tl.bfloat16, block_shape=[0, 32],
            B_bias=torch.ones(1, 32),
            config=dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=32),
            **{flag: True},
        )


@MXFP_REQUIRED
@pytest.mark.parametrize('bottom', [False, True])
@pytest.mark.parametrize('width', [160, 256])
def test_moe_mxfp8_scaled_dot(bottom, width):
    """CPU-prepared native K32 E8M0 scales, including K/N tails and routing."""
    a_group, block_shape = 32, (1, 32)
    torch.manual_seed(541)
    m, topk, experts, out_width = 3, 2, 3, 130
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]], dtype=torch.int32)
    router = torch.tensor([[.2, .8], [.7, .3], [.4, .6]])
    rows = m * topk if bottom else m
    a = (torch.randn(rows, width) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(experts, out_width, width) * 2).to(torch.float8_e4m3fn)
    sa = torch.randint(121, 127, (rows, width // a_group), dtype=torch.uint8)
    bo, bk = block_shape
    sb = torch.randint(121, 127, (experts, triton.cdiv(out_width, bo), triton.cdiv(width, bk)), dtype=torch.uint8)
    av = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(a_group, -1)
    bv = b.float() * torch.exp2(sb.float() - 127).repeat_interleave(bo, 1)[:, :out_width].repeat_interleave(bk, -1)[..., :width]
    expected = torch.empty(m * topk, out_width, dtype=torch.bfloat16)
    for route, expert in enumerate(ids.flatten().tolist()):
        acc = av[route if bottom else route // topk] @ bv[expert].T
        if bottom:
            acc *= router.flatten()[route]
        expected[route] = acc.bfloat16()

    sorted_ids, expert_ids, padded = _cpu_moe_routing(ids, experts, 16)
    c = torch.empty(m * topk, out_width, dtype=torch.bfloat16, device='cuda')
    fused_moe(a.cuda(), b.cuda(), c, sa.cuda(), sb.cuda(), None,
              router.cuda(), ids.cuda(), sorted_ids.cuda(), None,
              expert_ids.cuda(), padded.cuda(), bottom, 1 if bottom else topk,
              tl.bfloat16, use_mxfp8=True, block_shape=list(block_shape))
    torch.testing.assert_close(c.cpu(), expected, rtol=.015, atol=.002)
