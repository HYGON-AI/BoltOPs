# SPDX-License-Identifier: MIT
"""Single-K specialization: legality, rounding tolerance and unchanged fallbacks."""
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from boltops.fused_moe.triton import moe_op
from boltops.utility import arch_info


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
        a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=in_dtype)
        b = torch.randint(-127, 128, (experts, n, k), device="cuda", dtype=in_dtype)
    else:
        a = (torch.randn((rows, k), device="cuda") * 2).to(in_dtype)
        b = (torch.randn((experts, n, k), device="cuda") * 2).to(in_dtype)
        if w8a16:
            b = torch.randint(-127, 128, (experts, n, k), device="cuda", dtype=torch.int8)
    sa = sb = None
    if quant:
        sa = torch.rand((rows, triton.cdiv(k, group_k) if block else 1), device="cuda") * .01
        sb = torch.rand((experts, triton.cdiv(n, group_n) if block else n,
                         triton.cdiv(k, group_k) if block else 1), device="cuda") * .01
        if tensor:
            sa, sb = sa[0, 0].clone(), sb[:, 0, 0].clone()
    if w8a16:
        sb = torch.rand((experts, n), device="cuda") * .01
    ids = torch.randint(experts, (m, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand((m, topk), device="cuda")
    bias_value = torch.randn((experts, n), device="cuda") * .05 if bias else None
    if special:
        values = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"),
                               2.**-126, 2.**-133, 2.**-149, 65504., 65520.], device="cuda")
        if quant and not tensor:
            sa[:len(values), 0] = values
        elif not quant:
            a[:len(values), 0] = values.to(in_dtype)
        if bias:
            bias_value[:, :len(values)] = values
        weights.flatten()[:4] = torch.tensor([0., -0., 1., -1.], device="cuda")
    # Independent reference: materialize FP32 products/scales, then the public
    # epilogue order. Do not reuse the specialized Triton implementation.
    route_ids = ids.flatten().long()
    sources = torch.arange(routes, device="cuda") // tk
    effective_bk = min(bk, group_n, group_k) if block else bk
    ref = torch.zeros((routes, n), device="cuda")
    if block:
        ni = torch.arange(n, device="cuda") // group_n
        for start in range(0, k, effective_bk):
            dot = (a.float()[sources, start:start + effective_bk][:, None, :]
                   * b.float()[route_ids, :, start:start + effective_bk]).sum(-1)
            aa = sa[sources, start // group_k, None]
            bb = sb[route_ids[:, None], ni[None, :], start // group_k]
            ref += dot * aa * bb if bn > group_n else dot * (aa * bb)
    else:
        ref = (a.float()[sources, None, :] * b.float()[route_ids]).sum(-1)
    if routed:
        ref *= weights.flatten()[:, None]
    if quant and not block:
        ref = ref * (sa if tensor else sa[sources]) * (sb[route_ids, None] if tensor else sb[route_ids, :, 0])
    if w8a16:
        ref *= sb[route_ids]
    ref = ref.to(dtype)
    if bias:
        bias_ref = bias_value[route_ids]
        if scale_bias:
            bias_ref = bias_ref * weights.flatten()[:, None]
        ref = ref + bias_ref.to(dtype)
    sid_list, eid_list = [], []
    flat = ids.cpu().flatten()
    for expert in range(experts):
        selected = (flat == expert).nonzero().flatten().tolist()
        padded = triton.cdiv(len(selected), bm) * bm
        sid_list.extend(selected + [routes] * (padded - len(selected)))
        eid_list.extend([expert] * (padded // bm))
    sid = torch.tensor(sid_list, device="cuda", dtype=torch.int32)
    eid = torch.tensor(eid_list, device="cuda", dtype=torch.int32)
    count = torch.tensor([sid.numel()], device="cuda", dtype=torch.int32)
    valid = sid < routes
    if sorted_io:
        af = torch.zeros((sid.numel(), k), device="cuda")
        af[valid] = a.float()[sid[valid].long() // tk]
        a = af.to(in_dtype)
        if quant and not tensor:
            sf = torch.zeros((sid.numel(), sa.shape[1]), device="cuda")
            sf[valid] = sa[sid[valid].long() // tk]
            sa = sf
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
        moe_op.fused_moe(a, b, c, sa, sb, None, weights, ids, sid, None, eid, count,
                         routed, tk, compute, use_int8_w8a8=quant and not fp8,
                         use_fp8_w8a8=fp8, use_int8_w8a16=w8a16,
                         per_channel_quant=quant and not block and not tensor,
                         block_shape=[group_n, group_k] if block else None,
                         a_sorted=sorted_io, c_sorted=sorted_io, B_bias=bias_value,
                         scale_bias_with_routed_weight=scale_bias, config=cfg)
        torch.cuda.synchronize()
        assert cfg == saved
        assert (storage[-1] == -123).all() and (storage[:, -1] == -123).all()
        if sorted_io:
            assert (c[~valid] == -123).all()
            out = torch.empty((routes, n), device="cuda", dtype=dtype)
            out[sid[valid].long()] = c[valid]
        else:
            out = c.clone()
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
