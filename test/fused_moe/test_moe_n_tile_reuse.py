"""Legality, search pruning and persistence of adjacent-N A reuse."""
import json

import pytest
import torch
import triton.language as tl

from boltops.fused_moe.triton import moe_op
from boltops.fused_moe.triton.moe_activation import adjust_N_for_activation
from boltops.tools.fused_moe_triton_tune import autotune_patches as tune
from boltops.tools.fused_moe_triton_tune import moe_log_parser


def _args(**overrides):
    return dict({
        "N": 512, "K": 128, "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
        "MUL_ROUTED_WEIGHT": True,
        "COMBINE_SCALE_LOAD": False, "USE_MLS_LOAD": False,
        "bottom_a_use_mls_load": False, "num_warps": 4,
        "use_int8_w8a8": True, "use_fp8_w8a8": False,
        "use_int8_w8a16": False, "per_channel_quant": True,
        "group_n": 0, "group_k": 0, "compute_type": tl.float16,
    }, **overrides)


@pytest.fixture
def stub_launch(monkeypatch):
    captured = {}

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                captured.update(kwargs)
                captured["grid"] = grid(kwargs)
            return run

    monkeypatch.setattr(moe_op, "fused_moe_kernel", Kernel())
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", False)

    def launch(**config_changes):
        config = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128,
                      GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False,
                      num_warps=4, num_stages=1, SPLIT_K=0)
        config.update(config_changes)
        saved = dict(config)
        a = torch.empty((1, 128), dtype=torch.bfloat16)
        b = torch.empty((1, 512, 128), dtype=torch.bfloat16)
        c = torch.empty((1, 512), dtype=torch.bfloat16)
        ids = torch.zeros((1, 1), dtype=torch.int32)
        sid = torch.zeros((32,), dtype=torch.int32)
        eid = torch.zeros((1,), dtype=torch.int32)
        count = torch.tensor([32], dtype=torch.int32)
        weights = torch.ones((1, 1))
        moe_op.fused_moe(a, b, c, None, None, None, weights, ids, sid,
                         None, eid, count, True, 1, tl.bfloat16, config=config)
        assert config == saved
        return captured

    return launch


@pytest.mark.parametrize("value", [None, 0, 3, -1, True, 2.0, "2"])
def test_invalid_parameter(stub_launch, value):
    with pytest.raises(ValueError, match="integer 1 or 2"):
        stub_launch(N_TILES_PER_CTA=value)


@pytest.mark.parametrize("changes,reason", [
    ({"K": 129}, "single K tile"),
    ({"K": 0}, "single K tile"),
    ({"N": 256}, "more than one N tile"),
    ({"USE_MLS_LOAD": True}, "MLS"),
    ({"bottom_a_use_mls_load": True}, "MLS"),
    ({"COMBINE_SCALE_LOAD": True}, "COMBINE"),
    ({"use_block_pingpong": True}, "BlockPingpong"),
    ({"SPLIT_K": 2}, "non-Split-K"),
    ({"persistent": True}, "non-Split-K"),
    ({"use_fp8_w8a8": True}, "supports only"),
    ({"use_int8_w8a16": True}, "supports only"),
    ({"use_int4_w4a16": True}, "supports only"),
    ({"use_int4_w4a8": True}, "supports only"),
    ({"use_mxfp4_w4a4": True}, "supports only"),
    ({"per_channel_quant": False}, "supports only"),
    ({"num_warps": 8}, "4 waves"),
    ({"num_warps": 2}, "4 waves"),
    ({"group_n": 128, "group_k": 128, "MUL_ROUTED_WEIGHT": False}, "rounding"),
    ({"group_n": 128, "group_k": 64}, "quantization group"),
    ({"use_int8_w8a8": False, "compute_type": tl.float16}, "supports only"),
    ({"use_int8_w8a8": False, "compute_type": tl.bfloat16,
      "MUL_ROUTED_WEIGHT": False}, "supports only"),
])
def test_reject_unsupported_reuse(changes, reason):
    assert reason in moe_op._n_tile_reuse_error(_args(**changes))


@pytest.mark.parametrize("changes", [
    {}, {"MUL_ROUTED_WEIGHT": False}, {"K": 63}, {"N": 257},
    {"N": 769, "a_sorted": True, "c_sorted": True},
    {"group_n": 128, "group_k": 128, "num_warps": 8},
    {"use_int8_w8a8": False, "compute_type": tl.bfloat16, "num_warps": 8},
])
def test_supported_reuse(changes):
    assert moe_op._n_tile_reuse_error(_args(**changes)) is None


@pytest.mark.parametrize("config,expected,grid_n", [
    ({}, 1, 2), ({"N_TILES_PER_CTA": 1}, 1, 2),
    ({"N_TILES_PER_CTA": 2}, 2, 1),
])
def test_explicit_choice_controls_kernel_and_grid(stub_launch, config, expected, grid_n):
    result = stub_launch(**config)
    assert result["N_TILES_PER_CTA"] == expected
    assert result["grid"] == (grid_n,)
    assert "n_tiles_per_cta" not in result


@pytest.mark.parametrize("config,reason", [
    ({"BLOCK_SIZE_K": 64}, "single K tile"),
    ({"BLOCK_SIZE_N": 512}, "more than one N tile"),
    ({"use_block_pingpong": True}, "BlockPingpong"),
])
def test_runtime_checks_only_explicit_reuse(stub_launch, config, reason):
    assert stub_launch(**config)["N_TILES_PER_CTA"] == 1
    with pytest.raises(ValueError, match=reason):
        stub_launch(**config, N_TILES_PER_CTA=2)


def test_n_tile_choice_is_not_a_heuristic():
    assert "n_tiles_per_cta" not in moe_op.fused_moe_kernel.values
    assert "N_TILES_PER_CTA" not in moe_op.fused_moe_kernel.values
    assert "n_tiles_per_cta" not in moe_op.fused_moe_kernel.arg_names


@pytest.mark.parametrize("gemm1_n,is_gated,legal", [
    (256, True, True), (256, False, False),
    (128, False, True), (512, True, False),
])
def test_reuse_checks_actual_k_after_activation(gemm1_n, is_gated, legal):
    k = adjust_N_for_activation(gemm1_n, is_gated)
    assert (moe_op._n_tile_reuse_error(_args(K=k)) is None) == legal


def test_partial_second_tile_is_legal_but_not_in_default_search():
    args = _args(N=257, N_TILES_PER_CTA=2)
    assert moe_op._n_tile_reuse_error(args) is None
    assert not tune._keep_n_tile_reuse({}, args, 4)
    assert tune._keep_n_tile_reuse({}, {**args, "N": 512}, 4)


@pytest.fixture
def narrow_search(monkeypatch):
    class NoCompilerOptions:
        @staticmethod
        def supports(name):
            return False
    monkeypatch.setattr(tune, "triton_capabilities", NoCompilerOptions())
    monkeypatch.setattr(tune, "capMLS", False)
    monkeypatch.setattr(tune, "compile_only", 0)
    monkeypatch.setattr(tune, "mode_best_config_tune", False)
    for suffix in ("", "1", "2"):
        for bound in ("min", "max"):
            monkeypatch.setattr(tune, f"mode_autotune_{bound}_block_size_k{suffix}", 128, raising=False)
    for suffix in ("1", "2"):
        for bound in ("min", "max"):
            monkeypatch.setattr(tune, f"mode_autotune_{bound}_block_size_n{suffix}", 256)


@pytest.mark.parametrize("mode", ["int8", "int8_channel", "bf16"])
@pytest.mark.parametrize("m", [512, 4096, 8192, 16384])
def test_search_has_both_choices(narrow_search, monkeypatch, mode, m):
    monkeypatch.setattr(tune, "test_type", mode)
    configs = tune.generate_config2_lists(
        BLOCK_SIZE_M=128, N=7168, K=128, MUL_ROUTED_WEIGHT=True,
        per_channel_quant=mode == "int8_channel", num_input_tokens=m,
    )
    assert {cfg.kwargs["N_TILES_PER_CTA"] for cfg in configs} == {1, 2}
    for cfg in configs:
        if cfg.kwargs["N_TILES_PER_CTA"] == 2:
            assert not cfg.kwargs["USE_MLS_LOAD"]
            assert not cfg.kwargs["COMBINE_SCALE_LOAD"]
            if mode == "int8_channel":
                assert cfg.num_warps == 4


@pytest.mark.parametrize("overrides", [
    {"K": 7168}, {"N": 384}, {"runtime_args": {"use_fp8_w8a8": True}},
    {"runtime_args": {"use_int8_w8a8": False, "compute_type": tl.float16}},
    {"runtime_args": {"group_n": 128, "group_k": 64}},
])
def test_search_prunes_unsupported_reuse(narrow_search, monkeypatch, overrides):
    monkeypatch.setattr(tune, "test_type", "int8_channel")
    args = dict(BLOCK_SIZE_M=128, N=7168, K=128,
                MUL_ROUTED_WEIGHT=True, per_channel_quant=True)
    configs = tune.generate_config2_lists(**{**args, **overrides})
    assert configs
    assert {cfg.kwargs["N_TILES_PER_CTA"] for cfg in configs} == {1}


def test_specialized_kernel_does_not_get_new_keyword(narrow_search, monkeypatch):
    monkeypatch.setattr(tune, "test_type", "int8")
    configs = tune.generate_config2_lists(
        BLOCK_SIZE_M=128, N=7168, K=128,
        MUL_ROUTED_WEIGHT=True, supports_n_tiles=False,
    )
    assert configs and all("N_TILES_PER_CTA" not in cfg.kwargs for cfg in configs)


def test_runtime_guard_uses_actual_group_and_warp_count():
    assert tune._keep_n_tile_reuse({}, _args(), 4)
    assert not tune._keep_n_tile_reuse({}, _args(), 8)
    assert not tune._keep_n_tile_reuse({}, _args(group_n=128, group_k=64), 4)
    assert not tune._keep_n_tile_reuse({"USE_MLS_LOAD": True}, _args(), 4)


def test_log_parser_roundtrip_keeps_gemm_choices_independent(tmp_path):
    kernel = (
        "fused_moe_kernel:\n('key',) Configs:\n"
        "* 0  : BLOCK_SIZE_N: 256, BLOCK_SIZE_K: 128, GROUP_SIZE_M: 1, "
        "COMBINE_SCALE_LOAD: False, USE_MLS_LOAD: False, "
        "N_TILES_PER_CTA: {}, num_warps: 4, num_stages: 1\n"
        "\t : times(20 statics, rep:20 ms): 1.0 ms\n"
    )
    log = (
        "Test: m=8192, n=128, k=7168, e=256, topk=8\n"
        "Config 1: {'BLOCK_SIZE_M': 128}\n"
        + kernel.format(1) + kernel.format(2)
    )
    results = moe_log_parser.parse_log_file(log)
    top, bottom = tmp_path / "top.json", tmp_path / "bottom.json"
    moe_log_parser.save_results_to_json(results, top, bottom)
    assert json.loads(top.read_text())["8192"]["N_TILES_PER_CTA"] == 1
    assert json.loads(bottom.read_text())["8192"]["N_TILES_PER_CTA"] == 2
    assert json.loads(top.read_text())["8192"]["BLOCK_SIZE_M"] == 128
    assert json.loads(bottom.read_text())["8192"]["BLOCK_SIZE_M"] == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["int8", "int8_channel", "bf16"])
@pytest.mark.parametrize("n", [64, 257, 512])
def test_launch_explicit_choices_and_tail_masks(mode, n):
    torch.manual_seed(71)
    m, e, k, bm, bn, bk = 7, 2, 63, 32, 128, 128
    quant = mode != "bf16"
    dtype = torch.float16 if quant else torch.bfloat16
    if quant:
        a = torch.randint(-32, 32, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-32, 32, (e, n, k), device="cuda", dtype=torch.int8)
        sa = torch.rand((m, 1), device="cuda") * 0.01
        scale_n = (n + 127) // 128 if mode == "int8" else n
        sb = torch.rand((e, scale_n, 1), device="cuda") * 0.01
    else:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((e, n, k), device="cuda", dtype=dtype)
        sa = sb = None
    ids = (torch.arange(m, device="cuda", dtype=torch.int32) % e)[:, None]
    weights = torch.rand((m, 1), device="cuda")
    sid = torch.tensor([0, 2, 4, 6] + [m] * 28 + [1, 3, 5] + [m] * 29,
                       device="cuda", dtype=torch.int32)
    eid = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    count = torch.tensor([sid.numel()], device="cuda", dtype=torch.int32)
    config = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                  GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False,
                  num_warps=4, num_stages=1)

    def launch(choice):
        candidate = {**config, **({"N_TILES_PER_CTA": choice} if choice else {})}
        saved = dict(candidate)
        c = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
        moe_op.fused_moe(
            a, b, c, sa, sb, None, weights, ids, sid, None, eid, count, True, 1,
            tl.float16 if quant else tl.bfloat16,
            use_int8_w8a8=quant, per_channel_quant=mode == "int8_channel",
            block_shape=[128, 128] if mode == "int8" else None, config=candidate,
        )
        assert candidate == saved, "launch must not mutate the saved config"
        return c

    ordinary, default = launch(1), launch(None)
    assert torch.equal(ordinary, default)
    if n > bn:
        assert torch.equal(ordinary, launch(2))
    else:
        with pytest.raises(ValueError, match="more than one N tile"):
            launch(2)
    ref = (a.float()[:, None, :] * b[ids[:, 0].long()].float()).sum(-1)
    if mode == "int8":
        scale_idx = torch.arange(n, device="cuda") // 128
        ref = ref * (sa * sb[ids[:, 0].long()][:, scale_idx, 0])
    ref = ref * weights
    if mode == "int8_channel":
        ref = ref * sa * sb[ids[:, 0].long(), :, 0]
    torch.testing.assert_close(ordinary, ref.to(dtype), rtol=0.01, atol=0.002)
