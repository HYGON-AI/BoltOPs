import json
import importlib
import re
from pathlib import Path

import pytest
from triton.runtime.errors import HSACOError

from boltops.fused_moe.triton import moe_config_utils
from boltops.tools.fused_moe_triton_tune import moe_log_parser
from boltops.tools.fused_moe_triton_tune import autotune_patches
from boltops.tools.fused_moe_triton_tune import tune_moe_cli
from boltops.utility.triton_capability import TritonCapabilities


fused_moe_impl = importlib.import_module("boltops.fused_moe.triton.fused_moe")


def _caps(version: str, *options: str) -> TritonCapabilities:
    return TritonCapabilities(version, frozenset(options), "hcu", version)


@pytest.fixture(autouse=True)
def _clear_moe_config_cache():
    moe_config_utils.get_moe_configs.cache_clear()
    yield
    moe_config_utils.get_moe_configs.cache_clear()


def _write_config(config_root: Path, version: str, file_name: str, marker: str):
    config_dir = config_root / f"triton_{version.replace('.', '_')}"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / file_name).write_text(
        json.dumps({"1": {"marker": marker}}), encoding="utf-8"
    )


def _load_configs(monkeypatch, tmp_path, capability_version: str):
    config_root = tmp_path / "configs"
    monkeypatch.setattr(moe_config_utils, "_TRITON_CONFIGS_DIR", config_root)
    monkeypatch.setattr(moe_config_utils.arch_info, "get_arch", lambda: "gfx936")
    monkeypatch.setattr(
        moe_config_utils,
        "get_triton_capabilities",
        lambda: _caps(capability_version),
    )
    moe_config_utils.get_moe_configs.cache_clear()
    return config_root


def test_moe_config_prefers_current_capability_then_falls_back(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    file_name = "E=256,N=256,arch=gfx936.json"
    _write_config(config_root, "3.2", file_name, "lower")

    loaded = moe_config_utils.get_moe_configs(256, 256, None)
    assert loaded == {1: {"marker": "lower"}}

    _write_config(config_root, "3.6", file_name, "current")
    moe_config_utils.get_moe_configs.cache_clear()
    loaded = moe_config_utils.get_moe_configs(256, 256, None)
    assert loaded == {1: {"marker": "current"}}


def test_lower_capability_never_reads_higher_config(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.2")
    file_name = "E=256,N=256,arch=gfx936.json"
    _write_config(config_root, "3.6", file_name, "higher")

    assert moe_config_utils.get_moe_configs(256, 256, None) is None


def test_exact_k_topk_match_precedes_newer_legacy_config(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    legacy_name = "E=256,N=128,arch=gfx936,dtype=int8_w8a8.json"
    exact_name = (
        "E=256,N=128,K=7168,topk=8,arch=gfx936,dtype=int8_w8a8.json"
    )
    _write_config(config_root, "3.6", legacy_name, "newer-legacy")
    _write_config(config_root, "3.2", exact_name, "lower-exact")

    loaded = moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=8
    )
    assert loaded == {1: {"marker": "lower-exact"}}


def test_fuzzy_k_topk_selector_and_legacy_fallback(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    fuzzy_name = (
        "E=256,N=128,KMin=4096,KMod=1024,topkSet=4-8,"
        "arch=gfx936,dtype=int8_w8a8.json"
    )
    legacy_name = "E=256,N=128,arch=gfx936,dtype=int8_w8a8.json"
    _write_config(config_root, "3.6", fuzzy_name, "fuzzy")
    _write_config(config_root, "3.6", legacy_name, "legacy")

    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=8
    ) == {1: {"marker": "fuzzy"}}
    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=4
    ) == {1: {"marker": "fuzzy"}}
    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=3584, top_k=8
    ) == {1: {"marker": "legacy"}}
    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=2
    ) == {1: {"marker": "legacy"}}


def test_newer_fuzzy_config_precedes_more_specific_old_fuzzy(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    newer = (
        "E=256,N=128,KMin=4096,KMod=1024,topkSet=4-8,"
        "arch=gfx936,dtype=int8_w8a8.json"
    )
    older = (
        "E=256,N=128,KSet=6144-7168,topk=8,"
        "arch=gfx936,dtype=int8_w8a8.json"
    )
    _write_config(config_root, "3.6", newer, "newer-broad")
    _write_config(config_root, "3.2", older, "older-narrow")

    loaded = moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=8
    )
    assert loaded == {1: {"marker": "newer-broad"}}


@pytest.mark.parametrize(
    "fragment",
    (
        "KMin=4096,topk=8",
        "KMin=1,KMod=0,topk=8",
        "KMin=4096,KMax=8192,KMod=1024,topk=8",
        "K=4096,KSet=4096-7168,topk=8",
    ),
)
def test_invalid_shape_selector_is_rejected(fragment):
    assert moe_config_utils._parse_config_selector(fragment) is None


def test_kmin_kmod_selector_has_no_upper_bound():
    selector = moe_config_utils._parse_config_selector(
        "KMin=4096,KMod=1024,topk=8"
    )
    assert selector is not None
    assert moe_config_utils._selector_matches(selector["K"], 4096)
    assert moe_config_utils._selector_matches(selector["K"], 16384)
    assert not moe_config_utils._selector_matches(selector["K"], 4608)


def test_k_topk_are_part_of_config_cache_key(monkeypatch, tmp_path):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    for K, marker in ((4096, "k4096"), (7168, "k7168")):
        name = (
            f"E=256,N=128,K={K},topk=8,arch=gfx936,"
            "dtype=int8_w8a8.json"
        )
        _write_config(config_root, "3.6", name, marker)

    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=4096, top_k=8
    ) == {1: {"marker": "k4096"}}
    assert moe_config_utils.get_moe_configs(
        256, 128, "int8_w8a8", K=7168, top_k=8
    ) == {1: {"marker": "k7168"}}


@pytest.mark.parametrize(
    "e,n,k,topk,arch,dtype",
    (
        (392, 1536, 1232, 1, "gfx936", None),
        (128, 192, 2048, 8, "gfx938", "int8_w8a8"),
    ),
)
def test_exact_only_config_does_not_match_other_shapes(
    monkeypatch, tmp_path, e, n, k, topk, arch, dtype
):
    config_root = _load_configs(monkeypatch, tmp_path, "3.6")
    monkeypatch.setattr(moe_config_utils.arch_info, "get_arch", lambda: arch)
    dtype_suffix = f",dtype={dtype}" if dtype is not None else ""
    exact_name = f"E={e},N={n},K={k},topk={topk},arch={arch}{dtype_suffix}.json"
    _write_config(config_root, "3.6", exact_name, "exact-only")

    assert moe_config_utils.get_moe_configs(
        e, n, dtype, K=k, top_k=topk
    ) == {1: {"marker": "exact-only"}}
    assert moe_config_utils.get_moe_configs(
        e, n, dtype, K=k + 16, top_k=topk
    ) is None
    assert moe_config_utils.get_moe_configs(
        e, n, dtype, K=k, top_k=topk + 1
    ) is None


def test_tune_cli_names_exact_k_and_topk_configs():
    top, bottom = tune_moe_cli.expected_json_names(
        device_name="bmz",
        test_type="int8_channel",
        local_e=256,
        local_n=128,
        K=7168,
        top_k=8,
    )

    assert top == (
        "E=256,N=128,K=7168,topk=8,arch=gfx936,dtype=int8_w8a8.json"
    )
    assert bottom == (
        "E=256,N=128,K=7168,topk=8,arch=gfx936,dtype=int8_w8a8,"
        "is_bottom=True.json"
    )


def _autotune_log(
    use_block_pingpong: str | None,
    buffer_cache_swizzle: str | None = None,
    bottom_a_use_mls_load: str | None = None,
    a_sorted: str | None = None,
) -> str:
    pingpong = (
        "" if use_block_pingpong is None else f", use_block_pingpong: {use_block_pingpong}"
    )
    cache_swizzle = (
        ""
        if buffer_cache_swizzle is None
        else f", buffer_cache_swizzle: {buffer_cache_swizzle}"
    )
    bottom_a_mls = (
        ""
        if bottom_a_use_mls_load is None
        else f", bottom_a_use_mls_load: {bottom_a_use_mls_load}"
    )
    sorted_a = "" if a_sorted is None else f", a_sorted: {a_sorted}"
    kernel = (
        "fused_moe_kernel:\n"
        "('key',) Configs:\n"
        "* 0  : BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 1, "
        "COMBINE_SCALE_LOAD: False, USE_MLS_LOAD: False, waves_per_eu: 1, "
        "instruction_sched_variant: none, sched_latency: none, kpack: 1"
        f"{pingpong}{cache_swizzle}{bottom_a_mls}{sorted_a}, "
        "num_warps: 8, num_ctas: 1, "
        "num_stages: 2, maxnreg: None\n"
        "\t : times(20 statics, rep:20 ms): 1.0 ms\n"
    )
    return "Config 1: {'BLOCK_SIZE_M': 64}\n" + kernel + kernel


def test_log_parser_preserves_supported_pingpong_option(tmp_path):
    results = moe_log_parser.parse_log_file(_autotune_log("True"))
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert json.loads(top.read_text())["1"]["use_block_pingpong"] is True
    assert json.loads(bottom.read_text())["1"]["use_block_pingpong"] is True
    assert "waves_per_eu" not in json.loads(top.read_text())["1"]
    assert "waves_per_eu" not in json.loads(bottom.read_text())["1"]


def test_log_parser_omits_unsupported_pingpong_option(tmp_path):
    results = moe_log_parser.parse_log_file(_autotune_log(None))
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert "use_block_pingpong" not in json.loads(top.read_text())["1"]
    assert "use_block_pingpong" not in json.loads(bottom.read_text())["1"]


def test_log_parser_preserves_supported_buffer_cache_swizzle_option(tmp_path):
    results = moe_log_parser.parse_log_file(_autotune_log("False", "True"))
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert json.loads(top.read_text())["1"]["buffer_cache_swizzle"] is True
    assert json.loads(bottom.read_text())["1"]["buffer_cache_swizzle"] is True


def test_log_parser_preserves_bottom_a_mls_only_for_gemm2(tmp_path):
    results = moe_log_parser.parse_log_file(
        _autotune_log("False", "False", "True")
    )
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert "bottom_a_use_mls_load" not in json.loads(top.read_text())["1"]
    assert json.loads(bottom.read_text())["1"]["bottom_a_use_mls_load"] is True


def test_log_parser_preserves_sorted_bridge_with_ordinary_load(tmp_path):
    results = moe_log_parser.parse_log_file(
        _autotune_log("False", bottom_a_use_mls_load="False", a_sorted="True")
    )
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert "a_sorted" not in json.loads(top.read_text())["1"]
    config = json.loads(bottom.read_text())["1"]
    assert config["a_sorted"] is True
    assert config["bottom_a_use_mls_load"] is False


def test_log_parser_uses_logged_batch_key_and_sorts_output(tmp_path):
    missing = (
        "Test: m=1024, n=1536, k=1232, e=392, topk=1\n"
        "Config 1: {'BLOCK_SIZE_M': 64}\n"
    )
    valid_bf16 = (
        "Test: m=2048, n=1536, k=1232, e=392, topk=1\n"
        + _autotune_log("True")
    )
    valid_fp8 = (
        "Test Count 3: m=4096, n=1536, k=1232, e=392, topk=1\n"
        + _autotune_log("True")
    )
    results = moe_log_parser.parse_log_file(missing + valid_fp8 + valid_bf16)
    top = tmp_path / "top.json"
    bottom = tmp_path / "bottom.json"

    moe_log_parser.save_results_to_json(results, top, bottom)

    assert list(json.loads(top.read_text())) == ["2048", "4096"]
    assert list(json.loads(bottom.read_text())) == ["2048", "4096"]


def test_tuner_copies_results_to_current_capability_directory(monkeypatch):
    monkeypatch.setattr(
        moe_log_parser, "get_triton_capabilities", lambda: _caps("3.6")
    )

    expected = (
        Path(moe_log_parser.__file__).resolve().parents[2]
        / "fused_moe"
        / "triton"
        / "configs"
        / "triton_3_6"
    )
    assert moe_log_parser.resolve_boltops_moe_config_dir() == expected


def test_gemm2_large_option_pruning_uses_input_token_count(monkeypatch):
    class _BufferCacheSwizzleOnly:
        @staticmethod
        def supports(name):
            return name == "buffer_cache_swizzle"

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _BufferCacheSwizzleOnly()
    )
    monkeypatch.setattr(autotune_patches, "test_type", "bf16")
    monkeypatch.setattr(autotune_patches, "capMLS", False)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)

    def configs(num_input_tokens):
        return autotune_patches.generate_config2_lists(
            BLOCK_SIZE_M=128,
            N=256,
            K=4096,
            MUL_ROUTED_WEIGHT=True,
            num_valid_tokens=4096,
            top_k=1,
            num_input_tokens=num_input_tokens,
        )

    small = configs(512)
    large = configs(2048)

    assert small
    assert large
    assert not any(c.kwargs.get("buffer_cache_swizzle") for c in small)
    assert any(c.kwargs.get("buffer_cache_swizzle") for c in large)


def test_int8_channel_autotune_includes_legal_kpack2_configs(monkeypatch):
    class _KpackOnly:
        @staticmethod
        def supports(name):
            return name == "kpack"

    monkeypatch.setattr(autotune_patches, "triton_capabilities", _KpackOnly())
    monkeypatch.setattr(autotune_patches, "test_type", "int8_channel")
    monkeypatch.setattr(autotune_patches, "capMLS", False)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n1", 128)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 128)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k1", 32, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 128, raising=False
    )

    configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=256,
        K=7168,
        MUL_ROUTED_WEIGHT=False,
        per_channel_quant=True,
        num_valid_tokens=8192 * 8,
        top_k=8,
        num_input_tokens=8192,
    )

    generated = {
        (c.kwargs["BLOCK_SIZE_K"], c.kwargs.get("kpack")) for c in configs
    }
    assert not any("waves_per_eu" in c.kwargs for c in configs)
    assert (32, 1) in generated
    assert (32, 2) not in generated
    assert (64, 1) in generated
    assert (64, 2) in generated
    assert (128, 2) in generated


@pytest.mark.parametrize("tune_type", ["fp8_channel", "int8_channel"])
def test_mls_autotune_excludes_ignored_kpack2_and_pingpong(monkeypatch, tune_type):
    class _KpackAndPingpong:
        @staticmethod
        def supports(name):
            return name in ("kpack", "use_block_pingpong")

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _KpackAndPingpong()
    )
    monkeypatch.setattr(autotune_patches, "test_type", tune_type)
    monkeypatch.setattr(autotune_patches, "capMLS", True)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n1", 128)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 128)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k1", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 64, raising=False
    )
    configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=256,
        K=7168,
        MUL_ROUTED_WEIGHT=False,
        per_channel_quant=True,
        num_valid_tokens=8192 * 8,
        top_k=8,
        num_input_tokens=8192,
    )

    assert any(c.kwargs["USE_MLS_LOAD"] for c in configs)
    assert any(c.kwargs.get("use_block_pingpong", False) for c in configs)
    assert not any(
        (c.kwargs["USE_MLS_LOAD"] or c.kwargs.get("bottom_a_use_mls_load", False))
        and (c.kwargs.get("kpack") == 2 or c.kwargs.get("use_block_pingpong", False))
        for c in configs
    )


@pytest.mark.parametrize("tune_type", ["fp8_channel", "int8_channel"])
def test_bottom_a_mls_does_not_prune_gemm1_load_options(monkeypatch, tune_type):
    class _KpackAndPingpong:
        @staticmethod
        def supports(name):
            return name in ("kpack", "use_block_pingpong")

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _KpackAndPingpong()
    )
    monkeypatch.setattr(autotune_patches, "test_type", tune_type)
    monkeypatch.setattr(autotune_patches, "capMLS", True)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n1", 128)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 128)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k1", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k2", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k2", 64, raising=False
    )

    gemm1_configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=256,
        K=7168,
        MUL_ROUTED_WEIGHT=False,
        per_channel_quant=True,
        bottom_a_use_mls_load=True,
        num_valid_tokens=8192 * 8,
        top_k=8,
        num_input_tokens=8192,
    )
    assert any(
        not c.kwargs["USE_MLS_LOAD"] and c.kwargs.get("kpack") == 2
        for c in gemm1_configs
    )
    assert any(
        not c.kwargs["USE_MLS_LOAD"]
        and c.kwargs.get("use_block_pingpong", False)
        for c in gemm1_configs
    )
    assert not any(
        c.kwargs["USE_MLS_LOAD"]
        and (c.kwargs.get("kpack") == 2 or c.kwargs.get("use_block_pingpong", False))
        for c in gemm1_configs
    )

    gemm2_configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=7168,
        K=256,
        MUL_ROUTED_WEIGHT=True,
        per_channel_quant=True,
        bottom_a_use_mls_load=True,
        num_valid_tokens=8192 * 8,
        top_k=1,
        num_input_tokens=8192,
    )
    assert gemm2_configs
    mls_configs = [c for c in gemm2_configs if
                   c.kwargs["USE_MLS_LOAD"] or c.kwargs["bottom_a_use_mls_load"]]
    assert mls_configs
    assert not any(
        c.kwargs.get("kpack") == 2
        or c.kwargs.get("use_block_pingpong", False)
        for c in mls_configs
    )
    assert any(c.kwargs.get("kpack") == 2 for c in gemm2_configs)
    assert all(c.kwargs["a_sorted"] for c in gemm2_configs)


def test_bottom_a_sorted_buffer_load_keeps_gemm2_load_options(monkeypatch):
    class _KpackAndPingpong:
        @staticmethod
        def supports(name):
            return name in ("kpack", "use_block_pingpong")

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _KpackAndPingpong()
    )
    monkeypatch.setattr(autotune_patches, "test_type", "int8_channel")
    monkeypatch.setattr(autotune_patches, "capMLS", False)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n2", 128)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n2", 128)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k2", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k2", 64, raising=False
    )

    configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=7168,
        K=512,
        MUL_ROUTED_WEIGHT=True,
        per_channel_quant=True,
        bottom_a_use_mls_load=True,
        num_valid_tokens=8192 * 8,
        top_k=1,
        num_input_tokens=8192,
    )

    assert configs
    assert any(c.kwargs.get("kpack") == 2 for c in configs)
    assert any(c.kwargs.get("use_block_pingpong", False) for c in configs)
    assert all(not c.kwargs["USE_MLS_LOAD"] for c in configs)
    assert all(c.kwargs["a_sorted"] for c in configs)
    assert all(not c.kwargs["bottom_a_use_mls_load"] for c in configs)


@pytest.mark.parametrize("cap_mls", [False, True])
def test_bottom_a_sorted_layout_is_an_outer_fused_moe_choice(
    monkeypatch, cap_mls
):
    monkeypatch.setattr(autotune_patches, "capMLS", cap_mls)
    monkeypatch.setattr(autotune_patches, "test_type", "fp8_channel")
    monkeypatch.setattr(autotune_patches, "mode_best_config_tune", False)

    configs = autotune_patches.generate_config_lists()

    assert len(configs) == 10
    assert {
        (config["BLOCK_SIZE_M"], config["bottom_a_use_mls_load"])
        for config in configs
    } == {
        (block_m, bottom_a_mls)
        for block_m in (16, 32, 64, 128, 256)
        for bottom_a_mls in (False, True)
    }


def test_autotune_buffer_reserves_sorted_bridge_padding_without_mls(monkeypatch):
    monkeypatch.setattr(autotune_patches, "capMLS", False)

    total = autotune_patches._max_autotune_total_tokens(
        13104,
        1,
        392,
        allow_sorted_bridge=True,
    )

    assert total == 13104 + 393 * 255
    assert total > 13104


def test_gemm2_inner_search_keeps_outer_bottom_a_mls_choice(monkeypatch):
    class _NoCompilerOptions:
        @staticmethod
        def supports(_name):
            return False

    monkeypatch.setattr(autotune_patches, "triton_capabilities", _NoCompilerOptions())
    monkeypatch.setattr(autotune_patches, "test_type", "fp8_channel")
    monkeypatch.setattr(autotune_patches, "capMLS", True)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k2", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k2", 128, raising=False
    )
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n2", 16)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n2", 512)

    configs = autotune_patches.generate_config2_lists(
        BLOCK_SIZE_M=128,
        N=256,
        K=7168,
        MUL_ROUTED_WEIGHT=True,
        per_channel_quant=True,
        bottom_a_use_mls_load=True,
        num_valid_tokens=8192 * 8,
        top_k=1,
        num_input_tokens=8192,
    )

    assert configs
    assert all(c.kwargs["a_sorted"] is True for c in configs)
    assert {c.kwargs["bottom_a_use_mls_load"] for c in configs} == {False, True}


def test_bottom_a_mls_config_is_explicit(monkeypatch):
    monkeypatch.setattr(fused_moe_impl, "_bottom_moe_use_mls", lambda: True)
    assert not fused_moe_impl._bottom_a_uses_sorted_layout(
        {"USE_MLS_LOAD": True}, False
    )
    assert not fused_moe_impl._bottom_a_uses_mls({"USE_MLS_LOAD": True}, False)
    assert not fused_moe_impl._bottom_a_uses_mls(
        {"USE_MLS_LOAD": True, "bottom_a_use_mls_load": False}, False
    )
    assert fused_moe_impl._bottom_a_uses_mls(
        {"USE_MLS_LOAD": False, "bottom_a_use_mls_load": True}, False
    )
    assert fused_moe_impl._bottom_a_uses_sorted_layout(
        {"USE_MLS_LOAD": False, "bottom_a_use_mls_load": True}, False
    )
    assert not fused_moe_impl._bottom_a_uses_mls(
        {"USE_MLS_LOAD": True, "bottom_a_use_mls_load": True}, True
    )

    monkeypatch.setattr(fused_moe_impl, "_bottom_moe_use_mls", lambda: False)
    assert not fused_moe_impl._bottom_a_uses_mls(
        {"USE_MLS_LOAD": True, "bottom_a_use_mls_load": True}, False
    )
    assert fused_moe_impl._bottom_a_uses_sorted_layout(
        {"USE_MLS_LOAD": True, "bottom_a_use_mls_load": True}, False
    )


def test_checked_in_bottom_configs_have_explicit_a_side_mls():
    config_root = (
        Path(__file__).resolve().parents[2]
        / "boltops"
        / "fused_moe"
        / "triton"
        / "configs"
    )
    for version in ("triton_3_2", "triton_3_6"):
        paths = sorted((config_root / version).glob("*is_bottom=True*.json"))
        assert paths
        for path in paths:
            if path.stat().st_size == 0:
                continue
            entries = json.loads(path.read_text(encoding="utf-8"))
            for m, config in entries.items():
                assert isinstance(config.get("bottom_a_use_mls_load"), bool), (
                    f"{path.name}: M={m} lacks explicit A-side MLS state"
                )


def test_checked_in_triton_3_6_configs_are_sorted_by_batch_size():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "boltops"
        / "fused_moe"
        / "triton"
        / "configs"
        / "triton_3_6"
    )
    paths = sorted(config_dir.glob("*.json"))
    assert paths
    for path in paths:
        if path.stat().st_size == 0:
            continue
        keys = list(json.loads(path.read_text(encoding="utf-8")))
        assert keys == sorted(keys, key=int), f"{path.name} is not numerically sorted"


def test_checked_in_moe_configs_have_explicit_shape_selectors():
    config_root = (
        Path(__file__).resolve().parents[2]
        / "boltops"
        / "fused_moe"
        / "triton"
        / "configs"
    )

    for version in ("triton_3_2", "triton_3_6"):
        paths = sorted((config_root / version).glob("*.json"))
        assert paths
        selector_kinds = set()
        top_families = set()
        bottom_families = set()

        for path in paths:
            match = re.match(r"^(E=\d+,N=\d+),(.*)(,arch=.*\.json)$", path.name)
            assert match, f"unexpected MoE config name: {path.name}"
            selector = moe_config_utils._parse_config_selector(match.group(2))
            assert selector is not None, f"invalid shape selector: {path.name}"
            assert "KMax=" not in path.name

            kind = (
                "exact"
                if selector["K"][0] == selector["topk"][0] == "exact"
                else "fuzzy"
            )
            selector_kinds.add(kind)
            if kind == "fuzzy":
                assert selector["K"][0] == "lower_mod"
            family = match.group(1) + match.group(3)
            normalized_family = family.replace(",is_bottom=True", "")
            if ",is_bottom=True" in family:
                bottom_families.add(normalized_family)
            else:
                top_families.add(normalized_family)

        assert top_families == bottom_families
        if version == "triton_3_2":
            assert selector_kinds == {"fuzzy"}
        # A 3.6 family may provide exact configs, validated fuzzy configs, or
        # both. Do not require an unvalidated fuzzy range for an exact config.


def test_autotune_skips_unloadable_hip_candidate():
    def broken_bench(_kernel_call, _quantiles):
        raise RuntimeError("Triton Error [HIP]: Code: 500, named symbol not found")

    robust_bench = autotune_patches._skip_invalid_hip_loads(broken_bench)

    with pytest.raises(HSACOError, match="named symbol not found"):
        robust_bench(lambda: None, (0.5, 0.2, 0.8))


def test_autotune_does_not_hide_other_runtime_errors():
    def broken_bench(_kernel_call, _quantiles):
        raise RuntimeError("device-side assert triggered")

    robust_bench = autotune_patches._skip_invalid_hip_loads(broken_bench)

    with pytest.raises(RuntimeError, match="device-side assert"):
        robust_bench(lambda: None, (0.5, 0.2, 0.8))


def test_fp8_channel_autotune_prunes_unsafe_small_configs(monkeypatch):
    class _KpackCapability:
        @staticmethod
        def supports(name):
            return name == "kpack"

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _KpackCapability()
    )
    monkeypatch.setattr(autotune_patches, "test_type", "fp8_channel")
    monkeypatch.setattr(autotune_patches, "capMLS", False)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n1", 32)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 64)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k1", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 64, raising=False
    )

    def configs(num_input_tokens):
        configs = autotune_patches.generate_config2_lists(
            BLOCK_SIZE_M=16,
            N=320,
            K=7168,
            MUL_ROUTED_WEIGHT=False,
            per_channel_quant=True,
            num_valid_tokens=num_input_tokens * 8,
            top_k=8,
            num_input_tokens=num_input_tokens,
        )
        assert configs
        return configs

    small = configs(4)
    assert any(c.kwargs.get("kpack") == 2 for c in small)

    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 256)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 256, raising=False
    )
    small = configs(4)
    assert not any(
        c.kwargs["BLOCK_SIZE_N"] >= 256
        and c.kwargs["BLOCK_SIZE_K"] >= 256
        for c in small
    )
    assert not any(
        c.kwargs["BLOCK_SIZE_N"] >= 256 and c.num_warps >= 8
        for c in small
    )
    assert any(
        c.kwargs["BLOCK_SIZE_N"] >= 256
        and c.kwargs["BLOCK_SIZE_K"] < 256
        and c.num_warps <= 4
        for c in small
    )


def test_fp8_channel_autotune_avoids_wide_decode_mls(monkeypatch):
    class _KpackCapability:
        @staticmethod
        def supports(name):
            return name == "kpack"

    monkeypatch.setattr(
        autotune_patches, "triton_capabilities", _KpackCapability()
    )
    monkeypatch.setattr(autotune_patches, "test_type", "fp8_channel")
    monkeypatch.setattr(autotune_patches, "capMLS", True)
    monkeypatch.setattr(autotune_patches, "compile_only", 0)
    monkeypatch.setattr(autotune_patches, "mode_autotune_min_block_size_n1", 16)
    monkeypatch.setattr(autotune_patches, "mode_autotune_max_block_size_n1", 64)
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_min_block_size_k1", 64, raising=False
    )
    monkeypatch.setattr(
        autotune_patches, "mode_autotune_max_block_size_k1", 64, raising=False
    )

    def configs(n, num_input_tokens):
        return autotune_patches.generate_config2_lists(
            BLOCK_SIZE_M=16,
            N=n,
            K=7168,
            MUL_ROUTED_WEIGHT=False,
            per_channel_quant=True,
            num_valid_tokens=num_input_tokens * 6,
            top_k=6,
            num_input_tokens=num_input_tokens,
        )

    assert not any(
        config.kwargs["USE_MLS_LOAD"] for config in configs(1536, 4)
    )
    assert any(config.kwargs["USE_MLS_LOAD"] for config in configs(1536, 128))
    assert any(config.kwargs["USE_MLS_LOAD"] for config in configs(768, 4))


def test_sorted_bridge_does_not_force_mls(monkeypatch):
    monkeypatch.setattr(fused_moe_impl, "_bottom_moe_use_mls", lambda: True)
    config = {"a_sorted": True, "bottom_a_use_mls_load": False, "N_TILES_PER_CTA": 2}
    assert fused_moe_impl._bottom_a_uses_sorted_layout(config, False)
    assert not fused_moe_impl._bottom_a_uses_mls(config, False)
    assert not fused_moe_impl._bottom_a_uses_sorted_layout(config, True)
