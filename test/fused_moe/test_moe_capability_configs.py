import json
from pathlib import Path

import pytest

from boltops.fused_moe.triton import moe_config_utils
from boltops.tools.fused_moe_triton_tune import moe_log_parser
from boltops.tools.fused_moe_triton_tune import autotune_patches
from boltops.utility.triton_capability import TritonCapabilities


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


def _autotune_log(
    use_block_pingpong: str | None,
    buffer_cache_swizzle: str | None = None,
) -> str:
    pingpong = (
        "" if use_block_pingpong is None else f", use_block_pingpong: {use_block_pingpong}"
    )
    cache_swizzle = (
        ""
        if buffer_cache_swizzle is None
        else f", buffer_cache_swizzle: {buffer_cache_swizzle}"
    )
    kernel = (
        "fused_moe_kernel:\n"
        "('key',) Configs:\n"
        "* 0  : BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 1, "
        "COMBINE_SCALE_LOAD: False, USE_MLS_LOAD: False, waves_per_eu: 1, "
        "instruction_sched_variant: none, sched_latency: none, kpack: 1"
        f"{pingpong}{cache_swizzle}, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None\n"
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


def test_log_parser_uses_logged_batch_key_after_missing_result(tmp_path):
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
    results = moe_log_parser.parse_log_file(missing + valid_bf16 + valid_fp8)
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
