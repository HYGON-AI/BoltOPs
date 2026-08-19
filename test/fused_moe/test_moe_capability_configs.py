import json
from pathlib import Path

import pytest

from boltops.fused_moe.triton import moe_config_utils
from boltops.tools.fused_moe_triton_tune import moe_log_parser
from boltops.utility.triton_capability import TritonCapabilities


def _caps(version: str, *options: str) -> TritonCapabilities:
    return TritonCapabilities(version, frozenset(options), "hcu", version)


@pytest.fixture(autouse=True)
def _clear_moe_config_cache():
    moe_config_utils.get_moe_configs.cache_clear()
    yield
    moe_config_utils.get_moe_configs.cache_clear()


def _write_config(config_root: Path, version: str, file_name: str, marker: str):
    config_dir = config_root / "moe" / f"triton_{version.replace('.', '_')}"
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


def _autotune_log(use_block_pingpong: str | None) -> str:
    pingpong = (
        "" if use_block_pingpong is None else f", use_block_pingpong: {use_block_pingpong}"
    )
    kernel = (
        "fused_moe_kernel:\n"
        "('key',) Configs:\n"
        "* 0  : BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 1, "
        "COMBINE_SCALE_LOAD: False, USE_MLS_LOAD: False, waves_per_eu: 1, "
        "instruction_sched_variant: none, sched_latency: none, kpack: 1"
        f"{pingpong}, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None\n"
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


def test_tuner_copies_results_to_current_capability_directory(monkeypatch):
    monkeypatch.setattr(
        moe_log_parser, "get_triton_capabilities", lambda: _caps("3.6")
    )

    assert moe_log_parser.resolve_boltops_moe_config_dir().name == "triton_3_6"
