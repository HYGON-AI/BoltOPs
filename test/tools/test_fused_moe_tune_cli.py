# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli

from boltops.tools.fused_moe_triton_tune.tune_moe_cli import (
    is_moe_process_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "python3 -m boltops.tools.fused_moe_triton_tune.tune_moe_cli nmz bf16",
        "/usr/bin/python3 -u -m boltops.tools.fused_moe_triton_tune.tune_moe_cli bmz int8",
        "python3 /zhenxin/BoltOPs/boltops/tools/fused_moe_triton_tune/tune_moe_cli.py nmz fp8",
        "boltops-fused-moe-tune-cli nmz fp8_channel",
        "/usr/bin/python3 /usr/local/bin/boltops-fused-moe-tune-cli bmz bf16",
    ],
)
def test_matches_cli_entry_points(command: str) -> None:
    assert is_moe_process_command(command)


def test_does_not_match_parent_shell_containing_module_command() -> None:
    command = (
        "bash -lc python3 -m "
        "boltops.tools.fused_moe_triton_tune.tune_moe_cli nmz bf16"
    )
    assert not is_moe_process_command(command)


def test_does_not_match_unrelated_python_process() -> None:
    assert not is_moe_process_command("python3 -m pytest test/test_package_smoke.py")


@pytest.mark.parametrize("n,k,valid", [(32, 160, False), (160, 32, False),
                                      (128, 128, True), (16, 128, False),
                                      (128, 16, False)])
def test_mxfp4_cli_requires_dynamic_g128_dimensions(monkeypatch, tmp_path, n, k, valid):
    calls = []
    monkeypatch.setattr(cli.sys, "argv", [
        "boltops-fused-moe-tune-cli", "sb", "mxfp4", "--tp-list", "1",
        "--hidden-size-default", str(n), "--ffn-hidden-size", str(k),
        "--benchmark", "--device-count", "1", "--disable-auto-select-devices-by-hy-smi",
    ])
    monkeypatch.setattr(cli, "configure_output_streams", lambda: None)
    monkeypatch.setattr(cli, "resolve_launcher_env", lambda *args: {})
    monkeypatch.setattr(cli, "check_patch_contract", lambda **kwargs: (True, ""))
    script_dir = Path(cli.__file__).parent
    monkeypatch.setattr(cli, "resolve_runtime_dirs", lambda: (script_dir, tmp_path, "source"))

    def benchmark(**kwargs):
        assert kwargs["tp_env"]["MOE_HIDDEN_SIZE"] == str(n)
        assert kwargs["tp_env"]["MOE_FFN_HIDDEN_SIZE"] == str(k)
        calls.append(True)
        return 0
    monkeypatch.setattr(cli, "run_benchmark_step", benchmark)
    assert cli.main() == (0 if valid else 1)
    assert bool(calls) == valid


@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_native_mx_benchmark_export_keeps_quantization_type(monkeypatch, tmp_path, kind):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return 0
    monkeypatch.setattr(cli, "run_cmd", run)
    assert cli.run_benchmark_step(
        script_dir=tmp_path, test_script_name="tune_moe_bf16.py",
        tp_env={"AUTOTUNE_TEST_TYPE": kind}, tp=1, ep_size=1,
        break_on_error=True,
    ) == 0
    export = commands[-1]
    assert export[export.index("--test-type") + 1] == kind
