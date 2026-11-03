# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest
import json
import os
from types import SimpleNamespace

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
        "python3 -m boltops.tools.fused_moe_triton_tune.performance_csv --test-type fp8",
        "/usr/bin/python3 -u -m boltops.tools.fused_moe_triton_tune.performance_csv --test-type fp8 --_single-batch --batch-sizes 8192",
        "python3 -m boltops.tools.fused_moe_triton_tune.performance_csv --config exact.json",
        "python3 /zhenxin/BoltOPs/boltops/tools/fused_moe_triton_tune/performance_csv.py --test-type fp8",
    ],
)
def test_matches_cli_entry_points(command: str) -> None:
    assert is_moe_process_command(command)


@pytest.mark.parametrize("module", ["tune_moe_cli", "performance_csv"])
def test_does_not_match_parent_shell_containing_module_command(module) -> None:
    command = (
        "bash -lc python3 -m "
        f"boltops.tools.fused_moe_triton_tune.{module} nmz bf16"
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


@pytest.mark.parametrize("kind,count", [("fp8_channel", 10), ("fp8", 10), ("bf16", 10), ("int4", 5),
                                      ("mxfp4", 4), ("mxfp8", 4)])
def test_outer_config_shards_cover_original_space(kind, count):
    from boltops.tools.fused_moe_triton_tune.common import autotune_outer_configs
    configs = autotune_outer_configs(kind)
    assert len(configs) == count
    for config in configs:
        assert autotune_outer_configs(kind, json.dumps(config)) == [config]


@pytest.mark.parametrize("selection", [
    '{"BLOCK_SIZE_M": 512, "bottom_a_use_mls_load": false}',
    '{"BLOCK_SIZE_M": 64, "bottom_a_use_mls_load": 1}',
    '{"BLOCK_SIZE_M": 64}', '[]',
])
def test_outer_config_rejects_invalid_selection(selection):
    from boltops.tools.fused_moe_triton_tune.common import autotune_outer_configs
    with pytest.raises(ValueError):
        autotune_outer_configs("fp8", selection)


@pytest.mark.parametrize("kind,script", [("fp8", "tune_moe_fp8.py"),
                                       ("mxfp4", "tune_moe_bf16.py"),
                                       ("mxfp8", "tune_moe_bf16.py")])
def test_isolated_group_keeps_gpu_and_all_outer_configs(monkeypatch, tmp_path, kind, script):
    from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli
    from boltops.tools.fused_moe_triton_tune.common import autotune_outer_configs
    monkeypatch.setenv("AUTOTUNE_TEST_TYPE", kind)
    for key, value in dict(MOE_BATCH_SIZES="1,4096", AUTOTUNE_DEVICE_START_ID="3",
                           AUTOTUNE_DEVICE_COUNT="2", AUTOTUNE_NUM_GROUPS_RUN="2", CUR_GROUP="1").items():
        monkeypatch.setenv(key, value)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs['env'].copy()))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.run_isolated_group(tmp_path, script) == 0
    assert len(calls) == len(autotune_outer_configs(kind))
    assert [json.loads(env['AUTOTUNE_OUTER_CONFIG']) for _, env in calls] == list(reversed(autotune_outer_configs(kind)))
    for command, env in calls:
        assert command[0] == cli.sys.executable and command[-1] == script
        assert env['MOE_BATCH_SIZES'] == "4096"
        assert env['AUTOTUNE_DEVICE_START_ID'] == "4"
        assert env['AUTOTUNE_DEVICE_COUNT'] == env['AUTOTUNE_NUM_GROUPS_RUN'] == "1"
        assert env['CUR_GROUP'] == "0"
    assert os.environ['CUR_GROUP'] == "1"


def test_isolated_group_stops_on_first_failure(monkeypatch, tmp_path):
    from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli
    for key, value in dict(MOE_BATCH_SIZES="1,4096", AUTOTUNE_DEVICE_START_ID="0",
                           AUTOTUNE_DEVICE_COUNT="1", AUTOTUNE_NUM_GROUPS_RUN="1", CUR_GROUP="0").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('AUTOTUNE_OUTER_CONFIG', 'stale selector')
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs['env'].copy())
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.run_isolated_group(tmp_path, "tune_moe_fp8.py") == 7
    assert len(calls) == 1 and 'AUTOTUNE_OUTER_CONFIG' not in calls[0]


def test_explicit_run_group_count(monkeypatch):
    from boltops.tools.fused_moe_triton_tune.common import resolve_device_settings
    monkeypatch.setenv('AUTOTUNE_DEVICE_START_ID', '2')
    monkeypatch.setenv('AUTOTUNE_DEVICE_COUNT', '4')
    monkeypatch.setenv('AUTOTUNE_NUM_GROUPS_RUN', '1')
    assert resolve_device_settings(8) == (2, 4, 1)
    monkeypatch.setenv('AUTOTUNE_NUM_GROUPS_RUN', '0')
    with pytest.raises(ValueError, match='NUM_GROUPS_RUN'):
        resolve_device_settings(8)
    monkeypatch.setenv('AUTOTUNE_NUM_GROUPS_RUN', '5')
    with pytest.raises(ValueError, match='NUM_GROUPS_RUN'):
        resolve_device_settings(8)


@pytest.mark.parametrize('count,batches', [(3, [1,768,4096,16384]), (4, [1,1024,10000,16384,32768])])
def test_run_batch_partition_does_not_drop_nonstandard_m(count, batches):
    from boltops.tools.fused_moe_triton_tune.common import split_run_batches
    groups = split_run_batches(batches, count)
    assert len(groups) == count and all(groups)
    assert sorted(m for group in groups for m in group) == batches


@pytest.mark.parametrize('batch_key', ['m', 'BATCH_SIZE'])
def test_log_parser_selects_complete_pair_across_processes(tmp_path, batch_key):
    from boltops.tools.fused_moe_triton_tune import moe_log_parser as parser

    def kernel(time):
        return ("fused_moe_kernel:\n* 0 : BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 64, "
                "GROUP_SIZE_M: 1, num_warps: 8, num_stages: 1\n"
                f" : times(3 statics, rep:20 ms): {time} ms\n")

    text = ''.join(f"Test: {batch_key}=8192,\nConfig 1: {{'BLOCK_SIZE_M': {bm}}}\n" + kernel(t1) + kernel(t2)
                   for bm, t1, t2 in [(64, .5, 2), (128, 1, 1.1), (256, 5, 5)])
    results = parser.parse_log_file(text)
    assert len(results) == 1 and results[0][4] == pytest.approx(2.1)
    parser.validate_expected_batches(results, '8192')
    with pytest.raises(ValueError, match='Incomplete autotune'):
        parser.validate_expected_batches(results, '8192,16384')
    top, bottom = tmp_path/'top.json', tmp_path/'bottom.json'
    parser.save_results_to_json(results, top, bottom)
    assert json.loads(top.read_text())['8192']['BLOCK_SIZE_M'] == 128
    assert json.loads(bottom.read_text())['8192']['BLOCK_SIZE_M'] == 128


@pytest.mark.parametrize('compile_only', [False, True])
def test_group_launch_uses_worker_only_for_training(monkeypatch, tmp_path, compile_only):
    from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli
    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs['env'].copy()))
        return SimpleNamespace(pid=100, wait=lambda: 0)

    monkeypatch.setattr(cli.subprocess, 'Popen', start)
    assert cli.run_parallel_pytest_groups(script_dir=tmp_path, test_script_name='tune_moe_fp8.py',
        env_base={}, group_ids=[0,1], log_dir=tmp_path, log_name_builder=lambda g: f'g{g}.log',
        triton_compile_only=compile_only, break_on_error=True, startup_sleep_sec=0)
    assert [env['CUR_GROUP'] for _, env in calls] == ['0','1']
    assert all(('_run-group' in command) == (not compile_only) for command, _ in calls)


@pytest.mark.parametrize('failure', ['missing_batch', 'publish_error'])
def test_parser_does_not_report_success_for_incomplete_or_unpublished_config(monkeypatch, tmp_path, failure):
    from boltops.tools.fused_moe_triton_tune import moe_log_parser as parser
    log, top, bottom = tmp_path/'input.log', tmp_path/'top.json', tmp_path/'bottom.json'
    log.write_text('test log')
    cfg = dict(BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=8, num_stages=1)
    result = (1, "{'BLOCK_SIZE_M': 64}", 0, 0, 2.0, cfg, cfg, 8192)
    monkeypatch.setattr(parser, 'parse_log_file', lambda content: [result])
    monkeypatch.setattr(parser.sys, 'argv', ['parser', str(log), str(top), str(bottom)])
    monkeypatch.setenv('AUTOTUNE_EXPECTED_BATCHES', '8192,16384' if failure == 'missing_batch' else '8192')
    target = tmp_path/'config_dir'
    target.write_text('not a directory')
    monkeypatch.setattr(parser, 'resolve_boltops_moe_config_dir', lambda: target)
    with pytest.raises(SystemExit) as error:
        parser.main()
    assert error.value.code == 1
    if failure == 'missing_batch':
        assert not top.exists() and not bottom.exists()
