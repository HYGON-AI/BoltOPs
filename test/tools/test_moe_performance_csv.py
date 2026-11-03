# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from boltops.tools.fused_moe_triton_tune.performance_csv import (
    FIELDS, config_pair, make_row, parse_config, unsupported_batch_sizes, upsert_row,
)


def _shape(directory):
    return parse_config(Path(directory) / "E=256,N=128,K=7168,topk=8,arch=gfx938,dtype=int8_w8a8.json")


def _write(directory, m):
    row = make_row(_shape(directory), m, "torch.bfloat16", 40 + m)
    return str(upsert_row(directory, "int8_channel", row))


def test_csv_schema_upsert(tmp_path):
    path = Path(_write(tmp_path, 1))
    _write(tmp_path, 1)
    _write(tmp_path, 8)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["inter_dim"] == "128"
    assert rows[0]["model_dim"] == "7168"
    assert rows[0]["sol_type"] == "triton"
    assert rows[0]["quant_type"] == "int8_w8a8_channel"
    assert b"\r" not in path.read_bytes()
    assert not (path.parent / "performance_metadata").exists()
    assert not list(tmp_path.glob("*.json"))


def test_concurrent_writers_preserve_rows(tmp_path):
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_write, [tmp_path] * 12, range(1, 13)))
    path = tmp_path / "tuned_fmoe_triton_w8a8_channel.csv"
    with path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 12


def test_writer_lock_is_bounded_and_not_stolen(tmp_path):
    from boltops.tools.fused_moe_triton_tune.performance_csv import csv_write_lock
    path = tmp_path / 'test.csv'
    with csv_write_lock(path):
        with pytest.raises(TimeoutError):
            with csv_write_lock(path, timeout=0):
                pytest.fail('Lock was stolen')
    assert not path.with_suffix('.writer-lock').exists()


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_bad_times_not_published(tmp_path, latency):
    row = make_row(_shape(tmp_path), 1, "torch.float16", latency)
    with pytest.raises(ValueError):
        upsert_row(tmp_path, "int8_channel", row)
    assert not list(tmp_path.glob("*.csv"))


def test_dtypes_have_independent_rows(tmp_path):
    _write(tmp_path, 1)
    row = make_row(_shape(tmp_path), 1, "torch.float16", 33)
    path = upsert_row(tmp_path, "int8_channel", row)
    with path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_implementations_have_independent_rows(tmp_path):
    path = Path(_write(tmp_path, 1))
    row = make_row(_shape(tmp_path), 1, "torch.bfloat16", 21)
    row["sol_id"] = "triton:fused_moe_alternative"
    upsert_row(tmp_path, "int8_channel", row)
    with path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_backend_prefix_must_agree(tmp_path):
    row = make_row(_shape(tmp_path), 1, "torch.bfloat16", 21)
    row["sol_id"] = "tilelang:fused_moe"
    with pytest.raises(ValueError, match="prefix"):
        upsert_row(tmp_path, "int8_channel", row)


@pytest.mark.parametrize("suffix,test_type,block", [
    ("", "bf16", (0, 0)),
    (",dtype=fp8_w8a8", "fp8_channel", (0, 0)),
    (",dtype=int8_w8a8,block_shape=[128,128]", "int8", (128, 128)),
    (",dtype=int4_w4a16", "int4", (0, 64)),
])
def test_config_parsing(tmp_path, suffix, test_type, block):
    path = tmp_path / f"E=256,N=128,K=7168,topk=8,arch=gfx938{suffix}.json"
    shape = parse_config(path)
    assert shape["test_type"] == test_type
    assert (shape["block_n"], shape["block_k"]) == block
    assert ",is_bottom=True" in shape["bottom"].name
    if "block_shape" in suffix:
        assert ",is_bottom=True,block_shape=" in shape["bottom"].name


@pytest.mark.parametrize("selector", ["KMin=4096,KMod=1024,topk=8", "K=4096,topkSet=1-8"])
def test_fuzzy_is_not_a_measurement(tmp_path, selector):
    with pytest.raises(ValueError):
        parse_config(tmp_path / f"E=256,N=128,{selector},arch=gfx938.json")


def test_pair_requires_shared_bm(tmp_path):
    shape = _shape(tmp_path)
    shape["top"].write_text(json.dumps({"1": {"BLOCK_SIZE_M": 16}}))
    shape["bottom"].write_text(json.dumps({"1": {"BLOCK_SIZE_M": 32}}))
    with pytest.raises(ValueError, match="BLOCK_SIZE_M"):
        config_pair(shape)


def test_unsupported_dtype_is_per_m_not_entire_config(tmp_path):
    shape = parse_config(tmp_path / "E=256,N=128,K=3072,topk=8,arch=gfx938.json")
    top = {"1": {}, "2": {}, "4": {}}
    bottom = {"1": {}, "2": {"N_TILES_PER_CTA": 2}, "4": {}}
    assert unsupported_batch_sizes(shape, "float16", top, bottom) == [2]
    assert unsupported_batch_sizes(shape, "torch.bfloat16", top, bottom) == []
    shape["test_type"] = "int8_channel"
    assert unsupported_batch_sizes(shape, "float16", top, bottom) == []


def test_memory_efficient_weights_preserve_values_and_rng():
    import torch
    from boltops.tools.fused_moe_triton_tune.common import random_int8_weights

    # CPU regression test; the same comparison is also exercised on the GPU
    # before using the generator to backfill large-E measurements.
    shapes = ((7, 512, 256), (7, 256, 256))
    torch.manual_seed(123)
    a = (torch.rand(shapes[0], dtype=torch.float32) - 0.5) * 2
    b = (torch.rand(shapes[1], dtype=torch.float32) - 0.5) * 2
    expected = [(value * 127).clamp(-128, 127).to(torch.int8) for value in (a, b)]
    expected_state = torch.get_rng_state()
    torch.manual_seed(123)
    actual = [random_int8_weights(shape, "cpu") for shape in shapes]
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_stability_rejects_foreign_queue():
    from boltops.tools.fused_moe_triton_tune.performance_csv import DeviceGuard
    guard = object.__new__(DeviceGuard)
    guard.processes = lambda: {101, 102}
    with pytest.raises(RuntimeError, match="not exclusive"):
        guard.__enter__()


def test_stability_spread():
    from boltops.tools.fused_moe_triton_tune.performance_csv import spread_fraction
    assert spread_fraction([40, 40.1, 40.2, 40.3, 40.4]) < 0.05
    assert spread_fraction([40, 40, 40, 40, 48]) > 0.05


def test_busy_gpu_wait_is_bounded():
    from boltops.tools.fused_moe_triton_tune.performance_csv import DeviceGuard
    guard = object.__new__(DeviceGuard)
    guard.processes = lambda: {101, 102}
    with pytest.raises(RuntimeError, match="remained occupied"):
        guard.wait_until_exclusive(timeout=0)


def test_foreign_queue_during_measurement_invalidates_result():
    import threading
    from boltops.tools.fused_moe_triton_tune.performance_csv import DeviceGuard
    guard = object.__new__(DeviceGuard)
    guard.stop = threading.Event()
    guard.processes = lambda: {101}
    with guard:
        guard.processes = lambda: {101, 102}
    assert guard.contaminated


def _device_guard_with_kfd(monkeypatch, tmp_path, nodes):
    """Build DeviceGuard against a synthetic KFD topology.

    ``nodes`` maps node name to ``(properties, gpu_id)``; a ``None``
    ``properties`` makes the node unreadable to the process.
    """
    import ctypes
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter

    root = tmp_path / "kfd" / "topology" / "nodes"
    for name, (properties, gpu_id) in nodes.items():
        node = root / name
        node.mkdir(parents=True)
        if properties is None:
            (node / "properties").mkdir()
        else:
            (node / "properties").write_text(properties)
        if gpu_id is not None:
            (node / "gpu_id").write_text(gpu_id)

    def get_bus(buf, _size, _device):
        ctypes.memmove(buf, b"0000:07:00.0\0", 13)
        return 0

    get_bus.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    get_bus.restype = ctypes.c_int

    class _Hip:
        @property
        def hipDeviceGetPCIBusId(self):
            return get_bus

    real_path = exporter.Path

    def fake_path(*args, **kwargs):
        target = real_path(*args, **kwargs)
        return root if str(target) == "/sys/class/kfd/kfd/topology/nodes" else target

    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libamdhip64.so")
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: _Hip())
    monkeypatch.setattr(exporter, "Path", fake_path)
    return exporter


def test_topology_skips_unreadable_nodes(monkeypatch, tmp_path):
    exporter = _device_guard_with_kfd(monkeypatch, tmp_path, {
        "1": (None, None),
        "3": ("location_id 1792\ndomain 0\ndev 3\n", "42\n"),
    })
    assert exporter.DeviceGuard(0).gpu_id == 42


def test_topology_still_requires_test_gpu_identified(monkeypatch, tmp_path):
    exporter = _device_guard_with_kfd(monkeypatch, tmp_path, {
        "1": ("location_id 999\ndomain 0\ndev 1\n", "7\n"),
        "2": (None, None),
    })
    with pytest.raises(RuntimeError, match="Cannot map HIP device to KFD topology"):
        exporter.DeviceGuard(0)


def test_cli_only_exports_after_success(monkeypatch, tmp_path):
    from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return 0

    monkeypatch.setattr(cli, "run_cmd", run)
    args = dict(script_dir=tmp_path, test_script_name="tune_moe_bf16.py",
                tp_env={}, tp=1, ep_size=1, break_on_error=True)
    assert cli.run_benchmark_step(**args) == 0
    assert commands[-1][-1] == "bf16"
    assert "boltops.tools.fused_moe_triton_tune.performance_csv" in commands[-1]
    commands.clear()
    monkeypatch.setattr(cli, "run_cmd", lambda cmd, **kwargs: commands.append(cmd) or 1)
    assert cli.run_benchmark_step(**args) == 1
    assert len(commands) == 1


def test_environment_is_normalized_before_config_resolution(monkeypatch):
    import os
    from types import SimpleNamespace
    import torch
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter

    # Both config-resolution imports and callbacks must see the selected GPU
    # and synthetic routing, not the inherited training environment.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.update(TRAINING_MODE="1", BENCHMARK_MODE="1", TRITON_COMPILE_ONLY="1",
                      AUTOTUNE_DEVICE_START_ID="0", AUTOTUNE_DEVICE_COUNT="8",
                      CUR_GROUP="7", TOPK_IDS_DIR="/unused/custom-routing")
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: SimpleNamespace(zero_=lambda: None))
    monkeypatch.setattr(exporter, "DeviceGuard", lambda device: None)

    class Resolved(Exception):
        pass

    def resolve(test_type):
        assert test_type == "bf16"
        assert selected == [3]
        assert os.environ["AUTOTUNE_DEVICE_START_ID"] == "3"
        assert os.environ["AUTOTUNE_DEVICE_COUNT"] == "1"
        assert os.environ["CUR_GROUP"] == "0"
        for key in ("TRAINING_MODE", "BENCHMARK_MODE", "TRITON_COMPILE_ONLY"):
            assert os.environ[key] == "0"
        assert "TOPK_IDS_DIR" not in os.environ
        raise Resolved()

    monkeypatch.setattr(exporter, "_from_environment", resolve)
    with pytest.raises(Resolved):
        exporter.benchmark(SimpleNamespace(device=3, config=None, test_type="bf16"))


def test_remeasurement_replaces_existing_time(tmp_path):
    _write(tmp_path, 1)
    row = make_row(_shape(tmp_path), 1, "torch.bfloat16", 35)
    path = upsert_row(tmp_path, "int8_channel", row)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["time_us"]) == 35
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize('kind,recipe', [
    ('mxfp4', 'deepgemm-a128-b1x32'),
    ('mxfp8', 'vllm-a32-b1x32-e8m0-e8m0'),
])
def test_native_mx_config_and_csv_keep_logical_shapes(tmp_path, kind, recipe):
    dtype = 'mxfp4_w4a8' if kind == 'mxfp4' else 'mxfp8_w8a8'
    suffix = f'dtype={dtype},gemm_type=dot_scaled,scale_recipe={recipe}'
    shape = parse_config(tmp_path / f'E=80,N=512,K=2048,topk=8,arch=gfx946,{suffix}.json')
    assert shape['test_type'] == kind
    assert shape['gemm_type'] == 'dot_scaled'
    assert shape['scale_recipe'] == recipe
    assert shape['bottom'].name.endswith(suffix + ',is_bottom=True.json')
    assert (shape['block_n'], shape['block_k']) == (0, 0)
    row = make_row(shape, 12, 'torch.bfloat16', 10.)
    assert (row['inter_dim'], row['model_dim']) == (512, 2048)
    assert (row['q_size_n'], row['q_size_k']) == (1, 32)
    path = upsert_row(tmp_path, kind, row)
    assert path.name == f'tuned_fmoe_triton_{dtype}.csv'


@pytest.mark.parametrize('suffix', [
    'dtype=mxfp4_w4a8',
    'dtype=mxfp8_w8a8,gemm_type=dot_scaled,scale_recipe=deepgemm-a128-b1x32',
    'dtype=fp8_w8a8,gemm_type=dot_scaled',
])
def test_csv_rejects_incompatible_scale_recipe(tmp_path, suffix):
    with pytest.raises(ValueError):
        parse_config(tmp_path / f'E=80,N=512,K=2048,topk=8,arch=gfx946,{suffix}.json')


@pytest.mark.parametrize('kind,recipe', [
    ('mxfp4', 'deepgemm-a128-b1x32'),
    ('mxfp8', 'vllm-a32-b1x32-e8m0-e8m0'),
])
def test_native_mx_environment_resolves_saved_recipe(monkeypatch, kind, recipe):
    from types import SimpleNamespace
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    from boltops.utility import arch_info

    monkeypatch.setenv('MOE_NUM_EXPERTS', '80')
    monkeypatch.setenv('MOE_HIDDEN_SIZE', '512')
    monkeypatch.setenv('AUTOTUNE_TEST_TYPE', kind)
    monkeypatch.setattr(arch_info, 'get_arch', lambda: 'gfx946')
    def load(name):
        assert name.endswith('.tune_moe_bf16')
        return SimpleNamespace(FFN_HIDDEN_SIZES=[2048], TOP_KS=[8])
    monkeypatch.setattr(exporter.importlib, 'import_module', load)
    shape = parse_config(exporter._from_environment(kind))
    assert shape['test_type'] == kind
    assert shape['scale_recipe'] == recipe
    assert (shape['e'], shape['n'], shape['k'], shape['topk']) == (80, 512, 2048, 8)


def test_fresh_csv_process_timeout_retries_only_owned_leaf(monkeypatch):
    from types import SimpleNamespace
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    calls, killed = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(calls)
            calls.append((command, kwargs))

        def wait(self, timeout=None):
            if self.pid == 100 and timeout is not None:
                raise exporter.subprocess.TimeoutExpired(calls[0][0], timeout)
            return 0

    monkeypatch.setattr(exporter.subprocess, 'Popen', Child)
    monkeypatch.setattr(exporter.os, 'killpg', lambda *args: killed.append(args))
    args = SimpleNamespace(batch_sizes='8,1,8', max_attempts=2, process_timeout=5)
    exporter.benchmark_in_fresh_processes(args, ['--config', 'exact.json'])
    assert [command[-1] for command, _ in calls] == ['1', '1', '8']
    assert all(kwargs['start_new_session'] for _, kwargs in calls)
    assert all('--_single-batch' in command for command, _ in calls)
    assert killed == [(100, exporter.signal.SIGKILL)]


def test_cli_checks_every_m_in_separate_process_and_keeps_failures(monkeypatch, tmp_path):
    from boltops.tools.fused_moe_triton_tune import tune_moe_cli as cli
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs['env'].copy()))
        return 1 if kwargs['env'].get('MOE_BATCH_SIZES') == '1' else 0

    monkeypatch.setattr(cli, 'run_cmd', run)
    assert cli.run_benchmark_step(script_dir=tmp_path, test_script_name='tune_moe_fp8.py',
           tp_env={'MOE_BATCH_SIZES': '1,8192'}, tp=1, ep_size=1, break_on_error=False) == 1
    assert [env['MOE_BATCH_SIZES'] for _, env in calls] == ['1', '8192']
    assert all('pytest' in command for command, _ in calls)  # No CSV after failure.


def test_csv_does_not_retry_a_failed_correctness_or_config_check(monkeypatch):
    from types import SimpleNamespace
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    calls = []

    def start(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(wait=lambda **kwargs: 3)

    monkeypatch.setattr(exporter.subprocess, 'Popen', start)
    args = SimpleNamespace(batch_sizes='1,8', max_attempts=3, process_timeout=5)
    with pytest.raises(exporter.subprocess.CalledProcessError):
        exporter.benchmark_in_fresh_processes(args, ['--config', 'exact.json'])
    assert len(calls) == 1


@pytest.mark.parametrize('ignore_term', [False, True])
def test_csv_cancellation_cleans_up_only_owned_process_group(monkeypatch, ignore_term):
    from types import SimpleNamespace
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    killed = []

    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            raise KeyboardInterrupt
        if ignore_term and len(waits) == 2:
            raise exporter.subprocess.TimeoutExpired('csv child', timeout)
        return 0

    monkeypatch.setattr(exporter.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=123, wait=wait))
    monkeypatch.setattr(exporter.os, 'killpg', lambda *args: killed.append(args))
    args = SimpleNamespace(batch_sizes='1', max_attempts=1, process_timeout=5)
    with pytest.raises(KeyboardInterrupt):
        exporter.benchmark_in_fresh_processes(args, ['--config', 'exact.json'])
    expected = [(123, exporter.signal.SIGTERM)]
    if ignore_term:
        expected.append((123, exporter.signal.SIGKILL))
    assert killed == expected
    assert waits == ([5, 5, None] if ignore_term else [5, 5])


@pytest.mark.parametrize('groups', [1, 2, 4])
def test_csv_resets_inherited_training_groups(monkeypatch, groups):
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    from boltops.tools.fused_moe_triton_tune.common import resolve_device_settings

    # Keep all environment writes local to this test.
    monkeypatch.setattr(exporter.os, 'environ', dict(exporter.os.environ))
    exporter.os.environ.update(AUTOTUNE_DEVICE_COUNT=str(groups),
        AUTOTUNE_NUM_GROUPS_RUN=str(groups), AUTOTUNE_START_GROUP='3', CUR_GROUP='3')
    exporter._prepare_environment(2)
    assert resolve_device_settings(8) == (2, 1, 1)
    assert exporter.os.environ['AUTOTUNE_START_GROUP'] == exporter.os.environ['CUR_GROUP'] == '0'


def test_csv_sigterm_reaps_leaf_and_restores_handler(monkeypatch):
    from types import SimpleNamespace
    from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
    killed = []
    previous = exporter.signal.getsignal(exporter.signal.SIGTERM)

    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            exporter.signal.raise_signal(exporter.signal.SIGTERM)
        return 0

    monkeypatch.setattr(exporter.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=123, wait=wait))
    monkeypatch.setattr(exporter.os, 'killpg', lambda *args: killed.append(args))
    args = SimpleNamespace(batch_sizes='1', max_attempts=1, process_timeout=5)
    monkeypatch.setattr(exporter, '_main', lambda: exporter.benchmark_in_fresh_processes(args, []))
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 128 + exporter.signal.SIGTERM
    assert killed == [(123, exporter.signal.SIGTERM)]
    assert waits == [5, 5]
    assert exporter.signal.getsignal(exporter.signal.SIGTERM) == previous


def test_csv_sigterm_cleans_up_real_detached_leaf():
    import os
    import select
    import signal
    import subprocess
    import sys

    # No CUDA context or files: exercise the real supervisor with a CPU sleeper.
    code = """
import subprocess, sys
from types import SimpleNamespace
from boltops.tools.fused_moe_triton_tune import performance_csv as exporter
real_popen = subprocess.Popen
def start(command, **kwargs):
    child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
    def wait(timeout=None):
        if timeout is not None:
            print(child.pid, flush=True)
        return child.wait(timeout=timeout)
    return SimpleNamespace(pid=child.pid, wait=wait)
exporter.subprocess.Popen = start
args = SimpleNamespace(batch_sizes='1', max_attempts=1, process_timeout=60)
exporter._main = lambda: exporter.benchmark_in_fresh_processes(args, [])
exporter.main()
"""
    parent = subprocess.Popen([sys.executable, '-u', '-c', code], stdout=subprocess.PIPE, text=True)
    leaf = None
    try:
        assert select.select([parent.stdout], [], [], 10)[0], 'Supervisor did not start'
        leaf = int(parent.stdout.readline())
        parent.terminate()
        assert parent.wait(timeout=5) == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(leaf, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if leaf is not None:
            try:
                os.killpg(leaf, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()
