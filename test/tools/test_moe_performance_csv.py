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
