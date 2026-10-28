# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2021 AlQuraishi Laboratory
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/protenix/model/triangular/layers.py
"""Functional and configuration tests for TriangleAttention."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("triton")

import torch
import yaml

from boltops.fa import triangle_attention_fwd
from boltops.fa.triangle_attention.triton import config
from boltops.tools.triangle_attention_triton_tune import tune_triangle_attention as tune
from boltops.tools.triangle_attention_triton_tune.tune_triangle_attention import (
    make_inputs, torch_production_attention,
)

GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


_CASE_FILE = Path(__file__).parents[2] / "boltops/tools/triangle_attention_triton_tune/benchmark_cases.yaml"
_CASES = tuple(row for row in yaml.safe_load(_CASE_FILE.read_text())["cases"] if row["enabled"])


def _reference(query, key, value, biases):
    # Match Protenix's protenix/model/triangular/layers.py::_attention.
    # Upstream: https://github.com/bytedance/Protenix
    key = key.transpose(-1, -2)
    a = torch.matmul(query, key)
    for b in biases:
        a += b
    with torch.amp.autocast("cuda", enabled=False):
        a = torch.softmax(a, dim=-1)
    return torch.matmul(a, value)


def test_public_api_exports_triangle_attention():
    import inspect

    params = tuple(inspect.signature(triangle_attention_fwd).parameters)
    assert params == ("q", "k", "v", "biases")


@pytest.mark.parametrize("case", _CASES, ids=lambda row: row["name"])
@pytest.mark.parametrize("seed", [0, 17, 42])
@GPU_REQUIRED
def test_triangle_attention_matches_reference(case, seed):
    torch.manual_seed(seed)
    b, h, n, d = case["batch"], case["heads"], case["seq_len"], case["head_dim"]
    q, k, v, mask, mask_bias, bias = make_inputs(
        b, h, n, d, case["mask_density"], case.get("mask_layout", "contiguous"),
        bias_dtype=torch.bfloat16, mask_dtype=torch.bfloat16,
    )
    assert q.stride() == (n * h * d, d, h * d, 1)
    assert all(t.dtype == torch.bfloat16 for t in (q, k, v, mask_bias, bias))
    assert mask_bias.stride(-1) == (b if case.get("mask_layout") == "strided" else 1)
    expected = _reference(q, k, v, [mask_bias, bias])
    actual = triangle_attention_fwd(q, k, v, [mask_bias, bias])
    assert actual.data_ptr() != q.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=2e-2)
    assert torch.isfinite(actual).all()

    # The compact [B,N] mask is part of the public contract as well.
    compact_bias = mask_bias[:, 0, 0, :]
    compact = triangle_attention_fwd(q, k, v, [compact_bias, bias])
    torch.testing.assert_close(compact.float(), actual.float(), atol=0, rtol=0)


@GPU_REQUIRED
@pytest.mark.parametrize("n", [1, 63, 64, 65, 129])
@pytest.mark.parametrize("batched_bias", [False, True])
def test_strided_inputs(n, batched_bias):
    torch.manual_seed(17)
    b, h, d = 3, 2, 32
    q, k, v = [torch.randn((b, n, h, d * 2), device="cuda", dtype=torch.bfloat16)
               .transpose(1, 2)[..., ::2] for _ in range(3)]
    q.div_(d ** 0.5)
    bias = torch.randn((b if batched_bias else 1, n, n, h), device="cuda", dtype=torch.bfloat16).permute(0, 3, 1, 2)
    mask = (torch.rand((n, b), device="cuda") > .3).t()
    mask[:, 0] = True
    mask_bias = torch.zeros((b, 1, 1, n), device="cuda", dtype=torch.bfloat16).masked_fill_(~mask[:, None, None, :], -1.0e9)
    expected = _reference(q, k, v, [mask_bias, bias])
    actual = triangle_attention_fwd(q, k, v, [mask_bias, bias])
    torch.testing.assert_close(actual.float(), expected.float(), atol=.03, rtol=.02)
    assert torch.isfinite(actual).all()



@GPU_REQUIRED
@pytest.mark.parametrize("mask_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask_layout", ["contiguous", "strided"])
def test_mixed_bias_dtypes(mask_dtype, bias_dtype, mask_layout):
    torch.manual_seed(17)
    q, k, v, _, mask_bias, bias = make_inputs(
        3, 2, 65, 32, .9, mask_layout,
        mask_dtype=mask_dtype, bias_dtype=bias_dtype,
    )
    actual = triangle_attention_fwd(q, k, v, [mask_bias, bias])
    expected = _reference(q, k, v, [mask_bias, bias])
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=.03, rtol=.02)


@GPU_REQUIRED
@pytest.mark.parametrize("prep", [0, 1, 2])
def test_bf16_bias_preprocessing(prep):
    from boltops.fa.triangle_attention.triton.triangle_attention import (
        _prepare_bias, triangle_attention_fwd as triangle_attention_fwd_backend,
    )
    from boltops.fa.triangle_attention.triton.config import DEFAULT_CONFIG

    torch.manual_seed(17)
    q, k, v, _, mb, bias = make_inputs(2, 2, 65, 32, .9, "strided")
    padded = _prepare_bias(bias, 64)
    assert padded.dtype == bias.dtype == mb.dtype == torch.bfloat16
    torch.testing.assert_close(padded[..., :65], bias, atol=0, rtol=0)
    assert torch.isneginf(padded[..., 65:]).all()
    cfg = dict(DEFAULT_CONFIG, BIAS_PREP=prep)
    actual = triangle_attention_fwd_backend(q, k, v, [mb, bias], config=cfg)
    torch.testing.assert_close(actual, _reference(q, k, v, [mb, bias]), atol=.03, rtol=.02)
    with pytest.raises(ValueError, match="preserve bias dtype"):
        triangle_attention_fwd_backend(q, k, v, [mb, bias], config=dict(cfg, BIAS_PREP=3))


def test_only_production_cases_are_timed():
    assert {case["name"] for case in tune.load_cases()} == {
        "production-h2", "production-h2-mask-strided", "production-h8"
    }


def test_search_space_is_valid():
    assert len(tune.candidates()) == 320
    assert tune.candidates(True) == [config.DEFAULT_CONFIG]
    assert all(config._validate(candidate) == candidate for candidate in tune.candidates())


def test_compiler_pass_failure_skips_candidate(monkeypatch):
    monkeypatch.setattr(tune, "override", lambda candidate: nullcontext())
    monkeypatch.setattr(tune, "measure", lambda run, args: (_ for _ in ()).throw(
        RuntimeError("PassManager::run failed")
    ))
    candidate = tune.candidates()[3]
    args = SimpleNamespace(skip_correctness=True)
    record = tune.evaluate_candidate(candidate, lambda: None, None, args)
    assert record == {
        "config": candidate, "status": "failed", "error_type": "RuntimeError",
        "error": "PassManager::run failed",
    }


def test_unrelated_runtime_error_is_not_skipped(monkeypatch):
    monkeypatch.setattr(tune, "override", lambda candidate: nullcontext())
    monkeypatch.setattr(tune, "measure", lambda run, args: (_ for _ in ()).throw(
        RuntimeError("device unavailable")
    ))
    with pytest.raises(RuntimeError, match="device unavailable"):
        tune.evaluate_candidate(tune.candidates()[0], lambda: None, None,
                                SimpleNamespace(skip_correctness=True))


def test_config_roundtrip_and_missing_shape(tmp_path, monkeypatch):
    info = SimpleNamespace(arch="test", cu_count=1)
    monkeypatch.setattr(tune, "get_device_info", lambda: info)
    monkeypatch.setattr(config, "get_device_info", lambda device=None: info)
    monkeypatch.setattr(config, "CONFIG_ROOT", tmp_path)
    case = tune.load_cases()[0]
    candidate = dict(BLOCK_M=32, BLOCK_N=128, num_warps=4, num_stages=1)
    path = tune.write_config(case, candidate, tmp_path)
    assert path.is_file()
    case_strided = next(case for case in tune.load_cases()
                        if case["name"] == "production-h2-mask-strided")
    candidate_strided = dict(BLOCK_M=64, BLOCK_N=32, num_warps=1, num_stages=1)
    assert tune.write_config(case_strided, candidate_strided, tmp_path) == path
    case_h8 = next(case for case in tune.load_cases() if case["name"] == "production-h8")
    candidate_h8 = dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=1)
    assert tune.write_config(case_h8, candidate_h8, tmp_path) == path
    assert path.name == "triangle_attention-device=test_cu1.json"
    config.clear_config_cache()
    kwargs = {key: case[key] for key in config.CONFIG_KEY_FIELDS}
    try:
        assert config.get_kernel_config(**kwargs) == candidate
        assert config.get_kernel_config(**{
            key: case_strided[key] for key in config.CONFIG_KEY_FIELDS
        }) == candidate_strided
        assert config.get_kernel_config(**{
            key: case_h8[key] for key in config.CONFIG_KEY_FIELDS
        }) == candidate_h8
        assert len(json.loads(path.read_text())["config"]) == 3
        kwargs["batch"] = 1
        assert config.get_kernel_config(**kwargs) == config.DEFAULT_CONFIG
    finally:
        config.clear_config_cache()


@pytest.mark.parametrize("payload", [
    [], None,
    {"key": list(config.CONFIG_KEY_FIELDS), "config": []},
    {"key": list(config.CONFIG_KEY_FIELDS), "config": None},
], ids=["array", "null", "config-array", "config-null"])
@pytest.mark.parametrize("compatible_config", [False, True], ids=["default", "compatible"])
def test_malformed_config_falls_back(tmp_path, monkeypatch, payload, compatible_config):
    info = SimpleNamespace(arch="test", cu_count=1)
    monkeypatch.setattr(config, "get_device_info", lambda device=None: info)
    monkeypatch.setattr(config, "get_triton_capabilities", lambda: None)
    directories = [tmp_path / "current", tmp_path / "compatible"]
    monkeypatch.setattr(config, "get_compatible_triton_config_dirs", lambda root, caps: directories)
    filename = config.make_config_filename(device_label="test_cu1")
    directories[0].mkdir()
    (directories[0] / filename).write_text(json.dumps(payload))
    kwargs = dict(batch=2, heads=2, seq_len=65, head_dim=32, mask_layout="contiguous")
    expected = dict(config.DEFAULT_CONFIG)
    if compatible_config:
        expected = dict(BLOCK_M=32, BLOCK_N=128, num_warps=4, num_stages=1)
        directories[1].mkdir()
        (directories[1] / filename).write_text(json.dumps({
            "key": list(config.CONFIG_KEY_FIELDS),
            "config": {str(config.make_config_key(**kwargs)): expected},
        }))
    config.clear_config_cache()
    try:
        assert config.get_kernel_config(**kwargs) == expected
    finally:
        config.clear_config_cache()


def test_mask_layout_is_part_of_config_key():
    kwargs = dict(batch=693, heads=2, seq_len=693, head_dim=32)
    assert config.make_config_key(**kwargs, mask_layout="contiguous") != config.make_config_key(
        **kwargs, mask_layout="strided"
    )
    with pytest.raises(ValueError, match="Unsupported mask layout"):
        config.make_config_key(**kwargs, mask_layout="unknown")


@pytest.mark.parametrize("candidate", [
    {}, {"BLOCK_M": 3}, dict(BLOCK_M=32, BLOCK_N=63, num_warps=4, num_stages=2)
])
def test_invalid_config_rejected(candidate):
    assert config._validate(candidate) is None


@GPU_REQUIRED
def test_nondefault_launch_tiles():
    torch.manual_seed(0)
    q, k, v, _, mask_bias, bias = make_inputs(2, 2, 65, 32, .9)
    reference = torch_production_attention(q, k, v, mask_bias, bias)
    for block_m, block_n in [(16, 32), (32, 128)]:
        with tune.override(dict(BLOCK_M=block_m, BLOCK_N=block_n,
                                num_warps=4, num_stages=1)):
            tune.check(triangle_attention_fwd(q, k, v, [mask_bias, bias]), reference)
