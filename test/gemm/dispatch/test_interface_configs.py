# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Each public GEMM interface reads only its own backend configuration map."""
import json

import pytest
import torch

from boltops.gemm import performance_map
from boltops.gemm.descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    performance_map._entries.cache_clear()
    monkeypatch.setattr(performance_map, '_CONFIG_DIR', tmp_path)
    yield tmp_path
    performance_map._entries.cache_clear()


def problem(interface):
    return GemmProblem(
        interface if interface in ('batched', 'grouped') else 'dense', 32, 64, 64,
        OperandSpec(torch.bfloat16, strides=(64, 1)),
        OperandSpec(torch.bfloat16, strides=(64, 1)), torch.bfloat16, 'gfx946',
        a_scale=ScaleSpec(torch.float32, (1, 32)) if interface == 'scaled' else None,
        b_scale=ScaleSpec(torch.float32, (32, 32)) if interface == 'scaled' else None,
        groups=GroupSpec('equal', 2) if interface == 'grouped' else None)


@pytest.mark.parametrize('interface', ['scaled', 'dense', 'batched', 'grouped'])
def test_only_own_configuration_file_is_required(config_dir, interface):
    (config_dir / f'gemm_{interface}.json').write_text(json.dumps(
        {'schema_version': 3, 'interface': f'gemm_{interface}', 'entries': []}))
    # The other interfaces' files deliberately do not exist.
    assert performance_map.tuned_configs(problem(interface), 'unregistered') == ()


@pytest.mark.parametrize('interface', ['scaled', 'dense', 'batched', 'grouped'])
def test_configuration_cannot_belong_to_a_different_interface(config_dir, interface):
    (config_dir / f'gemm_{interface}.json').write_text(json.dumps(
        {'schema_version': 3, 'interface': 'another_interface', 'entries': []}))
    with pytest.raises(ValueError, match=f'configuration file must belong to gemm_{interface}'):
        performance_map.tuned_configs(problem(interface), 'unregistered')
