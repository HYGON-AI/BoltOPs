# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Backend selection and dependency boundaries of the common interfaces."""
from dataclasses import replace
import os
import subprocess
import sys

import pytest
import torch

from boltops.gemm import catalog
from boltops.gemm.descriptors import GemmProblem, OperandSpec, ScaleSpec
from boltops.gemm.registry import Implementation, KernelConfig, launch, select_plan


def problem():
    return GemmProblem('dense', 128, 256, 256,
                       OperandSpec(torch.float8_e4m3fn, strides=(256, 1)),
                       OperandSpec(torch.float8_e4m3fn, strides=(256, 1)),
                       torch.bfloat16, 'gfx946', num_cus=48,
                       a_scale=ScaleSpec(torch.float32, (1, 128)),
                       b_scale=ScaleSpec(torch.float32, (128, 128)))


def test_existing_scaled_path_and_unimplemented_gluon():
    assert select_plan(problem()).implementation.backend == 'triton'
    assert select_plan(problem(), backend='triton').implementation.name == 'triton.scaled.block_fp8'
    with pytest.raises(NotImplementedError, match='backend=gluon'):
        select_plan(problem(), backend='gluon')


@pytest.mark.parametrize('interface', ['scaled', 'dense', 'batched', 'grouped'])
def test_tilelang_has_explicit_empty_capabilities(interface):
    assert catalog.implementations('tilelang', interface) == ()


def test_unimplemented_tilelang_does_not_delegate():
    with pytest.raises(NotImplementedError, match='backend=tilelang'):
        select_plan(problem(), backend='tilelang')


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match='unknown GEMM backend'):
        select_plan(problem(), backend='gulon')


def test_explicit_missing_runtime_is_reported(monkeypatch):
    monkeypatch.setattr(catalog, 'available', lambda name: False)
    with pytest.raises(ImportError, match='triton is not installed'):
        select_plan(problem(), backend='triton')
    with pytest.raises(NotImplementedError):
        select_plan(problem())


def test_selected_launch_error_propagates(monkeypatch):
    import boltops.gemm.registry as registry
    plan = select_plan(problem(), backend='triton')
    class FailedKernel:
        @staticmethod
        def launch_block_scaled(*args, **kwargs):
            raise RuntimeError('native compilation failed')
    monkeypatch.setattr(registry, 'import_module', lambda name: FailedKernel)
    with pytest.raises(RuntimeError, match='native compilation failed'):
        launch(plan, problem())


def test_provider_ownership_checked(monkeypatch):
    class InvalidProvider:
        INTERFACES = {'scaled': (Implementation('foreign', 'dense', 'none', 'none',
                      lambda p: (), lambda p, c: False, backend='gluon'),)}
    monkeypatch.setattr(catalog, 'import_module', lambda name: InvalidProvider)
    with pytest.raises(ValueError, match="another backend"):
        catalog.implementations('tilelang', 'scaled')


def test_frontends_and_tilelang_provider_import_without_any_dsl():
    code = '''
import torch
import importlib.abc
import sys
class BlockDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('triton', 'tilelang', 'tvm'):
            raise AssertionError('unexpected DSL import: ' + fullname)
for name in list(sys.modules):
    if name.split('.')[0] in ('triton', 'tilelang', 'tvm'):
        del sys.modules[name]
sys.meta_path.insert(0, BlockDSL())
from boltops.gemm import scaled, dense, batched, grouped, specialized
from boltops.gemm.catalog import implementations
assert all(implementations('tilelang', i) == () for i in ('scaled', 'dense', 'batched', 'grouped', 'batched_a16wfp4_online_quant'))
'''
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())



@pytest.mark.parametrize('backend', ['gluon', 'tilelang'])
@pytest.mark.parametrize('interface', ['scaled', 'dense', 'batched', 'grouped'])
def test_unimplemented_backend_interfaces_are_empty(backend, interface):
    assert catalog.implementations(backend, interface) == ()


def test_cross_backend_map_wins_over_general_default(monkeypatch):
    from boltops.gemm import performance_map
    p = problem()
    tile = KernelConfig(32, 64, 32)
    general = Implementation('default', 'dense', 'unused', 'unused',
                             lambda p: (tile,), lambda p, c: True, backend='triton')
    tuned = replace(general, name='tuned', backend='gluon')
    monkeypatch.setattr(performance_map, 'tuned_configs',
                        lambda p, name, **kw: (tile,) if name == 'tuned' else ())
    plan = select_plan(p, implementations=(general, tuned))
    assert plan.implementation.backend == 'gluon'
    assert select_plan(p, implementations=(general, tuned), backend='triton').implementation == general


@pytest.mark.parametrize('encoding', ['numeric', 'e8m0'])
def test_unregistered_scale_combinations_fail(encoding):
    p = replace(problem(), a_scale=ScaleSpec(torch.uint8, (1, 64), encoding),
                b_scale=ScaleSpec(torch.uint8, (128, 32), encoding))
    with pytest.raises(NotImplementedError):
        select_plan(p)


def test_unregistered_fp4_never_uses_a_native_byte_dot():
    from dataclasses import replace
    from boltops.gemm.descriptors import OperandSpec
    from boltops.gemm.registry import select_plan
    p = replace(problem(), b=OperandSpec(torch.uint8, 'fp4', 'e2m1_low_high', -1, 2, strides=(128, 1)))
    with pytest.raises(NotImplementedError):
        select_plan(p)
