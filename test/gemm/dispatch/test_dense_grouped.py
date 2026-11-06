# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Small native tests for the three common GEMM interfaces; no tuning scans."""
from dataclasses import replace

import pytest
import torch
import triton

from boltops.gemm import gemm_dense, gemm_grouped, gemm_scaled
from boltops.gemm.descriptors import GemmProblem, OperandSpec
from boltops.gemm.registry import select_plan


def require_hcu():
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')


def dense_problem(m=4096, n=4608, k=4096, num_cus=48):
    return GemmProblem('dense', m, n, k,
                       OperandSpec(torch.float16, strides=(k, 1)),
                       OperandSpec(torch.float16, strides=(1, n)),
                       torch.float16, 'gfx946', num_cus=num_cus)


def test_dense_selects_existing_fp16_config():
    p = dense_problem()
    plan = select_plan(p)
    assert plan.implementation.name == 'triton.dense.fp16'
    assert plan.config.source_config.endswith('M4096_N4608_K4096_TT_wdra48_bcs_mls_mlf3_pp_mstore.json')
    assert select_plan(replace(p, num_cus=96)).implementation.name == 'triton.dense.native'
    assert select_plan(dense_problem(512, 512, 512)).implementation.name == 'triton.dense.native'
    tn = replace(p, b=OperandSpec(torch.float16, strides=(p.k, 1)))
    assert select_plan(tn).implementation.name == 'triton.dense.native'


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_dense_native_small(dtype):
    require_hcu()
    torch.manual_seed(317)
    a = (torch.randint(-4, 5, (5, 64)).float() / 4).to(dtype)
    b = (torch.randint(-4, 5, (7, 64)).float() / 4).to(dtype)
    bias = torch.full((7,), .25, dtype=dtype)
    out = gemm_dense(a.cuda(), b.cuda(), bias=bias.cuda())
    torch.testing.assert_close(out.cpu(), (a.float() @ b.float().T + bias.float()).to(dtype), atol=0, rtol=0)


def test_existing_tuned_fp16_recipe_single_cta():
    require_hcu()
    full = dense_problem()
    plan = select_plan(full)
    # Reuse the selected production recipe on its complete 256x256 TG.
    # This checks integration, not full-size performance or a fake reduced grid.
    p = dense_problem(256, 256, 256)
    torch.manual_seed(319)
    a = (torch.randint(-2, 3, (256, 256)).float() / 4).half()
    nn_b = (torch.randint(-2, 3, (256, 256)).float() / 4).half()
    out = torch.empty((256, 256), dtype=torch.float16, device='cuda')
    from boltops.gemm.triton.dense_dispatch import launch_dense_fp16
    weight = nn_b.cuda().T
    p = replace(p, b=OperandSpec(torch.float16, strides=tuple(weight.stride())))
    launch_dense_fp16(plan.config, p, a.cuda(), weight, None, out)
    torch.testing.assert_close(out.cpu(), (a.float() @ nn_b.float()).half(), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('n_contiguous', [False, True])
def test_grouped_native_equal(dtype, n_contiguous):
    require_hcu()
    a = torch.ones((2, 3, 32), dtype=dtype)
    b = ((torch.arange(2 * 5 * 32).reshape(2, 5, 32) % 5 + 1).float() / 4).to(dtype)
    bias = torch.full((2, 1, 5), .25, dtype=dtype)
    weight = b.transpose(1, 2).contiguous().transpose(1, 2) if n_contiguous else b
    out = gemm_grouped(a.cuda(), weight.cuda(), bias=bias.cuda())
    golden = torch.bmm(a.float(), b.float().transpose(1, 2)) + bias.float()
    torch.testing.assert_close(out.cpu(), golden.to(dtype), atol=0, rtol=0)


def test_unimplemented_native_ragged_does_not_launch_dense_per_group():
    require_hcu()
    a = torch.ones((3, 32), device='cuda', dtype=torch.float16)
    b = torch.ones((2, 5, 32), device='cuda', dtype=torch.float16)
    offsets = torch.tensor([0, 1, 3], device='cuda', dtype=torch.int32)
    with pytest.raises(NotImplementedError, match='no registered grouped GEMM'):
        gemm_grouped(a, b, group_offsets=offsets)
