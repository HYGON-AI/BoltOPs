# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Public operation contracts have one entry point each, independent of kernels."""
import pytest
import torch
import triton

from boltops import gemm


def require_hcu():
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')


def test_public_surface_has_no_legacy_duplicates():
    expected = {'gemm_dense', 'gemm_scaled', 'gemm_batched', 'gemm_grouped',
                'gemm_batched_a16wfp4_online_quant'}
    assert set(gemm.__all__) == expected
    assert all(callable(getattr(gemm, name)) for name in expected)
    for old in ('matmul_persistent', 'gemm_a16w16', 'gemm_a8w8_blockscale',
                'w8a8_triton_block_scaled_mm', 'triton_scaled_mm', 'w8a8_block_int8_matmul',
                'awq_gemm_triton', 'gemm_afp4wfp4', 'gemm_awq', 'gemm_mxfp4', 'gemm_batched_a16wfp4', 'batched_gemm_bf16', 'batched_gemm_a16wfp4'):
        assert not hasattr(gemm, old)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('n_contiguous', [False, True])
def test_batched_native(dtype, n_contiguous):
    require_hcu()
    torch.manual_seed(631)
    a = (torch.randint(-2, 3, (2, 3, 32)).float() / 4).to(dtype)
    b = (torch.randint(-2, 3, (2, 5, 32)).float() / 4).to(dtype)
    weight = b.transpose(1, 2).contiguous().transpose(1, 2) if n_contiguous else b
    bias = torch.full((2, 1, 5), .25, dtype=dtype)
    y = gemm.gemm_batched(a.cuda(), weight.cuda(), bias=bias.cuda())
    torch.testing.assert_close(y.cpu(), (torch.bmm(a.float(), b.float().transpose(1, 2)) + bias).to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.int8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('block_scale', [False, True])
def test_scaled_non_fp8(dtype, block_scale):
    require_hcu()
    torch.manual_seed(634)
    a = torch.randint(-2, 3, (3, 64)).to(dtype)
    b = torch.randint(-2, 3, (17, 64)).to(dtype)
    recipe = ((1, 32), (32, 32)) if block_scale else ((1, 64), (1, 64))
    sa = torch.full((3, 2 if block_scale else 1), .5)
    sb = torch.full((1 if block_scale else 17, 2 if block_scale else 1), .25)
    bias = None if block_scale else torch.full((17,), .5, device='cuda')
    y = gemm.gemm_scaled(a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), scale_recipe=recipe, bias=bias,
                         out_dtype=torch.float32, backend='triton')
    golden = (a.float() @ b.float().T) * .125 + (0 if bias is None else .5)
    torch.testing.assert_close(y.cpu(), golden, atol=0, rtol=0)


def test_awq_explicit_zero_points():
    require_hcu()
    # Every packed nibble: weight=3, zero=1 => dequantized weight=1.
    a = torch.full((2, 64), .25, dtype=torch.float16, device='cuda')
    w = torch.full((64, 8), 0x33333333, dtype=torch.int32, device='cuda')
    z = torch.full((1, 8), 0x11111111, dtype=torch.int32, device='cuda')
    s = torch.full((1, 64), .5, dtype=torch.float16, device='cuda')
    y = gemm.gemm_scaled(a, w, b_scale=s, b_zero_points=z, b_format="awq_int4",
                         scale_recipe=((1, 1), (1, 64)))
    torch.testing.assert_close(y.cpu(), torch.full((2, 64), 16., dtype=torch.float16), atol=0, rtol=0)


def test_batched_and_grouped_use_distinct_plans():
    from dataclasses import replace
    from boltops.gemm.descriptors import GemmProblem, OperandSpec, GroupSpec
    from boltops.gemm.registry import select_plan
    p = GemmProblem('batched', 3, 5, 32, OperandSpec(torch.bfloat16, strides=(96, 32, 1)),
                    OperandSpec(torch.bfloat16, strides=(160, 32, 1)), torch.bfloat16, 'gfx946', batch=2)
    assert select_plan(p).implementation.name == 'triton.batched.native16'
    q = replace(p, kind='grouped', batch=1, m=6, groups=GroupSpec('equal', 2))
    assert select_plan(q).implementation.name == 'triton.grouped.native16'


def test_awq_zero_point_defaults_to_zero():
    require_hcu()
    a = torch.full((2, 64), .25, dtype=torch.float16, device='cuda')
    w = torch.full((64, 8), 0x33333333, dtype=torch.int32, device='cuda')
    s = torch.full((1, 64), .5, dtype=torch.float16, device='cuda')
    y = gemm.gemm_scaled(a, w, b_scale=s, b_format='awq_int4', scale_recipe=((1, 1), (1, 64)))
    torch.testing.assert_close(y.cpu(), torch.full((2, 64), 24., dtype=torch.float16), atol=0, rtol=0)


def test_scaled_prequantized_mxfp4():
    require_hcu()
    a = torch.full((16, 32), 0x22, dtype=torch.uint8, device='cuda')
    b = torch.full((16, 32), 0x22, dtype=torch.uint8, device='cuda')
    scale = torch.full((16, 2), 127, dtype=torch.uint8, device='cuda')
    y = gemm.gemm_scaled(a, b, scale, scale, a_format='fp4_e2m1', b_format='fp4_e2m1',
                         scale_recipe=((1, 32), (1, 32)), scale_encodings=('e8m0', 'e8m0'))
    torch.testing.assert_close(y.cpu(), torch.full((16, 16), 64., dtype=torch.bfloat16), atol=0, rtol=0)


def test_batched_a16wfp4_online_quant():
    require_hcu()
    a = torch.ones((1, 16, 64), dtype=torch.bfloat16, device='cuda')
    b = torch.full((1, 16, 32), 0x22, dtype=torch.uint8, device='cuda')
    scale = torch.full((1, 16, 2), 127, dtype=torch.uint8, device='cuda')
    y = gemm.gemm_batched_a16wfp4_online_quant(a, b, scale)
    torch.testing.assert_close(y.cpu(), torch.full((1, 16, 16), 64., dtype=torch.bfloat16), atol=0, rtol=0)
