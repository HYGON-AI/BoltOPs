# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Unified scaled adapters execute existing kernels on small shapes only."""
import pytest
import torch
import triton

from boltops.gemm import gemm_scaled


@pytest.mark.parametrize('recipe', [((1, 32), (32, 32)), ((1, 128), (128, 128)),
                                   ((1, 131), (1, 131)), ((5, 131), (17, 131)),
                                   ((1, 131), (17, 131)), ((5, 131), (1, 131))])
@pytest.mark.parametrize('out_dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_existing_scaled_kernels(recipe, out_dtype):
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != 'gfx946':
        pytest.skip('requires gfx946')
    torch.manual_seed(823)
    m, n, k = 5, 17, 131
    a = (torch.randint(-2, 3, (m, k)).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-2, 3, (n, k)).float() / 4).to(torch.float8_e4m3fn)
    (am, ak), (bn, bk) = recipe
    sa = (torch.arange(triton.cdiv(m, am) * triton.cdiv(k, ak)).reshape(triton.cdiv(m, am), -1) % 3 + 1).float() / 2
    sb = (torch.arange(triton.cdiv(n, bn) * triton.cdiv(k, bk)).reshape(triton.cdiv(n, bn), -1) % 3 + 1).float() / 2
    # CPU golden only; device operands stay native FP8 throughout the launch.
    av = a.float() * sa.repeat_interleave(am, 0).repeat_interleave(ak, 1)[:m, :k]
    bv = b.float() * sb.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    out = gemm_scaled(a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), scale_recipe=recipe,
                      out_dtype=out_dtype, backend='triton')
    torch.testing.assert_close(out.cpu(), (av @ bv.T).to(out_dtype), atol=0, rtol=0)
