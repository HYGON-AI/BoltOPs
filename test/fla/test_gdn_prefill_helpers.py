import os

import importlib
import json
from functools import partial

import pytest
import torch
import torch.nn.functional as functional
import triton

from boltops.fla import gdn
from boltops.fla.gdn.triton import prefill_helpers


@pytest.fixture
def device():
    if os.environ.get('TRITON_INTERPRET') == '1':
        return 'cpu'
    if not torch.cuda.is_available():
        pytest.skip('CUDA/HIP device required unless TRITON_INTERPRET=1')
    return 'cuda'


def strided_matrix(values):
    storage = torch.empty((values.shape[0], values.shape[1] * 2 + 3),
                          dtype=values.dtype, device=values.device)
    view = storage[:, 1:2 * values.shape[1]:2]
    view.copy_(values)
    return view


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('tokens,heads_qk,heads_v,key_width,value_width', [
    (3, 2, 4, 7, 9), (1, 1, 1, 17, 13), (2, 2, 8, 16, 32),
])
@pytest.mark.parametrize('strided', [False, True])
def test_qkvzba_split(device, dtype, tokens, heads_qk, heads_v, key_width, value_width, strided):
    torch.manual_seed(73)
    widths = [heads_qk * key_width, heads_qk * key_width, heads_v * value_width, heads_v * value_width]
    segments = [torch.randn(tokens, width, dtype=dtype, device=device) + index
                for index, width in enumerate(widths)]
    b_values = torch.randn(tokens, heads_v, dtype=dtype, device=device) - 2
    a_values = torch.randn(tokens, heads_v, dtype=dtype, device=device) + 2
    projected_qkvz = torch.cat(segments, dim=-1)
    projected_ba = torch.cat((b_values, a_values), dim=-1)
    before_qkvz, before_ba = projected_qkvz.clone(), projected_ba.clone()
    if strided:
        projected_qkvz, projected_ba = map(strided_matrix, (projected_qkvz, projected_ba))
    outputs = gdn.gdn_qkvzba_split(projected_qkvz, projected_ba, heads_qk, heads_v, key_width, value_width)
    expected = (torch.cat(segments[:3], dim=-1), segments[3], b_values, a_values)
    for actual, reference in zip(outputs, expected):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(projected_qkvz, before_qkvz, rtol=0, atol=0)
    torch.testing.assert_close(projected_ba, before_ba, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('beta,threshold', [(1.0, 20.0), (0.5, 8.0), (2.0, 3.0)])
def test_fused_gating(device, dtype, beta, threshold):
    if os.environ.get('TRITON_INTERPRET') == '1':
        pytest.skip('Triton interpreter does not implement libdevice.log1p; use CUDA/HIP')
    torch.manual_seed(79)
    a_values = torch.linspace(-30, 30, 39).reshape(3, 13).to(dtype=dtype, device=device)
    b_values = torch.linspace(30, -30, 39).reshape(3, 13).to(dtype=dtype, device=device)
    a_projection, b_projection = map(strided_matrix, (a_values, b_values))
    log_decay = torch.linspace(-2, 2, 26, device=device)[::2]
    time_bias = torch.linspace(-1, 1, 26, device=device)[::2]
    decay, update = gdn.gdn_fused_gating(a_projection, b_projection, log_decay, time_bias, beta, threshold)
    reference = -log_decay.exp() * functional.softplus(a_projection.float() + time_bias, beta, threshold)
    assert decay.dtype == update.dtype == torch.float32
    torch.testing.assert_close(decay, reference, rtol=3e-5, atol=1e-12)
    torch.testing.assert_close(update, b_projection.float().sigmoid(), rtol=3e-5, atol=1e-12)
    assert torch.isfinite(decay).all()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('width,eps', [(7, 1e-6), (128, 1e-5), (193, 1e-3)])
def test_rmsnorm_gated(device, dtype, width, eps):
    torch.manual_seed(83)
    hidden = torch.randn(3, width, dtype=dtype, device=device)
    hidden[0] = 0
    hidden[1] *= 1e-3
    gate = torch.randn_like(hidden) * 3
    weight = torch.randn(width * 2, dtype=dtype, device=device)[::2]
    hidden, gate = map(strided_matrix, (hidden, gate))
    output = gdn.gdn_rmsnorm_gated(hidden, gate, weight, eps)
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + eps)
    reference = (normalized * weight.float() * functional.silu(gate.float())).to(dtype)
    assert output.dtype == dtype and output.is_contiguous()
    torch.testing.assert_close(output, reference, rtol=2e-5 if dtype == torch.float32 else 1e-2, atol=2e-6)


def test_public_exports_match_backend():
    for name in ('gdn_qkvzba_split', 'gdn_fused_gating', 'gdn_rmsnorm_gated'):
        assert getattr(gdn, name) is getattr(prefill_helpers, name)


@pytest.mark.parametrize('rows,width,dtype,strided,supported,config,expected_rows', [
    (384, 128, torch.bfloat16, False, True, None, 4),
    (2048, 128, torch.bfloat16, False, True, None, 16),
    (65536, 128, torch.bfloat16, False, True, None, 16),
    (383, 128, torch.bfloat16, False, True, None, 1),
    (384, 64, torch.bfloat16, False, True, None, 1),
    (384, 128, torch.float32, False, True, None, 1),
    (384, 128, torch.bfloat16, True, True, None, 1),
    (384, 128, torch.bfloat16, False, False, None, 1),
    (384, 128, torch.bfloat16, False, True, {'ROWS': 1}, 1),
    (384, 128, torch.bfloat16, False, True, {'ROWS': 32}, 32),
])
def test_gfx946_rms_grouped_dispatch(rows, width, dtype, strided, supported,
                                    config, expected_rows, monkeypatch):
    launches = []

    class Capture:
        def __init__(self, grouped):
            self.grouped = grouped

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.grouped, grid, kwargs))
            return launch

    monkeypatch.setattr(prefill_helpers, '_gfx946_helper_device_supported', lambda device: supported)
    monkeypatch.setattr(prefill_helpers, '_gdn_rmsnorm_gated_kernel', Capture(False))
    monkeypatch.setattr(prefill_helpers, '_gdn_rmsnorm_gated_rows_kernel', Capture(True))
    hidden = torch.empty(rows, width * (2 if strided else 1), dtype=dtype)
    if strided:
        hidden = hidden[:, ::2]
    output = prefill_helpers.gdn_rmsnorm_gated(hidden, hidden, torch.empty(width, dtype=dtype),
                                             kernel_cfg=config)
    assert output.shape == hidden.shape and output.is_contiguous()
    grouped, grid, kwargs = launches.pop()
    assert not launches
    assert grouped == (expected_rows > 1)
    assert grid == (triton.cdiv(rows, expected_rows),)
    if grouped:
        assert kwargs['ROWS'] == expected_rows and kwargs['SILU_RCP']


@pytest.mark.parametrize('tokens,dtype,supported,config,expected_block', [
    (2048, torch.bfloat16, True, None, 4096),
    (2047, torch.bfloat16, True, None, 512),
    (2048, torch.float32, True, None, 512),
    (2048, torch.bfloat16, False, None, 512),
    (2048, torch.bfloat16, True, {'BLOCK': 512}, 512),
])
def test_gfx946_split_launch_config(tokens, dtype, supported, config, expected_block, monkeypatch):
    launches = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))
            return launch

    monkeypatch.setattr(prefill_helpers, '_gfx946_helper_device_supported', lambda device: supported)
    monkeypatch.setattr(prefill_helpers, '_gdn_qkvzba_split_kernel', Capture())
    outputs = prefill_helpers.gdn_qkvzba_split(
        torch.empty(tokens, 12288, dtype=dtype), torch.empty(tokens, 64, dtype=dtype),
        16, 32, kernel_cfg=config)
    assert all(output.is_contiguous() for output in outputs)
    grid, kwargs = launches.pop()
    assert not launches
    assert grid == (triton.cdiv(12288, expected_block), tokens)
    assert kwargs['BLOCK'] == expected_block


@pytest.mark.parametrize('tokens,dtype,supported,strided,config,block,rows', [
    (28672, torch.bfloat16, True, False, None, 4096, 8),
    (28671, torch.bfloat16, True, False, None, 512, 1),
    (28672, torch.bfloat16, False, False, None, 512, 1),
    (28672, torch.float32, True, False, None, 512, 1),
    (28672, torch.bfloat16, True, True, None, 512, 1),
    (28672, torch.bfloat16, True, False, {}, 512, 1),
    (28672, torch.bfloat16, True, False, {'BLOCK': 4096}, 4096, 1),
    (28672, torch.bfloat16, True, False, {'BLOCK': 4096, 'ROWS': 8}, 4096, 8),
    (3, torch.float32, False, False, {'ROWS': 1}, 512, 1),
    (37, torch.bfloat16, True, True, {'ROWS': 8}, 512, 8),
])
def test_gfx946_split_grouped_launch_config(tokens, dtype, supported, strided,
                                           config, block, rows, monkeypatch):
    launches = []

    class Capture:
        def __init__(self, grouped):
            self.grouped = grouped

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.grouped, grid, kwargs))
            return launch

    monkeypatch.setattr(prefill_helpers, '_gfx946_helper_device_supported', lambda device: supported)
    monkeypatch.setattr(prefill_helpers, '_gdn_qkvzba_split_kernel', Capture(False))
    monkeypatch.setattr(prefill_helpers, '_gdn_qkvzba_split_rows_kernel', Capture(True))
    step = 2 if strided else 1
    projection = torch.empty(tokens, 2560 * step, dtype=dtype)[:, ::step]
    gates = torch.empty(tokens, 16 * step, dtype=dtype)[:, ::step]
    outputs = prefill_helpers.gdn_qkvzba_split(projection, gates, 2, 8, kernel_cfg=config)
    assert all(output.is_contiguous() for output in outputs)
    grouped, grid, options = launches.pop()
    assert grouped is (rows > 1)
    assert grid == (triton.cdiv(2560, block), triton.cdiv(tokens, rows))
    assert options['BLOCK'] == block
    if grouped:
        assert options['ROWS'] == rows and options['TOTAL_ROWS'] == tokens


@pytest.mark.parametrize('dtype,supported,config,expected', [
    (torch.bfloat16, True, None, True),
    (torch.bfloat16, False, None, False),
    (torch.float32, True, None, False),
    (torch.bfloat16, True, {}, False),
    (torch.bfloat16, True, {'FAST_MATH': False}, False),
    (torch.float32, False, {'FAST_MATH': True}, True),
])
def test_gfx946_gating_launch_config(dtype, supported, config, expected, monkeypatch):
    launches = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)
            return launch

    monkeypatch.setattr(prefill_helpers, '_gfx946_helper_device_supported', lambda device: supported)
    monkeypatch.setattr(prefill_helpers, '_gdn_fused_gating_kernel', Capture())
    values = torch.empty(7, 3, dtype=dtype)
    prefill_helpers.gdn_fused_gating(values, values, torch.zeros(3), torch.zeros(3),
                                    kernel_cfg=config)
    assert launches.pop()['FAST_MATH'] is expected


@pytest.mark.parametrize('tokens,heads,config,expected_block', [
    (2048, 32, None, 256),
    (6656, 32, None, 1024),
    (6656, 64, None, 1024),
    (3584, 64, None, 1024),
    (28672, 8, None, 1024),
    (53248, 8, None, 1024),
    (6656, 64, {}, 256),
    (6656, 64, {'BLOCK': 512}, 512),
])
def test_gfx946_gating_large_launch_config(tokens, heads, config, expected_block, monkeypatch):
    launches = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))
            return launch

    monkeypatch.setattr(prefill_helpers, '_gfx946_helper_device_supported', lambda device: True)
    monkeypatch.setattr(prefill_helpers, '_gdn_fused_gating_kernel', Capture())
    values = torch.empty(tokens, heads, dtype=torch.bfloat16)
    prefill_helpers.gdn_fused_gating(values, values, torch.zeros(heads), torch.zeros(heads),
                                    kernel_cfg=config)
    grid, options = launches.pop()
    assert options['BLOCK'] == expected_block
    assert grid == (triton.cdiv(tokens * heads, expected_block),)


def test_zero_token_inputs_do_not_launch(monkeypatch):
    class RejectLaunch:
        def __getitem__(self, grid):
            pytest.fail('empty inputs must not launch')

    for name in ('_gdn_qkvzba_split_kernel', '_gdn_fused_gating_kernel', '_gdn_rmsnorm_gated_kernel'):
        monkeypatch.setattr(prefill_helpers, name, RejectLaunch())
    split = gdn.gdn_qkvzba_split(torch.empty(0, 12), torch.empty(0, 2), 1, 1, 3, 3)
    assert [tuple(output.shape) for output in split] == [(0, 9), (0, 3), (0, 1), (0, 1)]
    gates = gdn.gdn_fused_gating(torch.empty(0, 4), torch.empty(0, 4), torch.ones(4), torch.ones(4))
    assert all(output.shape == (0, 4) for output in gates)
    assert gdn.gdn_rmsnorm_gated(torch.empty(0, 7), torch.empty(0, 7), torch.ones(7)).shape == (0, 7)


@pytest.mark.parametrize('case', [
    'split-width', 'split-dtype', 'split-heads', 'split-groups', 'split-stride',
    'gate-shape', 'gate-vector', 'gate-beta', 'gate-threshold',
    'norm-shape', 'norm-weight', 'norm-eps', 'norm-width',
])
def test_invalid_arguments(case):
    hidden = torch.empty(3, 8)
    weight = torch.ones(8)
    with pytest.raises(ValueError):
        if case == 'split-width':
            gdn.gdn_qkvzba_split(hidden, hidden, 1, 1)
        elif case == 'split-dtype':
            gdn.gdn_qkvzba_split(hidden.to(torch.int32), hidden, 1, 1)
        elif case == 'split-heads':
            gdn.gdn_qkvzba_split(hidden, hidden, 0, 1)
        elif case == 'split-groups':
            gdn.gdn_qkvzba_split(hidden, hidden, 2, 3)
        elif case == 'split-stride':
            gdn.gdn_qkvzba_split(hidden[:1].expand(3, 8), hidden, 1, 1)
        elif case == 'gate-shape':
            gdn.gdn_fused_gating(hidden, hidden[:, :4], weight, weight)
        elif case == 'gate-vector':
            gdn.gdn_fused_gating(hidden, hidden, weight[:4], weight)
        elif case == 'gate-beta':
            gdn.gdn_fused_gating(hidden, hidden, weight, weight, softplus_beta=0)
        elif case == 'gate-threshold':
            gdn.gdn_fused_gating(hidden, hidden, weight, weight, softplus_threshold=float('nan'))
        elif case == 'norm-shape':
            gdn.gdn_rmsnorm_gated(hidden, hidden[:, :4], weight)
        elif case == 'norm-weight':
            gdn.gdn_rmsnorm_gated(hidden, hidden, weight[:4])
        elif case == 'norm-eps':
            gdn.gdn_rmsnorm_gated(hidden, hidden, weight, eps=-1)
        else:
            gdn.gdn_rmsnorm_gated(torch.empty(3, 0), torch.empty(3, 0), torch.empty(0))


_BENCH_1102_DTYPES = {'BF16': torch.bfloat16, 'FP16': torch.float16,
                      'FP32': torch.float32, 'FP8E4M3': torch.float8_e4m3fn}


_BENCH_1102_SHAPES = [
    {'id': 's000', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 12288], [6656, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[6656, 8192], [6656, 4096], [6656, 32], [6656, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 11}, {'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 10}]},
    {'id': 's002', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 32], [6656, 32], [32], [32]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[6656, 32], [6656, 32]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 13}, {'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 12}]},
    {'id': 's004', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Prefill_Chunk', 'inputs': [[212992, 128], [212992, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[212992, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 15}, {'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 14}]},
    {'id': 's006', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Remainder', 'inputs': [[2560, 12288], [2560, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[2560, 8192], [2560, 4096], [2560, 32], [2560, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 73}]},
    {'id': 's008', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Remainder', 'inputs': [[2560, 32], [2560, 32], [32], [32]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[2560, 32], [2560, 32]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 75}]},
    {'id': 's010', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Prefill_Remainder', 'inputs': [[81920, 128], [81920, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[81920, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 77}]},
    {'id': 's012', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Decode', 'inputs': [[64, 12288], [64, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[64, 8192], [64, 4096], [64, 32], [64, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'decode_shape', 'row': 11}]},
    {'id': 's015', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Decode', 'inputs': [[2048, 128], [2048, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[2048, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'decode_shape', 'row': 14}]},
    {'id': 's037', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Remainder', 'inputs': [[2048, 12288], [2048, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[2048, 8192], [2048, 4096], [2048, 32], [2048, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 85}]},
    {'id': 's039', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Remainder', 'inputs': [[2048, 32], [2048, 32], [32], [32]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[2048, 32], [2048, 32]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 87}]},
    {'id': 's041', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Prefill_Remainder', 'inputs': [[65536, 128], [65536, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[65536, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 89}]},
    {'id': 's043', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Decode', 'inputs': [[12, 12288], [12, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[12, 8192], [12, 4096], [12, 32], [12, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'decode_shape', 'row': 10}]},
    {'id': 's046', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Decode', 'inputs': [[384, 128], [384, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[384, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'decode_shape', 'row': 13}]},
    {'id': 's098', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 20480], [6656, 128]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[6656, 12288], [6656, 8192], [6656, 64], [6656, 64]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 11}]},
    {'id': 's100', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 64], [6656, 64], [64], [64]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[6656, 64], [6656, 64]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 13}]},
    {'id': 's102', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Prefill_Chunk', 'inputs': [[425984, 128], [425984, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[425984, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 15}, {'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 16}]},
    {'id': 's123', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Remainder', 'inputs': [[3584, 20480], [3584, 128]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[3584, 12288], [3584, 8192], [3584, 64], [3584, 64]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 98}]},
    {'id': 's125', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Remainder', 'inputs': [[3584, 64], [3584, 64], [64], [64]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[3584, 64], [3584, 64]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 100}]},
    {'id': 's127', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Prefill_Remainder', 'inputs': [[229376, 128], [229376, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[229376, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 102}, {'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 89}]},
    {'id': 's129', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Chunk', 'inputs': [[53248, 2560], [53248, 16]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[53248, 1536], [53248, 1024], [53248, 8], [53248, 8]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 12}]},
    {'id': 's131', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Chunk', 'inputs': [[53248, 8], [53248, 8], [8], [8]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[53248, 8], [53248, 8]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 14}]},
    {'id': 's136', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Prefill_Remainder', 'inputs': [[28672, 2560], [28672, 16]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[28672, 1536], [28672, 1024], [28672, 8], [28672, 8]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 85}]},
    {'id': 's138', 'op': 'gdn_fused_gating', 'type': 'gdn_fused_gating', 'stage': 'Prefill_Remainder', 'inputs': [[28672, 8], [28672, 8], [8], [8]], 'dtypes': ['BF16', 'BF16', 'FP32', 'FP32'], 'outputs': [[28672, 8], [28672, 8]], 'output_dtypes': ['FP32', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 87}]},
    {'id': 's140', 'op': 'gdn_qkvzba_split', 'type': 'gdn_qkvzba_split', 'stage': 'Decode', 'inputs': [[192, 10240], [192, 64]], 'dtypes': ['BF16', 'BF16'], 'outputs': [[192, 6144], [192, 4096], [192, 32], [192, 32]], 'output_dtypes': ['BF16', 'BF16', 'BF16', 'BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 11}]},
    {'id': 's143', 'op': 'gdn_rmsnorm_gated', 'type': 'gdn_rmsnorm_gated', 'stage': 'Decode', 'inputs': [[6144, 128], [6144, 128], [128]], 'dtypes': ['BF16', 'BF16', 'BF16'], 'outputs': [[6144, 128]], 'output_dtypes': ['BF16'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'decode_shape', 'row': 14}]},
]


def _bench_1102_cases():
    cases = []
    for shape in _BENCH_1102_SHAPES:
        routes = ('helper',)
        for route in routes:
            cases.append({
                'shape_id': shape['id'], 'source_op': shape['op'], 'route': route,
                'measurement': 'component' if shape['op'] == 'gdn_chunk_delta' else 'operator',
                'input_shapes': json.dumps(shape['inputs'], separators=(',', ':')),
                'input_dtypes': ','.join(shape['dtypes']),
                'output_shapes': json.dumps(shape['outputs'], separators=(',', ':')),
                'output_dtypes': ','.join(shape['output_dtypes']),
                'sources': json.dumps(shape['locations'], separators=(',', ':')),
                'shape': shape,
            })
    return cases


def _bench_1102_case_id(case):
    return f"{case['source_op']}-{case['route']}-{case['shape_id']}"


def _build_bench_1102_runner(case, device='cuda'):
    """Allocate inputs once; return an operator/component invocation for timing."""
    shape = case['shape']
    inputs = shape['inputs']
    dtypes = [_BENCH_1102_DTYPES[name] for name in shape['dtypes']]
    route = case['route']

    def tensor(*dims, dtype=torch.bfloat16, value=None):
        tensor = torch.empty(dims, dtype=dtype, device=device)
        if tensor.device.type != 'meta':
            if value is not None:
                tensor.fill_(value)
            elif dtype == torch.float8_e4m3fn:
                tensor.view(torch.uint8).random_(24, 40)
            else:
                tensor.uniform_(-0.02, 0.02)
        return tensor

    def indices(count):
        return torch.arange(count, dtype=torch.int32, device=device)
    target = importlib.import_module('boltops.fla.gdn')
    args = [tensor(*dims, dtype=dtype) for (dims, dtype) in zip(inputs, dtypes)]
    if shape['op'] == 'gdn_qkvzba_split':
        heads_v = inputs[1][-1] // 2
        width = shape['outputs'][1][-1] // heads_v
        heads_qk = (inputs[0][-1] - 2 * heads_v * width) // (2 * width)
        return partial(target.gdn_qkvzba_split, *args, heads_qk, heads_v, width, width)
    return partial(getattr(target, shape['op']), *args)


@pytest.mark.bench
@pytest.mark.parametrize('case', _bench_1102_cases(), ids=_bench_1102_case_id)
def test_bench_1102_gdn_prefill_helpers_triton(case, bench_warmup, bench_repeat, time_ms):
    if not torch.cuda.is_available():
        pytest.skip('CUDA/HIP device required for benchmarks')
    run = _build_bench_1102_runner(case)
    run()
    latency = time_ms(run, bench_warmup, bench_repeat, dtype=case['input_dtypes'])
    print(f'{_bench_1102_case_id(case)} ({case["measurement"]}): {latency * 1000:.3f} us')
