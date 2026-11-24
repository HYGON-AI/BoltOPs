"""CPU routing checks for the gfx946 scalar scan specialization."""
import importlib

import pytest
import torch


@pytest.mark.parametrize('feature,grouped', [
    ('plain', True), ('head8', True), ('head64', True), ('head64_long', True),
    ('other_device', False), ('short', False), ('other_heads', False),
    ('bf16_input', False), ('bf16_output', False), ('head_first', False),
    ('strided', False), ('varlen', False), ('chunk32', False),
    ('explicit_empty', False), ('explicit_grouped', True),
])
def test_gfx946_scalar_cumsum_dispatch(feature, grouped, monkeypatch):
    backend = importlib.import_module('boltops.fla.common.triton.cumsum')
    calls = []

    class Capture:
        def __init__(self, grouped):
            self.grouped = grouped

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.grouped, grid, kwargs))
            return launch

    monkeypatch.setattr(backend, '_grouped_scalar_device_supported',
                        lambda device: feature != 'other_device')
    monkeypatch.setattr(backend, 'chunk_local_cumsum_scalar_kernel', Capture(False))
    monkeypatch.setattr(backend, 'chunk_local_cumsum_scalar_heads_kernel', Capture(True))
    rows = 2047 if feature == 'short' else (6656 if feature == 'head64_long' else 2048)
    heads = {'head8': 8, 'head64': 64, 'head64_long': 64, 'other_heads': 13}.get(feature, 32)
    dtype = torch.bfloat16 if feature == 'bf16_input' else torch.float32
    source = torch.empty(1, rows, heads * (2 if feature == 'strided' else 1), dtype=dtype)
    if feature == 'strided':
        source = source[..., ::2]
    if feature == 'head_first':
        source = source.transpose(1, 2).contiguous()
    options = dict(head_first=feature == 'head_first')
    if feature == 'varlen':
        options.update(cu_seqlens=torch.tensor([0, rows], dtype=torch.int32),
                       chunk_indices=torch.tensor([[0, 0]], dtype=torch.int32))
    if feature == 'bf16_output':
        options['output_dtype'] = torch.bfloat16
    if feature == 'explicit_empty':
        options['kernel_cfg'] = {}
    if feature == 'explicit_grouped':
        options['kernel_cfg'] = {'BH': 8, 'num_warps': 4}
    chunk = 32 if feature == 'chunk32' else 64
    result = backend.chunk_local_cumsum_scalar(source, chunk, **options)
    assert result.shape == source.shape
    selected, grid, config = calls.pop()
    assert selected is grouped
    if grouped:
        assert config['BH'] == 8 and config['num_warps'] == (1 if feature == 'head64_long' else 4)
        assert grid == ((rows + 63) // 64, heads // 8)


def test_gfx946_scalar_cumsum_device_guard(monkeypatch):
    backend = importlib.import_module('boltops.fla.common.triton.cumsum')
    monkeypatch.setattr(backend.arch_info, 'get_arch', lambda device: 'gfx946')
    assert not backend._grouped_scalar_device_supported(torch.device('cpu'))
    assert backend._grouped_scalar_device_supported(torch.device('cuda'))
    monkeypatch.setattr(backend.arch_info, 'get_arch', lambda device: 'gfx950')
    assert not backend._grouped_scalar_device_supported(torch.device('cuda'))
