import functools
import re

import importlib
import json
from functools import partial

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from boltops.fla.gdn.triton.chunk_wasp_async import (
    chunk_gated_delta_rule_fwd_h_wasp_async,
    chunk_wasp_async_kernel,
)
from boltops.utility import arch_info
from ._gdn_reference import naive_chunk_h


def _wasp_compile_source(block_v, weighted_k_major):
    constants = dict(H=32, HG=16, BV=block_v, USE_INITIAL=True, STATE_ROWS=1,
                     HAS_INDICES=False, IS_VARLEN=False, SAVE_VALUE=True,
                     SAVE_FINAL=True, USE_EXP2=False, indices=None,
                     cu_seqlens=None, chunk_offsets=None, STATE_VK=True, SMALL=False,
                     PREDICTION_KWIDTH=4, WEIGHTED_K_MAJOR=weighted_k_major,
                     PARTITION_REGS=(48, 128) if block_v == 32 else (64, 160))
    pointers = dict(keys='*bf16', weights='*bf16', values='*bf16', gates='*fp32',
                    initial='*fp32', states='*bf16', residuals='*bf16', final='*fp32', T='i32')
    signature = {name: 'constexpr' if name in constants else pointers[name]
                 for name in chunk_wasp_async_kernel.arg_names}
    attributes = {(chunk_wasp_async_kernel.arg_names.index(name),):
                  [('tt.divisibility', 16), ('tt.pointer_range', 32)]
                  for name, kind in pointers.items() if kind.startswith('*')}
    options = dict(num_warps=4, num_stages=2, use_async_copy=True,
                   wasp_partition_warps=(4, 4), wasp_wdra=True,
                   wasp_partition_regs=(48, 128) if block_v == 32 else (64, 160))
    source = GluonASTSource(chunk_wasp_async_kernel, signature, constants, attributes)
    return source, options


@functools.cache
def _wasp_backend_compiles() -> bool:
    """Probe HCU support without hiding real WASP compilation failures."""
    from triton._C import libtriton

    if not hasattr(libtriton, 'hcu'):
        return False
    source, options = _wasp_compile_source(32, False)
    compiled = triton.compile(source, target=GPUTarget('hip', 'gfx946', 64),
                              options=options)
    return bool(compiled.metadata.wasp_enabled)


def _require_wasp_device():
    if not torch.cuda.is_available():
        pytest.skip('HIP device required')
    if arch_info.get_arch(torch.device('cuda:0')) != 'gfx946' or not _wasp_backend_compiles():
        pytest.skip('WASP path requires a gfx946 device with the in-tree '
                   'WASP Triton backend')
@pytest.mark.parametrize('tokens,kv_heads,heads,varlen,expected', [
    # fixed: measured points inside the envelope
    (2048, 16, 32, False, True), (2560, 16, 32, False, True),
    (4096, 16, 32, False, True), (6656, 16, 32, False, True),
    (3584, 16, 64, False, True), (6656, 16, 64, False, True),
    (28672, 2, 8, False, True), (53248, 2, 8, False, True),
    # fixed: envelope generalization (unmeasured but in-bounds)
    (2112, 16, 32, False, True), (2048, 16, 64, False, True),
    (2048, 2, 8, False, True), (2048, 2, 16, False, True),
    # fixed: out of envelope
    (1984, 16, 32, False, False), (2080, 16, 32, False, False),
    (2048, 16, 4, False, False), (1984, 16, 64, False, False),
    # varlen: only the measured workload
    (6656, 16, 32, True, True), (2560, 16, 32, True, False),
    (6656, 16, 64, True, False), (6656, 2, 8, True, False)])
def test_measured_dispatch_scope(tokens, kv_heads, heads, varlen, expected, monkeypatch):
    from types import SimpleNamespace
    from boltops.fla.gdn.triton import chunk_wasp_async as module

    monkeypatch.setattr(module, '_supports_async_wasp', lambda options: True)
    monkeypatch.setattr(module.arch_info, 'get_arch', lambda *args: 'gfx946')
    monkeypatch.setattr(module.arch_info, 'get_num_sms', lambda *args: 48)
    keys = SimpleNamespace(shape=(1, tokens, kv_heads, 128), device='cuda:0')
    values = SimpleNamespace(shape=(1, tokens, heads, 128))
    lengths = SimpleNamespace(numel=lambda: 4) if varlen else None
    config = module._measured_async_config(dict(k=keys, u=values, cu_seqlens=lengths,
        use_exp2=False, output_final_state=True, save_new_value=True))
    assert (config is not None) == expected
    if config is not None:
        assert config['BV'] == 64
        assert config['weighted_k_major'] == (not varlen)
        assert config['wasp_partition_regs'] == ((64, 192) if varlen else (64, 160))


@pytest.mark.parametrize('option,value', [
    ('use_exp2', True), ('transpose_state_layout', False),
    ('inplace_final_state', True), ('null_state_index', 0),
    ('chunk_size', 128), ('output_final_state', False), ('save_new_value', False),
    ('g', None), ('gk', object()), ('kernel_cfg', dict(BK=64)), ('cu_count', 96)])
def test_measured_dispatch_preserves_unsupported_modes(option, value, monkeypatch):
    import math
    from types import SimpleNamespace
    from boltops.fla.gdn.triton import chunk_wasp_async as module

    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
            device='cuda:0', is_cuda=True, is_contiguous=lambda: True,
            numel=lambda: math.prod(shape), element_size=lambda: 2 if dtype == torch.bfloat16 else 4)

    keys = tensor((1, 2048, 16, 128), torch.bfloat16)
    values = tensor((1, 2048, 32, 128), torch.bfloat16)
    bound = module._SIGNATURE.bind(keys, values, values,
        g=tensor((1, 2048, 32), torch.float32))
    bound.apply_defaults()
    monkeypatch.setattr(module.arch_info, 'get_arch', lambda *args: 'gfx946')
    monkeypatch.setattr(module.arch_info, 'get_num_sms', lambda *args: 48)
    assert module._measured_async_config(bound.arguments) is not None
    if option == 'cu_count':
        monkeypatch.setattr(module.arch_info, 'get_num_sms', lambda *args: value)
    else:
        bound.arguments[option] = value
    assert module._measured_async_config(bound.arguments) is None


@pytest.mark.parametrize('config,eligible,architecture,expected', [
    (None, True, 'gfx946', 'async'), (None, False, 'gfx946', 'baseline'),
    ({'BV': 32}, True, 'gfx946', 'baseline'), (None, True, 'gfx90a', 'baseline')])
def test_public_dispatch_respects_explicit_config(config, eligible, architecture, expected, monkeypatch):
    from boltops.fla.gdn.triton import chunk_delta_h, chunk_wasp_async

    calls = []
    tensors = tuple(torch.empty(1) for _ in range(3))

    def try_async(*args, **kwargs):
        calls.append('try')
        return tensors if eligible else None

    monkeypatch.setattr(chunk_wasp_async, '_try_measured_async', try_async)
    monkeypatch.setattr(chunk_delta_h.arch_info, 'get_arch', lambda *args: architecture)
    monkeypatch.setattr(chunk_delta_h, 'launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64',
                        lambda **kwargs: calls.append('baseline'))
    values = torch.empty(1, 2, 1, 1)
    result = chunk_delta_h.chunk_gated_delta_rule_fwd_h(values, values, values, kernel_cfg=config)
    if expected == 'async':
        assert result is tensors
        assert calls == ['try']
    else:
        assert calls == (['try', 'baseline'] if config is None and architecture == 'gfx946'
                         else ['baseline'])
        # The baseline path must own the returned tensors; a dispatch that
        # leaked the async result (e.g. ignoring an explicit kernel_cfg or a
        # non-gfx946 architecture) returns the sentinel above.
        assert result is not tensors


@pytest.mark.parametrize('block_v', [32, 64])
@pytest.mark.parametrize('weighted_k_major', [False, True])
def test_async_wasp_uses_native_matrix_reads(block_v, weighted_k_major):
    if not _wasp_backend_compiles():
        pytest.skip('Triton backend cannot compile WASP partitions')
    source, options = _wasp_compile_source(block_v, weighted_k_major)
    compiled = triton.compile(
        source, target=GPUTarget('hip', 'gfx946', 64), options=options)
    assembly = compiled.asm['amdgcn']
    assert compiled.metadata.wasp_enabled
    assert compiled.metadata.num_warps == 8
    assert tuple(compiled.metadata.wasp_partition_regs_requested) == options['wasp_partition_regs']
    assert compiled.metadata.shared <= 65536
    assert re.search(r'buffer_load.*lds', assembly)
    assert 'ds_read_m32x16_b16' in assembly
    if weighted_k_major:
        assert 'ds_read_u16' not in assembly
    assert not re.search(r'\.[sv]gpr_spill_count:\s+[1-9]', assembly)


@pytest.mark.parametrize('requested', [None, (48, 128), (64, 160)])
def test_async_wasp_forwards_partition_registers(requested, monkeypatch):
    from boltops.fla.gdn.triton import chunk_wasp_async as module

    calls = []

    class Launch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append(kwargs)

    monkeypatch.setattr(module, '_supports_async_wasp', lambda options: True)
    monkeypatch.setattr(module, 'chunk_wasp_async_kernel', Launch())
    keys = torch.empty(1, 128, 2, 128, dtype=torch.bfloat16)
    values = torch.empty(1, 128, 4, 128, dtype=torch.bfloat16)
    config = {} if requested is None else {'wasp_partition_regs': requested}
    module.chunk_gated_delta_rule_fwd_h_wasp_async(
        keys, values, values, g=torch.empty(1, 128, 4), kernel_cfg=config)
    assert len(calls) == 1
    expected = (64, 192) if requested is None else requested
    assert calls[0]['PARTITION_REGS'] == calls[0]['wasp_partition_regs'] == expected


@pytest.mark.parametrize('block_v,varlen,tail', [
    (32, False, False), (64, False, False), (32, True, False), (64, True, False),
    (32, False, True), (64, True, True)])
@pytest.mark.parametrize('weighted_k_major', [False, True])
def test_async_wasp_matches_indexed_recurrence(block_v, varlen, tail, weighted_k_major):
    _require_wasp_device()
    torch.set_num_threads(2)
    torch.manual_seed(20261106)
    lengths = ([65, 129, 64] if tail else [128, 192, 64]) if varlen else [65 if tail else 128]
    tokens = sum(lengths)
    keys = (torch.randn(1, tokens, 2, 128) * 0.05).bfloat16()
    weights = (torch.randn(1, tokens, 4, 128) * 0.05).bfloat16()
    values = torch.randn_like(weights)
    gates = -torch.rand(1, tokens, 4) * 0.04
    begin = 0
    for length in lengths:
        for chunk in range(0, length, 64):
            gates[:, begin + chunk:begin + min(chunk + 64, length)] = gates[
                :, begin + chunk:begin + min(chunk + 64, length)].cumsum(1)
        begin += length
    initial = torch.randn(len(lengths), 4, 128, 128) * 0.01
    indices = torch.tensor(([2, -1, 0] if tail else [2, 0, 1]) if varlen else [0], dtype=torch.int32)
    cu_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int() if varlen else None
    expected = naive_chunk_h(keys, weights, values, g=gates, initial_state=initial,
                             initial_state_indices=indices, cu_seqlens=cu_cpu)
    actual = chunk_gated_delta_rule_fwd_h_wasp_async(
        *[tensor.cuda() for tensor in (keys, weights, values)], g=gates.cuda(),
        initial_state=initial.cuda(), initial_state_indices=indices.cuda(),
        cu_seqlens=cu_cpu.cuda() if varlen else None,
        kernel_cfg=dict(BV=block_v, weighted_k_major=weighted_k_major))
    for result, reference in zip(actual, expected):
        result_cpu = result.cpu().float()
        reference = reference.float()
        torch.testing.assert_close(result_cpu, reference, atol=0.08, rtol=0.08)
        assert (result_cpu - reference).norm() / reference.norm() < 0.005


_BENCH_1102_DTYPES = {'BF16': torch.bfloat16, 'FP16': torch.float16,
                      'FP32': torch.float32, 'FP8E4M3': torch.float8_e4m3fn}


_BENCH_1102_SHAPES = [
    {'id': 's003', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 16, 128], [6656, 16, 128], [6656, 32, 128], [6656, 32], [6656, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[6656, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 14}, {'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 13}]},
    {'id': 's009', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[2560, 16, 128], [2560, 16, 128], [2560, 32, 128], [2560, 32], [2560, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[2560, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_07_q35_fp8_4k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 76}]},
    {'id': 's040', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[2048, 16, 128], [2048, 16, 128], [2048, 32, 128], [2048, 32], [2048, 32], [1, 32, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[2048, 32, 128], [1, 32, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_08_q35_mxfp4_128k_s150.xlsx', 'sheet': 'prefill_shape', 'row': 88}]},
    {'id': 's101', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[6656, 16, 128], [6656, 16, 128], [6656, 64, 128], [6656, 64], [6656, 64], [1, 64, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[6656, 64, 128], [1, 64, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 14}]},
    {'id': 's126', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[3584, 16, 128], [3584, 16, 128], [3584, 64, 128], [3584, 64], [3584, 64], [1, 64, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[3584, 64, 128], [1, 64, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape', 'row': 101}]},
    {'id': 's132', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Chunk', 'inputs': [[53248, 2, 128], [53248, 2, 128], [53248, 8, 128], [53248, 8], [53248, 8], [1, 8, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[53248, 8, 128], [1, 8, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 15}]},
    {'id': 's139', 'op': 'gdn_chunk_delta', 'type': 'gdn_chunk_delta', 'stage': 'Prefill_Remainder', 'inputs': [[28672, 2, 128], [28672, 2, 128], [28672, 8, 128], [28672, 8], [28672, 8], [1, 8, 128, 128]], 'dtypes': ['BF16', 'BF16', 'BF16', 'FP32', 'FP32', 'FP32'], 'outputs': [[28672, 8, 128], [1, 8, 128, 128]], 'output_dtypes': ['BF16', 'FP32'], 'locations': [{'file': 'shape_11_q397b_128k_s2000.xlsx', 'sheet': 'prefill_shape_cp8', 'row': 88}]},
]


def _bench_1102_cases():
    cases = []
    for shape in _BENCH_1102_SHAPES:
        routes = ('wasp',)
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
    (total, heads, width) = inputs[0]
    (value_heads, value_width) = inputs[2][1:]
    gates = tensor(1, total, value_heads, dtype=torch.float32, value=-0.01)
    keys = tensor(1, total, heads, width)
    values = tensor(1, total, value_heads, value_width)
    weights = tensor(1, total, value_heads, width)
    initial = tensor(*inputs[-1], dtype=torch.float32, value=0)
    target = importlib.import_module('boltops.fla.gdn.triton.chunk_wasp_async')
    states = tensor(1, total // 64, value_heads, value_width, width)
    residuals = tensor(1, total, value_heads, value_width)
    final = tensor(*inputs[-1], dtype=torch.float32)
    return partial(target.chunk_wasp_async_kernel[2, value_heads], keys, weights, values, gates, initial, states, residuals, final, None, None, None, total, H=value_heads, HG=heads, BV=64, USE_INITIAL=True, STATE_ROWS=1, HAS_INDICES=False, IS_VARLEN=False, SAVE_VALUE=True, SAVE_FINAL=True, USE_EXP2=False, STATE_VK=True, SMALL=False, PREDICTION_KWIDTH=4, WEIGHTED_K_MAJOR=True, PARTITION_REGS=(64, 160), num_warps=4, num_stages=2, use_async_copy=True, wasp_partition_warps=(4, 4), wasp_partition_regs=(64, 160), wasp_wdra=True)


@pytest.mark.bench
@pytest.mark.parametrize('case', _bench_1102_cases(), ids=_bench_1102_case_id)
def test_bench_1102_gdn_wasp_triton(case, bench_warmup, bench_repeat, time_ms):
    _require_wasp_device()
    run = _build_bench_1102_runner(case)
    run()
    latency = time_ms(run, bench_warmup, bench_repeat, dtype=case['input_dtypes'])
    print(f'{_bench_1102_case_id(case)} ({case["measurement"]}): {latency * 1000:.3f} us')
