"""Opt-in gfx946 GDN recurrence with explicit Triton WASP partitions."""

import inspect

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy

from boltops.utility import arch_info
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from boltops.fla.common.triton.utils import prepare_chunk_indices, prepare_chunk_offsets


_SIGNATURE = inspect.signature(chunk_gated_delta_rule_fwd_h)


@gluon.jit
def _empty_partition():
    pass


@gluon.jit
def _exchange(SMALL: gl.constexpr):
    if SMALL:
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n\ts_ebarrier_sync 14, 3\n\ts_mov_b32 $0, 0",
            constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    else:
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n\ts_ebarrier_sync 14, 8\n\ts_mov_b32 $0, 0",
            constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _ready(BARRIER: gl.constexpr):
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\ts_abarrier_arrive $0, " + BARRIER + ", 1",
        constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _wait(BARRIER: gl.constexpr, phase):
    if phase == 0:
        gl.inline_asm_elementwise(
            "s_abarrier_try_wait $0, " + BARRIER + ", 0",
            constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    else:
        gl.inline_asm_elementwise(
            "s_abarrier_try_wait $0, " + BARRIER + ", 1",
            constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _sequence(BARRIER: gl.constexpr):
    gl.inline_asm_elementwise(
        "s_abarrier_seq " + BARRIER + "\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _produce(keys, weights, values, states, residuals, shared, key_shared, h_shared, u_shared, v_shared, T, cu_seqlens, chunk_offsets, H: gl.constexpr, HG: gl.constexpr, BV: gl.constexpr, SAVE_VALUE: gl.constexpr, IS_VARLEN: gl.constexpr, SMALL: gl.constexpr):
    waves: gl.constexpr = 1 if SMALL else 4
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [waves, 1], [1, 0])
    sequence = gl.program_id(1) // H
    head = gl.program_id(1) % H
    if IS_VARLEN:
        begin = gl.load(cu_seqlens + sequence).to(gl.int32)
        length = (gl.load(cu_seqlens + sequence + 1) - begin).to(gl.int32)
        first_chunk = gl.load(chunk_offsets + sequence).to(gl.int32)
    else:
        begin = sequence * T
        length = T
        first_chunk = sequence * gl.cdiv(T, 64)
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    columns = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    gl.inline_asm_elementwise(
        "s_abarrier_init 0, 4\n\ts_abarrier_init 1, 4\n\ts_abarrier_init 2, 4\n\ts_abarrier_init 3, 4\n\ts_abarrier_init 4, 4\n\ts_abarrier_init 5, 4\n\ts_abarrier_init 6, 4\n\ts_abarrier_init 7, 4\n\ts_abarrier_init 8, 4\n\ts_abarrier_init 9, 4\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    _exchange(False)
    h_rows = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    u_layout: gl.constexpr = gl.BlockedLayout([1,8], [512 // BV, BV // 8], [4,1], [1,0])
    u_rows = gl.arange(0,64,layout=gl.SliceLayout(1,u_layout))
    u_columns = gl.program_id(0) * BV + gl.arange(0,BV,layout=gl.SliceLayout(0,u_layout))
    for chunk in range(gl.cdiv(length, 64)):
        if chunk > 0:
            _wait("1", (chunk - 1) & 1)
        times = chunk * 64 + rows
        _sequence("0")
        async_copy.buffer_load_to_shared(shared, weights,
            (((begin + times[:, None]) * H + head) * 128 + columns[None, :]).to(gl.int32),
            times[:, None] < length, other=0)
        _ready("0")
        if chunk > 0:
            _wait("7", (chunk - 1) & 1)
        _sequence("6")
        u_offsets = (((begin + chunk * 64 + u_rows[:,None]) * H + head) * 128 + u_columns[None,:]).to(gl.int32)
        async_copy.buffer_load_to_shared(u_shared,values,u_offsets,chunk * 64 + u_rows[:,None] < length,other=0)
        _ready("6")
        if chunk > 0:
            _wait("3", (chunk - 1) & 1)
        _sequence("2")
        async_copy.buffer_load_to_shared(key_shared, keys,
            (((begin + times[:, None]) * HG + head // (H // HG)) * 128 + columns[None, :]).to(gl.int32),
            times[:, None] < length, other=0)
        _ready("2")
        if SAVE_VALUE:
            _wait("8", chunk & 1)
            value_tile = v_shared.load(u_layout)
            _ready("9")
            gl.store(residuals + u_offsets,value_tile,chunk * 64 + u_rows[:,None] < length)
        _wait("4", chunk & 1)
        state_tile = h_shared.load(layout)
        _ready("5")
        state_base = ((first_chunk + chunk) * H + head) * 16384
        gl.store(states + state_base + h_rows[:,None] * 128 + columns[None,:], state_tile)


@gluon.jit
def _consume(values, gates, initial, states, residuals, final, indices, shared, key_shared, h_shared, u_shared, v_shared, T, cu_seqlens,
             H: gl.constexpr, BV: gl.constexpr, USE_INITIAL: gl.constexpr,
             HAS_INDICES: gl.constexpr, STATE_ROWS: gl.constexpr,
             IS_VARLEN: gl.constexpr,
             USE_EXP2: gl.constexpr, SAVE_VALUE: gl.constexpr, SAVE_FINAL: gl.constexpr,
             STATE_VK: gl.constexpr, SMALL: gl.constexpr, PREDICTION_KWIDTH: gl.constexpr,
             WEIGHTED_K_MAJOR: gl.constexpr):
    state_waves: gl.constexpr = [1,4] if STATE_VK else [4,1]
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], mmac_layout=gl.amd.MmacLayout.LIT,
        warps_per_cta=state_waves, tiles_per_warp=[1,2])
    prediction_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16,16,16], mmac_layout=gl.amd.MmacLayout.LIT, warps_per_cta=[4,1])
    prediction_a: gl.constexpr = gl.DotOperandLayout(0,prediction_mma,PREDICTION_KWIDTH)
    prediction_b: gl.constexpr = gl.DotOperandLayout(1,prediction_mma,PREDICTION_KWIDTH)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    sequence = gl.program_id(1) // H
    head = gl.program_id(1) % H
    if IS_VARLEN:
        begin = gl.load(cu_seqlens + sequence).to(gl.int32)
        length = (gl.load(cu_seqlens + sequence + 1) - begin).to(gl.int32)
    else:
        begin = sequence * T
        length = T
    if STATE_VK:
        state_keys = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
        state_values = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
        state_elements = state_values[:, None] * 128 + state_keys[None, :]
        state = gl.zeros((BV, 128), gl.float32, layout=mma)
    else:
        state_keys = gl.arange(0, 128, layout=gl.SliceLayout(1, mma))
        state_values = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
        state_elements = state_values[None, :] * 128 + state_keys[:, None]
        state = gl.zeros((128, BV), gl.float32, layout=mma)
    state_offsets = (sequence * H + head) * 16384 + state_elements
    if USE_INITIAL:
        state_index = gl.load(indices + sequence) if HAS_INDICES else sequence
        state = gl.load(initial + (state_index.to(gl.int64) * H + head) * 16384 + state_elements,
                        (state_index >= 0) & (state_index < STATE_ROWS), other=0)
    time_rows = gl.arange(0, 64, layout=gl.SliceLayout(1, prediction_mma))
    value_columns = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, prediction_mma))
    if WEIGHTED_K_MAJOR:
        weighted_layout: gl.constexpr = gl.SwizzledSharedLayout(8,1,8,[1,0])
        weighted_shared = v_shared._reinterpret(gl.bfloat16,[BV,64],weighted_layout)
    _exchange(False)
    for chunk in range(gl.cdiv(length, 64)):
        times = chunk * 64 + time_rows
        gate = gl.load(gates + (begin + times) * H + head, times < length, other=0)
        last_time = gl.minimum((chunk + 1) * 64, length) - 1
        last_gate = gl.load(gates + (begin + last_time) * H + head)
        if chunk > 0:
            _wait("5", (chunk - 1) & 1)
        if STATE_VK:
            h_shared.store(state.to(gl.bfloat16))
        else:
            h_shared.store(state.permute((1,0)).to(gl.bfloat16))
        _ready("4")
        _wait("0", chunk & 1)
        weight_tile = shared.load(prediction_a)
        _ready("1")
        state_operand = h_shared.permute((1,0)).load(prediction_b)
        prediction = gl.amd.cdna3.mfma(
            weight_tile, state_operand, gl.zeros((64, BV), gl.float32, layout=prediction_mma))
        _wait("6", chunk & 1)
        value_tile = u_shared.load(prediction_mma)
        _ready("7")
        residual = value_tile.to(gl.float32) - prediction
        if SAVE_VALUE:
            if chunk > 0:
                _wait("9", (chunk - 1) & 1)
            v_shared.store(residual.to(gl.bfloat16))
            _ready("8")
        difference = last_gate - gate
        difference = gl.where((times < length) & (difference <= 0), difference, float("-inf"))
        if USE_EXP2:
            weighted = residual * gl.exp2(difference)[:, None]
            state = state * gl.exp2(last_gate)
        else:
            weighted = residual * gl.exp2(difference * 1.4426950408889634)[:, None]
            state = state * gl.exp2(last_gate * 1.4426950408889634)
        if SAVE_VALUE:
            _wait("9", chunk & 1)
        if STATE_VK:
            if WEIGHTED_K_MAJOR:
                weighted_shared.store(weighted.permute((1,0)).to(gl.bfloat16))
                weighted_operand = weighted_shared.load(dot_a)
            else:
                v_shared.store(weighted.to(gl.bfloat16))
                weighted_operand = v_shared.permute((1,0)).load(dot_a)
            _wait("2", chunk & 1)
            key_operand = key_shared.load(dot_b)
            _ready("3")
            state = gl.amd.cdna3.mfma(weighted_operand, key_operand, state)
        else:
            v_shared.store(weighted.to(gl.bfloat16))
            weighted_operand = v_shared.load(dot_b)
            _wait("2", chunk & 1)
            key_operand = key_shared.permute((1, 0)).load(dot_a)
            _ready("3")
            state = gl.amd.cdna3.mfma(key_operand, weighted_operand, state)
    if SAVE_FINAL:
        gl.store(final + state_offsets, state)


@gluon.jit
def chunk_wasp_async_kernel(keys, weights, values, gates, initial, states, residuals, final, indices, cu_seqlens, chunk_offsets,
                      T, H: gl.constexpr, HG: gl.constexpr, BV: gl.constexpr,
                      USE_INITIAL: gl.constexpr, USE_EXP2: gl.constexpr,
                      SAVE_VALUE: gl.constexpr, SAVE_FINAL: gl.constexpr,
                      HAS_INDICES: gl.constexpr, STATE_ROWS: gl.constexpr,
                      IS_VARLEN: gl.constexpr,
                      STATE_VK: gl.constexpr = False, SMALL: gl.constexpr = False,
                      PREDICTION_KWIDTH: gl.constexpr = 4,
                      WEIGHTED_K_MAJOR: gl.constexpr = False,
                      PARTITION_REGS: gl.constexpr = (64, 192)):
    layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    shared = gl.allocate_shared_memory(gl.bfloat16, (64, 128), layout)
    matrix_layout: gl.constexpr = gl.SwizzledSharedLayout(8,2,2,[1,0])
    key_shared = gl.allocate_shared_memory(gl.bfloat16, (64, 128), matrix_layout)
    h_shared = gl.allocate_shared_memory(gl.bfloat16, (BV, 128), layout)
    u_layout: gl.constexpr = gl.SwizzledSharedLayout(8,1,BV // 8,[1,0])
    u_shared = gl.allocate_shared_memory(gl.bfloat16,(64,BV),u_layout)
    v_shared = gl.allocate_shared_memory(gl.bfloat16,(64,BV),u_layout)
    if SMALL:
        gl.warp_specialize([
            (_empty_partition, ()),
            (_produce, (keys, weights, values, states, residuals, shared, key_shared, h_shared, u_shared, v_shared, T, cu_seqlens, chunk_offsets, H, HG, BV, SAVE_VALUE, IS_VARLEN, SMALL)),
            (_consume, (values, gates, initial, states, residuals, final, indices, shared, key_shared, h_shared, u_shared, v_shared,
                        T, cu_seqlens, H, BV, USE_INITIAL, HAS_INDICES, STATE_ROWS, IS_VARLEN, USE_EXP2, SAVE_VALUE, SAVE_FINAL, STATE_VK, SMALL, PREDICTION_KWIDTH, WEIGHTED_K_MAJOR)),
        ], [1, 2], [128, 256])
    else:
        gl.warp_specialize([
            (_empty_partition, ()),
            (_produce, (keys, weights, values, states, residuals, shared, key_shared, h_shared, u_shared, v_shared, T, cu_seqlens, chunk_offsets, H, HG, BV, SAVE_VALUE, IS_VARLEN, SMALL)),
            (_consume, (values, gates, initial, states, residuals, final, indices, shared, key_shared, h_shared, u_shared, v_shared,
                        T, cu_seqlens, H, BV, USE_INITIAL, HAS_INDICES, STATE_ROWS, IS_VARLEN, USE_EXP2, SAVE_VALUE, SAVE_FINAL, STATE_VK, SMALL, PREDICTION_KWIDTH, WEIGHTED_K_MAJOR)),
        ], [4, 4], PARTITION_REGS)


def _supports_async_wasp(options):
    k, w, u = options['k'], options['w'], options['u']
    gate = options['g']
    initial = options['initial_state']
    indices = options['initial_state_indices']
    lengths = options['cu_seqlens']
    config = options['kernel_cfg'] or {}
    supported = (
        k.ndim == w.ndim == u.ndim == 4 and k.shape[-1] == u.shape[-1] == 128
        and k.shape[:2] == w.shape[:2] == u.shape[:2]
        and w.shape[2:] == (u.shape[2], 128) and k.shape[0] > 0 and k.shape[1] > 0
        and k.shape[2] > 0 and u.shape[2] % k.shape[2] == 0
        and all(tensor.is_cuda and tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
                and tensor.device == k.device for tensor in (k, w, u))
        and gate is not None and gate.shape == u.shape[:3] and gate.dtype == torch.float32
        and gate.device == k.device and gate.is_contiguous()
        and options['gk'] is None and options['chunk_size'] == 64
        and options['transpose_state_layout'] and not options['inplace_final_state']
        and options['null_state_index'] == -1 and config.get('BK', 128) == 128
        and config.get('BV', 64) in (32, 64)
        and config.get('num_warps', 4) == 4 and config.get('num_stages', 2) == 2
        and config.get('prediction_kwidth', 4) in (4, 8)
        and isinstance(config.get('weighted_k_major', False), bool)
        and all(tensor.numel() * tensor.element_size() < 2**31 for tensor in (k, w, u))
        and (initial is None or initial.numel() * initial.element_size() < 2**31)
        and (initial is None or (initial.ndim == 4 and initial.shape[1:] == (u.shape[2], 128, 128)
             and initial.dtype == torch.float32 and initial.device == k.device and initial.is_contiguous()))
        and (indices is None or (initial is not None and indices.ndim == 1
             and indices.dtype in (torch.int32, torch.int64) and indices.is_contiguous()
             and indices.device == k.device))
        and (lengths is None or (k.shape[0] == 1 and lengths.ndim == 1
             and lengths.numel() >= 2 and lengths.dtype in (torch.int32, torch.int64)
             and lengths.device == k.device and lengths.is_contiguous())))
    return supported and arch_info.get_arch(k.device) == 'gfx946'


def _measured_async_config(options):
    if not _supports_async_wasp(options):
        return None
    keys, values = options['k'], options['u']
    if arch_info.get_num_sms(keys.device) != 48:
        return None
    lengths = options['cu_seqlens']
    # Fixed layout: generalized measured envelope. WASP won at every
    # measured point (1.25x-2.03x over the faster BV32/BV64 baseline):
    # H=32 at T=2048/2560/4096/6656, H=64 at T=3584/6656, H=8 at
    # T=28672/53248; the envelope below was spot-checked at T=4096
    # (1.40x). T and H are only lower-bounded because the original
    # recurrence degrades linearly in both while the WASP pipeline
    # amortizes its setup. Varlen keeps the single measured workload.
    if (keys.shape[0] != 1 or options['use_exp2']
            or not options['output_final_state']
            or not options['save_new_value']):
        return None
    if lengths is None:
        if (keys.shape[1] % 64 or keys.shape[1] < 2048
                or values.shape[2] < 8):
            return None
    elif (values.shape[2] != 32 or keys.shape[2] != 16
            or keys.shape[1] != 6656 or lengths.numel() != 4):
        return None
    return dict(BK=128, BV=64, num_warps=4, num_stages=2,
                prediction_kwidth=4, weighted_k_major=lengths is None,
                wasp_partition_regs=(64, 160) if lengths is None else (64, 192))


def _try_measured_async(k, w, u, **kwargs):
    bound = _SIGNATURE.bind(k, w, u, **kwargs)
    bound.apply_defaults()
    config = _measured_async_config(bound.arguments)
    if config is None:
        return None
    kwargs['kernel_cfg'] = config
    return chunk_gated_delta_rule_fwd_h_wasp_async(k, w, u, **kwargs)


def chunk_gated_delta_rule_fwd_h_wasp_async(k, w, u, **kwargs):
    bound = _SIGNATURE.bind(k, w, u, **kwargs)
    bound.apply_defaults()
    options = bound.arguments
    config = options['kernel_cfg'] or {}
    if not _supports_async_wasp(options):
        return chunk_gated_delta_rule_fwd_h(k, w, u, **kwargs)
    initial = options['initial_state']
    indices = options['initial_state_indices']
    lengths = options['cu_seqlens']
    gate = options['g']
    batch, tokens = k.shape[:2]
    heads = u.shape[2]
    if lengths is None:
        sequences = batch
        chunks = triton.cdiv(tokens, 64)
        offsets = None
    else:
        sequences = lengths.numel() - 1
        chunk_indices = options['chunk_indices']
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(lengths, 64)
        chunks = chunk_indices.shape[0]
        offsets = prepare_chunk_offsets(lengths, 64)
    if indices is not None and indices.numel() != sequences:
        raise ValueError('initial_state_indices must have one entry per sequence')
    if max(batch * chunks * heads * 16384 * 2, sequences * heads * 16384 * 4) >= 2**31:
        return chunk_gated_delta_rule_fwd_h(k, w, u, **kwargs)
    states = k.new_empty(batch, chunks, heads, 128, 128)
    residuals = torch.empty_like(u) if options['save_new_value'] else None
    final = torch.empty((sequences, heads, 128, 128), device=k.device,
                        dtype=torch.float32) if options['output_final_state'] else None
    block_v = config.get('BV', 64)
    compiler_options = dict(num_warps=4, num_stages=2, use_async_copy=True,
                            wasp_partition_warps=(4, 4), wasp_wdra=True,
                            wasp_partition_regs=tuple(config.get('wasp_partition_regs', (64, 192))))
    chunk_wasp_async_kernel[(128 // block_v, sequences * heads)](
        k, w, u, gate, initial, states, residuals, final, indices, lengths, offsets, tokens,
        H=heads, HG=k.shape[2], BV=block_v, USE_INITIAL=initial is not None,
        STATE_ROWS=0 if initial is None else initial.shape[0], HAS_INDICES=indices is not None,
        IS_VARLEN=lengths is not None, SAVE_VALUE=residuals is not None,
        SAVE_FINAL=final is not None, USE_EXP2=options['use_exp2'],
        STATE_VK=True, SMALL=False, PREDICTION_KWIDTH=config.get('prediction_kwidth', 4),
        WEIGHTED_K_MAJOR=config.get('weighted_k_major', False),
        PARTITION_REGS=compiler_options['wasp_partition_regs'],
        **compiler_options)
    return states, residuals, final
