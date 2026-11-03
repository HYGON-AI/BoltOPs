# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Correctness tests for fused MoE quantization helpers."""

import pytest
import torch

triton = pytest.importorskip("triton")

from boltops.fused_moe import fused_experts
from boltops.utility import arch_info
from boltops.fused_moe.triton.moe_compat import _upcast_e8m0_to_fp32
from boltops.fused_moe.triton.quant import dynamic_per_group_quant_fp8_power2
from boltops.fused_moe.triton.mxfp8 import mxfp8_e4m3_quantize
from boltops.fused_moe.triton.mxfp4_quant import (
    per_token_group_quant_fp8_mxfp4,
    silu_mul_quant_fp8_packed_triton,
)


from boltops.fused_moe.triton.quant import (  # noqa: E402
    dynamic_mxfp4_quant,
    dynamic_per_token_quant_fp8_i8,
    static_per_tensor_quant_fp8_i8,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
MXFP_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None
    or arch_info.get_arch() != "gfx946",
    reason="native MXFP tests require gfx946",
)


@GPU_REQUIRED
@pytest.mark.parametrize("shape", [(2, 128), (257, 128), (1281, 1232)])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.int8])
def test_dynamic_per_token_quant_zero_rows(shape, quant_dtype):
    """Zero rows stay finite on the 1-D, N=128, and four-row paths."""
    torch.manual_seed(20)
    rows, cols = shape
    x = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    zero_rows = torch.tensor(
        sorted({0, rows // 2, rows - 1}), dtype=torch.long, device=DEVICE
    )
    x[zero_rows] = 0

    quantized = torch.empty(shape, dtype=quant_dtype, device=DEVICE)
    scales = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    dynamic_per_token_quant_fp8_i8(quantized, x, scales)

    dtype_max = (
        torch.finfo(quant_dtype).max
        if torch.is_floating_point(quantized)
        else torch.iinfo(quant_dtype).max
    )
    expected_zero_scale = torch.tensor(
        1.0e-10 / dtype_max, dtype=torch.float32, device=DEVICE
    )

    assert torch.isfinite(scales).all()
    assert torch.isfinite(quantized.float()).all()
    torch.testing.assert_close(
        scales[zero_rows], expected_zero_scale.expand_as(scales[zero_rows])
    )
    torch.testing.assert_close(
        quantized[zero_rows].float(), torch.zeros_like(x[zero_rows].float())
    )


@GPU_REQUIRED
@pytest.mark.parametrize("tensor_name", ["x_in", "qx", "scale_out"])
def test_dynamic_per_token_quant_rejects_noncontiguous_tensors(tensor_name):
    values = torch.randn((4, 128), dtype=torch.float16, device=DEVICE)
    quantized = torch.empty(values.shape, dtype=torch.float8_e4m3fn, device=DEVICE)
    scales = torch.empty(values.shape[0], dtype=torch.float32, device=DEVICE)
    if tensor_name == "x_in":
        values = torch.randn((4, 256), dtype=values.dtype, device=DEVICE)[:, ::2]
    elif tensor_name == "qx":
        quantized = torch.empty((4, 256), dtype=quantized.dtype, device=DEVICE)[
            :, ::2
        ]
    else:
        scales = torch.empty(8, dtype=scales.dtype, device=DEVICE)[::2]

    with pytest.raises(AssertionError, match=f"{tensor_name} must be contiguous"):
        dynamic_per_token_quant_fp8_i8(quantized, values, scales)


def _int8_rounding_input(rows, cols):
    values = torch.zeros((rows, cols), device=DEVICE, dtype=torch.float16)
    values[:, :6] = torch.tensor(
        [0.5, -0.5, 1.5, -1.5, 126.5, 127.0],
        device=DEVICE,
        dtype=torch.float16,
    )
    return values


@GPU_REQUIRED
@pytest.mark.parametrize("shape", [(1, 8), (256, 128)])
def test_dynamic_per_token_int8_rounds_to_nearest(shape):
    values = _int8_rounding_input(*shape)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(shape[0], device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, values.round().to(torch.int8))


@GPU_REQUIRED
def test_dynamic_per_token_int8_tiled_path_rounds_to_nearest():
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    rows = max(1025, 16 * num_cus)
    values = _int8_rounding_input(rows, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(rows, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized, values.round().to(torch.int8))


@GPU_REQUIRED
def test_dynamic_per_token_fp8_keeps_fractional_values():
    values = torch.tensor(
        [[0.5, 1.5, 2.5, 448.0]], device=DEVICE, dtype=torch.float16
    )
    quantized = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, device=DEVICE, dtype=torch.float32)

    dynamic_per_token_quant_fp8_i8(quantized, values, scales)

    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(quantized.float(), values.float())


@GPU_REQUIRED
def test_static_per_tensor_int8_rounds_to_nearest():
    values = _int8_rounding_input(2, 8)
    quantized = torch.empty_like(values, dtype=torch.int8)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)

    static_per_tensor_quant_fp8_i8(quantized, values, scale)

    torch.testing.assert_close(quantized, values.round().to(torch.int8))


@MXFP_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32), (5, 96)])
def test_dynamic_mxfp4_quant_gfx946(dtype, shape):

    # Every block contains 6, so its expected E8M0 scale is known exactly.
    values = torch.tensor([
        -0., 0., .25, -.25, .75, -.75, 1.25, -1.25,
        1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., -5.,
        .5, -.5, 1., -1., 1.5, -1.5, 2., -2.,
        3., -3., 4., -4., 6., -6., .125, -.125,
    ], dtype=torch.float32)
    rows, cols = shape
    exponents = torch.arange(cols // 32, dtype=torch.int32) * 4 - 4
    scale = torch.exp2(exponents.float()).repeat_interleave(32)
    x = (values.repeat(cols // 32) * scale).expand(rows, -1).to(dtype).contiguous()
    packed, scales = dynamic_mxfp4_quant(x.to(DEVICE))

    normalized = x.float() / scale
    thresholds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    codes = (normalized.abs()[..., None] >= thresholds).sum(-1).to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    expected = codes[:, 0::2] | (codes[:, 1::2] << 4)
    torch.testing.assert_close(packed.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(scales.cpu(), (exponents + 127).to(torch.uint8).expand(rows, -1), atol=0, rtol=0)


def _swiglu_fp8_cpu_reference(x, group_size, clamp_limit=None, alpha=1., beta=0.):
    gate, up = x.float().chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(-clamp_limit, clamp_limit)
    # This BF16 boundary is part of the pinned vLLM algorithm, even for FP16 input.
    glu = gate / (1. + torch.exp(-alpha * gate))
    y = ((up + beta) * glu).to(torch.bfloat16).float()
    groups = y.reshape(x.shape[0], -1, group_size)
    raw_scale = (groups.abs().amax(-1) / 448.).clamp_min(1.e-10)
    exponent = torch.ceil(torch.log2(raw_scale))
    scale = torch.exp2(exponent)
    q = (groups / scale[..., None]).clamp(-448., 448.).to(torch.float8_e4m3fn).reshape(y.shape)
    scale_byte = (exponent + 127).clamp(0, 255).to(torch.int64)
    group_count = groups.shape[1]
    packed_groups = ((group_count + 3) // 4) * 4
    # Unused groups in the last pack are zeros entering the quantizer, whose
    # scale floor is 1.e-10 -> ceil(log2(...))=-33 -> encoded exponent 94.
    padded = torch.full((x.shape[0], packed_groups), 94, dtype=torch.int64)
    padded[:, :group_count] = scale_byte
    bytes4 = padded.reshape(x.shape[0], -1, 4)
    packed = sum(bytes4[..., i] << (8*i) for i in range(4)).to(torch.int32)
    return q, packed


def _assert_packed_fp8_result(q, packed, expected_q, expected_packed, *, exact=False):
    assert q.dtype == torch.float8_e4m3fn
    assert packed.dtype == torch.int32
    # Scale bytes/layout are part of the algorithm: no numeric tolerance.
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    actual = q.float()
    expected = expected_q.float()
    if exact:
        torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
    else:
        # CPU exp and the target's exp approximation may straddle a BF16
        # midpoint. Bound that effect by one E4M3 step; do not use a loose
        # relative tolerance near zero/subnormal quantized values.
        step = torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(2.**-6))) - 3)
        assert torch.all((actual - expected).abs() <= step), (
            f'FP8 error exceeds one quantization step; max={(actual-expected).abs().max().item()}')
    assert torch.isfinite(actual).all()


@GPU_REQUIRED
@pytest.mark.parametrize('rows,hidden,group,pattern', [
    (1, 128, 128, 'zeros'),
    (3, 640, 128, 'random'),     # non-four group count / partial scale pack
    (4, 1152, 128, 'random'),    # >8 groups / multiple column CTAs
    (513, 128, 128, 'random'),   # BM=4 launch branch, tail row
    (3, 160, 32, 'random'),      # group's smaller launch branch
    (2, 512, 128, 'large'),     # signed packed int32 words
])
def test_swiglu_fp8_packed_cpu_golden(rows, hidden, group, pattern):
    torch.manual_seed(318)
    x = torch.randn(rows, hidden*2, dtype=torch.bfloat16)
    if pattern == 'zeros':
        x.zero_()
    elif pattern == 'large':
        x.fill_(1024.)
    expected_q, expected_s = _swiglu_fp8_cpu_reference(x, group)
    q, s = silu_mul_quant_fp8_packed_triton(x.cuda(), group_size=group, use_mxfp8_quant=False)
    assert s.shape == (rows, (hidden//group+3)//4)
    assert s.stride(0) == 1 and s.stride(1) == ((rows+3)//4)*4
    _assert_packed_fp8_result(q.cpu(), s.cpu(), expected_q, expected_s, exact=pattern != 'random')


@GPU_REQUIRED
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_swiglu_fp8_clamp_alpha_beta_output_buffer(dtype):
    torch.manual_seed(97)
    x = (torch.randn(3, 512) * 9).to(dtype)
    expected_q, expected_s = _swiglu_fp8_cpu_reference(x, 128, clamp_limit=7., alpha=1.702, beta=1.)
    output_cpu = torch.full((3, 256), 3.).to(torch.float8_e4m3fn)
    output_gpu = output_cpu.cuda()
    q, s = silu_mul_quant_fp8_packed_triton(
        x.cuda(), group_size=128, output_q=output_gpu, clamp_limit=7., alpha=1.702, beta=1.)
    assert q.data_ptr() == output_gpu.data_ptr()
    _assert_packed_fp8_result(q.cpu(), s.cpu(), expected_q, expected_s)


@GPU_REQUIRED
def test_swiglu_fp8_expert_alignment_gaps_and_empty_expert():
    torch.manual_seed(59)
    rows, hidden = 11, 128
    x = torch.randn(rows, hidden*2, dtype=torch.bfloat16)
    ends = torch.tensor([3, 8, 11], dtype=torch.int32)
    # Live expert ranges [0,3), [8,8), [8,11); [3,8) is padding.
    live = torch.tensor([0, 1, 2, 8, 9, 10], dtype=torch.int64)
    output_cpu = torch.full((rows, hidden), 3.).to(torch.float8_e4m3fn)
    expected_q, expected_s = _swiglu_fp8_cpu_reference(x, 128)
    q, s = silu_mul_quant_fp8_packed_triton(
        x.cuda(), group_size=128, output_q=output_cpu.cuda(), expert_ends=ends.cuda(), expert_alignment=8)
    q_cpu, s_cpu = q.cpu(), s.cpu()
    # Torch CPU does not implement advanced indexing for float8 tensors.
    # Select the underlying bytes to preserve the quantized values exactly.
    live_q = q_cpu.view(torch.uint8)[live].view(torch.float8_e4m3fn)
    live_expected = expected_q.view(torch.uint8)[live].view(torch.float8_e4m3fn)
    _assert_packed_fp8_result(live_q, s_cpu[live], live_expected, expected_s[live])
    torch.testing.assert_close(q_cpu[3:8].float(), output_cpu[3:8].float(), rtol=0, atol=0)


@GPU_REQUIRED
def test_swiglu_fp8_bf16_rounding_boundary():
    # Construct a stable case that distinguishes the upstream BF16 boundary
    # from directly quantizing the FP32 activation. The anchor fixes scale=1.
    gates = torch.linspace(-4., 4., 257).bfloat16().float()
    ups = torch.linspace(.125, 4., 257).bfloat16().float()
    raw = gates[:, None] / (1. + torch.exp(-gates[:, None])) * ups[None, :]
    rounded = raw.bfloat16().float()
    different = raw.to(torch.float8_e4m3fn).view(torch.uint8) != rounded.to(torch.float8_e4m3fn).view(torch.uint8)
    # Keep away from BF16 round midpoints so CPU/target exp approximation
    # cannot explain a different expected FP8 code.
    different &= (raw - rounded).abs() > 1.e-5
    gi, ui = different.nonzero()[0].tolist()
    x = torch.empty(1, 256, dtype=torch.bfloat16)
    x[:, :128] = gates[gi]
    x[:, 128:] = ups[ui]
    x[0, 127] = 1024.
    x[0, 255] = .4375
    expected_q, expected_s = _swiglu_fp8_cpu_reference(x, 128)
    no_bf16 = raw[gi, ui].to(torch.float8_e4m3fn).view(torch.uint8)
    assert expected_q.view(torch.uint8)[0, 0] != no_bf16
    q, s = silu_mul_quant_fp8_packed_triton(x.cuda(), group_size=128)
    _assert_packed_fp8_result(q.cpu(), s.cpu(), expected_q, expected_s, exact=True)


@pytest.mark.parametrize('on_device', [False, True])
def test_checkpoint_scale_preparation_matches_upstream_bits(on_device):
    # Match the actual vLLM helper, including its bit-shift boundary behavior.
    # This helper is not the native MXFP4 scale decoder.
    encoded = torch.tensor([0, 1, 126, 127, 128, 254, 255], dtype=torch.uint8)
    expected = torch.tensor([0, 0x00800000, 0x3f000000, 0x3f800000,
                             0x40000000, 0x7f000000, 0x7f800000], dtype=torch.int32)
    prepared = _upcast_e8m0_to_fp32(encoded.cuda() if on_device else encoded)
    assert torch.equal(prepared.cpu().view(torch.int32), expected)


@pytest.mark.parametrize('name', ['w1_scale', 'w2_scale', 'a1_scale', 'a2_scale'])
@pytest.mark.parametrize('storage', ['uint8', 'int32', 'float8_e8m0fnu'])
def test_public_fp8_rejects_unprepared_checkpoint_scales(name, storage):
    if not hasattr(torch, storage):
        pytest.skip(f'Torch has no {storage} dtype')
    scale = torch.tensor([127], dtype=torch.uint8)
    if storage == 'int32':
        scale = torch.tensor([0x7f7f7f7f], dtype=torch.int32)
    elif storage != 'uint8':
        scale = scale.view(getattr(torch, storage))
    with pytest.raises(ValueError, match=name + ' must be a decoded'):
        fused_experts(
            torch.zeros(1, 128, dtype=torch.bfloat16),
            torch.zeros(1, 256, 128).to(torch.float8_e4m3fn),
            torch.zeros(1, 128, 128).to(torch.float8_e4m3fn),
            torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32),
            use_fp8_w8a8=True, **{name: scale})


@GPU_REQUIRED
@pytest.mark.parametrize('width', [32, 128, 256])
def test_group_fp8_power2_matches_vllm_quantization_order(width):
    torch.manual_seed(225)
    x = torch.randn(19, width)
    x[0].zero_()
    # Put group maxima immediately around power-of-two scale transitions.
    boundaries = torch.tensor([448 * 2.**e for e in range(-9, 8)])
    x[1:18] *= .01
    x[1:18, 0] = boundaries
    x[18, 0] = torch.nextafter(boundaries[8], torch.tensor(float('inf')))
    maximum = x.abs().amax(-1).clamp_min(1.e-10)
    raw = maximum * (1.0 / 448.0)
    scales = torch.exp2(torch.ceil(torch.log2(raw)))
    expected = (x / scales[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = torch.empty(x.shape, device='cuda', dtype=torch.float8_e4m3fn)
    s = torch.empty(x.shape[0], device='cuda')
    dynamic_per_group_quant_fp8_power2(q, x.cuda(), s)
    assert torch.equal(s.cpu(), scales)
    assert torch.equal(q.cpu().view(torch.uint8), expected.view(torch.uint8))


@GPU_REQUIRED
def test_power2_scale_storage_preserves_exponent_across_fp32_range():
    # For maximum=2**e, ceil(log2(maximum/448)) is e-8. Cover all
    # reachable normal scale binades above the quantizer's amax floor.
    exponents = torch.arange(-33, 128, dtype=torch.float32)
    maximum = torch.exp2(exponents)
    x = torch.zeros(len(exponents) + 2, 128)
    x[:-2, 0] = maximum
    x[:-2, 1] = -maximum
    x[-1, 0] = torch.finfo(torch.float32).tiny
    expected = torch.cat((torch.exp2(exponents - 8), torch.full((2,), 2.**-42)))
    q = torch.empty(x.shape, device='cuda', dtype=torch.float8_e4m3fn)
    s = torch.empty(x.shape[0], device='cuda')
    dynamic_per_group_quant_fp8_power2(q, x.cuda(), s)
    actual = s.cpu()
    assert torch.equal(actual, expected)
    assert torch.equal(actual.view(torch.int32) & 0x7fffff,
                       torch.zeros(actual.numel(), dtype=torch.int32))


@MXFP_REQUIRED
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_mxfp4_native_quantization_scale_floor_and_exponent_boundaries(dtype):
    emin, emax = (-20, 6) if dtype == torch.float16 else (-42, 118)
    maxima = 448 * torch.exp2(torch.arange(emin, emax + 1, dtype=torch.float32))
    maxima = torch.cat((torch.zeros(1), torch.full((1,), 1.e-20), maxima,
                        torch.nextafter(maxima, torch.zeros_like(maxima)),
                        torch.nextafter(maxima, torch.full_like(maxima, float('inf')))))
    x = torch.zeros(maxima.numel(), 128, dtype=dtype)
    x[:, 0] = maxima.to(dtype)
    x[:, 1] = -maxima.to(dtype)
    # C++ divides in float32 before flooring the scale, then rounds upward
    # using exponent/mantissa bits. CPU double division models div.rn.f32.
    maximum = x.float().abs().amax(-1).clamp_min(1.e-10)
    raw = (maximum.double() / 448).float().clamp_min(1.e-10)
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    expected = (x.float() * (1. / scale[:, None])).clamp(-448, 448).to(torch.float8_e4m3fn)
    q, s = per_token_group_quant_fp8_mxfp4(x.cuda(), group_size=128)
    assert torch.equal(s.cpu().flatten(), exponent.to(torch.uint8))
    assert torch.equal(q.cpu().view(torch.uint8), expected.view(torch.uint8))


@MXFP_REQUIRED
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('hidden', [32, 160, 512, 2048])
def test_mxfp4_aiter_g32_quantization(fused, hidden):
    """CPU oracle for AITER's power-of-two amax rounding, including zero groups."""
    torch.manual_seed(83)
    x = (torch.randn(3, hidden * (2 if fused else 1)) * .5).bfloat16()
    x[0].zero_()
    if fused:
        gate, up = x.float().chunk(2, -1)
        values = up * gate / (1 + torch.exp(-gate))
    else:
        values = x.float()
    groups = values.reshape(3, -1, 32)
    amax = groups.abs().amax(-1)
    rounded = ((amax.view(torch.int32) + 0x200000) & -0x800000).view(torch.float32)
    exponent = (torch.floor(torch.log2(rounded)) - 8).clamp(-127, 127)
    expected_scales = (exponent + 127).to(torch.uint8)
    expected_q = (groups * torch.exp2(-exponent)[..., None]).to(torch.float8_e4m3fn)
    if fused:
        q, scale = silu_mul_quant_fp8_packed_triton(
            x.cuda(), group_size=32, use_mxfp8_quant=True)
    else:
        q, scale = per_token_group_quant_fp8_mxfp4(x.cuda(), group_size=32)
    assert scale.dtype == torch.uint8
    assert torch.equal(scale.cpu(), expected_scales)
    torch.testing.assert_close(q.cpu().float(), expected_q.reshape(3, hidden).float(),
                               rtol=0., atol=0.)


@MXFP_REQUIRED
def test_mxfp8_native_vllm_ceil_quantization():
    torch.manual_seed(681)
    x = torch.randn(4, 128)
    x[0].zero_()
    x[1] *= 1.e-12
    x[2] *= 1.e-30
    groups = x.reshape(4, 4, 32)
    amax = groups.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (torch.ceil(torch.log2(amax / 448.)) + 127).clamp(0, 254)
    expected_q = (groups * torch.exp2(127-exponent)[..., None]).to(torch.float8_e4m3fn).reshape(x.shape)
    q, scale = mxfp8_e4m3_quantize(x.cuda())
    torch.testing.assert_close(scale.cpu(), exponent.to(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(q.cpu().float(), expected_q.float(), rtol=0, atol=0)
