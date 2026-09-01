"""Indexer RMS/LayerNorm + partial NeoX RoPE, and the RoPE cache builder.

Vendored from Step-4 DSA handoff ``inference/kernel.py``
(``dsa_kernels.indexer_norm_rope``, ``dsa_attention.build_rope_cache``).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

@triton.jit
def _affine_normalize(
    raw,
    mean,
    rstd,
    weight_ptr,
    bias_ptr,
    offsets,
    mask,
    weight_bias,
    HAS_BIAS: tl.constexpr,
):
    """Apply ``(x - mean) * rstd * (w + weight_bias) [+ b]`` to one dim slice.

    A device function rather than a closure because Triton cannot compile nested
    ``def`` inside a kernel, and the normalization has to be applied to three different
    dim slices of the same head: the tail, and each half of the rotated span.
    """
    scale = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (raw - mean) * rstd * (scale + weight_bias)
    if HAS_BIAS:
        result += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    return result


@triton.jit
def _indexer_norm_rope_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_x_token,
    stride_out_token,
    stride_cos,
    eps,
    weight_bias,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Normalize one head of one token, then rotate its leading ``2 * rotary_dim`` dims.

    ``SUBTRACT_MEAN`` selects LayerNorm over RMSNorm. The indexer needs both: q is
    RMSNorm with a zero-centered weight, k is LayerNorm with a weight and a bias. They
    differ by one reduction and are trivial to confuse, which would shift every key
    without changing any shape.

    The rotated span is written from a separate load rather than by patching the
    normalized vector in place, because Triton cannot index a value tensor by another
    tensor -- and overlapping stores from one program would depend on store ordering the
    compiler does not promise.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = token * stride_x_token + head * head_dim
    out_base = token * stride_out_token + head * head_dim

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    values = tl.load(x_ptr + base + dims, mask=dim_ok, other=0.0).to(tl.float32)

    sum_sq = tl.sum(values * values)
    if SUBTRACT_MEAN:
        # Match the deployed LayerNorm reduction and fast-rsqrt path.  Computing
        # ``mean((x - mean) ** 2)`` is algebraically equivalent, but changes the
        # FP32 reduction/rounding before every sparse-indexer key.
        mean = tl.sum(values) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = 0.0
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)

    # Dims past the rotated span keep the plain normalized value.
    tl.store(
        out_ptr + out_base + dims,
        _affine_normalize(
            values,
            mean,
            rstd,
            weight_ptr,
            bias_ptr,
            dims,
            dim_ok,
            weight_bias,
            HAS_BIAS=HAS_BIAS,
        ).to(out_ptr.dtype.element_ty),
        mask=dim_ok & (dims >= 2 * rotary_dim),
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(cos_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + position * stride_cos + half, mask=half_ok, other=0.0).to(tl.float32)

    real = _affine_normalize(
        tl.load(x_ptr + base + half, mask=half_ok, other=0.0).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)
    imaginary = _affine_normalize(
        tl.load(x_ptr + base + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32),
        mean,
        rstd,
        weight_ptr,
        bias_ptr,
        rotary_dim + half,
        half_ok,
        weight_bias,
        HAS_BIAS=HAS_BIAS,
    ).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + out_base + half,
        (real * cos - imaginary * sin).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )
    tl.store(
        out_ptr + out_base + rotary_dim + half,
        (real * sin + imaginary * cos).to(out_ptr.dtype.element_ty),
        mask=half_ok,
    )


def _norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    weight_bias: float,
    subtract_mean: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _indexer_norm_rope_kernel[(x.shape[0], num_heads)](
        x,
        out,
        weight,
        bias if bias is not None else weight,
        cos,
        sin,
        positions,
        x.stride(0),
        out.stride(0),
        cos.stride(0),
        eps,
        weight_bias,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        SUBTRACT_MEAN=subtract_mean,
        HAS_BIAS=bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
    )
    return out


def indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the indexer's q/k/z from their fused projection.

    Args:
        index_q: ``[tokens, num_q_heads * head_dim]``.
        index_k: ``[tokens, num_k_heads * head_dim]``.
        index_z: ``[tokens, num_k_heads * head_dim]`` -- returned untouched.
        cos, sin: ``[seq, rotary_dim]``, half the width of the rotated span.
        q_norm_weight_bias: ``1.0`` for a zero-centered RMSNorm weight (``x * (1 + w)``),
            ``0.0`` otherwise. This follows the checkpoint's norm class, so it must be
            threaded through from config rather than hardcoded.
    """
    return (
        _norm_rope(
            index_q,
            q_norm_weight,
            None,
            cos,
            sin,
            positions,
            num_heads=num_q_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=q_norm_weight_bias,
            subtract_mean=False,
        ),
        _norm_rope(
            index_k,
            k_norm_weight,
            k_norm_bias,
            cos,
            sin,
            positions,
            num_heads=num_k_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            weight_bias=0.0,
            subtract_mean=True,
        ),
        index_z.contiguous(),
    )



def build_rope_cache(
    *,
    rotary_span: int,
    theta: float,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for a NEOX-style partial rotation.

    Args:
        rotary_span: how many of the head's dimensions rotate. The returned tables are
            ``[max_position, rotary_span // 2]`` -- *half* the span, because each frequency
            drives the pair ``(d, d + rotary_span // 2)``. The kernels take that halved
            width as their ``rotary_dim``, which is easy to misread as the span itself.
        scaling: the ``rope_scaling`` block, or ``None``. Only ``llama3`` is implemented,
            because it is the only type step4's config asks for; an unrecognised type
            raises rather than silently falling back to unscaled, which would look like a
            long-context quality regression rather than a bug.
    """
    if rotary_span % 2:
        raise ValueError(f"rotary span must be even, got {rotary_span}")
    half = rotary_span // 2
    # Match vLLM's RotaryEmbedding cache construction exactly.  In particular this must
    # stay float32 throughout, even though computing the table in float64 looks more
    # accurate in isolation.  The deployed model rounds the float32 frequencies/trig
    # results to the requested cache dtype; using float64 changes BF16 table entries as
    # early as position 24 (and therefore changes Q/K before the first long-context
    # feature is involved).
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_span, 2, device=device, dtype=torch.float32) / rotary_span)
    )
    if scaling:
        inv_freq = _apply_llama3_scaling(inv_freq, scaling)

    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    assert freqs.shape[1] == half
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    rope_type = str(scaling.get("rope_type", scaling.get("type", ""))).lower()
    if rope_type != "llama3":
        raise NotImplementedError(f"unsupported rope_scaling type {rope_type!r}")

    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max = float(scaling["original_max_position_embeddings"])

    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths are stretched by the full factor, short ones are left alone, and the
    # band between is interpolated. Applying the stretch everywhere is the classic mistake:
    # it destroys the high-frequency positional detail the short-range heads rely on.
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    in_band = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(in_band, smoothed, scaled)
