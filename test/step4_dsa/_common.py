# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared pytest helpers for Step-4 DSA correctness tests and benchmarks."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa.geometry import Step4Geometry

pytest.importorskip("triton")

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def tp8_geometry() -> Step4Geometry:
    """Real TP8 per-rank geometry, including one four-head indexer provider group."""
    return Step4Geometry(
        num_attention_heads=8,
        num_attention_groups=1,
        indexer_num_heads=4,
        num_provider_groups=1,
    )


def last_dim_padded_view(src: torch.Tensor, pad: int, poison: float) -> torch.Tensor:
    """Same logical values as ``src``, larger leading strides, last dim still contiguous."""
    padded = torch.empty(
        (*src.shape[:-1], src.shape[-1] + pad),
        device=src.device,
        dtype=src.dtype,
    )
    padded[..., : src.shape[-1]] = src
    padded[..., src.shape[-1] :] = poison
    view = padded[..., : src.shape[-1]]
    assert view.shape == src.shape
    assert view.stride(-1) == 1
    assert view.stride(0) != src.stride(0)
    return view


def strided_last_dim(src: torch.Tensor) -> torch.Tensor:
    """Same shape as ``src`` with ``stride(-1) != 1``, via a stepped last-dim slice."""
    expanded = src.new_empty(*src.shape[:-1], src.shape[-1] * 2)
    expanded[..., ::2] = src
    view = expanded[..., ::2]
    assert view.shape == src.shape
    assert view.stride(-1) != 1
    return view


def make_norm_inputs(
    geo: Step4Geometry,
    *,
    tokens: int,
    max_position: int,
    device: torch.device,
    seed: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    rotary_dim = geo.indexer_rope_dim // 2

    def randn(*shape: int, out_dtype: torch.dtype = dtype) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, device=device, dtype=out_dtype
        ).contiguous()

    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    angles = torch.arange(max_position, device=device, dtype=torch.float32)[
        :, None
    ] * (inverse_frequency[None, :])
    return {
        "index_q": randn(tokens, geo.indexer_num_heads * geo.proxy_dim),
        "index_k": randn(tokens, geo.indexer_num_k_heads * geo.proxy_dim),
        "index_z": randn(tokens, geo.indexer_num_k_heads * geo.proxy_dim),
        "q_norm_weight": randn(geo.proxy_dim),
        "k_norm_weight": randn(geo.proxy_dim),
        "k_norm_bias": randn(geo.proxy_dim),
        "cos": angles.cos().to(dtype).contiguous(),
        "sin": angles.sin().to(dtype).contiguous(),
        "positions": torch.randint(
            0,
            max_position,
            (tokens,),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ),
    }


FP8_E4M3_MAX = 448.0


def quantize_kv_fp8(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize K/V to ``float8_e4m3fn`` as ``clamp(x / scale, ±448)``.

    Matches the vLLM DSA store path: the sparse kernels later multiply the same
    ``scale`` back on the fp32 side.
    """
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available")
    scaled = tensor.float() if float(scale) == 1.0 else tensor.float() / float(scale)
    return scaled.clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(dtype)


def apply_fp8_kv_cache(
    data: dict, k_scale: float = 1.0, v_scale: float = 1.0
) -> tuple[float, float]:
    """Replace ``key_cache`` / ``value_cache`` in-place with ``float8_e4m3fn``."""
    data["key_cache"] = quantize_kv_fp8(data["key_cache"], k_scale)
    data["value_cache"] = quantize_kv_fp8(data["value_cache"], v_scale)
    return float(k_scale), float(v_scale)
