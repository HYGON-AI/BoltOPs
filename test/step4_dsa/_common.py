"""Shared pytest helpers for Step-4 DSA correctness tests."""

from __future__ import annotations

import pytest
import torch

from boltops.step4_dsa.geometry import Step4Geometry

pytest.importorskip("triton")

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


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
