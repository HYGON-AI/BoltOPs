# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Internal-only tests for HCU WASP/WDRA ``gemm_fp16`` (not a public API)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm.triton.gemm_a16w16_wasp import (  # noqa: E402
    MODES,
    build_autotune_configs,
    _gemm_fp16_load_kernel,
    _gemm_fp16_mls_kernel,
    default_launch_config,
    mode_launch_kwargs,
    supports_wasp_wdra,
)

GOLDEN_ATOL = 1e-3
GOLDEN_RTOL = 1e-3


def fp16_gemm_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32-accumulate reference, rounded to fp16 to match kernel store."""
    return torch.matmul(a.detach().cpu().float(), b.detach().cpu().float()).to(
        torch.float16
    )


def assert_fp16_gemm_close(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = GOLDEN_ATOL,
    rtol: float = GOLDEN_RTOL,
) -> tuple[float, float]:
    """``assert_close`` vs :func:`fp16_gemm_golden`. Returns ``(max_abs, max_rel)``."""
    out_cpu = out.detach().cpu()
    ref = fp16_gemm_golden(a, b)
    torch.testing.assert_close(out_cpu, ref, atol=atol, rtol=rtol)
    diff = (out_cpu.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-8)
    return float(diff.max().item()), float(rel.max().item())


def _make_ab(
    M: int,
    N: int,
    K: int,
    ab_layout: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build A ``[M,K]`` and B ``[K,N]`` for the given AB layout tag.

    TT: contiguous A (K-major), contiguous B (N-major).
    TN: contiguous A (K-major); B is a ``[N,K].T`` view (K-major, ``stride_bk=1``).
    """
    a = torch.randn((M, K), device=device, dtype=dtype)
    if ab_layout == "TT":
        b = torch.randn((K, N), device=device, dtype=dtype)
    else:  # TN
        b = torch.randn((N, K), device=device, dtype=dtype).T
    return a, b


def run_pipeline(a, b, *, use_mls=False, config=None):
    import os
    import triton

    os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    params = dict(config) if config is not None else default_launch_config()
    params["HAS_BIAS"] = False
    if use_mls:
        params["USE_MATRIX_STORE"] = False
    kernel = _gemm_fp16_mls_kernel if use_mls else _gemm_fp16_load_kernel
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    kernel[grid](
        a, b, out, b, m, n, k, *a.stride(), *b.stride(), *out.stride(), **params
    )
    return out


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
WASP_REQUIRED = pytest.mark.skipif(
    not supports_wasp_wdra(), reason="wasp/wdra require gfx946"
)


def test_autotune_configs_include_barrier_knobs():
    cfgs = build_autotune_configs(wasp=True, wdra=True, load_warps=4, mma_warps=8)
    assert len(cfgs) == 18
    keys = cfgs[0].kwargs
    assert "empty_arrive_after_mmac" in keys
    assert "enable_v_mmac_cluster" in keys
    assert "sched_barrier_before_mmac" not in keys


def test_mode_launch_kwargs_modes():
    assert set(MODES) == {"nowasp", "wasp44", "wdra48", "wdra88"}
    wdra48 = mode_launch_kwargs("wdra48")
    assert wdra48["wasp_partition_warps"] == (4, 4, 4)
    assert wdra48["wasp_wdra"] is True
    assert wdra48["wasp_partition_regs"] == (16, 244, 244)
    assert (wdra48["BLOCK_SIZE_M"], wdra48["BLOCK_SIZE_K"]) == (64, 64)
    assert "enable_consumer_pingpong" not in wdra48


@pytest.mark.parametrize("mode,waves,wdra", [
    ("nowasp", None, False), ("wasp44", (4, 4), False),
    ("wdra48", (4, 4, 4), True), ("wdra88", (4, 4, 4, 4), True),
])
def test_launch_options_match_current_hcu_backend(mode, waves, wdra):
    from triton.backends.hcu.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    cfg = mode_launch_kwargs(mode)
    assert cfg.get("wasp_partition_warps") == waves
    assert cfg["wasp_wdra"] is wdra
    backend = HIPBackend(GPUTarget("hip", "gfx946", 64))
    constexprs = {"GROUP_SIZE_M", "WARP_SPECIALIZE", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"}
    options = {k: v for k, v in cfg.items() if k not in constexprs}
    parsed = backend.parse_options(dict(options))
    # parse_options ignores unknown keys, so also check the dataclass contract.
    assert set(options) <= set(parsed.__dataclass_fields__)
    assert not ({"wasp_enabled", "wdra_enabled", "wasp_num_load_warps", "wasp_num_mma_warps", "enable_consumer_pingpong"} & cfg.keys())
    if waves:
        assert parsed.wasp_partition_warps == waves


def test_fp16_golden_helper_is_tighter_than_1e2():
    """Dropped K-tile must fail 1e-3; identical tensors pass."""
    assert (GOLDEN_ATOL, GOLDEN_RTOL) == (1e-3, 1e-3)
    torch.manual_seed(0)
    a, b = _make_ab(64, 64, 256, "TN", device="cpu", dtype=torch.float16)
    ref = fp16_gemm_golden(a, b)
    assert_fp16_gemm_close(ref, a, b)
    dropped = fp16_gemm_golden(a[:, :-64], b[:-64, :])
    try:
        assert_fp16_gemm_close(dropped, a, b)
    except AssertionError:
        return
    raise AssertionError("dropped BK=64 tile should fail atol/rtol=1e-3")


_K_TAIL_XFAIL = pytest.mark.xfail(
    reason="MLS K-tail unmasked when K % BLOCK_SIZE_K != 0",
    strict=False,
)


@GPU_REQUIRED
@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 64, 64),
        (48, 80, 96),
        (64, 64, 128),
        (48, 80, 192),
    ],
)
@pytest.mark.parametrize("ab_layout", ["TT", "TN"])
def test_internal_gemm_fp16_nowasp(M: int, N: int, K: int, ab_layout: str):
    torch.manual_seed(0)
    a_cpu, b_cpu = _make_ab(M, N, K, ab_layout, device="cpu", dtype=torch.float16)
    # Poison the backing storage after each input. Unmasked K-tail reads
    # must fail deterministically instead of depending on allocator contents.
    a_storage = torch.full((M * K + 128 * 128,), float("nan"), dtype=torch.float16).to(DEVICE)
    b_storage = torch.full((K * N + 128 * 128,), float("nan"), dtype=torch.float16).to(DEVICE)
    a = a_storage[:M * K].view(M, K)
    b = (b_storage[:K * N].view(K, N) if ab_layout == "TT"
         else b_storage[:K * N].view(N, K).T)
    a.copy_(a_cpu)
    b.copy_(b_cpu)
    out = run_pipeline(
        a,
        b,
    )
    assert torch.isfinite(out.detach().cpu()).all(), "GEMM returned NaN/Inf with finite inputs"
    max_abs, max_rel = assert_fp16_gemm_close(out, a_cpu, b_cpu)
    print(f"nowasp {ab_layout} {M}x{N}x{K} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")


@GPU_REQUIRED
@WASP_REQUIRED
@pytest.mark.parametrize(
    "ab_layout,M,N,K,bm,bn,bk",
    [
        pytest.param("TN", 64, 64, 64, 64, 64, 64, id="TN_64cube_tile64"),
        pytest.param("TT", 64, 64, 64, 64, 64, 64, id="TT_64cube_tile64"),
        pytest.param("TN", 48, 64, 64, 64, 64, 64, id="TN_48x64x64_Mrem"),
        pytest.param("TN", 64, 80, 64, 64, 64, 64, id="TN_64x80x64_Nrem"),
        pytest.param(
            "TN", 64, 64, 96, 64, 64, 64, id="TN_64x64x96_Krem", marks=_K_TAIL_XFAIL
        ),
        pytest.param("TN", 48, 80, 64, 64, 64, 64, id="TN_48x80x64_MNrem"),
        pytest.param(
            "TN",
            48,
            80,
            96,
            64,
            64,
            64,
            id="TN_48x80x96_rem_tile64",
            marks=_K_TAIL_XFAIL,
        ),
        pytest.param(
            "TT",
            48,
            80,
            96,
            64,
            64,
            64,
            id="TT_48x80x96_rem_tile64",
            marks=_K_TAIL_XFAIL,
        ),
        pytest.param("TN", 128, 64, 64, 64, 64, 64, id="TN_128x64x64_2M"),
        pytest.param("TN", 128, 128, 128, 64, 64, 64, id="TN_128cube_tile64"),
        pytest.param("TN", 96, 64, 192, 64, 64, 64, id="TN_96x64x192_tile64"),
        # 256×256×64 needs ~256KB LDS here (PMD lds-size=128kB) → SIGSEGV.
        # 4k perf runs use the same tile with a compiled kernel that fits.
    ],
)
def test_internal_gemm_fp16_wdra48_mls_sizes(
    ab_layout: str, M: int, N: int, K: int, bm: int, bn: int, bk: int
):
    """wdra48 + MLS, same knobs as the 4k TN best (mlf=3, pp=1, ea=0)."""
    torch.manual_seed(0)
    a_cpu, b_cpu = _make_ab(M, N, K, ab_layout, device="cpu", dtype=torch.float16)
    cfg = mode_launch_kwargs(
        "wdra48",
        empty_arrive_after_mmac=False,
        mmac_layout_force=3,
        num_stages=2,
    )
    cfg.update(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk)
    out = run_pipeline(
        a_cpu.to(DEVICE),
        b_cpu.to(DEVICE),
        use_mls=True,
        config=cfg,
    )
    torch.cuda.synchronize()
    max_abs, max_rel = assert_fp16_gemm_close(out, a_cpu, b_cpu)
    print(
        f"wdra48+mls {ab_layout} {M}x{N}x{K} tile={bm}x{bn}x{bk} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
    )


@GPU_REQUIRED
@pytest.mark.bench
def test_bench_gemm_fp16_nowasp_skeleton(time_ms, bench_warmup: int, bench_repeat: int):
    torch.manual_seed(0)
    M = N = K = 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    run_pipeline(a, b)
    torch.cuda.synchronize()
    ms = time_ms(lambda: run_pipeline(a, b), bench_warmup, bench_repeat)
    print(f"[bench] gemm_fp16 nowasp {M}x{N}x{K}: {ms:.3f} ms")
