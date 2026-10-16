"""Internal-only tests for HCU WASP/WDRA ``gemm_fp16`` (not a public API)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm.triton.gemm_fp16 import (  # noqa: E402
    MODES,
    build_autotune_configs,
    gemm_fp16,
    mode_launch_kwargs,
    supports_wasp_wdra,
)
from boltops.tools.tune_gemm_wasp import (  # noqa: E402
    SHAPES,
    assert_fp16_gemm_close,
    _make_ab,
)

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


def test_removed_pingpong_sweep_rejected():
    from boltops.tools.tune_gemm_wasp import main
    with pytest.raises(SystemExit):
        main(["--consumer-pingpong", "false"])



@pytest.mark.parametrize("mode,quotas", [("wdra48", (16, 244, 244)), ("wdra88", (16, 224, 240))])
def test_promote_emits_current_partition_options(tmp_path, monkeypatch, mode, quotas):
    import csv
    import json
    from boltops.tools import tune_gemm_wasp as tune
    monkeypatch.setattr(tune, "_CONFIG_DIR", tmp_path / "configs")
    row = dict(mode=mode, M=256, N=256, K=4096, ab_layout="TT",
               BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
               num_stages=2, empty_arrive_after_mmac=0, enable_v_mmac_cluster=0,
               enable_consumer_pingpong=1, mmac_layout_force=3,
               wasp_partition_warps="(4,4,4)",
               wdra_num_load_regs=quotas[0], wdra_num_mma_regs_main=quotas[1],
               wdra_num_mma_regs_tail=quotas[2], tops=7.4, cycles=159000)
    source = tmp_path / "measurements.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    result = json.loads(tune._promote(source, "test").read_text())
    producers = 2 if mode == "wdra88" else 1
    assert result["wasp_partition_warps"] == [4] * (producers + 2)
    assert result["wasp_partition_regs"] == [quotas[0]] * producers + list(quotas[1:])
    assert "enable_consumer_pingpong" not in result
    assert "wasp_enabled" not in result
    assert "wdra_enabled" not in result
    assert "wdra_num_load_regs" not in result


def test_saved_configs_use_current_hcu_options():
    import json
    from pathlib import Path
    from boltops.tools import tune_gemm_wasp as tune
    from triton.backends.hcu.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    backend = HIPBackend(GPUTarget("hip", "gfx946", 64))
    from triton.backends.hcu import wasp
    configs = list((Path(tune._CONFIG_DIR) / "gfx946").glob("*.json"))
    assert configs, f"Missing packaged GEMM configs under {tune._CONFIG_DIR}/gfx946"
    for path in configs:
        cfg = json.loads(path.read_text())
        assert not ({"wasp_enabled", "wdra_enabled", "wasp_num_load_warps", "wasp_num_mma_warps", "wdra_num_load_regs", "wdra_num_mma_regs_main", "wdra_num_mma_regs_tail", "enable_consumer_pingpong"} & cfg.keys()), path
        options = {k: cfg[k] for k in tune._LAUNCH_KEYS if k in cfg and k not in {"GROUP_SIZE_M", "WARP_SPECIALIZE", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"}}
        parsed = backend.parse_options(options)
        assert set(options) <= set(parsed.__dataclass_fields__), path
        if parsed.wasp_partition_regs is not None:
            wasp.normalize_regs(parsed.wasp_partition_regs, "gfx946")


def test_cycle_attribution_ignores_other_output_directories(tmp_path, monkeypatch):
    from boltops.tools import tune_gemm_wasp as tune
    current = tmp_path / "current/m5out/0/0/stats.txt"
    foreign = tmp_path / "other/m5out/0/0/stats.txt"
    monkeypatch.setattr(tune, "_iter_stats", lambda: [(current, 159360, 2.197802, 2)])
    assert tune._best_stats_since({str(foreign): (1, 27041)})[1] == 159360
    prior = current.parent.parent / "previous/stats.txt"
    assert tune._best_stats_since({str(prior): (1, 10000)})[1] == 149360


def test_tune_shapes_table():
    assert SHAPES[0][0] == "512cube"
    assert SHAPES[0][1:] == (512, 512, 512)


def test_fp16_golden_helper_is_tighter_than_1e2():
    """Dropped K-tile must fail 1e-3; identical tensors pass."""
    from boltops.tools.tune_gemm_wasp import GOLDEN_ATOL, GOLDEN_RTOL, fp16_gemm_golden

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
    out = gemm_fp16(
        a,
        b,
        wasp=False,
        wdra=False,
        autotune=False,
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
    out = gemm_fp16(
        a_cpu.to(DEVICE),
        b_cpu.to(DEVICE),
        wasp=True,
        wdra=True,
        use_mls=True,
        wasp_num_load_warps=4,
        wasp_num_mma_warps=8,
        autotune=False,
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
    gemm_fp16(a, b, autotune=False)
    torch.cuda.synchronize()
    ms = time_ms(lambda: gemm_fp16(a, b, autotune=False), bench_warmup, bench_repeat)
    print(f"[bench] gemm_fp16 nowasp {M}x{N}x{K}: {ms:.3f} ms")
