"""Configuration tests for TileLang MQA logits prefill."""

import inspect

from boltops.mqa_logits import tilelang_qsa_mqa_prefill
from boltops.mqa_logits.tilelang.configs.mqa_prefill import resolve_get_config
from boltops.mqa_logits.tilelang._dispatch import _resolve_for_arch


def test_mqa_logits_tilelang_dispatch_falls_back_by_symbol():
    prefill_938 = _resolve_for_arch("qsa_mqa_prefill", "gfx938")
    decode_938 = _resolve_for_arch("qsa_mqa_decode", "gfx938")
    prefill_936 = _resolve_for_arch("qsa_mqa_prefill", "gfx936")

    assert prefill_938.__module__.endswith("kernels.gfx938.mqa")
    assert decode_938.__module__.endswith("kernels.gfx938.mqa")
    assert prefill_936.__module__.endswith("kernels.common.mqa")


def test_gfx938_mqa_prefill_config_tracks_cu_wave_boundary():
    get_config = resolve_get_config("gfx938")

    for cu_count in (64, 72):
        for block_q in (8, 16, 32, 64):
            one_round = get_config(cu_count * block_q, 16384, cu_count)
            expected_block_q = max(block_q, 32)
            assert one_round.block_q == expected_block_q
            assert one_round.logical_m == expected_block_q * 4
            assert one_round.rows_divisible

        capped = get_config(cu_count * 64 + 1, 16384, cu_count)
        assert capped.block_q == 64
        assert not capped.rows_divisible


def test_gfx938_mqa_prefill_config_derives_q_tile_from_heads():
    get_config = resolve_get_config("gfx938")

    config = get_config(512, 16384, 64, num_heads=8)

    assert config.block_q == 16
    assert config.heads == 8
    assert config.logical_m == 128

    expected_block_q = {
        8: (16, 32),
        32: (8, 8),
        64: (2, 4),
    }
    for heads, (small_block_q, large_block_q) in expected_block_q.items():
        assert get_config(64, 16384, 72, num_heads=heads).block_q == small_block_q
        assert (
            get_config(8192, 16384, 72, num_heads=heads).block_q
            == large_block_q
        )

    assert get_config(64, 65536, 72, num_heads=8).split_n == 16
    assert get_config(64, 65536, 72, num_heads=32).split_n == 8
    assert get_config(64, 65536, 72, num_heads=64).split_n == 4

    # The singleton consumer-Q geometry must use block_q=2 only for the
    # short-row/short-range cases where its block-N-only wave split wins.
    assert get_config(64, 2048, 72, num_heads=32).block_q == 2
    assert get_config(8192, 2048, 72, num_heads=32).block_q == 8


def test_gfx938_h4_small_rows_use_key_dependent_logical_m_floor():
    get_config = resolve_get_config("gfx938")

    expected = {
        2048: (8, 2, 1),
        4096: (8, 2, 2),
        8192: (32, 8, 1),
        16384: (32, 8, 1),
        32768: (64, 16, 1),
        65536: (64, 16, 2),
    }
    for keys, choice in expected.items():
        config = get_config(256, keys, 72, num_heads=4)
        assert (config.block_q, config.split_n, config.n_tile_phases) == choice


def test_gfx938_mqa_prefill_config_tracks_key_stride():
    get_config = resolve_get_config("gfx938")

    assert get_config(8192, 2048, 64).n_tile_phases == 1
    assert get_config(8192, 2049, 64).n_tile_phases == 4
    assert get_config(8192, 4096, 64).n_tile_phases == 4
    assert get_config(4096, 8192, 64).n_tile_phases == 4
    assert get_config(4096, 16383, 64).n_tile_phases == 4
    assert get_config(4096, 16384, 64).n_tile_phases == 8
    assert get_config(4096, 32768, 64).n_tile_phases == 16
    assert get_config(4096, 65536, 64).n_tile_phases == 32
    assert get_config(4096, 32768, 64, 512).n_tile_phases == 1
    assert get_config(4096, 65536, 64, 24576).n_tile_phases == 8


def test_gfx938_mqa_prefill_config_jointly_selects_split_and_phase():
    get_config = resolve_get_config("gfx938")

    full_round = get_config(4608, 16384, 72)
    assert (full_round.split_n, full_round.n_tile_phases) == (1, 8)

    # One leftover Q CTA: split-N shortens the sparse second round instead
    # of doubling wave count at split_n=1.
    expected = {
        4096: (4, 1),
        8192: (4, 1),
        16384: (8, 1),
        32768: (8, 2),
        65536: (16, 2),
    }
    for keys, split_phase in expected.items():
        config = get_config(4609, keys, 72)
        assert (config.split_n, config.n_tile_phases) == split_phase

    occupied_64 = {
        16384: (1, 8),
        32768: (1, 16),
        65536: (1, 32),
    }
    for keys, split_phase in occupied_64.items():
        config = get_config(8192, keys, 64)
        assert (config.split_n, config.n_tile_phases) == split_phase

    occupied_72 = {
        16384: (1, 8),
        32768: (1, 16),
        65536: (8, 4),
    }
    for keys, split_phase in occupied_72.items():
        config = get_config(8192, keys, 72)
        assert (config.split_n, config.n_tile_phases) == split_phase


def test_gfx938_prefill_mask_launch_kwargs():
    from boltops.mqa_logits.tilelang.kernels.gfx938.mqa import (
        prefill_mask_launch_kwargs,
    )

    assert prefill_mask_launch_kwargs(256, 1024) == dict(
        steady_phases=1, row_phases=1, reverse_suffix=False
    )
    assert prefill_mask_launch_kwargs(256, 8192) == dict(
        steady_phases=1, row_phases=1, reverse_suffix=True
    )
    assert prefill_mask_launch_kwargs(4608, 32768) == dict(
        steady_phases=1, row_phases=1, reverse_suffix=True
    )
    assert prefill_mask_launch_kwargs(256, 65536) == dict(
        steady_phases=16, row_phases=1, reverse_suffix=True
    )
    assert prefill_mask_launch_kwargs(4608, 65536) == dict(
        steady_phases=16, row_phases=4, reverse_suffix=True
    )


def test_public_prefill_effective_range_defaults_like_get_config():
    params = inspect.signature(tilelang_qsa_mqa_prefill).parameters
    assert params["effective_range"].default is None
    assert "kv_scales" in params
    assert params["kv_scales"].default is None

    get_config = resolve_get_config("gfx938")
    full = get_config(4096, 65536, 64)
    short = get_config(4096, 65536, 64, effective_range=512)
    assert (full.split_n, full.n_tile_phases) != (
        short.split_n,
        short.n_tile_phases,
    )
    assert short.n_tile_phases == 1


def test_gfx938_fp8_config_does_not_change_bf16():
    from boltops.mqa_logits.tilelang.configs.mqa_prefill import (
        resolve_get_config_fp8,
    )
    from boltops.mqa_logits.tilelang.configs.mqa_prefill.gfx938 import (
        get_config,
        get_config_fp8,
    )

    bf16 = resolve_get_config("gfx938")
    fp8 = resolve_get_config_fp8("gfx938")
    assert bf16 is get_config
    assert fp8 is get_config_fp8

    for rows, keys, heads in (
        (4608, 8192, 4),
        (4608, 65536, 4),
        (8192, 65536, 4),
        (4608, 8192, 32),
        (64, 2048, 32),
        (4608, 8192, 64),
        (8192, 65536, 64),
    ):
        assert get_config(rows, keys, 72, num_heads=heads) == get_config_fp8(
            rows, keys, 72, num_heads=heads
        )

    h32_8k = get_config(4608, 8192, 72, num_heads=32)
    assert h32_8k.block_q == 8
    assert h32_8k.n_tile_phases == 1
    assert h32_8k.split_n == 1

    h32_16k = get_config_fp8(4608, 16384, 72, num_heads=32)
    h32_16k_bf16 = get_config(4608, 16384, 72, num_heads=32)
    assert h32_16k.block_q == h32_16k_bf16.block_q
    assert h32_16k.split_n == h32_16k_bf16.split_n
    assert h32_16k.n_tile_phases == 1
    assert h32_16k_bf16.n_tile_phases == 1

    h64_8k = get_config_fp8(4608, 8192, 72, num_heads=64)
    assert h64_8k.block_q == 4
    assert h64_8k.n_tile_phases == 1
    assert get_config(4608, 8192, 72, num_heads=64).n_tile_phases == 1

    h64_64k = get_config_fp8(8192, 65536, 72, num_heads=64)
    assert h64_64k.block_q == 4
    assert h64_64k.split_n == 1
    assert h64_64k.n_tile_phases == 1

    h4_64k = get_config_fp8(4608, 65536, 72, num_heads=4)
    assert h4_64k == get_config(4608, 65536, 72, num_heads=4)
    assert h4_64k.n_tile_phases == 32
