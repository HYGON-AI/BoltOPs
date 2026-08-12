from types import SimpleNamespace

import pytest

from boltops.mhc.tilelang.configs import pre_big_fuse
from boltops.mhc.tilelang.configs.pre_big_fuse.common import get_config
from boltops.mhc.tilelang.configs.pre_big_fuse.gfx946 import (
    _score_common_candidate,
    get_config as get_gfx946_config,
)


def test_pre_big_fuse_config_is_cached_and_valid():
    config = get_config(128, 4 * 4096, 64)
    assert config is get_config(128, 4 * 4096, 64)
    assert config.token_block > 0
    assert config.hidden_block > 0
    assert config.n_splits_pre >= 1


def test_large_token_config_disables_split_k():
    config = get_config(65536, 4 * 4096, 64)
    assert config.n_splits_pre == 1
    assert not config.use_small_token_splitk


def test_gfx946_config_uses_power_of_two_shapes():
    for num_tokens in (1, 16, 64, 128, 256, 512, 1024, 4096, 16384):
        config = get_gfx946_config(num_tokens, 4 * 4096, 48)
        for value in (
            config.token_block,
            config.hidden_block,
            config.n_splits_pre,
        ):
            assert value > 0 and value & (value - 1) == 0
        assert config.target_occupancy > 0


def test_gfx946_small_shape_fills_common_residency_slots():
    config = get_gfx946_config(16, 4 * 4096, 48)
    total_ctas = (
        (16 + config.token_block - 1) // config.token_block * config.n_splits_pre
    )
    assert config.target_occupancy == 3
    assert config.token_block == 32
    assert config.n_splits_pre == 64
    assert total_ctas == 64
    assert config.separate_x_fn_smem


def test_gfx946_small_shape_caps_split_and_targets_128_ctas():
    config64 = get_gfx946_config(64, 4 * 4096, 48)
    config128 = get_gfx946_config(128, 4 * 4096, 48)

    assert config64.token_block == 32
    assert config128.token_block == 64
    assert config64.hidden_block == config128.hidden_block == 64
    assert config64.n_splits_pre == 64
    assert config128.n_splits_pre == 64
    assert config64.separate_x_fn_smem
    assert config128.separate_x_fn_smem
    assert (64 // 32) * config64.n_splits_pre == 128
    assert (128 // 64) * config128.n_splits_pre == 128


def test_gfx946_tokens256_targets_one_power_of_two_cu_wave():
    config = get_gfx946_config(256, 4 * 4096, 48)

    assert config.token_block == 128
    assert config.hidden_block == 64
    assert config.n_splits_pre == 16
    assert (256 // config.token_block) * config.n_splits_pre == 32
    assert config.target_occupancy == 1
    assert not config.separate_x_fn_smem


def test_gfx946_tokens512_uses_generic_persistent_score():
    config = get_gfx946_config(512, 4 * 4096, 48)

    assert config.token_block == 128
    assert config.hidden_block == 64
    assert config.n_splits_pre == 32
    assert (512 // config.token_block) * config.n_splits_pre == 128


def test_gfx946_persistent_config_enforces_kernel_k_tiling_constraints():
    mhc_hidden_size = 8192
    config = get_gfx946_config(512, mhc_hidden_size, 48)
    assert mhc_hidden_size % config.n_splits_pre == 0
    assert (mhc_hidden_size // config.n_splits_pre) % (2 * config.hidden_block) == 0

    with pytest.raises(ValueError, match="No valid gfx946 persistent split-K"):
        get_gfx946_config(512, 8256, 48)


def test_gfx946_split_one_still_uses_persistent_stage_zero():
    config = get_gfx946_config(16384, 4 * 4096, 48)

    assert config.n_splits_pre == 1
    assert config.use_small_token_splitk


def test_gfx946_common_score_fills_residency_then_uses_larger_token_block():
    three_rounds = _score_common_candidate(64, 32, 64, 48)
    two_rounds = _score_common_candidate(64, 64, 64, 48)
    assert three_rounds < two_rounds

    tb32_split32 = _score_common_candidate(128, 32, 32, 48)
    tb64_split64 = _score_common_candidate(128, 64, 64, 48)
    assert tb64_split64 < tb32_split32


def test_gfx946_tokens96_fills_more_common_residency_slots():
    config = get_gfx946_config(96, 4 * 4096, 48)
    total_ctas = (
        (96 + config.token_block - 1) // config.token_block * config.n_splits_pre
    )

    assert config.token_block == 64
    assert config.n_splits_pre == 64
    assert total_ctas == 128


def test_gfx946_large_shape_uses_occupancy_one_persistent_policy():
    config = get_gfx946_config(1024, 4 * 4096, 48)
    assert config.target_occupancy == 1
    assert config.token_block == 128
    assert config.hidden_block == 64
    assert config.n_splits_pre == 16
    total_ctas = (
        (1024 + config.token_block - 1) // config.token_block * config.n_splits_pre
    )
    assert (total_ctas + 47) // 48 == 3
    assert not config.separate_x_fn_smem


def test_gfx946_tokens2048_uses_three_high_utilization_cu_rounds():
    config = get_gfx946_config(2048, 4 * 4096, 48)
    total_tiles = (
        (2048 + config.token_block - 1) // config.token_block * config.n_splits_pre
    )

    assert config.token_block == 128
    assert config.hidden_block == 64
    assert config.n_splits_pre == 8
    assert config.target_occupancy == 1
    assert total_tiles == 128


def test_gfx946_tokens6144_uses_two_full_cu_rounds():
    config = get_gfx946_config(6144, 4 * 4096, 48)
    total_tiles = (
        (6144 + config.token_block - 1) // config.token_block * config.n_splits_pre
    )

    assert config.token_block == 128
    assert config.hidden_block == 64
    assert config.n_splits_pre == 2
    assert total_tiles == 2 * 48


def _install_fake_config_modules(monkeypatch, available: set[str]):
    imports: list[str] = []

    def sentinel(*_args):
        return None

    monkeypatch.setattr(
        pre_big_fuse,
        "find_spec",
        lambda module_name: object() if module_name in available else None,
    )

    def fake_import(module_name: str):
        imports.append(module_name)
        return SimpleNamespace(get_config=sentinel)

    monkeypatch.setattr(pre_big_fuse, "import_module", fake_import)
    pre_big_fuse.resolve_get_config.cache_clear()
    return sentinel, imports


def test_config_falls_back_to_common(monkeypatch):
    common = "boltops.mhc.tilelang.configs.pre_big_fuse.common"
    sentinel, imports = _install_fake_config_modules(monkeypatch, {common})

    assert pre_big_fuse.resolve_get_config("gfx946") is sentinel
    assert imports == [common]


def test_config_family_override_takes_precedence(monkeypatch):
    family = "boltops.mhc.tilelang.configs.pre_big_fuse.gfx94x"
    common = "boltops.mhc.tilelang.configs.pre_big_fuse.common"
    sentinel, imports = _install_fake_config_modules(monkeypatch, {family, common})

    assert pre_big_fuse.resolve_get_config("gfx946") is sentinel
    assert imports == [family]


def test_config_exact_override_takes_precedence(monkeypatch):
    exact = "boltops.mhc.tilelang.configs.pre_big_fuse.gfx946"
    family = "boltops.mhc.tilelang.configs.pre_big_fuse.gfx94x"
    common = "boltops.mhc.tilelang.configs.pre_big_fuse.common"
    sentinel, imports = _install_fake_config_modules(
        monkeypatch, {exact, family, common}
    )

    assert pre_big_fuse.resolve_get_config("gfx946") is sentinel
    assert imports == [exact]


def test_resolved_config_is_cached(monkeypatch):
    common = "boltops.mhc.tilelang.configs.pre_big_fuse.common"
    sentinel, imports = _install_fake_config_modules(monkeypatch, {common})

    assert pre_big_fuse.resolve_get_config("gfx946") is sentinel
    assert pre_big_fuse.resolve_get_config("gfx946") is sentinel
    assert imports == [common]
