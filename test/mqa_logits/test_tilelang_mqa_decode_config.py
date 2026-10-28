# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Unit tests for gfx938 and shared MQA decode launch policies."""

import pytest

from boltops.mqa_logits.tilelang.configs.mqa_decode import (
    resolve_get_config,
    resolve_get_config_fp8,
)
from boltops.mqa_logits.tilelang.configs.mqa_decode.common import (
    get_config as get_common_config,
)
from boltops.mqa_logits.tilelang.configs.mqa_decode.gfx938 import (
    get_config,
    get_config_fp8,
)


def _schedule(batch: int, keys: int, heads: int = 16) -> tuple[str, int]:
    config = get_config(batch, keys, 72, heads)
    if config.use_persistent:
        return "persistent", config.persistent_splits
    return "normal", config.groups_per_cta


def test_measured_decode_schedule_table():
    assert _schedule(16, 8 * 1024) == ("normal", 8)
    assert _schedule(32, 8 * 1024) == ("persistent", 4)
    assert _schedule(64, 8 * 1024) == ("persistent", 8)

    assert _schedule(16, 16 * 1024) == ("persistent", 8)
    assert _schedule(32, 16 * 1024) == ("persistent", 8)
    assert _schedule(64, 16 * 1024) == ("persistent", 16)

    assert _schedule(16, 32 * 1024) == ("persistent", 8)
    assert _schedule(32, 32 * 1024) == ("persistent", 16)
    assert _schedule(64, 32 * 1024) == ("persistent", 16)

    assert _schedule(16, 64 * 1024) == ("persistent", 8)
    assert _schedule(32, 64 * 1024) == ("persistent", 32)
    assert _schedule(64, 64 * 1024) == ("persistent", 32)


def test_128k_reuses_64k_decode_policy():
    for heads in (4, 16, 32, 64):
        for batch in (16, 32, 64, 128):
            assert _schedule(batch, 128 * 1024, heads) == _schedule(
                batch, 64 * 1024, heads
            )


def test_wide_head_decode_schedule_overrides():
    for heads in (32, 64):
        assert _schedule(16, 8 * 1024, heads) == ("persistent", 8)
        assert _schedule(32, 16 * 1024, heads) == ("persistent", 16)
        assert _schedule(128, 32 * 1024, heads) == ("persistent", 8)
        assert _schedule(128, 64 * 1024, heads) == ("persistent", 16)


def test_small_head_decode_schedule_is_shared():
    for heads in (4, 8, 16):
        assert _schedule(16, 8 * 1024, heads) == ("normal", 8)
        assert _schedule(32, 16 * 1024, heads) == ("persistent", 8)
        assert _schedule(128, 64 * 1024, heads) == ("persistent", 8)


def test_common_policy_matches_reference_at_72_cus():
    for heads in (4, 16, 32, 64):
        for batch in (1, 8, 16, 32, 64, 128):
            for keys in (8, 16, 32, 64):
                assert get_common_config(
                    batch, keys * 1024, 72, heads
                ) == get_config(batch, keys * 1024, 72, heads)


def test_non_938_arch_resolves_common_policy():
    assert resolve_get_config("gfx936") is get_common_config


def test_common_persistent_split_scales_with_cu_count():
    assert get_common_config(16, 16 * 1024, 36, 16).persistent_splits == 4
    assert get_common_config(16, 16 * 1024, 72, 16).persistent_splits == 8
    assert get_common_config(16, 16 * 1024, 144, 16).persistent_splits == 16


def test_fp8_decode_config_is_gfx938_only():
    assert resolve_get_config_fp8("gfx938") is get_config_fp8
    assert get_config_fp8(32, 16 * 1024, 72, 16) == get_config(
        32, 16 * 1024, 72, 16
    )
    with pytest.raises(RuntimeError, match="FP8 QSA MQA decode config"):
        resolve_get_config_fp8("gfx936")


def test_small_batch_targets_about_256_ctas():
    for getter in (get_config, get_config_fp8):
        for heads in (4, 32):
            batch1_64k = getter(1, 64 * 1024, 72, heads)
            assert not batch1_64k.use_persistent
            assert batch1_64k.groups_per_cta == 2

            batch1_8k = getter(1, 8 * 1024, 72, heads)
            assert not batch1_8k.use_persistent
            assert batch1_8k.groups_per_cta == 1

            batch8_64k = getter(8, 64 * 1024, 72, heads)
            assert batch8_64k.persistent_splits == 32

            # 8k has only 64 groups, so 32 splits would leave too little
            # work per CTA.  H64 measured S16 ahead of S32 here.
            batch8_8k = getter(8, 8 * 1024, 72, heads)
            assert batch8_8k.persistent_splits == 16

        h64_64k = getter(1, 64 * 1024, 72, 64)
        assert not h64_64k.use_persistent
        assert h64_64k.groups_per_cta == 4


def test_fp8_batch_16_uses_more_splits_than_bf16():
    assert get_config(16, 64 * 1024, 72, 4).persistent_splits == 8
    assert get_config_fp8(16, 64 * 1024, 72, 4).persistent_splits == 16
    assert get_config(16, 8 * 1024, 72, 4).groups_per_cta == 8
    assert get_config_fp8(16, 8 * 1024, 72, 4).persistent_splits == 16


def test_common_ordinary_gpc_is_not_scaled():
    for cu_count in (36, 72, 144):
        config = get_common_config(16, 8 * 1024, cu_count, 16)
        assert not config.use_persistent
        assert config.groups_per_cta == 8
