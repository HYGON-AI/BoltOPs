"""Regression tests for shared Step-4 runtime configuration helpers."""

from __future__ import annotations

import pytest

from boltops.step4_dsa.triton.common import (
    _choose_kv_splits,
    _clamp_auto_kv_splits,
    _config_match_score,
    _load_device_configs,
    _portable_launch_options,
    _select_config,
)


def test_config_specificity_and_missing_keys():
    shape = {"s_q": 4, "dtype": "bfloat16"}
    assert _config_match_score({"s_q": 4}, shape) == (1, 0, -0.0)
    assert _config_match_score({"s_q_min": 1, "s_q_max": 8}, shape) == (0, 2, -7.0)
    assert _config_match_score({"s_q_min": 5}, shape) is None
    assert _config_match_score({"s_q_max": 3}, shape) is None
    assert _config_match_score({"s_kv": None}, shape) is None
    assert _config_match_score({"s_kv_min": 1}, shape) is None


def test_config_selection_preserves_specificity_order_and_copies():
    broad = {"match": {"s_q_min": 1, "s_q_max": 128}, "config": {"BLOCK_R": 8}}
    narrow = {"match": {"s_q_min": 1, "s_q_max": 8}, "config": {"BLOCK_R": 4}}
    exact = {"match": {"s_q": 4}, "config": {"BLOCK_R": 2}}
    tied = {"match": {"s_q": 4}, "config": {"BLOCK_R": 1}}
    shape = {"s_q": 4}
    assert _select_config({"configs": [broad, narrow]}, shape) == narrow["config"]
    selected = _select_config({"configs": [broad, narrow, exact, tied]}, shape)
    assert selected == exact["config"]
    selected["BLOCK_R"] = 16
    assert exact["config"]["BLOCK_R"] == 2


@pytest.mark.parametrize("entries", [None, {}, "invalid", [None, 3, {"match": []}]])
def test_config_selection_ignores_malformed_entries(entries):
    assert _select_config({"configs": entries}, {"s_q": 4}) is None


def test_config_loading_and_device_alias(tmp_path):
    filename = lambda device: f"attention_{device}.json"
    assert _load_device_configs(str(tmp_path), filename, "BW1101") == {}
    path = tmp_path / filename("BW1100")
    path.write_text('{"configs": []}', encoding="utf-8")
    assert _load_device_configs(str(tmp_path), filename, "BW1101") == {"configs": []}
    for invalid in ("[1, 2]", "{invalid"):
        path.write_text(invalid, encoding="utf-8")
        assert _load_device_configs(str(tmp_path), filename, "BW1101") == {}


def test_portable_launch_options_filter_consumed_fields():
    config = {
        "BLOCK_R": 4, "AUTO_KV_SPLITS": 18, "num_warps": "4",
        "waves_per_eu": 2, "unsupported": 1,
    }
    consumed = frozenset({"BLOCK_R", "AUTO_KV_SPLITS"})
    defaults = {"num_stages": 2}
    assert _portable_launch_options(config, consumed_keys=consumed, defaults=defaults) == {
        "num_warps": 4, "num_stages": 2,
    }
    assert _portable_launch_options(
        config, consumed_keys=consumed, supports=lambda name: name == "waves_per_eu"
    ) == {"num_warps": 4, "waves_per_eu": 2}
    assert defaults == {"num_stages": 2}


@pytest.mark.parametrize(
    "queries,groups,width,target,min_slots,expected",
    [(4, 1, 513, 72, 8, 18), (128, 1, 513, 72, 8, 1),
     (1, 1, 9, 72, 8, 1), (1, 1, 9, 72, 1, 9), (0, 1, 0, 0, 8, 1)],
)
def test_shared_kv_split_policy(queries, groups, width, target, min_slots, expected):
    assert _choose_kv_splits(
        queries, groups, width, target_programs=target, min_slots_per_split=min_slots
    ) == expected
    assert 1 <= _clamp_auto_kv_splits(target, width) <= max(width, 1)
