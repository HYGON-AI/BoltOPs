# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Aiter-style configuration lookup and backend isolation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from boltops.gemm import config

PARAMS = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    config.clear_cache()


@pytest.fixture
def trees(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.delenv("BOLTOPS_GEMM_CONFIG_DIR", raising=False)
    for backend in config.BACKENDS:
        write(
            tmp_path / "gfx946" / backend / "gemm_a16w16/DEFAULT.json",
            {"any": PARAMS},
        )
    yield tmp_path
    config.clear_cache()


@pytest.mark.parametrize("backend", config.BACKENDS)
def test_backend_isolation_and_copy(trees, backend):
    folder = trees / "gfx946" / backend / "gemm_a16w16"
    write(
        folder / "GEMM-A16W16-N=64-K=32.json",
        {"M_LEQ_8": dict(PARAMS, num_stages=3), "any": PARAMS},
    )
    result, tuned = config.get_gemm_config(
        "GEMM-A16W16", 8, 64, 32, arch="gfx946", backend=backend
    )
    assert tuned and result["num_stages"] == 3
    result["BLOCK_SIZE_M"] = 999
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, arch="gfx946", backend=backend
        )[0]["BLOCK_SIZE_M"]
        == 32
    )
    for other in config.BACKENDS:
        if other != backend:
            assert (
                "num_stages"
                not in config.get_gemm_config(
                    "GEMM-A16W16", 8, 64, 32, arch="gfx946", backend=other
                )[0]
            )


def test_batch_priority_and_custom_filename(trees):
    folder = trees / "gfx946/triton/gemm_a16w16"
    for suffix, stages in [("N=64-K=32", 1), ("B=2-N=64-K=32", 2), ("custom", 3)]:
        write(
            folder / f"GEMM-A16W16-{suffix}.json",
            {"any": dict(PARAMS, num_stages=stages)},
        )
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, B=2, arch="gfx946")[0][
            "num_stages"
        ]
        == 2
    )
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, B=3, arch="gfx946")[0][
            "num_stages"
        ]
        == 1
    )
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, B=2, arch="gfx946", specialized_filename="custom"
        )[0]["num_stages"]
        == 3
    )


def test_cu_tree_is_selected_whole(trees):
    base = trees / "gfx946/triton/gemm_a16w16"
    write(base / "GEMM-A16W16-N=64-K=32.json", {"any": dict(PARAMS, num_stages=3)})
    cu = trees / "gfx946/48cu/triton/gemm_a16w16"
    # An incomplete CU directory does not activate a CU tree.
    write(cu / "GEMM-A16W16-N=32-K=32.json", {"any": dict(PARAMS, num_stages=2)})
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, arch="gfx946", cu_num=48)[0][
            "num_stages"
        ]
        == 3
    )
    write(cu / "DEFAULT.json", {"any": dict(PARAMS, num_stages=1)})
    # Missing shape files do not fall through to architecture-level tuned data.
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, arch="gfx946", cu_num=48)[0][
            "num_stages"
        ]
        == 1
    )
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, arch="gfx946", cu_num=96)[0][
            "num_stages"
        ]
        == 3
    )


@pytest.mark.parametrize(
    "m,stage",
    [
        (1, 1),
        (8, 1),
        (9, 2),
        (16, 2),
        (17, 3),
        (64, 3),
        (65, 4),
        (127, 4),
        (128, 5),
        (256, 5),
    ],
)
def test_m_bucket_order(trees, m, stage):
    write(
        trees / "gfx946/triton/gemm_a16w16/DEFAULT.json",
        {
            "M_LEQ_8": dict(PARAMS, num_stages=1),
            "M_LEQ_16": dict(PARAMS, num_stages=2),
            "M_LEQ_64": dict(PARAMS, num_stages=3),
            "M_GEQ_128": dict(PARAMS, num_stages=5),
            "any": dict(PARAMS, num_stages=4),
        },
    )
    assert (
        config.get_gemm_config("GEMM-A16W16", m, arch="gfx946")[0]["num_stages"]
        == stage
    )


def test_default_fallback_merges_sparse_shape(trees):
    folder = trees / "gfx946/triton/gemm_a16w16"
    write(
        folder / "DEFAULT.json", {"M_LEQ_8": dict(PARAMS, num_stages=1), "any": PARAMS}
    )
    write(
        folder / "GEMM-A16W16-N=64-K=32.json",
        {"DEFAULT_FALLBACK": True, "M_LEQ_16": dict(PARAMS, num_stages=2)},
    )
    assert (
        config.get_gemm_config("GEMM-A16W16", 8, 64, 32, arch="gfx946")[0]["num_stages"]
        == 1
    )
    assert (
        config.get_gemm_config("GEMM-A16W16", 16, 64, 32, arch="gfx946")[0][
            "num_stages"
        ]
        == 2
    )
    assert config.get_gemm_config("GEMM-A16W16", 9000, 64, 32, arch="gfx946") == (
        PARAMS,
        False,
    )


@pytest.mark.parametrize("bad", ["M_LEQ_0", "M_GEQ_-8", "M_LEQ_no", "M_BOUNDS"])
def test_invalid_bucket(trees, bad):
    write(
        trees / "gfx946/triton/gemm_a16w16/DEFAULT.json",
        {bad: PARAMS, "any": PARAMS},
    )
    with pytest.raises(ValueError, match="M bucket"):
        config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946")


def test_missing_backend_never_retries(trees):
    with pytest.raises(FileNotFoundError):
        config.get_gemm_config("GEMM-A16W16", 8, arch="gfx999")
    with pytest.raises(ValueError, match="select a backend"):
        config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946", backend="auto")


def test_override_family_and_cache(trees, tmp_path, monkeypatch):
    user = tmp_path / "user"
    path = user / "gfx946/triton/gemm_a16w16/DEFAULT.json"
    write(path, {"any": dict(PARAMS, num_stages=3)})
    monkeypatch.setenv("BOLTOPS_GEMM_CONFIG_DIR", str(user))
    assert config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946")[0]["num_stages"] == 3
    path.write_text(json.dumps({"any": dict(PARAMS, num_stages=2)}))
    assert config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946")[0]["num_stages"] == 3
    config.clear_cache()
    assert config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946")[0]["num_stages"] == 2


def test_no_eager_dsl_import():
    code = """
import torch, sys, importlib.abc
for name in list(sys.modules):
 if name.split('.')[0] in ('triton','tilelang','tvm'): del sys.modules[name]
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('triton','tilelang','tvm'): raise AssertionError(fullname)
sys.meta_path.insert(0,Reject())
import boltops.gemm
from boltops.gemm.config import get_gemm_config
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_layout_files_share_family_and_preserve_batch_priority(trees):
    directory = trees / "gfx946/triton/gemm_a16w16"
    for suffix, stages in [
        ("N=64-K=32-TN", 1),
        ("B=2-N=64-K=32-TN", 2),
        ("N=64-K=32-TT", 3),
    ]:
        write(
            directory / f"GEMM-A16W16-{suffix}.json",
            {"any": dict(PARAMS, num_stages=stages)},
        )
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, B=2, arch="gfx946", layout="TN"
        )[0]["num_stages"]
        == 2
    )
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, B=3, arch="gfx946", layout="TN"
        )[0]["num_stages"]
        == 1
    )
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, B=2, arch="gfx946", layout="TT"
        )[0]["num_stages"]
        == 3
    )
    assert (
        config.get_gemm_config(
            "GEMM-A16W16", 8, 64, 32, B=2, arch="gfx946", layout="NN"
        )[0]
        == PARAMS
    )


@pytest.mark.parametrize(
    "profile,default_m,winner_m", [("fp32_recipe", 32, 768), ("bf16_recipe", 64, 512)]
)
def test_profiles_share_shape_file_without_cross_matching(
    trees, profile, default_m, winner_m
):
    folder = trees / "gfx946/triton/gemm_a16w16"
    write(
        folder / "DEFAULT.json",
        {
            "profiles": {
                "fp32_recipe": {"any": dict(PARAMS, BLOCK_SIZE_M=32)},
                "bf16_recipe": {"any": dict(PARAMS, BLOCK_SIZE_M=64)},
            }
        },
    )
    write(
        folder / "GEMM-A16W16-N=64-K=32.json",
        {
            "profiles": {
                "fp32_recipe": {
                    "DEFAULT_FALLBACK": True,
                    "M_EQ_768": dict(PARAMS, num_stages=3),
                },
                "bf16_recipe": {
                    "DEFAULT_FALLBACK": True,
                    "M_EQ_512": dict(PARAMS, num_stages=2),
                },
            }
        },
    )
    for m in (512, 640, 768):
        params, tuned = config.get_gemm_config(
            "GEMM-A16W16", m, 64, 32, arch="gfx946", profile=profile
        )
        assert tuned == (m == winner_m)
        if m != winner_m:
            assert params["BLOCK_SIZE_M"] == default_m and "num_stages" not in params
    with pytest.raises(KeyError, match="profile"):
        config.get_gemm_config(
            "GEMM-A16W16", 768, 64, 32, arch="gfx946", profile="absent"
        )
    with pytest.raises(ValueError, match="requires a profile"):
        config.get_gemm_config("GEMM-A16W16", 768, 64, 32, arch="gfx946")


def test_exact_m_precedes_range_bucket(trees):
    write(
        trees / "gfx946/triton/gemm_a16w16/DEFAULT.json",
        {
            "M_EQ_7": dict(PARAMS, num_stages=3),
            "M_LEQ_8": dict(PARAMS, num_stages=1),
            "any": PARAMS,
        },
    )
    for m, stage in [(6, 1), (7, 3), (8, 1)]:
        assert (
            config.get_gemm_config("GEMM-A16W16", m, arch="gfx946")[0]["num_stages"]
            == stage
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"profiles": []},
        {"profiles": {"recipe": []}},
        {"profiles": {"recipe": {"M_EQ_0": PARAMS}}},
        {"profiles": {"recipe": {"M_LEQ_bad_8": PARAMS}}},
    ],
)
def test_invalid_profile_buckets(trees, bad):
    write(trees / "gfx946/triton/gemm_a16w16/DEFAULT.json", bad)
    with pytest.raises(ValueError):
        config.get_gemm_config("GEMM-A16W16", 8, arch="gfx946", profile="recipe")
