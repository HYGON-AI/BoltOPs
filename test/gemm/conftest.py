# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

from copy import deepcopy
from importlib import import_module

import pytest


@pytest.fixture
def override_gemm_config(monkeypatch):
    """Supply test launch settings through a backend's normal config lookup."""
    def override(config_name, config, *, backend="triton"):
        operation = config_name.lower().replace("-", "_")
        module = f"triton.{operation}" if backend == "triton" else "gluon"
        dispatch = import_module(f"boltops.gemm.{module}")
        lookup = dispatch.get_gemm_config

        def get_config(name, *args, **kwargs):
            if name == config_name:
                return deepcopy(config), True
            return lookup(name, *args, **kwargs)

        monkeypatch.setattr(dispatch, "get_gemm_config", get_config)

    return override
