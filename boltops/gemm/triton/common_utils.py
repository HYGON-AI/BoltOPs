# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/utils/common_utils.py
"""Small helpers shared by vendored aiter GEMM wrappers."""

from __future__ import annotations

import json
from typing import Any


def serialize_dict(d: dict | None) -> str | None:
    if d is None:
        return None
    return json.dumps(d)


def deserialize_str(s: str | dict | None) -> dict | None:
    if s is None:
        return None
    if isinstance(s, dict):
        return dict(s)
    return json.loads(s)
