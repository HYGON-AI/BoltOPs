# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""MQA logits collection rules.

``--bench`` runs BoltOPs-only latency cases. Hipc comparisons stay marked
``bench`` (so a plain correctness run does not execute them) but are skipped
unless ``-k`` selects ``vs_hipc``. optest ``per`` invokes pytest with
``--bench -m bench`` and no keyword, so CI does not time hipc.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--bench"):
        return
    keyword = config.getoption("keyword") or ""
    if "vs_hipc" in keyword:
        return
    skip_hipc = pytest.mark.skip(reason="pass -k vs_hipc to compare against hipc")
    for item in items:
        if "vs_hipc" not in item.name:
            continue
        path = str(getattr(item, "path", "") or "")
        if "/mqa_logits/" not in path.replace("\\", "/"):
            continue
        item.add_marker(skip_hipc)
