# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Redact backend identifiers only at owned HCU output boundaries."""

from __future__ import annotations

import re


_PLATFORM_NAMES = re.compile(r"amdgpu|amd|xgmi", re.IGNORECASE)


def format_hcu_message(message: object) -> str:
    """Format owned user-facing messages without changing runtime identifiers."""
    def display_name(match: re.Match[str]) -> str:
        value = match.group()
        if value.lower() == "xgmi":
            return "[interconnect]"
        return "[backend]"

    return _PLATFORM_NAMES.sub(display_name, str(message))
