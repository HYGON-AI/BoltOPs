# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared architecture fallback policy for MHC TileLang components."""

from __future__ import annotations


_ARCH_FALLBACKS = {
    "gfx936": ("gfx936", "gfx93x", "common"),
    "gfx938": ("gfx938", "gfx93x", "common"),
    "gfx946": ("gfx946", "gfx94x", "common"),
}


def targets_for_arch(arch: str) -> tuple[str, ...]:
    """Return exact, family, then common targets for an architecture."""
    return _ARCH_FALLBACKS.get(arch, (arch, "common"))
