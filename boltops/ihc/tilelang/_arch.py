# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
_FALLBACKS = {"gfx936": ("gfx936", "gfx93x", "common"), "gfx938": ("gfx938", "gfx93x", "common"), "gfx946": ("gfx946", "gfx94x", "common")}
def targets_for_arch(arch):
    return _FALLBACKS.get(arch, (arch, "common"))
