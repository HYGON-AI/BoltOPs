# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Default launch policy for ``ihc_head``.

Shares the GEMM tile/split heuristic with ``ihc_pre`` for now.
"""

from boltops.ihc.tilelang.configs.pre.common import (
    BLOCK_N,
    PreConfig as HeadConfig,
    get_config,
)

__all__ = ["BLOCK_N", "HeadConfig", "get_config"]
