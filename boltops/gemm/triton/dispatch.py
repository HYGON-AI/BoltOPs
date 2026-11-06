# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Register adapters for implementations already present on main."""
from .scaled_dispatch import IMPLEMENTATIONS as SCALED
from .dense_dispatch import IMPLEMENTATIONS as DENSE
from .batched_dispatch import IMPLEMENTATIONS as BATCHED
from .specialized_dispatch import IMPLEMENTATIONS as SPECIALIZED, SCALED_IMPLEMENTATIONS as PACKED_SCALED
from .grouped_dispatch import IMPLEMENTATIONS as GROUPED

INTERFACES = {"scaled": SCALED + PACKED_SCALED, "batched": BATCHED, "dense": DENSE, "grouped": GROUPED}
INTERFACES.update(SPECIALIZED)
