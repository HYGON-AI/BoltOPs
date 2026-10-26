# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon in 2026 for BoltOPs DCU/HCU integration.
"""Helpers used by vLLM ``TritonMLA`` decode scheduling."""

from __future__ import annotations

import triton

from boltops.utility.arch_info import get_num_sms

# Matches vLLM ``triton_mla.py``.
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2


def compute_num_kv_splits(
    max_seq_len: int,
    sm_count: int | None = None,
) -> int:
    """Pick split-KV count for MLA decode (power-of-two, SM-capped).

    Same formula as vLLM ``TritonMLAImpl`` / ``_compute_num_kv_splits``.
    """
    if sm_count is None:
        sm_count = get_num_sms()
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)
