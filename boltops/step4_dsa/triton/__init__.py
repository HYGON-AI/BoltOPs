# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton backend for the Step-4 DSA family.

Public callers go through ``boltops.step4_dsa`` (lazy per-op imports). This
package has no re-exports so importing one kernel module does not load the rest.
"""
