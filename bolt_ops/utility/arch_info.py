"""Hygon / Triton architecture queries."""

from __future__ import annotations

import torch
import triton

# Per-arch LDS / shared-memory capacity (bytes). Unknown arch -> 64KiB.
_LDS_CAP_BYTES = {
    "gfx1250": 327680,
    "gfx950": 163840,
    "gfx942": 65536,
    "gfx946": 65536,
    "gfx948": 65536,
    "gfx938": 65536,
    "gfx936": 65536,
    "gfx928": 65536,
    "gfx926": 65536,
    "gfx92a": 65536,
}


def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    return current_device.multi_processor_count


def get_lds_capacity_bytes(arch: str | None = None) -> int:
    """LDS / shared-memory capacity in bytes for launch sizing."""
    return int(_LDS_CAP_BYTES.get(arch or get_arch(), 65536))


def is_fp4_avail():
    return get_arch() in ("gfx946",)


def is_fp8_avail():
    return get_arch() in ("gfx938", "gfx92a", "gfx946")


def is_mls_avail():
    return get_arch() in ("gfx938", "gfx92a", "gfx946")


def get_fp8_dtypes():
    return torch.float8_e5m2, torch.float8_e4m3fn


def get_fp8_e4m3_dtype():
    return torch.float8_e4m3fn


