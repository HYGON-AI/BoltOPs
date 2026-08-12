"""Cached accelerator architecture and capability queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import torch

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


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    arch: str
    cu_count: int


def _normalize_arch(value: str) -> str | None:
    match = re.search(r"gfx[0-9a-f]+", value.lower())
    return match.group(0) if match else None


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            f"Accelerator metadata requires a CUDA/HCU device, got {resolved}"
        )
    return (
        torch.cuda.current_device()
        if resolved.index is None
        else resolved.index
    )


@lru_cache(maxsize=None)
def _get_device_info(index: int) -> DeviceInfo:
    props = torch.cuda.get_device_properties(index)
    candidates = (
        getattr(props, "gcnArchName", ""),
        getattr(props, "arch_name", ""),
        getattr(props, "name", ""),
    )
    arch = next(
        (arch for value in candidates if (arch := _normalize_arch(str(value)))),
        None,
    )
    if arch is None:
        raise RuntimeError(
            f"Unable to determine gfx architecture for device {index}: {props!r}"
        )
    return DeviceInfo(
        index=index,
        arch=arch,
        cu_count=props.multi_processor_count,
    )


def get_device_info(
    device: torch.device | str | int | None = None,
) -> DeviceInfo:
    """Return cached metadata for an explicit or current accelerator."""
    return _get_device_info(_device_index(device))


def get_arch(device: torch.device | str | int | None = None):
    return get_device_info(device).arch


def get_num_sms(device: torch.device | str | int | None = None):
    return get_device_info(device).cu_count


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
