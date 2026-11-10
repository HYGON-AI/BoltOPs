# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Aiter-style architecture/CU/backend GEMM configuration trees."""

from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path

BACKENDS = ("triton", "gluon", "tilelang")
_CONFIG_DIR = Path(__file__).with_name("configs")


def validate_backend(backend):
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"unknown GEMM backend {backend!r}")


@lru_cache(maxsize=1024)
def _load_json(path):
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be an object: {path}")
    groups = data.get("profiles")
    if groups is not None:
        if set(data) != {"profiles"} or not isinstance(groups, dict):
            raise ValueError(f"profiles must be a dictionary of M buckets: {path}")
        if any(
            not name or not isinstance(value, dict) for name, value in groups.items()
        ):
            raise ValueError(f"each profile requires a name and M buckets: {path}")
        groups = groups.values()
    else:
        groups = [data]
    for buckets in groups:
        for key, value in buckets.items():
            if key == "DEFAULT_FALLBACK":
                if type(value) is not bool:
                    raise ValueError("DEFAULT_FALLBACK must be a boolean")
                continue
            if key != "any" and not key.startswith(("M_EQ_", "M_LEQ_", "M_GEQ_")):
                raise ValueError(f"unknown M bucket {key}: {path}")
            if key != "any":
                bound = key.split("_", 2)[2]
                if not bound.isdigit() or int(bound) <= 0:
                    raise ValueError(f"invalid M bucket {key}: {path}")
            if not isinstance(value, dict):
                raise ValueError(f"bucket {key} must be a parameter dictionary: {path}")
            for dim in ("M", "N", "K"):
                tile = value.get("BLOCK_SIZE_" + dim)
                if tile is not None and (type(tile) is not int or tile <= 0):
                    raise ValueError(f"BLOCK_SIZE_{dim} must be a positive integer")
    return data


def resolve_config_dir(config_name, *, backend="triton", arch=None, cu_num=None):
    """Choose one complete CU-specific or architecture tree, as aiter does."""
    validate_backend(backend)
    if backend == "auto":
        raise ValueError("the operation must select a backend before config lookup")
    if arch is None:
        from boltops.utility.arch_info import get_device_info

        device = get_device_info()
        arch, cu_num = device.arch, device.cu_count if cu_num is None else cu_num
    for component in (config_name, arch):
        if (
            not isinstance(component, str)
            or not component
            or "/" in component
            or "\\" in component
            or ".." in component
        ):
            raise ValueError("invalid configuration path component")
    if cu_num is not None and (type(cu_num) is not int or cu_num <= 0):
        raise ValueError("cu_num must be a positive integer or None")
    roots = (
        [Path(os.environ["BOLTOPS_GEMM_CONFIG_DIR"])]
        if os.environ.get("BOLTOPS_GEMM_CONFIG_DIR")
        else []
    )
    roots.append(_CONFIG_DIR)
    family = Path(backend) / config_name.lower().replace("-", "_")
    for root in roots:
        if cu_num is not None:
            directory = root / arch / f"{cu_num}cu" / family
            if _load_json(str(directory / "DEFAULT.json")) is not None:
                return directory
        directory = root / arch / family
        if _load_json(str(directory / "DEFAULT.json")) is not None:
            return directory
    raise FileNotFoundError(f"no DEFAULT.json for {arch}/{cu_num}cu/{family}")


@lru_cache(maxsize=1024)
def _get_cached(
    directory, config_name, M, N, K, specialized_filename, B, layout=None, profile=None
):
    data = _load_json(str(Path(directory) / "DEFAULT.json"))
    if "profiles" in data and profile is None:
        raise ValueError(f"{config_name} requires a profile")
    if profile is not None:
        data = data.get("profiles", {}).get(profile)
    suffixes = [specialized_filename] if specialized_filename is not None else []
    if specialized_filename is None and N is not None and K is not None:
        if B is not None:
            suffixes.append(f"B={B}-N={N}-K={K}" + (f"-{layout}" if layout else ""))
        suffixes.append(f"N={N}-K={K}" + (f"-{layout}" if layout else ""))
    tuned_keys = set()
    for suffix in suffixes:
        if "/" in suffix or "\\" in suffix or ".." in suffix:
            raise ValueError("invalid specialized filename")
        specialized = _load_json(str(Path(directory) / f"{config_name}-{suffix}.json"))
        if specialized is not None and "profiles" in specialized and profile is None:
            raise ValueError(f"{config_name} requires a profile")
        if specialized is not None and profile is not None:
            specialized = specialized.get("profiles", {}).get(profile)
        if specialized is not None:
            tuned_keys = set(specialized)
            data = (
                {**(data or {}), **specialized}
                if specialized.get("DEFAULT_FALLBACK")
                else specialized
            )
            break
    if data is None:
        raise KeyError(f"no configuration profile {profile!r} for {config_name}")
    exact = f"M_EQ_{M}"
    if exact in data:
        return data[exact], exact in tuned_keys
    for prefix, reverse in (("M_LEQ_", False), ("M_GEQ_", True)):
        for bound in sorted(
            (int(key[6:]) for key in data if key.startswith(prefix)), reverse=reverse
        ):
            if (not reverse and M <= bound) or (reverse and M >= bound):
                key = f"{prefix}{bound}"
                return data[key], key in tuned_keys
    if "any" in data:
        return data["any"], False
    raise KeyError(f"no M bucket for {config_name}, M={M}, N={N}, K={K}, B={B}")


def get_gemm_config(
    config_name,
    M,
    N=None,
    K=None,
    specialized_filename=None,
    backend="triton",
    B=None,
    cu_num=None,
    arch=None,
    layout=None,
    profile=None,
):
    """Return a private parameter dict and specialized-bucket flag.

    File priority: custom suffix, otherwise B/N/K then N/K. Within a file:
    exact M_EQ, ascending M_LEQ, descending M_GEQ, then any. No cross-backend fallback.
    """
    if (
        type(M) is not int
        or M < 0
        or any(v is not None and (type(v) is not int or v <= 0) for v in (N, K, B))
    ):
        raise ValueError("invalid GEMM shape")
    if layout is not None and layout not in ("TN", "TT", "NN", "NT"):
        raise ValueError("layout must be TN, TT, NN or NT")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise ValueError("profile must be a nonempty string or None")
    directory = resolve_config_dir(
        config_name, backend=backend, arch=arch, cu_num=cu_num
    )
    params, tuned = _get_cached(
        str(directory),
        config_name,
        M,
        N,
        K,
        specialized_filename,
        B,
        layout,
        profile,
    )
    return deepcopy(params), tuned


def clear_cache():
    _load_json.cache_clear()
    _get_cached.cache_clear()
