# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import SimpleNamespace

from boltops.mhc.tilelang import _dispatch


def _resolve_with_arch(monkeypatch, arch: str):
    imports: list[str] = []
    sentinel = object()
    available = {"boltops.mhc.tilelang.kernels.common.pre_big_fuse"}

    monkeypatch.setattr(
        _dispatch,
        "get_device_info",
        lambda _device: SimpleNamespace(arch=arch),
    )

    def fake_import(module_name: str):
        imports.append(module_name)
        return SimpleNamespace(mhc_pre_big_fuse=sentinel)

    monkeypatch.setattr(_dispatch, "import_module", fake_import)
    monkeypatch.setattr(
        _dispatch,
        "find_spec",
        lambda module_name: object() if module_name in available else None,
    )
    _dispatch._resolve_for_arch.cache_clear()
    resolved = _dispatch.resolve("mhc_pre_big_fuse", "hcu:0")
    return resolved, sentinel, imports, available


def test_known_arch_without_override_falls_back_to_common(monkeypatch):
    resolved, sentinel, imports, _ = _resolve_with_arch(monkeypatch, "gfx946")

    assert resolved is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.common.pre_big_fuse"]


def test_unknown_arch_falls_back_to_common(monkeypatch):
    resolved, sentinel, imports, _ = _resolve_with_arch(monkeypatch, "gfx999")

    assert resolved is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.common.pre_big_fuse"]


def test_arch_family_override_takes_precedence(monkeypatch):
    _, sentinel, imports, available = _resolve_with_arch(monkeypatch, "gfx946")
    available.add("boltops.mhc.tilelang.kernels.gfx94x.pre_big_fuse")
    imports.clear()
    _dispatch._resolve_for_arch.cache_clear()
    resolved = _dispatch.resolve("mhc_pre_big_fuse", "hcu:0")

    assert resolved is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.gfx94x.pre_big_fuse"]


def test_arch_exact_override_takes_precedence(monkeypatch):
    _, sentinel, imports, available = _resolve_with_arch(monkeypatch, "gfx946")
    available.add("boltops.mhc.tilelang.kernels.gfx946.pre_big_fuse")
    imports.clear()
    _dispatch._resolve_for_arch.cache_clear()
    resolved = _dispatch.resolve("mhc_pre_big_fuse", "hcu:0")

    assert resolved is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.gfx946.pre_big_fuse"]


def test_arch_family_name_falls_back_to_common(monkeypatch):
    resolved, sentinel, imports, _ = _resolve_with_arch(monkeypatch, "gfx93x")

    assert resolved is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.common.pre_big_fuse"]


def test_resolved_implementation_is_cached(monkeypatch):
    resolved, sentinel, imports, _ = _resolve_with_arch(monkeypatch, "gfx946")
    resolved_again = _dispatch.resolve("mhc_pre_big_fuse", "hcu:0")

    assert resolved is sentinel
    assert resolved_again is sentinel
    assert imports == ["boltops.mhc.tilelang.kernels.common.pre_big_fuse"]
