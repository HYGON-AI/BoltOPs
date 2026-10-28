# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

from boltops.ihc.tilelang import _dispatch


def test_unknown_arch_falls_back_to_common(monkeypatch):
    imports: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        _dispatch, "get_device_info", lambda _device: SimpleNamespace(arch="gfx938")
    )

    def fake_import(name: str):
        imports.append(name)
        return SimpleNamespace(ihc_post=sentinel)

    monkeypatch.setattr(_dispatch, "import_module", fake_import)
    monkeypatch.setattr(
        _dispatch,
        "find_spec",
        lambda name: object() if name.endswith("kernels.common.post") else None,
    )
    _dispatch._resolve_for_arch.cache_clear()
    try:
        resolved = _dispatch.resolve("ihc_post", "hcu:0")
        assert resolved is sentinel
        assert imports == ["boltops.ihc.tilelang.kernels.common.post"]
    finally:
        _dispatch._resolve_for_arch.cache_clear()
