# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Lazy, independently owned GEMM backend providers."""
from importlib import import_module
from importlib.util import find_spec

BACKENDS = ("triton", "gluon", "tilelang")


def validate_backend(backend):
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"unknown GEMM backend {backend!r}; expected auto, triton, gluon or tilelang")
    return backend


def implementations(backend, interface):
    validate_backend(backend)
    if backend == "auto":
        raise ValueError("a backend provider must be explicit")
    provider = import_module(f"boltops.gemm.{backend}.dispatch")
    result = provider.INTERFACES[interface]
    if any(impl.backend != backend for impl in result):
        raise ValueError(f"GEMM provider {backend} registered another backend's implementation")
    return result


def available(backend):
    # Probe only top-level packages: dotted find_spec imports parent packages.
    package = "triton" if backend == "gluon" else backend
    return find_spec(package) is not None
