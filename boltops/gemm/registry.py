# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Implementation registry independent of operator family, dtype and side tensors."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .descriptors import GemmProblem


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    source_config: str | None = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.block_m, self.block_n, self.block_k)):
            raise ValueError("kernel block sizes must be positive integers")


@dataclass(frozen=True)
class Implementation:
    name: str
    family: str
    module: str
    entry: str
    configurations: Callable[[GemmProblem], tuple[KernelConfig, ...]]
    supports: Callable[[GemmProblem, KernelConfig], bool]
    backend: str

    def matches(self, problem, config):
        return self.family == problem.kind and self.supports(problem, config)


@dataclass(frozen=True)
class ExecutionPlan:
    implementation: Implementation
    config: KernelConfig


def select_plan(problem, implementations=None, *, backend="auto"):
    from .catalog import BACKENDS, available, implementations as load, validate_backend
    validate_backend(backend)
    interface = problem.interface
    if implementations is None:
        candidates = []
        for owner in BACKENDS if backend == "auto" else (backend,):
            registered = load(owner, interface)
            if not registered:
                continue
            if not available(owner):
                if backend != "auto":
                    raise ImportError(f"GEMM backend {owner} is not installed")
                continue
            candidates.extend(registered)
    else:
        candidates = [impl for impl in implementations
                      if backend == "auto" or impl.backend == backend]
    # Consult this interface's map across all backends before defaults.
    # Compilation/execution failures are never caught to select a fallback.
    from .performance_map import tuned_configs
    for implementation in candidates:
        for config in tuned_configs(problem, implementation.name, backend=implementation.backend):
            if implementation.matches(problem, config):
                return ExecutionPlan(implementation, config)
    for implementation in candidates:
        if implementation.family != problem.kind:
            continue
        for config in implementation.configurations(problem):
            if implementation.matches(problem, config):
                return ExecutionPlan(implementation, config)
    raise NotImplementedError(
        f"no registered {problem.kind} GEMM implements {problem} (interface={interface}, backend={backend})")


def launch(plan, problem, **tensors):
    implementation = plan.implementation
    if not implementation.matches(problem, plan.config):
        raise ValueError(f"configuration {plan.config} is incompatible with {implementation.name}")
    module = import_module(implementation.module)
    getattr(module, implementation.entry)(plan.config, problem, **tensors)
