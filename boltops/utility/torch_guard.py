# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Register custom ops so ``torch.compile`` can trace them.

Origin: ported from ROCm aiter ``aiter/jit/utils/torch_guard.py``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Union

from packaging import version
from packaging.version import Version

boltops_lib = None


def is_torch_equal_or_newer(target: str) -> bool:
    """Return True if installed torch version is >= ``target`` (e.g. ``\"2.6.0\"``)."""
    import torch

    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO (needed by doc gen without a full torch import).
        return Version(importlib.metadata.version("torch")) >= Version(target)


def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


def torch_compile_guard(
    mutates_args: Union[list[str], str] = "unknown",
    device: str = "cpu",
    calling_func_: Optional[Callable[..., Any]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
):
    """Wrap ``func`` as ``torch.ops.boltops.<name>`` with optional ``gen_fake``."""

    def decorator(func):
        calling_func = calling_func_ if calling_func_ is not None else func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        try:
            import inspect

            import torch
            from torch.library import Library  # noqa: F401
        except ImportError:
            return wrapper

        def infer_op_schema(calling_func):
            import inspect

            import torch

            global boltops_lib
            boltops_lib = (
                Library("boltops", "FRAGMENT") if boltops_lib is None else boltops_lib
            )
            sig = inspect.signature(calling_func)
            if hasattr(torch.library, "infer_schema"):
                return torch.library.infer_schema(
                    calling_func, mutates_args=mutates_args
                )

            # PyTorch 2.4 fallback
            import torch._custom_op.impl

            mutates_args_custom = mutates_args
            if mutates_args == "unknown":
                mutates_args_custom = [
                    param_name
                    for param_name, param in sig.parameters.items()
                    if param.annotation == torch.Tensor
                ]
            return torch._custom_op.impl.infer_schema(
                calling_func, mutates_args_custom
            )

        schema = infer_op_schema(calling_func)

        sig = inspect.signature(calling_func)
        input_is_tensor = False
        parameters = list(sig.parameters.values())

        if parameters:
            first_param = parameters[0]
            if (
                first_param.annotation is not inspect.Parameter.empty
                and first_param.annotation is torch.Tensor
            ):
                input_is_tensor = True

        input_part, output_part = schema.split("->", 1)
        if input_is_tensor:
            new_input = input_part
        elif not sig.parameters:
            new_input = "(Tensor dummy)"
        else:
            new_input = "(Tensor dummy, " + input_part[1:]

        return_non_tensor = False
        return_annotation = sig.return_annotation
        if return_annotation in [int, bool, float]:
            output_part = "(Tensor, " + output_part + ")"
            return_non_tensor = True

        schema = f"{new_input} -> {output_part}".strip()

        loadName = calling_func.__name__

        def wrapper_custom(*args, **kwargs):
            result = (
                getattr(torch.ops.boltops, f"{loadName}")(*args, **kwargs)
                if input_is_tensor
                else getattr(torch.ops.boltops, f"{loadName}")(
                    torch.empty(1, device=device), *args, **kwargs
                )
            )
            return result[1] if return_non_tensor else result

        if hasattr(torch.ops.boltops, loadName):
            return wrapper_custom

        def abstract_impl(*args, **kwargs):
            if gen_fake is not None:
                if return_non_tensor:
                    return torch.empty(1, device=device), gen_fake(*args, **kwargs)
                return gen_fake(*args, **kwargs)
            if return_non_tensor:
                return torch.empty(1, device=device), calling_func(*args, **kwargs)
            return calling_func(*args, **kwargs)

        def outer_wrapper(*args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_non_tensor
                else (torch.empty(1, device=device), wrapper(*args, **kwargs))
            )

        def abstract_impl_dummy(dummy, *args, **kwargs):
            if gen_fake is not None:
                if return_non_tensor:
                    return torch.empty(1, device=device), gen_fake(*args, **kwargs)
                return gen_fake(*args, **kwargs)
            if return_non_tensor:
                return torch.empty(1, device=device), calling_func(*args, **kwargs)
            return calling_func(*args, **kwargs)

        def outer_wrapper_dummy(dummy, *args, **kwargs):
            return (
                wrapper(*args, **kwargs)
                if not return_non_tensor
                else (torch.empty(1, device=device), wrapper(*args, **kwargs))
            )

        custom_func = outer_wrapper
        fake_func = abstract_impl
        if not input_is_tensor:
            custom_func = outer_wrapper_dummy
            fake_func = abstract_impl_dummy

        if not hasattr(torch.ops.boltops, calling_func.__name__):
            tags = (
                ()
                if is_torch_equal_or_newer("2.8.0")
                else (torch.Tag.needs_fixed_stride_order,)
            )
            op_schema = f"boltops::{loadName}" + schema
            boltops_lib.define(op_schema, tags=tags)
            boltops_lib.impl(f"boltops::{loadName}", custom_func, dispatch_key="CUDA")
            boltops_lib.impl(f"boltops::{loadName}", custom_func, dispatch_key="CPU")
            boltops_lib._register_fake(f"{loadName}", fake_func)

        return wrapper_custom

    return decorator
