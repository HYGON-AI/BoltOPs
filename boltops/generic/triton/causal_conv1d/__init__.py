"""Causal depthwise-convolution Triton operator.

Re-exports the public entry points so callers can use either
``boltops.generic`` (recommended) or this subpackage directly.

The implementation is split into ``core.py`` (public API and routing) and
``kernels.py`` (the two source-specific Triton kernel variants).
"""

from .core import causal_conv1d_fn, causal_conv1d_update

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
]
