# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triangle attention operator family."""

def triangle_attention_fwd(q, k, v, biases):
    """Compute production triangle attention for q/k/v shaped ``[B, H, N, D]``.

    BF16 ``q`` must be pre-scaled by ``1/sqrt(D)``. ``biases`` is ordered as
    ``[mask_bias, triangle_bias]``. Floating-point biases retain their input
    dtype and are converted to FP32 for accumulation. The result is
    returned as a newly allocated tensor.

    ``mask_bias`` uses ``0`` for visible keys and finite ``-1e9`` for masked
    keys, stored as BF16, FP32, or FP64. Each query must have a visible key;
    negative-infinity input masks are unsupported.
    """
    from .triton.triangle_attention import triangle_attention_fwd as impl
    return impl(q, k, v, biases)


__all__ = ["triangle_attention_fwd"]
