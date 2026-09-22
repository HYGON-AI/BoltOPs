"""Local replacements for SGLang runtime dependencies.

KV strides originate in attention/decode_attention.py, aux handling in
attention/score_mod.py, and platform/environment helpers in srt utils/environ.
The 4-D paged-cache branch is a BoltOPs extension to the upstream 3-D helper.
"""

from __future__ import annotations

import os
import types

import triton


_HCU_ARCHS = frozenset({"gfx926", "gfx928", "gfx936", "gfx938", "gfx92a", "gfx946"})


def _target():
    return triton.runtime.driver.active.get_current_target()


def is_cuda() -> bool:
    return _target().backend == "cuda"


def is_hip() -> bool:
    return _target().backend == "hip"


def is_hcu() -> bool:
    target = _target()
    return target.backend == "hip" and target.arch.split(":", 1)[0] in _HCU_ARCHS


def is_gfx95_supported() -> bool:
    return is_hip() and _target().arch.startswith("gfx95")


def is_gfx1250_supported() -> bool:
    return is_hip() and _target().arch.startswith("gfx1250")


def _extract_kv_strides(buf, page_size: int):
    # Keep upstream slot indexing while accepting explicit page/token strides.
    if buf.ndim == 4:
        if buf.shape[1] != page_size:
            raise ValueError(
                f"4-D KV buffer dim 1 must equal page_size; got {buf.shape} "
                f"and page_size={page_size}"
            )
        page_stride = buf.stride(0)
        tok_stride = buf.stride(1)
        head_stride = buf.stride(2)
        slot_stride = page_stride // page_size
    elif buf.ndim == 3:
        slot_stride = buf.stride(0)
        head_stride = buf.stride(1)
        page_stride = slot_stride * page_size
        tok_stride = slot_stride
    else:
        raise ValueError(f"unexpected KV buffer ndim={buf.ndim}, shape={buf.shape}")
    return slot_stride, head_stride, page_stride, tok_stride


def unpack_aux_tensors(score_mod, aux_tensors):
    if score_mod is None:
        return None, 0, 0, 0
    assert aux_tensors is not None and len(aux_tensors) == 1, (
        "Triton score_mod requires exactly one aux tensor"
    )
    aux0 = aux_tensors[0]
    assert aux0.dim() == 3 and aux0.stride(2) == 1, (
        "aux_tensors[0] must be 3D with a contiguous last dimension"
    )
    return aux0, aux0.stride(0), aux0.stride(1), aux0.shape[2]


class _EnvBool:
    def __init__(self, name: str, default: bool):
        self.name = name
        self.default = default

    def get(self) -> bool:
        value = os.environ.get(self.name)
        if value is None:
            return self.default
        return value.lower() not in {"0", "false", "no", "off"}


envs = types.SimpleNamespace(
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION=_EnvBool(
        "BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION", True
    ),
    SGLANG_TRITON_FP8_PREFILL_ATTN=_EnvBool(
        "BOLT_OPS_TRITON_FP8_PREFILL_ATTN", True
    ),
)
