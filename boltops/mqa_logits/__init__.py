"""MQA logits operator package (sparse-attention indexer scoring).

Public signature and ship-time / runtime selection live here.

* docs: ``docs/ops/mqa_logits.md``
* tests: ``test/mqa_logits/``

Public operators:

* ``fp8_mqa_logits`` — prefill / dense KV
"""

from __future__ import annotations

from typing import Optional

import torch


def fp8_mqa_logits(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
    impl: str = "triton",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """FP8 MQA logits for sparse-attention top-k (prefill / non-paged).

    ``impl`` selects the framework backend (currently Triton only):

    * ``\"triton\"`` (default) / ``None`` / ``\"auto\"`` — Triton

    Within Triton, ``backend`` selects the kernel path:

    * ``None`` / ``\"auto\"`` — shape + capability: Gluon → ``row``|``gluon``;
      else ``row``|``bn``
    * ``\"row\"`` / ``\"bn\"`` / ``\"gluon\"`` — force that path (``gluon``
      raises if unavailable)

    ``D_out``: optional preallocated ``[M, N]`` float32 buffer written in-place.
    """
    impl_norm = None if impl is None else str(impl).strip().lower()
    if impl_norm in (None, "", "auto", "triton"):
        return _fp8_mqa_logits_triton(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            D_out=D_out,
            backend=backend,
        )
    raise ValueError(f"impl must be None/'auto'/'triton', got {impl!r}")


def _fp8_mqa_logits_triton(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    *,
    D_out,
    backend,
):
    from boltops.mqa_logits.triton.mqa_config_utils import get_fp8_mqa_shape_dispatch

    seq_len = int(Q.shape[0])
    seq_len_kv = int(KV.shape[0])
    num_heads = int(Q.shape[1])

    backend_norm = None if backend is None else str(backend).strip().lower()
    if backend_norm in (None, "", "auto"):
        gluon_ok = False
        try:
            from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
                gluon_prefill_available,
            )

            gluon_ok = bool(gluon_prefill_available())
        except ImportError:
            gluon_ok = False

        if gluon_ok:
            from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
                fp8_mqa_logits_gluon,
            )

            return fp8_mqa_logits_gluon(
                Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out
            )

        disp = get_fp8_mqa_shape_dispatch(
            seq_len, seq_len_kv, num_heads, gluon_available=False
        )
        return _run_triton_prefill(
            Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out, disp=disp
        )

    if backend_norm == "gluon":
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
            fp8_mqa_logits_gluon,
            gluon_prefill_available,
        )

        if not gluon_prefill_available():
            raise RuntimeError("backend='gluon' requested but Gluon is unavailable")
        return fp8_mqa_logits_gluon(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            D_out=D_out,
            force_gluon_kernel=True,
        )

    if backend_norm == "row":
        # Apply the same shape-tuned row knobs auto uses on tiny KV
        # (block_kv / warps / stages). Mid/large forced-row keeps JSON defaults.
        disp = get_fp8_mqa_shape_dispatch(
            seq_len, seq_len_kv, num_heads, gluon_available=True
        )
        if disp.get("path") == "row":
            return _run_triton_prefill(
                Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out, disp=disp
            )
        from boltops.mqa_logits.triton.fp8_mqa_logits import fp8_mqa_logits as impl

        return impl(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            auto_bn_tile=False,
            D_out=D_out,
        )

    if backend_norm == "bn":
        from boltops.mqa_logits.triton.mqa_config_utils import get_bn_tile_dispatch

        # Force BN geometry even on tiny shapes that auto would send to row.
        bn = get_bn_tile_dispatch(seq_len, seq_len_kv, num_heads)
        disp = {"path": "bn", **bn}
        return _run_triton_prefill(
            Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out, disp=disp
        )

    raise ValueError(
        f"backend must be None/'auto'/'row'/'bn'/'gluon', got {backend!r}"
    )


def _run_triton_prefill(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    *,
    D_out,
    disp: dict,
):
    from boltops.mqa_logits.triton.fp8_mqa_logits import fp8_mqa_logits as impl

    if disp["path"] == "row":
        row_kw = {
            "auto_bn_tile": False,
            "D_out": D_out,
            "head_split_k": int(disp.get("head_split_k", 1)),
        }
        for key in ("block_kv", "num_warps", "num_stages"):
            if key in disp:
                row_kw[key] = int(disp[key])
        return impl(Q, KV, kv_scales, weights, cu_starts, cu_ends, **row_kw)
    return impl(
        Q,
        KV,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        auto_bn_tile=False,
        D_out=D_out,
        bn_tile=int(disp["bn_tile"]),
        block_m_tile=int(disp["block_m_tile"]),
        head_split_k=int(disp["head_split_k"]),
        num_warps=int(disp["num_warps"]),
        num_stages=int(disp["num_stages"]),
        waves_per_eu=int(disp["waves_per_eu"]),
        matrix_instr_nonkdim=int(disp["matrix_instr_nonkdim"]),
        kpack=int(disp["kpack"]),
        schedule_hint=str(disp["schedule_hint"]),
        use_block_pingpong=bool(disp["use_block_pingpong"]),
    )


__all__ = [
    "fp8_mqa_logits",
]
