"""MQA logits operator package (sparse-attention indexer scoring).

Public Triton entry lives here; TileLang can be added later as a sibling
export (e.g. ``tilelang_mqa_logits``) without renaming this API.

* docs: ``docs/ops/mqa_logits.md``
* tests: ``test/mqa_logits/``

Public operator:

* ``triton_mqa_logits`` — prefill / dense KV (Triton)
  - ``kv_scales`` provided → float8 path
  - ``kv_scales is None`` → fp16/bf16 path
"""

from __future__ import annotations

from typing import Optional

import torch

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    name = str(dtype)
    return "float8_e4m3" in name


def _is_half_dtype(dtype: torch.dtype) -> bool:
    return dtype in _HALF_DTYPES


def triton_mqa_logits(
    Q: torch.Tensor,
    KV: torch.Tensor,
    kv_scales: Optional[torch.Tensor],
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """MQA logits for sparse-attention top-k (prefill / non-paged, Triton).

    Path selection is by ``kv_scales``:

    * ``kv_scales`` is a float32 tensor → float8 Q/KV path
    * ``kv_scales is None`` → fp16/bf16 path (dedicated half kernels)

    ``Q`` / ``KV`` must share dtype and match the chosen path.

    ``backend`` selects the Triton kernel path:

    * ``None`` / ``\"auto\"`` — float8: Gluon → ``row``|``gluon``, else
      ``row``|``bn``; half: ``row``|``bn`` (row only on short-KV shapes where
      tune beats BN; else full-size tuned BN).
    * ``\"row\"`` / ``\"bn\"`` / ``\"gluon\"`` — float8 as named; half ``row``
      uses per-row KV scan, half ``bn`` forces BN; half ``gluon`` unsupported
      (use experimental ``gluon_bn_align``).
    * ``\"gluon_bn_align\"`` / ``\"gluon-exp\"`` / ``\"gluon_exp\"`` —
      **experimental** BN-aligned Gluon (float8 or half); not selected by auto.
    * ``\"gluon_opt\"`` — **experimental** float8 alias of ship Gluon
      (ablation-backed); prefer ``\"gluon\"`` in product code.

    ``D_out``: optional preallocated ``[M, N]`` float32 buffer written in-place.
    """
    return _triton_mqa_logits(
        Q,
        KV,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        D_out=D_out,
        backend=backend,
    )


def _triton_mqa_logits(
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

    if Q.dtype != KV.dtype:
        raise ValueError(
            f"Q and KV must share dtype, got Q={Q.dtype} KV={KV.dtype}"
        )

    use_fp8 = kv_scales is not None
    if use_fp8:
        if not _is_fp8_dtype(Q.dtype):
            raise ValueError(
                "kv_scales is set → float8 path requires float8_e4m3* Q/KV, "
                f"got {Q.dtype}"
            )
    else:
        if not _is_half_dtype(Q.dtype):
            raise ValueError(
                "kv_scales is None → half path requires float16/bfloat16 Q/KV, "
                f"got {Q.dtype}"
            )

    seq_len = int(Q.shape[0])
    seq_len_kv = int(KV.shape[0])
    num_heads = int(Q.shape[1])
    backend_norm = None if backend is None else str(backend).strip().lower()

    if not use_fp8:
        if backend_norm == "gluon":
            raise ValueError(
                "backend='gluon' does not support float16/bfloat16 inputs yet; "
                "use backend=None/'auto'/'row'/'bn' or 'gluon_bn_align'"
            )
        if backend_norm in ("gluon_bn_align", "gluon-exp", "gluon_exp"):
            from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
                gluon_prefill_available,
            )
            from boltops.mqa_logits.triton.mqa_logits_half_gluon_bn_align import (
                mqa_logits_half_gluon_bn_align,
            )

            if not gluon_prefill_available():
                raise RuntimeError(
                    "backend='gluon_bn_align' requested but Gluon is unavailable"
                )
            return mqa_logits_half_gluon_bn_align(
                Q, KV, weights, cu_starts, cu_ends, D_out=D_out
            )
        if backend_norm == "row":
            from boltops.mqa_logits.triton.mqa_logits_half_row import (
                mqa_logits_half_row,
            )

            return mqa_logits_half_row(
                Q, KV, weights, cu_starts, cu_ends, D_out=D_out
            )
        if backend_norm not in (None, "", "auto", "bn"):
            raise ValueError(
                "backend must be None/'auto'/'row'/'bn'/'gluon'/"
                f"'gluon_bn_align', got {backend!r}"
            )
        if backend_norm in (None, "", "auto"):
            from boltops.mqa_logits.triton.mqa_config_utils import (
                get_half_mqa_shape_dispatch,
            )

            disp = get_half_mqa_shape_dispatch(seq_len, seq_len_kv, num_heads)
            if disp.get("path") == "row":
                from boltops.mqa_logits.triton.mqa_logits_half_row import (
                    mqa_logits_half_row,
                )

                row_kw = {"D_out": D_out}
                for key in (
                    "block_kv",
                    "num_warps",
                    "num_stages",
                    "waves_per_eu",
                    "head_split_k",
                    "clear_block_kv",
                    "clear_num_warps",
                    "clear_num_stages",
                ):
                    if key in disp:
                        row_kw[key] = int(disp[key])
                return mqa_logits_half_row(
                    Q, KV, weights, cu_starts, cu_ends, **row_kw
                )
        from boltops.mqa_logits.triton.mqa_logits_half import mqa_logits_half

        return mqa_logits_half(
            Q, KV, weights, cu_starts, cu_ends, D_out=D_out
        )

    # float8 path (kv_scales provided)
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

    if backend_norm in ("gluon_bn_align", "gluon-exp", "gluon_exp"):
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon_bn_align import (
            fp8_mqa_logits_gluon_bn_align,
        )
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
            gluon_prefill_available,
        )

        if not gluon_prefill_available():
            raise RuntimeError(
                "backend='gluon_bn_align' requested but Gluon is unavailable"
            )
        return fp8_mqa_logits_gluon_bn_align(
            Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out
        )

    if backend_norm in ("gluon_opt", "gluon-opt"):
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon import (
            gluon_prefill_available,
        )
        from boltops.mqa_logits.triton.fp8_mqa_logits_gluon_opt import (
            fp8_mqa_logits_gluon_opt,
        )

        if not gluon_prefill_available():
            raise RuntimeError("backend='gluon_opt' requested but Gluon is unavailable")
        return fp8_mqa_logits_gluon_opt(
            Q, KV, kv_scales, weights, cu_starts, cu_ends, D_out=D_out
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
        "backend must be None/'auto'/'row'/'bn'/'gluon'/'gluon_bn_align'/'gluon_opt', "
        f"got {backend!r}"
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
    "triton_mqa_logits",
]
