"""Chunk-local cumulative sum helpers used by the KDA prefill orchestration.

Port of the vLLM/FLA ``chunk_local_cumsum`` family with the repository's
static-config convention: no runtime autotuning, a small set of pre-selected
configs per kernel and a shape-based selector.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import prepare_chunk_indices

CHUNK_LOCAL_CUMSUM_SCALAR_CONFIGS: tuple[dict[str, int], ...] = (
    {"num_warps": 1},
    {"num_warps": 2},
    {"num_warps": 4},
)

CHUNK_LOCAL_CUMSUM_VECTOR_CONFIGS: tuple[dict[str, int], ...] = (
    {"BS": 32, "num_warps": 4},
    {"BS": 64, "num_warps": 4},
    {"BS": 128, "num_warps": 4},
)


def get_chunk_local_cumsum_scalar_config(
    T: int, H: int, is_varlen: bool
) -> dict[str, int]:
    del T, H, is_varlen
    return CHUNK_LOCAL_CUMSUM_SCALAR_CONFIGS[0]


def get_chunk_local_cumsum_vector_config(
    T: int, H: int, S: int, is_varlen: bool
) -> dict[str, int]:
    """Pick the widest S-block the row affords.

    The kernel is purely bandwidth-bound, and ``BS`` decides how much of each
    row is read in one contiguous run: the rows are ``H * S`` apart, so a
    narrow block turns every row into its own short burst.  At ``S=128``,
    ``T=16384, H=64`` on BW1100 the measured cost is 2.64 ms at ``BS=32``,
    1.98 ms at 64 and 1.71 ms at 128 (407 / 542 / 627 GB/s); warp count and
    stage count are flat to within 1% at every width, so only ``BS`` is tuned.
    Replacing the ``dot(tril, s)`` scan with ``tl.cumsum`` was measured within
    noise of the matmul, so the mask-matmul formulation is kept.
    """
    del T, H, is_varlen
    if S >= 128:
        return CHUNK_LOCAL_CUMSUM_VECTOR_CONFIGS[2]
    if S >= 64:
        return CHUNK_LOCAL_CUMSUM_VECTOR_CONFIGS[1]
    return CHUNK_LOCAL_CUMSUM_VECTOR_CONFIGS[0]


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    output_scale,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    b_o = b_o * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    output_scale,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False) * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    cumsum_scale: float = 1.0,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    config = get_chunk_local_cumsum_scalar_config(T, H, cu_seqlens is not None)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        cumsum_scale=cumsum_scale,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        output_scale=output_scale,
        **config,
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    cumsum_scale: float = 1.0,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    config = get_chunk_local_cumsum_vector_config(
        T, H, S, cu_seqlens is not None
    )

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    chunk_local_cumsum_vector_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        cumsum_scale=cumsum_scale,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        output_scale=output_scale,
        **config,
    )
    return g


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    cumsum_scale: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """``cumsum_scale`` scales the result inside the kernel.

    KDA needs the log2-domain gate, i.e. the running sum times ``1/ln 2``.
    Doing that as a torch multiply costs a full extra read-modify-write of an
    FP32 ``[B,T,H,K]`` tensor -- 0.61 ms at ``T=16384, H=64, K=128`` on
    BW1100 -- while folding it into the store is free.
    """
    if cu_seqlens is not None:
        assert g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g,
            chunk_size,
            reverse,
            cu_seqlens,
            chunk_indices,
            head_first,
            output_dtype,
            cumsum_scale,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g,
            chunk_size,
            reverse,
            cu_seqlens,
            chunk_indices,
            head_first,
            output_dtype,
            cumsum_scale,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}. "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
