"""Single-query MHA over FP8 KV; reuse the existing split-KV decode kernel."""

from __future__ import annotations

import math
import numbers

import torch

# The kernel takes 128-wide heads and computes int32 element offsets.
_HEAD_DIM = 128
_MAX_SPAN = 2**31


def _span(shape, stride):
    """Elements addressed from the first to the last element of a strided view."""
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride))


def _positive_scale(value, name, device):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype != torch.float32 or value.device != device:
            raise ValueError(f"{name} must be a scalar float32 tensor on {device}")
        number = value.item()
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
    else:
        raise TypeError(f"{name} must be a positive scalar or scalar float32 tensor")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return torch.tensor(number, device=device, dtype=torch.float32)


def decode_attention_fwd_4d(
    q, k, v, *, sm_scale, k_scale, v_scale,
    block_table=None, seq_lens=None, num_kv_splits=None,
):
    """Return BF16 ``[B,H,1,128]`` MHA output, reading FP8 KV directly.

    Without ``block_table``, K/V are logical ``[B,H,S,128]`` tensors.
    With a table they are physical ``[num_pages,page_size,H,128]`` buffers;
    ``seq_lens[B]`` is required and only used table entries are validated.
    Q is the last token and attends every valid cached token (including its
    own K/V, which the caller must already have written).

    Scales are explicit positive scalars (or same-device scalar FP32 tensors),
    with real K/V = FP8 values * scale. Per-head/block scales, tree masks and
    autograd are outside this contract. Validation reads metadata on the host;
    this allocating convenience entry is not CUDA Graph capture compatible.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.ndim != 4 or any(size == 0 for size in t.shape):
            raise ValueError(f"{name} must be nonempty and four-dimensional")
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA/HIP device as q")
        if t.stride(-1) != 1:
            raise ValueError(f"{name} must have unit last-dimension stride")
    if q.dtype != torch.bfloat16:
        raise TypeError("q must be BF16")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise TypeError("k/v must be torch.float8_e4m3fn (E4M3FN)")
    if q.shape[2:] != (1, _HEAD_DIM) or k.shape[-1] != _HEAD_DIM or v.shape != k.shape:
        raise ValueError("requires Q [B,H,1,128] and matching 128-dimensional K/V")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("decode_attention_fwd_4d does not support autograd")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, numbers.Real):
        raise TypeError("sm_scale must be a positive number")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be finite and positive")

    batch, heads = q.shape[:2]
    if _span(q.shape, q.stride()) >= _MAX_SPAN:
        raise ValueError("Q stride span exceeds kernel int32 indexing capacity")
    if block_table is None:
        if seq_lens is not None:
            raise ValueError("seq_lens requires block_table; logical input uses its full length")
        if k.shape[:2] != (batch, heads):
            raise ValueError("logical K/V must match Q batch and heads")
        max_len = k.shape[2]
        tables = None
    else:
        if k.shape[2] != heads:
            raise ValueError("paged K/V heads must equal Q heads (MHA)")
        for name, t, ndim in (("block_table", block_table, 2), ("seq_lens", seq_lens, 1)):
            if not isinstance(t, torch.Tensor) or t.ndim != ndim:
                raise ValueError(f"{name} must be a {ndim}-D tensor")
            if t.device != q.device or t.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor on Q device")
            if t.shape[0] != batch:
                raise ValueError(f"{name} must have B rows")
        lengths = seq_lens.tolist()
        tables = block_table.tolist()
        page_size = k.shape[1]
        for length, table in zip(lengths, tables):
            if length <= 0 or length > len(table) * page_size:
                raise ValueError("seq_lens must be positive and fit block_table capacity")
            if any(page < 0 or page >= k.shape[0] for page in table[:math.ceil(length / page_size)]):
                raise ValueError("block_table contains an invalid used page index")
        max_len = max(lengths)
        if block_table.numel() >= _MAX_SPAN:
            raise ValueError("block_table exceeds kernel int32 indexing capacity")

    # Page bases are int64 in the existing kernel; intra-page token/head
    # products are int32 and must remain bounded even for padded views.
    for cache in (k, v):
        if _span(cache.shape[1:], cache.stride()[1:]) >= _MAX_SPAN:
            raise ValueError("K/V request/page stride span exceeds kernel int32 indexing capacity")

    from boltops.mla import compute_num_kv_splits, decode_attention_fwd_normal

    # Each split stores its partial output plus one LSE slot.
    scratch_width = _HEAD_DIM + 1
    if num_kv_splits is None:
        with torch.cuda.device(q.device):
            num_kv_splits = compute_num_kv_splits(max_len)
    if isinstance(num_kv_splits, bool) or not isinstance(num_kv_splits, int) or num_kv_splits < 1:
        raise ValueError("num_kv_splits must be a positive integer")
    if max_len >= _MAX_SPAN or batch * heads * num_kv_splits * scratch_width >= _MAX_SPAN:
        raise ValueError("sequence/workspace exceeds kernel int32 indexing capacity")

    with torch.cuda.device(q.device):
        ks = _positive_scale(k_scale, "k_scale", q.device)
        vs = _positive_scale(v_scale, "v_scale", q.device)
        out = torch.empty_like(q, memory_format=torch.contiguous_format)

        def launch(query, keys, values, output, table, lengths, size):
            n = query.shape[0]
            lse = torch.empty((n, heads), device=q.device, dtype=torch.float32)
            scratch = torch.empty(
                (n, heads, num_kv_splits, scratch_width), device=q.device, dtype=torch.float32
            )
            decode_attention_fwd_normal(
                query, keys, values, output, lse, table.contiguous(), lengths.contiguous(),
                scratch, num_kv_splits, float(sm_scale), size, k_scale=ks, v_scale=vs,
            )

        if tables is None:
            # Each request is one strided page: no full-cache transpose copy.
            # Bucket the virtual page size to avoid specializing PAGE_SIZE for
            # every decode length. All valid tokens stay in page zero; the
            # sequence-length mask still bounds loads to the actual cache.
            logical_page_size = 1 << (max_len - 1).bit_length()
            table = torch.zeros((1, 1), device=q.device, dtype=torch.int32)
            lengths = torch.full((1,), max_len, device=q.device, dtype=torch.int32)
            for b in range(batch):
                launch(
                    q[b:b + 1, :, 0, :], k[b].transpose(0, 1).unsqueeze(0),
                    v[b].transpose(0, 1).unsqueeze(0), out[b:b + 1, :, 0, :],
                    table, lengths, logical_page_size,
                )
        else:
            launch(q[:, :, 0, :], k, v, out[:, :, 0, :], block_table, seq_lens, page_size)
    return out
