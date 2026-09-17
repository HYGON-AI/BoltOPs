"""Half-precision (fp16/bf16) MQA logits — dedicated Triton BN path.

Code defaults (overridden by shipped JSON when present):

* ``N <= 8192``: ``BM=16``, ``BN=64``, lightop-like ``head_split_k``
* ``N > 8192``: ``BM=16``, ``BN=128``, ``split_k=1``

Shipped ``mqa_logits_half,arch=*.json`` prefers exact ``by_shape`` (full-size
tune), then ``by_seq_kv`` / ``default``.

No ``kv_scales`` (unlike the FP8 kernels). Not shared with the FP8 BN/Gluon
paths so those stay tuned independently.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def _half_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    """Match lightop half small-path ``split_k`` heuristic."""
    split_k = 4
    if seq_len <= 32 and seq_len_kv <= 128:
        split_k = 16
    if num_heads < 32:
        split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


def get_half_bn_dispatch(seq_len: int, seq_len_kv: int, num_heads: int) -> dict:
    """Half-precision tile / split defaults + shipped JSON overrides.

    Code defaults follow lightop geometry; JSON may override ``block_m``,
    ``block_n``, ``head_split_k``, and launch knobs when present.
    """
    if seq_len_kv > 8192:
        out = {
            "block_m": 16,
            "block_n": 128,
            "head_split_k": 1,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 0,
            "w_group": 8,
            "init_neginf": True,
        }
    else:
        out = {
            "block_m": 16,
            "block_n": 64,
            "head_split_k": _half_split_k(seq_len, seq_len_kv, num_heads),
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 0,
            "w_group": 8,
            "init_neginf": False,
        }

    from boltops.mqa_logits.triton.mqa_config_utils import (
        get_mqa_logits_half_launch_config,
    )

    cfg = get_mqa_logits_half_launch_config(seq_len, seq_len_kv, num_heads)
    for key in (
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "w_group",
        "block_m",
        "block_n",
    ):
        if key in cfg:
            out[key] = int(cfg[key])

    if "head_split_k" in cfg:
        sk = int(cfg["head_split_k"])
        if sk >= 1 and num_heads % sk == 0:
            out["head_split_k"] = sk
    return out


@triton.jit
def _mqa_logits_half_bn_kernel(
    Q_ptr,
    KV_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.constexpr,
    stride_logits_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ASSUME_KS_ZERO: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_CHUNK: tl.constexpr,
    W_GROUP: tl.constexpr,
):
    """Half BN tile: KV once, Q+W grouped prefetch, no kv_scales."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    m_start = pid_m * BLOCK_M
    start_h = split_id * HEAD_CHUNK
    if start_h >= NUM_HEADS:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    m_mask = offs_m < seq_len
    n_mask = offs_n < seq_len_kv

    if ASSUME_KS_ZERO:
        cu_start = tl.zeros([BLOCK_M], dtype=tl.int32)
    else:
        cu_start = tl.load(cu_start_ptr + offs_m, mask=m_mask, other=0)
        cu_start = tl.maximum(cu_start, 0)
    cu_end = tl.load(cu_end_ptr + offs_m, mask=m_mask, other=0)
    cu_end = tl.minimum(cu_end, seq_len_kv)

    tile_past = n_start >= tl.max(cu_end, axis=0)
    tile_before = False
    if not ASSUME_KS_ZERO:
        tile_before = (n_start + BLOCK_N) <= tl.min(cu_start, axis=0)
    if tile_past or tile_before:
        if NUM_SPLITS == 1 or split_id == 0:
            logits_ptrs = (
                logits_ptr
                + offs_m[:, None] * stride_logits_s
                + offs_n[None, :] * stride_logits_k
            )
            tl.store(
                logits_ptrs,
                tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=m_mask[:, None] & n_mask[None, :],
            )
        return

    kv_ptrs = KV_ptr + offs_d[:, None] * stride_kv_d + offs_n[None, :] * stride_kv_s
    kv = tl.load(kv_ptrs, mask=n_mask[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    tl.static_assert(HEAD_CHUNK % W_GROUP == 0)
    tl.static_assert(W_GROUP >= 1)
    # W_GROUP==1: lightop-style per-head Q/W ping-pong (prefetch next head).
    # W_GROUP>=2: batched weight load + intra-group Q software pipeline.
    if W_GROUP == 1:
        h0 = start_h
        q_cur = tl.load(
            Q_ptr
            + offs_m[:, None] * stride_q_s
            + h0 * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=m_mask[:, None] & (h0 < NUM_HEADS),
            other=0.0,
        )
        w_cur = tl.load(
            weights_ptr + offs_m * stride_w_s + h0 * stride_w_h,
            mask=m_mask & (h0 < NUM_HEADS),
            other=0.0,
        ).to(tl.float32)
        for hi in tl.static_range(HEAD_CHUNK):
            h = start_h + hi
            if hi + 1 < HEAD_CHUNK:
                q_next = tl.load(
                    Q_ptr
                    + offs_m[:, None] * stride_q_s
                    + (h + 1) * stride_q_h
                    + offs_d[None, :] * stride_q_d,
                    mask=m_mask[:, None] & ((h + 1) < NUM_HEADS),
                    other=0.0,
                )
                w_next = tl.load(
                    weights_ptr + offs_m * stride_w_s + (h + 1) * stride_w_h,
                    mask=m_mask & ((h + 1) < NUM_HEADS),
                    other=0.0,
                ).to(tl.float32)
            scores = tl.dot(q_cur, kv, input_precision="ieee")
            scores = tl.maximum(scores, 0.0)
            acc += scores * w_cur[:, None]
            if hi + 1 < HEAD_CHUNK:
                q_cur = q_next
                w_cur = w_next
    else:
        n_groups: tl.constexpr = HEAD_CHUNK // W_GROUP
        offs_wg = tl.arange(0, W_GROUP)
        for g in tl.static_range(n_groups):
            offs_h_g = start_h + g * W_GROUP + offs_wg
            w_g = tl.load(
                weights_ptr
                + offs_m[:, None] * stride_w_s
                + offs_h_g[None, :] * stride_w_h,
                mask=m_mask[:, None] & (offs_h_g[None, :] < NUM_HEADS),
                other=0.0,
            ).to(tl.float32)
            h0 = start_h + g * W_GROUP
            q_cur = tl.load(
                Q_ptr
                + offs_m[:, None] * stride_q_s
                + h0 * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=m_mask[:, None] & (h0 < NUM_HEADS),
                other=0.0,
            )
            for hi in tl.static_range(W_GROUP):
                h = start_h + g * W_GROUP + hi
                if hi + 1 < W_GROUP:
                    q_next = tl.load(
                        Q_ptr
                        + offs_m[:, None] * stride_q_s
                        + (h + 1) * stride_q_h
                        + offs_d[None, :] * stride_q_d,
                        mask=m_mask[:, None] & ((h + 1) < NUM_HEADS),
                        other=0.0,
                    )
                w = tl.sum(
                    tl.where(tl.arange(0, W_GROUP)[None, :] == hi, w_g, 0.0),
                    axis=1,
                )
                scores = tl.dot(q_cur, kv, input_precision="ieee")
                scores = tl.maximum(scores, 0.0)
                acc += scores * w[:, None]
                if hi + 1 < W_GROUP:
                    q_cur = q_next

    if ASSUME_KS_ZERO:
        in_window = offs_n[None, :] < cu_end[:, None]
    else:
        in_window = (offs_n[None, :] >= cu_start[:, None]) & (
            offs_n[None, :] < cu_end[:, None]
        )
    store_ok = m_mask[:, None] & n_mask[None, :]
    logits_ptrs = (
        logits_ptr
        + offs_m[:, None] * stride_logits_s
        + offs_n[None, :] * stride_logits_k
    )
    if NUM_SPLITS == 1:
        out = tl.where(in_window & store_ok, acc, float("-inf"))
        tl.store(logits_ptrs, out, mask=store_ok)
    else:
        tl.atomic_add(logits_ptrs, acc, mask=store_ok & in_window)
        if split_id == 0:
            tl.store(
                logits_ptrs,
                tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=store_ok & (~in_window),
            )


def mqa_logits_half(
    Q: torch.Tensor,
    KV: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    D_out: Optional[torch.Tensor] = None,
    assume_ks_zero: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    head_split_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    waves_per_eu: Optional[int] = None,
    w_group: Optional[int] = None,
) -> torch.Tensor:
    """Prefill MQA logits for float16 / bfloat16 Q·KV (no scales)."""
    if Q.dtype != KV.dtype or Q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"mqa_logits_half expects fp16/bf16 Q/KV of same dtype, got "
            f"{Q.dtype}/{KV.dtype}"
        )
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = int(KV.shape[0])
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    assert head_size == 128, "half path currently expects head_dim=128"

    disp = get_half_bn_dispatch(seq_len, seq_len_kv, num_heads)
    tile_bm = int(block_m if block_m is not None else disp["block_m"])
    tile_bn = int(block_n if block_n is not None else disp["block_n"])
    num_splits = int(
        head_split_k if head_split_k is not None else disp["head_split_k"]
    )
    assert num_splits >= 1 and num_heads % num_splits == 0
    head_chunk = num_heads // num_splits

    n_warps = int(num_warps if num_warps is not None else disp["num_warps"])
    n_stages = int(num_stages if num_stages is not None else disp["num_stages"])
    waves = int(waves_per_eu if waves_per_eu is not None else disp["waves_per_eu"])

    w_grp = int(disp.get("w_group", 8) if w_group is None else w_group)
    while w_grp > 1 and head_chunk % w_grp != 0:
        w_grp //= 2
    if w_grp < 1 or head_chunk % w_grp != 0:
        w_grp = 1
    # W_GROUP=1 is lightop-style per-head; >=2 may be odd only if it divides chunk.

    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv)
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if num_splits > 1:
            logits.zero_()
        elif disp["init_neginf"]:
            logits.fill_(float("-inf"))
    else:
        if num_splits > 1:
            logits = torch.zeros(
                (seq_len, seq_len_kv), device=Q.device, dtype=torch.float32
            )
        elif disp["init_neginf"]:
            logits = torch.full(
                (seq_len, seq_len_kv),
                float("-inf"),
                device=Q.device,
                dtype=torch.float32,
            )
        else:
            logits = torch.empty(
                (seq_len, seq_len_kv), device=Q.device, dtype=torch.float32
            )

    grid_n = (seq_len_kv + tile_bn - 1) // tile_bn
    grid_m = (seq_len + tile_bm - 1) // tile_bm
    grid = (grid_n, grid_m, num_splits)

    kwargs = {
        "num_warps": n_warps,
        "num_stages": n_stages,
    }
    if waves:
        kwargs["waves_per_eu"] = waves

    _mqa_logits_half_bn_kernel[grid](
        Q,
        KV,
        weights,
        cu_starts,
        cu_ends,
        logits,
        seq_len,
        seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=Q.stride(0),
        stride_q_h=Q.stride(1),
        stride_q_d=Q.stride(2),
        stride_kv_s=KV.stride(0),
        stride_kv_d=KV.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_M=tile_bm,
        BLOCK_N=tile_bn,
        ASSUME_KS_ZERO=assume_ks_zero,
        NUM_SPLITS=num_splits,
        HEAD_CHUNK=head_chunk,
        W_GROUP=w_grp,
        **kwargs,
    )
    return logits


__all__ = [
    "mqa_logits_half",
    "get_half_bn_dispatch",
]
