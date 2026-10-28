# SPDX-License-Identifier: MIT
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py
"""Half-precision (fp16/bf16) MQA logits — dedicated Triton BN path.

Code defaults (overridden by shipped JSON when present):

* ``N <= 8192``: ``BM=16``, ``BN=64``, short-KV ``head_split_k`` by heads
* ``N > 8192``: ``BM=16``, ``BN=128``, ``split_k=1``

``head_split_k`` comes from ``_half_split_k``: short KV (``seq_len_kv <= 256``)
splits the *head* dimension (``16`` for ``num_heads >= 32``, else ``1``); long KV
keeps the lightop-like rule.  Shipped JSON ``by_shape`` may still override it.

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

from boltops.mqa_logits.triton.fp8_mqa_logits import _filter_hip_launch_kwargs


def _half_split_k(seq_len: int, seq_len_kv: int, num_heads: int) -> int:
    """Pick ``head_split_k``.

    Long KV: lightop-like heuristic (unchanged).

    Short KV (``seq_len_kv <= 256``): the parallelism must come from splitting
    the *head* dimension, and the useful amount depends on ``num_heads`` -- not
    on ``seq_len``.  Measured on BW1100/gfx938, the best ``split_k`` is ~1 for
    ``num_heads < 32`` and ~16 for ``num_heads >= 32``; the old
    ``seq_len <= 32`` gate left the whole ``seq_len > 32`` short-KV family at
    ``split_k`` 1-4, which was up to 2.5x off (e.g. 64x66 H=64).

    The ``256`` boundary matches the ``by_seq_kv`` bucket that carries the
    short-KV geometry, so both agree on where the rule hands over.
    """
    if seq_len_kv <= 256:
        split_k = 16 if num_heads >= 32 else 1
    else:
        split_k = 4
        if num_heads < 32:
            split_k = 2
    while split_k > 1 and num_heads % split_k != 0:
        split_k //= 2
    return max(1, split_k)


def _half_w_group(chunk: int, preferred: int) -> int:
    """Clamp ``w_group`` to a divisor of ``head_chunk``."""
    cand = max(1, int(preferred))
    while cand > 1 and chunk % cand != 0:
        cand //= 2
    return max(1, cand)


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
            "enable_v_mmac_cluster": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
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
            "enable_v_mmac_cluster": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
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
        "block_m",
        "block_n",
        "enable_v_mmac_cluster",
        "matrix_instr_nonkdim",
        "kpack",
        "q_depth",
        "assume_no_nans",
        "misched_regpressure",
    ):
        if key in cfg:
            out[key] = int(cfg[key])

    # Low-head geometry.  The shipped ``by_seq_kv`` buckets were tuned at
    # ``seq_len=127, num_heads=64`` and are reused verbatim for every
    # ``seq_len`` and ``num_heads``; ``default`` carries ``block_m=32``.  With
    # few heads there is not enough parallel work to keep those narrow tiles
    # busy, so head counts <= 8 want the wider ``block_m=64`` instead.
    #
    # ``block_m`` is the sole lever (``num_stages``/``clu``/``w_group`` held at
    # the bucket values).  It only pays off with enough parallel work on both
    # axes, so gate on both: a tile of 64 rows needs ~16 tiles along M
    # (``seq_len >= 1024``) before it helps, and below that it wins only once
    # the ``N`` axis carries the work (``seq_len_kv >= 2048``).  Measured on
    # gfx938 (BW1100) through this entry point: 1.06x-1.78x inside the gate,
    # while ``N <= 512`` or ``seq_len < 1024 & N < 2048`` regresses up to 30%,
    # hence the guards.  ``seq_len < 256`` is left untouched -- it is fully
    # covered by the dense ``by_shape`` grid at its measured optimum.
    if num_heads <= 8 and seq_len >= 256:
        need = 1024 if seq_len >= 1024 else 2048
        if seq_len_kv >= need:
            out["block_m"] = 64

    # Mid-head geometry (H=8).  ``block_n=128`` halves the N-grid, which pays
    # off only where the N axis already carries enough work.  Measured at H=8
    # through this entry point on gfx938 (BW1100), ``block_n`` 64 -> 128:
    #   seq_len_kv = 32768  0.979x-1.001x   (A=256/1024/4608)
    #   seq_len_kv = 65536  0.963x-0.981x
    #   seq_len_kv = 131072 0.958x-0.960x
    # i.e. neutral-to-better at >= 32768, but a wash at 16384 and a 4%-22%
    # regression at other head counts (H=4 +4~+16%, H=32 +1~+13%,
    # H=64 +5~+22%), so the ``num_heads == 8`` gate is load-bearing.
    #
    # It also rides on the ``block_m=64`` tile set just above: the narrow
    # ``block_m=32`` that ``seq_len < 256`` keeps (alongside ``w_group=1``)
    # makes ``block_n=128`` a ~5% regression (127x65536), so reuse the same
    # ``seq_len >= 256`` gate rather than keying on ``seq_len_kv`` alone.
    # ``waves_per_eu`` is handled by the rule just below, which subsumes this
    # geometry: the two knobs interact destructively (shipped JSON carries
    # ``waves_per_eu=4``, which is fine at ``block_n=64`` but costs 11%-12% at
    # ``block_n=128`` -- 4608x65536 H=8: 5304us vs 4750us), so ``block_n=128``
    # is only profitable together with ``waves_per_eu=0``.  Applying the pin
    # separately keeps each rule's evidence independent.
    if num_heads == 8 and seq_len >= 256 and seq_len_kv >= 32768:
        out["block_n"] = 128

    # Low-head ``waves_per_eu``.  The shipped JSON sets ``waves_per_eu=4`` on
    # its long-KV buckets, but that value is a slight pessimisation for every
    # low-head geometry measured here, and the effect is concentrated on the
    # ``seq_len_kv = 16384`` bucket (on ``seq_len_kv >= 32768`` it is within
    # noise for ``num_heads <= 8`` except where the rule above already pins 0):
    #   num_heads = 4 (A=256/1024/4608)   16384: +0.9%/+0.3%/+0.4%
    #   num_heads = 8 (A=256/1024/4608)   16384: +1.3%/+1.7%/+1.7%
    #   num_heads = 1, 2                  neutral (1.000x-1.019x)
    #   num_heads = 8, seq_len_kv >= 32768  neutral (already 0 via the rule above)
    # ``waves_per_eu`` 0/1/2 are interchangeable in these geometries, so 0 is
    # the safe representative.  The gate mirrors the ``block_m`` rule above so
    # that all three low-head rules share one ``seq_len >= 256`` boundary --
    # below it the dense ``by_shape`` grid owns the shape and none of this was
    # measured.
    if num_heads <= 8 and seq_len >= 256 and seq_len_kv >= 16384:
        out["waves_per_eu"] = 0

    if "head_split_k" in cfg:
        sk = int(cfg["head_split_k"])
        if sk >= 1 and num_heads % sk == 0:
            out["head_split_k"] = sk

    # ``w_group`` must divide the *effective* head chunk; shipped JSON carries
    # values tuned against a different ``head_split_k``.
    chunk = max(1, num_heads // out["head_split_k"])
    out["w_group"] = _half_w_group(chunk, int(cfg.get("w_group", out["w_group"])))
    # ``q_depth`` is a ring-buffer size over the head chunk and must not exceed
    # it (the kernel needs exactly that many distinct slots).
    qd = int(out.get("q_depth", 1))
    out["q_depth"] = max(1, min(qd, chunk))

    # Large-``seq_len`` short-KV geometry (``seq_len_kv <= 4096``).  The same
    # buckets also serve large-``seq_len`` prefill, where the ``seq_len=127``
    # tiles are far too small -- and there the *whole* tile wants to change,
    # not just ``block_m``.  Measured on gfx938 (BW1100) through this entry
    # point at ``seq_len_kv`` 512-4096, ``(block_m, block_n, num_stages,
    # w_group)`` 16/128/2/1 -> 64/64/1/4:
    #   num_heads = 64    1024x4096:  +3.5% -> -22%    4608x4096: +25% -> -13%
    #                     1024x2048:  -0.3% -> -21%    4608x2048: +22% -> -14%
    #   num_heads = 4, 8  seq_len 1024/4608: 2% ~ 43% closer to lightop
    # Every ``seq_len < 256`` shape regresses badly at the wide tile (e.g.
    # 127x2048 H=64 is ~34% *worse*), and those are exactly the shapes the
    # dense ``by_shape`` grid owns, hence the same ``seq_len >= 256`` gate as
    # the ``block_m`` rule above -- which also guarantees no ``by_shape`` entry
    # (all of which carry ``seq_len < 256``) is ever clobbered.  ``seq_len_kv >
    # 4096`` is left alone: those buckets are tuned separately and were not
    # re-measured here.
    #
    # This must run *after* the ``w_group`` normalization above: that clamp
    # derives from the JSON bucket (``cfg["w_group"]``), so it would otherwise
    # overwrite the pin with the bucket's ``1`` (measured: the pin was silently
    # dropped, costing the whole win).
    if seq_len >= 256 and seq_len_kv <= 4096:
        need = 1024 if seq_len >= 1024 else 2048
        if seq_len_kv >= need:
            out["block_m"] = 64
            out["block_n"] = 64
            out["num_stages"] = 1
            if chunk % 4 == 0:
                out["w_group"] = 4

    # --- backend scheduling knobs (证据见 docs/backend_knob_sweep.md) ---
    # assume_no_nans: 发 `no-nans-fp-math` LLVM 函数属性, 让后端丢掉 fmin/fmax
    #   展开时用于静默 signaling NaN 的 FCANONICALIZE。效果是每个 relu 少一条
    #   冗余 `v_max_f32 x, x` (4608x16384 H=4 上 128 -> 64 条 v_max_f32),
    #   实测 -3.9%~-4.5%, 输出与 baseline 逐位相同。lightop 本身就是按有限数学
    #   假设编译的, 所以这是向 lightop 的数值契约对齐, 而非额外放松; 代价是
    #   NaN 输入行为未定义, 故逐形状可选 (JSON/autotune 可覆盖)。
    out.setdefault("assume_no_nans", 1)
    # misched_regpressure=0 -> `-mllvm=-misched-regpressure=false`。只对计算
    #   受限的低头数形状有利 (gfx938 上 H<=8 实测 +1.7%~+2.0%), 其余中性偏负
    #   (H=16 -0.3%, 访存受限的超大 N 中性), 且与
    #   amdgpu-enable-max-ilp-scheduling-strategy 反协同, 所以保留 H<=8 闸门。
    out.setdefault("misched_regpressure", 0 if num_heads <= 8 else 1)
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
    Q_DEPTH: tl.constexpr = 2,
    QK_TF32: tl.constexpr = False,
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
    # QK_TF32: KV 以 f32 进 LDS, 走 f32 16x16x8 MMAC + ds_read_m32x8_b32(flat)。
    # 编译器 (AccelerateHCUMatmul) 要求 A/B 位宽相同且 16x16x8 的 K==kPerTile(8)。
    if QK_TF32:
        kv = kv.to(tl.float32)

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
            if QK_TF32:
                scores = tl.dot(q_cur.to(tl.float32), kv,
                                input_precision="tf32")
            else:
                scores = tl.dot(q_cur, kv, input_precision="ieee")
            scores = tl.maximum(scores, 0.0)
            acc += scores * w_cur[:, None]
            if hi + 1 < HEAD_CHUNK:
                q_cur = q_next
                w_cur = w_next
    else:
        n_groups: tl.constexpr = HEAD_CHUNK // W_GROUP
        offs_wg = tl.arange(0, W_GROUP)
        # ---- Q 环形: 跨 group 连续的 Q_DEPTH 槽软流水 ----
        # 旧实现每个 group 开头都重新 load 一次 Q(g*W_GROUP), 组边界存在
        # 流水断点. 这里改为: prologue 一次性填满 Q_DEPTH 个槽, 之后每拍
        # 整体左移并补最远槽 Q(h + Q_DEPTH), 使预取跨过 group 边界连续进行.
        # 注意: 必须恰好有 Q_DEPTH 个槽, 否则移位会跳过 head (曾导致静默错值).
        qs0 = tl.load(
            Q_ptr
            + offs_m[:, None] * stride_q_s
            + start_h * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=m_mask[:, None] & (start_h < NUM_HEADS),
            other=0.0,
        )
        if Q_DEPTH >= 2:
            qs1 = tl.load(
                Q_ptr
                + offs_m[:, None] * stride_q_s
                + (start_h + 1) * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=m_mask[:, None] & ((start_h + 1) < NUM_HEADS),
                other=0.0,
            )
        else:
            qs1 = qs0
        if Q_DEPTH >= 3:
            qs2 = tl.load(
                Q_ptr
                + offs_m[:, None] * stride_q_s
                + (start_h + 2) * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=m_mask[:, None] & ((start_h + 2) < NUM_HEADS),
                other=0.0,
            )
        else:
            qs2 = qs0
        if Q_DEPTH >= 4:
            qs3 = tl.load(
                Q_ptr
                + offs_m[:, None] * stride_q_s
                + (start_h + 3) * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=m_mask[:, None] & ((start_h + 3) < NUM_HEADS),
                other=0.0,
            )
        else:
            qs3 = qs0

        for g in tl.static_range(n_groups):
            offs_h_g = start_h + g * W_GROUP + offs_wg
            w_g = tl.load(
                weights_ptr
                + offs_m[:, None] * stride_w_s
                + offs_h_g[None, :] * stride_w_h,
                mask=m_mask[:, None] & (offs_h_g[None, :] < NUM_HEADS),
                other=0.0,
            ).to(tl.float32)
            # ---- 逐 head 权重提取: 静态解交织, 避免跨 lane 归约 ----
            # 原实现用 tl.sum(tl.where(arange==hi, w_g, 0), axis=1)。该归约依赖
            # 循环变量 hi, 无法提出循环, 于是每个 head 都要重做一次跨 lane(DPP)
            # 归约 + layout 转换; 4608x16384 H=4 实测占约 4% 并多出 6 道 s_barrier。
            # 这里对 (BLOCK_M, W_GROUP) 做一次 tl.split 解交织。注意 tl.split 取
            # 最后一维, 故 (M,W)->(M,W/2,2) 切出的是偶/奇列, 需按列号重排回 head
            # 顺序 (W=2/4/8 的重排都是位反转型, 见下面的取值顺序)。
            if W_GROUP == 2:
                _e, _o = tl.split(w_g)
                w_sel = [_e, _o]
            elif W_GROUP == 4:
                _e, _o = tl.split(tl.reshape(w_g, (BLOCK_M, 2, 2)))
                _e0, _e1 = tl.split(_e)
                _o0, _o1 = tl.split(_o)
                w_sel = [_e0, _o0, _e1, _o1]
            elif W_GROUP == 8:
                _e, _o = tl.split(tl.reshape(w_g, (BLOCK_M, 4, 2)))
                _e0, _e1 = tl.split(tl.reshape(_e, (BLOCK_M, 2, 2)))
                _o0, _o1 = tl.split(tl.reshape(_o, (BLOCK_M, 2, 2)))
                _e00, _e01 = tl.split(_e0)
                _e10, _e11 = tl.split(_e1)
                _o00, _o01 = tl.split(_o0)
                _o10, _o11 = tl.split(_o1)
                w_sel = [_e00, _o00, _e10, _o10, _e01, _o01, _e11, _o11]
            for hi in tl.static_range(W_GROUP):
                h = start_h + g * W_GROUP + hi
                q_cur = qs0
                # 取第 Q_DEPTH 槽 (最远), 用于补位
                q_far = qs0
                if Q_DEPTH == 2:
                    q_far = qs1
                elif Q_DEPTH == 3:
                    q_far = qs2
                elif Q_DEPTH >= 4:
                    q_far = qs3
                if (h + Q_DEPTH) < (start_h + HEAD_CHUNK):
                    q_far = tl.load(
                        Q_ptr
                        + offs_m[:, None] * stride_q_s
                        + (h + Q_DEPTH) * stride_q_h
                        + offs_d[None, :] * stride_q_d,
                        mask=m_mask[:, None] & ((h + Q_DEPTH) < NUM_HEADS),
                        other=0.0,
                    )
                if (W_GROUP == 2) or (W_GROUP == 4) or (W_GROUP == 8):
                    w = w_sel[hi]
                else:
                    # 非 2/4/8 的 W_GROUP 极少见, 退回逐 head 归约
                    w = tl.sum(
                        tl.where(tl.arange(0, W_GROUP)[None, :] == hi, w_g, 0.0),
                        axis=1,
                    )
                if QK_TF32:
                    scores = tl.dot(q_cur.to(tl.float32), kv,
                                    input_precision="tf32")
                else:
                    scores = tl.dot(q_cur, kv, input_precision="ieee")
                scores = tl.maximum(scores, 0.0)
                acc += scores * w[:, None]
                if (h + 1) < (start_h + HEAD_CHUNK):
                    if Q_DEPTH >= 2:
                        qs0 = qs1
                    if Q_DEPTH >= 3:
                        qs1 = qs2
                    if Q_DEPTH >= 4:
                        qs2 = qs3
                    if Q_DEPTH == 1:
                        qs0 = q_far
                    elif Q_DEPTH == 2:
                        qs1 = q_far
                    elif Q_DEPTH == 3:
                        qs2 = q_far
                    else:
                        qs3 = q_far

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

    # A single-split kernel writes the *entire* output: ``acc`` inside the window
    # and ``-inf`` outside it, covering every in-grid element (see both
    # ``tl.store`` sites in ``_mqa_logits_half_bn_kernel``).  Pre-filling is
    # therefore pure overhead there -- a full ``fill_(-inf)`` costs ~2.4 GB of
    # traffic, i.e. 2.3 ms at 4608x131072.  Only the multi-split path needs an
    # initialised buffer, because it accumulates with ``tl.atomic_add``.
    if D_out is not None:
        assert D_out.shape == (seq_len, seq_len_kv)
        assert D_out.dtype == torch.float32 and D_out.device == Q.device
        logits = D_out
        if num_splits > 1:
            logits.zero_()
    else:
        if num_splits > 1:
            logits = torch.zeros(
                (seq_len, seq_len_kv), device=Q.device, dtype=torch.float32
            )
        else:
            logits = torch.empty(
                (seq_len, seq_len_kv), device=Q.device, dtype=torch.float32
            )

    grid_n = (seq_len_kv + tile_bn - 1) // tile_bn
    grid_m = (seq_len + tile_bm - 1) // tile_bm
    grid = (grid_n, grid_m, num_splits)

    # Kernel constexprs (not HIPOptions).
    q_depth = int(disp.get("q_depth", 1))
    qk_tf32 = bool(int(disp.get("qk_tf32", 0)))
    # HIP launch knobs: drop unsupported options instead of KeyError on older
    # Triton (same filter as FP8 / half-row). See get_half_bn_dispatch notes.
    hip_kwargs = {
        # enable_v_mmac_cluster: sched barrier pass; do not add
        # empty_arrive_after_mmac (measured +38% trap).
        "enable_v_mmac_cluster": int(disp.get("enable_v_mmac_cluster", 1)),
        "matrix_instr_nonkdim": int(disp.get("matrix_instr_nonkdim", 16)),
        "kpack": int(
            disp.get(
                "kpack",
                2 if int(disp.get("matrix_instr_nonkdim", 16)) == 16 else 1,
            )
        ),
        "assume_no_nans": int(disp.get("assume_no_nans", 0)),
        "misched_regpressure": bool(int(disp.get("misched_regpressure", 1))),
    }
    if waves:
        hip_kwargs["waves_per_eu"] = waves
    hip = _filter_hip_launch_kwargs(hip_kwargs)

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
        Q_DEPTH=q_depth,
        QK_TF32=qk_tf32,
        num_warps=n_warps,
        num_stages=n_stages,
        **hip,
    )
    return logits


__all__ = [
    "mqa_logits_half",
    "get_half_bn_dispatch",
]
