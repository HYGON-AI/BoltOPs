# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimized long-query and short-query kernels for Extend Attention.

Host-side routing and configuration live in ``extend_attention`` and
``config`` so historical version suffixes do not leak into the BoltOPs API.
"""

import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.heuristics({"CONST_LEN_EXTEND": lambda META: False})
@triton.jit
def _fwd_kernel_long_query(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    head_num: tl.constexpr,
    USE_MLS: tl.constexpr,
    batch_size,
    max_len_extend,
    CONST_LEN_EXTEND: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    tl.assume(Q_Extend.to(tl.int64) >= 0)
    tl.assume(K_Extend.to(tl.int64) >= 0)
    tl.assume(V_Extend.to(tl.int64) >= 0)

    tl.assume(kv_group_num >= 0)
    tl.assume(stride_qbs >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_kbs >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_vbs >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_obs >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_buf_kbs >= 0)
    tl.assume(stride_buf_kh >= 0)
    tl.assume(stride_buf_vbs >= 0)
    tl.assume(stride_buf_vh >= 0)
    tl.assume(head_num >= 0)
    tl.assume(batch_size >= 0)
    tl.assume(max_len_extend >= 0)

    kv_head_num = head_num // kv_group_num
    cur_kv_head = cur_head // kv_group_num

    if CONST_LEN_EXTEND:
        cur_seq_extend_start_idx = cur_seq * max_len_extend
        cur_seq_len_extend = max_len_extend
        cur_seq_kv_start_idx = 0
        cur_seq_len_prefix = 0
    else:
        cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
        cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
        cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
        cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    ALL_MASK_M = tl.min(mask_m.to(tl.int32), axis=0) == 1
    ALL_MASK_D = tl.min(mask_d.to(tl.int32), axis=0) == 1
    ALL_MASK_DV = tl.min(mask_dv.to(tl.int32), axis=0) == 1

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    if USE_MLS:
        q = tl.matrix_load(
            Q_Extend + cur_head * stride_qh,
            shape=(head_num, Lq),
            strides=(stride_qbs, 1),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            offsets=((cur_seq_extend_start_idx + cur_block_m * BLOCK_M).to(tl.int32), 0),
        )
        if not (ALL_MASK_M & ALL_MASK_D):
            q = tl.where((mask_m[:, None]) & (mask_d[None, :]), q, 0.0)
    else:
        offs_q = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
        )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        if USE_MLS:
            qpe = tl.matrix_load(Q_Extend + cur_head * stride_qh,
                                shape=(head_num, Lq),
                                strides=(stride_qbs, 1),
                                block_shape=(BLOCK_M, BLOCK_DPE),
                                offsets=((cur_seq_extend_start_idx + cur_block_m * BLOCK_M).to(tl.int32),
                                         BLOCK_DMODEL),
                                )
            if not ALL_MASK_M:
                qpe = tl.where(mask_m[:, None], qpe, 0.0)
        else:
            offs_qpe = (
                (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qbs
                + cur_head * stride_qh
                + offs_dpe[None, :]
            )
            qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        ALL_MASK_N = tl.min(mask_n.to(tl.int32), axis=0) == 1

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            if USE_MLS:
                custom_mask = tl.matrix_load(
                    mask_ptr + cur_seq_mask_start_idx,
                    shape=(cur_seq_len_extend, cur_seq_len + window_kv_offset),
                    strides=(cur_seq_len + window_kv_offset, 1),
                    block_shape=(BLOCK_M, BLOCK_N),
                    offsets=((cur_block_m * BLOCK_M).to(tl.int32),
                             (window_kv_offset + start_n).to(tl.int32)),
                )
                if not (ALL_MASK_M & ALL_MASK_N):
                    custom_mask = tl.where((mask_m[:, None]) & (mask_n[None, :]), custom_mask, 0)
            else:
                custom_mask = tl.load(
                    mask_ptr
                    + cur_seq_mask_start_idx
                    + (cur_block_m * BLOCK_M + offs_m[:, None])
                    * (cur_seq_len + window_kv_offset)
                    + window_kv_offset
                    + start_n
                    + offs_n[None, :],
                    mask=(mask_m[:, None] & mask_n[None, :]),
                    other=0,
                )
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)
            qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale

            e_max = n_e_max

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end
        ALL_MASK_N = tl.min(mask_n.to(tl.int32), axis=0) == 1

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            if USE_MLS:
                custom_mask = tl.matrix_load(
                    mask_ptr + cur_seq_mask_start_idx,
                    shape=(cur_block_m_end, cur_seq_len + window_kv_offset),
                    strides=(cur_seq_len + window_kv_offset, 1),
                    block_shape=(BLOCK_M, BLOCK_N),
                    offsets=((cur_block_m * BLOCK_M).to(tl.int32),
                             (window_kv_offset + cur_seq_len_prefix + start_n).to(tl.int32)),
                )
                if not (ALL_MASK_M & ALL_MASK_N):
                    custom_mask = tl.where((mask_m[:, None]) & (mask_n[None, :]), custom_mask, 0)
            else:
                custom_mask = tl.load(
                    mask_ptr
                    + cur_seq_mask_start_idx
                    + (cur_block_m * BLOCK_M + offs_m[:, None])
                    * (cur_seq_len + window_kv_offset)
                    + window_kv_offset
                    + cur_seq_len_prefix
                    + start_n
                    + offs_n[None, :],
                    mask=(mask_m[:, None] & mask_n[None, :]),
                    other=0,
                )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            if USE_MLS:
                k = tl.matrix_load(
                    K_Extend + cur_kv_head * stride_kh,
                    shape=(kv_head_num, Lq),
                    strides=(1, stride_kbs),
                    block_shape=(BLOCK_DMODEL, BLOCK_N),
                    offsets=(0, (cur_seq_extend_start_idx + start_n).to(tl.int32)),
                )
                if not (ALL_MASK_N & ALL_MASK_D):
                    k = tl.where((mask_d[:, None]) & (mask_n[None, :]), k, 0.0)
            else:
                offs_k = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_d[:, None]
                )
                k = tl.load(
                    K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
                )

            qk = tl.dot(q.to(k.dtype), k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                if USE_MLS:
                    kpe = tl.matrix_load(
                        K_Extend + cur_kv_head * stride_kh,
                        shape=(kv_head_num, Lq),
                        strides=(1, stride_kbs),
                        block_shape=(BLOCK_DPE, BLOCK_N),
                        offsets=(BLOCK_DMODEL, (cur_seq_extend_start_idx + start_n).to(tl.int32)),
                    )
                    if not ALL_MASK_N:
                        kpe = tl.where(mask_n[None, :], kpe, 0.0)
                else:
                    offs_kpe = (
                        (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                        + cur_kv_head * stride_kh
                        + offs_dpe[:, None]
                    )
                    kpe = tl.load(
                        K_Extend + offs_kpe,
                        mask=mask_n[None, :],
                        other=0.0,
                    )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if USE_MLS:
                v = tl.matrix_load(
                    V_Extend + cur_kv_head * stride_vh,
                    shape=(kv_head_num, Lv),
                    strides=(stride_vbs, 1),
                    block_shape=(BLOCK_N, BLOCK_DV),
                    offsets=((cur_seq_extend_start_idx + start_n).to(tl.int32), 0),
                )
                if not (ALL_MASK_N & ALL_MASK_DV):
                    v = tl.where((mask_n[:, None]) & (mask_dv[None, :]), v, 0.0)
            else:
                offs_v = (
                    (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                    + cur_kv_head * stride_vh
                    + offs_dv[None, :]
                )
                v = tl.load(
                    V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
                )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    deno_safe = tl.where(deno == 0.0, 1.0, deno)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno_safe[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno_safe[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


@triton.heuristics({
    # 'CONST_LEN_EXTEND': lambda META: True
    'CONST_LEN_EXTEND': lambda META: False
})
@triton.jit
def _fwd_kernel_short_query(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    kv_group_num: tl.constexpr,
    num_query_heads: tl.constexpr,
    USE_MLS: tl.constexpr,
    batch_size,
    max_len_extend,
    CONST_LEN_EXTEND: tl.constexpr,
):
    """
    v2 decode: grid (batch, num_kv_heads, cdiv(max_len_extend, Q_SEQ)) with
    Q_SEQ = BLOCK_M // kv_group_num (same as unified ``BLOCK_Q``; floor, not ceil). If BLOCK_M is not
    a multiple of G, adjacent ``cur_block_m`` may overlap in query_pos like unified, but `mask_m`
    and sequence bounds keep correctness. BLOCK_M is a power of 2 (host). Require BLOCK_M // G >= 1 to launch.
    """
    # Per unified: BLOCK_Q = BLOCK_M // G; stride in query token index for each +1 of cur_block_m.
    Q_SEQ: tl.constexpr = BLOCK_M // kv_group_num
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    tl.assume(Q_Extend.to(tl.int64) >= 0)
    tl.assume(K_Extend.to(tl.int64) >= 0)
    tl.assume(V_Extend.to(tl.int64) >= 0)

    tl.assume(stride_qbs >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_kbs >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_vbs >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_obs >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_buf_kbs >= 0)
    tl.assume(stride_buf_kh >= 0)
    tl.assume(stride_buf_vbs >= 0)
    tl.assume(stride_buf_vh >= 0)
    tl.assume(batch_size >= 0)
    tl.assume(max_len_extend >= 0)

    kv_head_num = num_query_heads // kv_group_num

    if CONST_LEN_EXTEND:
        cur_seq_extend_start_idx = cur_seq * max_len_extend
        cur_seq_len_extend = max_len_extend
    else:
        cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
        cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if cur_block_m * Q_SEQ >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    # unified_attention-style: per offs_m, row = offs_m // G, h = offs_m % G (G = kv_group_num)
    query_pos = cur_block_m * Q_SEQ + (offs_m // kv_group_num)
    q_head_in_group = offs_m % kv_group_num
    query_offset_0 = cur_seq_extend_start_idx + query_pos
    query_offset_1 = cur_kv_head * kv_group_num + q_head_in_group
    mask_m = (query_pos < cur_seq_len_extend) & (query_offset_1 < num_query_heads)

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    ALL_MASK_M = tl.min(mask_m.to(tl.int32), axis=0) == 1
    ALL_MASK_D = tl.min(mask_d.to(tl.int32), axis=0) == 1
    ALL_MASK_DV = tl.min(mask_dv.to(tl.int32), axis=0) == 1

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + query_pos
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    offs_q = (
        query_offset_0[:, None] * stride_qbs
        + query_offset_1[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            query_offset_0[:, None] * stride_qbs
            + query_offset_1[:, None] * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        ALL_MASK_N = tl.min(mask_n.to(tl.int32), axis=0) == 1

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            # if USE_MLS:
            #     group_id = offs_m // kv_group_num
            #     custom_mask_group = tl.matrix_load(
            #         mask_ptr + cur_seq_mask_start_idx,
            #         shape=(cur_seq_len_prefix, cur_seq_len + window_kv_offset),
            #         strides=(cur_seq_len + window_kv_offset, 1),
            #         block_shape=(BLOCK_M // kv_group_num, BLOCK_N),
            #         offsets=((cur_block_m * Q_SEQ).to(tl.int32),
            #                  (window_kv_offset + start_n).to(tl.int32)),
            #     )
            #     custom_mask = custom_mask_group[group_id[:, None], offs_n[None, :]]
            #     if not (ALL_MASK_M & ALL_MASK_N):
            #         custom_mask = tl.where((mask_m[:, None] & mask_n[None, :]), custom_mask, 0)
            # else:
            #     custom_mask = tl.load(
            #         mask_ptr
            #         + cur_seq_mask_start_idx
            #         + (query_pos[:, None]) * (cur_seq_len + window_kv_offset)
            #         + window_kv_offset
            #         + start_n
            #         + offs_n[None, :],
            #         mask=(mask_m[:, None] & mask_n[None, :]),
            #         other=0,
            #     )
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (query_pos[:, None]) * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            window_mask = (
                cur_seq_len_prefix + query_pos[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)
            qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale

            e_max = n_e_max

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * Q_SEQ)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end
        ALL_MASK_N = tl.min(mask_n.to(tl.int32), axis=0) == 1

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            # if USE_MLS:
            #     group_id = offs_m // kv_group_num
            #     custom_mask_group = tl.matrix_load(
            #         mask_ptr + cur_seq_mask_start_idx,
            #         shape=(cur_block_m_end, cur_seq_len + window_kv_offset),
            #         strides=(cur_seq_len + window_kv_offset, 1),
            #         block_shape=(BLOCK_M // kv_group_num, BLOCK_N),
            #         offsets=((cur_block_m * Q_SEQ).to(tl.int32),
            #                  (window_kv_offset + cur_seq_len_prefix + start_n).to(tl.int32)),
            #     )
            #     custom_mask = custom_mask_group[group_id[:, None], offs_n[None, :]]
            #     if not (ALL_MASK_M & ALL_MASK_N):
            #         custom_mask = tl.where((mask_m[:, None] & mask_n[None, :]), custom_mask, 0)
            # else:
            #     custom_mask = tl.load(
            #         mask_ptr
            #         + cur_seq_mask_start_idx
            #         + (query_pos[:, None]) * (cur_seq_len + window_kv_offset)
            #         + window_kv_offset
            #         + cur_seq_len_prefix
            #         + start_n
            #         + offs_n[None, :],
            #         mask=(mask_m[:, None] & mask_n[None, :]),
            #         other=0,
            #     )
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (query_pos[:, None]) * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = query_pos[:, None] >= (start_n + offs_n[None, :])
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            window_mask = query_pos[:, None] <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            if USE_MLS:
                k = tl.matrix_load(
                    K_Extend + cur_kv_head * stride_kh,
                    shape=(kv_head_num, Lq),
                    strides=(1, stride_kbs),
                    block_shape=(BLOCK_DMODEL, BLOCK_N),
                    offsets=(0, (cur_seq_extend_start_idx + start_n).to(tl.int32)),
                )
                if not (ALL_MASK_D & ALL_MASK_N):
                    k = tl.where((mask_d[:, None] & (mask_n[None, :])), k, 0.0)
            else:
                offs_k = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_d[:, None]
                )
                k = tl.load(
                    K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
                )

            qk = tl.dot(q.to(k.dtype), k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                if USE_MLS:
                    kpe = tl.matrix_load(
                        K_Extend + cur_kv_head * stride_kh,
                        shape=(kv_head_num, Lq),
                        strides=(1, stride_kbs),
                        block_shape=(BLOCK_DPE, BLOCK_N),
                        offsets=(BLOCK_DMODEL, (cur_seq_extend_start_idx + start_n).to(tl.int32)),
                    )
                    if not ALL_MASK_N:
                        kpe = tl.where(mask_n[None, :], kpe, 0.0)
                else:
                    offs_kpe = (
                        (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                        + cur_kv_head * stride_kh
                        + offs_dpe[:, None]
                    )
                    kpe = tl.load(
                        K_Extend + offs_kpe,
                        mask=mask_n[None, :],
                        other=0.0,
                    )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if USE_MLS:
                v = tl.matrix_load(
                    V_Extend + cur_kv_head * stride_vh,
                    shape=(kv_head_num, Lv),
                    strides=(stride_vbs, 1),
                    block_shape=(BLOCK_N, BLOCK_DV),
                    offsets=((cur_seq_extend_start_idx + start_n).to(tl.int32), 0),
                )
                if not (ALL_MASK_N & ALL_MASK_DV):
                    v = tl.where((mask_n[:, None] & mask_dv[None, :]), v, 0.0)
            else:
                offs_v = (
                    (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                    + cur_kv_head * stride_vh
                    + offs_dv[None, :]
                )
                v = tl.load(
                    V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
                )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(
            sink_ptr + cur_kv_head * kv_group_num + q_head_in_group,
            mask=mask_m,
            other=0.0,
        )
        deno += tl.exp(cur_sink - e_max)

    deno_safe = tl.where(deno == 0.0, 1.0, deno)

    offs_o = (
        query_offset_0[:, None] * stride_obs
        + query_offset_1[:, None] * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno_safe[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno_safe[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )
