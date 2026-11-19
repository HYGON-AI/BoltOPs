"""Gluon WASP path for the measured gfx946 UA decode case."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
from triton.experimental.gluon.language.amd.hcu import ebarrier_sync, wait_lgkmcnt


@gluon.jit
def _empty_partition():
    pass


@gluon.jit
def _barrier(barrier_id: gl.constexpr, count: gl.constexpr):
    wait_lgkmcnt(0)
    ebarrier_sync(barrier_id, count)


@gluon.jit
def _ua_wasp_producer(
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    k_scale_ptr,
    v_scale_ptr,
    key_shared,
    value_shared,
    query_start_len_ptr,
    seq_lens_ptr,
    sequence,
    kv_head,
    BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    stride_k_cache_0: gl.constexpr,
    stride_k_cache_1: gl.constexpr,
    stride_k_cache_2: gl.constexpr,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.constexpr,
    stride_v_cache_1: gl.constexpr,
    stride_v_cache_2: gl.constexpr,
    stride_v_cache_3: gl.constexpr,
    block_table_stride: gl.constexpr,
    num_stages: gl.constexpr,
    BUFFER_KV: gl.constexpr,
    WARPS: gl.constexpr,
    TOTAL_WARPS: gl.constexpr,
):
    producer_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [WARPS, 1], [1, 0]
    )
    tile_rows = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(1, producer_layout))
    head_dims = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, producer_layout))
    sequence_length = gl.load(seq_lens_ptr + sequence)
    num_tiles = (sequence_length + TILE_SIZE - 1) // TILE_SIZE

    for tile in range(0, num_tiles):
        if tile > 0:
            _barrier(15, TOTAL_WARPS)
        token_offsets = tile * TILE_SIZE + tile_rows
        block_indices = gl.load(
            block_tables_ptr + sequence * block_table_stride
            + token_offsets // BLOCK_SIZE,
            token_offsets < sequence_length,
            other=0,
        )
        token_in_block = token_offsets % BLOCK_SIZE
        key_offsets = (
            block_indices[:, None] * stride_k_cache_0
            + token_in_block[:, None] * stride_k_cache_1
            + kv_head * stride_k_cache_2
            + head_dims[None, :] * stride_k_cache_3
        ).to(gl.int64)
        value_offsets = (
            block_indices[:, None] * stride_v_cache_0
            + token_in_block[:, None] * stride_v_cache_1
            + kv_head * stride_v_cache_2
            + head_dims[None, :] * stride_v_cache_3
        ).to(gl.int64)
        valid = token_offsets < sequence_length
        if BUFFER_KV:
            key = buffer_load(key_cache_ptr, key_offsets.to(gl.int32), valid[:, None], other=0.0)
        else:
            key = gl.load(key_cache_ptr + key_offsets, valid[:, None], other=0.0)
        key_shared.store(key.to(gl.bfloat16))
        if BUFFER_KV:
            value = buffer_load(value_cache_ptr, value_offsets.to(gl.int32), valid[:, None], other=0.0)
        else:
            value = gl.load(value_cache_ptr + value_offsets, valid[:, None], other=0.0)
        value_shared.store(value.to(gl.bfloat16))
        _barrier(14, TOTAL_WARPS)


@gluon.jit
def _ua_wasp_consumer(
    output_ptr,
    query_ptr,
    block_tables_ptr,
    key_shared,
    value_shared,
    query_start_len_ptr,
    seq_lens_ptr,
    sequence,
    kv_head,
    NUM_QUERY_HEADS: gl.constexpr,
    NUM_QUERIES_PER_KV: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    query_stride_0: gl.constexpr,
    query_stride_1: gl.constexpr,
    output_stride_0: gl.constexpr,
    output_stride_1: gl.constexpr,
    scale,
    WARPS: gl.constexpr,
    TOTAL_WARPS: gl.constexpr,
    k_scale_ptr,
    v_scale_ptr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        mmac_layout=gl.amd.MmacLayout.LIT,
        warps_per_cta=[1, WARPS],
        tiles_per_warp=[1, 2] if TILE_SIZE == 32 else [1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    softmax_layout: gl.constexpr = mma
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [2, 32], [1, WARPS], [1, 0]
    )
    query_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))
    query_dims = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, mma))
    sequence_start = gl.load(query_start_len_ptr + sequence)
    sequence_length = gl.load(seq_lens_ptr + sequence)
    query_heads = kv_head * NUM_QUERIES_PER_KV + query_rows
    query_offsets = (
        sequence_start * query_stride_0
        + query_heads[:, None] * query_stride_1
        + query_dims[None, :]
    ).to(gl.int64)
    query_mask = query_rows < NUM_QUERIES_PER_KV
    query = gl.load(query_ptr + query_offsets, query_mask[:, None], other=0).to(gl.bfloat16)
    query = gl.convert_layout(query, dot_a)
    acc = gl.full([BLOCK_M, HEAD_SIZE], 0.0, gl.float32, layout=mma)
    row_layout: gl.constexpr = gl.SliceLayout(1, softmax_layout)
    running_max = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=row_layout)
    running_sum = gl.full([BLOCK_M], 0.0, gl.float32, layout=row_layout)
    score_scale = scale * gl.load(k_scale_ptr)
    output_scale = gl.load(v_scale_ptr)
    num_tiles = (sequence_length + TILE_SIZE - 1) // TILE_SIZE
    for tile in range(0, num_tiles):
        _barrier(14, TOTAL_WARPS)
        key = key_shared.permute((1, 0)).load(dot_b)
        value = value_shared.load(dot_b)
        # Release the shared tile after both operands reach registers. The
        # producer can load/convert the next KV tile during current MMACs.
        if tile + 1 < num_tiles:
            _barrier(15, TOTAL_WARPS)
        scores = gl.amd.cdna3.mfma(query, key, gl.full([BLOCK_M, TILE_SIZE], 0.0, gl.float32, layout=mma))
        scores = scores * score_scale
        scores = gl.convert_layout(scores, softmax_layout)
        token_offsets = tile * TILE_SIZE + gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, softmax_layout))
        scores = gl.where(token_offsets[None, :] < sequence_length, scores, float("-inf"))
        tile_max = gl.max(scores, axis=1)
        tile_max = gl.maximum(running_max, tile_max)
        tile_max = gl.where(tile_max > float("-inf"), tile_max, 0.0)
        probabilities = gl.exp(scores - tile_max[:, None])
        tile_sum = gl.sum(probabilities, axis=1)
        alpha = gl.exp(running_max - tile_max)
        acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, mma))[:, None]
        acc = gl.amd.cdna3.mfma(gl.convert_layout(probabilities.to(gl.bfloat16), dot_a), value, acc)
        running_sum = running_sum * alpha + tile_sum
        running_max = tile_max

    acc = acc / gl.convert_layout(running_sum, gl.SliceLayout(1, mma))[:, None]
    acc = acc * output_scale
    store_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
    store_dims = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, store_layout))
    output_offsets = (
        sequence_start * output_stride_0
        + (kv_head * NUM_QUERIES_PER_KV + store_rows)[:, None] * output_stride_1
        + store_dims[None, :]
    ).to(gl.int64)
    store_mask = store_rows[:, None] < NUM_QUERIES_PER_KV
    gl.store(output_ptr + output_offsets, gl.convert_layout(acc.to(gl.bfloat16), store_layout), store_mask)


@gluon.jit
def unified_attention_wasp_gluon(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    query_start_len_ptr,
    seq_lens_ptr,
    k_scale,
    v_scale,
    scale,
    NUM_QUERY_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    NUM_QUERIES_PER_KV: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    query_stride_0: gl.constexpr,
    query_stride_1: gl.constexpr,
    output_stride_0: gl.constexpr,
    output_stride_1: gl.constexpr,
    block_table_stride: gl.constexpr,
    stride_k_cache_0: gl.constexpr,
    stride_k_cache_1: gl.constexpr,
    stride_k_cache_2: gl.constexpr,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.constexpr,
    stride_v_cache_1: gl.constexpr,
    stride_v_cache_2: gl.constexpr,
    stride_v_cache_3: gl.constexpr,
    BUFFER_KV: gl.constexpr,
    PARTITION_REGS: gl.constexpr,
    PARTITION_WARPS: gl.constexpr,
):
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    key_shared = gl.allocate_shared_memory(gl.bfloat16, [TILE_SIZE, HEAD_SIZE], shared_layout)
    value_shared = gl.allocate_shared_memory(gl.bfloat16, [TILE_SIZE, HEAD_SIZE], shared_layout)
    sequence = gl.program_id(0)
    kv_head = gl.program_id(1)
    producer_warps: gl.constexpr = PARTITION_WARPS[0]
    consumer_warps: gl.constexpr = PARTITION_WARPS[1]
    total_warps: gl.constexpr = producer_warps + consumer_warps
    gl.warp_specialize([
        (_empty_partition, ()),
        (_ua_wasp_producer, (
            key_cache_ptr, value_cache_ptr, block_tables_ptr, k_scale, v_scale,
            key_shared, value_shared, query_start_len_ptr, seq_lens_ptr, sequence, kv_head,
            BLOCK_SIZE, TILE_SIZE, HEAD_SIZE, stride_k_cache_0, stride_k_cache_1,
            stride_k_cache_2, stride_k_cache_3, stride_v_cache_0, stride_v_cache_1,
            stride_v_cache_2, stride_v_cache_3, block_table_stride, 2, BUFFER_KV,
            producer_warps, total_warps)),
        (_ua_wasp_consumer, (
            output_ptr, query_ptr, block_tables_ptr, key_shared, value_shared,
            query_start_len_ptr, seq_lens_ptr, sequence, kv_head, NUM_QUERY_HEADS,
            NUM_QUERIES_PER_KV, BLOCK_M, TILE_SIZE, HEAD_SIZE, query_stride_0, query_stride_1,
            output_stride_0, output_stride_1, scale,
            consumer_warps, total_warps, k_scale, v_scale)),
    ], PARTITION_WARPS, PARTITION_REGS)


def launch_unified_attention_wasp_gluon(q, k, v, out, cu_seqlens_q, seqused_k,
                                       block_table, k_scale, v_scale, scale):
    assert q.dtype == torch.bfloat16 and k.dtype == v.dtype
    assert q.shape[-1] == 128 and k.shape[-1] == v.shape[-1] == q.shape[-1]
    assert q.shape[0] == cu_seqlens_q.numel() - 1
    assert q.shape[0] == out.shape[0]
    tile_size = 64
    partition_regs = (40, 116)
    partition_warps = (4, 4)
    grid = (cu_seqlens_q.numel() - 1, k.shape[2])
    return unified_attention_wasp_gluon[grid](
        out, q, k, v, block_table, cu_seqlens_q, seqused_k, k_scale, v_scale, scale,
        NUM_QUERY_HEADS=q.shape[1], NUM_KV_HEADS=k.shape[2],
        NUM_QUERIES_PER_KV=q.shape[1] // k.shape[2], BLOCK_SIZE=k.shape[1],
        BLOCK_M=max(16, triton.next_power_of_2(q.shape[1] // k.shape[2])),
        TILE_SIZE=tile_size, HEAD_SIZE=q.shape[-1],
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        BUFFER_KV=max(k.numel() * k.element_size(), v.numel() * v.element_size()) < 2**31,
        PARTITION_REGS=partition_regs,
        PARTITION_WARPS=partition_warps,
        num_warps=4, num_stages=2, wasp_partition_warps=partition_warps,
        wasp_wdra=True, wasp_partition_regs=partition_regs,
        enable_v_mmac_cluster=1,
    )


__all__ = ["unified_attention_wasp_gluon", "launch_unified_attention_wasp_gluon"]
