# SPDX-License-Identifier: MIT
"""Independent weighted BF16 / packed E4M3FN paged decode on gfx938.

QSA-compatible FullRow geometry and ordinary/persistent partitioning.
Signed head weights follow ReLU; per-Key FP8 scales follow head reduction.
Runtime attention scaling follows head reduction; QSA kernels are unchanged.
"""

import math

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-ffast-math"],
    }
)
def _weighted_decode_kernel(
    heads: int, head_dim: int, page_size: int, group_size: int,
    groups_per_cta: int, persistent_splits: int, threads: int,
    num_stages: int, use_async: bool, packed_fp8: bool, scale_async: bool,
):
    assert heads in (4, 8, 16, 32, 64)
    assert head_dim == 128
    assert page_size in (8, 16, 32, 64)
    assert group_size in (64, 128) and group_size % page_size == 0
    assert groups_per_cta > 0 and persistent_splits >= 0
    assert threads == 256
    assert not scale_async or packed_fp8
    # Small-page async gathers and high-split software pipelines are not
    # reliably lowered by the current HCU compiler. Keep their partitioning
    # but use synchronous gathers, with explicit CTA barriers for high splits.
    explicit_sync = persistent_splits >= 16
    async_keys = (use_async and not explicit_sync
                  and not (page_size == 8 or (group_size == 64 and page_size < 64)))
    scale_async = scale_async and async_keys
    pipeline_stages = num_stages if async_keys else 0
    compile_options = {}
    if explicit_sync:
        # Automatic ThreadSync misclassifies some S16/S32 uniform partitions
        # as partial-wave regions. All shared dependencies below instead have
        # explicit CTA barriers, under CTA-uniform group/tail conditions.
        compile_options[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    if packed_fp8:
        compile_options.update({
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_Z3_RLIMIT: 50000,
        })
    if not async_keys:
        compile_options[tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY] = False
    sub_pages = group_size // page_size
    block_heads = max(16, heads)
    dtype = T.float8_e4m3fn if packed_fp8 else T.bfloat16
    batch = T.dynamic("batch")
    pages = T.dynamic("pages")
    max_pages = T.dynamic("max_pages")
    max_model_len = T.dynamic("max_model_len")
    use_offen = heads > 16
    packed_page_size = page_size * (head_dim + 4)
    cache_shape = ([pages, 1, packed_page_size] if use_offen
                   else [pages, packed_page_size]) if packed_fp8 else [pages, page_size, 1, head_dim]

    @T.macro
    def load_keys(KCache, PageTable, k_shared, bx, group, context_len, tail):
        for token, dim in T.Parallel(group_size, head_dim, prefer_async=async_keys and not tail):
            if not tail or group * group_size + token < context_len:
                page_id = PageTable[bx, group * sub_pages + token // page_size]
                if packed_fp8:
                    if use_offen:
                        k_shared[token, dim] = KCache[page_id, 0, (token % page_size) * head_dim + dim]
                    else:
                        k_shared[token, dim] = KCache[page_id, (token % page_size) * head_dim + dim]
                else:
                    k_shared[token, dim] = KCache[page_id, token % page_size, 0, dim]
            else:
                k_shared[token, dim] = T.cast(0, dtype)

    @T.macro
    def load_scales(ScaleCache, PageTable, scale_buf, bx, group, context_len, tail):
        for token in T.Parallel(group_size, prefer_async=scale_async and not tail):
            if not tail or group * group_size + token < context_len:
                page_id = PageTable[bx, group * sub_pages + token // page_size]
                scale_buf[token] = ScaleCache[page_id, page_size * head_dim // 4 + token % page_size]
            else:
                scale_buf[token] = T.float32(0)

    @T.macro
    def compute(k_shared, k_local, q_local, weights, scores, reduced,
                scale_buf, scale_local, Logits, Scale, bx, group_start, context_len, tx, tail):
        if explicit_sync:
            T.sync_threads()
        T.copy(k_shared, k_local)
        if scale_async:
            T.copy(scale_buf, scale_local)
        if explicit_sync:
            T.sync_threads()
        T.gemm(k_local, q_local, scores, transpose_B=True, clear_accum=True,
               k_pack=2, policy=T.GemmWarpPolicy.FullRow)
        for token, head in T.Parallel(group_size, block_heads):
            scores[token, head] = T.max(scores[token, head], 0.0) * weights[head]
        T.reduce_sum(scores, reduced, dim=1, clear=True)
        for token in T.Parallel(group_size):
            if (tx % 64) // 16 == 0:
                if packed_fp8:
                    reduced[token] = reduced[token] * scale_local[token]
                reduced[token] = reduced[token] / Scale
                if tail:
                    if group_start + token < context_len:
                        Logits[bx, group_start + token] = reduced[token]
                else:
                    Logits[bx, group_start + token] = reduced[token]

    @T.prim_func
    def weighted_mqa_decode_kernel(
        Q: T.Tensor([batch, heads, head_dim], dtype),
        KCache: T.Tensor(cache_shape, dtype),
        Weights: T.Tensor([batch, heads], T.float32),
        PageTable: T.Tensor([batch, max_pages], T.int32),
        ContextLens: T.Tensor([batch], T.int32),
        Logits: T.Tensor([batch, max_model_len], T.float32),
        Scale: T.float32,
    ):
        if packed_fp8:
            ScaleCache = T.view(KCache, [pages, packed_page_size // 4], dtype=T.float32)
        with T.Kernel(batch, persistent_splits if persistent_splits else
                      T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                      threads=threads) as (bx, partition):
            tx = T.get_thread_binding()
            q_shared = T.alloc_shared([block_heads, head_dim], dtype)
            k_shared = T.alloc_shared([group_size, head_dim], dtype)
            layouts = {q_shared: tilelang.layout.make_hcu_swizzled_layout(q_shared, major_pack=2)}
            if not async_keys:
                layouts[k_shared] = tilelang.layout.make_hcu_swizzled_layout(k_shared, major_pack=2)
            T.annotate_layout(layouts)
            q_local = T.alloc_fragment([block_heads, head_dim], dtype)
            k_local = T.alloc_fragment([group_size, head_dim], dtype)
            weights = T.alloc_fragment([block_heads], T.float32)
            scores = T.alloc_fragment([group_size, block_heads], T.float32)
            reduced = T.alloc_fragment([group_size], T.float32)
            # BF16's unused scale fragments are eliminated at compile time.
            scale_local = T.alloc_fragment([group_size], T.float32)
            if scale_async:
                scale_shared = T.alloc_shared([group_size], T.float32)
            scale_buf = scale_shared if scale_async else scale_local
            for head, dim in T.Parallel(block_heads, head_dim):
                q_shared[head, dim] = T.if_then_else(head < heads, Q[bx, head, dim], T.cast(0, dtype))
            if explicit_sync:
                T.sync_threads()
            T.copy(q_shared, q_local)
            if explicit_sync:
                T.sync_threads()
            for head in T.Parallel(block_heads):
                weights[head] = T.if_then_else(head < heads, Weights[bx, head], T.float32(0))

            context_len = ContextLens[bx]
            T.assume(context_len >= 0)
            T.assume(context_len <= max_model_len)
            T.assume(context_len <= max_pages * page_size)
            full_context_groups = context_len // group_size
            context_groups = T.ceildiv(context_len, group_size)
            if persistent_splits:
                begin = context_groups * partition // persistent_splits
                end = context_groups * (partition + 1) // persistent_splits
            else:
                begin = T.min(partition * groups_per_cta, context_groups)
                end = T.min((partition + 1) * groups_per_cta, context_groups)
            full_begin = T.min(begin, full_context_groups)
            full_end = T.min(end, full_context_groups)
            for offset in T.Pipelined(full_end - full_begin, num_stages=pipeline_stages):
                group = full_begin + offset
                load_keys(KCache, PageTable, k_shared, bx, group, context_len, False)
                if packed_fp8:
                    load_scales(ScaleCache, PageTable, scale_buf, bx, group, context_len, False)
                compute(k_shared, k_local, q_local, weights, scores, reduced,
                        scale_buf, scale_local, Logits, Scale, bx, group * group_size, context_len, tx, False)
            if begin <= full_context_groups and full_context_groups < end:
                if full_context_groups < context_groups:
                    group = full_context_groups
                    load_keys(KCache, PageTable, k_shared, bx, group, context_len, True)
                    if packed_fp8:
                        load_scales(ScaleCache, PageTable, scale_buf, bx, group, context_len, True)
                    compute(k_shared, k_local, q_local, weights, scores, reduced,
                            scale_buf, scale_local, Logits, Scale, bx, group * group_size, context_len, tx, True)

    return weighted_mqa_decode_kernel.with_attr("tilelang_pass_configs", compile_options)


def _tilelang_weighted_mqa_decode_kernel(
    group_size=128, groups_per_cta=8, num_stages=1, use_async=True,
    *, heads=4, head_dim=128, page_size=64, threads=256,
):
    return _weighted_decode_kernel(heads, head_dim, page_size, group_size,
                                   groups_per_cta, 0, threads, num_stages, use_async, False, False)


def _tilelang_weighted_mqa_decode_persistent_kernel(
    heads, head_dim=128, page_size=64, group_size=128,
    persistent_splits=2, threads=256, num_stages=1, use_async=True,
):
    assert persistent_splits > 0
    return _weighted_decode_kernel(heads, head_dim, page_size, group_size,
                                   1, persistent_splits, threads, num_stages, use_async, False, False)


def _tilelang_weighted_mqa_decode_fp8_kernel(
    heads, head_dim=128, page_size=64, group_size=64, groups_per_cta=1,
    threads=256, num_stages=0, use_async=False, scale_async=False,
):
    return _weighted_decode_kernel(heads, head_dim, page_size, group_size,
                                   groups_per_cta, 0, threads, num_stages, use_async, True, scale_async)


def _tilelang_weighted_mqa_decode_fp8_persistent_kernel(
    heads, head_dim=128, page_size=64, group_size=128,
    persistent_splits=2, threads=256, num_stages=1, use_async=True, scale_async=False,
):
    assert persistent_splits > 0
    return _weighted_decode_kernel(heads, head_dim, page_size, group_size,
                                   1, persistent_splits, threads, num_stages, use_async, True, scale_async)


def weighted_mqa_decode(q, k_cache, weights, page_table, context_lens,
                        max_model_len, score_scale=None, *, D_out=None):
    """BF16 or packed E4M3FN decode; valid page IDs/lengths are caller guarantees.

    Packed cache holds all page Key bytes followed by page FP32 scales, as
    in pack_fp8_paged_k_cache, not interleaved token records. Query scales
    belong in FP32 weights. The runtime attention denominator is applied
    inside the kernel and defaults to sqrt(D). No host sync is added.
    """
    if q.ndim != 3 or q.shape[1] not in (4, 8, 16, 32, 64) or q.shape[2] != 128:
        raise ValueError("weighted decode requires Q [batch,H,128], H in 4/8/16/32/64")
    scale = math.sqrt(q.shape[2]) if score_scale is None else float(score_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("score_scale must be finite and positive")
    packed = q.dtype == torch.float8_e4m3fn
    dim = 132 if packed else 128
    if (k_cache.ndim != 4 or k_cache.shape[1] not in (8, 16, 32, 64)
            or k_cache.shape[2:] != (1, dim)):
        raise ValueError(f"k_cache must be [pages,page_size,1,{dim}], page_size in 8/16/32/64")
    if packed:
        if k_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("packed FP8 k_cache requires uint8 or float8_e4m3fn storage")
    elif q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        raise ValueError("weighted decode requires BF16 Q/K or E4M3FN Q with packed FP8 cache")
    batch, heads = q.shape[:2]
    if weights.shape != (batch, heads) or weights.dtype != torch.float32:
        raise ValueError("weights must be float32 [batch,H]")
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must be [batch,max_pages]")
    if context_lens.shape != (batch,):
        raise ValueError("context_lens must be [batch]")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (page_table, context_lens)):
        raise ValueError("page_table and context_lens must be int32/int64")
    if not isinstance(max_model_len, int) or max_model_len < 0:
        raise ValueError("max_model_len must be a nonnegative integer")
    if not q.is_cuda or any(t.device != q.device for t in (k_cache, weights, page_table, context_lens)):
        raise ValueError("all inputs must be on the same GPU")
    if D_out is None:
        logits = torch.empty((batch, max_model_len), dtype=torch.float32, device=q.device)
    else:
        if (D_out.shape != (batch, max_model_len) or D_out.dtype != torch.float32
                or D_out.device != q.device or not D_out.is_contiguous()):
            raise ValueError("D_out must be contiguous float32 [batch,max_model_len] on Q.device")
        logits = D_out
    if not batch or not max_model_len:
        return logits
    if not page_table.shape[1] or not k_cache.shape[0]:
        return logits.fill_(-float("inf"))

    from boltops.utility.arch_info import get_device_info
    from ...configs.mqa_decode.weighted import get_config, get_config_fp8
    from ..common.mqa import _get_decode_suffix_splits, _tilelang_qsa_mqa_decode_mask_kernel

    config = (get_config_fp8 if packed else get_config)(
        batch, max_model_len, get_device_info(q.device).cu_count, heads)
    kwargs = dict(heads=heads, head_dim=128, page_size=k_cache.shape[1],
                  group_size=config.group_size, num_stages=config.num_stages,
                  use_async=config.use_async)
    if config.use_persistent:
        kwargs['persistent_splits'] = config.persistent_splits
        factory = (_tilelang_weighted_mqa_decode_fp8_persistent_kernel if packed
                   else _tilelang_weighted_mqa_decode_persistent_kernel)
    else:
        kwargs['groups_per_cta'] = config.groups_per_cta
        factory = _tilelang_weighted_mqa_decode_fp8_kernel if packed else _tilelang_weighted_mqa_decode_kernel
    cache = k_cache.contiguous()
    if packed:
        shape = ((cache.shape[0], 1, cache.shape[1] * 132) if heads > 16
                 else (cache.shape[0], cache.shape[1] * 132))
        cache = cache.view(torch.float8_e4m3fn).reshape(shape)
    lengths = context_lens.to(dtype=torch.int32).contiguous()
    factory(**kwargs)(q.contiguous(), cache, weights.contiguous(),
                      page_table.to(dtype=torch.int32).contiguous(), lengths, logits, scale)
    # Reuse the suffix helper without altering its implementation.
    splits = _get_decode_suffix_splits(batch, max_model_len)
    _tilelang_qsa_mqa_decode_mask_kernel(
        steady_phases=16 if splits == 1 and max_model_len >= 64 * 1024 else 1,
        suffix_splits=splits, reverse_suffix=max_model_len > 8 * 1024,
    )(logits, lengths)
    return logits
