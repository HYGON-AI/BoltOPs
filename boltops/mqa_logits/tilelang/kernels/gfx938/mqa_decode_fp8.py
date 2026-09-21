# SPDX-License-Identifier: MIT
"""gfx938 occupancy FP8 paged MQA decode (packed D+4 cache).

Fork of ``common/mqa_decode.py``.  K uses the same swizzled LDS gather
(``T.Parallel`` + optional ``prefer_async``) and ``T.copy`` into fragments.
Per-token KV scales are gathered with ``T.Parallel`` directly into
``scale_local`` (no scale LDS / ``T.async_copy``).
"""

import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-ffast-math"],
        tilelang.PassConfigKey.TL_Z3_RLIMIT: 50000,
    }
)
def _tilelang_qsa_mqa_decode_fp8_kernel(
    heads: int,
    head_dim: int,
    page_size: int = 64,
    group_size: int = 64,
    groups_per_cta: int = 1,
    threads: int = 256,
    num_stages: int = 0,
    use_async: bool = False,
    scale_async: bool = False,
):
    assert group_size % page_size == 0, page_size
    sub_pages = group_size // page_size
    block_heads = max(16, 1 << (heads - 1).bit_length())
    batch = T.dynamic("batch")
    pages = T.dynamic("pages")
    max_pages = T.dynamic("max_pages")
    max_model_len = T.dynamic("max_model_len")
    gemm_policy = T.GemmWarpPolicy.FullRow
    use_offen = heads > 16
    packed_page_size = page_size * (head_dim + 4)
    k_cache_shape = (
        [pages, 1, packed_page_size]
        if use_offen
        else [pages, packed_page_size]
    )

    @T.macro
    def load_full_group(
        KCache, ScaleCache, PageTable, k_shared, scale_buf, bx, group
    ):
        for token, dim in T.Parallel(
            group_size,
            head_dim,
            prefer_async=use_async,
        ):
            page_slot = token // page_size
            page_offset = token % page_size
            page_id = PageTable[bx, group * sub_pages + page_slot]
            if use_offen:
                k_shared[token, dim] = KCache[
                    page_id, 0, page_offset * head_dim + dim
                ]
            else:
                k_shared[token, dim] = KCache[
                    page_id, page_offset * head_dim + dim
                ]
        for token in T.Parallel(group_size, prefer_async=scale_async):
            page_slot = token // page_size
            page_offset = token % page_size
            page_id = PageTable[bx, group * sub_pages + page_slot]
            scale_buf[token] = ScaleCache[
                page_id, page_size * head_dim // 4 + page_offset
            ]

    @T.macro
    def load_tail_group(
        KCache,
        ScaleCache,
        PageTable,
        k_shared,
        scale_buf,
        bx,
        group,
        context_len,
    ):
        for token, dim in T.Parallel(group_size, head_dim):
            position = group * group_size + token
            page_slot = token // page_size
            page_offset = token % page_size
            if position < context_len:
                page_id = PageTable[bx, group * sub_pages + page_slot]
                if use_offen:
                    k_shared[token, dim] = KCache[
                        page_id, 0, page_offset * head_dim + dim
                    ]
                else:
                    k_shared[token, dim] = KCache[
                        page_id, page_offset * head_dim + dim
                    ]
            else:
                k_shared[token, dim] = T.float8_e4m3fn(0)
        for token in T.Parallel(group_size):
            position = group * group_size + token
            page_slot = token // page_size
            page_offset = token % page_size
            if position < context_len:
                page_id = PageTable[bx, group * sub_pages + page_slot]
                scale_buf[token] = ScaleCache[
                    page_id, page_size * head_dim // 4 + page_offset
                ]
            else:
                scale_buf[token] = T.float32(0)

    @T.macro
    def compute_and_store(
        k_shared,
        k_local,
        q_local,
        scale_local,
        scale_src,
        scores,
        reduced,
        Logits,
        Scale,
        bx,
        group_start,
        context_len,
        tx,
        check_context,
    ):
        T.copy(k_shared, k_local)
        if scale_async:
            T.copy(scale_src, scale_local)
        T.gemm(
            k_local,
            q_local,
            scores,
            transpose_B=True,
            clear_accum=True,
            k_pack=2,
            policy=gemm_policy,
        )
        for token, head in T.Parallel(group_size, block_heads):
            scores[token, head] = T.max(scores[token, head], 0.0)
        T.reduce_sum(scores, reduced, dim=1, clear=True)
        for token in T.Parallel(group_size):
            position = group_start + token
            if (tx % 64) // 16 == 0:
                reduced[token] = reduced[token] * scale_local[token] / Scale
                if check_context:
                    if position < context_len:
                        Logits[bx, position] = reduced[token]
                else:
                    Logits[bx, position] = reduced[token]

    @T.prim_func
    def qsa_mqa_decode_fp8_kernel(
        Q: T.Tensor([batch, 1, heads, head_dim], T.float8_e4m3fn),  # type: ignore
        KCache: T.Tensor(k_cache_shape, T.float8_e4m3fn),  # type: ignore
        PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
        ContextLens: T.Tensor([batch], T.int32),  # type: ignore
        Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
        Scale: T.float32,
    ):
        ScaleCache = T.view(
            KCache,
            [pages, page_size * (head_dim + 4) // 4],
            dtype=T.float32,
        )
        with T.Kernel(
            batch,
            T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
            threads=threads,
        ) as (bx, group_block):
            tx = T.get_thread_binding()
            q_shared = T.alloc_shared([block_heads, head_dim], T.float8_e4m3fn)
            k_shared = T.alloc_shared([group_size, head_dim], T.float8_e4m3fn)
            if scale_async:
                scale_shared = T.alloc_shared([group_size], T.float32)
            shared_layouts = {
                q_shared: tilelang.layout.make_hcu_swizzled_layout(
                    q_shared, major_pack=2
                )
            }
            if not use_async:
                shared_layouts[k_shared] = tilelang.layout.make_hcu_swizzled_layout(
                    k_shared, major_pack=2
                )
            T.annotate_layout(shared_layouts)
            q_local = T.alloc_fragment([block_heads, head_dim], T.float8_e4m3fn)
            k_local = T.alloc_fragment([group_size, head_dim], T.float8_e4m3fn)
            scale_local = T.alloc_fragment([group_size], T.float32)
            scale_buf = scale_shared if scale_async else scale_local
            scores = T.alloc_fragment([group_size, block_heads], T.float32)
            reduced = T.alloc_fragment([group_size], T.float32)
            for head, dim in T.Parallel(block_heads, head_dim):
                q_shared[head, dim] = T.if_then_else(
                    head < heads, Q[bx, 0, head, dim], T.float8_e4m3fn(0)
                )
            T.copy(q_shared, q_local)
            context_len = ContextLens[bx]
            T.assume(context_len >= 0)
            T.assume(context_len <= max_pages * page_size)
            block_group_start = group_block * groups_per_cta
            full_context_groups = context_len // group_size
            context_groups = T.ceildiv(context_len, group_size)
            full_groups = T.max(
                0,
                T.min(groups_per_cta, full_context_groups - block_group_start),
            )
            valid_groups = T.max(
                0, T.min(groups_per_cta, context_groups - block_group_start)
            )

            for gi in T.Pipelined(full_groups, num_stages=num_stages):
                group = block_group_start + gi
                group_start = group * group_size
                load_full_group(
                    KCache,
                    ScaleCache,
                    PageTable,
                    k_shared,
                    scale_buf,
                    bx,
                    group,
                )
                compute_and_store(
                    k_shared,
                    k_local,
                    q_local,
                    scale_local,
                    scale_buf,
                    scores,
                    reduced,
                    Logits,
                    Scale,
                    bx,
                    group_start,
                    context_len,
                    tx,
                    False,
                )

            if full_groups < valid_groups:
                group = block_group_start + full_groups
                group_start = group * group_size
                load_tail_group(
                    KCache,
                    ScaleCache,
                    PageTable,
                    k_shared,
                    scale_buf,
                    bx,
                    group,
                    context_len,
                )
                compute_and_store(
                    k_shared,
                    k_local,
                    q_local,
                    scale_local,
                    scale_buf,
                    scores,
                    reduced,
                    Logits,
                    Scale,
                    bx,
                    group_start,
                    context_len,
                    tx,
                    True,
                )

    return qsa_mqa_decode_fp8_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-ffast-math"],
        tilelang.PassConfigKey.TL_Z3_RLIMIT: 50000,
    }
)
def _tilelang_qsa_mqa_decode_fp8_persistent_kernel(
    heads: int,
    head_dim: int,
    page_size: int = 64,
    group_size: int = 128,
    persistent_splits: int = 2,
    threads: int = 256,
    num_stages: int = 1,
    use_async: bool = True,
    scale_async: bool = False,
):
    assert group_size % page_size == 0, page_size
    assert persistent_splits > 0, persistent_splits
    sub_pages = group_size // page_size
    block_heads = max(16, 1 << (heads - 1).bit_length())
    batch = T.dynamic("batch")
    pages = T.dynamic("pages")
    max_pages = T.dynamic("max_pages")
    max_model_len = T.dynamic("max_model_len")
    use_offen = heads > 16
    packed_page_size = page_size * (head_dim + 4)
    k_cache_shape = (
        [pages, 1, packed_page_size]
        if use_offen
        else [pages, packed_page_size]
    )

    @T.macro
    def load_full_group(
        KCache, ScaleCache, PageTable, k_shared, scale_buf, bx, group
    ):
        for token, dim in T.Parallel(
            group_size,
            head_dim,
            prefer_async=use_async,
        ):
            page_slot = token // page_size
            page_offset = token % page_size
            page_id = PageTable[bx, group * sub_pages + page_slot]
            if use_offen:
                k_shared[token, dim] = KCache[
                    page_id, 0, page_offset * head_dim + dim
                ]
            else:
                k_shared[token, dim] = KCache[
                    page_id, page_offset * head_dim + dim
                ]
        for token in T.Parallel(group_size, prefer_async=scale_async):
            page_slot = token // page_size
            page_offset = token % page_size
            page_id = PageTable[bx, group * sub_pages + page_slot]
            scale_buf[token] = ScaleCache[
                page_id, page_size * head_dim // 4 + page_offset
            ]

    @T.macro
    def load_tail_group(
        KCache,
        ScaleCache,
        PageTable,
        k_shared,
        scale_buf,
        bx,
        group,
        context_len,
    ):
        for token, dim in T.Parallel(group_size, head_dim):
            position = group * group_size + token
            page_slot = token // page_size
            page_offset = token % page_size
            if position < context_len:
                page_id = PageTable[bx, group * sub_pages + page_slot]
                if use_offen:
                    k_shared[token, dim] = KCache[
                        page_id, 0, page_offset * head_dim + dim
                    ]
                else:
                    k_shared[token, dim] = KCache[
                        page_id, page_offset * head_dim + dim
                    ]
            else:
                k_shared[token, dim] = T.float8_e4m3fn(0)
        for token in T.Parallel(group_size):
            position = group * group_size + token
            page_slot = token // page_size
            page_offset = token % page_size
            if position < context_len:
                page_id = PageTable[bx, group * sub_pages + page_slot]
                scale_buf[token] = ScaleCache[
                    page_id, page_size * head_dim // 4 + page_offset
                ]
            else:
                scale_buf[token] = T.float32(0)

    @T.macro
    def compute_and_store(
        k_shared,
        k_local,
        q_local,
        scale_local,
        scale_src,
        scores,
        reduced,
        Logits,
        Scale,
        bx,
        group_start,
        context_len,
        tx,
        check_context,
    ):
        T.copy(k_shared, k_local)
        if scale_async:
            T.copy(scale_src, scale_local)
        T.gemm(
            k_local,
            q_local,
            scores,
            transpose_B=True,
            clear_accum=True,
            k_pack=2,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for token, head in T.Parallel(group_size, block_heads):
            scores[token, head] = T.max(scores[token, head], 0.0)
        T.reduce_sum(scores, reduced, dim=1, clear=True)
        for token in T.Parallel(group_size):
            position = group_start + token
            if (tx % 64) // 16 == 0:
                reduced[token] = reduced[token] * scale_local[token] / Scale
                if check_context:
                    if position < context_len:
                        Logits[bx, position] = reduced[token]
                else:
                    Logits[bx, position] = reduced[token]

    @T.prim_func
    def qsa_mqa_decode_fp8_persistent_kernel(
        Q: T.Tensor([batch, 1, heads, head_dim], T.float8_e4m3fn),  # type: ignore
        KCache: T.Tensor(k_cache_shape, T.float8_e4m3fn),  # type: ignore
        PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
        ContextLens: T.Tensor([batch], T.int32),  # type: ignore
        Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
        Scale: T.float32,
    ):
        ScaleCache = T.view(
            KCache,
            [pages, page_size * (head_dim + 4) // 4],
            dtype=T.float32,
        )
        with T.Kernel(batch, persistent_splits, threads=threads) as (bx, split):
            tx = T.get_thread_binding()
            q_shared = T.alloc_shared([block_heads, head_dim], T.float8_e4m3fn)
            k_shared = T.alloc_shared([group_size, head_dim], T.float8_e4m3fn)
            if scale_async:
                scale_shared = T.alloc_shared([group_size], T.float32)
            shared_layouts = {
                q_shared: tilelang.layout.make_hcu_swizzled_layout(
                    q_shared, major_pack=2
                )
            }
            if not use_async:
                shared_layouts[k_shared] = tilelang.layout.make_hcu_swizzled_layout(
                    k_shared, major_pack=2
                )
            T.annotate_layout(shared_layouts)

            q_local = T.alloc_fragment([block_heads, head_dim], T.float8_e4m3fn)
            k_local = T.alloc_fragment([group_size, head_dim], T.float8_e4m3fn)
            scale_local = T.alloc_fragment([group_size], T.float32)
            scale_buf = scale_shared if scale_async else scale_local
            scores = T.alloc_fragment([group_size, block_heads], T.float32)
            reduced = T.alloc_fragment([group_size], T.float32)
            for head, dim in T.Parallel(block_heads, head_dim):
                q_shared[head, dim] = T.if_then_else(
                    head < heads, Q[bx, 0, head, dim], T.float8_e4m3fn(0)
                )
            T.copy(q_shared, q_local)

            context_len = ContextLens[bx]
            T.assume(context_len >= 0)
            T.assume(context_len <= max_pages * page_size)
            full_context_groups = context_len // group_size
            context_groups = T.ceildiv(context_len, group_size)
            split_begin = context_groups * split // persistent_splits
            split_end = context_groups * (split + 1) // persistent_splits
            full_begin = T.min(split_begin, full_context_groups)
            full_end = T.min(split_end, full_context_groups)

            for group_offset in T.Pipelined(
                full_end - full_begin, num_stages=num_stages
            ):
                group = full_begin + group_offset
                group_start = group * group_size
                load_full_group(
                    KCache,
                    ScaleCache,
                    PageTable,
                    k_shared,
                    scale_buf,
                    bx,
                    group,
                )
                compute_and_store(
                    k_shared,
                    k_local,
                    q_local,
                    scale_local,
                    scale_buf,
                    scores,
                    reduced,
                    Logits,
                    Scale,
                    bx,
                    group_start,
                    context_len,
                    tx,
                    False,
                )

            if split_begin <= full_context_groups and full_context_groups < split_end:
                if full_context_groups < context_groups:
                    group = full_context_groups
                    group_start = group * group_size
                    load_tail_group(
                        KCache,
                        ScaleCache,
                        PageTable,
                        k_shared,
                        scale_buf,
                        bx,
                        group,
                        context_len,
                    )
                    compute_and_store(
                        k_shared,
                        k_local,
                        q_local,
                        scale_local,
                        scale_buf,
                        scores,
                        reduced,
                        Logits,
                        Scale,
                        bx,
                        group_start,
                        context_len,
                        tx,
                        True,
                    )

    return qsa_mqa_decode_fp8_persistent_kernel


__all__ = [
    "_tilelang_qsa_mqa_decode_fp8_kernel",
    "_tilelang_qsa_mqa_decode_fp8_persistent_kernel",
]
