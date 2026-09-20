# SPDX-License-Identifier: MIT
"""Weight-free TileLang MQA logits (prefill + paged decode).

Port of qwen38 ``src/qsa/mqa.py``. Prefill N-range loads Starts/Ends, then
``reduce_min`` / ``reduce_max``, then clamps the scalar range against ``keys``.
"""

import math
from typing import Optional

import torch

from boltops.mqa_logits.tilelang.configs.mqa_decode import resolve_get_config
from boltops.utility.arch_info import get_device_info

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


def _is_hip() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def _get_decode_suffix_splits(batch: int, max_model_len: int) -> int:
    """Choose enough suffix CTAs to cover roughly one gfx938 CU round."""
    if max_model_len <= 8 * 1024 or batch >= 64:
        return 1
    splits_to_fill = (64 + batch - 1) // batch
    return min(16, 1 << (splits_to_fill - 1).bit_length())


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


if HAS_TILELANG:
    from .mqa_decode import (
        _tilelang_qsa_mqa_decode_kernel,
        _tilelang_qsa_mqa_decode_persistent_kernel,
    )

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        apply_scale: bool = True,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def qsa_mqa_prefill_generic(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
            InvScale: T.float32,
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                starts_f = T.alloc_fragment([block_q], T.int32)
                ends_f = T.alloc_fragment([block_q], T.int32)
                start_min_f = T.alloc_fragment([1], T.int32)
                end_max_f = T.alloc_fragment([1], T.int32)
                row_base = bx * block_q
                for qi in T.Parallel(block_q):
                    starts_f[qi] = Starts[row_base + qi]
                    ends_f[qi] = Ends[row_base + qi]
                T.reduce_min(starts_f, start_min_f, dim=0, clear=True)
                T.reduce_max(ends_f, end_max_f, dim=0, clear=True)
                start_min = T.min(start_min_f[0], keys)
                end_max = T.min(end_max_f[0], keys)

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        if apply_scale:
                            Logits[row_base + qi, start_min + ni * block_n + n] = (
                                reduced[n, qi] * InvScale
                            )
                        else:
                            Logits[row_base + qi, start_min + ni * block_n + n] = (
                                reduced[n, qi]
                            )

        return qsa_mqa_prefill_generic

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def _tilelang_qsa_mqa_mask_kernel(
        threads: int = 256,
        steady_phases: int = 1,
        row_phases: int = 1,
        reverse_suffix: bool = False,
    ):
        assert steady_phases in (1, 2, 4, 8, 16, 32)
        assert row_phases in (1, 2, 4, 8, 16, 32)
        stores_per_thread = 4
        elements_per_round = threads * stores_per_thread
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.macro
        def store_round(logits, bx, tx, segment_start, item):
            base = segment_start + item * elements_per_round + tx
            for v in T.Unroll(stores_per_thread):
                logits[bx, base + v * threads] = -T.infinity(T.float32)

        @T.macro
        def store_steady(logits, bx, tx, segment_start, rounds):
            if steady_phases > 1:
                phase_offset = T.bitwise_and(bx, steady_phases - 1) * (
                    rounds // steady_phases
                )
                for item in T.serial(rounds):
                    candidate_item = phase_offset + item
                    phased_item = T.if_then_else(
                        candidate_item >= rounds,
                        candidate_item - rounds,
                        candidate_item,
                    )
                    store_round(logits, bx, tx, segment_start, phased_item)
            else:
                for item in T.serial(rounds):
                    store_round(logits, bx, tx, segment_start, item)

        @T.macro
        def store_steady_reverse(logits, bx, tx, segment_start, rounds):
            if steady_phases > 1:
                phase_offset = T.bitwise_and(bx, steady_phases - 1) * (
                    rounds // steady_phases
                )
                for item in T.serial(rounds):
                    candidate_item = phase_offset + item
                    phased_item = T.if_then_else(
                        candidate_item >= rounds,
                        candidate_item - rounds,
                        candidate_item,
                    )
                    store_round(
                        logits,
                        bx,
                        tx,
                        segment_start,
                        rounds - phased_item - 1,
                    )
            else:
                for item in T.serial(rounds):
                    store_round(
                        logits,
                        bx,
                        tx,
                        segment_start,
                        rounds - item - 1,
                    )

        @T.macro
        def mask_row(logits, starts, ends, row, tx):
            start = T.max(0, T.min(starts[row], keys))
            end = T.max(start, T.min(ends[row], keys))

            # Enumerate only the invalid prefix and suffix.  Each steady round
            # issues four scalar dword stores per thread while keeping wave
            # stores contiguous.
            prefix_rounds = start // elements_per_round
            store_steady(logits, row, tx, 0, prefix_rounds)
            prefix_tail = prefix_rounds * elements_per_round + tx
            if prefix_rounds * elements_per_round < start:
                for v in T.Unroll(stores_per_thread):
                    column = prefix_tail + v * threads
                    if column < start:
                        logits[row, column] = -T.infinity(T.float32)

            suffix = keys - end
            suffix_rounds = suffix // elements_per_round
            if reverse_suffix:
                # Anchor full rounds at keys so their wave stores stay aligned
                # even when each row has a different, unaligned end.  Enumerate
                # the rounds backwards; only the short interval adjacent to end
                # remains predicated.
                suffix_steady_start = keys - suffix_rounds * elements_per_round
                store_steady_reverse(
                    logits,
                    row,
                    tx,
                    suffix_steady_start,
                    suffix_rounds,
                )
                suffix_head = end + tx
                if end < suffix_steady_start:
                    for v in T.Unroll(stores_per_thread):
                        column = suffix_head + v * threads
                        if column < suffix_steady_start:
                            logits[row, column] = -T.infinity(T.float32)
            else:
                store_steady(logits, row, tx, end, suffix_rounds)
                suffix_tail = end + suffix_rounds * elements_per_round + tx
                if end + suffix_rounds * elements_per_round < keys:
                    for v in T.Unroll(stores_per_thread):
                        column = suffix_tail + v * threads
                        if column < keys:
                            logits[row, column] = -T.infinity(T.float32)

        @T.prim_func
        def qsa_mqa_mask(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                if row_phases > 1:
                    # Interleave equally sized row groups without changing the
                    # CTA count.  The base/remainder formula is a bijection for
                    # non-divisible row counts such as 4609.
                    phase = bx % row_phases
                    offset = bx // row_phases
                    rows_per_phase = rows // row_phases
                    extra_rows = rows % row_phases
                    row = (
                        phase * rows_per_phase
                        + T.min(phase, extra_rows)
                        + offset
                    )
                    mask_row(Logits, Starts, Ends, row, tx)
                else:
                    mask_row(Logits, Starts, Ends, bx, tx)

        return qsa_mqa_mask

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def _tilelang_qsa_mqa_decode_mask_kernel(
        threads: int = 256,
        steady_phases: int = 1,
        suffix_splits: int = 1,
        reverse_suffix: bool = False,
    ):
        """Fill only the invalid decode suffix after ``context_len``."""
        assert steady_phases in (1, 2, 4, 8, 16, 32)
        assert suffix_splits in (1, 2, 4, 8, 16)
        stores_per_thread = 4
        elements_per_round = threads * stores_per_thread
        batch = T.dynamic("batch")
        keys = T.dynamic("keys")

        @T.macro
        def store_round(logits, bx, tx, segment_start, item):
            base = segment_start + item * elements_per_round + tx
            for v in T.Unroll(stores_per_thread):
                logits[bx, base + v * threads] = -T.infinity(T.float32)

        @T.prim_func
        def qsa_mqa_decode_mask(
            Logits: T.Tensor([batch, keys], T.float32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
        ):
            if suffix_splits == 1:
                with T.Kernel(batch, threads=threads) as bx:
                    tx = T.thread_binding(0, threads, thread="threadIdx.x")
                    end = T.max(0, T.min(ContextLens[bx], keys))
                    suffix = keys - end
                    rounds = suffix // elements_per_round
                    if reverse_suffix:
                        steady_start = keys - rounds * elements_per_round
                        if steady_phases > 1:
                            phase_offset = T.bitwise_and(
                                bx, steady_phases - 1
                            ) * (rounds // steady_phases)
                            for item in T.serial(rounds):
                                candidate_item = phase_offset + item
                                phased_item = T.if_then_else(
                                    candidate_item >= rounds,
                                    candidate_item - rounds,
                                    candidate_item,
                                )
                                store_round(
                                    Logits,
                                    bx,
                                    tx,
                                    steady_start,
                                    rounds - phased_item - 1,
                                )
                        else:
                            for item in T.serial(rounds):
                                store_round(
                                    Logits,
                                    bx,
                                    tx,
                                    steady_start,
                                    rounds - item - 1,
                                )

                        head = end + tx
                        if end < steady_start:
                            for v in T.Unroll(stores_per_thread):
                                column = head + v * threads
                                if column < steady_start:
                                    Logits[bx, column] = -T.infinity(T.float32)
                    else:
                        if steady_phases > 1:
                            phase_offset = T.bitwise_and(
                                bx, steady_phases - 1
                            ) * (rounds // steady_phases)
                            for item in T.serial(rounds):
                                candidate_item = phase_offset + item
                                phased_item = T.if_then_else(
                                    candidate_item >= rounds,
                                    candidate_item - rounds,
                                    candidate_item,
                                )
                                store_round(Logits, bx, tx, end, phased_item)
                        else:
                            for item in T.serial(rounds):
                                store_round(Logits, bx, tx, end, item)

                        tail = end + rounds * elements_per_round + tx
                        if rounds * elements_per_round < suffix:
                            for v in T.Unroll(stores_per_thread):
                                column = tail + v * threads
                                if column < keys:
                                    Logits[bx, column] = -T.infinity(T.float32)
            else:
                # Keep split as blockIdx.x so adjacent CTAs work on different
                # contiguous pieces of one row before advancing to the next
                # batch.  About 64 total CTAs is enough to fill gfx938.
                with T.Kernel(suffix_splits, batch, threads=threads) as (
                    split,
                    bx,
                ):
                    tx = T.thread_binding(0, threads, thread="threadIdx.x")
                    end = T.max(0, T.min(ContextLens[bx], keys))
                    if reverse_suffix:
                        rounds = (keys - end) // elements_per_round
                        steady_start = keys - rounds * elements_per_round
                        split_begin = rounds * split // suffix_splits
                        split_end = rounds * (split + 1) // suffix_splits
                        for item in T.serial(split_end - split_begin):
                            reverse_item = rounds - (split_begin + item) - 1
                            store_round(
                                Logits,
                                bx,
                                tx,
                                steady_start,
                                reverse_item,
                            )
                        if split == suffix_splits - 1 and end < steady_start:
                            head = end + tx
                            for v in T.Unroll(stores_per_thread):
                                column = head + v * threads
                                if column < steady_start:
                                    Logits[bx, column] = -T.infinity(T.float32)
                    else:
                        rounds = T.ceildiv(keys - end, elements_per_round)
                        split_begin = rounds * split // suffix_splits
                        split_end = rounds * (split + 1) // suffix_splits
                        for item in T.serial(split_end - split_begin):
                            base = (
                                end
                                + (split_begin + item) * elements_per_round
                                + tx
                            )
                            for v in T.Unroll(stores_per_thread):
                                column = base + v * threads
                                if column < keys:
                                    Logits[bx, column] = -T.infinity(T.float32)

        return qsa_mqa_decode_mask

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def _tilelang_qsa_mqa_fill_kernel(
        threads: int = 256,
        blocks: int = 72,
    ):
        """Fill a contiguous logits view using a fixed persistent CTA grid."""
        stores_per_thread = 4
        elements_per_cta = threads * stores_per_thread
        elements = T.dynamic("elements")

        @T.prim_func
        def qsa_mqa_fill(
            Logits: T.Tensor([elements], T.float32),  # type: ignore
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for item in T.serial(T.ceildiv(elements, blocks * elements_per_cta)):
                    base = (item * blocks + bx) * elements_per_cta + tx
                    for v in T.Unroll(stores_per_thread):
                        index = base + v * threads
                        if index < elements:
                            Logits[index] = -T.infinity(T.float32)

        return qsa_mqa_fill

def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    effective_range: Optional[int] = None,
) -> torch.Tensor:
    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    # Keep the generic GEMM at a stable logical-M=128 across head counts.
    # Pipeline depth remains backend-specific, but launch geometry should not
    # silently grow from M=128 to M=32*heads on HIP.
    threads = 256
    block_q = max(1, 128 // heads)
    num_stages = 1 if _is_hip() else 3
    # The generic kernel keeps TileLang's safe-memory pass enabled, so its
    # final partial CTA is predicated in generated device code. Preserve the
    # true row extent instead of padding and copying every external tensor.
    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    q_kernel = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    inv_scale = 1.0 / (score_scale or math.sqrt(head_dim))
    _ = effective_range

    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
        num_stages=num_stages,
        threads=threads,
        apply_scale=inv_scale != 1.0,
    )(
        q_kernel.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
        inv_scale,
    )
    _tilelang_qsa_mqa_mask_kernel()(logits, starts, ends)
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    if not q.is_cuda:
        raise ValueError("TileLang QSA decode requires device tensors")
    query_heads, head_dim = q.shape[1:]
    if query_heads not in (4, 8, 16, 32, 64):
        raise ValueError(
            "TileLang QSA decode requires query heads in 4/8/16/32/64, "
            f"got {query_heads}"
        )
    if head_dim != 128:
        raise ValueError(
            f"TileLang QSA decode requires head_dim=128, got {head_dim}"
        )
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.empty(
        (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
    )
    if not q.shape[0] or not max_model_len:
        return logits
    q_kernel = q.to(torch.bfloat16)
    device_info = get_device_info(q.device)
    config = resolve_get_config(device_info.arch)(
        q.shape[0],
        max_model_len,
        device_info.cu_count,
        num_heads=query_heads,
    )
    kernel_factory = (
        _tilelang_qsa_mqa_decode_persistent_kernel
        if config.use_persistent
        else _tilelang_qsa_mqa_decode_kernel
    )
    kernel_kwargs = dict(
        heads=query_heads,
        head_dim=head_dim,
        page_size=page_size,
        group_size=config.group_size,
        num_stages=config.num_stages,
        use_async=config.use_async,
    )
    if config.use_persistent:
        kernel_kwargs["persistent_splits"] = config.persistent_splits
    else:
        kernel_kwargs["groups_per_cta"] = config.groups_per_cta
    context_lens_kernel = context_lens.to(
        device=q.device, dtype=torch.int32
    ).contiguous()
    kernel_factory(**kernel_kwargs)(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens_kernel,
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    suffix_splits = _get_decode_suffix_splits(q.shape[0], max_model_len)
    _tilelang_qsa_mqa_decode_mask_kernel(
        steady_phases=(
            16 if suffix_splits == 1 and max_model_len >= 64 * 1024 else 1
        ),
        suffix_splits=suffix_splits,
        reverse_suffix=max_model_len > 8 * 1024,
    )(logits, context_lens_kernel)
    return logits


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    effective_range: Optional[int] = None,
) -> torch.Tensor:
    return tilelang_qsa_mqa_prefill(
        q, k, row_starts, row_ends, score_scale, effective_range
    )


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    return tilelang_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )
