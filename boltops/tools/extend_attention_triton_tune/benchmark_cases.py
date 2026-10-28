# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Inputs for Extend Attention tuning and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


KV_CACHE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


@dataclass(frozen=True)
class ExtendAttentionTuneCase:
    name: str
    batch_size: int
    query_heads: int
    prefix_length: int
    extend_length: int
    kv_heads: int
    sliding_window_size: int = -1
    with_sinks: bool = False
    use_custom_mask: bool = False
    qk_dim: int = 192
    value_dim: int = 128
    tune_equal_seqlens: bool = True
    tune_qo_indptr_dtype: torch.dtype = torch.int32
    tune_kv_indptr_dtype: torch.dtype = torch.int32
    tune_indices_dtype: torch.dtype = torch.int32
    tune_mask_indptr_dtype: torch.dtype = torch.int32
    tune_sinks_dtype: torch.dtype = torch.bfloat16
    tune_zero_sinks: bool = False
    tune_window_offsets_dtype: torch.dtype = torch.int32
    tune_with_window_offsets: bool | None = None
    kv_cache_dtypes: tuple[str, ...] = ("bf16", "fp8")

    @property
    def kind(self) -> str:
        return "short_query" if self.extend_length < 32 else "long_query"


# Fixed production benchmark rows. The default is BF16 with ragged lengths.
BENCHMARK_CASES = (
    ExtendAttentionTuneCase("long-kv1-b2-e555", 2, 16, 4096, 555, 1),
    ExtendAttentionTuneCase("short-kv1-b16-e3", 16, 16, 556, 3, 1),
    ExtendAttentionTuneCase("long-kv1-b16-e32", 16, 16, 556, 32, 1),
    ExtendAttentionTuneCase("long-kv1-b16-e56", 16, 16, 556, 56, 1),
    ExtendAttentionTuneCase("short-kv1-b32-e3", 32, 16, 556, 3, 1),
    ExtendAttentionTuneCase("short-kv1-b64-e3", 64, 16, 556, 3, 1),
    ExtendAttentionTuneCase(
        "long-kv2-swa-b2-e1024", 2, 16, 4096, 1024, 2, 128, True
    ),
    ExtendAttentionTuneCase(
        "short-kv2-swa-b64-e3", 64, 16, 556, 3, 2, 128, True
    ),
)


# Representative long-query and short-query autotune rows.
AUTOTUNE_CASES = (
    ExtendAttentionTuneCase(
        "long-kv2-swa", 4, 16, 8192, 1024, 2, 128, True
    ),
    ExtendAttentionTuneCase("long-kv1", 4, 16, 8192, 1024, 1),
    ExtendAttentionTuneCase(
        "long-kv2-swa-mask", 4, 16, 8192, 1024, 2, 128, True, True
    ),
    ExtendAttentionTuneCase(
        "long-kv1-mask", 4, 16, 8192, 1024, 1, use_custom_mask=True
    ),
    ExtendAttentionTuneCase(
        "short-kv2-swa", 64, 16, 256, 4, 2, 128, True
    ),
    ExtendAttentionTuneCase("short-kv1", 64, 16, 256, 4, 1),
    ExtendAttentionTuneCase(
        "short-kv2-swa-mask", 64, 16, 256, 4, 2, 128, True, True
    ),
    ExtendAttentionTuneCase(
        "short-kv1-mask", 64, 16, 256, 4, 1, use_custom_mask=True
    ),
)


# The benchmark itself remains unchanged. These tuning copies reproduce its
# BF16 input/key profile so the BF16 config JSON also covers every benchmark
# row. Prefix/extend lengths remain ragged, kv_indices is int64, sinks is
# float32, and no window-offset tensor is passed.
BENCHMARK_TUNING_CASES = tuple(
    replace(
        case,
        name=f"benchmark-{case.name}",
        tune_equal_seqlens=False,
        tune_indices_dtype=torch.int64,
        tune_sinks_dtype=torch.float32,
        tune_zero_sinks=True,
        tune_with_window_offsets=False,
        kv_cache_dtypes=("bf16",),
    )
    for case in BENCHMARK_CASES
)


def _historical_case(
    name: str,
    batch_size: int,
    prefix_length: int,
    extend_length: int,
    kv_heads: int,
    *,
    sliding_window_size: int = -1,
    with_sinks: bool = False,
    use_custom_mask: bool = False,
    qo_indptr_dtype: torch.dtype,
    mask_indptr_dtype: torch.dtype = torch.int64,
    window_offsets_dtype: torch.dtype = torch.int32,
    with_window_offsets: bool = False,
) -> ExtendAttentionTuneCase:
    """Describe one full key from the packaged historical FP8 tables."""
    return ExtendAttentionTuneCase(
        name=name,
        batch_size=batch_size,
        query_heads=16,
        prefix_length=prefix_length,
        extend_length=extend_length,
        kv_heads=kv_heads,
        sliding_window_size=sliding_window_size,
        with_sinks=with_sinks,
        use_custom_mask=use_custom_mask,
        tune_qo_indptr_dtype=qo_indptr_dtype,
        tune_indices_dtype=torch.int64,
        tune_mask_indptr_dtype=mask_indptr_dtype,
        tune_window_offsets_dtype=window_offsets_dtype,
        tune_with_window_offsets=with_window_offsets,
        kv_cache_dtypes=("fp8",),
    )


_HISTORICAL_SHORT_BATCHES = (1, 2, 4, 8, 16, 24, 27, 32, 64)

# Union of the gfx938 CU64/CU72 FP8 short-query keys. Each B/extend=4 grid has
# the three attribute/dtype combinations present in the packaged tables.
HISTORICAL_SHORT_TUNING_CASES = tuple(
    case
    for batch_size in _HISTORICAL_SHORT_BATCHES
    for case in (
        _historical_case(
            f"historical-short-mha-mask-b{batch_size}-e4",
            batch_size,
            256,
            4,
            1,
            use_custom_mask=True,
            qo_indptr_dtype=torch.int32,
        ),
        _historical_case(
            f"historical-short-gqa-swa-b{batch_size}-e4",
            batch_size,
            256,
            4,
            2,
            sliding_window_size=128,
            with_sinks=True,
            qo_indptr_dtype=torch.int64,
            with_window_offsets=True,
        ),
        _historical_case(
            f"historical-short-gqa-swa-mask-b{batch_size}-e4",
            batch_size,
            256,
            4,
            2,
            sliding_window_size=128,
            with_sinks=True,
            use_custom_mask=True,
            qo_indptr_dtype=torch.int32,
            window_offsets_dtype=torch.int64,
            with_window_offsets=True,
        ),
    )
)

HISTORICAL_LONG_TUNING_CASES = (
    _historical_case(
        "historical-long-gqa-swa-b1-e16429", 1, 8192, 16429, 2,
        sliding_window_size=128, with_sinks=True,
        qo_indptr_dtype=torch.int64, with_window_offsets=True,
    ),
    _historical_case(
        "historical-long-mha-b1-e16507", 1, 8192, 16507, 1,
        qo_indptr_dtype=torch.int64,
    ),
    _historical_case(
        "historical-long-mha-b1-e20003", 1, 8192, 20003, 1,
        qo_indptr_dtype=torch.int64,
    ),
    _historical_case(
        "historical-long-gqa-swa-b1-e20003", 1, 8192, 20003, 2,
        sliding_window_size=128, with_sinks=True,
        qo_indptr_dtype=torch.int64, with_window_offsets=True,
    ),
)


# Superset manifest: current autotune keys for both cache dtypes, unchanged
# benchmark keys for BF16, and the union of historical CU64/CU72 FP8 keys.
TUNING_CASES = (
    AUTOTUNE_CASES
    + BENCHMARK_TUNING_CASES
    + HISTORICAL_SHORT_TUNING_CASES
    + HISTORICAL_LONG_TUNING_CASES
)


def _causal_custom_mask(
    case: ExtendAttentionTuneCase,
    device: torch.device,
    mask_indptr_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = case.prefix_length + case.extend_length
    query_positions = torch.arange(case.extend_length, device=device)
    query_positions += case.prefix_length
    key_positions = torch.arange(total, device=device)
    one_request = (key_positions[None, :] <= query_positions[:, None]).reshape(-1)
    custom_mask = one_request.repeat(case.batch_size).contiguous()
    segment = case.extend_length * total
    mask_indptr = torch.arange(
        0,
        (case.batch_size + 1) * segment,
        segment,
        device=device,
        dtype=mask_indptr_dtype,
    )
    return custom_mask, mask_indptr


def _make_inputs(
    case: ExtendAttentionTuneCase,
    *,
    seed: int,
    equal_seqlens: bool,
    qo_indptr_dtype: torch.dtype,
    kv_indptr_dtype: torch.dtype,
    indices_dtype: torch.dtype,
    sinks_dtype: torch.dtype,
    zero_sinks: bool,
    mask_indptr_dtype: torch.dtype,
    window_offsets_dtype: torch.dtype,
    with_window_offsets: bool,
    kv_cache_dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    if equal_seqlens:
        extend_lens = torch.full(
            (case.batch_size,), case.extend_length, dtype=torch.int32
        )
        prefix_lens = torch.full(
            (case.batch_size,), case.prefix_length, dtype=torch.int32
        )
    else:
        extend_lens = torch.randint(
            1, case.extend_length + 1, (case.batch_size,), dtype=torch.int32
        )
        prefix_lens = torch.randint(
            1, case.prefix_length + 1, (case.batch_size,), dtype=torch.int32
        )

    qo_indptr_cpu = torch.cat(
        (
            torch.tensor([0], dtype=torch.int32),
            extend_lens.cumsum(0, dtype=torch.int32),
        ),
        dim=0,
    )
    kv_indptr_cpu = torch.cat(
        (
            torch.tensor([0], dtype=torch.int32),
            prefix_lens.cumsum(0, dtype=torch.int32),
        ),
        dim=0,
    )
    total_extend = int(qo_indptr_cpu[-1])
    total_prefix = int(kv_indptr_cpu[-1])
    qo_indptr = qo_indptr_cpu.to(device=device, dtype=qo_indptr_dtype)
    kv_indptr = kv_indptr_cpu.to(device=device, dtype=kv_indptr_dtype)

    q = torch.randn(
        total_extend,
        case.query_heads,
        case.qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        total_extend,
        case.kv_heads,
        case.qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_buffer_bf16 = torch.randn(
        total_prefix,
        case.kv_heads,
        case.qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k[..., : case.value_dim])
    v_buffer_bf16 = torch.randn_like(k_buffer_bf16[..., : case.value_dim])
    # Torch random generation does not support every FP8 backend. Generate
    # representative values in BF16 first, then cast only the persistent KV
    # cache. The current-chunk K/V tensors remain BF16 in both tuning modes.
    k_buffer = k_buffer_bf16.to(kv_cache_dtype)
    v_buffer = v_buffer_bf16.to(kv_cache_dtype)
    kv_indices = torch.arange(total_prefix, device=device, dtype=indices_dtype)
    custom_mask = mask_indptr = None
    if case.use_custom_mask:
        custom_mask, mask_indptr = _causal_custom_mask(
            case, device, mask_indptr_dtype
        )
    if case.with_sinks:
        make_sinks = torch.zeros if zero_sinks else torch.randn
        sinks = make_sinks(case.query_heads, device=device, dtype=sinks_dtype)
    else:
        sinks = None
    window_kv_offsets = None
    if with_window_offsets:
        window_kv_offsets = torch.zeros(
            case.batch_size, device=device, dtype=window_offsets_dtype
        )

    return {
        "q": q,
        "k": k,
        "v": v,
        "output": torch.empty(
            total_extend,
            case.query_heads,
            case.value_dim,
            device=device,
            dtype=torch.bfloat16,
        ),
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "custom_mask": custom_mask,
        "mask_indptr": mask_indptr,
        "sinks": sinks,
        "window_kv_offsets": window_kv_offsets,
    }


def make_benchmark_inputs(
    case: ExtendAttentionTuneCase,
) -> dict[str, torch.Tensor | None]:
    return _make_inputs(
        case,
        seed=0,
        equal_seqlens=False,
        qo_indptr_dtype=torch.int32,
        kv_indptr_dtype=torch.int32,
        indices_dtype=torch.int64,
        sinks_dtype=torch.float32,
        zero_sinks=True,
        mask_indptr_dtype=torch.int32,
        window_offsets_dtype=torch.int32,
        with_window_offsets=False,
        kv_cache_dtype=torch.bfloat16,
    )


def make_tune_inputs(
    case: ExtendAttentionTuneCase,
    *,
    seed: int = 0,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | None]:
    return _make_inputs(
        case,
        seed=seed,
        equal_seqlens=case.tune_equal_seqlens,
        qo_indptr_dtype=case.tune_qo_indptr_dtype,
        kv_indptr_dtype=case.tune_kv_indptr_dtype,
        indices_dtype=case.tune_indices_dtype,
        sinks_dtype=case.tune_sinks_dtype,
        zero_sinks=case.tune_zero_sinks,
        mask_indptr_dtype=case.tune_mask_indptr_dtype,
        window_offsets_dtype=case.tune_window_offsets_dtype,
        with_window_offsets=(
            case.tune_with_window_offsets
            if case.tune_with_window_offsets is not None
            else case.use_custom_mask and case.sliding_window_size > 0
        ),
        kv_cache_dtype=kv_cache_dtype,
    )
