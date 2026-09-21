# SPDX-License-Identifier: MIT
"""Measured gfx938 policy for the paged MQA decode kernel."""

from __future__ import annotations

from functools import lru_cache

from .common import QSAMQADecodeConfig, small_batch_config


def _persistent(splits: int) -> QSAMQADecodeConfig:
    return QSAMQADecodeConfig(persistent_splits=splits)


@lru_cache(maxsize=256)
def get_config(
    batch: int,
    max_model_len: int,
    cu_count: int,
    num_heads: int,
) -> QSAMQADecodeConfig:
    """Choose the robust full/ragged configuration measured on gfx938.

    The length buckets are intentionally based on the allocated decode queue,
    because inspecting ``context_lens`` on the host would introduce a device
    synchronization.  The 64-K policy is reused above 64 K until 128-K-specific
    measurements justify another specialization.
    """
    if batch <= 0:
        raise ValueError(f"QSA MQA decode requires batch > 0, got {batch}")
    if max_model_len <= 0:
        raise ValueError(
            "QSA MQA decode requires max_model_len > 0, got "
            f"{max_model_len}"
        )
    if cu_count <= 0:
        raise ValueError(f"QSA MQA decode requires cu_count > 0, got {cu_count}")
    if num_heads <= 0:
        raise ValueError(f"QSA MQA decode requires num_heads > 0, got {num_heads}")

    # Batch 1 and 8 were not in the original BF16 sweep.  S8 there is only
    # ``batch * 8`` CTAs and loses to a ~256-CTA launch on the FP8 measurements.
    if batch <= 8:
        return small_batch_config(batch, max_model_len, num_heads)

    wide_heads = num_heads > 16

    # Batches above the measured 64-request range already expose ample CTA
    # parallelism.  At 64 K, split16 consistently improves H32/H64 full and
    # ragged workloads; smaller heads and shorter queues retain split8.
    if batch > 64:
        if wide_heads and max_model_len > 32 * 1024:
            return _persistent(16)
        return _persistent(8)

    if max_model_len <= 8 * 1024:
        if batch <= 16:
            if wide_heads:
                return _persistent(8)
            return QSAMQADecodeConfig()
        return _persistent(4 if batch <= 32 else 8)

    if max_model_len <= 16 * 1024:
        if batch <= 16:
            return _persistent(8)
        if batch <= 32:
            return _persistent(16 if wide_heads else 8)
        return _persistent(16)

    if max_model_len <= 32 * 1024:
        if batch <= 16:
            return _persistent(8)
        return _persistent(16)

    # 64 K: B16/split8, B32/split32, B64/split32.  Reuse this bucket for
    # 128 K and larger queues for now.
    return _persistent(8 if batch <= 16 else 32)


@lru_cache(maxsize=256)
def get_config_fp8(
    batch: int,
    max_model_len: int,
    cu_count: int,
    num_heads: int,
) -> QSAMQADecodeConfig:
    """FP8 occupancy policy.

    Batch 1-16 target about 256 CTAs.  Batch 1 uses an ordinary G launch
    because persistent splits stop helping before the G grid does.  Batch 8
    and 16 use persistent splits so ragged rows are cut by their real
    ``context_len``.  Larger batches keep the BF16 table.
    """
    get_config(batch, max_model_len, cu_count, num_heads)
    if batch <= 16:
        return small_batch_config(batch, max_model_len, num_heads)
    return get_config(batch, max_model_len, cu_count, num_heads)


__all__ = ["get_config", "get_config_fp8"]
