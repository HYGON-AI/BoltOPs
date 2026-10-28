# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared data model for TileLang MQA decode launch configurations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class QSAMQADecodeConfig:
    """Compile-time parameters for the shared paged-decode kernel."""

    group_size: int = 128
    groups_per_cta: int = 8
    persistent_splits: int = 0
    num_stages: int = 1
    use_async: bool = True

    @property
    def use_persistent(self) -> bool:
        """Whether to use dynamic per-batch context partitioning."""
        return self.persistent_splits > 0


def _persistent(splits: int) -> QSAMQADecodeConfig:
    return QSAMQADecodeConfig(persistent_splits=splits)


# Enough CTAs to cover a few waves on 72 CUs.  Batch 1 reaches this with a
# G launch; batch 2-8 does it with persistent splits.
_TARGET_CTAS = 256


def _floor_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def small_batch_config(
    batch: int, max_model_len: int, num_heads: int
) -> QSAMQADecodeConfig:
    """Launch about ``_TARGET_CTAS`` CTAs for batches the old table skipped.

    ``groups_per_cta`` and ``persistent_splits`` come from the allocated queue
    (``max_model_len / 128``), not from the live ``context_lens``.  H64 wants
    about half as many CTAs on a single long request.  Splits are capped so
    each one still owns at least four groups.
    """
    groups = max(1, max_model_len // 128)
    target = 128 if num_heads >= 64 and batch <= 1 else _TARGET_CTAS
    if batch <= 1:
        groups_per_cta = _floor_pow2(max(1, groups // target))
        return QSAMQADecodeConfig(groups_per_cta=groups_per_cta)
    splits = _floor_pow2(min(groups, max(1, _TARGET_CTAS // batch)))
    splits = min(splits, _floor_pow2(max(1, groups // 4)))
    return _persistent(splits)


def _reference_config(
    batch: int, max_model_len: int, num_heads: int
) -> QSAMQADecodeConfig:
    """Return the policy measured on the 72-CU gfx938 reference device."""
    if batch <= 8:
        return small_batch_config(batch, max_model_len, num_heads)
    wide_heads = num_heads > 16
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

    return _persistent(8 if batch <= 16 else 32)


def _ceil_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _scale_parallelism(
    config: QSAMQADecodeConfig, cu_count: int
) -> QSAMQADecodeConfig:
    """Preserve approximately the gfx938 CTA/CU rounds on another CU count."""
    reference_cus = 72
    if config.use_persistent:
        scaled = (config.persistent_splits * cu_count + reference_cus - 1) // (
            reference_cus
        )
        splits = min(32, _ceil_power_of_two(max(1, scaled)))
        return QSAMQADecodeConfig(
            group_size=config.group_size,
            groups_per_cta=config.groups_per_cta,
            persistent_splits=splits,
            num_stages=config.num_stages,
            use_async=config.use_async,
        )

    # Keep the measured ordinary-kernel GPC.  Scaling GPC would also shorten
    # its steady loop, which has not been measured outside gfx938.
    return config


@lru_cache(maxsize=256)
def get_config(
    batch: int,
    max_model_len: int,
    cu_count: int,
    num_heads: int,
) -> QSAMQADecodeConfig:
    """Scale the measured gfx938 policy to preserve similar CTA/CU rounds."""
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

    return _scale_parallelism(
        _reference_config(batch, max_model_len, num_heads), cu_count
    )


__all__ = ["QSAMQADecodeConfig", "get_config"]
