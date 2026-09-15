"""Public lazy wrappers for query-specific attention operators."""

from __future__ import annotations


def qsa_mqa_paged(*args, **kwargs):
    """Compute paged MQA query-specific attention scores."""
    from boltops.generic.triton.qsa import qsa_mqa_paged as impl

    return impl(*args, **kwargs)


def expand_qsa_block_indices(*args, **kwargs):
    """Expand selected QSA block indices to logical token indices."""
    from boltops.generic.triton.qsa import expand_qsa_block_indices as impl

    return impl(*args, **kwargs)


def qsa_select_paged_tokens(*args, **kwargs):
    """Select paged QSA token indices."""
    from boltops.generic.triton.qsa import qsa_select_paged_tokens as impl

    return impl(*args, **kwargs)


def qsa_sparse_paged_attention(*args, **kwargs):
    """Compute sparse paged attention for selected QSA tokens."""
    from boltops.generic.triton.qsa import qsa_sparse_paged_attention as impl

    return impl(*args, **kwargs)


__all__ = [
    "qsa_mqa_paged",
    "expand_qsa_block_indices",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
]
