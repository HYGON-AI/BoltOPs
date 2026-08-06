# SPDX-License-Identifier: Apache-2.0
"""Public API and routing for the two causal conv1d Triton kernel variants.

Everything that is not kernel logic lives here:

- constants and platform stubs (``PAD_SLOT_ID`` / ``NULL_BLOCK_ID`` / PDL);
- backend routing used by the unified entry points (:mod:`kernels`).

Upstream layout
---------------
- vLLM: ``vllm/model_executor/layers/mamba/ops/causal_conv1d.py`` (main)
- SGLang: ``sglang/kernels/ops/mamba/causal_conv1d_triton.py`` (ROCm/Triton
  fallback; the SGLang fast path moved to ``sgl_kernel``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

# ---------------------------------------------------------------------------
# Constants and platform stubs
# ---------------------------------------------------------------------------

PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0


def is_arch_support_pdl() -> bool:
    """PDL (programmatic dependent launch) is CUDA-only and unsupported here."""
    return False


BACKEND_AUTO = "auto"
BACKEND_SGLANG = "sglang"
BACKEND_VLLM = "vllm"
BACKENDS = (BACKEND_AUTO, BACKEND_SGLANG, BACKEND_VLLM)


def normalize_backend(backend: str) -> str:
    """Validate a user-supplied backend name."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    return backend


def _any_set(*values: Any) -> bool:
    return any(v is not None for v in values)


def resolve_fwd_backend(
    backend: str = BACKEND_AUTO,
    *,
    seq_lens_cpu: Optional[List[int]] = None,
    metadata: Any = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
) -> str:
    """Pick the port for a fwd call.

    Explicit ``backend`` always wins. In ``auto`` mode:

    - ``seq_lens_cpu`` is the only SGLang-only fwd signal;
    - vLLM V1 scheduler metadata / APC tensors / block alignment are the
      vLLM-only signals;
    - with no signal at all the vLLM port is the default base.
    """
    backend = normalize_backend(backend)
    if backend != BACKEND_AUTO:
        return backend
    sglang_hint = seq_lens_cpu is not None
    vllm_hint = _any_set(
        metadata,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
    ) or (block_size_to_align not in (0, None))
    if sglang_hint and vllm_hint:
        raise ValueError(
            "ambiguous backend: both SGLang-only (seq_lens_cpu) and vLLM-only "
            "(metadata / APC tensors / block_size_to_align) arguments were "
            "provided; pass backend='sglang' or backend='vllm' explicitly"
        )
    return BACKEND_SGLANG if sglang_hint else BACKEND_VLLM


def resolve_update_backend(
    backend: str = BACKEND_AUTO,
    *,
    cache_seqlens: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: Optional[int] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> str:
    """Pick the port for an update call.

    Explicit ``backend`` always wins. In ``auto`` mode:

    - SGLang-only signals: circular ``cache_seqlens``, spec-decode / Eagle tree
      buffers (``num_accept_tokens``, ``intermediate_*``, ``retrieve_*``);
    - vLLM-only signals: ``num_accepted_tokens``, varlen ``query_start_loc`` /
      ``max_query_len``, APC tensors, ``null_block_id``, ``out``;
    - with no signal at all the vLLM port is the default base.
    """
    backend = normalize_backend(backend)
    if backend != BACKEND_AUTO:
        return backend
    sglang_hint = _any_set(
        cache_seqlens,
        num_accept_tokens,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
    )
    vllm_hint = _any_set(
        num_accepted_tokens,
        query_start_loc,
        null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
    ) or (max_query_len != -1)
    if sglang_hint and vllm_hint:
        raise ValueError(
            "ambiguous backend: both SGLang-only (cache_seqlens / spec-decode / "
            "Eagle tree) and vLLM-only (num_accepted_tokens / query_start_loc / "
            "APC / null_block_id / out) arguments were provided; pass "
            "backend='sglang' or backend='vllm' explicitly"
        )
    return BACKEND_SGLANG if sglang_hint else BACKEND_VLLM
def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Union[str, bool, None] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: Optional[int] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
    metadata: Any = None,
    backend: str = "auto",
    validate_data: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Varlen causal depthwise conv1d (extend / prefill).

    ``x`` is ``[dim, cu_seqlen]`` with sequences concatenated along the token
    axis; ``query_start_loc`` is the cumulative lengths tensor ``[batch + 1]``.
    See the per-port docstrings for the full parameter semantics; parameters
    that belong to the other backend are simply ignored.

    Args:
        backend: ``"auto"`` | ``"sglang"`` | ``"vllm"``.
        seq_lens_cpu: required by the SGLang port (per-sequence lengths on CPU,
            used for grid computation).
        metadata: vLLM V1 scheduler metadata (compact launch arrays). When
            omitted the vLLM port derives them from ``query_start_loc``.
    """
    if query_start_loc is None:
        raise ValueError("query_start_loc (cumulative sequence lengths) is required")
    if conv_states is None:
        raise ValueError(
            "conv_states is required: both kernels update the conv-state cache "
            "in place"
        )
    backend = resolve_fwd_backend(
        backend,
        seq_lens_cpu=seq_lens_cpu,
        metadata=metadata,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens,
        block_size_to_align=block_size_to_align,
    )
    if backend == BACKEND_SGLANG:
        if seq_lens_cpu is None:
            raise ValueError(
                "the 'sglang' backend requires seq_lens_cpu (per-sequence "
                "lengths as a CPU list)"
            )
        from .kernels import causal_conv1d_fn_sglang as impl

        return impl(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            seq_lens_cpu,
            cache_indices,
            has_initial_state,
            activation,
            pad_slot_id,
            validate_data,
            **kwargs,
        )

    # vLLM backend (default base).
    padded_batch = query_start_loc.size(0) - 1
    if cache_indices is None:
        if conv_states.size(0) < padded_batch:
            raise ValueError(
                f"conv_states has {conv_states.size(0)} cache lines but the batch "
                f"needs {padded_batch} (identity cache mapping)"
            )
        cache_indices = torch.arange(padded_batch, dtype=torch.int32, device=x.device)
    if has_initial_state is None:
        has_initial_state = torch.zeros(
            padded_batch, dtype=torch.bool, device=x.device
        )
    if null_block_id is None:
        # Unified contract: the skip marker is ``pad_slot_id`` on both
        # backends. vLLM's own default (block 0 = null block) is not used here.
        null_block_id = pad_slot_id
    from .kernels import causal_conv1d_fn_vllm as impl

    return impl(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
        metadata,
        validate_data,
        **kwargs,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[str, bool, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    # SGLang-only
    cache_seqlens: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # vLLM-only
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: Optional[int] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # control
    backend: str = "auto",
    validate_data: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Decode-step causal conv1d + conv-state sliding update.

    ``x`` is ``[batch, dim]`` (single token) or ``[batch, dim, seqlen]``
    (multi-token / draft verification). See the per-port docstrings for the
    full parameter semantics.

    ``num_accept_tokens`` (SGLang spelling) and ``num_accepted_tokens`` (vLLM
    spelling) mean the same thing; either is accepted, the vLLM one routes to
    the vLLM port.

    Args:
        backend: ``"auto"`` | ``"sglang"`` | ``"vllm"``.
        out: optional pre-allocated output. When omitted a fresh tensor is
            allocated (``x`` is never overwritten).
    """
    backend = resolve_update_backend(
        backend,
        cache_seqlens=cache_seqlens,
        num_accept_tokens=num_accept_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        null_block_id=null_block_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        out=out,
    )
    if backend == BACKEND_SGLANG:
        metadata = kwargs.pop("metadata", None)
        from .kernels import causal_conv1d_update_sglang as impl

        return impl(
            x,
            conv_state,
            weight,
            bias,
            activation,
            cache_seqlens,
            conv_state_indices,
            num_accept_tokens,
            intermediate_conv_window,
            intermediate_state_indices,
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
            pad_slot_id,
            metadata,
            validate_data,
        )

    # vLLM backend (default base).
    if num_accepted_tokens is None:
        num_accepted_tokens = num_accept_tokens
    if conv_state_indices is None:
        batch = (
            query_start_loc.size(0) - 1
            if query_start_loc is not None
            else x.shape[0]
        )
        if conv_state.size(0) < batch:
            raise ValueError(
                f"conv_state has {conv_state.size(0)} cache lines but the batch "
                f"needs {batch} (identity cache mapping)"
            )
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
    if out is None:
        out = torch.empty_like(x)
    if null_block_id is None:
        null_block_id = pad_slot_id
    from .kernels import causal_conv1d_update_vllm as impl

    return impl(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
        out,
    )


__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
