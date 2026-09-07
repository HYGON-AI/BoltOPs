"""Fused Triton kernel for masked Mamba state scatter operations.

This module is intentionally kept close to SGLang's implementation so upstream
changes can be reviewed and synchronized with a small diff.
"""

import torch
import triton
import triton.language as tl


# Tuned on Hygon BW1000_H for the captured TP=2 and derived TP=4 shapes.
# Keep this table deliberately limited to two fixed configurations.
_MAMBA_STATE_SCATTER_CONFIGS = (
    {"BLOCK_SIZE": 1024, "num_warps": 2, "num_stages": 1},
    {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 1},
)


def _select_mamba_state_scatter_config(
    elem_per_entry: int, _total_requests: int
) -> dict[str, int]:
    """Select one of the two fixed configs from the runtime tensor shape."""
    if elem_per_entry >= 131072:
        return _MAMBA_STATE_SCATTER_CONFIGS[1]
    return _MAMBA_STATE_SCATTER_CONFIGS[0]


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    # The physical innermost stride is implicit 1 (contiguous). These flattened
    # state-entry strides only participate in uniform base-address arithmetic.
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused gather-scatter kernel with built-in masking."""
    # Make the contiguous state block the fastest-changing grid dimension. On
    # BW1000_H this preserves long sequential memory streams across adjacent
    # workgroups and avoids the large penalty from request-major dispatch.
    pid_block = tl.program_id(0).to(tl.int64)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_req = tl.program_id(2)

    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)
    if step_idx < 0:
        return

    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)
    src_idx = pid_req

    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    data = tl.load(
        src_ptr + src_offset + offsets, mask=mask, cache_modifier=".cg"
    )
    tl.store(
        dst_ptr + dst_offset + offsets, data, mask=mask, cache_modifier=".cs"
    )


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw destination indices
    step_indices_raw: torch.Tensor,  # [total_requests] - step >= 0 means valid
):
    """Fully fused gather-scatter with built-in masking for Mamba state updates."""
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
        )
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()

    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    config = _select_mamba_state_scatter_config(elem_per_entry, total_requests)
    block_size = config["BLOCK_SIZE"]
    grid = (triton.cdiv(elem_per_entry, block_size), num_layers, total_requests)

    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


__all__ = ["fused_mamba_state_scatter_with_mask"]
