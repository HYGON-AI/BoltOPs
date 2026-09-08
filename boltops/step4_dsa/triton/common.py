"""Shared Python helpers for Step-4 DSA Triton kernels.

Vendored from the Step-4 DSA handoff ``inference/kernel.py`` (dsa_kernels /
qknorm fp-fusion launch flag). Triton ``@jit`` helpers stay next to the kernels
that call them; this module is Python-only.
"""

from __future__ import annotations

import importlib

import torch


def _device_name_for_config() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name().replace(" ", "_")
    compact = name.lower().replace("_", "").replace("-", "")
    # BW1100 must win over the BW200 substring check used by GEMM loaders.
    if "bw1100" in compact:
        return "BW1100"
    if "bw200" in compact:
        return "BW200"
    if "k100" in compact:
        return "K100_AI"
    return name


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype).replace("torch.", "")


# Step-4 compresses every 8 consecutive tokens into one summary vector. The CuTe
# kernels hard-specialize on this; we keep it a parameter but never exercise
# anything else.
REGION_BLOCK_SIZE = 8
PROXY_DIM = 256


def _shared_memory_limit() -> int:
    """Bytes of shared memory one block may use, or 0 when that is unknowable.

    The sparse attention kernels stage a whole region-block of keys *and* values in
    shared memory, so their tile size is bounded by the device rather than by the
    geometry.  Hopper allows 227 KiB per block and the ``block_regions=16`` default is
    sized for it; a 64 KiB device (CDNA/DCU) cannot launch that tile at all and Triton
    reports OutOfResources at first launch rather than at import.  Ask the driver
    instead of hardcoding either number.
    """
    try:
        from triton.runtime.driver import driver

        return int(
            driver.active.utils.get_device_properties(
                torch.cuda.current_device()
            )["max_shared_mem"]
        )
    except Exception:  # pragma: no cover - no driver, no device, or no such property.
        return 0


def _fit_block_regions(
    block_regions: int,
    *,
    region_size: int,
    block_d: int,
    element_size: int,
) -> int:
    """Shrink ``block_regions`` until one key+value tile fits in shared memory.

    Halved rather than reduced by one so the region count stays a power of two, which
    keeps the ``tl.dot`` operand shapes the kernels are tuned around.  Returns at least
    one region-block: a device that cannot stage even one is a real failure, and the
    OutOfResources error names the shape better than anything raised here would.
    """
    limit = _shared_memory_limit()
    if limit <= 0:
        return block_regions
    while block_regions > 1:
        if 2 * block_regions * region_size * block_d * element_size <= limit:
            break
        block_regions //= 2
    return block_regions


def _require_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    """Sparse attention kernels address the last axis as ``base + d``, not ``d * stride``.

    Token and head strides are kernel arguments; the innermost dimension is not.
    A legal shape with ``stride(-1) != 1`` (permute, stepped slice) would otherwise
    be silently mis-addressed. Callers may still pad token/head strides.
    """
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in the last dimension")


def shared_table_stride(block_table: torch.Tensor) -> int:
    """Row stride to pass for a block table, zero when the table is shared.

    A single-row table means "every request uses this mapping". Expressing that as a zero
    row stride rather than a branch inside the kernel keeps the two cases on one code path,
    and -- more importantly -- makes the shared case safe: the alternative, indexing a
    one-row table by a nonzero request id, reads past the tensor and returns plausible
    garbage instead of failing.
    """
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must be [num_reqs, pages], got {tuple(block_table.shape)}"
        )
    return 0 if block_table.shape[0] == 1 else block_table.stride(0)


def _detect_rounded_intrinsics() -> bool:
    """True when the active Triton backend exposes CUDA-style ``*_rn`` libdevice ops.

    CUDA exposes the explicitly rounded scalar intrinsics; HIP/DCU does not.  On HIP
    the plain fp32 operators are already round-to-nearest-even, so fusion is disabled
    at launch instead (see ``_NO_FP_FUSION``).
    """
    try:
        import triton.language as tl

        try:
            import triton.language.extra.libdevice as _libdevice  # noqa: F401
        except ImportError:  # pragma: no cover
            _libdevice = tl.libdevice  # noqa: F841
        from triton.runtime.driver import driver

        backend = driver.active.get_current_target().backend
        module = importlib.import_module(f"triton.language.extra.{backend}.libdevice")
    except Exception:  # pragma: no cover - no driver, no device, or Triton missing.
        return False
    return all(
        hasattr(module, name)
        for name in ("add_rn", "sub_rn", "mul_rn", "div_rn", "fma_rn")
    )


_HAS_ROUNDED_INTRINSICS = _detect_rounded_intrinsics()
# Passing ``enable_fp_fusion=False`` keeps ``a * b`` and ``+ c`` as two separately
# rounded operations. CSA compression needs this on backends without ``*_rn``.
_NO_FP_FUSION = {} if _HAS_ROUNDED_INTRINSICS else {"enable_fp_fusion": False}
