# SPDX-License-Identifier: MIT
"""Preserve HCU register modes across vendor Gluon constructor APIs."""

import inspect

from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import constexpr_function

_PARAMETERS = inspect.signature(gl.amd.AMDMFMALayout).parameters
_HAS_TRANSPOSED = "transposed" in _PARAMETERS
HAS_EXPLICIT_MMAC_LAYOUT = "mmac_layout" in _PARAMETERS and hasattr(gl.amd, "MmacLayout")


@constexpr_function
def make_mfma_layout(
    version, instr_shape, transposed, warps_per_cta, tiles_per_warp,
    explicit_mmac=False,
):
    # The old transpose-aware API owns its target-specific native mapping.
    if _HAS_TRANSPOSED and not explicit_mmac:
        return gl.amd.AMDMFMALayout(
            version=version,
            instr_shape=instr_shape,
            transposed=transposed,
            warps_per_cta=warps_per_cta,
            tiles_per_warp=tiles_per_warp,
        )
    # gfx928/gfx92a/gfx936 reject LIT/LTS in explicit-only native lowering.
    if explicit_mmac:
        mode = gl.amd.MmacLayout.LIT if transposed else gl.amd.MmacLayout.LIT_LTS
    else:
        mode = gl.amd.MmacLayout.LEGACY
    return gl.amd.AMDMFMALayout(
        version=version,
        instr_shape=instr_shape,
        mmac_layout=mode,
        warps_per_cta=warps_per_cta,
        tiles_per_warp=tiles_per_warp,
    )
