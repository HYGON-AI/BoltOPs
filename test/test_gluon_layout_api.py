# SPDX-License-Identifier: MIT
"""CPU native-IR contracts for MQA and WASP register layouts.

Load the actual utility source without importing unrelated GPU operators.
Device numerics and Graph replay are exercised by their existing suites.
"""

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton._C.libtriton import gluon_ir, ir
from triton.experimental.gluon import language as gl

_SOURCE = Path(__file__).resolve().parents[1] / "boltops/utility/gluon_layout.py"
_SPEC = importlib.util.spec_from_file_location("boltops_gluon_layout_contract", _SOURCE)
_LAYOUTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAYOUTS)


def _native_type(layout, arch):
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    builder.options = SimpleNamespace(arch=arch)
    return str(builder.get_distributed_ty(
        builder.get_float_ty(), [64, 128], layout._to_ir(builder)))


@pytest.mark.parametrize("arch", ["gfx928", "gfx92a", "gfx936", "gfx938", "gfx946"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("k,tiles", [(16, [4, 2]), (32, [1, 1]), (32, [1, 2])])
def test_mqa_native_layout_preserves_target_modes(arch, transposed, k, tiles):
    parameters = inspect.signature(gl.amd.AMDMFMALayout).parameters
    explicit = _LAYOUTS.HAS_EXPLICIT_MMAC_LAYOUT and arch in ("gfx938", "gfx946")
    kwargs = dict(version=4 if arch == "gfx946" else 3,
                  instr_shape=[16, 16, k], warps_per_cta=[1, 4],
                  tiles_per_warp=tiles)
    layout = _LAYOUTS.make_mfma_layout(
        transposed=transposed, explicit_mmac=explicit, **kwargs)
    native = _native_type(layout, arch)
    mode = (2 if transposed else 4) if arch in ("gfx938", "gfx946") else 1
    assert f"mmacLayout = {mode}" in native
    assert "isTransposed = false" in native
    assert f"instrShape = [16, 16, {k}]" in native
    assert "warpsPerCTA = [1, 4]" in native
    assert layout.tiles_per_warp == tiles
    if "transposed" in parameters:
        original = gl.amd.AMDMFMALayout(transposed=transposed, **kwargs)
        assert native == _native_type(original, arch)


@pytest.mark.parametrize("warps,tiles", [([1, 4], [1, 2]), ([4, 1], [1, 2]), ([4, 1], [1, 1])])
def test_wasp_native_layout_preserves_lit_mode(warps, tiles):
    layout = _LAYOUTS.make_mfma_layout(
        version=3, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=warps, tiles_per_warp=tiles,
        explicit_mmac=_LAYOUTS.HAS_EXPLICIT_MMAC_LAYOUT)
    native = _native_type(layout, "gfx946")
    assert "mmacLayout = 2" in native
    assert "isTransposed = false" in native
    assert "instrShape = [16, 16, 16]" in native
    assert f"warpsPerCTA = {warps}" in native
    assert layout.tiles_per_warp == tiles
