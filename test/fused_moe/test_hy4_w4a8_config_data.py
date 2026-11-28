# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU-only regressions for the serialized HY4 channel-W4A8 tile data.

Run directly with unittest without importing Torch, Triton or a GPU backend.
Hardware metadata is a fixture, as in test_moe_capability_configs; JSON loading
and selection execute the real source functions, never a replacement lookup.
"""
import ast
import functools
import json
import logging
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SMALL = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
             GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False)
LARGE = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
             GROUP_SIZE_M=8, COMBINE_SCALE_LOAD=False)
DECODE = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
              GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False)
PREFILL = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
               GROUP_SIZE_M=8, COMBINE_SCALE_LOAD=False)
TUNED = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
             GROUP_SIZE_M=8, COMBINE_SCALE_LOAD=False)


def _lookup_namespace(capability="3.8", arch="gfx938"):
    """Load only the pure selector functions; imports would require GPU SDKs."""
    source = ROOT / "boltops/fused_moe/triton/moe_config_utils.py"
    capability_source = ROOT / "boltops/utility/triton_capability.py"
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    namespace = {"Path": Path}
    capability_tree = ast.parse(capability_source.read_bytes())
    constants = [node for node in capability_tree.body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name)
                     and target.id == "_CAPABILITY_ORDER"
                     for target in node.targets)]
    helpers = [node for node in capability_tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "get_compatible_triton_config_dirs"]
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *constants, *helpers], type_ignores=[])),
                 str(capability_source), "exec"),
        namespace)
    logger = logging.Logger("hy4-config-data-test")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    namespace.update(
        json=json, re=re, functools=functools, logger=logger,
        arch_info=SimpleNamespace(get_arch=lambda: arch),
        get_triton_capabilities=lambda: SimpleNamespace(
            version=capability, compiler_options=frozenset(), backend="hcu",
            package_version="3.7.0" if capability == "3.8" else capability),
        _TRITON_CONFIGS_DIR=source.parent / "configs",
    )
    functions = [node for node in ast.parse(source.read_bytes()).body
                 if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *functions], type_ignores=[])), str(source), "exec"), namespace)
    return namespace


class HY4W4A8ConfigDataTest(unittest.TestCase):
    def setUp(self):
        self.lookup = _lookup_namespace()

    def _select(self, m, bottom=False, hidden=1536, topk=16, namespace=None):
        shape = (512, hidden, 192) if bottom else (512, 768, hidden // 2)
        return (namespace or self.lookup)["try_get_optimal_moe_config"](
            w_shape=shape, top_k=topk, dtype="int4_w4a8", M=m,
            is_bottom=bottom,
        )

    def _assert_exact_shape(self, bottom):
        # Independent literals pin the two new prefill points relative to the
        # existing M384/M2048/M8192 tuning. Guards preserve intervening batches.
        changed = []
        for m in range(1, 32770):
            default = SMALL if m <= 512 else LARGE
            before = {384: DECODE, 2048: TUNED,
                      8192: PREFILL}.get(m, default)
            expected = {4096: TUNED, 32768: TUNED}.get(m, before)
            selected = self._select(m, bottom)
            self.assertEqual(selected, (expected, expected["BLOCK_SIZE_M"])
                             if bottom else expected, (bottom, m))
            original_selected = ((before, before["BLOCK_SIZE_M"])
                                 if bottom else before)
            if selected != original_selected:
                changed.append(m)
            # Adding a tuned file must not mutate the original fallback policy.
            original = self.lookup["get_default_config"](
                m, 512, 384, 1536 if bottom else 768, 16,
                "int4_w4a8", False, None, bottom,
            )
            self.assertEqual(original, default, (bottom, m, "default"))
        self.assertEqual(changed, [4096, 32768])

    def test_top_tunes_only_qualified_points_and_preserves_other_batches(self):
        self._assert_exact_shape(False)

    def test_bottom_tunes_qualified_points_and_preserves_padding_capacity(self):
        self._assert_exact_shape(True)

    def test_serialized_guards_bound_both_new_points(self):
        expected_keys = {1, 383, 384, 385, 512, 513, 2047, 2048, 2049,
                         4095, 4096, 4097, 8191, 8192, 8193,
                         32767, 32768, 32769}
        for bottom in (False, True):
            name = self.lookup["get_config_file_name"](
                512, 384, "int4_w4a8", None, bottom, K=1536, top_k=16,
            )
            path = ROOT / "boltops/fused_moe/triton/configs/triton_3_8" / name
            data = {int(k): v for k, v in json.loads(path.read_bytes()).items()}
            self.assertEqual(set(data), expected_keys)
            for m in (383, 385):
                self.assertEqual(data[m], SMALL)
            for m in (2047, 2049, 4095, 4097, 8191, 8193, 32767, 32769):
                self.assertEqual(data[m], LARGE)

    def test_large_batches_return_original_default(self):
        for bottom in (False, True):
            for m in (8193, 16384, 32767, 32769, 65536, 1_000_000, 2**31 - 1):
                self.assertEqual(self._select(m, bottom),
                                 (LARGE, 64) if bottom else LARGE, (bottom, m))

    def test_other_k_or_topk_does_not_select_this_exact_family(self):
        for bottom in (False, True):
            for hidden, topk in ((2048, 16), (1536, 8)):
                for m in (2048, 4096, 32768):
                    self.assertEqual(self._select(m, bottom, hidden, topk),
                                     (LARGE, 64) if bottom else LARGE,
                                     (bottom, hidden, topk, m))
        # This intentionally makes no isolation claim for E != 512: the
        # existing selector may fall back from another E to its nearest E.

    def test_older_capability_does_not_consume_this_3_8_data(self):
        older = _lookup_namespace("3.6")
        for bottom in (False, True):
            self.assertEqual(self._select(2048, bottom, namespace=older),
                             (LARGE, 64) if bottom else LARGE)

    def test_gfx936_standard_buckets_and_compatible_lookup(self):
        expected_sizes = [1, 2, 4, 8, 16, 24, 32, 64, 96, 128, 256, 384,
                          512, 1024, 2048, 4096, 8192, 16384, 32768]
        for capability in ("3.6", "3.8"):
            lookup = _lookup_namespace(capability, arch="gfx936")
            for bottom in (False, True):
                with self.subTest(capability=capability, bottom=bottom):
                    name = lookup["get_config_file_name"](
                        512, 192, "int4_w4a8", None, bottom, K=1536, top_k=16,
                    )
                    configs = lookup["_TRITON_CONFIGS_DIR"]
                    data = json.loads((configs / "triton_3_6" / name).read_bytes())
                    self.assertEqual(list(map(int, data)), expected_sizes)
                    self.assertFalse((configs / "triton_3_8" / name).exists())
                    w_shape = (512, 1536, 96) if bottom else (512, 384, 768)
                    for m, bucket in ((33, 32), (384, 384), (8192, 8192),
                                      (16384, 16384)):
                        selected = lookup["try_get_optimal_moe_config"](
                            w_shape, 16, "int4_w4a8", m, is_bottom=bottom,
                        )
                        expected = data[str(bucket)]
                        self.assertEqual(selected, (expected, expected["BLOCK_SIZE_M"])
                                         if bottom else expected)
                    self.assertEqual(data["32"]["BLOCK_SIZE_N"], 128)

    def test_missing_config_preserves_default_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = _lookup_namespace()
            namespace["_TRITON_CONFIGS_DIR"] = Path(directory)
            for bottom in (False, True):
                for m in (4096, 32768):
                    self.assertEqual(self._select(m, bottom, namespace=namespace),
                                     (LARGE, 64) if bottom else LARGE)

    def test_fuzzy_and_nearest_expert_fallbacks_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = _lookup_namespace()
            namespace["_TRITON_CONFIGS_DIR"] = Path(directory)
            config_dir = Path(directory) / "triton_3_8"
            config_dir.mkdir()
            name = ("E=32,N=384,KMin=1024,KMod=512,topkSet=8-16,"
                    "arch=gfx938,dtype=int4_w4a8.json")
            (config_dir / name).write_text(json.dumps({"1": TUNED}),
                                           encoding="utf-8")
            # E=31 still falls back to E=32. Fuzzy matching is an existing
            # selector contract, not an isolation claim for these exact files.
            for experts in (32, 31):
                self.assertEqual(namespace["get_moe_configs"](
                    experts, 384, "int4_w4a8", K=1536, top_k=16), {1: TUNED})


if __name__ == "__main__":
    unittest.main()
