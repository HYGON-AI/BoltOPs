# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU contracts using the real Triton autotuner with synthetic kernel timings."""

import ast
from contextlib import redirect_stdout
from contextvars import ContextVar
import io
from pathlib import Path
import unittest

import torch
import triton


ROOT = Path(__file__).resolve().parents[2]


def load_functions(relative, names, namespace):
    path = ROOT / relative
    nodes = [node for node in ast.parse(path.read_text()).body
             if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class DynamicAutotuneApiTest(unittest.TestCase):
    def exercise(self, *, compile_only, count):
        initialized = torch.cuda.is_initialized()
        configs = [triton.Config({"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 64,
                                  "GROUP_SIZE_M": 1}, num_warps=4, num_stages=1)
                   for bn in (32, 64)[:count]]
        launches, measurements = [], []

        class Kernel:
            arg_names = ["N", "K", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]
            fn = staticmethod(lambda: None)
            __name__ = "synthetic_kernel"

            def run(self, *args, **kwargs):
                launches.append(kwargs.copy())
                return kwargs["BLOCK_SIZE_N"]

            def warmup(self, *args, **kwargs):
                return self.run(*args, **kwargs)

        def measure(launch, quantiles):
            launch()
            bn = launches[-1]["BLOCK_SIZE_N"]
            measurements.append(bn)
            # Deliberately prefer the second config; these are fixture data.
            return [0.3 if bn == 32 else 0.1] * 3

        namespace = dict(triton=triton, torch=torch, compile_only=compile_only,
                         apply_my_heuristics=lambda fn: fn,
                         generate_config2_lists=lambda **kwargs: configs,
                         _prune_configs_by_runtime_constraints=lambda cs, runtime: cs,
                         _check_hip_loads=lambda bench: measure,
                         _tuning_phase=ContextVar("phase", default=False))
        load_functions("boltops/tools/fused_moe_triton_tune/autotune_patches.py",
                       {"dynamic_autotune", "_ensure_single_config_timing", "_print_best_config"},
                       namespace)
        stream = io.StringIO()
        with redirect_stdout(stream):
            output = namespace["dynamic_autotune"](Kernel())[(1,)](
                N=128, K=256, BLOCK_SIZE_M=16, BLOCK_SIZE_N=999,
                BLOCK_SIZE_K=999, GROUP_SIZE_M=999)
        self.assertEqual(torch.cuda.is_initialized(), initialized)
        self.assertTrue(launches)
        self.assertTrue(all(call["BLOCK_SIZE_N"] in (32, 64) for call in launches))
        self.assertTrue(all(call["BLOCK_SIZE_K"] == 64 for call in launches))
        if compile_only:
            self.assertEqual(output, [32, 64][:count])
            self.assertFalse(measurements)
            self.assertNotIn("fused_moe_kernel:", stream.getvalue())
            return
        self.assertEqual(output, (32, 64)[count - 1])
        self.assertCountEqual(measurements, [32, 64][:count])
        parser = load_functions("boltops/tools/fused_moe_triton_tune/moe_log_parser.py",
                                {"_parse_config_value", "_parse_autotune_config", "_parse_log_cases"},
                                dict(ast=ast, re=__import__("re"),
                                     _SAVED_CONFIG_KEYS=("BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                                         "GROUP_SIZE_M", "num_warps", "num_stages")))
        # A complete top/bottom pair must be consumable without vendor debug logs.
        pair = "Config 1: {'BLOCK_SIZE_M': 16}\n" + stream.getvalue() * 2
        results = parser["_parse_log_cases"](pair)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][4], (0.6, 0.2)[count - 1])

    def test_single_candidate_execution_and_parser(self):
        self.exercise(compile_only=False, count=1)

    def test_multiple_candidates_execution_and_parser(self):
        self.exercise(compile_only=False, count=2)

    def test_single_candidate_compile_only(self):
        self.exercise(compile_only=True, count=1)

    def test_multiple_candidates_compile_only(self):
        self.exercise(compile_only=True, count=2)


if __name__ == "__main__":
    unittest.main()
