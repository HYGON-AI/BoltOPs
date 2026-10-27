# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU-only regression tests for owned HCU diagnostics."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from boltops.utility.logging import get_logger
from boltops.utility.runtime_text import format_hcu_message
from boltops.utility import triton_capability


ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def load_compiler():
    torch = ModuleType("torch")
    torch._C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)
    extensions = ModuleType("torch.utils.cpp_extension")
    extensions.get_default_build_root = MagicMock()
    extensions.include_paths = lambda: []
    extensions.library_paths = lambda: []
    baton = ModuleType("torch.utils.file_baton")
    baton.FileBaton = MagicMock()
    modules = {
        "torch": torch,
        "torch.utils": ModuleType("torch.utils"),
        "torch.utils.cpp_extension": extensions,
        "torch.utils.file_baton": baton,
    }
    path = ROOT / "boltops/fused_moe/triton/moe_sorting_hipc/compiler.py"
    spec = importlib.util.spec_from_file_location("hipc_compiler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class RuntimeMessagesTest(unittest.TestCase):
    def assert_platform_neutral(self, text):
        self.assertNotRegex(text.lower(), r"amd|xgmi")

    def test_display_does_not_mutate_configuration(self):
        config = {"amdgpu_enable_max_ilp_scheduling_strategy": True}
        shown = format_hcu_message(config)
        self.assert_platform_neutral(shown)
        self.assertIn("amdgpu_enable_max_ilp_scheduling_strategy", config)
        self.assertEqual(
            format_hcu_message("AMD XGMI amd xgmi"),
            "[backend] [interconnect] [backend] [interconnect]",
        )

    def test_generic_logger_preserves_diagnostic_information(self):
        logger = get_logger("boltops.runtime_message_test")
        stream = io.StringIO()
        try:
            with patch.object(logger.handlers[0], "stream", stream):
                try:
                    raise RuntimeError("AMD XGMI failure")
                except RuntimeError:
                    logger.exception("backend=%s", "amd")
            self.assertIn("AMD XGMI failure", stream.getvalue())
            self.assertIn("backend=amd", stream.getvalue())
            self.assertIn("failure", stream.getvalue())
        finally:
            logger.handlers.clear()

    def test_backend_keys_and_preference_are_unchanged(self):
        hcu, amd = object(), object()
        self.assertEqual(triton_capability._select_backend({"hcu": hcu, "amd": amd}), ("hcu", hcu))
        self.assertEqual(triton_capability._select_backend({"amd": amd}), ("amd", amd))
        with self.assertRaises(RuntimeError) as error:
            triton_capability._select_backend({"other": object()})
        self.assert_platform_neutral(str(error.exception))

    def test_backend_errors_do_not_expose_compatibility_key(self):
        compiler = type("Compiler", (), {})
        backends = SimpleNamespace(backends={"amd": SimpleNamespace(compiler=compiler)})
        with patch.object(triton_capability.importlib, "import_module", return_value=backends):
            with patch.object(triton_capability.inspect, "getmodule", return_value=SimpleNamespace()):
                with self.assertRaises(RuntimeError) as error:
                    triton_capability._get_compiler_options()
                self.assert_platform_neutral(str(error.exception))
            options = SimpleNamespace(__dataclass_fields__={})
            with patch.object(triton_capability.inspect, "getmodule", return_value=SimpleNamespace(HIPOptions=options)):
                with self.assertRaises(RuntimeError) as error:
                    triton_capability._get_compiler_options()
                self.assert_platform_neutral(str(error.exception))

    def test_actual_compiler_and_linker_identifiers_are_retained(self):
        module = load_compiler()
        flags = module._compile_flags("test", [])
        self.assertIn("-D__HIP_PLATFORM_AMD__=1", flags)
        self.assertIn("--amdgpu-kernarg-preload-count=16", flags)
        self.assertIn("-lamdhip64", module._link_flags())
        self.assertIn("-DCUDA_HAS_FP16=1", flags)

    def test_compiler_success_and_failure_cover_verbose_modes(self):
        for verbose in ("0", "1"):
            for failure in (False, True):
                with self.subTest(verbose=verbose, failure=failure), tempfile.TemporaryDirectory() as directory:
                    module = load_compiler()
                    root = Path(directory)
                    (root / "kernel.cu").write_text("// kernel\n")
                    module._aicc = lambda: root / "aicc"
                    module._digest = lambda *args: "digest"
                    module._load = MagicMock(return_value="loaded")
                    module.get_default_build_root = lambda: directory
                    lock = module.FileBaton.return_value
                    lock.try_acquire.return_value = True
                    output = "AMD compiler: -amdgpu-option; XGMI diagnostic\n"
                    process = MagicMock()
                    process.__enter__.return_value = process
                    process.stdout = io.StringIO(output)
                    process.wait.return_value = 1 if failure else 0
                    runner = MagicMock(return_value=process)
                    stream = io.StringIO()

                    def wait_after_output():
                        if verbose == "1":
                            self.assertIn("diagnostic", stream.getvalue())
                        return 1 if failure else 0

                    process.wait.side_effect = wait_after_output
                    with patch.object(module.subprocess, "Popen", runner), patch.dict(os.environ, {"BOLT_OPS_HIPC_VERBOSE": verbose}), redirect_stdout(stream):
                        if failure:
                            with self.assertRaises(RuntimeError) as error:
                                module.load_hipc_extension("test", ["kernel.cu"], csrc_dir=root)
                            self.assertTrue(error.exception.__suppress_context__)
                            self.assertIn("exit 1", str(error.exception))
                            self.assert_platform_neutral("".join(traceback.format_exception(error.exception)))
                        else:
                            self.assertEqual(module.load_hipc_extension("test", ["kernel.cu"], csrc_dir=root), "loaded")
                    self.assert_platform_neutral(stream.getvalue())
                    if verbose == "1" and not failure:
                        self.assertIn("diagnostic", stream.getvalue())
                    self.assertEqual(runner.call_args.kwargs["stdout"], subprocess.PIPE)
                    lock.release.assert_called_once()

    def test_redacted_diagnostics_do_not_invent_executable_paths(self):
        path = "/opt/amd/bin/compiler"
        shown = format_hcu_message(path)
        self.assertEqual(shown, "/opt/[backend]/bin/compiler")
        self.assertNotIn("/opt/hcu/", shown.lower())

    def test_runtime_loader_keeps_library_name_but_hides_exception_chain(self):
        import ctypes
        import ctypes.util

        namespace = {"ctypes": ctypes}
        guard = load_function("boltops/tools/fused_moe_triton_tune/performance_csv.py", "DeviceGuard", namespace)
        with patch.object(ctypes.util, "find_library", return_value=None), patch.object(ctypes, "CDLL", side_effect=OSError("libamdhip64.so not found")) as loader:
            with self.assertRaises(OSError) as error:
                guard(0)
            loader.assert_called_once_with("libamdhip64.so")
            self.assertTrue(error.exception.__suppress_context__)
            self.assert_platform_neutral("".join(traceback.format_exception(error.exception)))

    def test_batched_tuning_logs_keep_real_config_for_execution(self):
        config = {"amdgpu_enable_max_ilp_scheduling_strategy": True}
        kernel = MagicMock()
        namespace = {
            "torch": MagicMock(), "math": __import__("math"),
            "candidates": lambda *args: [config], "bench": lambda fn, *args: (fn(), 0.1)[1],
            "batched_gemm_bf16": kernel, "format_hcu_message": format_hcu_message,
        }
        tune = load_function("boltops/tools/tune_batched_gemm_bf16.py", "tune_shape", namespace)
        stream = io.StringIO()
        with redirect_stdout(stream):
            best, _ = tune(1, 16, 16, 16, 1, 1)
        self.assertIs(best, config)
        self.assertIs(kernel.call_args.kwargs["config"], config)
        self.assert_platform_neutral(stream.getvalue())
        kernel.side_effect = RuntimeError("AMD compiler failure")
        with redirect_stdout(stream), self.assertRaises(RuntimeError):
            tune(1, 16, 16, 16, 1, 1)
        self.assert_platform_neutral(stream.getvalue())

    def test_flash_tuning_logs_keep_real_config_for_execution(self):
        config = {
            "amdgpu_enable_max_ilp_scheduling_strategy": True,
            "USE_MATRIX_LOAD_Q": False,
        }
        kernel = MagicMock(return_value=(MagicMock(),))
        namespace = {
            "torch": MagicMock(), "DTYPE": "bf16", "math": __import__("math"),
            "triton": SimpleNamespace(testing=SimpleNamespace(do_bench=lambda *args, **kwargs: 0.1)),
            "flash_mla_sparse_fwd": kernel, "_close": lambda *args: None,
            "_supports_matrix_load": lambda *args: False,
            "_structural_candidates": lambda *args: [config],
            "_compiler_candidates": lambda *args: [config],
            "_optimization_candidates": lambda *args: [config],
            "format_hcu_message": format_hcu_message,
        }
        tune = load_function("boltops/tools/tune_flash_mla_sparse.py", "tune_shape", namespace)
        stream = io.StringIO()
        with redirect_stdout(stream):
            best, *_ = tune((1, 16, 1, 1, 4, 16, 16), 1, 1, True)
        self.assertIn("amdgpu_enable_max_ilp_scheduling_strategy", best)
        self.assertIn("amdgpu_enable_max_ilp_scheduling_strategy", kernel.call_args.kwargs["config"])
        self.assert_platform_neutral(stream.getvalue())

    def test_tuning_cli_help_is_platform_neutral(self):
        import argparse

        for name in ("tune_batched_gemm_bf16.py", "tune_flash_mla_sparse.py"):
            namespace = {"argparse": argparse, "Path": Path, "AITer_M_VALUES": (1,), "__doc__": "Tune HCU operators"}
            main = load_function("boltops/tools/" + name, "main", namespace)
            stream = io.StringIO()
            with patch.object(sys, "argv", [name, "--help"]), redirect_stdout(stream):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 0)
            self.assert_platform_neutral(stream.getvalue())


if __name__ == "__main__":
    unittest.main()
