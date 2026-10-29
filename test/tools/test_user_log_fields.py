# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU regressions for user-visible logs and unchanged execution inputs."""

import ast
from contextlib import redirect_stdout
import ctypes
import ctypes.util
from functools import partial
import importlib.util
import io
from pathlib import Path
import statistics
import re
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def compiler_module():
    torch = ModuleType("torch")
    torch._C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)
    extensions = ModuleType("torch.utils.cpp_extension")
    extensions.get_default_build_root = MagicMock()
    extensions.include_paths = lambda: []
    extensions.library_paths = lambda: []
    baton = ModuleType("torch.utils.file_baton")
    baton.FileBaton = MagicMock()
    modules = {"torch": torch, "torch.utils": ModuleType("torch.utils"),
               "torch.utils.cpp_extension": extensions, "torch.utils.file_baton": baton}
    path = ROOT / "boltops/fused_moe/triton/moe_sorting_hipc/compiler.py"
    spec = importlib.util.spec_from_file_location("log_test_compiler", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class UserLogFieldsTest(unittest.TestCase):
    def neutral(self, text):
        self.assertNotRegex(text, r"(?i)amd|xgmi")

    def test_compiler_preserves_diagnostics_and_execution_inputs(self):
        for verbose in ("0", "1"):
            for code in (0, 1, 2):
                with self.subTest(verbose=verbose, code=code), tempfile.TemporaryDirectory() as directory:
                    module = compiler_module()
                    root = Path(directory)
                    (root / "kernel.cu").write_text("// kernel\n", encoding="utf-8")
                    module._aicc = lambda: root / "aicc"
                    module._digest = lambda *args: "digest"
                    module.get_default_build_root = lambda: directory
                    module._load = MagicMock(return_value="loaded")
                    module.FileBaton.return_value.try_acquire.return_value = True
                    raw = "AMD --amdgpu-option -lamdhip64 XGMI\nsource.cu:7: useful diagnostic\n"
                    process = MagicMock()
                    process.__enter__.return_value = process
                    process.stdout = io.StringIO(raw)
                    process.wait.return_value = code
                    stream = io.StringIO()
                    with patch.object(module.subprocess, "Popen", return_value=process) as launch, \
                            patch.dict(module.os.environ, {"BOLT_OPS_HIPC_VERBOSE": verbose}), redirect_stdout(stream):
                        if code:
                            with self.assertRaises(RuntimeError) as caught:
                                module.load_hipc_extension("test", ["kernel.cu"], csrc_dir=root)
                            self.neutral("".join(traceback.format_exception(caught.exception)))
                            self.assertIn(f"exit {code}", str(caught.exception))
                        else:
                            self.assertEqual(module.load_hipc_extension("test", ["kernel.cu"], csrc_dir=root), "loaded")
                    self.neutral(stream.getvalue())
                    logs = list(root.rglob("*.log"))
                    self.assertEqual(len(logs), 1)
                    text = logs[0].read_text(encoding="utf-8")
                    self.neutral(text)
                    self.assertEqual(text, "[backend] --[backend]-option -l[backend]hip64 [backend]\nsource.cu:7: useful diagnostic\n")
                    ninja = (logs[0].parent / "build.ninja").read_text(encoding="utf-8")
                    self.assertIn("-D__HIP_PLATFORM_AMD__=1", ninja)
                    self.assertIn("-lamdhip64", ninja)
                    self.assertEqual(launch.call_args.args[0][0], "ninja")
                    module.FileBaton.return_value.release.assert_called_once()

    def test_cache_hit_does_not_launch_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            module = compiler_module()
            root = Path(directory)
            (root / "kernel.cu").write_text("// kernel\n", encoding="utf-8")
            module._aicc = lambda: root / "aicc"
            module._digest = lambda *args: "digest"
            module.get_default_build_root = lambda: directory
            target = root / "boltops_hipc/_boltops_test_digest/_boltops_test_digest.so"
            target.parent.mkdir(parents=True)
            target.touch()
            module._load = MagicMock(return_value="cached")
            with patch.object(module.subprocess, "Popen") as launch:
                self.assertEqual(module.load_hipc_extension("test", ["kernel.cu"], csrc_dir=root), "cached")
            launch.assert_not_called()

    def test_loader_errors_keep_real_library_and_readable_reason(self):
        for resolved in (None, "/opt/AMD/lib/libamdhip64.so"):
            with self.subTest(resolved=resolved), tempfile.TemporaryDirectory() as directory:
                namespace = {"ctypes": ctypes, "re": re, "tempfile": SimpleNamespace(
                    NamedTemporaryFile=partial(tempfile.NamedTemporaryFile, dir=directory))}
                functions("boltops/tools/fused_moe_triton_tune/performance_csv.py", {"_display_error", "DeviceGuard"}, namespace)
                with patch.object(ctypes.util, "find_library", return_value=resolved), \
                        patch.object(ctypes, "CDLL", side_effect=OSError("libamdhip64.so: cannot load runtime")) as load:
                    with self.assertRaises(OSError) as caught:
                        namespace["DeviceGuard"](0)
                self.assertEqual(load.call_args.args[0], resolved or "libamdhip64.so")
                self.neutral("".join(traceback.format_exception(caught.exception)))
                logs = list(Path(directory).glob("*.log"))
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0].read_text(encoding="utf-8"), "lib[backend]hip64.so: cannot load runtime")

    def test_csv_retry_preserves_error_detail_without_raw_exception_chain(self):
        path = "boltops/tools/fused_moe_triton_tune/performance_csv.py"
        namespace = {"re": re, "torch": MagicMock()}
        functions(path, {"_display_error"}, namespace)
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        measure = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "measure")
        guard = MagicMock()
        guard.wait_until_exclusive.side_effect = RuntimeError("AMD measurement failed at iteration 3")
        namespace.update(guard=guard, raw=MagicMock(), samples=[], statistics=statistics,
                         args=SimpleNamespace(max_attempts=2, retry_delay=0),
                         time=SimpleNamespace(sleep=MagicMock()))
        exec(compile(ast.Module(body=[measure], type_ignores=[]), path, "exec"), namespace)
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(RuntimeError) as caught:
            namespace["measure"]()
        self.neutral(stream.getvalue())
        self.neutral("".join(traceback.format_exception(caught.exception)))
        self.assertIn("measurement failed at iteration 3", str(caught.exception))
        self.assertEqual(guard.wait_until_exclusive.call_count, 2)


if __name__ == "__main__":
    unittest.main()
