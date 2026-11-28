# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU regressions for user-visible logs and unchanged execution inputs."""

import ast
from contextlib import redirect_stdout
import ctypes
import ctypes.util
from functools import partial
import io
from pathlib import Path
import statistics
import re
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class UserLogFieldsTest(unittest.TestCase):
    def neutral(self, text):
        self.assertNotRegex(text, r"(?i)amd|xgmi")

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
