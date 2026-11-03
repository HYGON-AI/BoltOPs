# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from urllib.parse import quote

from packaging.tags import Tag, sys_tags


SCRIPT = Path(__file__).resolve().parents[1] / "select-hcu-wheel.py"
SPEC = importlib.util.spec_from_file_location("select_hcu_wheel", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)
INDEX = "http://hcu.example/simple/tilelang/"
TAG = Tag("cp39", "abi3", "linux_x86_64")


def link(version, attrs="", tag=TAG, host=None):
    filename = f"tilelang-{version}-{tag}.whl"
    href = (host or "../../packages/") + quote(filename) + "#sha256=fixture"
    return f'<a href="{href}" {attrs}>{filename}</a>'


class SelectHcuWheelTests(unittest.TestCase):
    def select(self, html):
        return SELECTOR.select_wheel(
            html, INDEX, "tilelang", "2610", "2.13.0", [TAG], "3.12.0"
        )

    def test_latest_matching_hcu_wheel_wins(self):
        html = "".join([
            link("0.1.15+dtk2610.torch2130.261101"),
            link("0.1.15+dtk2610.torch2130.261103"),
            link("0.1.16+dtk2610.torch2110.261104"),
            link("0.1.16+dtk2604.torch2130.261104"),
            link("0.2.0"),
        ])
        version, url = self.select(html)
        self.assertEqual(version, "0.1.15+dtk2610.torch2130.261103")
        self.assertTrue(url.startswith("http://hcu.example/packages/"))
        self.assertIn("%2Bdtk2610.torch2130", url)
        self.assertTrue(url.endswith("#sha256=fixture"))

    def test_missing_vendor_build_cannot_fall_back_to_public(self):
        with self.assertRaisesRegex(ValueError, "No compatible HCU tilelang.*torch2.13.0"):
            self.select(link("0.1.15+dtk2610.torch2110.261103") + link("0.1.15"))

    def test_rejects_yanked_wrong_python_foreign_and_prerelease(self):
        html = "".join([
            link("0.1.15+dtk2610.torch2130.1", 'data-requires-python="&gt;=3.9"'),
            link("0.1.16+dtk2610.torch2130.1", 'data-yanked="bad build"'),
            link("0.1.17+dtk2610.torch2130.1", 'data-requires-python="&gt;=3.13"'),
            link("0.1.18+dtk2610.torch2130.1", host="https://public.example/"),
            link("0.1.19rc1+dtk2610.torch2130.1"),
        ])
        self.assertEqual(self.select(html)[0], "0.1.15+dtk2610.torch2130.1")

    def test_rejects_incompatible_platform_and_partial_tags(self):
        for html in [
            link("0.1.15+dtk2610.torch2130.1", tag=Tag("cp39", "abi3", "win_amd64")),
            link("0.1.15+dtk26101.torch2130.1"),
            link("0.1.15+dtk2610.torch21301.1"),
        ]:
            with self.subTest(html=html), self.assertRaises(ValueError):
                self.select(html)

    def test_cli_selects_url_or_exits_without_requirement(self):
        html = link("0.1.15+dtk2610.torch2130.1", tag=next(sys_tags()))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            command = [sys.executable, str(SCRIPT), "--index-url",
                       f"http://127.0.0.1:{server.server_port}/simple/",
                       "--package", "tilelang", "--dtk-ver", "2610", "--torch-version"]
            result = subprocess.run(command + ["2.13.0"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith(f"http://127.0.0.1:{server.server_port}/packages/"))
            self.assertIn("Selected HCU tilelang", result.stderr)
            result = subprocess.run(command + ["2.11.0"], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("Public-package fallback is disabled", result.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
