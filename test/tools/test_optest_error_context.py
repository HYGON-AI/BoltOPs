# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CI diagnostics must survive successful benchmarks after a failure."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/optest_to_junit.py"
spec = importlib.util.spec_from_file_location("optest_to_junit", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_long_log_keeps_traceback_before_successful_benchmarks(tmp_path):
    log = tmp_path / "bench.log"
    log.write_text("\n".join([
        "[1/2] backend=triton file=test/failing.py",
        "================ FAILURES ================",
        "________________ test_bench_shape ________________",
        "    assert actual == expected",
        "\x1b[31mE   AssertionError: bad output\x1b[0m",
        "test/failing.py:42: AssertionError",
        "================ short test summary info ================",
        "FAILED test/failing.py::test_bench_shape - AssertionError",
        "[summary] backend=triton file=test/failing.py: pytest rc=1",
        "[2/2] backend=triton file=test/passing.py",
        *([".[bench] passing: 0.003 ms"] * 3000),
        "80 passed in 42.73s",
        "1 benchmark failure(s)",
    ]))

    context = report._read_error_context(tmp_path, log.name)
    assert "test/failing.py:42: AssertionError" in context
    assert "E   AssertionError: bad output" in context
    assert "pytest rc=1" in context
    assert "80 passed in 42.73s" in context
    assert "\x1b" not in context
    assert len(context.splitlines()) < 1000

    data = {"per": {"tests": [{
        "name": "run_operator_benchmarks", "status": "FAIL", "log": log.name,
    }]}}
    suite = ElementTree.fromstring(report.build_junit(data, tmp_path))
    assert suite.attrib["failures"] == "1"
    assert "test/failing.py:42: AssertionError" in suite.find("testcase/failure").text
    assert report._failure_count(data) == 1

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--error-context", str(log)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "test/failing.py:42: AssertionError" in result.stdout


def test_crashed_process_keeps_file_and_last_output(tmp_path):
    log = tmp_path / "bench.log"
    log.write_text("\n".join([
        "[1/2] backend=tilelang file=test/crashing.py",
        "compiling kernel_that_crashed",
        ".[summary] backend=tilelang file=test/crashing.py: pytest rc=-9",
        "[2/2] backend=triton file=test/passing.py",
        *(["[bench] passed"] * 3000),
    ]))
    context = report._read_error_context(tmp_path, log.name)
    assert "compiling kernel_that_crashed" in context
    assert "file=test/crashing.py: pytest rc=-9" in context


def test_large_fatal_trace_is_bounded_and_keeps_location(tmp_path):
    log = tmp_path / "bench.log"
    log.write_text("\n".join([
        "Fatal Python error: Aborted",
        '  File "test/crashing.py", line 91 in test_bench',
        *(["[bench] later success"] * 4000),
    ]))
    context = report._read_error_context(tmp_path, log.name)
    assert 'File "test/crashing.py", line 91' in context
    assert len(context.splitlines()) < 1000
