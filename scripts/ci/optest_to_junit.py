#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Convert optest .optest-results.json to JUnit XML for GitLab CI.

optest output structure:
  /op_test_output/boltops/<branch>_<8char_commit>_<timestamp>/
    ├── .optest-results.json
    ├── acc/
    └── *.csv

Logs mirror the same version directory under /op_test_log/boltops/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_FATAL_RE = re.compile(r"Fatal Python error:")
_SUMMARY_RE = re.compile(r"^=* short test summary info =*$")
_TRACEBACK = "Traceback (most recent call last):"
_PYTEST_HEADER_RE = re.compile(r"=+ .+ =+$")
_PYTEST_SECTION_RE = re.compile(r"=+ (?:FAILURES|ERRORS) =+$")
_PYTEST_SUBSECTION_RE = re.compile(r"_+ .+ _+$")
_RUNNER_LINE_RE = re.compile(r"\[(?:\d+/\d+\] backend=|summary\] )")
_RUNNER_HEADER_RE = re.compile(r"\[\d+/\d+\] backend=")
_RUNNER_VERDICT_RE = re.compile(r"\[summary\] \S+ file=(\S+): (?:pytest rc=(-?\d+)|launch FAILED)")
_MAX_FULL_LINES = 2000
_MAX_CONTEXT = 990
_TAIL_LINES = 5
_FATAL_CONTEXT = 100
_FATAL_WINDOW = 400
_FAILURE_WINDOW = 400
_FAILURES_BUDGET = 500
_SUBSECTION_TAIL = 40
_SUMMARY_WINDOW = 200
_FILE_WINDOW = 150

RESULTS_ROOT = Path("/op_test_output/boltops")
LOG_ROOT = Path("/op_test_log/boltops")
_RUNNER_TAG = sys.argv[2] if len(sys.argv) >= 3 else "unknown"
JUNIT_OUTPUT = (
    Path(sys.argv[1]) / f"scripts/{_RUNNER_TAG}_boltops_output.xml"
    if len(sys.argv) > 1
    else Path(f"scripts/{_RUNNER_TAG}_boltops_output.xml")
)



def find_latest_results() -> Path | None:
    """Return the newest .optest-results.json under RESULTS_ROOT."""
    candidates = sorted(RESULTS_ROOT.glob("*/.optest-results.json"))
    return candidates[-1] if candidates else None


def make_error_suite(message: str) -> str:
    testsuite = Element(
        "testsuite", name=f"{_RUNNER_TAG}.boltops.optest",
        tests="1", failures="1", errors="0", skipped="0",
    )
    tc = SubElement(testsuite, "testcase",
                    classname=f"{_RUNNER_TAG}.boltops.optest", name="no_results")
    SubElement(tc, "failure", message=message)
    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def _is_boundary(line: str) -> bool:
    """A pytest ``==== X ====``, a runner ``[i/n]`` or ``[summary]`` line."""
    return bool(_PYTEST_HEADER_RE.match(line) or _RUNNER_LINE_RE.search(line))


def _bounded(lines: list[str], start: int, cap: int, stop=None) -> list[str]:
    """lines[start:start+cap], cut short at the next section boundary."""
    end = min(len(lines), start + cap)
    for i in range(start + 1, end):
        if (_is_boundary if stop is None else stop)(lines[i]):
            return lines[start:i]
    return lines[start:end]


def _failure_sections(lines: list[str]) -> list[list[str]]:
    """Per-test tails of every FAILURES/ERRORS region: E lines and the
    ``file:line`` locations that the short summary leaves out."""
    sections: list[list[str]] = []
    budget = _FAILURES_BUDGET
    for i, line in enumerate(lines):
        if not _PYTEST_SECTION_RE.match(line):
            continue
        region = _bounded(lines, i, _FAILURE_WINDOW)
        heads = [j for j in range(i, i + len(region))
                 if _PYTEST_SUBSECTION_RE.match(lines[j])]
        if heads:
            bounds = heads + [i + len(region)]
            segment: list[str] = []
            for a, b in zip(bounds, bounds[1:]):
                segment.extend(lines[max(a, b - _SUBSECTION_TAIL):b])
        else:
            segment = region
        if len(segment) > budget:
            segment = segment[:budget]
        sections.append(segment)
        budget -= len(segment)
        if budget <= 0:
            break
    return sections


def _read_error_context(log_dir: Path, log_rel: str) -> str:
    """Extract where a run failed from a log, not just that it failed.

    Prioritize failed files/cases, compiler diagnostics and exception details
    even in short logs. Success and skip summaries cannot displace a failure.
    Output is bounded to _MAX_CONTEXT lines, with room for the final tally.
    Logs without recognized failures are returned whole when short, otherwise
    their tail is used. ANSI escapes are stripped for the JUnit viewer.
    """
    log_file = log_dir / log_rel
    if not log_file.is_file():
        return f"(log file not found: {log_file})"
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(failed to read log: {exc})"

    lines = [_ANSI_RE.sub("", line) for line in content.splitlines()]
    verdicts = [(i, match) for i, line in enumerate(lines)
                if (match := _RUNNER_VERDICT_RE.search(line))
                and match[2] != "0"]
    failure_sections = _failure_sections(lines)
    fatal = next((i for i, line in enumerate(lines)
                  if _FATAL_RE.match(line)), None)
    traceback_at = next((i for i, line in enumerate(lines)
                         if line.rstrip() == _TRACEBACK), None)
    if not verdicts and not failure_sections and fatal is None and traceback_at is None:
        if len(lines) <= _MAX_FULL_LINES:
            return "\n".join(lines)
        return "\n".join(lines[-_MAX_CONTEXT:])

    # Put failed files and case IDs first. Successful/skip summaries must not
    # consume the budget before a failure later in a long performance run.
    segments: list[list[str]] = []
    if verdicts:
        segments.append([lines[i] for i, _ in verdicts])
    for i, line in enumerate(lines):
        if _SUMMARY_RE.match(line):
            summary = _bounded(
                lines, i, _SUMMARY_WINDOW,
                stop=lambda l: bool(_RUNNER_HEADER_RE.search(l)))
            if any(re.match(r"(?:FAILED|ERROR) ", item) for item in summary):
                segments.append(summary)

    # Compiler assertions precede the pytest traceback and may be followed by
    # hundreds of MLIR lines. Keep the cause without dumping the IR module.
    diagnostics = [line for line in lines if re.search(
        r"Assertion .+ failed\.|LLVM ERROR:|(?:^|\s)(?:fatal )?error:|note: Pipeline failed", line)]
    if diagnostics:
        segments.append(diagnostics[:40])
    if fatal is not None:
        segments.append(lines[max(0, fatal - _FATAL_CONTEXT):
                              fatal + _FATAL_WINDOW])
    segments.extend(failure_sections)
    if fatal is None and not failure_sections and traceback_at is not None:
        segments.append(lines[max(0, traceback_at - _FATAL_CONTEXT):
                              traceback_at + _FAILURE_WINDOW])
    for i, _ in verdicts:
        start = max((j for j in range(i, -1, -1)
                     if _RUNNER_HEADER_RE.search(lines[j])), default=max(0, i - 1))
        segment = lines[start:i + 1]
        if any(_PYTEST_SECTION_RE.match(item) for item in segment):
            continue
        if len(segment) > _FILE_WINDOW:
            segment = [segment[0]] + segment[-(_FILE_WINDOW - 1):]
        segments.append(segment)

    # Always reserve room for the final tally, including when many files fail.
    tail = lines[-_TAIL_LINES:]
    parts: list[str] = []
    budget = _MAX_CONTEXT - len(tail) - 1
    for segment in segments:
        room = budget - bool(parts)
        if room <= 0:
            break
        take = segment[:room]
        if not take:
            continue
        if parts:
            parts.append("[...]")
            budget -= 1
        parts.extend(take)
        budget -= len(take)
    if parts:
        parts.append("[...]")
    parts.extend(tail)
    return "\n".join(parts)


def _failure_count(data: dict) -> int:
    """Count failures even if an optest summary is missing or incomplete."""
    per_data = data.get("per", {})
    per_detail_failed = sum(
        1 for test in per_data.get("tests", []) if test.get("status") == "FAIL"
    )
    per_failed = max(per_detail_failed, per_data.get("summary", {}).get("failed", 0))

    acc_data = data.get("acc", {})
    acc_detail_failed = sum(row.get("failed", 0) for row in acc_data.get("rows", []))
    acc_failed = max(acc_detail_failed, acc_data.get("summary", {}).get("failed", 0))
    return per_failed + acc_failed


def build_junit(data: dict, log_dir: Path) -> str:
    testsuite = Element(
        "testsuite", name=f"{_RUNNER_TAG}.boltops.optest",
        tests="0", failures="0", errors="0", skipped="0",
    )
    total = 0
    failed = 0
    skipped = 0

    # ── per (performance) tests ──────────────────────────────────────
    per_data = data.get("per", {})
    for test in per_data.get("tests", []):
        total += 1
        name = test["name"]
        duration = test.get("duration_s")
        attrs = {"classname": f"{_RUNNER_TAG}.boltops.per", "name": name}
        if duration is not None:
            attrs["time"] = str(duration)
        tc = SubElement(testsuite, "testcase", **attrs)
        if test.get("status") == "FAIL":
            failed += 1
            log_rel = test.get("log", "")
            failure = SubElement(
                tc, "failure",
                message=f"{name} failed (log: {log_rel})",
            )
            failure.text = _read_error_context(log_dir, log_rel)

    per_summary = per_data.get("summary", {})
    per_failed = per_summary.get("failed", 0)
    if per_failed > failed:
        failed = per_failed

    # ── acc (accuracy / pytest) tests ─────────────────────────────────
    acc_data = data.get("acc", {})
    for row in acc_data.get("rows", []):
        total += 1
        name = row["name"]
        row_passed = row.get("passed", 0)
        row_failed = row.get("failed", 0)
        row_skipped = row.get("skipped", 0)

        tc = SubElement(
            testsuite, "testcase",
            classname=f"{_RUNNER_TAG}.boltops.acc",
            name=f"{name} (p:{row_passed} f:{row_failed} s:{row_skipped})",
        )
        if row_failed > 0:
            failed += 1
            failure = SubElement(
                tc, "failure",
                message=f"{name}: {row_failed}/{row.get('total', 0)} failed",
            )
            failure.text = _read_error_context(log_dir, f"acc/{name}.log")
        if row_skipped > 0:
            skipped += 1
            SubElement(tc, "skipped",
                       message=f"{name}: {row_skipped}/{row.get('total', 0)} skipped")

    testsuite.set("tests", str(total))
    testsuite.set("failures", str(failed))
    testsuite.set("skipped", str(skipped))

    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--error-context":
        log_file = Path(sys.argv[2])
        print(_read_error_context(log_file.parent, log_file.name))
        return 0
    # CI selects the exact matrix job's report. Manual use keeps auto-discovery.
    results_file = Path(sys.argv[3]) if len(sys.argv) >= 4 else find_latest_results()
    if results_file is None:
        xml_str = make_error_suite("No optest results found under " + str(RESULTS_ROOT))
        JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
        print(f"No optest results found; wrote empty JUnit report to {JUNIT_OUTPUT}")
        return 1

    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        xml_str = make_error_suite(f"Failed to read optest results {results_file}: {exc}")
        JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
        print(f"Failed to read optest results {results_file}: {exc}")
        print(f"Wrote failure JUnit report to {JUNIT_OUTPUT}")
        return 1
    # A readable JSON file alone does not prove that any tests actually ran.
    if not data.get("per", {}).get("tests") and not data.get("acc", {}).get("rows"):
        xml_str = make_error_suite(f"No test records in optest results {results_file}")
        JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
        print(f"No test records in optest results {results_file}; wrote failure JUnit report")
        return 1

    branch = data.get("branch", "?")
    commit = data.get("commit", "?")[:8]
    version_dir = results_file.parent.name
    log_dir = LOG_ROOT / version_dir
    print(f"Converting optest results: {branch=} commit={commit} {version_dir=}")

    xml_str = build_junit(data, log_dir)

    JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
    print(f"JUnit report generated: {JUNIT_OUTPUT}")

    # Return non-zero if any failures so GitLab can fail the job.
    return 1 if _failure_count(data) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
