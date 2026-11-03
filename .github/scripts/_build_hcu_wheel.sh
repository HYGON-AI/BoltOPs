#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Build entry point for local development, CI tests and nightly publication.
set -eo pipefail
usage() {
  cat <<'HELP'
Usage: bash .github/scripts/_build_hcu_wheel.sh [--out-dir DIR] [--check-only] [--help]
Requires Python >=3.9, importable HCU Torch, DTK and an executable aicc.
Environment: PYTHON (default python), DTK_ROOT (default /opt/dtk),
             BOLT_OPS_AICC (optional compiler path, also used by the JIT).
The environment must already be installed. This script does not install DTK/Torch/AICC.
HELP
}
out_dir=dist
check_only=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then echo "ERROR: --out-dir requires a directory" >&2; exit 2; fi
      out_dir="$2"; shift 2 ;;
    --check-only) check_only=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [ ! -f "${DTK_ROOT}/env.sh" ]; then
  echo "ERROR: DTK is missing: ${DTK_ROOT}/env.sh. Install/source DTK first." >&2
  exit 1
fi
# Vendor env.sh may refer to unset variables; load it before set -u.
source "${DTK_ROOT}/env.sh"
set -u
python_bin="${PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: Python not found: ${python_bin}. Set PYTHON to its executable." >&2
  exit 1
fi
"$python_bin" - <<'PY'
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

def fail(message):
    raise SystemExit(f"ERROR: {message}")

if sys.version_info < (3, 9):
    fail(f"Python >=3.9 is required; found {sys.version.split()[0]}")
try:
    import torch
except Exception as exc:
    fail(f"Torch cannot be imported in {sys.executable}: {exc}. Install a matching HCU Torch wheel.")
if not getattr(torch.version, 'hip', None):
    fail("Torch is not an HCU/HIP build. Install the Torch wheel matching your DTK.")
if importlib.util.find_spec('build') is None:
    fail(f"Python build is missing. Run: {sys.executable} -m pip install build")
configured = os.environ.get('BOLT_OPS_AICC')
candidates = [configured] if configured else [shutil.which('aicc'), str(Path(os.environ['DTK_ROOT']) / 'bin/aicc')]
compiler = next((shutil.which(c) for c in candidates if c and shutil.which(c)), None)
if not compiler:
    fail("AICC is missing. Install AICC, or set BOLT_OPS_AICC to the aicc executable.")
try:
    result = subprocess.run([compiler, '--version'], check=True, capture_output=True, text=True, timeout=30)
except (OSError, subprocess.SubprocessError) as exc:
    fail(f"AICC cannot run: {compiler}: {exc}")
print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print(f"DTK: {os.environ['DTK_ROOT']}")
print(f"Torch: {torch.__version__}; HIP: {torch.version.hip}")
print(f"AICC: {compiler}\n{result.stdout.strip()}")
print("Environment checks passed (GPU availability is checked by tests, not packaging).")
PY
if [ "$check_only" -eq 1 ]; then exit 0; fi
# Interpret relative output paths from the repository root, not the caller's cwd.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [ ! -f pyproject.toml ]; then
  echo "ERROR: Project configuration not found: ${repo_dir}/pyproject.toml" >&2
  exit 1
fi
# setuptools can retain deleted source files in build/lib between local builds.
# Keep the clean-build behavior of the original build_wheel.sh entry point.
rm -rf -- build
mkdir -p "$out_dir"
staging_dir=$(mktemp -d)
trap 'rm -rf -- "$staging_dir"' EXIT
"$python_bin" -m build --wheel --outdir "$staging_dir"
shopt -s nullglob
wheels=("$staging_dir"/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "ERROR: Expected one wheel, found ${#wheels[@]}" >&2
  exit 1
fi
cp -- "${wheels[0]}" "$out_dir/"
echo "Wheel output: ${out_dir}/$(basename "${wheels[0]}")"
