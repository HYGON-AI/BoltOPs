#!/usr/bin/env bash
# Build a pure-Python wheel: bolt_ops-*-py3-none-any.whl
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q build
python -m build --wheel
ls -lh dist/
