#!/usr/bin/env bash
# Build a pure-Python wheel: boltops-*-py3-none-any.whl
set -euo pipefail
cd "$(dirname "$0")"
rm -rf -- build
python -m pip install -q build
python -m build --wheel
ls -lh dist/
