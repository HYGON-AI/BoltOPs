#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

boltops-sparse-gqa-tune-cli \
  --tp 8 \
  --case D8 \
  --fp8-cache
