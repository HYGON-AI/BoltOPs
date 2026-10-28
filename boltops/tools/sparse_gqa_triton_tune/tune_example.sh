#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail

cd "$(dirname "$0")/../../../.."

boltops-sparse-gqa-tune-cli \
  --tp 8 \
  --case D8 \
  --fp8-cache
