#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail

cd "$(dirname "$0")/../../.."
export TRITON_HIP_CLANG_PATH="${TRITON_HIP_CLANG_PATH:-/root/dtk_llvm_ffe19a/aillvm/bin/clang}"
export AMDGPU_USE_BUFFER_OPS="${AMDGPU_USE_BUFFER_OPS:-1}"

python3 -m boltops.tools.extend_attention_triton_tune.tune_extend_attention \
  --kv-cache-dtype both "$@"
