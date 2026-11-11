#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Source this file in every step: exports do not persist between Actions steps.
set -eo pipefail
source /opt/dtk/env.sh
set -u
cd "${GITHUB_WORKSPACE:?}"
export CI_PROJECT_DIR="${GITHUB_WORKSPACE}"
export CI_JOB_ID="${GITHUB_RUN_ID:?}_${GITHUB_RUN_ATTEMPT:?}_${GITHUB_JOB:?}_${DTK_VER:?}_${PYTHON_VERSION:?}_${TORCH_VERSION:?}"
export CI_WORK_DIR="${RUNNER_TEMP:?}/boltops-${CI_JOB_ID}"
export WHEEL_OUTPUT="${CI_PROJECT_DIR}/output/dtk${DTK_VER}/py${PYTHON_VERSION}/torch${TORCH_VERSION}"
export RUNNER_TAG=BW1100
export OPTEST_RUN_BRANCH="ci_performance_${RUNNER_TAG}_${CI_JOB_ID}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 PIP_RESUME_RETRIES=10
: "${PYPI_URL:?Set the PYPI_URL repository variable}"
: "${TRITON_PYPI_URL:?Set the TRITON_PYPI_URL repository variable}"
export PIP_INDEX_URL="${PYPI_URL%/}/nightly/dtk2604/+simple/"
# pip requires a hostname, without scheme, port or path.
export PIP_TRUSTED_HOST
PIP_TRUSTED_HOST=$(python -c 'import os,urllib.parse; print(" ".join(sorted({urllib.parse.urlparse(os.environ[k]).hostname for k in ["PYPI_URL","TRITON_PYPI_URL"]})))')
export TORCH_INDEX_URL="${PYPI_URL%/}/nightly/dtk${DTK_VER}/+simple/"
export TRITON_INDEX_URL="${TRITON_PYPI_URL%/}/das_nightly/dtk2604-rc4/+simple/"
case "${DTK_VER}" in
  2604) export AICC_MAJOR_VERSION=1.0.0 ;;
  2610) export AICC_MAJOR_VERSION=2.0.0
        export TORCH_INDEX_URL="${TRITON_PYPI_URL%/}/das_nightly/dtk2610/+simple/"
        export TRITON_INDEX_URL="${TORCH_INDEX_URL}" ;;
  *) echo "Unsupported DTK ${DTK_VER}" >&2; exit 1 ;;
esac
mkdir -p "${CI_WORK_DIR}" "${WHEEL_OUTPUT}"
