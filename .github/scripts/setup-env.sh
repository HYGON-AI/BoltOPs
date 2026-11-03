#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python - <<'PY'
import os,sys,sysconfig
from pathlib import Path
assert f"{sys.version_info.major}.{sys.version_info.minor}" == os.environ['PYTHON_VERSION'], sys.version
assert Path(sysconfig.get_path('include'), 'Python.h').is_file()
PY
: "${AICC_INSTALLER_URL:=${RESOURCE_SERVER_URL:+${RESOURCE_SERVER_URL%/}/ai_cc/Nightly/hcu_llvm_installer.sh}}"
: "${AICC_INSTALLER_URL:?Set AICC_INSTALLER_URL or RESOURCE_SERVER_URL}"
curl -fsSL --retry 3 --retry-delay 2 \
  -o "${CI_WORK_DIR}/hcu_llvm_installer.sh" \
  "${AICC_INSTALLER_URL}"
bash "${CI_WORK_DIR}/hcu_llvm_installer.sh" --major "${AICC_MAJOR_VERSION}" --dtk-dir /opt/dtk
test -f /opt/dtk/hcu_llvm.info
cat /opt/dtk/hcu_llvm.info
printf 'torch==%s\nnumpy==%s\n' "${TORCH_VERSION}" "${NUMPY_VERSION}" > "${CI_WORK_DIR}/constraints.txt"
cat "${CI_WORK_DIR}/constraints.txt"

python -m pip install --force-reinstall --no-deps --constraint "${CI_WORK_DIR}/constraints.txt" --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install --constraint "${CI_WORK_DIR}/constraints.txt" "torch==${TORCH_VERSION}" pytest PyYAML "numpy==${NUMPY_VERSION}" pandas build wheel ciupload cismi ninja optest
python -m pip install --constraint "${CI_WORK_DIR}/constraints.txt" --index-url "${TRITON_INDEX_URL}" "triton==3.6.0"
python -m pip show torch triton
python -m pip install --constraint "${CI_WORK_DIR}/constraints.txt" "apache-tvm-ffi==0.1.11" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
python -m pip install --constraint "${CI_WORK_DIR}/constraints.txt" --index-url "${TRITON_INDEX_URL}" "tilelang==0.1.12" --extra-index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
python -m pip check
python -m pip show torch triton tilelang apache-tvm-ffi numpy
python - <<'PY'
import os
import sys
from importlib.metadata import version
import numpy
import torch
import triton

expected_torch = os.environ["TORCH_VERSION"]
actual_torch = torch.__version__.split("+", 1)[0]
assert actual_torch == expected_torch, (actual_torch, expected_torch)
# Vendor build tags are recorded in wheel metadata, not always in torch.__version__.
torch_package_version = version("torch")
assert torch_package_version.split("+", 1)[0] == expected_torch, (torch_package_version, expected_torch)
assert numpy.__version__ == os.environ["NUMPY_VERSION"], numpy.__version__
expected_dtk_tag = f"dtk{os.environ['DTK_VER']}"
assert expected_dtk_tag in torch_package_version, (torch_package_version, expected_dtk_tag)
triton_version = version("triton")
expected_torch_tag = {"2.10.0": "torch2100", "2.11.0": "torch2110"}[expected_torch]
assert triton_version.startswith("3.6.0+"), triton_version
assert expected_torch_tag in triton_version, (triton_version, expected_torch_tag)
assert expected_dtk_tag in triton_version, (triton_version, expected_dtk_tag)
print(f"Validated dependency set: torch={torch.__version__}, numpy={numpy.__version__}")
print(f"Python runtime: {sys.executable}; version={sys.version}; prefix={sys.prefix}; base_prefix={sys.base_prefix}")
print(f"Torch runtime: version={version('torch')}; path={torch.__file__}")
print(f"Triton runtime: version={version('triton')}; path={triton.__file__}")
PY
