#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
HIP_VISIBLE_DEVICES=$(CIsmi need 1 --timeout "${CISMI_TIMEOUT}")
test -n "${HIP_VISIBLE_DEVICES}"
export HIP_VISIBLE_DEVICES
echo "Allocated HCU: ${HIP_VISIBLE_DEVICES}"

wheel_test_dir=$(mktemp -d)
python -m pip install --no-deps --target "${wheel_test_dir}" "${WHEEL_OUTPUT}"/*.whl
cp -r test "${wheel_test_dir}/test"
cp pyproject.toml "${wheel_test_dir}/pyproject.toml"
(
  cd "${wheel_test_dir}"
  python -m pytest test -s --tb=short -m "not slow" \
    --junitxml="${WHEEL_OUTPUT}/boltops_test.xml"
)
