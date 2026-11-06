#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
shopt -s nullglob
wheels=("${WHEEL_OUTPUT}"/*.whl)
test "${#wheels[@]}" -eq 1
python -m pip install --force-reinstall --no-deps "${wheels[0]}"
git checkout -B "${OPTEST_RUN_BRANCH}"
HIP_VISIBLE_DEVICES=$(CIsmi need 1 --timeout "${CISMI_TIMEOUT}")
test -n "${HIP_VISIBLE_DEVICES}"
export HIP_VISIBLE_DEVICES
echo "Allocated performance HCU: ${HIP_VISIBLE_DEVICES}"

touch "${CI_WORK_DIR}/optest-start"
# Performance only: the accuracy suite still has pre-existing
# failures (see test/mqa_logits, test/qsa) that would gate every
# MR, so acc is not run by this job. --gpus pins the run to the
# HCU reserved by CIsmi above.
optest_rc=0
optest --op_name boltops --test_mode per --gpus "${HIP_VISIBLE_DEVICES}" \
  --test_dir "${CI_PROJECT_DIR}" || optest_rc=$?

artifact_dir="${CI_PROJECT_DIR}/optest_artifacts/${RUNNER_TAG}/dtk${DTK_VER}/py${PYTHON_VERSION}/torch${TORCH_VERSION}"
mkdir -p "${artifact_dir}"
result_pattern="/op_test_output/boltops/${OPTEST_RUN_BRANCH}_*/.optest-results.json"
echo "Looking for current-job results: ${result_pattern}" | tee "${artifact_dir}/collection.log"
latest_results=$(find /op_test_output/boltops -mindepth 2 -maxdepth 2 \
  -path "${result_pattern}" \
  -newer "${CI_WORK_DIR}/optest-start" -printf '%T@ %h\n' 2>/dev/null | \
  sort -n | tail -n1 | cut -d' ' -f2-) || latest_results=""
results_file="${artifact_dir}/missing-optest-results.json"
if [ -n "${latest_results}" ]; then
  results_file="${latest_results}/.optest-results.json"
  echo "Selected results: ${results_file}" | tee -a "${artifact_dir}/collection.log"
else
  echo "ERROR: No current-job report matched ${result_pattern} after ${CI_WORK_DIR}/optest-start" \
    | tee -a "${artifact_dir}/collection.log"
fi

# Collect logs independently: a crashed benchmark may not write JSON.
mkdir -p "${artifact_dir}/logs"
if [ -d /op_test_log/boltops ]; then
  while IFS= read -r -d '' log_dir; do
    echo "Collecting logs: ${log_dir}" | tee -a "${artifact_dir}/collection.log"
    cp -a "${log_dir}" "${artifact_dir}/logs/"
  done < <(find /op_test_log/boltops -mindepth 1 -maxdepth 1 -type d \
    -name "${OPTEST_RUN_BRANCH}_*" -print0)
fi

report_rc=0
python scripts/ci/optest_to_junit.py "${CI_PROJECT_DIR}" "${RUNNER_TAG}_dtk${DTK_VER}_py${PYTHON_VERSION}_torch${TORCH_VERSION}" \
  "${results_file}" || report_rc=$?
echo "optest exit=${optest_rc}; report exit=${report_rc}" | tee -a "${artifact_dir}/collection.log"
if [ "${optest_rc}" -ne 0 ] || [ "${report_rc}" -ne 0 ]; then
  # Keep console output bounded; the complete logs remain in artifacts.
  log_count=0
  while IFS= read -r -d '' log_file; do
    echo "===== Last 120 lines: ${log_file} ====="
    tail -n 120 "${log_file}"
    log_count=$((log_count + 1))
    if [ "${log_count}" -ge 10 ]; then
      break
    fi
  done < <(find "${artifact_dir}/logs" -type f -name '*.log' -print0 | sort -z)
  if [ "${log_count}" -eq 0 ]; then
    echo "No benchmark logs found under /op_test_log/boltops/${OPTEST_RUN_BRANCH}_*" \
      | tee -a "${artifact_dir}/collection.log"
  fi
fi
if [ "${report_rc}" -ne 0 ]; then
  exit "${report_rc}"
fi
if [ "${optest_rc}" -ne 0 ]; then
  exit "${optest_rc}"
fi
