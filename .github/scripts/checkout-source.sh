#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"
git config --global --add safe.directory "${GITHUB_WORKSPACE}"
# PR builds and tests receive the event's PR head SHA explicitly.
# Publish uses the push event SHA.
test "$(git rev-parse HEAD)" = "${EXPECTED_SOURCE_SHA:-${GITHUB_SHA:?}}"
echo "Source commit: $(git rev-parse HEAD)"
git show -s --format='Parents: %P'
if [[ -n "${PR_BASE_SHA:-}" ]]; then
  git -c user.name='BoltOPs CI' -c user.email='ci@localhost' \
    merge --no-edit "${PR_BASE_SHA}"
  echo "Integrated PR base: ${PR_BASE_SHA}"
fi
