#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"
: "${PYPI_URL:?}" "${PYPI_PASSWORD:?Missing PYPI_PASSWORD secret}"
export DEVPI_URL="${PYPI_URL%/}/"
export devpi_nightly_password="${PYPI_PASSWORD}"
export PIP_INDEX_URL="${PYPI_URL%/}/nightly/dtk2604/+simple/"
export PIP_TRUSTED_HOST
PIP_TRUSTED_HOST=$(python -c 'import os,urllib.parse; print(urllib.parse.urlparse(os.environ["PYPI_URL"]).hostname)')
python -m pip install ciupload
python - <<'PY'
from pathlib import Path
import shutil
import zipfile

def contents(path):
    with zipfile.ZipFile(path) as wheel:
        return {name: wheel.read(name) for name in wheel.namelist()}

import os

dtk_version = os.environ["DTK_VER"]
wheels = sorted(Path("output").glob(f"dtk{dtk_version}/py*/torch*/*.whl"))
assert wheels, f"No built wheels found for DTK {dtk_version}"
destination = Path("publish_wheels") / f"dtk{dtk_version}"
destination.mkdir(parents=True, exist_ok=True)
seen = {}
for wheel in wheels:
    if wheel.name in seen:
        assert contents(wheel) == contents(seen[wheel.name]), (
            f"Different contents for the same wheel filename: {wheel} and {seen[wheel.name]}"
        )
        continue
    seen[wheel.name] = wheel
    shutil.copy2(wheel, destination / wheel.name)
    print(f"Publish wheel: {wheel}")
PY

for wheel in "publish_wheels/dtk${DTK_VER}"/*.whl; do
  CIUpload NIGHTLY \
    --devpi_url "${DEVPI_URL}" \
    --password "${devpi_nightly_password}" \
    --dtk_pkg_name "${DTK_PKG_NAME}" \
    --record \
    -f "${wheel}"
done
