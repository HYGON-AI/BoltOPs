---
name: build-hcu-wheel
description: Build or diagnose the BoltOPs HCU wheel using the repository's shared DTK/Torch/AICC build entry point, locally or in CI.
---

Use the shared `.github/scripts/_build_hcu_wheel.sh` for wheel builds. Both CI validation and publication use this entry point. `build_wheel.sh` is a compatibility wrapper.

The caller supplies a Linux environment with Python >=3.9, DTK, matching HIP Torch, AICC, and Python `build`. The script validates these before packaging; it does not install the toolchain and does not require a GPU for packaging. Triton/TileLang and GPU access are additionally needed for the test suites.

```bash
# From the repository root, in a prepared DTK environment:
python -m pip install build
bash .github/scripts/_build_hcu_wheel.sh --check-only
bash .github/scripts/_build_hcu_wheel.sh --out-dir dist
```

Optional environment variables:
- `DTK_ROOT`: DTK directory containing `env.sh`; default `/opt/dtk`.
- `PYTHON`: Python executable; default `python`.
- `BOLT_OPS_AICC`: explicit aicc executable, shared with the MoE JIT. Without it, use `aicc` on PATH or `$DTK_ROOT/bin/aicc`.

Relative output paths are resolved from the repository root. The output is one newly built pure-Python wheel; CI subsequently runs `CIUpload REPAIR` for DTK/Torch/commit tagging. Do not run auditwheel on this pure-Python wheel.

For missing DTK/Torch/AICC, report the failing precondition; do not silently skip checks or substitute CPU Torch. The provided AICC installer is `http://42.228.13.241:18000/ai_cc/Nightly/hcu_llvm_installer.sh`; CI selects major 1.0.0 for DTK26.04 and 2.0.0 for DTK26.10. Install tools only within the user's requested setup scope.

GitHub CI setup is in `.github/scripts/setup-env.sh`; build and CIUpload tagging are in the Build and repair wheel step of `.github/workflows/_build-test-wheel.yml`. PRs run correctness/performance tests; main pushes build and publish without rerunning tests. Building a wheel does not authorize uploading or publishing it.
