# SPDX-License-Identifier: MIT
"""CPU registration and teardown contracts, without importing GPU operators."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("destroy_explicitly", [False, True])
def test_torch_guard_dispatch_compile_and_teardown(destroy_explicitly):
    guard_path = Path(__file__).resolve().parents[1] / "boltops/utility/torch_guard.py"
    script = """
import runpy
import sys
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

guard = runpy.run_path(sys.argv[1])
decorate = guard["torch_compile_guard"]

@decorate(mutates_args=[], device="cpu")
def lifecycle_add(x: torch.Tensor) -> torch.Tensor:
    return x + 1

@decorate(mutates_args=["x"], device="cpu")
def lifecycle_mutate(x: torch.Tensor) -> None:
    x.add_(2)

@decorate(mutates_args=[], device="cpu")
def lifecycle_size(n: int) -> int:
    return n + 3

for x in (torch.arange(12.).reshape(3, 4), torch.empty(0, 4), torch.arange(20.)[::2]):
    torch.testing.assert_close(lifecycle_add(x), x + 1, rtol=0, atol=0)
    mutated = x.clone()
    lifecycle_mutate(mutated)
    torch.testing.assert_close(mutated, x + 2, rtol=0, atol=0)
    assert lifecycle_size(x.numel()) == x.numel() + 3

compiled = torch.compile(lifecycle_add, backend="eager", fullgraph=True)
for x in (torch.zeros(3, 4), torch.ones(5, 2), torch.arange(12.)[::2]):
    torch.testing.assert_close(compiled(x), x + 1, rtol=0, atol=0)
with FakeTensorMode():
    x = torch.empty(7, 3)
    y = lifecycle_add(x)
    assert y.shape == x.shape and y.dtype == x.dtype

for name in ("lifecycle_add", "lifecycle_mutate", "lifecycle_size"):
    qualified = "boltops::" + name
    assert getattr(torch.ops.boltops, name).default._schema.name == qualified
    for key in ("CPU", "CUDA"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, key)

assert not torch.cuda.is_initialized()
if sys.argv[2] == "True":
    decorate.__globals__["boltops_lib"]._destroy()
print("DISPATCH_AND_TEARDOWN_OK", flush=True)
"""
    env = dict(
        os.environ,
        ROCR_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
        HCU_ADAPTER_HOST="none",
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(guard_path),
            str(destroy_explicitly),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "DISPATCH_AND_TEARDOWN_OK" in child.stdout
    assert "Traceback" not in child.stderr, child.stderr
