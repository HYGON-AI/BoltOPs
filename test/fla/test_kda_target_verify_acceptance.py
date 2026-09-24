"""Acceptance checks for the KDA target-verify kernel against SGLang."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as bolt_target_verify,
)
from ._gdn_reference import naive_recurrent_gated_delta_rule


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
ATOL = 3e-2
RTOL = 2e-2


def _load_sglang_target_verify():
    repo = Path(os.environ.get("SGLANG_REPO", "/workspace/sglang"))
    path = (
        repo
        / "python"
        / "sglang"
        / "kernels"
        / "ops"
        / "attention"
        / "fla"
        / "fused_sigmoid_gating_recurrent.py"
    )
    if not path.exists():
        pytest.skip(f"SGLang baseline source not found: {path}")

    module_name = "sglang_baseline_fused_sigmoid_gating_recurrent"
    if module_name in sys.modules:
        return sys.modules[module_name].fused_sigmoid_gating_delta_rule_update

    # The baseline has no relative imports. Stub its arch helper so loading the
    # kernel does not invoke SGLang's full inference-runtime package __init__.
    utils = types.ModuleType("sglang.kernels.jit.utils")
    utils.is_arch_support_pdl = lambda: False
    sys.modules["sglang.kernels.jit.utils"] = utils
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.fused_sigmoid_gating_delta_rule_update


def _make_verify_inputs(
    *,
    n_requests: int,
    max_steps: int,
    lengths: list[int] | None,
    dtype: torch.dtype,
    tree: bool,
    seed: int,
):
    torch.manual_seed(seed)
    if lengths is None:
        lengths = [max_steps] * n_requests
    if len(lengths) != n_requests:
        raise ValueError("lengths must have one entry per request")

    allocated_steps = n_requests * max_steps
    h, hv, k, v = 2, 4, 128, 128
    q = torch.randn(1, allocated_steps, h, k, device=DEVICE, dtype=dtype) * 0.1
    k_tensor = torch.randn_like(q) * 0.1
    v_tensor = torch.randn(1, allocated_steps, hv, v, device=DEVICE, dtype=dtype) * 0.1
    a = torch.randn(1, allocated_steps, hv, k, device=DEVICE, dtype=dtype) * 0.1
    b = torch.randn(1, allocated_steps, hv, device=DEVICE, dtype=dtype) * 0.1
    A_log = torch.randn(hv, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(hv * k, device=DEVICE, dtype=dtype) * 0.1

    boundaries = []
    offset = 0
    for length in lengths:
        boundaries.append(offset)
        offset += max_steps
    boundaries.append(allocated_steps)
    cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int32)

    parents = None
    if tree:
        parents = torch.zeros(n_requests, max_steps, device=DEVICE, dtype=torch.int32)
        for request, length in enumerate(lengths):
            for step in range(1, length):
                parents[request, step] = torch.randint(0, step, (1,), device=DEVICE)

    return {
        "lengths": lengths,
        "q": q,
        "k": k_tensor,
        "v": v_tensor,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "parents": parents,
    }


def _run_target_verify(function, inputs, *, lower_bound, disable_state_update=True):
    n_requests = len(inputs["lengths"])
    max_steps = max(inputs["lengths"])
    state = inputs["state"]
    intermediate = inputs["intermediate"]
    state_indices = inputs["state_indices"]
    intermediate_indices = inputs["intermediate_indices"]
    return function(
        inputs["A_log"],
        inputs["a"],
        inputs["dt_bias"],
        1.0,
        20.0,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b"],
        state,
        state_indices,
        scale=inputs["k"].shape[-1] ** -0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["cu_seqlens"],
        is_kda=True,
        lower_bound=lower_bound,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=intermediate_indices,
        cache_steps=max_steps,
        retrieve_parent_token=inputs["parents"],
    )


def _assert_verify_outputs_close(actual, expected, inputs):
    offset = 0
    for request, length in enumerate(inputs["lengths"]):
        torch.testing.assert_close(
            actual[..., offset : offset + length, :, :],
            expected[..., offset : offset + length, :, :],
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            inputs["intermediate_actual"][request, :length],
            inputs["intermediate_expected"][request, :length],
            atol=ATOL,
            rtol=RTOL,
        )
        offset += inputs["q"].shape[1] // len(inputs["lengths"])


@GPU_REQUIRED
@pytest.mark.parametrize("draft_token_num", [1, 2, 4, 8])
@pytest.mark.parametrize(
    ("dtype", "n_requests", "lengths", "lower_bound", "tree", "seed"),
    [
        pytest.param(torch.bfloat16, 1, None, -5.0, False, 101, id="single-chain-safe"),
        pytest.param(torch.bfloat16, 4, None, None, False, 102, id="multi-chain-softplus"),
        pytest.param(torch.bfloat16, 4, None, -5.0, False, 103, id="multi-chain-safe"),
        pytest.param(torch.float16, 4, None, -5.0, False, 104, id="multi-chain-fp16"),
        pytest.param(torch.bfloat16, 4, [1, 2, 4, 4], -5.0, False, 105, id="ragged-ghost-chain"),
        pytest.param(torch.bfloat16, 4, [1, 2, 4, 4], -5.0, True, 106, id="ragged-ghost-tree"),
        pytest.param(torch.float16, 4, [1, 2, 4, 4], -5.0, True, 107, id="ragged-ghost-tree-fp16"),
    ],
)
def test_target_verify_matches_sglang_baseline(
    draft_token_num: int,
    dtype: torch.dtype,
    n_requests: int,
    lengths: list[int] | None,
    lower_bound: float | None,
    tree: bool,
    seed: int,
) -> None:
    sglang_target_verify = _load_sglang_target_verify()
    effective_lengths = (
        [min(length, draft_token_num) for length in lengths]
        if lengths is not None
        else [draft_token_num] * n_requests
    )
    inputs = _make_verify_inputs(
        n_requests=n_requests,
        max_steps=draft_token_num,
        lengths=effective_lengths,
        dtype=dtype,
        tree=tree,
        seed=seed + draft_token_num,
    )
    inputs["state"] = torch.randn(8, 4, 128, 128, device=DEVICE) * 0.01
    inputs["intermediate"] = torch.full(
        (n_requests, draft_token_num, 4, 128, 128), 777.0, device=DEVICE
    )
    inputs["state_indices"] = torch.tensor(
        [3, 0, 5, 2][:n_requests], device=DEVICE, dtype=torch.int32
    )
    inputs["intermediate_indices"] = torch.tensor(
        [0, 1, 3, 2][:n_requests], device=DEVICE, dtype=torch.int32
    )
    state_before = inputs["state"].clone()

    inputs["intermediate_actual"] = inputs["intermediate"]
    inputs["intermediate_expected"] = inputs["intermediate"].clone()
    actual_state = state_before.clone()
    expected_state = state_before.clone()
    inputs["state"] = actual_state
    actual = _run_target_verify(
        bolt_target_verify, inputs, lower_bound=lower_bound
    )

    inputs["state"] = expected_state
    inputs["intermediate"] = inputs["intermediate_expected"]
    expected = _run_target_verify(
        sglang_target_verify, inputs, lower_bound=lower_bound
    )
    torch.cuda.synchronize()
    _assert_verify_outputs_close(actual, expected, inputs)

    # Ghost/padding output and scratch rows are intentionally excluded above.
    # Verify must never touch the committed pool when the SGLang flag is set.
    assert torch.equal(actual_state, state_before)
    assert torch.equal(expected_state, state_before)

    # Commit accepted prefixes, then compare a second verify from the resulting
    # committed states. This covers repeated verify + accept/reject rollback.
    accepted_lengths = [
        min(1 if request % 2 == 0 else draft_token_num, length)
        for request, length in enumerate(effective_lengths)
    ]
    for request, accepted_length in enumerate(accepted_lengths):
        slot = int(inputs["state_indices"][request])
        actual_state[slot] = inputs["intermediate_actual"][
            request, accepted_length - 1
        ]
        expected_state[slot] = inputs["intermediate_expected"][
            request, accepted_length - 1
        ]

    inputs["intermediate_actual"] = inputs["intermediate"].clone()
    inputs["intermediate_expected"] = inputs["intermediate"].clone()
    inputs["intermediate"] = inputs["intermediate_actual"]
    inputs["state"] = actual_state
    actual_second = _run_target_verify(
        bolt_target_verify, inputs, lower_bound=lower_bound
    )
    inputs["state"] = expected_state
    inputs["intermediate"] = inputs["intermediate_expected"]
    expected_second = _run_target_verify(
        sglang_target_verify, inputs, lower_bound=lower_bound
    )
    torch.cuda.synchronize()
    _assert_verify_outputs_close(actual_second, expected_second, inputs)
    assert torch.equal(actual_state, expected_state)


@GPU_REQUIRED
@pytest.mark.parametrize("draft_token_num", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_replayssm_ring_and_commit_matches_baseline(
    draft_token_num: int, dtype: torch.dtype
) -> None:
    sglang_target_verify = _load_sglang_target_verify()
    seed = 200 + draft_token_num + (dtype == torch.float16)
    inputs = _make_verify_inputs(
        n_requests=2,
        max_steps=draft_token_num,
        lengths=None,
        dtype=dtype,
        tree=False,
        seed=seed,
    )
    inputs["state"] = torch.randn(4, 4, 128, 128, device=DEVICE) * 0.01
    inputs["intermediate"] = torch.empty(
        2, draft_token_num, 4, 128, 128, device=DEVICE
    )
    inputs["state_indices"] = torch.tensor([0, 2], device=DEVICE, dtype=torch.int32)
    inputs["intermediate_indices"] = torch.arange(2, device=DEVICE, dtype=torch.int32)
    ring_length = draft_token_num
    sentinel = 888.0

    def make_rings():
        return {
            "rawv": torch.full((4, 4, ring_length, 128), sentinel, device=DEVICE, dtype=dtype),
            "rawk": torch.full((4, 2, ring_length, 128), sentinel, device=DEVICE, dtype=dtype),
            "g": torch.full((4, 4, ring_length, 128), sentinel, device=DEVICE, dtype=dtype),
            "beta": torch.full((4, 4, ring_length), sentinel, device=DEVICE, dtype=dtype),
        }

    actual_rings = make_rings()
    expected_rings = make_rings()
    common = {
        "scale": 128**-0.5,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": inputs["cu_seqlens"],
        "is_kda": True,
        "lower_bound": -5.0,
        "disable_state_update": True,
        "cache_ring": True,
    }
    actual = bolt_target_verify(
        inputs["A_log"], inputs["a"], inputs["dt_bias"], 1.0, 20.0,
        inputs["q"], inputs["k"], inputs["v"], inputs["b"],
        inputs["state"], inputs["state_indices"], **common,
        replayssm_rawv=actual_rings["rawv"],
        replayssm_rawk=actual_rings["rawk"],
        replayssm_g=actual_rings["g"],
        replayssm_beta=actual_rings["beta"],
    )
    expected = sglang_target_verify(
        inputs["A_log"], inputs["a"], inputs["dt_bias"], 1.0, 20.0,
        inputs["q"], inputs["k"], inputs["v"], inputs["b"],
        inputs["state"], inputs["state_indices"], **common,
        replayssm_rawv=expected_rings["rawv"],
        replayssm_rawk=expected_rings["rawk"],
        replayssm_g=expected_rings["g"],
        replayssm_beta=expected_rings["beta"],
    )
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    for name in ("rawv", "rawk", "g", "beta"):
        torch.testing.assert_close(
            actual_rings[name], expected_rings[name], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            actual_rings[name][1], torch.full_like(actual_rings[name][1], sentinel)
        )

    # Baseline's eager intermediate state is the committed-state oracle for the
    # ring fold. Replay each accepted prefix from its fused ring records.
    inputs["cache_ring"] = False
    eager_baseline = sglang_target_verify(
        inputs["A_log"], inputs["a"], inputs["dt_bias"], 1.0, 20.0,
        inputs["q"], inputs["k"], inputs["v"], inputs["b"],
        inputs["state"], inputs["state_indices"],
        scale=common["scale"], use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["cu_seqlens"], is_kda=True, lower_bound=-5.0,
        disable_state_update=True,
        intermediate_states_buffer=inputs["intermediate"],
        intermediate_state_indices=inputs["intermediate_indices"],
        cache_steps=draft_token_num,
    )
    torch.cuda.synchronize()
    for request, slot in enumerate(inputs["state_indices"].tolist()):
        accepted_length = draft_token_num if request == 0 else draft_token_num // 2
        ring_state = inputs["state"][slot].transpose(-2, -1).unsqueeze(0)
        token_start = request * draft_token_num
        replay_output, replay_state = naive_recurrent_gated_delta_rule(
            inputs["q"][:, token_start : token_start + accepted_length],
            actual_rings["rawk"][slot].transpose(0, 1).unsqueeze(0),
            actual_rings["rawv"][slot].transpose(0, 1).unsqueeze(0),
            actual_rings["beta"][slot].transpose(0, 1).unsqueeze(0),
            actual_rings["g"][slot].transpose(0, 1).unsqueeze(0),
            scale=common["scale"],
            initial_state=ring_state,
            output_final_state=True,
        )
        committed_actual = replay_state[0].transpose(-2, -1)
        committed_expected = inputs["intermediate"][request, accepted_length - 1]
        torch.testing.assert_close(
            committed_actual, committed_expected, atol=ATOL, rtol=RTOL
        )
        assert torch.isfinite(eager_baseline).all()
