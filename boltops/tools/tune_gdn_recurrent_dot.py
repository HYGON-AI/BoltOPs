# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from boltops.fla.common.triton import fused_sigmoid_gating_recurrent as recurrent
from boltops.utility import arch_info
from boltops.utility.bench import time_ms


K_VALUES = (64, 128, 256)
V_VALUES = (64, 96, 128, 256)
STEP_VALUES = (1, 2, 4, 8, 16)
BATCH_VALUES = (1, 4, 16, 20)
CANDIDATES = (
    {"BV": 32, "num_warps": 1, "num_stages": 1},
    {"BV": 64, "num_warps": 2, "num_stages": 1},
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--v", type=int, nargs="+", default=list(V_VALUES))
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEP_VALUES))
    parser.add_argument("--batch", type=int, nargs="+", default=list(BATCH_VALUES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--min-speedup", type=float, default=1.03)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def _heads(K: int) -> tuple[int, int]:
    return (4, 8) if K == 64 else (8, 16)


def _make_inputs(
    *, dtype: torch.dtype, N: int, steps: int, H: int, HV: int, K: int, V: int
) -> dict[str, torch.Tensor]:
    total = N * steps
    generator = torch.Generator(device="cuda").manual_seed(117)
    q = torch.randn(
        1, total, H, K, device="cuda", dtype=dtype, generator=generator
    ) * 0.15
    k = torch.randn_like(q) * 0.15
    v = torch.randn(
        1, total, HV, V, device="cuda", dtype=dtype, generator=generator
    ) * 0.15
    a = torch.randn(
        total, HV, device="cuda", dtype=dtype, generator=generator
    ) * 0.15
    b = torch.randn_like(a) * 0.15
    A_log = torch.randn(
        HV, device="cuda", dtype=torch.float32, generator=generator
    ) * 0.1
    dt_bias = torch.randn(
        HV, device="cuda", dtype=dtype, generator=generator
    ) * 0.1
    state = torch.randn(
        N + 2,
        HV,
        V,
        K,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) * 0.01
    state_indices = torch.arange(1, N + 1, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(
        0, total + 1, steps, device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state": state,
        "state_indices": state_indices,
        "cu_seqlens": cu_seqlens,
    }


def _run(inputs: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
    K = inputs["k"].shape[-1]
    return recurrent.fused_sigmoid_gating_delta_rule_update(
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
        inputs["state_indices"],
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["cu_seqlens"],
    )


def _key(dtype_name: str, K: int, V: int, steps: int, N: int) -> str:
    return (
        f"dtype={dtype_name},K={K},V={V},T={steps},"
        f"N={recurrent._gdn_dot_n_bucket(N)}"
    )


def _qualify(
    *,
    dtype_name: str,
    K: int,
    V: int,
    steps: int,
    N: int,
    warmup: int,
    repeat: int,
    min_speedup: float,
) -> tuple[dict | None, float, float | None]:
    H, HV = _heads(K)
    inputs = _make_inputs(
        dtype=_dtype(dtype_name), N=N, steps=steps, H=H, HV=HV, K=K, V=V
    )
    original_lookup = recurrent._get_gdn_dot_recurrent_config
    try:
        recurrent._get_gdn_dot_recurrent_config = lambda *_args: None
        baseline_state = inputs["state"].clone()
        expected_state = inputs["state"].clone()
        expected_output = _run(inputs, expected_state)
        torch.cuda.synchronize()
        baseline_ms, _ = time_ms(
            lambda: _run(inputs, baseline_state),
            warmup_ms=warmup,
            repeat_ms=repeat,
        )
        baseline_us = baseline_ms * 1e3

        best_config = None
        best_us = None
        for candidate in CANDIDATES:
            if candidate["BV"] > triton_next_power_of_2(V):
                continue
            recurrent._get_gdn_dot_recurrent_config = (
                lambda *_args, candidate=candidate: candidate
            )
            actual_state = inputs["state"].clone()
            try:
                actual_output = _run(inputs, actual_state)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual_output, expected_output, atol=3e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    actual_state, expected_state, atol=3e-2, rtol=2e-2
                )
                timing_state = inputs["state"].clone()
                candidate_ms, _ = time_ms(
                    lambda: _run(inputs, timing_state),
                    warmup_ms=warmup,
                    repeat_ms=repeat,
                )
                candidate_us = candidate_ms * 1e3
            except Exception as error:
                print(f"  reject {candidate}: {error}")
                continue
            if best_us is None or candidate_us < best_us:
                best_config = dict(candidate)
                best_us = candidate_us
        if best_us is None or baseline_us / best_us < min_speedup:
            return None, baseline_us, best_us
        return best_config, baseline_us, best_us
    finally:
        recurrent._get_gdn_dot_recurrent_config = original_lookup


def triton_next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _default_output() -> Path:
    return (
        Path(recurrent.__file__).resolve().parent
        / "configs"
        / "fused_sigmoid_gating_recurrent"
        / f"fused_sigmoid_gating_recurrent-{arch_info.get_arch()}.json"
    )


def main() -> None:
    args = _parse_args()
    shapes = list(itertools.product(args.k, args.v, args.steps, args.batch))
    if args.dry_run:
        for K, V, steps, N in shapes:
            print(
                _key(args.dtype, K, V, steps, N),
                json.dumps(CANDIDATES, separators=(",", ":")),
            )
        return

    qualified = {}
    for K, V, steps, N in shapes:
        key = _key(args.dtype, K, V, steps, N)
        config, baseline_us, dot_us = _qualify(
            dtype_name=args.dtype,
            K=K,
            V=V,
            steps=steps,
            N=N,
            warmup=args.warmup,
            repeat=args.repeat,
            min_speedup=args.min_speedup,
        )
        print(f"{key}: baseline={baseline_us:.2f}us dot={dot_us} config={config}")
        if config is not None:
            qualified[key] = config

    output = args.output or _default_output()
    payload = json.loads(output.read_text()) if output.exists() else {}
    payload["dot_config"] = {
        **payload.get("dot_config", {}),
        **dict(sorted(qualified.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
