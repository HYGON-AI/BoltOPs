#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Benchmark and tune KDA prefill-stage Triton configs.

The tuner writes exact ``(arch, B, dtype, shape)`` winners under
``boltops/fla/kda/triton/configs``. Missing keys always fall back to the
in-tree static selector. Before ``--write-configs`` writes a winner, the tuner
runs a correctness smoke check against the corresponding static selector.

Examples:
  # tune the default fused prefill mainline
  python scripts/tune/triton/tune_kda_prefill.py --write-configs --e2e

  # all exported tuneable stages (including standalone/fallback paths)
  python scripts/tune/triton/tune_kda_prefill.py --kernel all

  # only compare existing JSON configs against static selectors
  python scripts/tune/triton/tune_kda_prefill.py --e2e-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import triton

sys_path = str(Path(__file__).resolve().parents[3])
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from boltops.fla.kda.triton import (
    chunk as chunk_module,
    chunk_intra as intra_module,
    chunk_intra_token_parallel as intra_parallel_module,
    chunk_state as chunk_state_module,
)
from boltops.fla.kda.triton.chunk import (
    chunk_gla_fwd_o_gk,
    chunk_kda_with_fused_gate_fwd,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    recompute_w_u_fwd,
)
from boltops.fla.kda.triton.chunk_intra import chunk_kda_fwd_intra
from boltops.fla.kda.triton.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from boltops.fla.kda.triton.chunk_state import (
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_fwd_state_output_fused,
)
from boltops.fla.common.triton.utils import prepare_chunk_indices
from boltops.utility import arch_info

BT = 64
DEVICE = "cuda"

MAINLINE = (
    "kda_gate_chunk_cumsum",
    "chunk_kda_inter_solve",
    "recompute_w_u_fwd",
    "chunk_kda_state_output_fused",
    "chunk_gla_fwd_o_gk",
)
EXTRA = (
    "kda_gate_fwd",
    "chunk_kda_state",
    "chunk_kda_intra_token_parallel",
)
ALL = MAINLINE + EXTRA

SELECTOR = {
    "kda_gate_chunk_cumsum": "get_kda_gate_chunk_cumsum_vector_config",
    "chunk_kda_inter_solve": "get_chunk_kda_inter_solve_config",
    "recompute_w_u_fwd": "get_recompute_w_u_fwd_config",
    "chunk_kda_state_output_fused": "get_fused_state_output_config",
    "chunk_gla_fwd_o_gk": "get_chunk_gla_fwd_o_config",
    "kda_gate_fwd": "get_kda_gate_fwd_config",
    "chunk_kda_state": "get_chunk_state_config",
    "chunk_kda_intra_token_parallel": (
        "get_chunk_kda_intra_token_parallel_config"
    ),
}
MODULE = {
    "kda_gate_chunk_cumsum": chunk_module,
    "chunk_kda_inter_solve": intra_module,
    "recompute_w_u_fwd": chunk_module,
    "chunk_kda_state_output_fused": chunk_state_module,
    "chunk_gla_fwd_o_gk": chunk_module,
    "kda_gate_fwd": chunk_module,
    "chunk_kda_state": chunk_state_module,
    "chunk_kda_intra_token_parallel": intra_parallel_module,
}


def dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype).replace("torch.", ""))


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    b: int
    t: int
    h: int
    k: int = 128
    v: int = 128
    dtype_str: str = "bf16"
    varlen: bool = False
    lower_bound: float | None = None


SHAPE_PRESETS = {
    "t512": ShapeSpec("t512", 1, 512, 24),
    "t2048": ShapeSpec("t2048", 1, 2048, 24),
    "t4096": ShapeSpec("t4096", 1, 4096, 24),
    "t8192": ShapeSpec("t8192", 1, 8192, 24),
    "t16384": ShapeSpec("t16384", 1, 16384, 24),
    "varlen_t8192": ShapeSpec(
        "varlen_t8192", 1, 8192, 24, varlen=True
    ),
    "batch4_t2048": ShapeSpec("batch4_t2048", 4, 2048, 24),
    "wide_h48_t8192": ShapeSpec("wide_h48_t8192", 1, 8192, 48),
}


def build_prefill_inputs(
    b: int,
    t: int,
    h: int,
    k: int,
    v_dim: int,
    *,
    dtype: torch.dtype,
    varlen: bool,
    lower_bound: float | None,
):
    if varlen and b != 1:
        raise ValueError("varlen tuning currently uses packed B=1 input")
    torch.manual_seed(20260922)
    if varlen:
        split = max(BT, (t // 2 // BT) * BT)
        lens = [length for length in (split, t - split) if length > 0]
        total = sum(lens)
        cu = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            device=DEVICE, dtype=torch.int32,
        )
    else:
        lens, total, cu = [t] * b, b * t, None
    n_sequences = len(lens)

    q = torch.randn(b, total, h, k, device=DEVICE, dtype=dtype) * 0.2
    k_tensor = torch.randn_like(q) * 0.2
    v = torch.randn(b, total, h, v_dim, device=DEVICE, dtype=dtype) * 0.2
    raw_g = -torch.rand(b, total, h, k, device=DEVICE, dtype=dtype) * 0.2
    raw_beta = torch.rand(b, total, h, device=DEVICE, dtype=dtype)
    a_log = torch.randn(h, device=DEVICE, dtype=torch.float32) * 0.2
    a = torch.tril(
        torch.randn(b, total, h, BT, device=DEVICE, dtype=dtype) * 0.1
    )
    chunk_indices = prepare_chunk_indices(cu, BT) if cu is not None else None

    gk, beta = fused_kda_gate_chunk_cumsum(
        raw_g, raw_beta, a_log, lower_bound=lower_bound,
        cu_seqlens=cu, chunk_indices=chunk_indices, chunk_size=BT,
    )
    aqk, a = chunk_kda_fwd_intra(
        q=q, k=k_tensor, gk=gk, beta=beta, scale=k ** -0.5,
        cu_seqlens=cu, chunk_indices=chunk_indices, chunk_size=BT,
        safe_gate=lower_bound is not None,
        normalize_qk=k == 128,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k_tensor, v=v, beta=beta, A=a, gk=gk,
        cu_seqlens=cu, chunk_indices=chunk_indices,
    )
    h_state, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=gk, initial_state=None,
        output_final_state=False, cu_seqlens=cu,
        chunk_indices=chunk_indices, use_exp2=True,
    )
    return {
        "B": b, "T": total, "H": h, "K": k, "V": v_dim,
        "N": n_sequences, "dtype": dtype, "lens": lens, "cu": cu,
        "chunk_indices": chunk_indices, "lower_bound": lower_bound,
        "safe_gate": lower_bound is not None,
        "normalize_qk": k == 128, "scale": k ** -0.5,
        "q": q, "k": k_tensor, "v": v, "raw_g": raw_g,
        "raw_beta": raw_beta, "A_log": a_log, "gk": gk,
        "beta": beta, "A": a, "Aqk": aqk,
        "w": w, "u": u, "kg": kg, "h": h_state, "v_new": v_new,
        "o": torch.empty_like(v),
        "Aqk_tp": torch.empty(b, total, h, BT, device=DEVICE, dtype=dtype),
        "Akk_tp": torch.empty(b, total, h, 16, device=DEVICE, dtype=torch.float32),
    }


def bench_case(fn: Callable[[], Any], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


@contextmanager
def forced_config(kernel: str, cfg: dict | None = None):
    """Force a candidate, or the static selector when ``cfg is None``."""
    selector = SELECTOR[kernel]
    module = MODULE[kernel]
    original = getattr(module, selector)
    if cfg is None:
        static = getattr(module, "static_" + selector[4:])

        def replacement(*args, **kwargs):
            kwargs.pop("b", None)
            kwargs.pop("dtype", None)
            return static(*args, **kwargs)
    else:
        replacement = lambda *_args, **_kwargs: cfg
    setattr(module, selector, replacement)
    try:
        yield
    finally:
        setattr(module, selector, original)


def candidate_sets(k: int = 128) -> dict[str, list[dict[str, int]]]:
    # The token-parallel sub-chunk kernel loads exactly one ``(BC, BK)`` tile
    # with no K loop, so ``BK < K`` silently truncates the reduction.  Keep
    # only tiles that cover the full head dimension.
    tp_bks = [bk for bk in (64, 128) if bk >= k] or [triton.next_power_of_2(k)]
    return {
        "kda_gate_chunk_cumsum": [
            {"BS": bs, "num_warps": warps, "num_stages": stages}
            for bs in (16, 32, 64, 128)
            for warps in (1, 2, 4)
            for stages in (1, 2, 3)
        ],
        "chunk_kda_inter_solve": [
            {"BK": bk, "num_warps": warps, "num_stages": stages}
            for bk in (32, 64)
            for warps in (1, 2)
            for stages in (1, 2, 3)
        ],
        "recompute_w_u_fwd": [
            {"BK": 64, "BV": bv, "num_warps": warps, "num_stages": stages}
            for bv in (32, 64, 128)
            for warps in (2, 4, 8)
            for stages in (1, 2, 3)
        ],
        "chunk_kda_state_output_fused": [
            {"BV": bv, "num_warps": warps, "num_stages": stages}
            for bv in (32, 64, 128)
            for warps in (4, 8, 16)
            for stages in (1, 2)
        ],
        "chunk_gla_fwd_o_gk": [
            {"BK": bk, "BV": bv, "num_warps": warps, "num_stages": stages}
            for bk in (64, 128)
            for bv in (64, 128)
            for warps in (4, 8)
            for stages in (1, 2, 3)
        ],
        "kda_gate_fwd": [
            {"BT": bt, "num_warps": warps, "num_stages": stages}
            for bt in (32, 64, 128)
            for warps in (2, 4, 8, 16)
            for stages in (1, 2, 3)
        ],
        "chunk_kda_state": [
            {"BV": bv, "num_warps": warps, "num_stages": stages}
            for bv in (32, 64, 128)
            for warps in (2, 4, 8)
            for stages in (1, 2)
        ],
        "chunk_kda_intra_token_parallel": [
            {"BK": bk, "num_warps": warps, "num_stages": stages}
            for bk in tp_bks
            for warps in (1, 2, 4)
            for stages in (1, 2, 3)
        ],
    }


def make_cases(x: dict) -> dict[str, Callable[[], Any]]:
    return {
        "kda_gate_chunk_cumsum": lambda: fused_kda_gate_chunk_cumsum(
            x["raw_g"], x["raw_beta"], x["A_log"],
            lower_bound=x["lower_bound"], cu_seqlens=x["cu"],
            chunk_indices=x["chunk_indices"], chunk_size=BT,
        ),
        "chunk_kda_inter_solve": lambda: chunk_kda_fwd_intra(
            q=x["q"], k=x["k"], gk=x["gk"], beta=x["beta"], scale=x["scale"],
            cu_seqlens=x["cu"], chunk_indices=x["chunk_indices"],
            chunk_size=BT, safe_gate=x["safe_gate"],
            normalize_qk=x["normalize_qk"],
        ),
        "recompute_w_u_fwd": lambda: recompute_w_u_fwd(
            k=x["k"], v=x["v"], beta=x["beta"], A=x["A"], gk=x["gk"],
            cu_seqlens=x["cu"], chunk_indices=x["chunk_indices"],
        ),
        "chunk_kda_state_output_fused": lambda: chunk_kda_fwd_state_output_fused(
            q=x["q"], w=x["w"], u=x["u"], g=x["gk"], kg=x["kg"],
            Aqk=x["Aqk"], o=x["o"], scale=x["scale"],
            cu_seqlens=x["cu"], chunk_indices=x["chunk_indices"], chunk_size=BT,
        ),
        "chunk_gla_fwd_o_gk": lambda: chunk_gla_fwd_o_gk(
            q=x["q"], v=x["v_new"], g=x["gk"], A=x["Aqk"], h=x["h"],
            o=x["o"], scale=x["scale"], cu_seqlens=x["cu"],
            chunk_indices=x["chunk_indices"], chunk_size=BT,
        ),
        "kda_gate_fwd": lambda: fused_kda_gate(
            x["gk"].reshape(x["B"], x["T"], -1), x["A_log"], x["K"],
            lower_bound=x["lower_bound"],
        ),
        "chunk_kda_state": lambda: chunk_gated_delta_rule_fwd_h(
            k=x["kg"], w=x["w"], u=x["u"], gk=x["gk"], initial_state=None,
            output_final_state=False, cu_seqlens=x["cu"],
            chunk_indices=x["chunk_indices"], use_exp2=True,
        ),
        "chunk_kda_intra_token_parallel": lambda: chunk_kda_fwd_intra_token_parallel(
            x["q"], x["k"], x["gk"], x["beta"], x["Aqk_tp"], x["Akk_tp"],
            x["scale"], cu_seqlens=x["cu"], chunk_size=BT,
        ),
    }


def flatten_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from flatten_tensors(item)


def clone_outputs(value: Any):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_outputs(item) for item in value)
    if isinstance(value, list):
        return [clone_outputs(item) for item in value]
    return value


def assert_outputs_close(actual: Any, expected: Any) -> None:
    for a, e in zip(flatten_tensors(actual), flatten_tensors(expected)):
        atol = 2e-3 if a.dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 2e-2 if a.dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(a, e, rtol=rtol, atol=atol)


def correctness_smoke(
    kernel: str, cfg: dict, case: Callable[[], Any]
) -> tuple[bool, str]:
    try:
        with forced_config(kernel, None):
            expected = clone_outputs(case())
        with forced_config(kernel, cfg):
            actual = case()
        assert_outputs_close(actual, expected)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def shape_keys(x: dict) -> dict[str, str]:
    varlen = x["cu"] is not None
    dtype = dtype_name(x["dtype"])
    geometry = (
        f"B={x['B']},dtype={dtype},T={x['T']},H={x['H']},"
        f"K={x['K']},V={x['V']},varlen={varlen}"
    )
    return {
        "kda_gate_chunk_cumsum": (
            f"B={x['B']},dtype={dtype},T={x['T']},H={x['H']},"
            f"S={x['K']},varlen={varlen}"
        ),
        "chunk_kda_inter_solve": (
            f"B={x['B']},dtype={dtype},T={x['T']},H={x['H']},HV={x['H']},"
            f"varlen={varlen},safe_gate={x['safe_gate']},"
            f"fuse_diag={x['normalize_qk']}"
        ),
        "recompute_w_u_fwd": geometry,
        "chunk_kda_state_output_fused": (
            f"B={x['B']},dtype={dtype},"
            f"programs={x['N'] * x['H']},V={x['V']}"
        ),
        "chunk_gla_fwd_o_gk": geometry,
        "kda_gate_fwd": f"B={x['B']},dtype={dtype},T={x['T']},H={x['H']}",
        "chunk_kda_state": (
            f"B={x['B']},dtype={dtype},N={x['N']},H={x['H']},"
            f"V={x['V']},K={x['K']},T={x['T']},varlen={varlen}"
        ),
        "chunk_kda_intra_token_parallel": (
            f"B={x['B']},dtype={dtype},T={x['T']},H={x['H']},HV={x['H']},"
            f"varlen={varlen}"
        ),
    }


def write_config(kernel: str, key: str, cfg: dict, arch: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    path = (
        root / "boltops/fla/kda/triton/configs" / kernel
        / f"{kernel}-{arch}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": {}}
    if path.exists():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("config"), dict):
            payload["config"] = loaded["config"]
    payload["config"][key] = cfg
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_e2e_once(x: dict, configs: dict[str, dict] | None) -> Any:
    initial = torch.randn(
        x["N"], x["H"], x["V"], x["K"],
        device=DEVICE, dtype=torch.float32,
    ) * 0.05
    if configs is None:
        return chunk_kda_with_fused_gate_fwd(
            q=x["q"], k=x["k"], v=x["v"], raw_g=x["raw_g"],
            raw_beta=x["raw_beta"], A_log=x["A_log"], g_bias=None,
            scale=x["scale"], initial_state=initial,
            output_final_state=True, lower_bound=x["lower_bound"],
            cu_seqlens=x["cu"], normalize_qk=x["normalize_qk"],
        )

    @contextmanager
    def apply_configs():
        contexts = [forced_config(kernel, cfg) for kernel, cfg in configs.items()]
        for context in contexts:
            context.__enter__()
        try:
            yield
        finally:
            for context in reversed(contexts):
                context.__exit__(None, None, None)

    with apply_configs():
        return chunk_kda_with_fused_gate_fwd(
            q=x["q"], k=x["k"], v=x["v"], raw_g=x["raw_g"],
            raw_beta=x["raw_beta"], A_log=x["A_log"], g_bias=None,
            scale=x["scale"], initial_state=initial,
            output_final_state=True, lower_bound=x["lower_bound"],
            cu_seqlens=x["cu"], normalize_qk=x["normalize_qk"],
        )


def benchmark_e2e(
    x: dict, tuned: dict[str, dict] | None, *, warmup: int, repeat: int
) -> tuple[float, float]:
    with forced_config_mainline(None):
        static = bench_case(
            lambda: run_e2e_once(x, None), warmup=warmup, repeat=repeat
        )
    with forced_config_mainline(tuned):
        tuned_ms = bench_case(
            lambda: run_e2e_once(x, tuned), warmup=warmup, repeat=repeat
        )
    return static, tuned_ms


@contextmanager
def forced_config_mainline(configs: dict[str, dict] | None):
    contexts = []
    for kernel in MAINLINE:
        contexts.append(forced_config(
            kernel, None if configs is None else configs.get(kernel)
        ))
    for context in contexts:
        context.__enter__()
    try:
        yield
    finally:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def resolve_shapes(args) -> list[ShapeSpec]:
    if args.shape:
        return [SHAPE_PRESETS[name] for name in args.shape]
    return [
        ShapeSpec(
            "custom", args.b, args.t, args.h, args.k, args.v,
            dtype_str=args.dtype, varlen=args.varlen,
            lower_bound=args.lower_bound,
        )
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--t", type=int, default=8192)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--shape", action="append", choices=tuple(SHAPE_PRESETS),
        help="add a named scale; repeatable",
    )
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument(
        "--kernel", action="append",
        choices=(*ALL, "mainline", "all"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--e2e-only", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    if args.e2e_only and args.write_configs:
        parser.error("--e2e-only cannot be combined with --write-configs")
    return args


def warmup_device(seconds: float = 5.0) -> None:
    """Drive the device to steady-state clocks before any measurement.

    The first launches of a fresh process can run against low power-state
    clocks for tens of milliseconds; without this, the first candidate of
    each (shape, kernel) block inherits a 2-15x penalty that biases the
    winner away from the configurations tried first.
    """
    a = torch.randn(4096, 4096, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=DEVICE, dtype=torch.bfloat16)
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_shapes:
        for name, shape in SHAPE_PRESETS.items():
            varlen = " varlen" if shape.varlen else ""
            print(
                f"{name:<24} B={shape.b} T={shape.t} H={shape.h} "
                f"K={shape.k} V={shape.v} {shape.dtype_str}{varlen}"
            )
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA/HIP device is required")
    warmup_device()

    requested = args.kernel
    if not requested or "all" in requested:
        selected = list(ALL)
    elif "mainline" in requested:
        selected = list(MAINLINE)
    else:
        selected = list(dict.fromkeys(requested))
    all_candidates = candidate_sets(k=args.k)
    if args.quick:
        selected_candidates = {k: all_candidates[k][:2] for k in selected}
    else:
        selected_candidates = {k: all_candidates[k] for k in selected}

    arch = arch_info.get_arch()
    all_rows: list[dict[str, Any]] = []
    e2e_results = []

    for shape in resolve_shapes(args):
        print(
            f"\n=== {shape.name}: B={shape.b} T={shape.t} H={shape.h} "
            f"K={shape.k} V={shape.v} {shape.dtype_str} "
            f"{'varlen' if shape.varlen else 'fixed'} ==="
        )
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[shape.dtype_str]
        x = build_prefill_inputs(
            shape.b, shape.t, shape.h, shape.k, shape.v, dtype=dtype,
            varlen=shape.varlen, lower_bound=shape.lower_bound,
        )
        cases = make_cases(x)
        keys = shape_keys(x)
        rows: list[dict[str, Any]] = []
        fastest: dict[str, tuple[float, dict]] = {}
        best: dict[str, dict] = {}

        if not args.e2e_only:
            for kernel in selected:
                for cfg in selected_candidates[kernel]:
                    try:
                        with forced_config(kernel, cfg):
                            elapsed = bench_case(
                                cases[kernel], warmup=args.warmup,
                                repeat=args.repeat,
                            )
                        status = "ok"
                    except triton.OutOfResources:
                        elapsed, status = math.inf, "oor"
                    latency_us = (
                        elapsed * 1000 if math.isfinite(elapsed) else math.inf
                    )
                    rows.append({
                        "arch": arch, "shape": shape.name,
                        "kernel": kernel, "status": status, **cfg,
                        "latency_us": round(latency_us, 2),
                    })
                    if math.isfinite(elapsed) and (
                        kernel not in fastest or elapsed < fastest[kernel][0]
                    ):
                        fastest[kernel] = (elapsed, cfg)
                    print(
                        f"{kernel:<34} {cfg}: {latency_us:10.2f} us {status}"
                    )

            for kernel, (_, cfg) in fastest.items():
                ok, detail = correctness_smoke(kernel, cfg, cases[kernel])
                print(f"correctness {kernel}: {detail}")
                if ok:
                    best[kernel] = cfg

            print("\nbest configs:")
            for kernel, cfg in best.items():
                print(f"  {keys[kernel]}: {cfg}")
                if args.write_configs:
                    print("    ->", write_config(kernel, keys[kernel], cfg, arch))

        if args.e2e or args.e2e_only:
            tuned_configs = None if args.e2e_only else best
            static_ms, tuned_ms = benchmark_e2e(
                x, tuned_configs,
                warmup=max(args.warmup, 1), repeat=max(args.repeat, 3),
            )
            speedup = static_ms / tuned_ms
            e2e_results.append((shape.name, static_ms, tuned_ms, speedup))
            print(
                f"e2e static={static_ms:.3f} ms tuned={tuned_ms:.3f} ms "
                f"speedup={speedup:.3f}x"
            )

        all_rows.extend(rows)

    if e2e_results:
        print("\ne2e summary:")
        for name, static_ms, tuned_ms, speedup in e2e_results:
            print(
                f"  {name:<24} static={static_ms:.3f} ms "
                f"tuned={tuned_ms:.3f} ms speedup={speedup:.3f}x"
            )

    if args.output_csv and all_rows:
        with args.output_csv.open("w", newline="") as f:
            fieldnames = []
            for row in all_rows:
                fieldnames.extend(
                    key for key in row if key not in fieldnames
                )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
