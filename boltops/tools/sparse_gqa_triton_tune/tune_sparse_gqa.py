#!/usr/bin/env python3
"""Tune Qwen Sparse-GQA production cases on the current GPU/DCU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Union

import torch
import triton

from boltops.fa.sparse_gqa.triton.compact import _launch_compact_hcu_fa
from boltops.fa.sparse_gqa.triton.config import (
    CONFIG_DIR,
    config_match,
    config_path,
    load_config,
)
from boltops.fa.sparse_gqa.triton.packed_decode import (
    _launch_sparse_gqa_chunk_prefill,
)
from boltops.tools.sparse_gqa_triton_tune.benchmark_cases import (
    PRODUCTION_CASES,
    SparseGQABenchmarkCase,
    make_production_inputs,
)
from boltops.tools.sparse_gqa_triton_tune.reference import (
    reference_compact,
    reference_decode,
)


COMPACT_BASELINE = {
    "BLOCK_TOPK": 16,
    "num_warps": 8,
    "num_stages": 1,
}
DECODE_BASELINE = {
    "BLOCK_N": 32,
    "num_warps": 8,
    "num_stages": 2,
}
COMPACT_CANDIDATES = tuple(
    {"BLOCK_TOPK": block, "num_warps": warps, "num_stages": stages}
    for block in (8, 16, 32, 64)
    for warps in (2, 4, 8)
    for stages in (1, 2)
)
DECODE_CANDIDATES = tuple(
    {"BLOCK_N": block, "num_warps": warps, "num_stages": stages}
    for block in (16, 32, 64, 128)
    for warps in (2, 4, 8)
    for stages in (1, 2, 3)
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _bench(fn: Callable[[], object], warmup_ms: int, repeat_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup_ms,
            rep=repeat_ms,
            return_mode="median",
        )
    )


def _select(
    candidates: tuple[dict, ...],
    launch: Callable[[dict], object],
    check: Callable[[object], None],
    *,
    warmup_ms: int,
    repeat_ms: int,
    label: str,
) -> tuple[dict, float]:
    best_config = None
    best_ms = math.inf
    for index, candidate in enumerate(candidates, 1):
        try:
            output = launch(candidate)
            check(output)
            latency_ms = _bench(
                lambda candidate=candidate: launch(candidate),
                warmup_ms,
                repeat_ms,
            )
        except Exception as exc:
            print(
                f"[{label} {index:02d}/{len(candidates)}] "
                f"ERROR {candidate}: {exc}",
                flush=True,
            )
            continue
        print(
            f"[{label} {index:02d}/{len(candidates)}] "
            f"{latency_ms:.6f} ms {candidate}",
            flush=True,
        )
        if latency_ms < best_ms:
            best_config = candidate
            best_ms = latency_ms
    if best_config is None:
        raise RuntimeError(f"all {label} candidates failed")
    return best_config, best_ms


def _write_config(
    *,
    kind: str,
    config: dict,
    case: SparseGQABenchmarkCase,
    hq: int,
    dtype: Union[torch.dtype, str],
    output_dir: Path,
) -> Path:
    default_path = config_path(kind)
    match = config_match(
        kind,
        case.batch,
        case.topk,
        hq,
        case.kv_heads,
        case.head_dim,
        dtype,
        page_size=case.page_size,
    )
    path = output_dir / default_path.name
    payload = {"configs": []}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"refusing to overwrite invalid config file {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("configs"), list
        ):
            raise RuntimeError(
                f"refusing to overwrite invalid config file {path}"
            )
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("match"), dict)
            and isinstance(item.get("config"), dict)
            for item in payload["configs"]
        ):
            raise RuntimeError(
                f"refusing to overwrite invalid config entries in {path}"
            )
    entry = {"match": match, "config": config}
    for index, existing in enumerate(payload["configs"]):
        if isinstance(existing, dict) and existing.get("match") == match:
            payload["configs"][index] = entry
            break
    else:
        payload["configs"].append(entry)
    payload["configs"].sort(
        key=lambda item: json.dumps(item.get("match", {}), sort_keys=True)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    load_config.cache_clear()
    print(f"Wrote {path}")
    return path


def _tune_case(
    case: SparseGQABenchmarkCase,
    *,
    access_pattern: str,
    fp8_cache: bool,
    seed: int,
    warmup_ms: int,
    repeat_ms: int,
    quick: bool,
    output_dir: Path,
    compact_results: dict[tuple, dict],
) -> None:
    print(f"=== {case.case_id} ===", flush=True)
    data = make_production_inputs(
        case,
        access_pattern=access_pattern,
        fp8_cache=fp8_cache,
        seed=seed,
    )
    compact_data = data["compact"]
    q, packed_k, packed_v, indices, cu_q, cu_k, kv_lens = data["decode"]

    expected_k, expected_v = reference_compact(
        compact_data,
        k_scale=data["k_scale"],
        v_scale=data["v_scale"],
    )

    def launch_compact(config: dict) -> None:
        _launch_compact_hcu_fa(
            compact_data["k"],
            compact_data["v"],
            compact_data["req_to_token"],
            compact_data["req_indices"],
            compact_data["indices"],
            compact_data["seq_lens"],
            compact_data["cu_k"],
            packed_k,
            packed_v,
            case.batch,
            case.topk,
            case.page_size,
            data["k_scale"],
            data["v_scale"],
            config=config,
        )

    def check_compact(_: object) -> None:
        torch.testing.assert_close(packed_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(packed_v, expected_v, rtol=0, atol=0)

    compact_key = (
        case.batch,
        case.topk,
        case.kv_heads,
        case.head_dim,
        _dtype_name(compact_data["k"].dtype),
        _dtype_name(packed_k.dtype),
    )
    write_compact = compact_key not in compact_results
    if write_compact:
        compact_candidates = (
            (COMPACT_BASELINE,) if quick else COMPACT_CANDIDATES
        )
        best_compact, _ = _select(
            compact_candidates,
            launch_compact,
            check_compact,
            warmup_ms=warmup_ms,
            repeat_ms=repeat_ms,
            label=f"{case.case_id}:compact",
        )
        compact_results[compact_key] = best_compact
    else:
        best_compact = compact_results[compact_key]
        print(
            f"[{case.case_id}:compact] reuse TP-independent config "
            f"{best_compact}",
            flush=True,
        )
    launch_compact(best_compact)
    check_compact(None)

    scale = case.head_dim**-0.5
    expected_out = reference_decode(
        q,
        expected_k,
        expected_v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
    )

    def launch_decode(config: dict) -> torch.Tensor:
        return _launch_sparse_gqa_chunk_prefill(
            q,
            packed_k,
            packed_v,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            scale,
            1,
            config=config,
        )

    def check_decode(output: object) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("decode candidate did not return a tensor")
        torch.testing.assert_close(
            output,
            expected_out,
            rtol=1e-2,
            atol=4e-3,
        )

    decode_candidates = (DECODE_BASELINE,) if quick else DECODE_CANDIDATES
    best_decode, _ = _select(
        decode_candidates,
        launch_decode,
        check_decode,
        warmup_ms=warmup_ms,
        repeat_ms=repeat_ms,
        label=f"{case.case_id}:decode",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dtype = (
        f"{_dtype_name(compact_data['k'].dtype)}_to_"
        f"{_dtype_name(packed_k.dtype)}"
    )
    if write_compact:
        _write_config(
            kind="compact",
            config=best_compact,
            case=case,
            hq=0,
            dtype=compact_dtype,
            output_dir=output_dir,
        )
    _write_config(
        kind="decode",
        config=best_decode,
        case=case,
        hq=case.q_heads,
        dtype=q.dtype,
        output_dir=output_dir,
    )
    del data, expected_k, expected_v, expected_out
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tp",
        type=int,
        action="append",
        choices=(4, 8),
        help="TP value to tune; repeat to select both (default: TP4 and TP8)",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=("D1", "D8", "D32"),
        help="production case to tune; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--access-pattern",
        choices=("production", "contiguous"),
        default="production",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--fp8-cache",
        dest="fp8_cache",
        action="store_true",
        help="use the production FP8 cache path (default)",
    )
    cache_group.add_argument(
        "--bf16-cache",
        dest="fp8_cache",
        action="store_false",
        help="tune the non-production BF16 cache path",
    )
    parser.set_defaults(fp8_cache=True)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="validate and save only the baseline candidates",
    )
    parser.add_argument("--output-dir", type=Path, default=CONFIG_DIR)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP device is required")
    if args.warmup_ms < 0 or args.rep_ms <= 0:
        parser.error("warmup must be >= 0 and repeat must be > 0")

    selected_cases = [
        case
        for case in PRODUCTION_CASES
        if (not args.tp or case.tp in args.tp)
        and (not args.case or case.name in args.case)
    ]
    compact_results: dict[tuple, dict] = {}
    for case in selected_cases:
        _tune_case(
            case,
            access_pattern=args.access_pattern,
            fp8_cache=args.fp8_cache,
            seed=args.seed,
            warmup_ms=args.warmup_ms,
            repeat_ms=args.rep_ms,
            quick=args.quick,
            output_dir=args.output_dir,
            compact_results=compact_results,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
