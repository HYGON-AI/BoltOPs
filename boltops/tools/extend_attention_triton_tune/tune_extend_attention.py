#!/usr/bin/env python3
"""Tune Extend Attention cases on the current HCU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import triton

from boltops.fa.extend_attention.triton import config as config_module
from boltops.fa.extend_attention.triton import extend_attention as backend
from boltops.tools.extend_attention_triton_tune.benchmark_cases import (
    KV_CACHE_DTYPES,
    TUNING_CASES,
    ExtendAttentionTuneCase,
    make_tune_inputs,
)
from boltops.utility.arch_info import get_device_info
from boltops.utility.triton_capability import (
    get_triton_capabilities,
    get_triton_config_dir,
)


_CONFIG_KEY_FIELDS = [
    "batch_size",
    "max_len_extend",
    "kv_group_num",
    "Lq",
    "Lv",
    "USE_CUSTOM_MASK",
    "IS_CAUSAL",
    "SKIP_PREFIX_CUSTOM_MASK",
    "HAS_SINK",
    "SLIDING_WINDOW_SIZE",
    "xai_temperature_len",
    "Q_Extend",
    "K_Extend",
    "V_Extend",
    "O_Extend",
    "K_Buffer",
    "V_Buffer",
    "qo_indptr",
    "kv_indptr",
    "kv_indices",
    "mask_ptr",
    "mask_indptr",
    "sink_ptr",
    "window_kv_offset_ptr",
]


def _bench(fn: Callable[[], None], warmup_ms: int, repeat_ms: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup_ms,
            rep=repeat_ms,
            return_mode="median",
        )
    )


def _config_key(case: ExtendAttentionTuneCase, data: dict) -> tuple:
    kind = (
        config_module.SHORT_QUERY
        if case.kind == "short_query"
        else config_module.LONG_QUERY
    )
    return config_module.make_config_key(
        kind,
        batch_size=case.batch_size,
        max_len_extend=case.extend_length,
        kv_group_num=case.query_heads // case.kv_heads,
        Lq=case.qk_dim,
        Lv=case.value_dim,
        use_custom_mask=case.use_custom_mask,
        is_causal=True,
        skip_prefix_custom_mask=True,
        has_sink=case.with_sinks,
        sliding_window_size=case.sliding_window_size,
        xai_temperature_len=-1,
        q_extend=data["q"],
        k_extend=data["k"],
        v_extend=data["v"],
        o_extend=data["output"],
        k_buffer=data["k_buffer"],
        v_buffer=data["v_buffer"],
        qo_indptr=data["qo_indptr"],
        kv_indptr=data["kv_indptr"],
        kv_indices=data["kv_indices"],
        custom_mask=data["custom_mask"],
        mask_indptr=data["mask_indptr"],
        sinks=data["sinks"],
        window_kv_offsets=data["window_kv_offsets"],
    )


def _launch(case: ExtendAttentionTuneCase, data: dict, candidate: dict) -> None:
    kind = (
        config_module.SHORT_QUERY
        if case.kind == "short_query"
        else config_module.LONG_QUERY
    )
    block_dmodel, block_dpe, block_dv, _, _, _ = (
        backend._get_block_sizes_for_extend_attention(case.qk_dim, case.value_dim)
    )
    backend._launch_optimized_kernel(
        kind,
        q_extend=data["q"],
        k_extend=data["k"],
        v_extend=data["v"],
        o_extend=data["output"],
        k_buffer=data["k_buffer"],
        v_buffer=data["v_buffer"],
        qo_indptr=data["qo_indptr"],
        kv_indptr=data["kv_indptr"],
        kv_indices=data["kv_indices"],
        custom_mask=data["custom_mask"],
        mask_indptr=data["mask_indptr"],
        sinks=data["sinks"],
        window_kv_offsets=data["window_kv_offsets"],
        sm_scale=case.qk_dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
        is_causal=True,
        skip_prefix_custom_mask=True,
        sliding_window_size=case.sliding_window_size,
        logit_cap=0.0,
        xai_temperature_len=-1,
        max_len_extend=case.extend_length,
        block_dmodel=block_dmodel,
        block_dpe=block_dpe,
        block_dv=block_dv,
        Lq=case.qk_dim,
        Lv=case.value_dim,
        config_override=candidate,
    )


def _unique_configs(configs: list[dict]) -> tuple[dict, ...]:
    unique = {}
    for candidate in configs:
        cleaned = config_module.sanitize_launch_config(dict(candidate))
        token = json.dumps(cleaned, sort_keys=True)
        unique[token] = cleaned
    return tuple(unique.values())


def _search_candidates(kind: str, kv_cache_dtype: torch.dtype) -> list[dict]:
    """Return the dtype-specific search space after early pruning.

    BF16 uses a 54-config search space. FP8 also tries kpack=2 and excludes
    BLOCK_N=16. Both kpack values fail during FP8 lowering for the short-query
    kernel on Triton 3.6 HCU.
    """
    if kind not in (config_module.SHORT_QUERY, config_module.LONG_QUERY):
        raise ValueError(f"unknown Extend Attention config kind: {kind!r}")
    if kv_cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported KV cache dtype: {kv_cache_dtype}")
    kpacks = (1,) if kv_cache_dtype == torch.bfloat16 else (1, 2)
    candidates = []
    for block_m in (16, 32, 64):
        for block_n in (16, 32, 64):
            for num_warps in (4, 8):
                for num_stages in (1, 2):
                    if block_m == 64 and num_stages >= 2:
                        continue
                    if block_m >= 64 and num_warps >= 8:
                        continue
                    for sched_latency in ("none", "mmac5-ds10"):
                        for kpack in kpacks:
                            if (
                                kv_cache_dtype == torch.float8_e4m3fn
                                and block_n == 16
                            ):
                                continue
                            candidates.append(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "waves_per_eu": 1,
                                    "matrix_instr_nonkdim": 16,
                                    "sched_latency": sched_latency,
                                    "kpack": kpack,
                                    "USE_MLS": False,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return candidates


def _candidate_configs(
    case: ExtendAttentionTuneCase,
    *,
    quick: bool,
    kv_cache_dtype: torch.dtype,
) -> tuple[dict, ...]:
    kind = (
        config_module.SHORT_QUERY
        if case.kind == "short_query"
        else config_module.LONG_QUERY
    )
    defaults = dict(config_module._DEFAULT_CONFIGS[kind])
    if quick:
        return (config_module.sanitize_launch_config(defaults),)
    return _unique_configs(_search_candidates(kind, kv_cache_dtype))


def _requested_kv_cache_dtypes(
    requested: str, *, fp8_available: bool
) -> tuple[str, ...]:
    """Resolve CLI dtype selection for the supported FP8 architectures."""
    if requested == "fp8" and not fp8_available:
        raise ValueError("FP8 KV cache tuning requires gfx938 or gfx92a")
    if requested == "both":
        return tuple(KV_CACHE_DTYPES) if fp8_available else ("bf16",)
    return (requested,)


def _fp8_kv_cache_tuning_available(arch: str) -> bool:
    """Return whether Extend Attention supports FP8 KV cache tuning."""
    return arch in ("gfx938", "gfx92a")


def _select(
    case: ExtendAttentionTuneCase,
    data: dict,
    candidates: tuple[dict, ...],
    *,
    warmup_ms: int,
    repeat_ms: int,
) -> tuple[dict, float]:
    best_config = None
    best_ms = math.inf
    for index, candidate in enumerate(candidates, 1):
        try:
            _launch(case, data, candidate)
            latency_ms = _bench(
                lambda candidate=candidate: _launch(case, data, candidate),
                warmup_ms,
                repeat_ms,
            )
        except Exception as exc:
            print(
                f"[{case.name} {index:03d}/{len(candidates)}] "
                f"ERROR {candidate}: {exc}",
                flush=True,
            )
            continue
        print(
            f"[{case.name} {index:03d}/{len(candidates)}] "
            f"{latency_ms:.6f} ms {candidate}",
            flush=True,
        )
        if latency_ms < best_ms:
            best_config = candidate
            best_ms = latency_ms
    if best_config is None:
        raise RuntimeError(f"all candidates failed for {case.name}")
    return best_config, best_ms


def _config_path(case: ExtendAttentionTuneCase, data: dict, output_dir: Path) -> Path:
    info = get_device_info(data["q"].device)
    file_key = config_module.make_file_key(data["q"], data["k_buffer"])
    selectors = "-".join(file_key)
    versioned_output_dir = get_triton_config_dir(
        output_dir, get_triton_capabilities()
    )
    return (
        versioned_output_dir
        / f"{case.kind}-device={info.arch}_cu{info.cu_count}-{selectors}.json"
    )


def _write_config(
    case: ExtendAttentionTuneCase,
    data: dict,
    candidate: dict,
    output_dir: Path,
) -> Path:
    path = _config_path(case, data, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": _CONFIG_KEY_FIELDS, "config": {}}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"refusing to overwrite invalid config {path}: {exc}") from exc
        if not isinstance(payload.get("config"), dict):
            raise RuntimeError(f"refusing to overwrite invalid config {path}")

    payload["config"][str(_config_key(case, data))] = candidate
    payload["config"] = dict(sorted(payload["config"].items()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    temporary.replace(path)
    config_module._load_configs.cache_clear()
    config_module.get_kernel_config.cache_clear()
    print(f"Wrote {path}", flush=True)
    return path


def tune_case(
    case: ExtendAttentionTuneCase,
    *,
    seed: int,
    warmup_ms: int,
    repeat_ms: int,
    quick: bool,
    output_dir: Path,
    kv_cache_dtype_name: str,
    seen_keys: set[tuple],
) -> tuple[dict, float] | None:
    kv_cache_dtype = KV_CACHE_DTYPES[kv_cache_dtype_name]
    print(
        f"=== {case.name} ({case.kind}, KV Cache {kv_cache_dtype_name}) ===",
        flush=True,
    )
    data = make_tune_inputs(
        case, seed=seed, kv_cache_dtype=kv_cache_dtype
    )
    identity = (
        case.kind,
        config_module.make_file_key(data["q"], data["k_buffer"]),
        _config_key(case, data),
    )
    if identity in seen_keys:
        print(f"SKIP duplicate key: {case.name}", flush=True)
        del data
        torch.cuda.empty_cache()
        return None
    seen_keys.add(identity)
    candidates = _candidate_configs(
        case, quick=quick, kv_cache_dtype=kv_cache_dtype
    )
    print(f"Candidates: {len(candidates)}", flush=True)
    best_config, best_ms = _select(
        case,
        data,
        candidates,
        warmup_ms=warmup_ms,
        repeat_ms=repeat_ms,
    )
    _write_config(case, data, best_config, output_dir)
    print(
        f"BEST {case.name}: {best_ms:.6f} ms {best_config}",
        flush=True,
    )
    del data
    torch.cuda.empty_cache()
    return best_config, best_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in TUNING_CASES),
        help="case to tune; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--kind",
        choices=("short_query", "long_query"),
        help="limit tuning to one kernel family",
    )
    parser.add_argument("--warmup-ms", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("bf16", "fp8", "both"),
        default="both",
        help=(
            "KV Cache dtype to tune; on gfx938/gfx92a, 'both' writes "
            "separate BF16 and FP8 JSON files; on other architectures, "
            "'both' writes BF16 only (default: both)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="benchmark and save only the built-in default config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config_module.CONFIG_ROOT,
        help=(
            "config root; results are written below its current Triton "
            "capability directory (default: packaged config root)"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in TUNING_CASES:
            print(case.name)
        return 0
    if not torch.cuda.is_available() or not backend._is_hcu:
        parser.error("an HCU device is required")
    if args.warmup_ms < 0 or args.rep_ms <= 0:
        parser.error("warmup must be >= 0 and rep must be > 0")
    selected = [
        case
        for case in TUNING_CASES
        if (not args.case or case.name in args.case)
        and (not args.kind or case.kind == args.kind)
    ]
    info = get_device_info()
    fp8_available = _fp8_kv_cache_tuning_available(info.arch)
    try:
        kv_cache_dtypes = _requested_kv_cache_dtypes(
            args.kv_cache_dtype, fp8_available=fp8_available
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.kv_cache_dtype == "both" and not fp8_available:
        print(
            f"SKIP FP8 KV cache tuning on unsupported architecture {info.arch}; "
            "FP8 is enabled only on gfx938 and gfx92a.",
            flush=True,
        )
    seen_keys: set[tuple] = set()
    tuned_count = 0
    duplicate_count = 0
    skipped_dtype_count = 0
    for kv_cache_dtype_name in kv_cache_dtypes:
        for case in selected:
            if kv_cache_dtype_name not in case.kv_cache_dtypes:
                skipped_dtype_count += 1
                continue
            result = tune_case(
                case,
                seed=args.seed,
                warmup_ms=args.warmup_ms,
                repeat_ms=args.rep_ms,
                quick=args.quick,
                output_dir=args.output_dir,
                kv_cache_dtype_name=kv_cache_dtype_name,
                seen_keys=seen_keys,
            )
            if result is None:
                duplicate_count += 1
            else:
                tuned_count += 1
    print(
        f"SUMMARY tuned={tuned_count} duplicate_keys={duplicate_count} "
        f"inapplicable_dtype_jobs={skipped_dtype_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
