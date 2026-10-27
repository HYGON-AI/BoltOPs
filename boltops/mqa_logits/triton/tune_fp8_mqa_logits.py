#!/usr/bin/env python3
"""Autotune BoltOPs Triton ``fp8_mqa_logits`` launch configs on Hygon HCU.

Sweeps main-kernel configs (aiter-style + optional matrix_instr_nonkdim) and,
on the fast path, clear-kernel configs. Times the full host API with CUDA
events and writes a JSON summary under the capability config dir
(``configs/triton_3_2`` / ``triton_3_6`` / ``triton_3_8``).

Example::

    cd BoltOPs
    HIP_VISIBLE_DEVICES=7 python3 boltops/mqa_logits/triton/tune_fp8_mqa_logits.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits import (
    DEFAULT_BLOCK_KV,
    DEFAULT_CLEAR_BLOCK_KV,
    DEFAULT_CLEAR_NUM_STAGES,
    DEFAULT_CLEAR_NUM_WARPS,
    DEFAULT_CLEAR_WAVES_PER_EU,
    DEFAULT_MATRIX_INSTR_NONKDIM,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
    DEFAULT_WAVES_PER_EU,
    fp8_mqa_logits,
)
from boltops.utility import arch_info
from boltops.utility.arch_info import get_fp8_e4m3_dtype
from boltops.utility.triton_capability import (
    get_triton_capabilities,
    get_triton_config_dir,
)

DEVICE = "cuda"
e4m3_type = get_fp8_e4m3_dtype()
fp8_max = torch.finfo(e4m3_type).max

# Shapes: (s_q, s_k, H, D, use_cp, label)
# Causal H=64 D=128 plus one slow-path mid (s_k % 64 != 0 for default BLOCK_KV).
TUNE_SHAPES = [
    (128, 1024, 64, 128, False, "mid-fast-causal"),
    (128, 4096, 64, 128, False, "long-fast-causal"),
    (1024, 8192, 64, 128, False, "xl-fast-causal"),
    (256, 4109, 64, 128, False, "long-slow-causal"),
    (128, 1041, 64, 128, False, "mid-slow-causal"),
]

MAIN_BLOCK_KVS = (64, 128, 256)
MAIN_NUM_STAGES = (1, 2)
MAIN_NUM_WARPS = (4, 8)
MAIN_WAVES = (0, 2)
MAIN_MATRIX_INSTR = (16,)  # 32 fails to compile on current HCU Triton
MAIN_KPACKS = (1, 2)
MAIN_SCHEDULE_HINTS = ("none", "local-prefetch")

CLEAR_BLOCK_KVS = (64, 128, 256, 512, 1024)
CLEAR_NUM_STAGES = (1, 2)
CLEAR_NUM_WARPS = (4, 8)
CLEAR_WAVES = (0, 2)
CLEAR_KPACKS = (1, 2)
CLEAR_SCHEDULE_HINTS = ("none", "local-prefetch")

_CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"
_CONFIG_DIR = get_triton_config_dir(_CONFIGS_ROOT, get_triton_capabilities())
OUT_JSON = _CONFIG_DIR / "fp8_mqa_logits_tune.json"
SHIP_JSON = _CONFIG_DIR / "fp8_mqa_logits,arch={arch}.json"


def _schedule_hints() -> tuple[str, ...]:
    if get_triton_capabilities().supports("schedule_hint"):
        return MAIN_SCHEDULE_HINTS
    return ("none",)


@dataclass(frozen=True)
class MainConfig:
    block_kv: int
    num_warps: int
    num_stages: int
    waves_per_eu: int
    matrix_instr_nonkdim: int
    kpack: int
    schedule_hint: str

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "block_kv": self.block_kv,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "waves_per_eu": self.waves_per_eu,
            "matrix_instr_nonkdim": self.matrix_instr_nonkdim,
            "kpack": self.kpack,
            "schedule_hint": self.schedule_hint,
            "use_block_pingpong": False,
        }


@dataclass(frozen=True)
class ClearConfig:
    clear_block_kv: int
    clear_num_warps: int
    clear_num_stages: int
    clear_waves_per_eu: int
    clear_kpack: int
    clear_schedule_hint: str

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "clear_block_kv": self.clear_block_kv,
            "clear_num_warps": self.clear_num_warps,
            "clear_num_stages": self.clear_num_stages,
            "clear_waves_per_eu": self.clear_waves_per_eu,
            "clear_kpack": self.clear_kpack,
            "clear_schedule_hint": self.clear_schedule_hint,
        }


DEFAULT_MAIN = MainConfig(
    block_kv=DEFAULT_BLOCK_KV,
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    matrix_instr_nonkdim=DEFAULT_MATRIX_INSTR_NONKDIM,
    kpack=2,
    schedule_hint="none",
)
DEFAULT_CLEAR = ClearConfig(
    clear_block_kv=DEFAULT_CLEAR_BLOCK_KV,
    clear_num_warps=DEFAULT_CLEAR_NUM_WARPS,
    clear_num_stages=DEFAULT_CLEAR_NUM_STAGES,
    clear_waves_per_eu=DEFAULT_CLEAR_WAVES_PER_EU,
    clear_kpack=2,
    clear_schedule_hint="none",
)


def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_amax = x.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    x_scaled = (x * (1.0 / sf)).to(e4m3_type)
    return x_scaled, sf.squeeze(1)


def make_inputs(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    use_cp: bool,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(s_q, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device=DEVICE, dtype=torch.bfloat16)
    kv_fp8, scales = per_token_cast_to_fp8(kv)
    weights = torch.randn(s_q, num_heads, device=DEVICE, dtype=torch.float32)
    if use_cp:
        assert s_k % s_q == 0 and s_q % 2 == 0
        chunk_size = s_q // 2
        cp_size = s_k // s_q
        cp_id = cp_size // 3
        ks = torch.zeros(s_q, dtype=torch.int32, device=DEVICE)
        ke = torch.zeros(s_q, dtype=torch.int32, device=DEVICE)
        for i in range(chunk_size):
            ke[i] = cp_id * chunk_size + i
            ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    else:
        ks = torch.zeros(s_q, dtype=torch.int32, device=DEVICE)
        ke = torch.arange(s_q, dtype=torch.int32, device=DEVICE) + (s_k - s_q)
    q_fp8 = q.to(e4m3_type)
    return q_fp8, kv_fp8, scales, weights, ks, ke


def useful_flops(
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> float:
    starts = cu_starts.to("cpu").numpy()
    ends = cu_ends.to("cpu").numpy()
    total = 0.0
    for start, end in zip(starts, ends):
        total += 2.0 * num_heads * head_dim * max(int(end) - int(start), 0)
    return total


def time_ms(fn: Callable[[], None], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def iter_main_configs(matrix_instr: Iterable[int]) -> list[MainConfig]:
    configs: list[MainConfig] = []
    for block_kv, stages, warps, waves, mink, kpack, hint in itertools.product(
        MAIN_BLOCK_KVS,
        MAIN_NUM_STAGES,
        MAIN_NUM_WARPS,
        MAIN_WAVES,
        matrix_instr,
        MAIN_KPACKS,
        _schedule_hints(),
    ):
        configs.append(
            MainConfig(
                block_kv=block_kv,
                num_warps=warps,
                num_stages=stages,
                waves_per_eu=waves,
                matrix_instr_nonkdim=mink,
                kpack=kpack,
                schedule_hint=hint,
            )
        )
    return configs


def iter_clear_configs() -> list[ClearConfig]:
    configs: list[ClearConfig] = []
    for block_kv, stages, warps, waves, kpack, hint in itertools.product(
        CLEAR_BLOCK_KVS,
        CLEAR_NUM_STAGES,
        CLEAR_NUM_WARPS,
        CLEAR_WAVES,
        CLEAR_KPACKS,
        _schedule_hints(),
    ):
        configs.append(
            ClearConfig(
                clear_block_kv=block_kv,
                clear_num_warps=warps,
                clear_num_stages=stages,
                clear_waves_per_eu=waves,
                clear_kpack=kpack,
                clear_schedule_hint=hint,
            )
        )
    return configs


def path_for(s_k: int, block_kv: int) -> str:
    return "slow" if (s_k % block_kv != 0) else "fast"


def bench_one(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    main: MainConfig,
    clear: ClearConfig,
    warmup: int,
    repeat: int,
) -> float:
    kwargs = {**main.as_kwargs(), **clear.as_kwargs()}

    def _run() -> None:
        fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, auto_bn_tile=False, **kwargs
        )

    # Compile / first-touch outside timed region.
    _run()
    torch.cuda.synchronize()
    return time_ms(_run, warmup=warmup, repeat=repeat)


def tune_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    use_cp: bool,
    label: str,
    main_configs: list[MainConfig],
    clear_configs: list[ClearConfig],
    warmup: int,
    repeat: int,
    skip_clear: bool,
) -> dict[str, Any]:
    q_fp8, kv_fp8, scales, weights, ks, ke = make_inputs(
        s_q, s_k, num_heads, head_dim, use_cp
    )
    flops = useful_flops(ks, ke, num_heads, head_dim)

    print(
        f"\n=== shape {label}: s_q={s_q} s_k={s_k} H={num_heads} D={head_dim} "
        f"({len(main_configs)} main configs) ===",
        flush=True,
    )

    best_main: Optional[MainConfig] = None
    best_main_ms = float("inf")
    main_failures = 0
    main_rows: list[dict[str, Any]] = []

    for i, main in enumerate(main_configs, 1):
        p = path_for(s_k, main.block_kv)
        try:
            ms = bench_one(
                q_fp8,
                kv_fp8,
                scales,
                weights,
                ks,
                ke,
                main,
                DEFAULT_CLEAR,
                warmup,
                repeat,
            )
        except Exception as exc:  # noqa: BLE001 — skip bad configs
            main_failures += 1
            print(
                f"  [main {i}/{len(main_configs)}] FAIL {asdict(main)} path={p}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        tflops = (flops / (ms * 1e-3)) / 1e12 if ms > 0 else 0.0
        main_rows.append({**asdict(main), "path": p, "ms": ms, "tflops": tflops})
        marker = ""
        if ms < best_main_ms:
            best_main_ms = ms
            best_main = main
            marker = " *"
        print(
            f"  [main {i}/{len(main_configs)}] {asdict(main)} path={p} "
            f"-> {ms:.4f} ms, {tflops:.2f} TFLOPS{marker}",
            flush=True,
        )

    if best_main is None:
        raise RuntimeError(f"No valid main config for shape {label}")

    best_clear = DEFAULT_CLEAR
    best_clear_ms = best_main_ms
    clear_rows: list[dict[str, Any]] = []
    clear_failures = 0

    # Clear kernel only runs on the fast path for the chosen BLOCK_KV.
    if not skip_clear and path_for(s_k, best_main.block_kv) == "fast":
        print(
            f"  --- clear sweep with best main {asdict(best_main)} "
            f"({len(clear_configs)} configs) ---",
            flush=True,
        )
        for i, clear in enumerate(clear_configs, 1):
            try:
                ms = bench_one(
                    q_fp8,
                    kv_fp8,
                    scales,
                    weights,
                    ks,
                    ke,
                    best_main,
                    clear,
                    warmup,
                    repeat,
                )
            except Exception as exc:  # noqa: BLE001
                clear_failures += 1
                print(
                    f"  [clear {i}/{len(clear_configs)}] FAIL {asdict(clear)}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            tflops = (flops / (ms * 1e-3)) / 1e12 if ms > 0 else 0.0
            clear_rows.append({**asdict(clear), "ms": ms, "tflops": tflops})
            marker = ""
            if ms < best_clear_ms:
                best_clear_ms = ms
                best_clear = clear
                marker = " *"
            print(
                f"  [clear {i}/{len(clear_configs)}] {asdict(clear)} "
                f"-> {ms:.4f} ms, {tflops:.2f} TFLOPS{marker}",
                flush=True,
            )

    # Baseline with current fixed defaults for reference.
    try:
        baseline_ms = bench_one(
            q_fp8,
            kv_fp8,
            scales,
            weights,
            ks,
            ke,
            DEFAULT_MAIN,
            DEFAULT_CLEAR,
            warmup,
            repeat,
        )
    except Exception:  # noqa: BLE001
        baseline_ms = None
        traceback.print_exc()

    best_tflops = (flops / (best_clear_ms * 1e-3)) / 1e12 if best_clear_ms > 0 else 0.0
    speedup = (
        (baseline_ms / best_clear_ms) if baseline_ms and best_clear_ms > 0 else None
    )

    result = {
        "label": label,
        "s_q": s_q,
        "s_k": s_k,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "use_cp": use_cp,
        "path_at_best": path_for(s_k, best_main.block_kv),
        "baseline_ms": baseline_ms,
        "best_ms": best_clear_ms,
        "best_tflops": best_tflops,
        "speedup_vs_default": speedup,
        "best_main": asdict(best_main),
        "best_clear": asdict(best_clear),
        "main_failures": main_failures,
        "clear_failures": clear_failures,
        "top5_main": sorted(main_rows, key=lambda r: r["ms"])[:5],
        "top5_clear": sorted(clear_rows, key=lambda r: r["ms"])[:5] if clear_rows else [],
    }
    print(
        f"  BEST {label}: {best_clear_ms:.4f} ms ({best_tflops:.2f} TFLOPS) "
        f"main={asdict(best_main)} clear={asdict(best_clear)} "
        f"vs default {baseline_ms:.4f} ms"
        if baseline_ms is not None
        else f"  BEST {label}: {best_clear_ms:.4f} ms main={asdict(best_main)}",
        flush=True,
    )
    return result


def recommend_default(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a single default: prefer the XL fast shape, else majority vote on main."""
    by_label = {r["label"]: r for r in results}
    if "xl-fast-causal" in by_label:
        pick = by_label["xl-fast-causal"]
        reason = "xl-fast-causal (largest fast-path shape)"
    elif "long-fast-causal" in by_label:
        pick = by_label["long-fast-causal"]
        reason = "long-fast-causal"
    else:
        pick = min(results, key=lambda r: r["best_ms"])
        reason = f"lowest latency among results ({pick['label']})"

    return {
        "reason": reason,
        "main": pick["best_main"],
        "clear": pick["best_clear"],
        "reference_shape": pick["label"],
        "reference_ms": pick["best_ms"],
    }


def print_summary(results: list[dict[str, Any]], recommended: dict[str, Any]) -> None:
    print("\n========== SUMMARY ==========")
    hdr = (
        f"{'label':22s} {'s_q':>5} {'s_k':>5} {'path':5s} "
        f"{'ms':>8} {'TFLOPS':>8} {'vs_def':>7}  best_main / clear"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        spd = r["speedup_vs_default"]
        spd_s = f"{spd:.2f}x" if spd is not None else "n/a"
        m = r["best_main"]
        c = r["best_clear"]
        main_s = (
            f"BK={m['block_kv']} w={m['num_warps']} s={m['num_stages']} "
            f"waves={m['waves_per_eu']} mink={m['matrix_instr_nonkdim']} "
            f"kpack={m.get('kpack')} hint={m.get('schedule_hint')}"
        )
        clear_s = (
            f"cBK={c['clear_block_kv']} w={c['clear_num_warps']} "
            f"s={c['clear_num_stages']} waves={c['clear_waves_per_eu']} "
            f"kpack={c.get('clear_kpack')} hint={c.get('clear_schedule_hint')}"
        )
        print(
            f"{r['label']:22s} {r['s_q']:5d} {r['s_k']:5d} {r['path_at_best']:5s} "
            f"{r['best_ms']:8.4f} {r['best_tflops']:8.2f} {spd_s:>7}  "
            f"{main_s} | {clear_s}"
        )
    print("\nRecommended default:", json.dumps(recommended, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument(
        "--matrix-instr-32",
        action="store_true",
        help="Also sweep matrix_instr_nonkdim=32 (usually fails to compile).",
    )
    p.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip clear-kernel sweep (main only, default clear).",
    )
    p.add_argument(
        "--write-ship",
        action="store_true",
        help="Also write arch-specific ship JSON from recommended + buckets.",
    )
    p.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="Optional subset of shape labels to tune.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_JSON,
        help=f"JSON output path (default: {OUT_JSON})",
    )
    return p.parse_args()


def build_ship_payload(results: list[dict[str, Any]], recommended: dict[str, Any], arch: str) -> dict[str, Any]:
    """Compress tune results into the shipped by_seq_kv JSON format."""
    # Prefer causal fast shapes for buckets; fall back to any result.
    by_label = {r["label"]: r for r in results}
    buckets = []
    for label, seq_kv_leq in (
        ("mid-fast-causal", 2048),
        ("long-fast-causal", 4096),
        ("xl-fast-causal", 16384),
    ):
        r = by_label.get(label)
        if r is None:
            continue
        buckets.append(
            {
                "seq_kv_leq": seq_kv_leq,
                "label": label,
                "main": r["best_main"],
                "clear": r["best_clear"],
            }
        )
    return {
        "meta": {
            "arch": arch,
            "source": "fp8_mqa_logits_tune.json",
            "note": "Buckets by seq_len_kv (LEQ); includes kpack/schedule_hint.",
        },
        "default": {
            "main": recommended["main"],
            "clear": recommended["clear"],
        },
        "by_seq_kv": buckets,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")

    matrix_instr = (16, 32) if args.matrix_instr_32 else MAIN_MATRIX_INSTR
    main_configs = iter_main_configs(matrix_instr)
    clear_configs = iter_clear_configs()

    shapes = TUNE_SHAPES
    if args.shapes:
        wanted = set(args.shapes)
        shapes = [s for s in TUNE_SHAPES if s[5] in wanted]
        if not shapes:
            raise SystemExit(f"No shapes matched {wanted}")

    arch = arch_info.get_arch()
    caps = get_triton_capabilities()
    print(
        f"device={torch.cuda.get_device_name(0)} arch={arch} "
        f"triton={caps.package_version} config_dir={caps.config_dir_name} "
        f"warmup={args.warmup} repeat={args.repeat} "
        f"main_configs={len(main_configs)} clear_configs={len(clear_configs)} "
        f"shapes={len(shapes)}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for s_q, s_k, H, D, use_cp, label in shapes:
        results.append(
            tune_shape(
                s_q,
                s_k,
                H,
                D,
                use_cp,
                label,
                main_configs,
                clear_configs,
                args.warmup,
                args.repeat,
                args.skip_clear,
            )
        )

    recommended = recommend_default(results)
    print_summary(results, recommended)

    payload = {
        "meta": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "device": torch.cuda.get_device_name(0),
            "arch": arch,
            "main_search": {
                "BLOCK_KV": list(MAIN_BLOCK_KVS),
                "num_stages": list(MAIN_NUM_STAGES),
                "num_warps": list(MAIN_NUM_WARPS),
                "waves_per_eu": list(MAIN_WAVES),
                "matrix_instr_nonkdim": list(matrix_instr),
                "kpack": list(MAIN_KPACKS),
                "schedule_hint": list(_schedule_hints()),
            },
            "clear_search": {
                "BLOCK_KV": list(CLEAR_BLOCK_KVS),
                "num_stages": list(CLEAR_NUM_STAGES),
                "num_warps": list(CLEAR_NUM_WARPS),
                "waves_per_eu": list(CLEAR_WAVES),
                "kpack": list(CLEAR_KPACKS),
                "schedule_hint": list(_schedule_hints()),
            },
            "default_before_tune": {
                "main": asdict(DEFAULT_MAIN),
                "clear": asdict(DEFAULT_CLEAR),
            },
        },
        "results": results,
        "recommended_default": recommended,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out}", flush=True)

    if args.write_ship:
        ship_path = Path(str(SHIP_JSON).format(arch=arch))
        ship_path.write_text(
            json.dumps(build_ship_payload(results, recommended, arch), indent=2) + "\n"
        )
        print(f"Wrote ship config {ship_path}", flush=True)


if __name__ == "__main__":
    main()
