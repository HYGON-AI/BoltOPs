#!/usr/bin/env python3
"""Quick launch-parameter sweep for ``fp8_mqa_logits_tiled`` (skill timing)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from boltops.mqa_logits.triton.fp8_mqa_logits_tiled import fp8_mqa_logits_tiled
from boltops.mqa_logits.triton.tune_fp8_mqa_logits import make_inputs
from boltops.utility import arch_info
from boltops.utility.triton_capability import get_triton_capabilities


def per_iter_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return sum(times) / len(times)


def sweep_shape(
    s_q: int,
    s_k: int,
    num_heads: int,
    *,
    warmup: int,
    repeat: int,
) -> dict:
    q, kv, scales, weights, ks, ke = make_inputs(s_q, s_k, num_heads, 128, False)

    block_ms = [16] if s_q < 256 else [16, 32]
    block_ns = [64, 128, 256]
    warps = [4, 8]
    stages = [1, 2]
    waves = [0, 2, 4]
    kpacks = [1, 2]

    results: list[dict] = []
    best_ms = float("inf")
    best_cfg: dict = {}

    for bm, bn, nw, ns, wv, kp in itertools.product(
        block_ms, block_ns, warps, stages, waves, kpacks
    ):
        cfg = {
            "block_m": bm,
            "block_n": bn,
            "num_warps": nw,
            "num_stages": ns,
            "waves_per_eu": wv,
            "kpack": kp,
        }

        def run(cfg=cfg):
            fp8_mqa_logits_tiled(q, kv, scales, weights, ks, ke, **cfg)

        try:
            ms = per_iter_ms(run, warmup, repeat)
        except Exception as exc:  # noqa: BLE001
            results.append({**cfg, "ms": None, "error": str(exc)})
            continue

        row = {**cfg, "ms": ms}
        results.append(row)
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    results.sort(key=lambda r: r.get("ms") if r.get("ms") is not None else 1e9)
    return {
        "s_q": s_q,
        "s_k": s_k,
        "num_heads": num_heads,
        "best_ms": best_ms,
        "best_config": best_cfg,
        "top5": results[:5],
        "all": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    caps = get_triton_capabilities()
    arch = arch_info.get_arch()

    shapes = [
        (128, 4096, 64),
        (256, 4109, 64),
        (1024, 8192, 64),
    ]

    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(args.gpu),
            "arch": arch,
            "triton": caps.package_version,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "timing": "skill_per_iter_cuda_event",
        },
        "results": [],
    }

    for s_q, s_k, h in shapes:
        print(f"Sweep {s_q}×{s_k} H={h} ...", flush=True)
        row = sweep_shape(s_q, s_k, h, warmup=args.warmup, repeat=args.repeat)
        payload["results"].append(row)
        print(
            f"  best {row['best_ms']:.4f} ms  cfg={row['best_config']}",
            flush=True,
        )

    cap_dir = f"triton_{caps.version.replace('.', '_')}"
    out = (
        Path(__file__).resolve().parent
        / "configs"
        / cap_dir
        / f"fp8_mqa_logits_tiled_tune_{arch}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
