# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Read measured configuration contracts; never infer support from timing alone."""
import json
from functools import lru_cache
from pathlib import Path

from .registry import KernelConfig


_CONFIG_DIR = Path(__file__).with_name("configs")


@lru_cache(maxsize=5)
def _entries(interface):
    if interface not in ("scaled", "dense", "batched", "grouped", "batched_a16wfp4_online_quant"):
        raise ValueError(f"unknown GEMM configuration interface: {interface}")
    name = f"gemm_{interface}"
    data = json.loads((_CONFIG_DIR / f"{name}.json").read_text())
    if data["schema_version"] != 3:
        raise ValueError("unsupported GEMM performance-map schema")
    if data["interface"] != name:
        raise ValueError(f"configuration file must belong to {name}")
    return data["entries"]


def tuned_configs(problem, implementation, *, backend=None):
    p = problem
    interface = p.interface
    entries = _entries(interface)
    if (p.kind != "dense" or not p.native_operands or
            min(p.m, p.n, p.k) <= 0):
        return ()
    configs = []
    for entry in entries:
        if backend is not None and entry["config"]["backend"] != backend:
            continue
        match = entry["match"]
        if p.has_bias != match.get("has_bias", False):
            continue
        if p.arch != match["arch"] or p.num_cus != match["num_cus"]:
            continue
        if p.a.strides != (p.k, 1):
            continue
        b_layout = match.get("b_layout", "k_contiguous")
        expected_b = (p.k, 1) if b_layout == "k_contiguous" else (1, p.n)
        if p.b.strides != expected_b:
            continue
        dtype_names = [str(t).removeprefix("torch.") for t in (p.a_dtype, p.b_dtype, p.out_dtype)]
        if dtype_names != match["dtypes"]:
            continue
        scales = [{"dtype": str(s.dtype).removeprefix("torch."),
                   "block": list(s.block), "encoding": s.encoding}
                  for s in (p.a_scale, p.b_scale) if s is not None]
        if scales != match["scales"]:
            continue
        c = match["shape_contract"]
        if any(getattr(p, dim) != c[dim] for dim in ("m", "n", "k") if dim in c):
            continue
        if (p.m % c.get("m_multiple", 1) or p.n % c.get("n_multiple", 1) or
                p.k % c.get("k_multiple", 1) or not c.get("k_min", 0) <= p.k <= c.get("k_max", p.k) or
                not p.a_aligned_16 or not p.b_aligned_16):
            continue
        # Tile geometry is a selected implementation parameter, never a
        # property used to match the caller's GEMM problem.
        config = entry["config"]
        if config["implementation"] == implementation:
            configs.append(KernelConfig(*config["block_size"], source_config=config.get("source_config")))
    return tuple(configs)
