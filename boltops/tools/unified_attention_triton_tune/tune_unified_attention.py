#!/usr/bin/env python3
"""Offline tuner for the aiter-compatible UnifiedAttention core kernels."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import triton
import yaml

from boltops.utility.arch_info import get_device_info
from boltops.utility.triton_capability import (
    get_triton_capabilities,
    get_triton_config_dir,
)

from boltops.fa.unified_attention.triton import config as config_module


CASE_FILE = Path(__file__).with_name("benchmark_cases.yaml")
DTYPES = {"bfloat16": torch.bfloat16}
ATTENTION_CONFIG_KEYS = ("BLOCK_M", "TILE_SIZE", "num_warps", "num_stages")


@dataclass(frozen=True)
class TuneCase:
    name: str
    seq_lens: tuple[tuple[int, int], ...]
    query_heads: int
    kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    dtype: torch.dtype
    sliding_window: int | None
    soft_cap: float | None
    use_sinks: bool
    num_segments: int
    seq_threshold_3d: int
    enabled: bool

    @property
    def kind(self) -> str:
        use_3d = (
            max(q_len for q_len, _ in self.seq_lens) == 1
            and len(self.seq_lens) <= self.seq_threshold_3d
        )
        return config_module.KIND_3D if use_3d else config_module.KIND_2D

    @property
    def num_queries_per_kv(self) -> int:
        return self.query_heads // self.kv_heads


def load_cases(path: Path = CASE_FILE) -> tuple[TuneCase, ...]:
    """Load and validate the single YAML source of tuning shapes."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: 'cases' must be a non-empty list")
    required = {
        "name", "seq_lens", "query_heads", "kv_heads", "head_size",
        "block_size", "num_blocks", "dtype", "sliding_window", "soft_cap",
        "use_sinks", "num_segments", "seq_threshold_3d", "enabled",
    }
    cases = []
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: case {index} must be an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}: case {index} missing {sorted(missing)}")
        name = str(row["name"])
        if not name or name in names:
            raise ValueError(f"{path}: duplicate or empty case name {name!r}")
        names.add(name)
        dtype_name = str(row["dtype"])
        if dtype_name not in DTYPES:
            raise ValueError(f"{path}: {name} has unsupported dtype {dtype_name!r}")
        seq_lens = tuple(tuple(map(int, pair)) for pair in row["seq_lens"])
        if not seq_lens or any(len(pair) != 2 for pair in seq_lens):
            raise ValueError(f"{path}: {name} has invalid seq_lens")
        if any(q_len <= 0 or kv_len < q_len for q_len, kv_len in seq_lens):
            raise ValueError(f"{path}: {name} requires 0 < query_len <= kv_len")
        query_heads = int(row["query_heads"])
        kv_heads = int(row["kv_heads"])
        if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
            raise ValueError(f"{path}: {name} has incompatible head counts")
        block_size = int(row["block_size"])
        num_blocks = int(row["num_blocks"])
        required_blocks = len(seq_lens) * math.ceil(
            max(kv_len for _, kv_len in seq_lens) / block_size
        )
        if block_size <= 0 or num_blocks < required_blocks:
            raise ValueError(f"{path}: {name} has insufficient cache blocks")
        sliding_window = row["sliding_window"]
        if sliding_window is not None and int(sliding_window) <= 0:
            raise ValueError(f"{path}: {name} sliding_window must be positive")
        case = TuneCase(
            name=name,
            seq_lens=seq_lens,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_size=int(row["head_size"]),
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=DTYPES[dtype_name],
            sliding_window=(None if sliding_window is None else int(sliding_window)),
            soft_cap=(None if row["soft_cap"] is None else float(row["soft_cap"])),
            use_sinks=bool(row["use_sinks"]),
            num_segments=int(row["num_segments"]),
            seq_threshold_3d=int(row["seq_threshold_3d"]),
            enabled=bool(row["enabled"]),
        )
        if case.head_size <= 0 or case.num_segments <= 0 or case.seq_threshold_3d < 0:
            raise ValueError(f"{path}: {name} has non-positive dimensions")
        cases.append(case)
    return tuple(cases)


def candidate_configs(case: TuneCase, *, quick: bool = False) -> tuple[dict, ...]:
    """Return the kernel-family search space after deterministic pruning."""
    if quick:
        return (dict(config_module.DEFAULT_CONFIGS[case.kind]),)
    if case.kind == config_module.KIND_2D:
        block_m_values = (8, 16, 32, 64)
        tile_size_values = (8, 16, 32, 64, 128)
        num_warps_values = (1, 2, 4, 8)
    else:
        # Keep decode attention aligned with aiter; 3D expands by jointly
        # searching every valid attention config with all reduce configs.
        block_m_values = (16, 32, 64)
        tile_size_values = (16, 32, 64)
        num_warps_values = (4, 8)
    candidates = []
    for block_m in block_m_values:
        for tile_size in tile_size_values:
            for num_warps in num_warps_values:
                for num_stages in (1, 2):
                    if block_m < case.num_queries_per_kv:
                        continue
                    if case.head_size > 128 and block_m >= 64:
                        continue
                    if (
                        case.kind == config_module.KIND_2D
                        and case.head_size > 128
                        and tile_size >= 128
                        and num_stages > 1
                    ):
                        # A 256-wide padded head with a 128-wide double-buffered
                        # KV tile requires 128 KiB, beyond gfx938's 64 KiB LDS.
                        continue
                    candidate = {
                        "BLOCK_M": block_m,
                        "TILE_SIZE": tile_size,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                    if case.kind == config_module.KIND_3D:
                        candidate.update(
                            reduce_num_warps=1,
                            reduce_num_stages=1,
                        )
                    candidates.append(candidate)
    defaults = dict(config_module.DEFAULT_CONFIGS[case.kind])
    tokens = {json.dumps(candidate, sort_keys=True) for candidate in candidates}
    if json.dumps(defaults, sort_keys=True) not in tokens:
        candidates.insert(0, defaults)
    return tuple(candidates)


def _make_inputs(case: TuneCase) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    query_lens = [q_len for q_len, _ in case.seq_lens]
    kv_lens = [kv_len for _, kv_len in case.seq_lens]
    blocks_per_seq = math.ceil(max(kv_lens) / case.block_size)
    q = torch.randn(
        sum(query_lens), case.query_heads, case.head_size,
        device=device, dtype=case.dtype,
    )
    k = torch.randn(
        case.num_blocks, case.block_size, case.kv_heads, case.head_size,
        device=device, dtype=case.dtype,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device, dtype=torch.int32,
    )
    seqused_k = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    block_table = torch.randperm(case.num_blocks, device=device)[:
        len(case.seq_lens) * blocks_per_seq
    ].reshape(len(case.seq_lens), blocks_per_seq).int()
    sinks = (
        torch.randn(case.query_heads, device=device, dtype=case.dtype)
        if case.use_sinks else None
    )
    padded_head = triton.next_power_of_2(case.head_size)
    partial = torch.empty(
        case.seq_threshold_3d, case.query_heads, case.num_segments, padded_head,
        device=device, dtype=torch.float32,
    )
    partial_max = torch.empty(
        case.seq_threshold_3d, case.query_heads, case.num_segments,
        device=device, dtype=torch.float32,
    )
    return {
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "sinks": sinks,
        "partial": partial,
        "partial_max": partial_max,
        "partial_sum": torch.empty_like(partial_max),
    }


def _reference(case: TuneCase, data: dict) -> torch.Tensor:
    output = torch.empty_like(data["q"])
    cursor = 0
    scale = case.head_size ** -0.5
    for seq, (query_len, kv_len) in enumerate(case.seq_lens):
        blocks = data["block_table"][seq, : math.ceil(kv_len / case.block_size)].long()
        key = data["k"][blocks].reshape(-1, case.kv_heads, case.head_size)[:kv_len]
        value = data["v"][blocks].reshape(-1, case.kv_heads, case.head_size)[:kv_len]
        query = data["q"][cursor : cursor + query_len]
        repeat = case.num_queries_per_kv
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
        positions = torch.arange(query_len, device=query.device) + kv_len - query_len
        keys = torch.arange(kv_len, device=query.device)
        mask = keys[None, None, :] > positions[None, :, None]
        if case.sliding_window is not None:
            mask |= (positions[None, :, None] - keys[None, None, :]) >= case.sliding_window
        if case.soft_cap is not None and case.soft_cap > 0:
            scores = case.soft_cap * torch.tanh(scores / case.soft_cap)
        scores.masked_fill_(mask, float("-inf"))
        if data["sinks"] is not None:
            sink_logits = data["sinks"].float()[:, None, None].expand(-1, query_len, 1)
            scores = torch.cat((scores, sink_logits), dim=-1)
        probabilities = torch.softmax(scores, dim=-1)[..., :kv_len]
        output[cursor : cursor + query_len] = torch.einsum(
            "hqk,khd->qhd", probabilities.to(value.dtype), value
        )
        cursor += query_len
    return output


def _backend_module():
    return importlib.import_module(
        "boltops.fa.unified_attention.triton.unified_attention"
    )


@contextmanager
def _config_override(case: TuneCase, candidate: dict) -> Iterator[None]:
    backend = _backend_module()
    attribute = (
        "_get_unified_attention_3d_config"
        if case.kind == config_module.KIND_3D
        else "_get_unified_attention_2d_config"
    )
    original = getattr(backend, attribute)
    setattr(backend, attribute, lambda **_: dict(candidate))
    try:
        yield
    finally:
        setattr(backend, attribute, original)


def _launch(case: TuneCase, data: dict, candidate: dict | None = None) -> None:
    backend = _backend_module()

    def run() -> None:
        backend.unified_attention(
            data["q"], data["k"], data["v"], data["out"],
            data["cu_seqlens_q"], max(data["query_lens"]), data["seqused_k"],
            max(data["kv_lens"]), case.head_size ** -0.5, True,
            ((case.sliding_window - 1, 0) if case.sliding_window else (-1, -1)),
            data["block_table"], case.soft_cap or 0.0, None, None, None,
            case.seq_threshold_3d, case.num_segments, data["partial"],
            data["partial_max"], data["partial_sum"], sinks=data["sinks"],
        )

    if candidate is None:
        run()
    else:
        with _config_override(case, candidate):
            run()


def _bench(
    case: TuneCase,
    data: dict,
    candidate: dict,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    return float(
        triton.testing.do_bench(
            lambda: _launch(case, data, candidate),
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
    )


def _check_output(case: TuneCase, data: dict, reference: torch.Tensor) -> None:
    if not torch.isfinite(data["out"]).all():
        raise AssertionError("candidate produced non-finite output")
    torch.testing.assert_close(data["out"], reference, atol=2e-2, rtol=2e-2)


def _rank_candidates(
    case: TuneCase,
    data: dict,
    reference: torch.Tensor,
    candidates: tuple[dict, ...],
    *,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[tuple[dict, float], ...]:
    ranked = []
    for index, candidate in enumerate(candidates, 1):
        try:
            _launch(case, data, candidate)
            _check_output(case, data, reference)
            elapsed = _bench(case, data, candidate, warmup_ms, rep_ms)
        except Exception as exc:
            print(f"[{case.name} {index}/{len(candidates)}] ERROR {candidate}: {exc}")
            continue
        print(f"[{case.name} {index}/{len(candidates)}] {elapsed:.6f} ms {candidate}")
        ranked.append((dict(candidate), elapsed))
    if not ranked:
        raise RuntimeError(f"No valid candidate for {case.name}")
    return tuple(sorted(ranked, key=lambda item: item[1]))


def _select_best(
    case: TuneCase,
    data: dict,
    reference: torch.Tensor,
    candidates: tuple[dict, ...],
    *,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    return _rank_candidates(
        case, data, reference, candidates,
        warmup_ms=warmup_ms, rep_ms=rep_ms,
    )[0]


def _attention_config(candidate: dict) -> dict:
    return {key: candidate[key] for key in ATTENTION_CONFIG_KEYS}


def _tune_reduce(
    case: TuneCase,
    data: dict,
    reference: torch.Tensor,
    attention_configs: tuple[dict, ...],
    *,
    quick: bool,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[dict, float]:
    if case.kind != config_module.KIND_3D:
        attention_config = attention_configs[0]
        return attention_config, _bench(case, data, attention_config, warmup_ms, rep_ms)
    pairs = ((1, 1),) if quick else tuple(
        (warps, stages) for warps in (1, 2, 4, 8) for stages in (1, 2)
    )
    candidates = tuple(
        {
            **_attention_config(attention_config),
            "reduce_num_warps": warps,
            "reduce_num_stages": stages,
        }
        for attention_config in attention_configs
        for warps, stages in pairs
    )
    return _select_best(
        case, data, reference, candidates,
        warmup_ms=warmup_ms, rep_ms=rep_ms,
    )


def _filename(case: TuneCase) -> str:
    info = get_device_info()
    return config_module.make_config_filename(
        case.kind,
        device_label=f"{info.arch}_cu{info.cu_count}",
        cache_block_size=case.block_size,
        head_size=case.head_size,
        sliding_window=case.sliding_window or 0,
        use_alibi_slopes=False,
        use_qq_bias=False,
        use_softcap=bool(case.soft_cap and case.soft_cap > 0),
        use_sinks=case.use_sinks,
        use_mm_prefix=False,
        use_fp8_output=False,
        num_segments_per_seq=(
            case.num_segments if case.kind == config_module.KIND_3D else None
        ),
        kv_dtype=case.dtype,
    )


def _config_key(case: TuneCase) -> tuple[int, int, int, int, int]:
    return config_module.make_config_key(
        num_seqs=len(case.seq_lens),
        num_query_tokens=sum(q_len for q_len, _ in case.seq_lens),
        max_seqlen_q=max(q_len for q_len, _ in case.seq_lens),
        max_seqlen_k=max(kv_len for _, kv_len in case.seq_lens),
        num_queries_per_kv=case.num_queries_per_kv,
    )


def write_config(case: TuneCase, config: dict, output_root: Path) -> Path:
    directory = get_triton_config_dir(output_root, get_triton_capabilities())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(case)
    payload = {"key": list(config_module.CONFIG_KEY_FIELDS), "config": {}}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"refusing to overwrite invalid config {path}: {exc}") from exc
        if (
            not isinstance(existing, dict)
            or tuple(existing.get("key", ())) != config_module.CONFIG_KEY_FIELDS
            or not isinstance(existing.get("config"), dict)
        ):
            raise RuntimeError(f"refusing to overwrite invalid config {path}")
        payload["config"].update(existing["config"])
    payload["config"][str(_config_key(case))] = config
    payload["config"] = dict(sorted(payload["config"].items()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def verify_runtime_hit(
    case: TuneCase,
    data: dict,
    expected: dict,
    output_root: Path,
) -> None:
    original_root = config_module.CONFIG_ROOT
    config_module.CONFIG_ROOT = output_root
    config_module.clear_config_cache()
    try:
        kwargs = dict(
            cache_block_size=case.block_size,
            head_size=case.head_size,
            num_seqs=len(case.seq_lens),
            num_query_tokens=sum(data["query_lens"]),
            max_seqlen_q=max(data["query_lens"]),
            max_seqlen_k=max(data["kv_lens"]),
            num_queries_per_kv=case.num_queries_per_kv,
            sliding_window=case.sliding_window or 0,
            use_alibi_slopes=False,
            use_qq_bias=False,
            use_softcap=bool(case.soft_cap and case.soft_cap > 0),
            use_sinks=case.use_sinks,
            use_mm_prefix=False,
            kv_dtype=case.dtype,
        )
        if case.kind == config_module.KIND_2D:
            loaded = config_module.get_kernel_config(
                case.kind, use_fp8_output=False, **kwargs
            )
        else:
            loaded = config_module.get_kernel_config(
                case.kind, num_segments_per_seq=case.num_segments, **kwargs
            )
        if loaded != expected:
            raise AssertionError(f"runtime loaded {loaded}, expected {expected}")
        _launch(case, data)
    finally:
        config_module.CONFIG_ROOT = original_root
        config_module.clear_config_cache()


def _build_parser(cases: tuple[TuneCase, ...], default_case_file: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case-file", type=Path, default=default_case_file)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in cases],
        help="case to tune; repeatable; defaults to every enabled YAML case",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--quick", action="store_true", help="validate one default candidate")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=config_module.CONFIG_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--case-file", type=Path, default=CASE_FILE)
    pre_args, _ = pre_parser.parse_known_args(argv)
    cases = load_cases(pre_args.case_file)
    args = _build_parser(cases, pre_args.case_file).parse_args(argv)
    if args.list_cases:
        for case in cases:
            print(case.name)
        return 0
    if args.warmup_ms < 0 or args.rep_ms <= 0:
        raise ValueError("warmup-ms must be >= 0 and rep-ms must be > 0")
    selected = [
        case for case in cases
        if (case.name in args.case if args.case else case.enabled)
    ]
    identities = [(_filename(case), _config_key(case)) for case in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("selected UnifiedAttention cases contain duplicate config keys")
    for case in selected:
        data = _make_inputs(case)
        reference = _reference(case, data)
        defaults = dict(config_module.DEFAULT_CONFIGS[case.kind])
        _launch(case, data, defaults)
        _check_output(case, data, reference)
        baseline_ms = _bench(case, data, defaults, args.warmup_ms, args.rep_ms)
        candidates = candidate_configs(case, quick=args.quick)
        if case.kind == config_module.KIND_3D:
            best, best_ms = _tune_reduce(
                case, data, reference, candidates,
                quick=args.quick,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
        else:
            best, best_ms = _select_best(
                case, data, reference, candidates,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
        path = write_config(case, best, args.output_root)
        verify_runtime_hit(case, data, best, args.output_root)
        print(
            f"[{case.name}] baseline={baseline_ms:.6f} ms, best={best_ms:.6f} ms, "
            f"speedup={baseline_ms / best_ms:.3f}x, json={path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
