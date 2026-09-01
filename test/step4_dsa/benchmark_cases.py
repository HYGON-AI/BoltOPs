"""Concrete Step-4 DSA benchmark cases used by the operator handoff.

"TP8" means one production TP8 rank: the benchmark itself is single-process and does not
launch eight workers.  Exact 32K/64K entries are derived stress targets, not retained service
measurements, and are hidden behind an explicit CLI opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boltops.step4_dsa.geometry import STEP4, Step4Geometry

BENCHMARK_SCHEMA = "step4-dsa-benchmark-v1"
METADATA_WIDTH = STEP4.topk + 1
RECORDED_SMOKE_TOKENS = 31_929


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    profile: str
    operator: str
    params: dict[str, Any]
    description: str
    provenance: str = "production_geometry"
    measurement_claim: bool = True
    opt_in: bool = False
    required_capability: str | None = None


def tp8_geometry() -> Step4Geometry:
    """Real TP8 per-rank geometry, including one four-head indexer provider group."""
    return Step4Geometry(
        num_attention_heads=8,
        num_attention_groups=1,
        indexer_num_heads=4,
        num_provider_groups=1,
    )


def tp1_geometry() -> Step4Geometry:
    """Whole-model geometry used for reduced correctness and layout validation."""
    return STEP4


def derived_context_geometry(context_tokens: int) -> dict[str, int]:
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    physical_regions = -(-context_tokens // STEP4.region_block_size)
    return {
        "context_tokens": context_tokens,
        "pages": -(-context_tokens // STEP4.paged_block_size),
        "physical_regions": physical_regions,
        "max_history_regions": max(physical_regions - 1, 0),
        "metadata_width": METADATA_WIDTH,
    }


def estimate_case_bytes(case: BenchmarkCase) -> int:
    """Conservative live-tensor estimate used only for the pre-allocation memory guard."""
    p = case.params
    bf16 = 2
    fp32 = 4
    i32 = 4
    i64 = 8
    op = case.operator

    if op == "norm_rope":
        tokens = int(p["tokens"])
        heads = int(p["indexer_heads"])
        # q/k/z input + output, plus a small RoPE/norm allowance.
        return tokens * (heads + 2) * STEP4.proxy_dim * bf16 * 2 + 8 * 1024 * 1024
    if op == "csa":
        tokens = sum(p.get("seq_lens", (int(p["tokens"]),)))
        regions = sum(-(-length // STEP4.region_block_size) for length in p.get("seq_lens", (tokens,)))
        return (
            tokens * 2 * STEP4.proxy_dim * bf16
            + regions * STEP4.proxy_dim * (fp32 + 1)
            + 8 * 1024 * 1024
        )
    if op == "logits":
        q = int(p["queries"])
        groups = int(p["groups"])
        regions = int(p["regions"])
        return (
            q * groups * 4 * STEP4.proxy_dim * bf16
            + regions * STEP4.proxy_dim * bf16
            + q * groups * regions * fp32
            + 16 * 1024 * 1024
        )
    if op in {"selector", "metadata"}:
        rows = int(p["rows"])
        regions = int(p["regions"])
        width = int(p["topk"]) + (1 if op == "metadata" else 0)
        return rows * regions * fp32 + rows * width * (i32 * 3) + 16 * 1024 * 1024
    if op in {"prefill", "tiled_prefill"}:
        queries = int(p["queries"])
        groups = int(p["groups"])
        heads = int(p["heads"])
        cache_regions = int(p["cache_regions"])
        requests = len(p.get("request_lengths", (queries,)))
        width = int(p["topk"]) + 1
        cache_tokens = cache_regions * STEP4.region_block_size * requests
        return (
            queries * heads * STEP4.head_dim * bf16 * 2
            + cache_tokens * groups * STEP4.head_dim * bf16 * 2
            + queries * groups * width * i32 * 3
            + 64 * 1024 * 1024
        )
    if op == "decode_metadata":
        batch = int(p["batch"])
        groups = int(p["groups"])
        context = int(p["context"])
        regions = -(-context // STEP4.region_block_size)
        pages = -(-context // STEP4.paged_block_size)
        width = int(p["topk"]) + 1
        return (
            batch * groups * regions * fp32
            + batch * pages * i32
            + batch * groups * width * (i64 + i32)
            + 16 * 1024 * 1024
        )
    if op == "decode":
        batch = int(p["batch"])
        groups = int(p["groups"])
        heads = int(p["heads"])
        context = int(p["context"])
        width = int(p["topk"]) + 1
        return (
            batch * context * groups * STEP4.head_dim * bf16 * 2
            + batch * heads * STEP4.head_dim * bf16 * 2
            + batch * groups * width * (i64 + i32)
            + 64 * 1024 * 1024
        )
    if op == "merge":
        splits = int(p["splits"])
        batch = int(p["batch"])
        heads = int(p["heads"])
        return splits * batch * heads * (STEP4.head_dim * bf16 + fp32) * 2
    raise ValueError(f"no memory estimate for operator {op!r}")


def _case(
    case_id: str,
    profile: str,
    operator: str,
    description: str,
    **params: Any,
) -> BenchmarkCase:
    return BenchmarkCase(case_id, profile, operator, params, description)


CASES: tuple[BenchmarkCase, ...] = (
    # Fast adapter smoke: one case for every numerical stage.
    _case("tp8.norm.t1", "tp8-smoke", "norm_rope", "decode indexer norm + RoPE", tokens=1, indexer_heads=4),
    _case("tp8.csa.t512", "tp8-smoke", "csa", "aligned 64-region compression", tokens=512),
    _case("tp8.logits.q4.r256", "tp8-smoke", "logits", "decode-like indexer scoring", queries=4, regions=256, groups=1),
    _case("tp8.selector.rows64.r256.k32", "tp8-smoke", "selector", "signed selector with shuffled pages", rows=64, regions=256, topk=32, groups=1),
    _case("tp8.selector.identity.rows64.r256.k32", "tp8-smoke", "selector", "signed selector with identity pages", rows=64, regions=256, topk=32, groups=1, table_layout="identity"),
    _case("tp8.meta.rows64.r256.k32", "tp8-smoke", "metadata", "prefill metadata smoke", rows=64, regions=256, topk=32, groups=1),
    _case("tp8.decode_meta.b4.ctx512.k32", "tp8-smoke", "decode_metadata", "decode compaction with unmapped pages", batch=4, context=512, topk=32, heads=8, groups=1, unmapped_pages=2),
    _case("tp8.prefill.q64.r64.k16", "tp8-smoke", "prefill", "scalar sparse prefill smoke", queries=64, cache_regions=64, topk=16, heads=8, groups=1),
    BenchmarkCase(
        "tp8.tiled.requests65x191.k32",
        "tp8-smoke",
        "tiled_prefill",
        {"queries": 256, "cache_regions": 32, "topk": 32, "heads": 8, "groups": 1, "request_lengths": (65, 191)},
        "request boundary falls inside a 128-query tile",
        required_capability="tiled_prefill_v1",
    ),
    _case("tp8.decode.b2.ctx512.k32.s1", "tp8-smoke", "decode", "decode smoke", batch=2, context=512, topk=32, splits=1, heads=8, groups=1),
    _case("tp8.merge.s4.b2", "tp8-smoke", "merge", "split-state merge smoke", splits=4, batch=2, heads=8),

    # Real TP8 per-rank production shapes.
    _case("tp8.norm.t1.production", "tp8-production", "norm_rope", "one-token decode", tokens=1, indexer_heads=4),
    _case("tp8.norm.t512", "tp8-production", "norm_rope", "scheduler chunk", tokens=512, indexer_heads=4),
    _case("tp8.norm.t4096", "tp8-production", "norm_rope", "long prefill chunk", tokens=4096, indexer_heads=4),
    _case("tp8.csa.t4096", "tp8-production", "csa", "512 complete regions", tokens=4096),
    _case("tp8.csa.t8192", "tp8-production", "csa", "1024 complete regions", tokens=8192),
    BenchmarkCase(
        "tp8.csa.ragged",
        "tp8-production",
        "csa",
        {"tokens": 6729, "seq_lens": (64, 4096, 8, 2049, 512)},
        "existing ragged multi-request acceptance shape",
    ),
    _case("tp8.logits.decode16k", "tp8-production", "logits", "four local indexer heads at 16K context", queries=4, regions=2047, groups=1),
    _case("tp8.logits.recorded31929", "tp8-production", "logits", "recorded 31,929-token smoke geometry", queries=4, regions=3991, groups=1),
    _case("tp8.logits.chunk4096.r4095", "tp8-production", "logits", "long prefill score matrix", queries=4096, regions=4095, groups=1),
    _case("tp8.selector.rows512.r4096.k512", "tp8-production", "selector", "production selector", rows=512, regions=4096, topk=512, groups=1),
    _case("tp8.meta.rows512.r4096.k512", "tp8-production", "metadata", "production metadata width 513", rows=512, regions=4096, topk=512, groups=1),
    _case("tp8.decode_meta.b4.ctx31929.k512", "tp8-production", "decode_metadata", "recorded-context decode metadata", batch=4, context=RECORDED_SMOKE_TOKENS, topk=512, heads=8, groups=1),
    _case("tp8.prefill.q128.r4096.k512", "tp8-production", "prefill", "one production query tile", queries=128, cache_regions=4096, topk=512, heads=8, groups=1),
    BenchmarkCase(
        "tp8.tiled.q128.r4096.k512",
        "tp8-production",
        "tiled_prefill",
        {"queries": 128, "cache_regions": 4096, "topk": 512, "heads": 8, "groups": 1, "request_lengths": (128,)},
        "production query-tiled prefill",
        required_capability="tiled_prefill_v1",
    ),
    BenchmarkCase(
        "tp8.tiled.q512.r4096.k512",
        "tp8-production",
        "tiled_prefill",
        {"queries": 512, "cache_regions": 4096, "topk": 512, "heads": 8, "groups": 1, "request_lengths": (512,)},
        "four production query tiles",
        required_capability="tiled_prefill_v1",
    ),
    BenchmarkCase(
        "tp8.prefill.q4096.r4096.k512",
        "tp8-production",
        "prefill",
        {"queries": 4096, "cache_regions": 4096, "topk": 512, "heads": 8, "groups": 1},
        "full 4096-token prefill chunk",
        opt_in=True,
    ),
    _case("tp8.decode.b1.ctx16384.k512.s16", "tp8-production", "decode", "single-request production decode", batch=1, context=16384, topk=512, splits=16, heads=8, groups=1),
    _case("tp8.decode.b4.ctx16384.k512.s1", "tp8-production", "decode", "production decode, one split", batch=4, context=16384, topk=512, splits=1, heads=8, groups=1),
    _case("tp8.decode.b4.ctx16384.k512.s16", "tp8-production", "decode", "production decode occupancy split", batch=4, context=16384, topk=512, splits=16, heads=8, groups=1),
    _case("tp8.decode.b4.ctx31929.k512.s16", "tp8-production", "decode", "recorded smoke context", batch=4, context=RECORDED_SMOKE_TOKENS, topk=512, splits=16, heads=8, groups=1),
    _case("tp8.decode.b8.ctx31929.k512.s16", "tp8-production", "decode", "recorded request-ceiling batch", batch=8, context=RECORDED_SMOKE_TOKENS, topk=512, splits=16, heads=8, groups=1),
    _case("tp8.merge.s16.b4", "tp8-production", "merge", "production split-state merge", splits=16, batch=4, heads=8),

    # TP1 is a correctness/layout profile, not a production throughput claim.
    _case("tp1.norm.t64", "tp1-validation", "norm_rope", "all 16 indexer heads", tokens=64, indexer_heads=16),
    _case("tp1.logits.q64.r256", "tp1-validation", "logits", "four provider groups", queries=64, regions=256, groups=4),
    _case("tp1.logits.q256.r1024", "tp1-validation", "logits", "larger group-major matrix", queries=256, regions=1024, groups=4),
    _case("tp1.selector.rows256.r1024.k512", "tp1-validation", "selector", "group-major selector", rows=256, regions=1024, topk=512, groups=4),
    _case("tp1.decode_meta.b2.ctx2048.k128", "tp1-validation", "decode_metadata", "four-group decode metadata", batch=2, context=2048, topk=128, heads=64, groups=4),
    _case("tp1.prefill.q64.r256.k32", "tp1-validation", "prefill", "64 Q heads and four KV groups", queries=64, cache_regions=256, topk=32, heads=64, groups=4),
    _case("tp1.decode.b2.ctx2048.k128.s4", "tp1-validation", "decode", "four-group split decode", batch=2, context=2048, topk=128, splits=4, heads=64, groups=4),

    # Explicit opt-in derived stress. These are not retained service results.
    BenchmarkCase(
        "tp8.derived.ctx32768.logits",
        "derived-stress",
        "logits",
        {"queries": 32768, "regions": 4095, "groups": 1, **derived_context_geometry(32768)},
        "exact 32K full prefill score tensor",
        provenance="derived",
        measurement_claim=False,
        opt_in=True,
    ),
    BenchmarkCase(
        "tp8.derived.ctx65536.logits",
        "derived-stress",
        "logits",
        {"queries": 65536, "regions": 8191, "groups": 1, **derived_context_geometry(65536)},
        "exact 64K full prefill score tensor",
        provenance="derived",
        measurement_claim=False,
        opt_in=True,
    ),
)


def list_cases(
    profile: str | None = None,
    *,
    include_opt_in: bool = False,
    include_derived_stress: bool = False,
) -> list[BenchmarkCase]:
    selected = []
    for case in CASES:
        if profile is not None and case.profile != profile:
            continue
        if case.profile == "derived-stress" and not include_derived_stress:
            continue
        if case.opt_in and not (include_opt_in or include_derived_stress):
            continue
        selected.append(case)
    return selected


def case_by_id(case_id: str) -> BenchmarkCase:
    for case in CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in sorted(value.items())}
    return value


def expected_implementation_path(case: BenchmarkCase) -> str:
    paths = {
        "norm_rope": "fused_norm_rope",
        "csa": "layout_plus_compress",
        "logits": "weighted_relu_logits",
        "selector": "fused_select_pack",
        "metadata": "select_append_meta",
        "decode_metadata": "decode_select_compact_append",
        "prefill": "per_query_sparse_prefill",
        "tiled_prefill": "query_tiled_public_api",
        "merge": "split_state_merge",
    }
    if case.operator == "decode":
        return "decode_plus_split_merge" if int(case.params["splits"]) > 1 else "decode_single"
    try:
        return paths[case.operator]
    except KeyError as exc:
        raise ValueError(f"no implementation path contract for {case.operator!r}") from exc


def expected_shape_contract(case: BenchmarkCase) -> dict[str, Any]:
    """Return the registry fields that must be visible in one result's shape record."""
    p = case.params
    tp_size = 1 if case.case_id.startswith("tp1.") else 8
    common = {"tp_size": tp_size}
    if case.operator == "norm_rope":
        return {**common, "tokens": p["tokens"], "indexer_heads": p["indexer_heads"]}
    if case.operator == "csa":
        return {
            **common,
            "seq_lens": list(p.get("seq_lens", (p["tokens"],))),
            "region_size": STEP4.region_block_size,
        }
    if case.operator == "logits":
        return {
            **common,
            "queries": p["queries"],
            "regions": p["regions"],
            "groups": p["groups"],
        }
    if case.operator == "selector":
        return {
            **common,
            "rows": p["rows"],
            "regions": p["regions"],
            "topk": p["topk"],
            "table_layout": p.get("table_layout", "shuffled"),
        }
    if case.operator == "metadata":
        return {
            **common,
            "rows": p["rows"],
            "regions": p["regions"],
            "topk": p["topk"],
            "metadata_width": int(p["topk"]) + 1,
        }
    if case.operator == "decode_metadata":
        return {
            **common,
            "batch": p["batch"],
            "context_tokens": p["context"],
            "topk": p["topk"],
            "metadata_width": int(p["topk"]) + 1,
            "num_kv_groups": p["groups"],
            "unmapped_pages_per_request": p.get("unmapped_pages", 0),
        }
    if case.operator in {"prefill", "tiled_prefill"}:
        shape = {
            **common,
            "queries": p["queries"],
            "cache_regions": p["cache_regions"],
            "topk": p["topk"],
            "metadata_width": int(p["topk"]) + 1,
            "heads": p["heads"],
            "groups": p["groups"],
            "region_size": STEP4.region_block_size,
            "block_q": 128 if case.operator == "tiled_prefill" else None,
            "block_regions": 4 if case.operator == "tiled_prefill" else 16,
        }
        if case.operator == "tiled_prefill":
            shape["request_lengths"] = list(p.get("request_lengths", (p["queries"],)))
        return shape
    if case.operator == "decode":
        splits = int(p["splits"])
        return {
            **common,
            "batch": p["batch"],
            "context_tokens": p["context"],
            "topk": p["topk"],
            "metadata_width": int(p["topk"]) + 1,
            "num_splits": splits,
            "num_q_heads": p["heads"],
            "num_kv_groups": p["groups"],
            "timed_merge": splits > 1,
        }
    if case.operator == "merge":
        return {"splits": p["splits"], "batch": p["batch"], "heads": p["heads"]}
    raise ValueError(f"no shape contract for operator {case.operator!r}")


def case_evidence_contract(case: BenchmarkCase, *, seed: int) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("benchmark seed must be a non-negative integer")
    return {
        "schema": "step4-dsa-benchmark-case-v1",
        "case_id": case.case_id,
        "profile": case.profile,
        "operator": case.operator,
        "params": _json_value(case.params),
        "description": case.description,
        "provenance": case.provenance,
        "measurement_claim": case.measurement_claim,
        "opt_in": case.opt_in,
        "required_capability": case.required_capability,
        "seed": seed,
        "implementation_path": expected_implementation_path(case),
        "shape": expected_shape_contract(case),
    }
