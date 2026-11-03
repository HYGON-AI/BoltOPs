# FP8 / Half MQA Logits — Performance Notes

Measured primarily on **BW1100 / gfx938**. Prefer kernel-only timing with
preallocated ``D_out`` (CUDA events / Triton’s `do_bench`) so host alloc does
not dominate short shapes.

## Prefill backends

| Backend | When | Notes |
| --- | --- | --- |
| **head_serial** | **gfx946** float8 auto (`backend="wasp"` aliases here) | N-loop head-serial: H=8 prologue MLS; short-KV large-M H=8→`force_grouped` BM32/qg8; mid/large H→`chunk8_mls`; else `Q_GROUP` |
| **wasp** | gfx946 **half** auto / `backend="wasp"` | Half BN-WASP: one CTA owns all heads; default `Q_GROUP=1` + MLS Q; window on store |
| **row** | Tiny KV (`N≤128`) under auto (non-946) | Tuned `block_kv` / warps / stages from shipped JSON |
| **bn** | No Gluon, mid/large (non-946) | 2D tile `BN∈{64,128}`, optional `head_split_k`; Q prefetch + `W_GROUP` |
| **gluon** | Gluon available, mid/large (float8, non-946) | KV LDS + Q double-buf + W_GROUP ping-pong; BN-style epilogue; launch knobs from coarse Q/KV/H ``by_region`` |

Public auto (**gfx946**): float8 → **head_serial** only (FP8 BN-WASP removed;
``backend="wasp"`` aliases head-serial); half still **BN-WASP** (`path=wasp`).

Public auto (**other arches, float8**): Gluon → `row|gluon`; else → `row|bn`
(`get_fp8_mqa_shape_dispatch` in `mqa_config_utils.py`).

Public auto (**other arches, half**): dedicated ``row|bn`` via
`get_half_mqa_shape_dispatch` — short-KV shapes where shipped compare shows
row beating BN go to ``mqa_logits_half_row``; else tuned ``mqa_logits_half``
BN (exact ``by_shape``, then optional ``by_region`` / ``by_seq_kv``). Half Gluon
(`backend="gluon_bn_align"`, BN-style epilogue default) is **opt-in**, not auto.

## gfx946 (gem5)

``fangzx_boltops`` / ``GPU_CHIP=sb``. Speedup from host wall-clock after
synchronize; TFLOPS from sim/kernel event time on the same probes. Logical
FLOPs ≈ ``2·M·H·D·N`` (full window, D=128). Probe shapes only.

| Path | vs prior default |
| --- | --- |
| FP8 **head_serial** vs old BN-WASP / ``bn`` | short-KV H8 large-M often ~1.7–3× vs BN-WASP; mid/large beats prior ``bn``/``row`` |
| Half BN-WASP vs short-KV ``wasp_row`` (H8 / H32 / H64) | ~4–6× / ~2.7–3× / ~1.8–2× |
| Half BN-WASP (**qg1+MLS**) vs ORIG BN (938-style) | gem5 14-proxy: **geo ~2.4×** (min ~0.9× on tiny `16×66 H8`; long-N / H4–64 often **~2–5×**) |

| Shape | event ms | ~TFLOPS |
| --- | ---: | ---: |
| Half BN-WASP one-M-tile ``64×8192`` H4 | 0.022 | ~24 |
| Half BN-WASP one-M-tile ``16×8192`` H64 | 0.114 | ~19 |

## vs hipc (prefill size grid)

Speedup = `hipc_ms / boltops_ms` (>1 BoltOPs faster). Same shapes / windows /
preallocated `D_out`.

| Runtime | Path | ≥0.98× HIPC | geo mean | note |
| --- | --- | --- | --- | --- |
| Triton 3.6+ (Gluon) | float8 auto `row\|gluon` | **54/54** | **~1.39×** | BN-style epilogue; ``assume_no_nans=1``; Gluon launch via Q/KV/H ``by_region`` |
| No Gluon (e.g. 3.2) | float8 auto `row\|bn` | **41/54** | **~1.06×** | weaker on some large-`N`+`H=64` / small-`M`+`H=8`; not re-run after ``assume_no_nans`` |
| Triton 3.6 | **half auto `row\|bn`** | **66/66** | **1.70×** | official 66-shape half grid; 64 faster, 2 ties, 0 slower |
| Triton 3.6 | half Gluon `bn_align` (opt-in) | 48/54 | ~1.36× | 54-grid measurement, **not** re-run on the 66 grid |

Float8 auto meets ≥1.00× HIPC on all 54 grid points under ``by_region``
dispatch (short-KV stays on **row**; min ~1.03× on tiny ``16×66 H=64``).
Refreshed on BW1100 after defaulting ``assume_no_nans=1`` (was ~1.27× geo
before that flip). Coarse ``by_region`` still avoids shipping an exact hipc
shape table.

Half auto meets ≥1.00× HIPC on the 66-shape suite in
`test/mqa_logits/test_triton_mqa_logits_half.py` (`HALF_SIZE_GRID`:
54-grid + 12 production prefill shapes). 64 faster, 2 ties
(`4608×16384` H=4 / H=16), 0 slower. Auto uses **row** on
`32×66 H=64` / `64×66 H=64`, **bn** elsewhere.

Forced `backend="gluon"` on tiny float8 shapes (e.g. `16×66 H=64`) can lose to
hipc; auto dispatch keeps those on **row**.

## Reproduce

```bash
# Prefill FP8: non-gfx946 → row/bn/gluon suite; gfx946 → prod head-serial
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py

# Prefill half: non-gfx946 → 66-grid; gfx946 → BN-WASP gfx946_prod
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py

# gfx946 prod-size benches (FP8 head-serial / half BN-WASP; skipped off 946)
pytest -q --bench -k gfx946_prod test/mqa_logits/test_triton_fp8_mqa_logits.py
pytest -q --bench -k gfx946_prod test/mqa_logits/test_triton_mqa_logits_half.py

# --bench alone is BoltOPs-only. Hipc comparison needs -k vs_hipc
# (optest per does not pass it, so CI does not time hipc).

# Official 66-shape half grid: BoltOPs-only latency
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k test_bench_triton_mqa_logits_half

# ...vs hipc
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k vs_hipc

# ...sharded across 8 GPUs
for i in 0 1 2 3 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$i HALF_GRID_SHARD="$i/8" \
    pytest -q test/mqa_logits/test_triton_mqa_logits_half.py \
    --bench -s -k vs_hipc > /tmp/half_shard$i.log 2>&1 &
done; wait

# ...or emit the single-process table + JSON (not an optest bench; no --bench)
HALF_GRID_REPORT=1 HALF_GRID_JSON=/tmp/mqa_logits_half_66.json \
  pytest -q test/mqa_logits/test_triton_mqa_logits_half.py -s -k report

# Size-grid latency (shapes from hipc prefill cuda list)
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k size_grid

# Optional comparison when hipc is installed (package import name may be lightop)
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_size
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_prod

# Paged decode: pytest benches use time_ms (cudagraph-first median)
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k test_bench_triton_fp8_paged_mqa_logits_gpu
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc
# CLI path sweep (splitkv_nofuse / bucket_persistent) is not collected by optest:
# python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py [--vs-hipc] [--path ...]
```

Float8 / half vs-hipc numbers above come from the pytest suites; re-run those
commands to refresh.

Retune:

```bash
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_fp8_mqa_logits.py --write-ship
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_mqa_logits_half.py --write-ship
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_mqa_logits_half_row_short_kv.py --write-ship
```

## Other notes

* Gluon launch knobs come from coarse ``by_region`` buckets
  (``seq_len_leq`` / ``seq_kv_leq`` / ``num_heads_leq``) in
  ``fp8_mqa_logits_gluon,arch=*.json`` — not an exact hipc-grid shape table.
  Exact all-sizes tune results stay in ``*_all_sizes_tune_*.json`` only.
* HIP warps/stages for tiled helpers still come from the tiled JSON. Row launch
  JSON was autotuned via `tune_fp8_mqa_logits.py`.
* Row JSON tune (Triton 3.2 capability) gave ~1.3× on short-fast / XL vs code
  defaults (not vs hipc).
* Triton 3.2 containers without FP8 hipc can only report absolute BoltOPs ms
  (historically 51/54 ok; `{16,32,64}×66 H=8` row compile fail on some setups).
* FP8 prefill defaults ``assume_no_nans=1`` (HIP ``no-nans-fp-math``), same
  finite-math contract as half / hipc; mid/large shapes measured
  ~1.15–1.19× vs off on BW1100 (bn/gluon; row ~flat).

## Paged decode (FP8)

### Triton paths

Public Triton entries (TileLang is documented in ``docs/ops/mqa_logits_tilelang.md``):

1. **`triton_fp8_paged_mqa_logits`** — full logits; four dispatch branches:
   - ``kv_block_size=1``: per-token SplitKV
   - ``kv_block_size=64``: default **SplitKV block** (only path with ``fuse_output_init``)
   - ``kv_block_size=64`` + ``use_bucket=True``: host-built schedule (eager / reference; not for hot path)
   - ``kv_block_size=64`` + ``bucket_schedule=``: device persistent bucket (skewed batches)
2. **`triton_fp8_paged_mqa_logits_stage1`** — per-head scores (page size 1 / 64)

### Bench

Pytest benches use ``time_ms`` (cudagraph-first, median). BoltOPs-only covers
the default path; ``-k vs_hipc`` is skipped when hipc is missing. The CLI still
does best-of CUDA-graph replay and is the only place that sweeps extra paths.

```bash
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k test_bench_triton_fp8_paged_mqa_logits_gpu
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc
# CLI (not optest): python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py [--vs-hipc]
```

Default path timed by pytest: ``kv_block_size=64`` SplitKV + ``fuse_output_init=True``.
``splitkv_nofuse`` / ``bucket_persistent`` stay on the CLI (``--path``).
