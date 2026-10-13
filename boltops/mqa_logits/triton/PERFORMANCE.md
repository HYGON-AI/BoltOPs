# FP8 / Half MQA Logits — Performance Notes

Measured primarily on **BW1100 / gfx938**. Prefer kernel-only timing with
preallocated ``D_out`` (CUDA events / Triton’s `do_bench`) so host alloc does
not dominate short shapes.

## Prefill backends

| Backend | When | Notes |
| --- | --- | --- |
| **row** | Tiny KV (`N≤128`) under auto dispatch | Tuned `block_kv` / warps / stages from shipped JSON |
| **bn** | No Gluon, mid/large | 2D tile `BN∈{64,128}`, optional `head_split_k`; Q prefetch + `W_GROUP` |
| **gluon** | Gluon available, mid/large (float8) | KV LDS + Q double-buf + W_GROUP ping-pong; BN-style epilogue; launch knobs from coarse Q/KV/H ``by_region`` |

Public auto (**float8**): Gluon → `row|gluon`; else → `row|bn`
(`get_fp8_mqa_shape_dispatch` in `mqa_config_utils.py`).

Public auto (**half** fp16/bf16): dedicated ``row|bn`` via
`get_half_mqa_shape_dispatch` — short-KV shapes where shipped compare shows
row beating BN go to ``mqa_logits_half_row``; else tuned ``mqa_logits_half``
BN (exact ``by_shape``, then optional ``by_region`` / ``by_seq_kv``). Half Gluon
(`backend="gluon_bn_align"`, BN-style epilogue default) is **opt-in**, not auto.

## vs hipc (prefill size grid)

Speedup = `hipc_ms / boltops_ms` (>1 BoltOPs faster). Same shapes / windows /
preallocated `D_out`.

| Runtime | Path | ≥0.98× HIPC | geo mean | note |
| --- | --- | --- | --- | --- |
| Triton 3.6+ (Gluon) | float8 auto `row\|gluon` | **54/54** | **~1.27×** | BN-style epilogue; Gluon launch via Q/KV/H ``by_region`` |
| No Gluon (e.g. 3.2) | float8 auto `row\|bn` | **41/54** | **~1.06×** | weaker on some large-`N`+`H=64` / small-`M`+`H=8` |
| Triton 3.6 | **half auto `row\|bn`** | **66/66** | **1.70×** | official 66-shape half grid; 64 faster, 2 ties, 0 slower |
| Triton 3.6 | half Gluon `bn_align` (opt-in) | 48/54 | ~1.36× | 54-grid measurement, **not** re-run on the 66 grid |

Float8 auto meets ≥0.98× HIPC on all 54 grid points under ``by_region``
dispatch (short-KV stays on **row**). Exact-shape ship was ~1.28× geo;
coarse regions trade ~0.6% for not shipping the hipc grid.

Half auto meets ≥1.00× HIPC on the 66-shape suite in
`test/mqa_logits/test_triton_mqa_logits_half.py` (`HALF_SIZE_GRID`:
54-grid + 12 production prefill shapes). 64 faster, 2 ties
(`4608×16384` H=4 / H=16), 0 slower. Auto uses **row** on
`32×66 H=64` / `64×66 H=64`, **bn** elsewhere.

Forced `backend="gluon"` on tiny float8 shapes (e.g. `16×66 H=64`) can lose to
hipc; auto dispatch keeps those on **row**.

## Reproduce

```bash
# Prefill FP8 correctness (row / bn / gluon + auto grid)
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py

# Prefill half correctness (official 66-shape grid subset)
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py

# Official 66-shape half grid: perf vs hipc (all shapes)
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k vs_hipc

# ...sharded across 8 GPUs
for i in 0 1 2 3 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$i HALF_GRID_SHARD="$i/8" \
    pytest -q test/mqa_logits/test_triton_mqa_logits_half.py \
    --bench -s -k vs_hipc > /tmp/half_shard$i.log 2>&1 &
done; wait

# ...or emit the single-process table + JSON
HALF_GRID_REPORT=1 HALF_GRID_JSON=/tmp/mqa_logits_half_66.json \
  pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k report

# Size-grid latency (shapes from hipc prefill cuda list)
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k size_grid

# Optional comparison when hipc is installed (package import name may be lightop)
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_size
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_prod

# Paged decode pure-GPU (+ optional vs-hipc)
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc
# or: python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py [--vs-hipc]
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

Default is **BoltOPs-only pure GPU** (CUDA-graph replay; host / Triton launch excluded).
Optional ``--vs-hipc`` compares against hipc with the same method.

```bash
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc
# or: python test/mqa_logits/test_triton_fp8_paged_mqa_logits.py [--vs-hipc]
```

Default path timed: ``kv_block_size=64`` SplitKV + ``fuse_output_init=True``.
Other branches: ``--path splitkv_nofuse bucket_persistent``.
