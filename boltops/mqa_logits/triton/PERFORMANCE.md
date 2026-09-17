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

## vs lightop (prefill size grid)

Speedup = `lightop_ms / boltops_ms` (>1 BoltOPs faster). Same shapes / windows /
preallocated `D_out`.

| Runtime | Path | ≥0.98× LO | geo mean | note |
| --- | --- | --- | --- | --- |
| Triton 3.6+ (Gluon) | float8 auto `row\|gluon` | **54/54** | **~1.27×** | BN-style epilogue; Gluon launch via Q/KV/H ``by_region`` |
| No Gluon (e.g. 3.2) | float8 auto `row\|bn` | **41/54** | **~1.06×** | weaker on some large-`N`+`H=64` / small-`M`+`H=8` |
| Triton 3.6 | **half auto `row\|bn`** | **50/54** | **~1.41×** | ship default; row on `32×66`/`64×66` H=64 |
| Triton 3.6 | half Gluon `bn_align` (opt-in) | **48/54** | **~1.36×** | often close to BN; loses more on short-N H=64 |

Float8 auto meets ≥0.98× LO on all 54 grid points under ``by_region``
dispatch (short-KV stays on **row**). Exact-shape ship was ~1.28× geo;
coarse regions trade ~0.6% for not shipping the lightop grid.

Half auto below 0.98× LO (4/54, all `path=bn`): `16×4096 H=64` (~0.74×),
`32×66 H=8` (~0.85×), `16×65536 H=8` (~0.88×), `16×66 H=64` (~0.94×).
On the last two, opt-in Gluon can match/beat LO; on the first two all three
Triton paths (BN/row/Gluon) still lose.

Forced `backend="gluon"` on tiny float8 shapes (e.g. `16×66 H=64`) can lose to
lightop; auto dispatch keeps those on **row**.

## Reproduce

```bash
# Correctness (row / bn / gluon + auto grid; half backends included)
pytest -q test/mqa_logits/test_fp8_mqa_logits.py

# Size-grid latency (shapes from lightop prefill cuda list)
pytest -q test/mqa_logits/test_fp8_mqa_logits.py --bench -s -k size_grid

# Optional comparison when lightop is installed
pytest -q test/mqa_logits/test_fp8_mqa_logits.py --bench -s -k vs_lightop

# Full vs-lightop grids (local helpers)
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -u \
  boltops/mqa_logits/triton/_bench_fp8_auto_vs_lightop_sizes.py
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/_bench_36_gluon_sizes.py
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/_bench_three_backends_sizes.py
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -u \
  boltops/mqa_logits/triton/_bench_half_auto_row_bn_vs_lightop_all_sizes.py
```

Float8 auto table artifact:
`configs/triton_3_6/bench_row_gluon_vs_lightop_all_sizes.out`

Half auto table artifact:
`configs/triton_3_6/bench_half_auto_row_bn_vs_lightop_all_sizes.out`

Retune:

```bash
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_fp8_mqa_logits.py --write-ship
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_mqa_logits_half.py --write-ship
HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_mqa_logits_half_row_short_kv.py --write-ship
```

## Other notes

* Gluon launch knobs come from coarse ``by_region`` buckets
  (``seq_len_leq`` / ``seq_kv_leq`` / ``num_heads_leq``) in
  ``fp8_mqa_logits_gluon,arch=*.json`` — not an exact lightop-grid shape table.
  Exact all-sizes tune results stay in ``*_all_sizes_tune_*.json`` only.
* HIP warps/stages for tiled helpers still come from the tiled JSON. Row launch
  JSON was autotuned via `tune_fp8_mqa_logits.py`.
* Row JSON tune (Triton 3.2 capability) gave ~1.3× on short-fast / XL vs code
  defaults (not vs lightop).
* Triton 3.2 containers without FP8 lightop can only report absolute BoltOPs ms
  (`_bench_32_sizes.out.txt`: 51/54 ok; `{16,32,64}×66 H=8` row compile fail).
