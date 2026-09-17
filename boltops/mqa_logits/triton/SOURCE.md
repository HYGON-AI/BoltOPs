Vendored Triton MQA logits sources (prefill / non-paged):

- `fp8_mqa_logits.py` — hygon aiter lineage (row + BN-tile kernels).
  **Internal** float8 kernels used by public ``triton_mqa_logits`` when
  ``kv_scales`` is provided.
- `mqa_logits_half.py` — dedicated fp16/bf16 BN path (ship `by_shape` / defaults).
- `mqa_logits_half_row.py` — dedicated fp16/bf16 per-row path (short-KV).
- `mqa_logits_half_gluon_bn_align.py` — **experimental** half Gluon BN-aligned
  (`backend="gluon_bn_align"`; defaults `w_lds_pp=False`, `q_double_buf=False`).
- `fp8_mqa_logits_gluon.py` — Gluon tiled prefill (Triton 3.6+, **float8 only**).
- `fp8_mqa_logits_gluon_opt.py` — **experimental** alias for ship Gluon
  (`backend="gluon_opt"`); product code should prefer `backend="gluon"`.
- `fp8_mqa_logits_gluon_bn_align.py` — **experimental** BN-aligned Gluon + optional
  QPP/WPP / layout ablations.
- `fp8_mqa_logits_tiled.py` — thin BN forwarder (compat / tune helper).
- `mqa_config_utils.py` — arch JSON load + float8 / half dispatch helpers.

Semantic lineage: DeepGEMM `fp8_mqa_logits` (sparse indexer scoring).
Call sites: vLLM `rocm_fp8_mqa_logits`, SGLang DSA indexer (ROCm → aiter).

Public entry: **`boltops.mqa_logits.triton_mqa_logits`** (Triton only;
TileLang can land later as a sibling export).
`backend=None|"auto"|"row"|"bn"|"gluon"` plus experimental
`gluon_bn_align` / `gluon_opt`; optional `D_out`.
Dispatch:

* ``kv_scales`` provided → float8 path
* ``kv_scales is None`` → half path

Launch configs:
- `configs/triton_3_6/fp8_mqa_logits,arch=gfx938.json` — row
- `configs/triton_3_6/fp8_mqa_logits_gluon,arch=gfx938.json` — Gluon knobs
- `configs/triton_3_6/mqa_logits_half,arch=gfx938.json` — half BN
- `configs/triton_3_6/mqa_logits_half_row_short_kv,arch=gfx938.json` — half row
- `configs/triton_3_2/fp8_mqa_logits,arch=gfx938.json` — row (3.2)
- `configs/triton_3_2/fp8_mqa_logits_tiled,arch=gfx938.json` — BN/HIP knobs
- `configs/triton_3_2/mqa_logits_half*,arch=gfx938.json` — half (3.2)
- `tune_mqa_logits_half.py` / `tune_mqa_logits_half_row_short_kv.py` /
  `tune_mqa_logits_half_gluon_bn_align.py` — half retune
- `tune_fp8_mqa_logits.py` / `tune_fp8_mqa_logits_tiled.py` /
  `tune_fp8_mqa_logits_gluon.py` — FP8 retune
- `PERFORMANCE.md` — notes

Paged decode (`fp8_paged_mqa_logits`) is intentionally not shipped in this MR.
