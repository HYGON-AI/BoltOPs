Vendored Triton MQA logits sources (prefill / non-paged):

- `fp8_mqa_logits.py` — hygon aiter lineage (row + BN-tile kernels).
  Public backends: **row** and **bn**.
- `fp8_mqa_logits_gluon.py` — Gluon tiled prefill (Triton 3.6+).
- `fp8_mqa_logits_tiled.py` — thin BN forwarder (compat / tune helper).
- `mqa_config_utils.py` — arch JSON load + `get_fp8_mqa_shape_dispatch`
  (`row|gluon` or `row|bn`).

Semantic lineage: DeepGEMM `fp8_mqa_logits` (sparse indexer scoring).
Call sites: vLLM `rocm_fp8_mqa_logits`, SGLang DSA indexer (ROCm → aiter).

Public entry: `boltops.mqa_logits.fp8_mqa_logits`
(`impl="triton"` default, `backend=None|"auto"|"row"|"bn"|"gluon"`, optional `D_out`).

Launch configs:
- `configs/triton_3_6/fp8_mqa_logits,arch=gfx938.json` — row
- `configs/triton_3_6/fp8_mqa_logits_gluon,arch=gfx938.json` — Gluon knobs
  (hand-picked from w_group / w_cache sweeps; not a full autotune dump)
- `configs/triton_3_2/fp8_mqa_logits,arch=gfx938.json` — row (3.2)
- `configs/triton_3_2/fp8_mqa_logits_tiled,arch=gfx938.json` — BN/HIP knobs
  (also reused by Gluon host for warps/stages/kpack)
- `tune_fp8_mqa_logits.py` / `tune_fp8_mqa_logits_tiled.py` — maintainer retune
- `PERFORMANCE.md` — notes

Paged decode (`fp8_paged_mqa_logits`) is intentionally not shipped in this MR.
