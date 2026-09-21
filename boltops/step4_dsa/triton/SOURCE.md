Vendored Step-4 DSA Triton sources.

Handoff package: `step4-dsa-handoff-20260831-f9f74c6`
Canonical body: `payload/repository/inference/kernel.py` (generated merge of
`dsa_kernels.py`, `dsa_meta.py`, `sparse_attn.py`, `dsa_attention.build_rope_cache`).
Vendor contract: `step4-dsa-ops-v1`.

This directory is **Triton kernels and launch helpers only**. Model constants
live in `../geometry.py`; production shapes / synthetic inputs live in
`test/step4_dsa/` (not shipped in the wheel).

Split layout (DSA numerical ops only; MoE FP8 / `fused_qknorm_rope` / vLLM planner
are not vendored):

- `common.py` — shared-memory tile fit, shared block-table stride, fp-fusion flag
- `norm_rope.py` — `build_rope_cache`, `indexer_norm_rope` (`kernel.py` ~112-347, 2438-2501)
- `csa.py` — region layout + compress (`kernel.py` ~349-497)
- `logits.py` — weighted-ReLU indexer scores (`kernel.py` ~500-645)
- `topk.py` — signed top-k ids + fused prefill pack (`kernel.py` ~648-1004)
- `metadata.py` — prefill/decode sparse metadata (`kernel.py` ~1059-1391)
- `prefill.py` — sparse GQA prefill, scalar + query-tiled (`kernel.py` ~1429-2034). FP8 KV-cache *read*: host folds ``k_scale`` into ``softmax_scale``; kernel ``v_scale``; K/V widened to Q dtype after load. Write/quantize stays in the framework.
- `decode.py` — sparse GQA decode / split-KV (`kernel.py` ~2037-2181, 2288-2386). Same FP8 KV-cache read contract as prefill.
- `merge.py` — split-state LSE merge (`kernel.py` ~2183-2286)

Sibling (not Triton impl):

- `../geometry.py` — Step-4 / step3p5 constants from `operator_parity/geometry.py`

Test / bench fixtures (not in this package):

- `test/step4_dsa/benchmark_cases.py` — `operator_parity/dsa_benchmark_cases.py` production shapes
- `test/step4_dsa/gen.py` — synthetic input builders from `operator_parity/gen.py`
- `test/step4_dsa/bench_inputs.py` — timed-callable construction for production benches
