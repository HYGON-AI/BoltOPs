Vendored Triton MLA sources:

- `decode_attention.py` — vLLM `vllm/v1/attention/ops/triton_decode_attention.py`
  (SGLang / LightLLM DeepSeek-V2 GQA flash decoding lineage). Used by vLLM
  `TRITON_MLA` (`TritonMLAImpl.forward_mqa` with `is_mla=True`).
- `num_kv_splits.py` — split count heuristic from vLLM
  `vllm/v1/attention/backends/mla/triton_mla.py` (`_compute_num_kv_splits`).
