Vendored Triton GEMM sources (flat layout, no vendor subpackages):

FP16 / BF16
- `matmul_persistent.py` — vLLM / SGLang `batch_invariant`
- `gemm_a16w16.py` / `gemm_a16w16_op.py` — ROCm/aiter GEMM-A16W16

Quantized (vLLM / SGLang) — public
- `w8a8_triton_block_scaled_mm.py` — vLLM `fp8_utils`
- `w8a8_block_int8_matmul.py` — SGLang `int8_kernel.py` (HCU OK)
- `triton_scaled_mm.py` — vLLM / SGLang compressed-tensors path
- `awq_gemm_triton.py` — vLLM / SGLang `awq_triton.py`

Quantized (aiter) — public
- `gemm_a8w8_blockscale.py` / `_op.py`
- `gemm_afp4wfp4.py` / `_op.py`
- `batched_gemm_a16wfp4.py` / `_op.py`
- `_quant/quant.py`

Internal (vendored, not in `boltops` / `boltops.gemm` `__all__`)
- `gemm_fp16.py` — HCU WASP/WDRA experiments

Configs (on disk, HCU only where available)
- `configs/awq_gemm_triton/` — vLLM-style `AWQ_{N}_{K}_{K100_AI|BW200}.json`
- `configs/w8a8_block_int8_matmul/` — hygon aiter `block_w8a8` (`BW200` / `gfx938`)

Note: hygon aiter `awq_w4a16/` tables target aiter `gemm_a16w4` (different
schema) and are **not** wired to `awq_gemm_triton`. Other kernels use
`builtin_gemm_configs.py`. Gluon not vendored.
