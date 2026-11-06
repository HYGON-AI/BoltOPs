Triton GEMM implementations
===========================

Public host functions live in `boltops/gemm/`. Backend files contain device
kernels and their compilation/configuration helpers, without duplicate GEMM
host entry points.

| Backend file | Source / implementation |
| --- | --- |
| `gemm_a16w16.py` | ROCm/aiter A16W16 |
| `gemm_a16w16_wasp.py` | HCU WASP/WDRA implementation based on the Triton matmul tutorial |
| `gemm_a32w32.py` | vLLM / SGLang persistent matmul |
| `gemm_a8w8.py` | vLLM / SGLang compressed-tensors scaled matmul |
| `gemm_a8w8_blockscale.py` | ROCm/aiter block-scaled device kernels |
| `gemm_a8w8_blockscale_fp8.py` | vLLM FP8 block-scaled matmul |
| `gemm_a8w8_blockscale_int8.py` | SGLang INT8 block-scaled matmul |
| `gemm_a16w4.py` | vLLM / SGLang AWQ GEMM and dequantization device kernels |
| `gemm_w4a4.py` | ROCm/aiter FP4 × FP4 GEMM |
| `batched_gemm_a16w16.py` | ROCm/aiter batched FP16/BF16 GEMM |
| `batched_gemm_a16w4.py` | ROCm/aiter batched A16 × FP4 with online activation quantization |
| `_quant/quant.py` | ROCm/aiter quantization primitives |

Original source URLs, licenses and copyright notices are retained in each file.
Public configuration lookup uses `../configs/<arch>/[<CU>cu]/<backend>/<family>/`.
Historical backend-local JSON tables are not read by public operation dispatch.

## Relocated third-party sources and licenses

Relocation and host/device separation retain the source licenses; the repository's
MIT license does not replace the Apache-2.0 licenses below. Hygon copyright lines
in these files apply to modifications only.

| Current file | Source retained from the base revision | License | Original holder |
| --- | --- | --- | --- |
| `gemm_a16w16.py` | `gemm_a16w16_op.py`; original same-path host header also retained | MIT | Advanced Micro Devices, Inc. |
| `gemm_a16w4.py` | `awq_gemm_triton.py`; vLLM `awq_triton.py` | Apache-2.0 | Contributors to the vLLM project |
| `gemm_a32w32.py` | `matmul_persistent.py`; vLLM `batch_invariant.py` | Apache-2.0 | Contributors to the vLLM project |
| `gemm_a8w8.py` | `triton_scaled_mm.py`; vLLM / SGLang scaled matmul | Apache-2.0 | Contributors to the vLLM project (retained SPDX notice) |
| `gemm_a8w8_blockscale_fp8.py` | `w8a8_triton_block_scaled_mm.py`; vLLM `fp8_utils.py` | Apache-2.0 | Contributors to the vLLM project |
| `gemm_a8w8_blockscale_int8.py` | `w8a8_block_int8_matmul.py`; SGLang `int8_kernel.py` | Apache-2.0 | No copyright line in the referenced upstream source; existing license retained |
| `../../../test/gemm/test_gemm_a16w4.py` | `test/gemm/test_awq_gemm_triton.py`; vLLM `test_awq_triton.py` | Apache-2.0 | Contributors to the vLLM project |

The base revision is `4a179c81d1aa9f12fe5b63bb3caa800ea226c57f` of
[HYGON-AI/BoltOPs](https://github.com/HYGON-AI/BoltOPs/tree/4a179c81d1aa9f12fe5b63bb3caa800ea226c57f).
Pinned upstream URLs are in the corresponding file headers. The new dispatch
tests are Hygon-authored and use the repository's MIT license.
