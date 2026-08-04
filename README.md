# BoltOps

Backend Optimized Low-latency Triton/TileLang Operators
<br>

**English** | [中文](README.zh-CN.md)

## Introduction

**BoltOps** (*Backend Optimized Low-latency Triton/TileLang Operators*) is the AI Compiler team's high-performance operator library for Hygon DCU. The name also nods to the **lightning bolt** — low latency and lightning-fast execution by design. It delivers production-ready, benchmark-driven kernels written in Triton, TileLang, and other DSLs behind clean Python APIs, giving inference frameworks and model-side callers tuned DCU throughput without backend-specific glue.

The library focuses on latency-critical fused operators, correctness-first validation, and deployment-friendly packaging: fast paths where they matter, minimal friction where callers integrate.

![BoltOp is Fastest](docs/boltop_is_fastest.jpg)


|            | Name             |
| ---------- | ---------------- |
| DAS        | `bolt_ops`       |
| PyPI / pip | `bolt_ops`       |
| Wheel      | `bolt_ops-*.whl` |
| Import     | `bolt_ops`       |




## Install

```bash
pip install -e ".[triton]"          # common
pip install -e ".[triton,dev]"      # develop / test
pip install -e ".[tune]"            # MoE autotune / benchmark tools
pip install -e ".[all]"             # Triton + TileLang
./build_wheel.sh                    # pure-Python wheel → dist/
# MoE autotune / benchmark entry (shipped with the package)
# bolt-moe-tune-cli <device_name> <test_type> [--benchmark]
```

Dependencies: `torch`; Triton ops also need a working Triton install (Hygon build). For MoE tuning see
`[docs/ops/fused_moe.md](docs/ops/fused_moe.md)` and
`[tune_moe_triton_user_guide.md](bolt_ops/tools/triton_moe_tune/tune_moe_triton_user_guide.md)`.

## Usage

Callers should use the public API only:

```python
from bolt_ops import (
    attn_res,
    recompute_w_u_fwd,
    fused_mamba_state_scatter_with_mask,
    fused_experts,
    moe_activation,
    matmul_persistent,
    gemm_a16w16,
    w8a8_triton_block_scaled_mm,
    triton_scaled_mm,
    awq_gemm_triton,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    batched_gemm_a16wfp4,
    w8a8_block_int8_matmul,
)
from bolt_ops.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
    recompute_w_u_fwd as kda_recompute_w_u_fwd,
)

mixed = attn_res(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)

w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens)

fused_mamba_state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw)

out = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)

moe_activation(activated, ffn1_out, activation="silu")

# Gemm family (layouts match vLLM / aiter call sites)
c = matmul_persistent(a, b, bias)         # vLLM Triton persistent; B[K,N]
y = gemm_a16w16(x, weight, bias)          # aiter A16W16; weight [N,K]
c = w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128])
c = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float16)
c = awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters)
y = gemm_a8w8_blockscale(x, w, x_scale, w_scale)
y = gemm_afp4wfp4(x, w, x_scales, w_scales)          # needs FP4 GPU
y = batched_gemm_a16wfp4(x, w, w_scales)             # needs FP4 GPU
c = w8a8_block_int8_matmul(A, B, As, Bs, [128, 128]) # SGLang INT8 block
```

KDA exposes the directly migrated Triton kernels and launch wrappers through
`bolt_ops.kda`; see [the KDA operator guide](docs/ops/kda.md) for prefill and
packed-decode signatures.

For A/B of backends, import the backend module directly — do not go around the public API:

```python
from bolt_ops.recompute_w_u.triton.recompute_w_u import recompute_w_u_fwd as triton_impl
# from bolt_ops.recompute_w_u.tilelang.recompute_w_u import recompute_w_u_fwd as tilelang_impl
```



## Public ops (+ planned)


| Category        | API                                   | DSLs                | Maintainers                   | Default backend       | Intro & docs                            |
| --------------- | ------------------------------------- | ------------------- | ----------------------------- | --------------------- | --------------------------------------- |
| Mixed           | `attn_res`                            | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/attn_res.md)            |
| Mixed           | `recompute_w_u_fwd`                   | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/recompute_w_u.md)       |
| Mixed           | `fused_mamba_state_scatter_with_mask` | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/mamba_state_scatter.md) |
| MoE family      | `fused_experts`                       | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton (mixed+policy) | [docs](docs/ops/fused_moe.md)           |
| MoE family      | `moe_activation`                     | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/moe_activation.md)      |
| MHC             |                                       |                     |                               |                       |                                         |
| Gemm family     | `matmul_persistent`               | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `gemm_a16w16`                     | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `w8a8_triton_block_scaled_mm`     | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `triton_scaled_mm`                | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `awq_gemm_triton`                 | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `gemm_a8w8_blockscale`            | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `gemm_afp4wfp4`                   | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `batched_gemm_a16wfp4`            | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| Gemm family     | `w8a8_block_int8_matmul`          | Triton ✅ TileLang ❌ | Triton @xukang TileLang @xxx | Triton | [docs](docs/ops/gemm.md) |
| SageAttention   |                                       |                     |                               |                       |                                         |
| KDA family      | `chunk_gla_fwd_o_gk`                  | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `chunk_kda_fwd_intra`                 | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `chunk_kda_fwd_intra_token_parallel`  | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `fused_kda_gate`                      | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `fused_kda_gate_chunk_cumsum`         | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `fused_recurrent_kda`                 | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `fused_recurrent_kda_fwd`             | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `fused_recurrent_kda_packed_decode`   | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| KDA family      | `recompute_w_u_fwd`                   | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx | Triton                | [docs](docs/ops/kda.md)                 |
| FA family       |                                       |                     |                               |                       |                                         |
| DeepGemm family |                                       |                     |                               |                       |                                         |
| MLA family      |                                       |                     |                               |                       |                                         |
| …               |                                       |                     |                               |                       |                                         |


MoE tuning guide: `[tune_moe_triton_user_guide.md](bolt_ops/tools/triton_moe_tune/tune_moe_triton_user_guide.md)`.

## Layout

```text
bolt_ops/
  <op>/              # one directory per op
    __init__.py      # public signature + ship-time backend (lazy-import)
    triton/          # Triton impl (incl. configs)
    tilelang/        # TileLang impl (placeholder)
  tools/             # shipped developer tools (incl. MoE autotune)
  utility/           # shared runtime helpers
docs/ops/            # op docs
test/                # correctness + optional --bench; not in wheel
build_wheel.sh       # build wheel
```



## Development

1. **Positioning of BoltOps operators**
  As a DSL-built operator library, we pursue performance on specific sizes **on top of generality** — that is the design intent. For a given op, different kernels + DSLs(mix) may target specific sizes or scenarios when chasing peak performance.
2. **Triton and TileLang**
  Both are Python-like DSLs and, today, complementary / collaborative — BoltOps should reflect that:

  |           | Triton                                                                                                            | TileLang                                                                                                                       |
  | --------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
  | Strengths | Strong ecosystem & compatibility; fast performance uplift; light edits can reach ~80%+ of hand-written HIPC / asm | Rising DSL with warp / thread-level fine-grained APIs; fast to develop; scene-specific hand-tuned kernels can match HIPC / asm |
  | Limits    | Harder to push further toward peak                                                                                | Ecosystem and generality lag Triton                                                                                            |

   The same op may therefore ship both Triton and TileLang backends; the default backend is chosen after bake-off (or a mixed policy).
3. **API definition**
  Prefer aligning public signatures with latest vLLM, SGLang, and similar open-source frameworks for easier integration. If no matching API exists, align with HIPC / CUDA C counterparts.
4. **Testing**
  Each op has one test file / directory, with `test` and `bench` together: `test` covers interface functionality / generality; `bench` focuses on real-network shapes. They complement each other.

```bash
pip install -e ".[triton,dev]"
pytest test                  # correctness (functional / case coverage)
pytest test --bench -s       # correctness + real model-size performance
# single op:
# pytest test/test_recompute_w_u.py
# pytest test/test_recompute_w_u.py --bench -s
```

**Adding a new op:**

1. Create `bolt_ops/<op>/` with `triton/` and `tilelang/` impls; put correctness in `test/test_<op>.py` and model-size benches under `@pytest.mark.bench`, then use `--bench` to pick the winner.
2. Prefer public signatures aligned with vLLM / SGLang (or HIPC / CUDA C); wire ship-time backend selection in `bolt_ops/<op>/__init__.py` and export.
3. Re-export from `bolt_ops/__init__.py` and add `docs/ops/`.
