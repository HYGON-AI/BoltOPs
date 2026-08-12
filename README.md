# BoltOPs

Backend Optimized Low-latency Triton/TileLang Operators

**English** | [中文](README.zh-CN.md)

## Introduction

**BoltOPs** (*Backend Optimized Low-latency Triton/TileLang Operators*) is the AI Compiler team's high-performance operator library for Hygon DCU. The name also nods to the **lightning bolt** — low latency and lightning-fast execution by design. It delivers production-ready, benchmark-driven kernels written in Triton, TileLang, and other DSLs behind clean Python APIs, giving inference frameworks and model-side callers tuned DCU throughput without backend-specific glue.

The library focuses on latency-critical fused operators, correctness-first validation, and deployment-friendly packaging: fast paths where they matter, minimal friction where callers integrate.

![BoltOp is Fastest](docs/boltop_is_fastest.jpg)


|            | Name             |
| ---------- | ---------------- |
| Repository | `BoltOPs`        |
| DAS        | `boltops`        |
| PyPI / pip | `boltops`        |
| Wheel      | `boltops-*.whl`  |
| Import     | `boltops`        |




## Install

```bash
pip install -e ".[triton]"          # common
pip install -e ".[triton,dev]"      # develop / test
pip install -e ".[tune]"            # MoE autotune / benchmark tools
pip install -e ".[all]"             # Triton + TileLang
./build_wheel.sh                    # pure-Python wheel → dist/
# MoE autotune / benchmark entry (shipped with the package)
# boltops-fused-moe-tune-cli <device_name> <test_type> [--benchmark]
```

Dependencies: `torch`; Triton ops also need a working Triton install (Hygon build). For MoE tuning see
`[docs/ops/fused_moe.md](docs/ops/fused_moe.md)` and
`[tune_moe_triton_user_guide.md](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md)`.

## Usage

Callers should import public APIs from the corresponding operator package.
The root `boltops` package exposes version metadata only:

```python
from boltops.generic import (
    attn_res,
    causal_conv1d_fn,
    causal_conv1d_update,
    fused_mamba_state_scatter_with_mask,
)
from boltops.fused_moe import (
    fused_experts,
    moe_activation,
)
from boltops.gemm import (
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
from boltops.mla import (
    decode_attention_fwd,
    compute_num_kv_splits,
)
from boltops.fla.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
    recompute_w_u_fwd as recompute_w_u_fwd_kda,
)
from boltops.fla.gdn import (
    chunk_fwd_o,
    chunk_gated_delta_rule_fwd_h,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_recurrent,
    recompute_w_u_fwd as recompute_w_u_fwd_gdn,
)

mixed = attn_res(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)

# GDN / SGLang WY recomputation
w, u = recompute_w_u_fwd_gdn(
    k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, cu_seqlens=cu_seqlens
)

# KDA / vLLM WY recomputation
w, u, qg, kg = recompute_w_u_fwd_kda(k=k, v=v, beta=beta, A=A, gk=gk)

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

# MLA decode (aligned with vLLM TRITON_MLA; pass is_mla=True for absorb MLA)
splits = compute_num_kv_splits(max_seq_len)
decode_attention_fwd(
    q, k_buffer, v_buffer, o, lse,
    block_table, seq_lens, attn_logits, splits, sm_scale,
    page_size=64, is_mla=True,
)
```

For A/B of backends, import the backend module directly — do not go around the public API:

```python
from boltops.generic.triton.attn_res import attn_res as triton_impl
```



## Public ops (+ planned)


| Category                                                                             | API                                                                                                                                                                                                                                    | DSLs                | Maintainers                                  | Default backend                       | Intro & docs                                                                                               |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Generic                                                                              | `attn_res` `fused_mamba_state_scatter_with_mask`<br>`causal_conv1d_fn`<br>`causal_conv1d_update`                                                                                                                                                                                        | Triton ✅ TileLang ❌ | Triton @all                                  | Triton                                | [docs](docs/ops/generic/generic.md)                                                                        |
| FLA/KDA family                                                                       | `chunk_gla_fwd_o_gk` `chunk_kda_fwd_intra` `chunk_kda_fwd_intra_token_parallel` `fused_kda_gate` `fused_kda_gate_chunk_cumsum` `fused_recurrent_kda` `fused_recurrent_kda_fwd` `fused_recurrent_kda_packed_decode` `recompute_w_u_fwd` | Triton ✅ TileLang ❌ | Triton @liji                                 | Triton                                | [docs](docs/ops/fla/kda.md)                                                                                |
| FLA/GDN family                                                                       |                                                                                                                                                                                                                                        |                     | Triton @huangteng                            |                                       |                                                                                                            |
| MoE family                                                                           | `fused_experts` `moe_activation`                                                                                                                                                                                                       | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx                | Triton (`fused_experts` mixed+policy) | [docs](docs/ops/fused_moe.md) [tuning guide](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md) |
| FA family (UnifiedAttention / PagedAttention / ExtendAttention / SagedAttention/...) |                                                                                                                                                                                                                                        |                     | Triton @huangshuang @sunchunhao @fangzx |                                       |                                                                                                            |
| MLA / FlashMLA / Sparse MLA                                                          | `decode_attention_fwd` `decode_attention_fwd_grouped` `decode_attention_fwd_normal` `compute_num_kv_splits`                                                                                                                              | Triton ✅ TileLang ❌ | Triton @xukang                               | Triton                                | [docs](docs/ops/mla.md) (dense MLA decode now; FlashMLA / Sparse TBD)                                      |
| MQA Logits (paged)                                                                   |                                                                                                                                                                                                                                        |                     | Triton @fangzx                           |                                       |                                                                                                            |
| MHC                                                                                  | `mhc_pre_big_fuse` `pre_big_fuse_tilelang` `mhc_post_fwd` `hc_split_sinkhorn` `mhc_fused_tilelang`                                                                                                                                    | TileLang ✅          | TileLang @zhouweiwang                       | TileLang                              | [docs](docs/ops/mhc.md)                                                                                    |
| Gemm/AwqGemm family                                                                  | `matmul_persistent` `gemm_a16w16` `w8a8_triton_block_scaled_mm` `triton_scaled_mm` `awq_gemm_triton` `gemm_a8w8_blockscale` `gemm_afp4wfp4` `batched_gemm_a16wfp4` `w8a8_block_int8_matmul`                                            | Triton ✅ TileLang ❌ | Triton @xukang @duyubo                       | Triton                                | [docs](docs/ops/gemm.md)                                                                                   |
| DeepGemm family                                                                      |                                                                                                                                                                                                                                        |                     | Triton @zhenxin Tilelang @zhouweiwang        |                                       |                                                                                                            |




## Layout

```text
boltops/
  fla/               # Flash Linear Attention umbrella
    kda/             # FLA/KDA family (Triton @liji)
    gdn/             # FLA/GDN family (Triton @huangteng)
    common/          # helpers shared by FLA sub-families
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

1. **Positioning of BoltOPs operators**
  As a DSL-built operator library, we pursue performance on specific sizes **on top of generality** — that is the design intent. For a given op, different kernels + DSLs(mix) may target specific sizes or scenarios when chasing peak performance.
2. **Triton and TileLang**
  Both are Python-like DSLs and, today, complementary / collaborative — BoltOPs should reflect that:


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
# pytest test/fla/test_gdn_recompute_w_u.py
# pytest test/fla/test_gdn_recompute_w_u.py --bench -s
```

**Adding a new op:**

1. Create `boltops/<op>/` with `triton/` and `tilelang/` impls; put correctness in `test/test_<op>.py` and model-size benches under `@pytest.mark.bench`, then use `--bench` to pick the winner.
2. Prefer public signatures aligned with vLLM / SGLang (or HIPC / CUDA C); wire ship-time backend selection in `boltops/<op>/__init__.py` and export.
3. Re-export from `boltops/__init__.py` and add `docs/ops/`.
