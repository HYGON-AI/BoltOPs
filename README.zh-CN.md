# BoltOps

Backend Optimized Low-latency Triton/TileLang Operators

[English](README.md) | **中文**

## 介绍

**BoltOps**（*Backend Optimized Low-latency Triton/TileLang Operators*）是 AI Compiler 团队面向海光 DCU 的高性能算子库。命名同时取自 **lightning bolt（闪电）**：以低时延为目标，追求闪电般的执行速度。它以 Triton、TileLang 等 DSL 实现可量产的、由 benchmark 驱动的 kernel，并通过清晰的 Python API 对外提供，让推理框架与模型侧调用方无需后端胶水即可获得调优后的 DCU 吞吐。

库聚焦时延敏感的融合算子、正确性优先的验证，以及便于部署的打包方式：该走快路径的地方走快路径，该降低接入成本的地方尽量少摩擦。

![BoltOp is Fastest](docs/boltop_is_fastest.jpg)


|            | 名称               |
| ---------- | ---------------- |
| DAS        | `bolt_ops`       |
| PyPI / pip | `bolt_ops`       |
| Wheel      | `bolt_ops-*.whl` |
| Import     | `bolt_ops`       |




## 安装

```bash
pip install -e ".[triton]"          # 常用
pip install -e ".[triton,dev]"      # 开发 / 测试
pip install -e ".[tune]"            # MoE autotune / 测速工具
pip install -e ".[all]"             # Triton + TileLang
./build_wheel.sh                    # 打纯 Python wheel → dist/
# MoE autotune / 测速入口（随包发布）
# bolt-moe-tune-cli <device_name> <test_type> [--benchmark]
```

依赖：`torch`；跑 Triton 算子还需本机可用的 Triton（Hygon 版）。MoE 调参见
`[docs/ops/fused_moe.md](docs/ops/fused_moe.md)` 与
`[tune_moe_triton_user_guide.md](bolt_ops/tools/triton_moe_tune/tune_moe_triton_user_guide.md)`。

## 用法

业务侧从对应算子包导入公开 API；根包 `bolt_ops` 只暴露版本信息：

```python
from bolt_ops.generic import (
    attn_res,
    recompute_w_u_fwd,
    fused_mamba_state_scatter_with_mask,
)
from bolt_ops.fused_moe import (
    fused_experts,
    moe_activation,
)
from bolt_ops.gemm import (
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
from bolt_ops.mla import (
    decode_attention_fwd,
    compute_num_kv_splits,
)
from bolt_ops.fla.kda import (
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

# Gemm 系列（布局与 vLLM / aiter 调用一致）
c = matmul_persistent(a, b, bias)         # vLLM Triton persistent；B[K,N]
y = gemm_a16w16(x, weight, bias)          # aiter A16W16；weight 为 [N,K]
c = w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128])
c = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float16)
c = awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters)
y = gemm_a8w8_blockscale(x, w, x_scale, w_scale)
y = gemm_afp4wfp4(x, w, x_scales, w_scales)          # 需要 FP4 GPU
y = batched_gemm_a16wfp4(x, w, w_scales)             # 需要 FP4 GPU
c = w8a8_block_int8_matmul(A, B, As, Bs, [128, 128]) # SGLang INT8 block

# MLA decode（对齐 vLLM TRITON_MLA；MLA absorb 传 is_mla=True）
splits = compute_num_kv_splits(max_seq_len)
decode_attention_fwd(
    q, k_buffer, v_buffer, o, lse,
    block_table, seq_lens, attn_logits, splits, sm_scale,
    page_size=64, is_mla=True,
)
```

需要做实现 A/B 时，直接 import 后端模块，不要绕公开 API：

```python
from bolt_ops.generic.triton.recompute_w_u import recompute_w_u_fwd as triton_impl
```



## 公开算子（+待开发算子）


| 算子类别                                                                            | 接口名                                                                                                                                                                                                                                    | DSLs                | 维护者                                           | 出厂后端                         | 介绍和文档                                                                                              |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------- |
| Generic                                                                         | `attn_res` `recompute_w_u_fwd` `fused_mamba_state_scatter_with_mask`                                                                                                                                                                   | Triton ✅ TileLang ❌ | Triton @all                                   | Triton                       | [docs](docs/ops/generic/generic.md)                                                                |
| FLA/KDA 系列                                                                      | `chunk_gla_fwd_o_gk` `chunk_kda_fwd_intra` `chunk_kda_fwd_intra_token_parallel` `fused_kda_gate` `fused_kda_gate_chunk_cumsum` `fused_recurrent_kda` `fused_recurrent_kda_fwd` `fused_recurrent_kda_packed_decode` `recompute_w_u_fwd` | Triton ✅ TileLang ❌ | Triton @liji                                  | Triton                       | [docs](docs/ops/fla/kda.md)                                                                        |
| FLA/GDN 系列                                                                      |                                                                                                                                                                                                                                        |                     | Trtion @huangteng                             |                              |                                                                                                    |
| MoE 系列                                                                          | `fused_experts` `moe_activation`                                                                                                                                                                                                       | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx                 | Triton（`fused_experts` 混合策略） | [docs](docs/ops/fused_moe.md) [调参指南](bolt_ops/tools/triton_moe_tune/tune_moe_triton_user_guide.md) |
| FA 系列（UnifiedAttention / PagedAttention / ExtendAttention / SagedAttention/...） |                                                                                                                                                                                                                                        |                     | Triton @huangshuang @sunchunhao @fangzx |                              |                                                                                                    |
| MLA / FlashMLA / Sparse MLA                                                     | `decode_attention_fwd` `decode_attention_fwd_grouped` `decode_attention_fwd_normal` `compute_num_kv_splits`                                                                                                                              | Triton ✅ TileLang ❌ | Triton @xukang                                | Triton                       | [docs](docs/ops/mla.md)（目前 dense MLA decode；FlashMLA / Sparse 待迁）                              |
| MQA Logits（paged）                                                               |                                                                                                                                                                                                                                        |                     | Triton @fangzx                            |                              |                                                                                                    |
| MHC                                                                             |                                                                                                                                                                                                                                        | TileLang            | Triton @zhouweiwang                          |                              |                                                                                                    |
| Gemm/AwqGemm 系列                                                                 | `matmul_persistent` `gemm_a16w16` `w8a8_triton_block_scaled_mm` `triton_scaled_mm` `awq_gemm_triton` `gemm_a8w8_blockscale` `gemm_afp4wfp4` `batched_gemm_a16wfp4` `w8a8_block_int8_matmul`                                            | Triton ✅ TileLang ❌ | Triton @xukang @duyubo                        | Triton                       | [docs](docs/ops/gemm.md)                                                                           |
| DeepGemm 系列                                                                     |                                                                                                                                                                                                                                        |                     | Triton @zhenxin Tilelang @zhouweiwang         |                              |                                                                                                    |




## 目录

```text
bolt_ops/
  fla/               # Flash Linear Attention 总目录
    kda/             # FLA/KDA 系列（Triton @liji）
    gdn/             # FLA/GDN 系列（Triton @huangteng）
    common/          # FLA 各子系列共用 helper
  <op>/              # 每个算子一个目录
    __init__.py      # 公开接口 + 出厂后端分流（lazy-import）
    triton/          # Triton 实现（含 configs）
    tilelang/        # TileLang 实现（预留）
  tools/             # 随包发布的开发工具（含 MoE autotune）
  utility/           # 公共运行时工具
docs/ops/            # 算子文档
test/                # 正确性 + 可选 --bench 性能；不进入 wheel
build_wheel.sh       # 打 wheel
```



## 开发

1. **关于 BoltOps 算子的定位**
  由于是 DSL 语言开发的算子库，我们一定是在保证通用性的基础上去追求特定 size 下的性能，这是开发初衷。当然，具体到某一个算子，针对特定 size 或场景，可以由不同 kernel + DSL语言 混合实现以追求极致性能优化。
2. **关于 Triton 和 TileLang**
  二者都是类 Python DSL，目前看是相互补充、相互协作的关系，在 BoltOps 里也应有所体现：


|     | Triton                                          | TileLang                                                  |
| --- | ----------------------------------------------- | --------------------------------------------------------- |
| 特点  | 生态好、兼容性好，性能拉升快；简单改动能达到手写 HIPC / asm 约 80% 及以上水平 | 新起之秀，提供 warp / thread 级细粒度接口，开发速度快；面向具体场景手写可对标 HIPC / asm |
| 局限  | 再往上抠极致较难                                        | 生态与通用性不如 Triton                                           |


   因此同一算子可以并存 Triton / TileLang 实现，出厂后端按 bake-off 结果选择（或混合策略）。
3. **关于算子接口定义**
  优先与 vLLM、SGLang 等相关开源框架（最新版本）的接口对齐，这样适配会比较简单；如果没有相关接口定义，再参考 HIPC / CUDA C 相关接口对齐。
4. **关于测试**
  每个算子有一个测试文件 / 夹，`test` 和 `bench` 放在一起：`test` 侧重接口功能性（普适性），`bench` 专注实际网络 shape 的性能；二者相辅相成，并不冲突。

```bash
pip install -e ".[triton,dev]"
pytest test                  # 正确性（功能 / case 覆盖）
pytest test --bench -s       # 正确性 + 真实模型尺寸性能
# 单算子：
# pytest test/test_generic_recompute_w_u.py
# pytest test/test_generic_recompute_w_u.py --bench -s
```

**新算子：**

1. 新建 `bolt_ops/<op>/`，在 `triton/` 与 `tilelang/` 实现；`test/test_<op>.py` 写正确性，`@pytest.mark.bench` 写模型尺寸测速，用 `--bench` 比出最优。
2. 公开签名优先对齐 vLLM / SGLang（或 HIPC / CUDA C）；在 `bolt_ops/<op>/__init__.py` 写出厂后端分流并导出。
3. 在 `bolt_ops/__init__.py` 再导出，并补 `docs/ops/`。
