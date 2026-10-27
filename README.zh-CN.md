# BoltOPs

Backend Optimized Low-latency Triton/TileLang Operators

[English](README.md) | **中文**

## 介绍

**BoltOPs**（*Backend Optimized Low-latency Triton/TileLang Operators*）是 AI Compiler 团队面向海光 HCU 的高性能算子库。命名同时取自 **lightning bolt（闪电）**：以低时延为目标，追求闪电般的执行速度。它以 Triton、TileLang 等 DSL 实现可量产的、由 benchmark 驱动的 kernel，并通过清晰的 Python API 对外提供，让推理框架与模型侧调用方无需后端胶水即可获得调优后的 HCU 吞吐。

库聚焦时延敏感的融合算子、正确性优先的验证，以及便于部署的打包方式：该走快路径的地方走快路径，该降低接入成本的地方尽量少摩擦。

![BoltOp is Fastest](docs/boltop_is_fastest.jpg)

|            | 名称              |
| ---------- | ----------------- |
| 仓库       | `BoltOPs`         |
| DAS        | `boltops`         |
| PyPI / pip | `boltops`         |
| Wheel      | `boltops-*.whl`   |
| Import     | `boltops`         |

## 安装

```bash
pip install -e ".[triton]"          # 常用
pip install -e ".[triton,dev]"      # 开发 / 测试
pip install -e ".[tune]"            # MoE autotune / 测速工具
pip install -e ".[all]"             # Triton + TileLang
./build_wheel.sh                    # 打纯 Python wheel → dist/
# MoE autotune / 测速入口（随包发布）
# boltops-fused-moe-tune-cli <device_name> <test_type> [--benchmark]
```

依赖：`torch`；跑 Triton 算子还需本机可用的 Triton（Hygon 版）。MoE 调参见
`[docs/ops/fused_moe/fused_moe.md](docs/ops/fused_moe/fused_moe.md)` 与
`[tune_moe_triton_user_guide.md](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md)`。

## 用法

业务侧从对应算子包导入公开 API；根包 `boltops` 只暴露版本信息：

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
from boltops.fa import (
    extend_attention_fwd,
    extend_attention_fwd_unified,
    dense_prefill_attention_fwd,
    dense_prefill_attention_fwd_4d,
    dense_prefill_attention_fwd_segmented,
    decode_attention_fwd_4d,
    build_unified_kv_indices,
    can_use_dense_prefill_fp8,
    unified_attention,
    reshape_and_cache_int4,
    triangle_attention_fwd,
    KVQuantMode,
)
from boltops.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
from boltops.mqa_logits import (
    triton_mqa_logits,
    triton_fp8_paged_mqa_logits,
)
from boltops.qsa import (
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    expand_qsa_block_indices,
    qsa_sparse_paged_attention,
)
from boltops.step4_dsa import (
    triton_indexer_logits,
    triton_sparse_attention_prefill,
    triton_sparse_attention_decode,
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
    recompute_w_u_fwd as recompute_w_u_fwd_gdn,
)

mixed = attn_res(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)

# GDN / SGLang WY 重计算
w, u = recompute_w_u_fwd_gdn(
    k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, cu_seqlens=cu_seqlens
)

# KDA / vLLM WY 重计算
w, u, qg, kg = recompute_w_u_fwd_kda(k=k, v=v, beta=beta, A=A, gk=gk)

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

# Triangle Attention 前向；BF16 q/k/v 形状为 [B, H, N, D]，q 已乘 1/sqrt(D)
out = triangle_attention_fwd(q, k, v, [mask_bias, triangle_bias])

# Step-4 DSA（vendor 合同 step4-dsa-ops-v1）
logits = triton_indexer_logits(index_q, weights, index_k)
out, lse = triton_sparse_attention_decode(
    q, k_cache, v_cache, packed, counts, kv_seqlens,
    num_kv_groups=1,
)
```

需要做实现 A/B 时，直接 import 后端模块，不要绕公开 API：

```python
from boltops.generic.triton.attn_res import attn_res as triton_impl
```

## 公开算子（+待开发算子）

| 算子类别                                                                            | 接口名                                                                                                                                                                                                                                    | DSLs                | 维护者                                           | 出厂后端                         | 介绍和文档                                                                                              |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------- |
| Generic                                                                         | `attn_res` `fused_mamba_state_scatter_with_mask`<br>`causal_conv1d_fn`<br>`causal_conv1d_update`                                                                                                                                                                                        | Triton ✅ TileLang ❌ | Triton @all                                   | Triton                       | [docs](docs/ops/generic/generic.md)                                                                |
| FLA/KDA 系列                                                                      | `chunk_gla_fwd_o_gk` `chunk_kda_fwd_intra` `chunk_kda_fwd_intra_token_parallel` `fused_kda_gate` `fused_kda_gate_chunk_cumsum` `fused_recurrent_kda` `fused_recurrent_kda_fwd` `fused_recurrent_kda_packed_decode` `recompute_w_u_fwd` | Triton ✅ TileLang ❌ | Triton @liji                                  | Triton                       | [docs](docs/ops/fla/kda.md)                                                                        |
| FLA/GDN 系列                                                                      |                                                                                                                                                                                                                                        |                     | Triton @liji<br />TileLang @huangteng |                              |                                                                                                    |
| MoE 系列                                                                          | `fused_experts` `moe_activation`                                                                                                                                                                                                       | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx                 | Triton（`fused_experts` 混合策略） | [文档](docs/ops/fused_moe/fused_moe.md) [性能实测](docs/ops/fused_moe/performance_report.md) [调参指南](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md) |
| FA 系列（UnifiedAttention / PagedAttention / ExtendAttention/...） | `extend_attention_fwd` `extend_attention_fwd_unified` `dense_prefill_attention_fwd` `dense_prefill_attention_fwd_4d` `dense_prefill_attention_fwd_segmented` `decode_attention_fwd_4d` `build_unified_kv_indices` `can_use_dense_prefill_fp8`<br>`unified_attention` `reshape_and_cache_int4` (Quant type: `KVQuantMode`) | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao | Triton | [UnifiedAttention](docs/ops/fa/unified_attention.md) [ExtendAttention](docs/ops/fa/extend_attention.md) [DecodeAttention](docs/ops/fa/decode_attention.md) |
| Triangle Attention | `triangle_attention_fwd` | Triton ✅ TileLang ❌ | Triton @sunchunhao | Triton | [文档](docs/ops/fa/triangle_attention.md) |
| MLA / FlashMLA / Sparse MLA                                                     | `decode_attention_fwd` `decode_attention_fwd_grouped` `decode_attention_fwd_normal` `compute_num_kv_splits`                                                                                                                              | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao | Triton                       | [docs](docs/ops/mla.md)（目前 dense MLA decode；FlashMLA / Sparse 待迁）                              |
| Qwen Sparse GQA                                                                 | `qwen_sparse_kv_extraction_compact_hcu_fa_triton` `sparse_gqa_packed_decode_triton`                                                                                                                              | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao | Triton                       | [docs](docs/ops/sparse_gqa.md)                                                                      |
| Step-4 DSA                                                                      | `triton_build_rope_cache` `triton_indexer_norm_rope` `triton_csa_*` `triton_indexer_logits` `triton_region_topk_ids` `triton_topk_select_and_pack` `triton_*_sparse_meta` `triton_sparse_attention_*` `triton_merge_split_states`           | Triton ✅ TileLang ❌ | Triton @xukang @duyubo                                 | Triton                       | [docs](docs/ops/step4_dsa.md)                                                                      |
| MQA Logits                                                             | `triton_mqa_logits`（fp8 传 scales；fp16/bf16 不传）<br>`triton_fp8_paged_mqa_logits` / `_stage1`                                                                                                                                        | Triton ✅ TileLang ❌ | Triton @fangzx                            | Triton                       | [文档](docs/ops/mqa_logits_triton.md)（prefill + FP8 paged）                                                      |
| QSA（分页打分 / 选择 / 稀疏注意力）                                                             | `qsa_mqa_paged` `qsa_select_paged_tokens` `expand_qsa_block_indices` `qsa_sparse_paged_attention`                                                                                                                                | Triton ✅ TileLang ❌ |                                    | Triton             | [docs](docs/ops/qsa.md)（四步分页流水线；TileLang 打分入口在 `boltops.mqa_logits`）                     |
| MHC                                                                             | `mhc_pre_big_fuse` `pre_big_fuse_tilelang` `mhc_post_fwd` `hc_split_sinkhorn` `mhc_fused_tilelang`                                                                                                                                      | TileLang ✅          | TileLang @zhouweiwang                        | TileLang                     | [docs](docs/ops/mhc.md)                                                                            |
| IHC                                                                             | `ihc_pre` `ihc_post` `ihc_head`                                                                                                                                                                                                         | TileLang ✅          | TileLang @zhouweiwang                        | TileLang                     | [docs](docs/ops/ihc.md)                                                                            |
| Gemm/AwqGemm 系列                                                                 | `matmul_persistent` `gemm_a16w16` `w8a8_triton_block_scaled_mm` `triton_scaled_mm` `awq_gemm_triton` `gemm_a8w8_blockscale` `gemm_afp4wfp4` `batched_gemm_a16wfp4` `w8a8_block_int8_matmul`                                            | Triton ✅ TileLang ❌ | Triton @xukang @duyubo                        | Triton                       | [docs](docs/ops/gemm.md)                                                                           |
| DeepGemm 系列                                                                     |                                                                                                                                                                                                                                        |                     | Triton @zhenxin Tilelang @zhouweiwang         |                              |                                                                                                    |

## 目录

```text
boltops/
  fla/               # Flash Linear Attention 总目录
    kda/             # FLA/KDA 系列（Triton @liji）
    gdn/             # FLA/GDN 系列（Triton @huangteng）
    common/          # FLA 各子系列共用 helper
  <op>/              # 每个算子一个目录
    __init__.py      # 公开接口 + 出厂后端分流（lazy-import）
    triton/          # Triton 实现（含 configs）
    tilelang/        # TileLang 实现（预留）
  step4_dsa/         # Step-4 DSA 数值 Triton 算子（vendor 合同 v1）
    geometry.py      # 模型几何常量（不是 kernel）
    triton/          # kernel + launch helper
    tilelang/        # 占位
  tools/             # 随包发布的开发工具（含 MoE autotune）
  utility/           # 公共运行时工具
docs/ops/            # 算子文档
test/                # 正确性 + 可选 --bench 性能；不进入 wheel
build_wheel.sh       # 打 wheel
```

## 开发

1. **关于 BoltOPs 算子的定位**
   由于是 DSL 语言开发的算子库，我们一定是在保证通用性的基础上去追求特定 size 下的性能，这是开发初衷。当然，具体到某一个算子，针对特定 size 或场景，可以由不同 kernel + DSL语言 混合实现以追求极致性能优化。
2. **关于 Triton 和 TileLang**
   二者都是类 Python DSL，目前看是相互补充、相互协作的关系，在 BoltOPs 里也应有所体现：

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
# pytest test/fla/test_gdn_recompute_w_u.py
# pytest test/fla/test_gdn_recompute_w_u.py --bench -s
```

5. **optest**
  `optest` 是 CI 使用的统一算子测试入口，一次运行覆盖正确性（pytest 跑 `test/`）和性能（`scripts/run_operator_benchmarks.py`）：

```bash
pip install optest -i http://10.16.1.201:9929/nightly/dtk2604/+simple/ --trusted-host 10.16.1.201  --force-reinstall
optest --op_name boltops --test_dir . --test_mode all      # 正确性 + 性能
optest --op_name boltops --test_dir . --test_mode acc      # 只跑正确性
optest --op_name boltops --test_dir . --test_mode per      # 只跑性能
optest --op_name boltops --test_dir . --test_mode all --gpus 0,2  # 指定 HCU
optest --display                                            # 用最新结果生成 HTML 报告
```

  结果写入 `/op_test_output/boltops/<branch>_<commit>_<timestamp>/`
  （`.optest-results.json` + 各 kernel 的 CSV），日志在 `/op_test_log/boltops/`。
  CI（`performance:optest`）对每个进入 `main` 的 MR 运行同一入口
  （`--test_mode per`），并用 `ci_script/optest_to_junit.py` 把结果转成
  JUnit；正确性套件存在存量失败，acc 暂不作为门禁（修复后可升级为
  `--test_mode all`）。
  新增性能测例：在 `test/` 下正常写 `@pytest.mark.bench` 单测即可
  （不需要任何 perf 专属代码、不需要注册——perf 入口
  `scripts/run_operator_benchmarks.py` 自动发现）。运行时收集层
  （`test/conftest.py`）观测每次计时调用，按算子输出统一 CSV。完整
  约定（收集、CSV 列规范、可选增强）见 `docs/optest_perf.md`。
  本地跑子集：
  `python scripts/run_operator_benchmarks.py --ops causal_conv1d`。
  负责搜索/回写配置的调优脚本放 `scripts/tune/`，不接入 optest。

**新算子：**

1. 新建 `boltops/<op>/`，在 `triton/` 与 `tilelang/` 实现；`test/test_<op>.py` 写正确性，`@pytest.mark.bench` 写模型尺寸测速，用 `--bench` 比出最优。
2. 公开签名优先对齐 vLLM / SGLang（或 HIPC / CUDA C）；在 `boltops/<op>/__init__.py` 写出厂后端分流并导出。
3. 在 `boltops/__init__.py` 再导出，并补 `docs/ops/`。

## 许可证

BoltOPs 原创代码采用 [MIT 许可证](LICENSE)。基于第三方项目修改的部分保留其
原有许可证和版权声明；来源见文件头及模块说明：[GEMM](boltops/gemm/triton/SOURCE.md)、
[MLA](boltops/mla/triton/SOURCE.md)、[MQA logits](boltops/mqa_logits/triton/SOURCE.md)
和 [Step-4 DSA](boltops/step4_dsa/triton/SOURCE.md)。
