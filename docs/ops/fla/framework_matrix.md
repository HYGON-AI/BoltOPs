# FLA 框架 × 算子族矩阵

BoltOPs 按算法家族组织实现：`gdn` 是 GDN 家族 API，`kda` 是 KDA 家族 API，
`common` 是 GDN/KDA 共享 Triton kernel。接入时先看下表的框架和算子族组合，
不需要额外的 framework compatibility 包。

## 四个组合

| 框架 / 算子族 | 推荐导入 | 调用入口 / 关键参数 | 返回值 / 默认 `null_state_index` | 主要场景 |
|---|---|---|---|---|
| vLLM + GDN | `from boltops.fla.gdn import fused_sigmoid_gating_delta_rule_update` | `fused_sigmoid_gating_delta_rule_update(..., is_kda=False)` | `(output, final_state)`；`0` | continuous batching 和 linear/speculative decode |
| vLLM + KDA | `from boltops.fla.kda import fused_recurrent_kda_packed_decode` | packed QKV decode；也可按场景用 `fused_recurrent_kda_fwd` / `fused_recurrent_kda` | `(output, state)`；`0` | Kimi-K3 KDA packed/固定/变长 decode |
| SGLang + GDN | `from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update` | `fused_sigmoid_gating_delta_rule_update(..., is_kda=False)` | `output`；`-1` | SGLang GDN decode/verify |
| SGLang + KDA | `from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update` | `fused_sigmoid_gating_delta_rule_update(..., is_kda=True)`；target verify 额外传 `lower_bound=-5.0`、`disable_state_update=True`、intermediate state 参数 | `output`；`-1` | SGLang KDA decode 和 MTP `target_verify` |

## 语义要点

- `target_verify` 不是新的算子名；它是 SGLang adapter 对共享 decode/verify
  kernel 的一种调用模式。
- vLLM 的 GDN/KDA decode ABI 分别保留在 `gdn` 和 `kda`。
- SGLang 的 GDN/KDA decode/verify 共享 kernel 保留在 `common`，通过
  `is_kda` 选择算法族。
- framework adapter 必须显式确认 `null_state_index` 语义，不能混用默认值：
  vLLM 使用 `0`，SGLang 使用 `-1`。

## 源码位置

| 组合 / 共享实现 | 源码 |
|---|---|
| vLLM + GDN decode | `boltops/fla/gdn/triton/fused_sigmoid_gating.py` |
| vLLM + KDA decode | `boltops/fla/kda/triton/fused_recurrent.py` |
| SGLang + GDN/KDA decode/verify | `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py` |

相关详细文档：

- GDN：[`docs/ops/fla/gdn.md`](gdn.md)
- KDA：[`docs/ops/fla/kda.md`](kda/kda.md)
- SGLang KDA `target_verify`：[`docs/ops/fla/kda/kda_target_verify.md`](kda/kda_target_verify.md)
