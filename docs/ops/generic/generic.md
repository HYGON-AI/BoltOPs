# Generic 系列算子说明

Generic 用于归类不属于 Gemm、MoE、KDA 等专用系列的通用融合算子。目前包含
注意力残差融合、Gated Delta Net 的 W/U 重计算、带 mask 的 Mamba state
scatter，以及 causal conv1d（fwd + update，统一入口 + SGLang/vLLM 双 port）。

公开 API 统一从 `boltops.generic` 导入：

```python
from boltops.generic import (
    attn_res,
    causal_conv1d_fn,
    causal_conv1d_update,
    recompute_w_u_fwd,
    fused_mamba_state_scatter_with_mask,
)
```

## 当前支持一览

| 公开 API | 功能 | 公开包 | 实现 | 详细文档 |
| --- | --- | --- | --- | --- |
| `attn_res` | 对 prefix 与多路 residual 做 RMSNorm、softmax 打分和加权融合 | `boltops.generic` | Triton | [注意力残差](attn_res.md) |
| `recompute_w_u_fwd` | 根据 `k`、`v`、`beta`、`g_cumsum` 和 chunk 矩阵 `A` 重建 WY 张量 | `boltops.generic` | Triton | [recompute W/U](recompute_w_u.md) |
| `fused_mamba_state_scatter_with_mask` | 按目标索引和 step mask 更新 Mamba state cache | `boltops.generic` | Triton | [Mamba state scatter](mamba_state_scatter.md) |
| `causal_conv1d_fn` | varlen 连续批处理的 causal depthwise conv1d（extend/prefill），自动路由 SGLang/vLLM port | `boltops.generic` | Triton | [causal conv1d](causal_conv1d.md) |
| `causal_conv1d_update` | decode 单步的 causal conv1d + state 更新，自动路由 SGLang/vLLM port | `boltops.generic` | Triton | [causal conv1d](causal_conv1d.md) |

## 接口与实现边界

- `boltops` 根包只公开 `__version__`，不再转发具体算子接口。
- Generic 算子的公开入口统一位于 `boltops/generic/__init__.py`；通常使用时不
  直接依赖 `triton/` 下的实现文件。
- Generic 中的算子彼此独立，没有共享的融合入口或统一函数签名；causal conv1d
  是例外：`causal_conv1d_fn` / `causal_conv1d_update` 是唯一入口，内部按参数
  自动路由到 SGLang / vLLM 两套 Triton port；常量、PDL stub 与 torch 参考实现
  在 `triton/causal_conv1d/common.py` 共享，kernel 本体保持两套可 diff 上游的
  移植。
- 合并决策由 bake-off 数据驱动：`pytest test/test_generic_causal_conv1d.py
  --bench -s` 在相同真实 shape 上对比两个后端，判定标准见
  [causal conv1d](causal_conv1d.md)。
- `recompute_w_u_fwd` 是独立的 Gated Delta Net WY 重计算实现，与
  `boltops.fla.kda.recompute_w_u_fwd` 的逐通道 gate 语义和返回值不同，不能互换。

## 测试与性能基准

```bash
# 正确性与接口测试
pytest -q test/test_generic_attn_res.py
pytest -q test/test_generic_recompute_w_u.py
pytest -q test/test_generic_mamba_state_scatter.py
pytest -q test/test_generic_causal_conv1d.py

# 各算子的 benchmark case
pytest -q test/test_generic_attn_res.py --bench -s
pytest -q test/test_generic_recompute_w_u.py --bench -s
pytest -q test/test_generic_mamba_state_scatter.py --bench -s
pytest -q test/test_generic_causal_conv1d.py --bench -s
```

各算子的数学语义、参数约束、典型 shape、溯源及 benchmark shape 见上表中的
详细文档。
