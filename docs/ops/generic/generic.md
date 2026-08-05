# Generic 系列算子说明

Generic 用于归类不属于 Gemm、MoE、KDA 等专用系列的通用融合算子。目前包含
注意力残差融合、Gated Delta Net 的 W/U 重计算，以及带 mask 的 Mamba state
scatter。

公开 API 统一从 `bolt_ops.generic` 导入：

```python
from bolt_ops.generic import (
    attn_res,
    recompute_w_u_fwd,
    fused_mamba_state_scatter_with_mask,
)
```

## 当前支持一览

| 公开 API | 功能 | 公开包 | 实现 | 详细文档 |
| --- | --- | --- | --- | --- |
| `attn_res` | 对 prefix 与多路 residual 做 RMSNorm、softmax 打分和加权融合 | `bolt_ops.generic` | Triton | [注意力残差](attn_res.md) |
| `recompute_w_u_fwd` | 根据 `k`、`v`、`beta`、`g_cumsum` 和 chunk 矩阵 `A` 重建 WY 张量 | `bolt_ops.generic` | Triton | [recompute W/U](recompute_w_u.md) |
| `fused_mamba_state_scatter_with_mask` | 按目标索引和 step mask 更新 Mamba state cache | `bolt_ops.generic` | Triton | [Mamba state scatter](mamba_state_scatter.md) |

## 接口与实现边界

- `bolt_ops` 根包只公开 `__version__`，不再转发具体算子接口。
- Generic 算子的公开入口统一位于 `bolt_ops/generic/__init__.py`；通常使用时不
  直接依赖 `triton/` 下的实现文件。
- Generic 中的三个算子彼此独立，没有共享的融合入口或统一函数签名。
- `recompute_w_u_fwd` 是独立的 Gated Delta Net WY 重计算实现，与
  `bolt_ops.fla_kda.recompute_w_u_fwd` 的逐通道 gate 语义和返回值不同，不能互换。

## 测试与性能基准

```bash
# 正确性与接口测试
pytest -q test/test_generic_attn_res.py
pytest -q test/test_generic_recompute_w_u.py
pytest -q test/test_generic_mamba_state_scatter.py

# 各算子的 benchmark case
pytest -q test/test_generic_attn_res.py --bench -s
pytest -q test/test_generic_recompute_w_u.py --bench -s
pytest -q test/test_generic_mamba_state_scatter.py --bench -s
```

各算子的数学语义、参数约束、典型 shape、溯源及 benchmark shape 见上表中的
详细文档。
