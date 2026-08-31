# Fused MoE 使用说明

融合 MoE experts 计算：对 `hidden_states` 按 `topk_ids` / `topk_weights` 路由，
经专家 `w1`/`w2`（及可选量化与 bias）得到输出。内部可走 Triton GEMM 路径，
并依赖私有 HIPC MoE sorting。

相关文档：

- [BoltOPs 与 Aiter 实测性能报告](performance_report.md)
- [Fused MoE Triton autotune 使用手册](../../../boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md)

## 1. 算子用途与数学语义

对每个 token，按其 top-k 专家执行（以 gated SiLU 为例）：

```text
# 对选中的 expert e、路由权重 w：
gate_up = x @ w1[e]^T (+ b1)
y = act(gate_up) @ w2[e]^T (+ b2)     # act 默认 silu，可 gated
out += w * y                          # 再按 topk 聚合；可选 no_combine
```

支持 BF16/FP16 以及 FP8 / INT8 / INT4 等权重量化路径（由 `use_*` 与 scale
参数选择）。`inplace=True` 时可写回 `hidden_states`。

## 2. 接口说明

第一层 GEMM 后的激活也可通过 `boltops.fused_moe.moe_activation` 独立调用。

```python
from boltops.fused_moe import fused_experts

out = fused_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation="silu",
    is_gated=True,
)
```

函数签名：

```python
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: Optional[str] = None,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    is_gated: Optional[bool] = None,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int4_w4a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    output_dtype: Optional[torch.dtype] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = 1.0,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> torch.Tensor:
    ...
```

### 2.1 参数含义

| 参数                                                   | Shape / 类型                                | 含义                                       |
| ------------------------------------------------------ | ------------------------------------------- | ------------------------------------------ |
| `hidden_states`                                      | `[M, K]`                                  | token 激活                                 |
| `w1`                                                 | `[E, N, K]`（gated 时 `N` 为 2×inter） | 第一层专家权重                             |
| `w2`                                                 | `[E, K, inter]`                           | 第二层专家权重                             |
| `topk_weights`                                       | `[M, topk]`                               | 路由权重                                   |
| `topk_ids`                                           | `[M, topk]`                               | 路由专家 id（通常 int32）                  |
| `inplace`                                            | bool                                        | 是否写回`hidden_states`                  |
| `activation`                                         | str / None                                  | 默认`"silu"`                             |
| `is_gated`                                           | bool / None                                 | 是否 gated（`w1` 含 gate/up）            |
| `b1` / `b2`                                        | optional                                    | bias                                       |
| `use_fp8_w8a8` 等                                    | bool                                        | 量化模式开关（互斥语义由实现约束）         |
| `per_channel_quant`                                  | bool                                        | per-channel 量化                           |
| `global_num_experts`                                 | int                                         | 全局专家数；`-1` 表示用 `E`            |
| `expert_map`                                         | optional                                    | EP 本地专家映射                            |
| `w1_scale` / `w2_scale` / `w*_zp` / `a*_scale` | optional                                    | 量化 scale / zero-point                    |
| `block_shape`                                        | optional                                    | block-wise 量化块形状                      |
| `output_dtype`                                       | optional                                    | 输出 dtype；默认跟激活                     |
| `no_combine`                                         | bool                                        | 为 True 时按 topk 维展开输出，不做加权求和 |
| `routed_scaling_factor`                              | float                                       | 路由缩放                                   |
| `gemm1_alpha` / `gemm1_limit`                      | optional                                    | 部分激活路径参数                           |

### 2.2 支持的 activation

| 类型 | `activation` |
| --- | --- |
| Gated / non-gated | `silu`、`gelu`、`gelu_tanh` |
| 仅 gated | `situ`、`swigluoai`、`swiglustep` |
| 仅 non-gated | `silu_no_mul`、`gelu_no_mul`、`gelu_tanh_no_mul`、`relu2`、`relu2_no_mul` |
| 别名 | `gelu_pytorch_tanh`、`gelu_pytorch_tanh_no_mul` |

### 2.3 前置条件

- 张量在 GPU 上；`hidden_states` contiguous；`w1`/`w2` 最后一维 stride 为 1
- `topk_weights.shape == topk_ids.shape`
- `hidden_states.shape[1]` 与 `w1` 的 K 维匹配（INT4 路径另有约定）
- 量化路径须提供对应 scale（及必要时 zp）；需本机 `aicc` 以编译 HIPC sorter
- 公开入口为 `fused_experts`；底层 `fused_experts_impl` 仅供内部 / 调参使用

## 3. 正确性测试与性能基准

线上 shape 的精度回归、autotune、以及性能测速统一走随包发布的
**`boltops-fused-moe-tune-cli`**（封装 `boltops/tools/fused_moe_triton_tune/tune_moe_cli.py`）。

安装后（推荐 `pip install -e ".[tune]"` 或安装带 `tune` extra 的 wheel）可在任意目录执行：

```bash
# 训练 + 生成 configs/triton_<capability> JSON + Step4 正确性/性能
boltops-fused-moe-tune-cli <device_name> <test_type>
# 例：boltops-fused-moe-tune-cli nmz bf16
#     boltops-fused-moe-tune-cli nmz int8_channel --num-experts 256 --top-k 8

# 仅测速 / 复核（复用已有 JSON，不重新 autotune）
boltops-fused-moe-tune-cli <device_name> <test_type> --benchmark

# 只看参数计划
boltops-fused-moe-tune-cli nmz bf16 --dry-run
```

wheel、普通安装和 editable 安装均可在任意目录执行，日志写到当前工作目录。

`test_type` 含 `bf16` / `fp8` / `fp8_channel` / `int8` / `int8_channel` /
`int4` / `int4int8` 等。参数、TP 循环、选卡与排障见一线手册：

[`tune_moe_triton_user_guide.md`](../../../boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md)

## 附录 A. 仓库文件

| 用途              | 文件                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| 公开接口          | `boltops/fused_moe/__init__.py`                                                                  |
| MoE JSON 配置     | `boltops/fused_moe/triton/configs/triton_<capability>/`                                         |
| 正确性 / 调参 / 测速 | `boltops-fused-moe-tune-cli` → `boltops.tools.fused_moe_triton_tune.tune_moe_cli:main` |
| Autotune 使用手册 | [`tune_moe_triton_user_guide.md`](../../../boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md) |
| 性能实测报告      | [`performance_report.md`](performance_report.md) |

## 附录 B. 溯源

由 ROCm **aiter** 的 Triton fused MoE 迁入（triton-only 路径）；HIPC sorting
为独立 in-tree 实现，不依赖 aiter / CK 运行时。

## 附录 C. 实现备注

- 公开 `fused_experts` 转发至 `fused_experts_impl`（带 `torch_compile_guard`）。
- Sorting 使用同目录下 `moe_sorting_hipc/` 的 `aicc` JIT；kernel 源码在 `csrc/`。
- Launch 配置优先读当前 `configs/triton_<capability>`，再向低能力版本回退。
