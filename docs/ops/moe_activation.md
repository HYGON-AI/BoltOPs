# `moe_activation` 使用说明

`moe_activation` 是 `fused_moe` 目录内、用于第一层 GEMM 之后的 Triton
激活能力。公开接口采用显式输出 Tensor，适合推理框架复用缓存。

## 1. 公开接口

```python
from bolt_ops.fused_moe import moe_activation

# input: [M, 2D]，按 [gate, up] 打包；output: [M, D]
moe_activation(output, input, activation="silu")

# 非 gated 激活不改变 shape
moe_activation(output, input, activation="relu2")
```

签名：

```python
def moe_activation(
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    activation: str = "silu",
    is_gated: Optional[bool] = None,
    *,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    activation_situ_beta: Optional[float] = None,
    activation_situ_linear_beta: Optional[float] = None,
) -> None:
    ...
```

### 1.1 aiter / 框架集成

`moe_c`、`moe_asm`、`moe_triton` 已经持有第一层 GEMM 输出和激活输出缓存，
可直接使用同一个 `moe_activation`：

```python
from bolt_ops.fused_moe import (
    moe_activation,
    moe_activation_output_size,
    normalize_moe_activation,
)

activation, is_gated = normalize_moe_activation(activation, is_gated)
output_n = moe_activation_output_size(ffn1_out_2d.shape[-1], is_gated)
activated_out = torch.empty(
    (ffn1_out_2d.shape[0], output_n),
    device=ffn1_out_2d.device,
    dtype=ffn1_out_2d.dtype,
)

moe_activation(
    activation=activation,
    is_gated=is_gated,
    activated_out=activated_out,
    ffn1_out_2d=ffn1_out_2d,
    gemm1_alpha=gemm1_alpha,
    gemm1_limit=gemm1_limit,
    activation_situ_beta=activation_situ_beta,
    activation_situ_linear_beta=activation_situ_linear_beta,
)
```

参数名与 aiter 三条 MoE 路径原先调用 `_apply_activation` 时使用的关键字一致，
迁移时只需替换 import 和函数名。

## 2. 支持的激活

| `activation` | gated 默认值 | 输入布局 | 计算 |
| --- | --- | --- | --- |
| `silu` | True | `[gate, up]` | `SiLU(gate) * up` |
| `gelu` | True | `[gate, up]` | exact `GELU(gate) * up` |
| `gelu_tanh` | True | `[gate, up]` | tanh GELU `* up` |
| `situ` | True | `[gate, up]` | Kimi SituGLU；必须给 `activation_situ_beta` |
| `swigluoai` | True | `[g0,u0,g1,u1,...]` | OpenAI interleaved SwiGLU |
| `swiglustep` | True | `[gate, up]` | clamp 后的 step SwiGLU |
| `silu_no_mul` | False | 普通 Tensor | SiLU |
| `gelu_no_mul` | False | 普通 Tensor | exact GELU |
| `gelu_tanh_no_mul` | False | 普通 Tensor | tanh GELU |
| `relu2` / `relu2_no_mul` | False | 普通 Tensor | `ReLU(x)^2` |

别名 `gelu_pytorch_tanh` 和 `gelu_pytorch_tanh_no_mul` 会分别归一化为
`gelu_tanh` 与 `gelu_tanh_no_mul`。对 `silu`、`gelu`、`gelu_tanh` 显式传
`is_gated=False` 可调用对应的非 gated 版本。

## 3. 输入约束

- 仅支持 CUDA/HIP 上的浮点 Tensor。
- `ffn1_out_2d` / `activated_out` 必须 contiguous，并位于同一设备。
- gated 输入最后一维必须为偶数，输出最后一维是输入的一半。
- `moe_activation` 接收 2D `[rows, N]` 输入与
  `[rows, N/2]`（gated）或 `[rows, N]`（非 gated）输出。
- 算子用于推理前向；当前不提供 autograd backward。

## 4. 测试与性能

```bash
pytest test/test_moe_activation.py
pytest test/test_moe_activation.py --bench -s
```

正确性覆盖 gated / non-gated、aiter 显式输出调用约定、SITU 与 SwiGLU
变体；benchmark 使用 `[3394, 3072] -> [3394, 1536]` BF16 shape。

## 5. 文件

| 用途 | 文件 |
| --- | --- |
| 后端无关公开接口 | `bolt_ops/fused_moe/__init__.py` |
| Triton 实现 | `bolt_ops/fused_moe/triton/moe_activation.py` |
| 正确性 / 性能测试 | `test/test_moe_activation.py` |
