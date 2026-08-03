# `attn_res 使用说明`

注意力残差融合算子：对 prefix 与多路 residual 分别做 RMSNorm 和打分，在 source
维度执行 softmax，再将各路原始 residual 加权合并为一路输出。

## 1. 算子用途与数学语义

对每个 token，设有效 residual block 数为 `R=num_blocks`，hidden size 为 `D`：

```text
values = concat(blocks[:R], prefix[None], dim=source)  # [R+1, D]

keys_i   = rms_norm(values_i, norm_weight, eps)
logit_i  = dot(keys_i, qk_weight)
weight_i = softmax(logit)_i

output = sum_i(weight_i * values_i)
```

- prefix 作为最后一路 source 参与归一化、打分和加权
- softmax 权重只用于混合，实际累加的是归一化前的 `values`
- Kernel 使用 online softmax，遍历 source tile 时维护最大 logit、分母和加权和
- 输入不被修改；输出 shape/dtype 与 `prefix` 一致

## 2. 接口说明

```python
from bolt_ops import attn_res

output = attn_res(
    prefix=prefix,
    blocks=blocks,
    norm_weight=norm_weight,
    qk_weight=qk_weight,
    num_blocks=num_blocks,
    eps=1e-5,
)
```

函数签名：

```python
def attn_res(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    ...
```

### 2.1 参数含义

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `prefix` | `[T,D]` | BF16 | 每个 token 的 prefix residual |
| `blocks` | `[T,capacity,D]` | BF16 | residual block 容器；只读取前 `num_blocks` 路 |
| `norm_weight` | `[D]` | BF16 | 每路 source 使用的 RMSNorm weight |
| `qk_weight` | `[D]` | BF16 | 将归一化 source 投影为标量 logit 的权重 |
| `num_blocks` | scalar | int | 有效 residual block 数；要求 `1 <= num_blocks <= capacity` |
| `eps` | scalar | float | RMSNorm epsilon |
| 返回 `output` | `[T,D]` | 同 `prefix` | softmax 加权后的融合 residual |

### 2.2 前置条件

- 所有 tensor 位于同一 CUDA/HIP device
- `prefix.shape[0] == blocks.shape[0]`
- `norm_weight.numel() == qk_weight.numel() == D`
- `prefix`、`blocks`、`norm_weight`、`qk_weight` 的最后一维连续
- 外层 row 允许 padding stride，因此无需整 tensor contiguous
- 当前接口要求至少一个 block；`num_blocks=0` 不支持
- `T=0` 合法，返回 shape `[0,D]` 的空 tensor
- 算子为 forward-only，不包含 backward

## 3. 典型 Shape

Kimi-K3 attention residual 的典型 hidden size 为 `7168`，最多使用 8 路 block：

| 场景 | `prefix` | `blocks` | `num_blocks` | dtype |
|---|---|---|---:|---|
| Decode | `[1,7168]` | `[1,8,7168]` | `1–8` | BF16 |
| Small batch | `[17,7168]` | `[17,8,7168]` | `1–8` | BF16 |
| Prefill | `[320,7168]` | `[320,8,7168]` | `1–8` | BF16 |

正确性测试另覆盖 `D=128/1024`、block capacity 大于有效 block 数，以及最后一维
连续但 row 带 padding 的输入。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/test_attn_res.py

# Decode / small-batch / prefill 性能
pytest -q test/test_attn_res.py --bench -s
```

PyTorch reference 等价于：

```python
values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
keys = torch.nn.functional.rms_norm(values, (hidden_size,), norm_weight, eps)
probs = (keys @ qk_weight).softmax(dim=-1)
expected = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
```

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `bolt_ops/attn_res/__init__.py` |
| Triton 实现 | `bolt_ops/attn_res/triton/attn_res.py` |
| 测试 / 性能 | `test/test_attn_res.py`（`--bench`） |

范围：`attn_res` / `_attn_res_kernel`；不含 NVIDIA CUDA 实现，不含 backward。

## 附录 B. 溯源

- vLLM PR：[`#50000`](https://github.com/vllm-project/vllm/pull/50000)
- 固定来源 commit：[`5a3eba03481d879451d89b238f9160fa1c1faddf`](https://github.com/vllm-project/vllm/commit/5a3eba03481d879451d89b238f9160fa1c1faddf)
- 源文件：[`vllm/models/kimi_k3/amd/ops/attn_res.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/vllm/models/kimi_k3/amd/ops/attn_res.py)
- 正确性参考：[`tests/models/kimi_k3/test_amd_attn_res.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/tests/models/kimi_k3/test_amd_attn_res.py)

## 附录 C. 实现备注

- Grid 为 `(num_tokens,)`，每个 Triton program 处理一个 token。
- `BLOCK_D=next_power_of_2(hidden_size)`；超出 `D` 的 lane 使用 mask。
- source 数为 `num_blocks+1`，最后一路固定为 prefix。
- `T>=256` 或 `num_blocks<=1` 时使用 `BLOCK_L=1,num_warps=4`；否则使用
  `BLOCK_L=4,num_warps=8`。
- RMSNorm、logit、online softmax 与 residual 加权在一次 Kernel launch 内完成。
- 本地实现仅将 `vllm.triton_utils` 替换为直接 Triton import；Kernel 与 wrapper
  已按 AST 和固定来源 commit 核对，无计算逻辑修改。

## 附录 D. Benchmark Shape

| Case | `T` | `num_blocks` | `D` |
|---|---:|---:|---:|
| `decode` | 1 | 8 | 7168 |
| `small-batch` | 17 | 8 | 7168 |
| `prefill` | 320 | 8 | 7168 |
