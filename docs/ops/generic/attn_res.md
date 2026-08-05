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
from bolt_ops.generic import attn_res

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

Kimi-K3 attention residual 的 hidden size 为 `7168`。实际网络输入为
`prefix=[4096,7168]`、`blocks=[4096,5,7168]`，其中 token 数还覆盖
`1～1024` 和 `8192`，block 数在 `1～8` 之间变化：

| 场景 | `prefix` | `blocks` | `num_blocks` | dtype |
|---|---|---|---:|---|
| Decode / small token | `[1～1024,7168]` | `[1～1024,1～8,7168]` | `1–8` | BF16 |
| 实际网络点 | `[4096,7168]` | `[4096,5,7168]` | `5` | BF16 |
| Long prefill | `[8192,7168]` | `[8192,5,7168]` | `5` | BF16 |

正确性测试另覆盖 `D=128/1024`、block capacity 大于有效 block 数，以及最后一维
连续但 row 带 padding 的输入。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/test_generic_attn_res.py

# Decode / small-batch / prefill 性能
pytest -q test/test_generic_attn_res.py --bench -s
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
| 公开接口 | `bolt_ops/generic/__init__.py` |
| Triton 实现 | `bolt_ops/generic/triton/attn_res.py` |
| 测试 / 性能 | `test/test_generic_attn_res.py`（`--bench`） |

范围：`attn_res`、上游 `_attn_res_kernel` 与本地
`_attn_res_kernel_lazy_prefix` sibling；不含 NVIDIA CUDA 实现，不含 backward。

## 附录 B. 溯源

- vLLM PR：[`#50000`](https://github.com/vllm-project/vllm/pull/50000)
- 固定来源 commit：[`5a3eba03481d879451d89b238f9160fa1c1faddf`](https://github.com/vllm-project/vllm/commit/5a3eba03481d879451d89b238f9160fa1c1faddf)
- 源文件：[`vllm/models/kimi_k3/amd/ops/attn_res.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/vllm/models/kimi_k3/amd/ops/attn_res.py)
- 正确性参考：[`tests/models/kimi_k3/test_amd_attn_res.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/tests/models/kimi_k3/test_amd_attn_res.py)

## 附录 C. 实现备注

- Grid 为 `(num_tokens,)`，每个 Triton program 处理一个 token。
- `BLOCK_D=next_power_of_2(hidden_size)`；超出 `D` 的 lane 使用 mask。
- source 数为 `num_blocks+1`，最后一路固定为 prefix。
- BW1100 离线静态配置共三组：小 token 使用 `(BLOCK_L=4,w8,s2)`；大 token
  且 block 数为 2～3 使用 `(BLOCK_L=4,w4,s2)`；其余大 token 使用
  `(BLOCK_L=2,w4,s2)`。源码不使用 `@triton.autotune`。
- `T>=320,num_blocks>=4` 使用 lazy-prefix sibling，只在 constexpr 最后一个
  source tile 加载 prefix；其他 shape 回退上游 kernel，避免小 token 的噪声级回退。
- RMSNorm、logit、online softmax 与 residual 加权在一次 Kernel launch 内完成。
- 上游 kernel 保持不改，静态配置与结构优化放在本地 selector/sibling，便于后续同步。

## 附录 D. Benchmark Shape

为避免对 `T×num_blocks` 做无意义的全笛卡尔积，benchmark 分别扫描两个维度，
并保留原始三组 case：

| Case 组 | `T` | `num_blocks` | `D` |
|---|---|---|---:|
| 原始 case | `1`、`17`、`320` | `8` | 7168 |
| Token sweep | `1`、`16`、`64`、`256`、`1024`、`4096`、`8192` | `5` | 7168 |
| Block sweep | `4096` | `1～8` | 7168 |

Token sweep 与 block sweep 共享的 `T=4096,num_blocks=5` 只运行一次，因此总计
17 组 benchmark。

## 附录 E. BW1100 离线调优记录

### E.1 静态配置

离线搜索覆盖 `BLOCK_L={1,2,4,8}`、`num_warps={2,4,8,16}`、
`num_stages={1,2,3}`，最终只保留上文三组静态配置。运行时根据 `T` 和
`num_blocks` 直接选择一组并 launch 一次，没有首次在线试跑或缓存依赖。

### E.2 结构优化与性能

lazy-prefix 保持 blocks→prefix 的 online-softmax 顺序、FP32 RMS/logit/softmax
归约、任意 `D` tail 和 source mask。它去掉非最后 tile 中不可能成立的 prefix
分支，并将 prefix 的 live range 延后到最后 tile。

| 证据 | 静态配置基线 | lazy-prefix | 加速 / 资源 |
|---|---:|---:|---:|
| XCU `T4096/B5/D7168` | 356.96 μs | 338.24 μs | 1.0553x |
| XCU HBM | 1151.70 GB/s | 1215.43 GB/s | +5.53% |
| VGPR / scratch | 190 / 0 | 155 / 0 | VGPR -18.4% |
| 17-shape AB/BA 几何平均 | — | — | 1.0179x |

17 组均通过 PyTorch oracle，最大绝对误差不超过 `0.03125`。实际网络点相对
静态配置基线：`T4096/B5` 为 `1.0456x`，`T8192/B5` 为 `1.0536x`；再叠加
本轮静态配置替换旧分派后，这两个点相对旧实现约提升 28%～29%。

未采用方案：FP32 `norm_weight*qk_weight` 预计算回退 6.98%，prefix-first
回退 1.40%，blocks 后单独处理 prefix 明显回退；未尝试 BF16/MMAC 降精度。

### E.3 总体平均口径

相对本轮前的旧分派，三组静态配置在 17 shape 上的几何平均为 `1.1125x`；只在
有收益范围启用 lazy-prefix 后，结构优化因子为 `1.0217x`，因此 attn_res 的
全 shape 综合因子约为 `1.1366x`。结合 KDA 十个独立 kernel dispatch 的
`1.572461x` 平均，十一项总体算术平均约为：

```text
(10 * 1.572461 + 1.1366) / 11 ≈ 1.532837x
```

即按“所有 kernel 独立计数”的验收口径平均提升约 `53.28%`。若 KDA 的 inter
solve 与 diagonal inverse 去重，完整 pipeline 的保守口径约为 `45.45%`；两种
口径均保留，避免把工作前移重复计功。
