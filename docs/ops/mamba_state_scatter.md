# `fused_mamba_state_scatter_with_mask` 使用说明

MTP verify 后，将接受的 draft step 对应的临时 Mamba state 写回长期 cache；
`step_indices_raw < 0` 的请求跳过。融合原先的 `nonzero` / `index_select` /
advanced-indexing scatter 为一次 kernel launch。

## 1. 算子用途与数学语义

对每个有效请求 `r`、layer `l`：

```text
s = step_indices_raw[r]
d = dst_indices_raw[r]
dst[l, d, ...] = src[l, r, s, ...]
```

仅当以下条件全部成立时写入：

```text
s >= 0
0 <= d < dst.shape[1]
r < src.shape[1]
s < src.shape[2]
```

- 算子**原地**修改 `dst`，无返回值
- 多个请求写入同一 destination row 时存在并行写竞争，结果未定义；调用方须保证
  有效 destination index 唯一

## 2. 接口说明

```python
from bolt_ops import fused_mamba_state_scatter_with_mask

fused_mamba_state_scatter_with_mask(
    dst, src, dst_indices_raw, step_indices_raw
)
```

函数签名：

```python
def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    ...
```

### 2.1 参数含义

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `dst` | `[num_layers, cache_size, *state_shape]` | FP32/BF16 | 长期 state cache（原地更新） |
| `src` | `[num_layers, spec_size, draft_steps, *state_shape]` | 同 `dst` | speculative 临时 state |
| `dst_indices_raw` | `[requests]` | int32/int64 | 每个请求写入的 cache row |
| `step_indices_raw` | `[requests]` | int32/int64 | 选中的 draft step；负数表示无效 |

说明：`request` 下标即 `src` 第 1 维；接口无单独的 `src_indices`。

### 2.2 前置条件

- `dst` 与 `src` 位于同一 GPU device
- `dst` / `src` 的 `num_layers` 与 trailing `state_shape` 一致，且均为 **contiguous**
- 入口将 index tensor 转为 contiguous `int32`
- 有效 `dst_indices` 在调用方侧应唯一

## 3. 典型 Shape

| 场景 | `dst` | `src` | index | dtype |
|---|---|---|---|---|
| SSM（TP=2） | `[30, 287, 16, 128, 128]` | `[30, 49, 4, 16, 128, 128]` | `[requests]` | FP32 |
| Conv（TP=2） | `[30, 287, 4096, 3]` | `[30, 49, 4, 4096, 3]` | `[requests]` | BF16 |
| SSM（TP=4） | `[30, 287, 8, 128, 128]` | `[30, 49, 4, 8, 128, 128]` | `[requests]` | FP32 |
| Conv（TP=4） | `[30, 287, 2048, 3]` | `[30, 49, 4, 2048, 3]` | `[requests]` | BF16 |

其中 `30/287/49/4` 依次为 layer 数、cache 容量、临时 request 容量、draft step 数。
`requests` 常为 `1–32`（高频 32）；可小于 `src.shape[1]`。更多抓取明细见附录。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/test_mamba_state_scatter.py

# 性能（真实模型尺寸；与正确性 case 集不同）
pytest -q test/test_mamba_state_scatter.py --bench -s
```

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `bolt_ops/mamba_state_scatter/__init__.py` |
| Triton 实现 | `bolt_ops/mamba_state_scatter/triton/mamba_state_scatter.py` |
| 测试 / 性能 | `test/test_mamba_state_scatter.py`（`--bench`） |

范围：公开入口与 `_fused_mamba_state_scatter_with_mask_kernel`；不含同文件内
state tracking / conv-window scatter。

## 附录 B. 溯源

- SGLang commit：[`132ade55cd65c3679a60adec563c8e6ca81238c5`](https://github.com/sgl-project/sglang/commit/132ade55cd65c3679a60adec563c8e6ca81238c5)
- 源文件：[`mamba_state_scatter_triton.py` L128–L301](https://github.com/sgl-project/sglang/blob/132ade55cd65c3679a60adec563c8e6ca81238c5/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py#L128-L301)

## 附录 C. 实现备注

- Grid：`(ceil(elem_per_entry / BLOCK_SIZE), num_layers, requests)`；trailing
  state 展平为 `elem_per_entry`。
- BW1000_H 固定两档配置：`elem_per_entry < 131072` → `BLOCK=1024, warps=2`；
  否则 `BLOCK=4096, warps=16`；无运行时 autotune。
- 大 SSM payload 已接近同流量连续 copy 带宽上限。

## 附录 D. 需求 shape 抓取明细

抓取日志中目标算子约 1038 次（SSM / Conv 各半）。高权重：

| 类型 | `dst` | `src` | dtype |
|---|---|---|---|
| SSM | `[30, 287, 16, 128, 128]` | `[30, 49, 4, 16, 128, 128]` | FP32 |
| Conv | `[30, 287, 4096, 3]` | `[30, 49, 4, 4096, 3]` | BF16 |

测试 / benchmark 常用 `requests ∈ {1, 4, 16, 31, 32}`；index 含 int32 与 int64。
一套高权重 FP32 `dst+src` 约 14 GiB；带 reference 的正确性测试建议 ≥24 GiB 空闲显存。
