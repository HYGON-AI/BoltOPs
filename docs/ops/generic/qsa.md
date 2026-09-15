# QSA 分页选择与稀疏注意力

Generic QSA 提供四个互相独立、无需 vLLM 的公开 API：压缩 K cache 打分、
压缩 block 选择与 token 展开、单独展开，以及分页稀疏 GQA。调用方负责 cache
分配与写入、page table 和 request 映射；BoltOPs 不管理框架调度或 cache 生命周期。

```python
import torch

from boltops.generic import (
    expand_qsa_block_indices,
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
```

## 从 vLLM 迁移

BoltOPs 保持 vLLM QSA 的四步数据流，通常只需替换导入和函数名：

| vLLM 调用 | BoltOPs 调用 |
| --- | --- |
| `qsa_mqa_paged(...)` | `qsa_mqa_paged(...)` |
| `qsa_select_paged_tokens(...)` | `qsa_select_paged_tokens(...)` |
| `expand_qsa_block_indices_cuda(...)` | `expand_qsa_block_indices(...)` |
| `qsa_sparse_paged_attention(...)` | `qsa_sparse_paged_attention(...)` |

参数顺序和返回布局保持一致；BoltOPs 不需要 `vllm._custom_ops`，也不需要
传入 CUDA stream。典型替换如下：

```python
from boltops.generic import (
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    expand_qsa_block_indices,
    qsa_sparse_paged_attention,
)

scores, visible = qsa_mqa_paged(
    q, compressed_k, compressed_page_table, token_to_req,
    query_positions, sequence_lengths, compress_ratio=4,
)
blocks = torch.topk(scores, k=token_topk // 4, dim=1, sorted=False).indices
logical_indices = expand_qsa_block_indices(
    blocks, query_positions, sequence_lengths, token_to_req,
    compress_ratio=4, token_topk=token_topk,
)
output = qsa_sparse_paged_attention(
    attention_q, k_cache, v_cache, logical_indices, block_table, token_to_req,
)
```

`logical_indices` 中的 `-1`、非法 page 和 causal tail 仍由 sparse kernel 安全处理。

## API 与布局

### `qsa_mqa_paged`

```python
scores, visible_blocks = qsa_mqa_paged(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    compress_ratio,
    num_columns=None,
    score_scale=None,
)
```

| 参数或返回值 | dtype | 布局 | 含义 |
| --- | --- | --- | --- |
| `q` | BF16 | `[M, index_heads, head_dim]` | 每个 query row 的 indexer query |
| `k_cache` | BF16 | `[pages, page_size, 1, head_dim]` | 压缩 K 的分页 cache |
| `page_table` | int32 | `[requests, page_table_width]` | request 的逻辑页到物理页映射 |
| `token_to_req` | int32 | `[M]` | query row 到 request 的映射 |
| `query_positions` | int32 | `[M]` | query row 的零起始逻辑位置 |
| `sequence_lengths` | int32 | `[requests]` | request 当前序列长度 |
| `scores` | FP32 | `[M, num_columns]` | 压缩 block 分数；不可见列为 `-inf` |
| `visible_blocks` | int32 | `[M]` | 每行可参与选择的压缩 block 数 |

`num_columns` 默认是 `page_table_width * page_size`。`score_scale` 是分数的
除数，默认 `sqrt(head_dim)`。第 `m` 行的可见 block 数为
`min((query_positions[m] + 1) // compress_ratio,
sequence_lengths[token_to_req[m]] // compress_ratio)`。

### `qsa_select_paged_tokens`

```python
logical_indices = qsa_select_paged_tokens(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    out=None,
)
```

输入布局与 `qsa_mqa_paged` 相同。此接口先屏蔽不可见分数，再在 GPU 上选择
`token_topk // compress_ratio` 个压缩 block，最后展开成 request-relative 的
逻辑 token id。`token_topk` 必须为正且能被 `compress_ratio` 整除。返回值和可选
caller-owned `out` 都是连续的 int32
`[M, token_topk + compress_ratio - 1]`；额外的 `compress_ratio - 1` 列容纳
尚未组成完整压缩 block 的 causal tail，未使用位置写 `-1`。

### `expand_qsa_block_indices`

```python
logical_indices = expand_qsa_block_indices(
    block_indices,
    query_positions,
    sequence_lengths,
    token_to_req,
    compress_ratio,
    token_topk,
    out=None,
)
```

`block_indices` 是 int32
`[M, token_topk // compress_ratio]`；其余 metadata 与打分接口相同。返回布局、
causal tail 和 `-1` 规则与 `qsa_select_paged_tokens` 相同。该接口适用于调用方已经
完成压缩 block 选择的场景。

### `qsa_sparse_paged_attention`

```python
output = qsa_sparse_paged_attention(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    out=None,
)
```

| 参数或返回值 | dtype | 布局 | 含义 |
| --- | --- | --- | --- |
| `q` | BF16 | `[M, query_heads, head_dim]` | 稀疏注意力 query |
| `k_cache`, `v_cache` | BF16 | `[cache_blocks, page_size, kv_heads, head_dim]` | 主 K/V 分页 cache |
| `logical_indices` | int32 | `[M, topk]` | request-relative token id；`-1` 表示无效位置 |
| `block_table` | int32 | `[requests, page_table_width]` | request 的逻辑页到物理 cache block 映射 |
| `token_to_req` | int32 | `[M]` | query row 到 request 的映射 |
| `output`, `out` | BF16 | `[M, query_heads, head_dim]` | grouped-query attention 输出 |

`query_heads` 必须能被 `kv_heads` 整除，`head_dim` 必须是不小于 16 的 2 的
幂。所有张量必须位于同一 CUDA/HIP device；Q/K/V、metadata、selection 与输出
的最后一维必须连续。实现使用 FP32 online-softmax partial 和 split-K merge，
最终输出保持 BF16。

## 正确性测试

独立 Torch reference 覆盖打分、选择、展开、直接 sparse 路径和 split-K merge：

```bash
pytest test/test_generic_qsa.py -v
```

## 性能配置

运行时使用 `qsa.py` 内的静态 selector，不执行 autotune。HCU 上
`GROUP_SIZE=6, HEAD_DIM=256` 使用 `BLOCK_M=16, BLOCK_N=16,
num_warps=2, num_stages=1`，以满足 16×16 BF16 MMAC；其他形状回退到通用
配置。调优结果直接维护在静态 selector 中，运行时不执行 autotune。

### Prefill

#### `_qsa_mqa_paged_kernel`

Qwen3.8-Flash-Next-FP8-Channelwise + TP4。优化重点是 grouped-token MMA，
将可合并的 query token 组织为矩阵核心友好的 tile，减少 scalar dot、地址计算
和 softmax 周边开销。

| Kernel | M | H | D | baseline | Group MMA opt | speedup |
|---|---:|---:|---:|---:|---:|---:|
| ★ `_qsa_mqa_paged_kernel` | 512 | 4 | 128 | 15.860 ms | 1.9591 ms | 8.095x |

####  `_qsa_sparse_paged_gqa_splitk_kernel`

QSA sparse GQA 使用 `BLOCK_M=16, BLOCK_N=16, num_warps=2,
num_stages=1` 的 HCU MMA 配置；静态 selector 只对目标 TP4 shape 采用该路径，
其他 shape 回退通用配置。

| Kernel | M | baseline (ms) | opt (ms) | speedup |
|---|---:|---:|---:|---:|
| ★ `_qsa_sparse_paged_gqa_splitk_kernel` | 1024 | 35.442 | 1.388 | 25.536x |
|  | 2048 | 69.570 | 2.444 | 28.467x |
|  | 4096 | 138.267 | 4.655 | 29.707x |
|  | 8192 | 275.300 | 9.054 | 30.407x |
|  | 10240 | 344.330 | 11.267 | 30.558x |

当前 long-prefill 段最终配置为 `BLOCK_N=16, NUM_SPLITS=1,
num_warps=2, num_stages=1`。此前记录的 14.300 ms 对应旧的
`BLOCK_N=32, num_warps=4` 配置；该配置在 M=8192 复测约为 13.5 ms。

### Decode

按交接文档的 TP4 shape：`q=[M,6,256]`、`k_cache/v_cache=[num_blocks,16,1,256]`、
`logical_indices=[M,2051]`。vLLM 原始启发式仅在 `NUM_SPLITS > 1` 时启动 merge；
`M > 512` 时 `NUM_SPLITS=1`，不会调用 `_qsa_merge_splitk_kernel`。交接记录中
decode 的 `_qsa_merge_splitk_kernel` 平均 **1.769 ms**（15 次），而对照的
sparse kernel 平均约 **4.37 µs**；因此 merge 只对小 M 的尾延迟有影响。

#### `_qsa_sparse_paged_gqa_splitk_kernel`（decode / small M）

该 kernel 在 prefill 和 decode 共用；下面补充 `M<512` 区间的独立对比。
测量只统计 splitk kernel 自身的 device time，不包含后续 merge kernel。

| M | baseline (µs) | opt (µs) | speedup |
|---:|---:|---:|---:|
| 1 | 75.343 | 15.375 | 4.900x |
| 4 | 174.671 | 28.831 | 6.058x |
| 8 | 307.471 | 37.935 | 8.105x |
| 9 | 347.417 | 38.063 | 9.127x |
| 16 | 552.639 | 70.154 | 7.878x |
| 32 | 1338.243 | 134.580 | 9.944x |
| 40 | 1645.987 | 173.610 | 9.481x |
| 64 | 2522.365 | 248.313 | 10.158x |
| 128 | 4813.644 | 454.767 | 10.585x |
| 256 | 9439.395 | 869.566 | 10.855x |
| 512 | 18076.014 | 1612.195 | 11.212x |

`M<512` 区间相对 vLLM baseline 约为 **4.9x ~ 11.2x**。数据为 BW1100 上 20 轮重复测量的中位数

| Kernel | M | NUM_SPLITS | baseline (µs) | opt (µs) | speedup |
|---|---:|---:|---:|---:|---:|
| ★ `_qsa_merge_splitk_kernel` | 4 | 64 | 5.44 | 3.69 | 1.47x |
|  | 8 | 64 | 5.61 | 3.84 | 1.46x |
|  | 40 | 8 | 4.68 | 3.17 | 1.48x |
|  | 160 | 8 | 8.22 | 6.15 | 1.34x |
|  | 256 | 8 | 12.70 | 10.69 | 1.19x |
|  | 512 | 4 | 15.52 | 13.57 | 1.14x |
|  | 1024* | 8 | 35.22 | 33.18 | 1.06x |
|  | 4096* | 8 | 125.26 | 125.10 | 1.00x |

`baseline` 为 vLLM 原始 kernel（`DIM_TILE=256, merge_warps=2`）；
`opt` 为当前静态 selector：`M<=8` 使用 `DIM_TILE=64`，其余使用
`DIM_TILE=256`，两类配置均使用 `merge_warps=1`。
