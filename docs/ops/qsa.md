# QSA 分页选择与稀疏注意力

QSA 提供四个互相独立、无需 vLLM 的公开 API：压缩 K cache 打分、
压缩 block 选择与 token 展开、单独展开，以及分页稀疏 GQA。调用方负责 cache
分配与写入、page table 和 request 映射；BoltOPs 不管理框架调度或 cache 生命周期。

```python
import torch

from boltops.qsa import (
    expand_qsa_block_indices,
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
```

## 从 vLLM 迁移

BoltOPs 保持 vLLM QSA 的四步数据流，通常只需替换导入和函数名。QSA 原位于
`boltops.generic`，现已升级为独立算子族 `boltops.qsa`，旧的
`from boltops.generic import qsa_*` 导入不再可用。

| vLLM 调用 | BoltOPs 调用 |
| --- | --- |
| `qsa_mqa_paged(...)` | `qsa_mqa_paged(...)` |
| `qsa_select_paged_tokens(...)` | `qsa_select_paged_tokens(...)` |
| `expand_qsa_block_indices_cuda(...)` | `expand_qsa_block_indices(...)` |
| `qsa_sparse_paged_attention(...)` | `qsa_sparse_paged_attention(...)` |

参数顺序和返回布局保持一致；BoltOPs 不需要 `vllm._custom_ops`，也不需要
传入 CUDA stream。典型替换如下：

```python
from boltops.qsa import (
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

## 关系与边界

- 本包是框架无关的分页 QSA 入口：打分、压缩 block 选择、causal 展开与稀疏
  paged GQA 四步数据流，不依赖 vLLM，cache 生命周期与 page table 由调用方
  管理。
- `boltops.mqa_logits` 是 aiter 风格通用 MQA 打分包（`triton_mqa_logits`：
  dense、fp8/fp16/bf16、per-head `weights`、`kv_scales`），与 QSA 打分契约
  不同，见 [MQA Logits](mqa_logits_triton.md)。
- `boltops.sparse_gqa` 同样是稀疏 GQA 领域（packed decode / KV compact），
  但面向不同的调度数据布局，见 [Sparse GQA](sparse_gqa.md)。
- 实现位于 `boltops/qsa/triton/qsa.py`（单文件，对齐 vLLM
  `qwen4_exp` QSA ops 的布局）：打分 / 展开 / 选择 / 稀疏 GQA kernel、
  静态 launch 配置与四个公共入口都在该文件，公共入口顺序与上游一致。

## 正确性测试

独立 Torch reference 覆盖打分、选择、展开、直接 sparse 路径和 split-K merge：

```bash
pytest test/qsa/test_qsa.py -v
```

## 性能配置

运行时使用 `boltops/qsa/triton/qsa.py` 内的静态 selector，不执行
autotune。HCU 上 `GROUP_SIZE=6, HEAD_DIM=256` 固定 `BLOCK_M=16`（满足
16×16 BF16 MMAC）；目标 TP4 shape（Hkv=1、topk=2051、D=256）的 `BLOCK_N`
与 split-K 参数按 rows 分段维护在 `TUNED_TARGET_SEGMENTS`：

| rows | BLOCK_N | target_splits | num_warps | num_stages |
|---:|---:|---:|---:|---:|
| 2 | 32 | 32 | 4 | 1 |
| 3–23 | 32 | 16 | 2 | 1 |
| 24–256 | 32 | 1 | 2 | 1 |
| 257–512 | 16 | 1 | 2 | 1 |
| 513–262144 | 16 | 1 | 2 | 1 |

段外 rows 与其他 shape 回退到通用 baseline 配置。所有段均为 HCU 实测
（decode 区间对 69+ 个 launch 配置全量 sweep 取 min-of-5 最优；splits=1
且总 program 数 ≲16 的配置存在 ~270 µs 低程序数病理延迟，选段时避开）。
`rows ≥ 3` 使用 row-extent 截断 kernel（kernel 内扫描行最后有效列并截断
tile 范围，短行跳过全掩码 tile），`rows ≤ 2` 使用常量边界 kernel。

### 优化手段

- **HCU MMA tile 配置**：`GROUP_SIZE=6, HEAD_DIM=256` 固定 `BLOCK_M=16`
  （满足 16×16 BF16 MMAC）；`num_warps=2, num_stages=1` 全面优于更多 warp
  / 更深 stage（69+ 个 launch 配置全量 sweep 取 min-of-5 最优）。
- **MQA 打分 grouped-token MMA**（`_qsa_mqa_paged_kernel`）：把可合并的
  query token 组织为矩阵核心友好的 tile，减少 scalar dot、地址计算和
  softmax 周边开销，M=512 从 15.860 ms 降到 1.9591 ms（8.1x）。
- **分段 dispatch**（`TUNED_TARGET_SEGMENTS`）：`BLOCK_N`、split-K、
  `num_warps` 按 rows 分段取实测最优。3–23 段用 `(32,16)` 保持足够 program
  数——splits=1 且总 program 数 ≲16 时存在 ~270 µs 的低程序数病理延迟，
  选段时避开；24–256 段单遍 `(32,1)` 免 merge——实测 M=32 时 splits=1
  为 22.8 µs，splits=4/8 反而 34.1/35.3 µs（partial 往返 + merge 开销
  超过并行收益）；257 起用 `(16,1)`。selector 结果按 shape 缓存
  （lru_cache），配置选择不进热路径。
- **row-extent 截断**：split-K kernel
  （`_qsa_sparse_paged_gqa_splitk_clamped_kernel`）在 kernel 内按 program
  扫描该行 `logical_indices` 的最后有效列（8 KB、L2 驻留），把 tile 范围
  截断到有效列以内，短行不再遍历整段 topk 的全掩码 tile；短窗口 decode
  M=128/256/512 相对无截断版本分别再降 3.7x / 5.6x / 4.5x。扫描融合在
  kernel 内（无独立 extent pass），小 batch 保持单次 kernel dispatch，
  避免额外 launch 的 CPU 开销。
- **小 batch 门限**：`rows ≤ 2`（单/双 token decode）走常量边界 kernel
  （`_qsa_sparse_paged_gqa_splitk_kernel`），`rows ≥ 3` 走截断 kernel。
- **split-K merge 配置**（`_qsa_merge_splitk_kernel`）：`M<=8` 使用
  `DIM_TILE=64`，其余使用 `DIM_TILE=256`，统一 `merge_warps=1`，比 vLLM
  原始 merge 快 1.1x~1.5x。

### Prefill

#### `_qsa_mqa_paged_kernel`

Qwen3.8-Flash-Next-FP8-Channelwise + TP4。优化重点是 grouped-token MMA，
将可合并的 query token 组织为矩阵核心友好的 tile，减少 scalar dot、地址计算
和 softmax 周边开销。

| Kernel | M | H | D | baseline | Group MMA opt | speedup |
|---|---:|---:|---:|---:|---:|---:|
| ★ `_qsa_mqa_paged_kernel` | 512 | 4 | 128 | 15.860 ms | 1.9591 ms | 8.095x |

`qsa_sparse_paged_attention` prefill 全路径实测（splitk + merge + 输出
分配，`triton.testing.do_bench`，BW1100，TP4 目标 shape；baseline 为 vLLM
源 kernel 的 device time，opt 为全路径）：

| M | vLLM baseline (ms) | 最终 opt (ms) | speedup |
|---:|---:|---:|---:|
| 1024 | 35.442 | 0.401 | 88.4x |
| 2048 | 69.570 | 1.305 | 53.3x |
| 4096 | 138.267 | 3.462 | 39.9x |
| 8192 | 275.300 | 7.914 | 34.8x |
| 10240 | 344.330 | 10.107 | 34.1x |

### Decode

按交接文档的 TP4 shape：`q=[M,6,256]`、`k_cache/v_cache=[num_blocks,16,1,256]`、
`logical_indices=[M,2051]`。merge 仅在 `NUM_SPLITS > 1` 时启动。vLLM
原始启发式（也是非目标 shape 的回退 baseline）只在 `M > 512` 时使用
`NUM_SPLITS=1`；目标 shape 经 HCU 全量 sweep 后该阈值下移到 `M ≥ 24`
（行并行已足够，split-K partial 的全局内存往返加 merge 成为净成本），
此后不调用 `_qsa_merge_splitk_kernel`；M=3~23 为 16 splits，启动 merge。

`qsa_sparse_paged_attention` decode 全路径实测（splitk + merge + 输出
分配，`triton.testing.do_bench`，BW1100，TP4 目标 shape；baseline 为 vLLM
源 kernel 的 device time，opt 为全路径）：

| M | vLLM baseline (µs) | 最终 opt (µs) | speedup |
|---:|---:|---:|---:|
| 1 | 75.343 | 23.6 | 3.2x |
| 4 | 174.671 | 22.9 | 7.6x |
| 8 | 307.471 | 23.8 | 12.9x |
| 9 | 347.417 | 23.7 | 14.7x |
| 16 | 552.639 | 27.4 | 20.2x |
| 32 | 1338.243 | 18.6 | 71.9x |
| 40 | 1645.987 | 22.6 | 72.8x |
| 64 | 2522.365 | 22.9 | 110x |
| 128 | 4813.644 | 36.9 | 130x |
| 256 | 9439.395 | 64.8 | 146x |
| 512 | 18076.014 | 154.5 | 117x |

`_qsa_merge_splitk_kernel`（split-K 尾合并，仅 M≤23 启动）相对 vLLM 原始
merge kernel（`DIM_TILE=256, merge_warps=2`）：

| M | NUM_SPLITS | vLLM baseline (µs) | 最终 opt (µs) | speedup |
|---:|---:|---:|---:|---:|
| 4 | 64 | 5.44 | 3.69 | 1.47x |
| 8 | 64 | 5.61 | 3.84 | 1.46x |
| 40 | 8 | 4.68 | 3.17 | 1.48x |
| 160 | 8 | 8.22 | 6.15 | 1.34x |
| 256 | 8 | 12.70 | 10.69 | 1.19x |
| 512 | 4 | 15.52 | 13.57 | 1.14x |
| 1024* | 8 | 35.22 | 33.18 | 1.06x |
| 4096* | 8 | 125.26 | 125.10 | 1.00x |

表中 `NUM_SPLITS` 为测量当时的 dispatch；当前分段 dispatch 下 M=3~23 为
16 splits，M≥24 为 1（不启动 merge），`*` 行仅为 kernel 口径对比。
