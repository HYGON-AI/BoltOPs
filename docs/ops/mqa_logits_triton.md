# Triton MQA Logits

`boltops.mqa_logits` 提供 sparse-attention indexer 用的 **MQA 打分** kernel。
接口对齐 hygon aiter / DeepGEMM / hipc；已迁入 **prefill** 与 **paged decode**
（FP8，`kv_block_size=1` / `64`）。

公共 Triton 入口：

* **`triton_mqa_logits`** — prefill / dense KV
* **`triton_fp8_paged_mqa_logits`** / **`triton_fp8_paged_mqa_logits_stage1`** — FP8 paged decode

```python
from boltops.mqa_logits import triton_mqa_logits

# Prefill float8：必须传 kv_scales；auto 为 row|gluon / row|bn
logits = triton_mqa_logits(Q_fp8, KV_fp8, kv_scales, weights, cu_starts, cu_ends)
# Prefill half：kv_scales 必须为 None；auto 为 row|bn
logits = triton_mqa_logits(Q_bf16, KV_bf16, None, weights, cu_starts, cu_ends)
# 强制路径
logits = triton_mqa_logits(..., backend="bn", D_out=prealloc)
```

测试：

```bash
pytest -q test/mqa_logits/
# --bench 只跑 BoltOPs。hipc 对比要加 -k vs_hipc（默认 CI 不跑）
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k size_grid
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k test_bench_triton_mqa_logits_half
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k test_bench_triton_fp8_paged_mqa_logits_gpu
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_size
pytest -q test/mqa_logits/test_triton_fp8_mqa_logits.py --bench -s -k vs_hipc_prod
pytest -q test/mqa_logits/test_triton_mqa_logits_half.py --bench -s -k vs_hipc
pytest -q test/mqa_logits/test_triton_fp8_paged_mqa_logits.py --bench -s -k vs_hipc
```

half 官方网格：`test/mqa_logits/test_triton_mqa_logits_half.py`；
prefill / **paged** Triton 性能与路径说明见 `boltops/mqa_logits/triton/PERFORMANCE.md`。
TileLang：`docs/ops/mqa_logits_tilelang.md`。

溯源：`boltops/mqa_logits/triton/SOURCE.md`。  
性能笔记：`boltops/mqa_logits/triton/PERFORMANCE.md`。  
Launch 配置：`configs/triton_<capability>/fp8_mqa_logits*,arch=<gfx>.json` 与 half 对应 JSON；
已入库 gfx938（`triton_3_6` + `triton_3_2`）。

---

## `triton_mqa_logits`

Prefill / dense KV：对每个 query 行，在 `[cu_starts, cu_ends)` 内计算

\[
\mathrm{logits}[m,n]=\sum_h \mathrm{ReLU}\big((Q[m,h]\cdot KV[n])\cdot scale[n]\big)\,W[m,h]
\]

窗外为 `-inf`。半精度不使用 `kv_scales`（必须传 `None`）。

### Triton 路径（`backend`）

| `backend` | 含义 |
| --- | --- |
| `None` / `"auto"` | float8：有 Gluon → `row` \| `gluon`，否则 `row` \| `bn`；半精度：`row` \| `bn` |
| `"row"` / `"bn"` | 强制对应路径 |
| `"gluon"` | 强制 Gluon tiled（仅 float8） |
| `"gluon_bn_align"` / `"gluon_opt"` | **experimental**；不进 auto |

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `Q` | `[seq_len, H, D]` | float8 / fp16 / bf16 | query（与 KV 同 dtype） |
| `KV` | `[seq_len_kv, D]` | 同 Q | 连续 Key（MQA 共享） |
| `kv_scales` | `[seq_len_kv]` 或 `None` | float32 / `None` | 有值 → float8；`None` → half |
| `weights` | `[seq_len, H]` | float32 | 每 (query, head) 权重 |
| `cu_starts` / `cu_ends` | `[seq_len]` | int32 | KV 窗口 `[start, end)` |
| `D_out` | `[seq_len, seq_len_kv]` | float32 | 可选预分配输出 |
| `backend` | — | str | 见上表 |

### 返回

| 张量 | Shape | dtype |
| --- | --- | --- |
| `logits` | `[seq_len, seq_len_kv]` | float32；窗外 `-inf` |

约束：`H`、`D` 为 2 的幂；半精度当前要求 **`D == 128`**；Gluon 要求 `w_group` 整除 head chunk。

Float8 prefill 默认 ``assume_no_nans=1``（HIP ``no-nans-fp-math``），与 half / hipc
有限数学契约一致；详见 `PERFORMANCE.md`。

---

## `triton_fp8_paged_mqa_logits`（paged decode）

```
logits[b*next_n+n, t] = sum_h ReLU(Q[b,n,h] · K[t] * scale[t]) * W[b*next_n+n,h]
causal: t <= context_lens[b] - next_n + n；窗外为 -inf
```

只支持 **`kv_block_size ∈ {1, 64}`**（其它值在触碰 KV 前 `AssertionError`）。  
页内布局仅行主序；`preshuffle=True` → `NotImplementedError`。

```python
from boltops.mqa_logits import (
    triton_fp8_paged_mqa_logits,
    triton_fp8_paged_mqa_logits_prepare,
)

# 默认契约：调用方预填整行 -inf（kernel 只保证写有效窗口）
out = torch.full((batch * next_n, max_model_len), float("-inf"),
                 device=q.device, dtype=torch.float32)

# kv_block_size=1：per-token kv_indices
triton_fp8_paged_mqa_logits(
    q, kv_cache, weights, out, context_lens, kv_indices, max_model_len,
    kv_block_size=1,
)

# kv_block_size=64：block_table（须显式传 64；默认是 1）
triton_fp8_paged_mqa_logits(
    q, kv_cache, weights, out, context_lens, block_table, max_model_len,
    kv_block_size=64,
)
```

### 输出契约（`fuse_output_init`）

消费方把非 `-inf` 当合法 token；尾部脏数据会污染 top-k。算子**不校验**是否预填。

| `kv_block_size` | kernel 写入范围 | 调用方预填 `-inf` |
| --- | --- | --- |
| `1` | `[0, context_len)` | **必须** |
| `64` 默认 | 同上 | **必须** |
| `64` + `fuse_output_init=True` | `[0, max_model_len)` | 可省（`torch.empty` 即可） |

`fuse_output_init` 仅 **64 + 默认 SplitKV** 支持；`kv_block_size=1` 或 bucket 路径传了会
`NotImplementedError`（不会静默跳过填充）。

```python
out = torch.empty((batch * next_n, max_model_len), device=q.device, dtype=torch.float32)
info = triton_fp8_paged_mqa_logits_prepare(
    q, kv_cache, context_lens, block_table
)
triton_fp8_paged_mqa_logits(
    q, kv_cache, weights, out, context_lens, block_table, max_model_len,
    kv_block_size=64, batch_info=info, fuse_output_init=True,
)
```

### 生产路径：`prepare` + 图捕获

热路径上的固定开销主要来自 Triton Python launcher，不是 GPU kernel。推荐：

1. 每 batch 图外调用一次 `triton_fp8_paged_mqa_logits_prepare`（只解析 `max_ctx`，形状校验仍在 launch）
2. 每步传 `batch_info=`；指针/shape 固定时可整段 CUDA/HIP Graph 捕获

未传 `batch_info` 时仍可 launch（不做 `context_lens` 数值 D2H）；数值边界由调用方保证
（与 hipc / aiter 相同）。越界可能直接 device fault，无 Python 栈。

默认调度是 **SplitKV**。偏斜 batch 可选用设备端 bucket schedule：

```python
from boltops.mqa_logits import triton_fp8_paged_mqa_logits_build_bucket_schedule

info = triton_fp8_paged_mqa_logits_prepare(q, kv, cl, bt)
base = torch.empty(max_rows + 1, dtype=torch.int32, device="cuda")  # 地址稳定，可复用
sch = triton_fp8_paged_mqa_logits_build_bucket_schedule(
    cl, batch_size=B, next_n=next_n, base=base, max_ctx=info.max_ctx,
)
triton_fp8_paged_mqa_logits(
    q, kv, w, out, cl, bt, max_model_len,
    kv_block_size=64, batch_info=info, bucket_schedule=sch,
)
```

`use_bucket=True`（host 侧建 schedule）仅作对照/eager 回退，**不要进热路径**。


### 与 hipc / aiter 的差异（替换时必改）

| | hipc | aiter / BoltOPs |
| --- | --- | --- |
| 输出 | 内部申请并返回 | **调用方传入** `out_logits`（第 4 参） |
| 窗外 `-inf` | `clean_logits=True` 时可内部填 | **默认需调用方预填**（或 64 路 `fuse_output_init`） |
| 页大小 | 固定 64 | `kv_block_size=`（BoltOPs **默认 1**，paged 须显式 `64`） |
| 调度 | `schedule_meta` | `batch_info=` / `bucket_schedule=`（可省 → SplitKV） |
| KV | fp8 包或 bf16 | **仅 fp8 融合包** `[blocks, page, 1, D+4]` |

本包不提供 hipc 适配层；调用点改上述三点即可。

### `stage1`

`triton_fp8_paged_mqa_logits_stage1` 输出逐 head 分数
`[H, B*next_n, max_model_len]`，由调用方 `sum(dim=0)`。与主入口数学等价；同样只支持
页大小 1 / 64。

### KV 解析与其它约束

* `_prepare_paged_kv_views` 用 `as_strided` 拆 fp8 / scale，**不拷贝**；禁止非字节寻址
  的 `kv_cache`（如 bf16）。
* 入口只做形状校验：`context_lens.numel() == batch`、`head_dim <= kv` 末维。
* `chunk_q` / `chunk_k` 可自动选取（小 `heads`、小 M 时会钳制 `chunk_k`，避免 LDS 爆）。

---

## 实现位置

* 入口：`boltops/mqa_logits/__init__.py`
* Prefill float8：`fp8_mqa_logits.py` / `fp8_mqa_logits_gluon.py`
* Prefill half：`mqa_logits_half.py` / `mqa_logits_half_row.py`
* Paged decode：`fp8_paged_mqa_logits.py`
* 分发：`mqa_config_utils.py`

非目标：页内 **preshuffle（16×16 分块）** 布局。
