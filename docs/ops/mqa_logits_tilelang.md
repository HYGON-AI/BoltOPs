# TileLang MQA Logits

`boltops.mqa_logits` 对外的 **BF16 TileLang QSA** 打分入口。按设备 arch 自动选
`kernels/gfx938`（WASP）或 `kernels/common`（通用 GEMM）。调用方使用
`tilelang_qsa_mqa_prefill` / `tilelang_qsa_mqa_decode`。

FP8 Triton 入口见 `docs/ops/mqa_logits_triton.md`。

```python
from boltops.mqa_logits import tilelang_qsa_mqa_prefill, tilelang_qsa_mqa_decode

logits = tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends)
logits = tilelang_qsa_mqa_decode(
    q, k_cache, page_table, context_lens, max_model_len
)
```

测试：

```bash
# CI / 默认：config + 各 1 个 GPU prefill/decode smoke
pytest -q test/mqa_logits -m "not slow and not bench"
# 全量 GPU 网格
pytest -q test/mqa_logits --runslow -m "not bench"
```

---

## 计算

每条 query 对 MQA 共享 Key 做

\[
\mathrm{logits}[m,n]=\frac{1}{s}\sum_h \mathrm{ReLU}\big(Q[m,h]\cdot K[n]\big)
\]

`s` 默认 \(\sqrt{d}\)。窗外位置写 `-inf`（prefill：`[row_starts, row_ends)`；
decode：`[0, context_lens)`，其余列到 `max_model_len` 也是 `-inf`）。

输入会转成 BF16；输出 FP32。

---

## `tilelang_qsa_mqa_prefill`

稠密 KV prefill。

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `q` | `[rows, H, D]` | 任意，内部 BF16 | query |
| `k` | `[keys, 1, D]` | 任意，内部 BF16 | MQA Key |
| `row_starts` | `[rows]` | int32 | 每行有效 KV 起点（含） |
| `row_ends` | `[rows]` | int32 | 每行有效 KV 终点（不含） |
| `score_scale` | — | float，可选 | 除数 `s`；默认 \(\sqrt{D}\) |
| `effective_range` | — | int，可选 | **本次 launch** 里所有 query 行的最长有效 range：`max_i (row_ends[i]-row_starts[i])`。多 sequence 被拼进同一个 `[rows]` 时，取跨 sequence 的 max（一份 `split_n`/`n_tile_phases` 覆盖整次 launch，不能按 batch 各选一套）。多次独立 `forward` 则每次传各次的 max。默认 `None` 按 `keys`。host 标量；算子不对 GPU starts/ends 做 max |

### 返回

| 张量 | Shape | dtype |
| --- | --- | --- |
| `logits` | `[rows, keys]` | float32 |

### 约束与分发

- `q`、`k` 最后一维（`D`）须一致。
- **gfx938 WASP**：`H ∈ {4, 8, 32, 64}` 且 `D = 128`。tile / split / mask 相数由
  runtime 按 `rows`、`keys`、CU 数、以及可选的 `effective_range` 选择。
- 其它 `H`/`D`：同一入口，落到 common TileLang GEMM + mask，不回到 Triton/Torch。

实现：`boltops/mqa_logits/__init__.py` → `tilelang._dispatch.resolve("qsa_mqa_prefill")`。

---

## `tilelang_qsa_mqa_decode`

Paged decode：逻辑 token 经 `page_table` 聚到物理 `k_cache`。

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `q` | `[batch, H, D]` | 任意，内部 BF16 | 当前步 query |
| `k_cache` | `[pages, page_size, 1, D]` | 任意，内部 BF16 | 物理页 Key |
| `page_table` | `[batch, max_pages]` | int32 | 每条请求的逻辑页 → 物理页 |
| `context_lens` | `[batch]` | int32 | 每条请求有效 token 数 |
| `max_model_len` | — | int | 输出列数（通常 ≥ 最大 context） |
| `score_scale` | — | float，可选 | 除数 `s`；默认 \(\sqrt{D}\) |

### 返回

| 张量 | Shape | dtype |
| --- | --- | --- |
| `logits` | `[batch, max_model_len]` | float32 |

### 约束

- 需要 device tensor。
- `H ∈ {4, 8, 16, 32, 64}`，`D = 128`。
- `page_size ∈ {8, 16, 32, 64}`（64-row GEMM 子页打包）。
- gfx938 按 batch / `max_model_len` / heads 自动选普通或 persistent decode，
  以及 suffix mask 的 split / reverse。

实现：`boltops/mqa_logits/__init__.py` → `tilelang._dispatch.resolve("qsa_mqa_decode")`。
当前 decode 走 `kernels/common`（gfx938 配置驱动 persistent）。
