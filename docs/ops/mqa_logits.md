# MQA Logits

`boltops.mqa_logits` 提供 sparse-attention indexer 用的 **MQA 打分** kernel。
接口对齐 hygon aiter / DeepGEMM；本 MR 仅迁入 **prefill**（row / bn / gluon）。

```python
from boltops.mqa_logits import fp8_mqa_logits

# Prefill：默认 impl="triton"；Triton 内 auto（有 Gluon 时 row|gluon，否则 row|bn）
logits = fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends)
# 框架：impl="triton"（默认，可省略）
# Triton 路径：backend="row" | "bn" | "gluon"
logits = fp8_mqa_logits(..., impl="triton", backend="bn", D_out=prealloc)
```

测试：

```bash
pytest -q test/mqa_logits/
# lightop 预填 size grid 性能（可选 vs lightop）
pytest -q test/mqa_logits/ --bench -s -k size_grid
pytest -q test/mqa_logits/ --bench -s -k vs_lightop
```

溯源见 `boltops/mqa_logits/triton/SOURCE.md`。
加速数据见 `boltops/mqa_logits/triton/PERFORMANCE.md`。

Launch 参数默认从
`configs/triton_<capability>/fp8_mqa_logits,arch=<gfx>.json`
（以及 `_tiled` / `_gluon`）按 arch + shape 选取。
已入库：**gfx938**（`triton_3_6` + `triton_3_2`）。gfx936 早期 row tune 已移除，待重 tune 后再入库。
重 tune：`HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_fp8_mqa_logits.py --write-ship`。

---

## `fp8_mqa_logits`

Prefill / dense KV：对每个 query 行，在 `[cu_starts, cu_ends)` 窗口内计算

\[
\mathrm{logits}[m,n]=\sum_h \mathrm{ReLU}\big((Q[m,h]\cdot KV[n])\cdot scale[n]\big)\,W[m,h]
\]

窗外位置为 `-inf`（供后续 top-k）。

### 实现选择

| `impl` | 含义 |
| --- | --- |
| `"triton"`（默认；也可省略 / `"auto"`） | Triton |

### Triton 路径（`backend`，仅 `impl=triton`）

| `backend` | 含义 |
| --- | --- |
| `None` / `"auto"` | 有 Gluon：`row` \| `gluon`；否则：`row` \| `bn`（形状启发式） |
| `"row"` | 经典按 query 行扫 KV |
| `"bn"` | 2D BN-tile（Triton，无 Gluon 时的主力大 shape 路径） |
| `"gluon"` | 强制 Gluon tiled（Gluon 不可用则报错） |

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `Q` | `[seq_len, H, D]` | float8 | query |
| `KV` | `[seq_len_kv, D]` | float8 | 连续 Key（MQA 共享） |
| `kv_scales` | `[seq_len_kv]` | float32 | 每 token 反量化 scale |
| `weights` | `[seq_len, H]` | float32 | 每 (query, head) 权重 |
| `cu_starts` | `[seq_len]` | int32 | 每行 KV 窗口起点（含） |
| `cu_ends` | `[seq_len]` | int32 | 每行 KV 窗口终点（不含） |
| `D_out` | `[seq_len, seq_len_kv]` | float32 | 可选预分配输出，原地写入 |
| `impl` | — | str | 默认 `"triton"`（当前仅支持 Triton） |
| `backend` | — | str | Triton 路径，见上表 |

### 返回

| 张量 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `logits` | `[seq_len, seq_len_kv]` | float32 | 打分结果；窗外为 `-inf` |

约束：`H`、`D` 须为 2 的幂（与 aiter 一致）。Gluon 路径额外要求 `w_group` 整除 head chunk。

实现：

* 公共入口：`boltops/mqa_logits/__init__.py`
* row / bn：`boltops/mqa_logits/triton/fp8_mqa_logits.py`
* gluon：`boltops/mqa_logits/triton/fp8_mqa_logits_gluon.py`
* 形状分发：`boltops/mqa_logits/triton/mqa_config_utils.py`

---

---

## 待开发

- Paged decode（`fp8_paged_mqa_logits` / stage1）
- Gluon **preshuffle** ``KVBlockSize=64``（aiter `gluon/pa_mqa_logits.py` preshuffle 变体）
