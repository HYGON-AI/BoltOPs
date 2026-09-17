# MQA Logits

`boltops.mqa_logits` 提供 sparse-attention indexer 用的 **MQA 打分** kernel。
接口对齐 hygon aiter / DeepGEMM / lightop；本 MR 仅迁入 **prefill**（row / bn / gluon）。

公共 Triton 入口为 **`triton_mqa_logits`**（TileLang 可另文件再导出）。路径由 `kv_scales` 决定：

* 传入 `kv_scales` → float8 路径
* `kv_scales=None` → fp16/bf16 路径

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
# lightop 预填 size grid 性能（可选 vs lightop）
pytest -q test/mqa_logits/ --bench -s -k size_grid
pytest -q test/mqa_logits/ --bench -s -k vs_lightop
```

溯源见 `boltops/mqa_logits/triton/SOURCE.md`。
加速数据见 `boltops/mqa_logits/triton/PERFORMANCE.md`。

Launch 参数默认从
`configs/triton_<capability>/fp8_mqa_logits,arch=<gfx>.json`
（以及 `_tiled` / `_gluon`）与 half 的
`mqa_logits_half` / `mqa_logits_half_row_short_kv`
按 arch + shape 选取。
已入库：**gfx938**（`triton_3_6` + `triton_3_2`）。gfx936 早期 row tune 已移除，待重 tune 后再入库。
重 tune：`HIP_VISIBLE_DEVICES=0 python3 boltops/mqa_logits/triton/tune_fp8_mqa_logits.py --write-ship`。

---

## `triton_mqa_logits`

Prefill / dense KV：对每个 query 行，在 `[cu_starts, cu_ends)` 窗口内计算

\[
\mathrm{logits}[m,n]=\sum_h \mathrm{ReLU}\big((Q[m,h]\cdot KV[n])\cdot scale[n]\big)\,W[m,h]
\]

窗外位置为 `-inf`（供后续 top-k）。半精度不使用 `kv_scales`（必须传 `None`）。

### Triton 路径（`backend`）

| `backend` | 含义 |
| --- | --- |
| `None` / `"auto"` | float8：有 Gluon → `row` \| `gluon`，否则 `row` \| `bn`；半精度：`row` \| `bn`（short-KV 上 row 更快的 shape 走 half row，否则 half BN） |
| `"row"` / `"bn"` | float8：对应路径；半精度：强制 half row / half BN |
| `"gluon"` | 强制 Gluon tiled（仅 float8；半精度报错） |
| `"gluon_bn_align"` | **experimental**：BN-aligned Gluon（float8 或 half）；不进 auto |
| `"gluon_opt"` | **experimental**：float8 下等价于 ship Gluon 别名；产品代码请用 `"gluon"` |

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `Q` | `[seq_len, H, D]` | float8 / fp16 / bf16 | query（与 KV 同 dtype） |
| `KV` | `[seq_len_kv, D]` | 同 Q | 连续 Key（MQA 共享） |
| `kv_scales` | `[seq_len_kv]` 或 `None` | float32 / `None` | 有值 → float8 路径；`None` → half 路径 |
| `weights` | `[seq_len, H]` | float32 | 每 (query, head) 权重 |
| `cu_starts` | `[seq_len]` | int32 | 每行 KV 窗口起点（含） |
| `cu_ends` | `[seq_len]` | int32 | 每行 KV 窗口终点（不含） |
| `D_out` | `[seq_len, seq_len_kv]` | float32 | 可选预分配输出，原地写入 |
| `backend` | — | str | Triton 路径，见上表 |

### 返回

| 张量 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `logits` | `[seq_len, seq_len_kv]` | float32 | 打分结果；窗外为 `-inf` |

约束：`H`、`D` 须为 2 的幂（与 aiter 一致）。半精度路径当前要求 **`D == 128`**。Gluon 路径额外要求 `w_group` 整除 head chunk。

实现：

* 公共入口：`boltops/mqa_logits/__init__.py`（`triton_mqa_logits`）
* row / bn（float8，内部）：`boltops/mqa_logits/triton/fp8_mqa_logits.py`
* half BN / row（fp16/bf16）：`mqa_logits_half.py`、`mqa_logits_half_row.py`
* half Gluon（experimental opt-in）：`mqa_logits_half_gluon_bn_align.py`
* gluon（float8）：`boltops/mqa_logits/triton/fp8_mqa_logits_gluon.py`
* experimental：`fp8_mqa_logits_gluon_bn_align.py`、`fp8_mqa_logits_gluon_opt.py`
* 形状分发：`boltops/mqa_logits/triton/mqa_config_utils.py`

非目标（本包后续可扩）：

- Paged decode（`fp8_paged_mqa_logits` / stage1）
- Gluon **preshuffle** ``KVBlockSize=64``（aiter `gluon/pa_mqa_logits.py` preshuffle 变体）
