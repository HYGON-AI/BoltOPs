# `recompute_w_u_fwd 使用说明`

GDN 与 KDA 的分块 WY 重计算同名但协议不同，必须从家族入口调用：

```python
from boltops.fla.gdn import recompute_w_u_fwd as recompute_w_u_fwd_gdn
from boltops.fla.kda import recompute_w_u_fwd as recompute_w_u_fwd_kda
```

## 1. 算子用途与数学语义

给定每个 chunk 的三角矩阵 `A`，重建 state 更新所需的 `w`、`u`。

```text
# GDN，Hg 可小于 H
u_c = A_c @ (v_c * beta_c[..., None])
w_c = A_c @ (k_c[:, floor(h / (H / Hg)), :] * beta_c[:, h, None]
             * exp(g_cumsum_c[:, h, None]))

# KDA，逐通道 gate
u_c = A_c @ (v_c * beta_c[..., None])
w_c = A_c @ (k_c * beta_c[..., None] * exp2(gk_c))
kg_c = k_c * exp2(gk_last_c - gk_c)
```

GDN 使用自然对数空间的 `g_cumsum`，返回 `(w,u)`；KDA 使用 log2 空间的
`gk`，返回 `(w,u,qg,kg)`，当前 `qg` 固定为 `None`。

## 2. 接口说明

```python
w, u = recompute_w_u_fwd_gdn(k, v, beta, g_cumsum, A, cu_seqlens)
w, u, qg, kg = recompute_w_u_fwd_kda(k, v, beta, A, gk=gk)
```

### 2.1 参数含义

| 参数 | GDN | KDA |
| --- | --- | --- |
| `k` | `[B,T,Hg,K]`，允许 GQA/MQA | `[B,T,H,K]` |
| `v` | `[B,T,H,V]` | `[B,T,H,V]` |
| `beta` | `[B,T,H]` | `[B,T,H]` |
| gate | `g_cumsum:[B,T,H]`，自然对数 | `gk:[B,T,H,K]`，log2，可选 |
| `A` | `[B,T,H,BT]` | `[B,T,H,BT]` |
| 返回 | `(w,u)` | `(w,u,None,kg)` |

### 2.2 前置条件

- 张量在同一 GPU 且 contiguous；`BT=A.shape[-1]`。
- GDN 要求 `H % Hg == 0`；KDA 的 `k/v` head 数均为 `H`。
- packed varlen 使用 `B=1`、`cu_seqlens:[N+1]` 与可选 `chunk_indices:[NT,2]`。
- GDN 不接收 `gk`；KDA 不接收 `g_cumsum`。错误混用会产生参数绑定错误；将
  GDN 解包为四项或 KDA 解包为两项会产生 `ValueError`。

## 3. 典型 Shape

| 字段 | GDN | KDA |
| --- | --- | --- |
| `B/H/Hg` | `1/16/8` | `1/24/24` |
| `K/V/BT` | `128/128/64` | `128/128/64` |
| `k` | `[1,T,8,128]` | `[1,T,24,128]` |
| `v/u` | `[1,T,16,128]` | `[1,T,24,128]` |
| gate | `[1,T,16]` FP32 | `[1,T,24,128]` FP32 |
| `A` | `[1,T,16,64]` BF16 | `[1,T,24,64]` BF16 |

## 4. 正确性测试与性能基准

```bash
pytest -q test/fla/test_gdn_recompute_w_u.py
pytest -q test/fla/test_gdn_recompute_w_u.py --bench -s
pytest -q test/fla/test_kda_chunk.py -k recompute_w_u
```

## 附录 A. 仓库文件

| 用途 | GDN | KDA |
| --- | --- | --- |
| 接口 | `boltops/fla/gdn/__init__.py` | `boltops/fla/kda/triton/__init__.py` |
| 实现 | `boltops/fla/gdn/triton/recompute_w_u.py` | `boltops/fla/kda/triton/chunk.py` |
| 测试 | `test/fla/test_gdn_recompute_w_u.py` | `test/fla/test_kda_chunk.py` |

## 附录 B. 溯源

- GDN：SGLang `fla/wy_fast.py`，上游为 flash-linear-attention Gated Delta Rule。
- KDA：vLLM Kimi K3 AMD `ops/third_party/kda/chunk.py`。

## 附录 C. 实现备注

- GDN grid 为 `(chunk, batch × value_head)`，按 `K/V` 选择三档固定配置。
- KDA 在 BF16、`BT=64`、`K/V<=128` 且传入 `gk` 时可使用 head-first 路径。
- 两者不能合并 kernel：GDN 需要 GQA 与 `exp`，KDA 需要逐通道 `exp2` 与 `kg`。

## 附录 D. 需求 shape 抓取明细

GDN packed 测试覆盖 `T=60–13674`、`N=1–3`、`NT=1–216`；代表点为
`T=60、4156、8652、12617、13674`。KDA 主模型 shape 为
`B=1,T=8192,H=24,K=V=128,BT=64`。
