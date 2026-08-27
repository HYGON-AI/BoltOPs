# KDA Prefill WY 重建阶段使用说明

## 1. 阶段职责

WY 阶段根据 intra 生成的 `A` 重建 state update 所需的 `w/u`，并生成衰减 key
`kg`：

```text
u_c  = A_c @ (v_c * beta_c[..., None])
w_c  = A_c @ (k_c * beta_c[..., None] * exp2(gk_c))
kg_c = k_c * exp2(gk_last_c - gk_c)
```

KDA 使用逐 key channel 的 log2 gate，不同于 GDN 的 scalar natural-log gate。

## 2. 接口与 I/O

```python
w, u, qg, kg = recompute_w_u_fwd(
    k, v, beta, A, q=None, gk=gk,
    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
)
```

`k/v:[B,T,H,K/V]`、`beta:[B,T,H]`、`A:[B,T,H,BT]`、`gk:[B,T,H,K]`；
返回 `w/u`、固定为 `None` 的 `qg` 和可选 `kg:[B,T,H,K]`。

## 3. fixed/varlen

fixed 使用规则 chunk；varlen 使用 `cu_seqlens` 和可选 `chunk_indices`，当前
packed Prefill wrapper 要求 `B=1`。输入必须 contiguous 且位于同一 device。

## 4. Kernel、配置与 fallback

- Kernel：`recompute_w_u_fwd_kernel_beta_factored_head_first`；wrapper：
  `recompute_w_u_fwd`；
- BF16、`BT=64`、`K/V<=128` 且存在 `gk` 时使用 beta-factored head-first，
  复用 `A*beta` tile；
- 主配置为 `BK=64,BV=128,w4,s2`；其它 shape/dtype 回退 upstream kernel；
- 返回 tuple 和 `kg` 语义保持不变。

## 5. 性能与证据

BW1000_H、`H=24,T=8192,K=V=128`：

| Layout | Baseline XCU | 最终 XCU | 加速 | 有效 HBM GB/s（model） |
|---|---:|---:|---:|---:|
| fixed | `0.873120 ms` | `0.375520 ms` | `2.325x` | `2424.91` |
| varlen | `0.506160 ms` | `0.370400 ms` | `1.367x` | `1426.44` |

VGPR `189→171`，无 scratch/spill。

2026-08-27 XCU 复核采用 dispatch `execution`；HBM 是相同 traffic model 的
有效带宽换算值，因本机 XCU memory PMC 返回 `0 metrics`，不代表 counter 实测。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_chunk.py -k recompute_w_u
```

源码：`boltops/fla/kda/triton/chunk.py`。
