# KDA Prefill Gate 阶段使用说明

## 1. 阶段职责

Gate 阶段把 raw gate/beta 变成后续 Prefill 使用的 `gk` 和 `beta`，并在每个
chunk 内累计 gate。该阶段包含单独 gate activation 和 gate/cumsum 融合两条入口。

```text
raw_g/raw_beta/A_log/g_bias
  -> gate activation + beta sigmoid
  -> chunk-local cumsum in log2 space
  -> gk, beta
```

## 2. 接口与 I/O

只计算 gate 时：

```python
gate = fused_kda_gate(
    raw_g, A_log, head_k_dim,
    g_bias=None, beta=1.0, threshold=20.0,
    lower_bound=None,
)
```

完整阶段：

```python
gk, beta = fused_kda_gate_chunk_cumsum(
    raw_g, raw_beta, A_log,
    g_bias=g_bias, lower_bound=lower_bound,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices, chunk_size=64,
)
```

| 张量 | Shape | 说明 |
|---|---|---|
| `raw_g` | `[B,T,H,K]` | raw key-wise gate |
| `raw_beta` | `[B,T,H]` | raw beta |
| `A_log` | `[H]` | head gate 参数 |
| `g_bias` | `[H,K]` 或 `None` | 可选 gate bias |
| `gk` | `[B,T,H,K]` | chunk-local log2 cumulative gate |
| `beta` | `[B,T,H]` | sigmoid 后 beta |

## 3. fixed/varlen

- fixed：`cu_seqlens=None`，按规则 batch/chunk 计算；
- varlen：`cu_seqlens=[N+1]`，cumsum 在每条 sequence 起点重置，chunk 不跨
  sequence，可附带 `chunk_indices:[NT,2]`；
- Prefill packed wrapper 要求 `B=1`，所有输入 contiguous 且在同一 GPU。

## 4. Kernel、配置与 fallback

- `fused_kda_gate` 使用 `kda_gate_fwd_kernel`，按 `BT=32/64/128` 选择；
- `fused_kda_gate_chunk_cumsum` 使用 `kda_gate_chunk_cumsum_vector_kernel`，
  保留 `BS32/w2`、`BS32/w4`、`BS64/w2`，BW1000_H 主 shape 选择 `BS64/w2`；
- softplus 的显式 `1/beta` 在后端支持时使用 `tl.fdiv_fast`；有界 sigmoid 不能
  用 `fdiv_fast` 替换；
- 不满足 shape、dtype、stride 或 packed 边界条件时回退 upstream-compatible
  gate/cumsum 路径。

## 5. 性能与证据

BW1000_H、`H=24,T=8192,K=V=128` 的 XCU dispatch time：

| 子路径 | Layout | Baseline XCU | 最终 XCU | 加速 | 有效 HBM GB/s（model） |
|---|---|---:|---:|---:|---:|
| gate activation | layout-agnostic | `0.205920 ms` | `0.187840/0.188160 ms` | `1.096/1.094x` | `1081.10/1079.26` |
| gate/cumsum | fixed | `0.285440 ms` | `0.248960 ms` | `1.147x` | `849.40` |
| gate/cumsum | varlen | `0.246080 ms` | `0.250560 ms` | `0.982x` | `844.31` |

2026-08-27 XCU 复核采用 dispatch `execution`；HBM 为相同 traffic model 的
有效带宽换算值，本机 XCU memory PMC 返回 `0 metrics`。gate activation 使用
扁平 token 入口，不区分 fixed/varlen；此前标注的 varlen `0.461280 ms` 是
driver 将第二次相同调用误标为 varlen，已删除，不代表 kernel 回退。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_gate.py
```

源码：`boltops/fla/kda/triton/chunk.py`。
