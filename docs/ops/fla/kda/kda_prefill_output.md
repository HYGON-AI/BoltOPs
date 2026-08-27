# KDA Prefill Output 阶段使用说明

## 1. 阶段职责

Output 阶段合并 chunk state 项和 chunk 内局部 causal 项：

```text
o = scale * q @ (exp2(g) * h) + tril(Aqk) @ v_new
```

它只负责输出，不负责独立的 state propagation。

## 2. 接口与 I/O

```python
o = chunk_gla_fwd_o_gk(
    q, v_new, g, Aqk, h, o,
    scale=K**-0.5,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_size=64,
)
```

输入 `q:[B,T,H,K]`、`v_new:[B,T,H,V]`、`g:[B,T,H,K]`、
`Aqk:[B,T,H,BT]`、`h:[B,NT,H,V,K]`；`o:[B,T,H,V]` 由调用方提供并原地写入。

## 3. fixed/varlen

fixed 使用规则 chunk offset；varlen 使用 `cu_seqlens/chunk_indices`，只在本条
sequence 的有效范围内计算。`Aqk` 上三角和尾部 padding 不读取。

## 4. Kernel、配置与 fallback

- optimized kernel：`chunk_gla_fwd_kernel_o_local_first`；generic kernel：
  `chunk_gla_fwd_kernel_o`；wrapper：`chunk_gla_fwd_o_gk`；
- `K,V>=128,BT=64` 主路径使用 `BK128,BV128,w8,s2`，先执行 local causal 矩阵
  乘，再累加 state 项；
- 其它 shape、dtype 或 layout 回退 generic output kernel，保持 scale 和 output
  buffer 语义；
- state/output fusion 是跨 State/Output 阶段的可选路径，详见
  [kda_prefill_state.md](./kda_prefill_state.md)。

## 5. 性能与证据

BW1000_H、`H=24,T=8192,K=V=128`：

| Layout | Baseline XCU | 最终 XCU | 加速 | 有效 HBM GB/s（model） |
|---|---:|---:|---:|---:|
| fixed | `0.492160 ms` | `0.398880 ms` | `1.234x` | `968.16` |
| varlen | `0.622560 ms` | `0.392160 ms` | `1.588x` | `970.99` |

2026-08-27 复核使用 XCU `execution` 字段；本 shape 命中 `local causal tri32`
路径。HBM 为沿用相同 tensor-traffic model 的有效带宽换算值，本机 XCU
`memory_workload` 未返回 PMC metric，不能视为硬件 counter 实测值。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_chunk.py
```

源码：`boltops/fla/kda/triton/chunk.py`。
