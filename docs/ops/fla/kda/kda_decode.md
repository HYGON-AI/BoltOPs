# KDA Decode/Recurrent 阶段使用说明

## 1. 阶段职责

Decode 阶段逐 token 或逐 speculative token batch 更新 FP32 recurrent state。它包含：

```text
raw gate/beta -> gate/beta materialization（可选）
q/k/v 或 packed QKV -> recurrent state update -> output/state
```

当 gate 融合开启时，gate/beta materialization 与 dense recurrent 在同一个 dispatch
中完成。

## 2. 接口与 I/O

已处理 `g/beta` 时：

```python
output, state = fused_recurrent_kda_fwd(
    q, k, v, g, beta,
    scale=K**-0.5,
    initial_state=state,
    cu_seqlens=cu_seqlens,
    ssm_state_indices=state_indices,
)
```

raw gate/beta 时：

```python
output, state = fused_recurrent_kda(
    q, k, v, raw_g, raw_beta,
    A_log, dt_bias, lower_bound,
    state, cu_seqlens, state_indices,
    fuse_gate=True,
)
```

packed QKV 使用 `fused_recurrent_kda_packed_decode`，其输入为
`mixed_qkv:[tokens,2*H*K+H*V]`、`raw_g:[1,tokens,H,K]`、
`raw_beta:[1,tokens,H]`，输出为 `[1,tokens,H,V]`。

## 3. fixed/varlen 与 state index

Dense recurrent tensor batch 固定为 `B=1`。`cu_seqlens:[N+1]` 表示多序列
dispatch，`ssm_state_indices` 为一维 `[N]` 或单序列多 token 的二维 `[1,T]`。
`N` 是 sequence/token-batch 数，不是 tensor batch `B=1`。Packed decode 使用
`state_indices:[tokens]`，不使用 Prefill 的 `cu_seqlens`。

## 4. 子路径、配置与 fallback

- decode gate/beta 使用 `_kda_gate_beta_fwd_kernel`，配置为 `BT2/w2/s3`、
  `BT2/w8/s2`、`BT8/w4/s3`；
- `fuse_gate=True` 时 raw gate/beta 不物化；`False` 时先执行独立 materialization；
- dense recurrent `N=1` 使用 `BV8/w1/s2`，`2<=N<32,H>=24` 使用
  `BV16/w1/s1`，`N>=32,H>=24` 使用 `BV32/w4/s1`；
- packed bounded gate、K=V=128、H≥24、FP32 state 使用 direct-3D
  `(V tile,head,token)` kernel；softplus、小 head、非方形 K/V 或非 FP32 state
  回退 generic packed kernel；
- H≥24 的 q/k normalization 使用 FP32 `x*rsqrt(sum)`，小 head 回退普通路径。

## 5. 性能与证据

BW1000_H、BF16 输入、FP32 state、`H=24,K=V=128`：

| 子路径 | Baseline XCU | 最终 XCU | 加速 | HBM GB/s | VGPR |
|---|---:|---:|---:|---:|---:|
| decode gate/beta，tokens=6 | `5.280 µs` | `5.120 µs` | `1.031x` | `23.89→24.50` | `15→15` |
| dense recurrent，tokens=6 | `21.760 µs` | `19.200 µs` | `1.133x` | `876.60→992.94` | `67→66` |
| packed decode，tokens=6 | `22.560 µs` | `18.720 µs` | `1.205x` | `845.59→1019.38` | `54→50` |

大 token batch 的 gate/state dispatch 摊销：

| N | 当前 separate | 原始 vLLM fused | 当前 fused | separate→fused | 原始 fused→当前 |
|---:|---:|---:|---:|---:|---:|
| 16 | `49.60 µs` | `48.32 µs` | `41.44 µs` | `1.197x` | `1.166x` |
| 32 | `113.12 µs` | `86.40 µs` | `82.08 µs` | `1.378x` | `1.053x` |
| 64 | — | — | `163.04 µs` | — | — |

N=32 最终 active waves/CU `28.32`、VALU Util `69.09%`、MMOP Util `0%`、
VMEM latency `1722.21 cycles`，无 spill/scratch；N=64 最终 HBM 为
`1247.06 GB/s`。N=64 未采集同批次 baseline，因此不推导加速比。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_decode.py
```

源码：`boltops/fla/kda/triton/fused_recurrent.py`。
