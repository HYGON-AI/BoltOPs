# KDA Prefill Intra 阶段使用说明

## 1. 阶段职责

Intra 阶段在每个 `BT=64` chunk 内生成 `Aqk/Akk`，并完成 diagonal 与跨
sub-chunk 的三角求解。它包含三个实现子路径，但对外表现为一个
`chunk_kda_fwd_intra` 阶段接口：

```text
q/k/gk/beta -> token-parallel 或 polynomial diagonal -> inter solve -> Aqk/Akk
```

## 2. 接口与 I/O

```python
Aqk, Akk = chunk_kda_fwd_intra(
    q, k, gk=gk, beta=beta,
    scale=K**-0.5,
    cu_seqlens=cu_seqlens,
    chunk_size=64,
    chunk_indices=chunk_indices,
    safe_gate=False,
)
```

| 参数/返回值 | Shape | 说明 |
|---|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16/FP16 Q/K |
| `gk` | `[B,T,H,K]` | log2 cumulative gate，必须提供 |
| `beta` | `[B,T,H]` | update coefficient |
| `Aqk`, `Akk` | `[B,T,H,BT]` | chunk 系数矩阵 |

只有有效下三角区域有定义，上三角和 padding 不得读取。

## 3. fixed/varlen 与公共 normalization

fixed 使用规则 chunk offset；varlen 使用 `cu_seqlens` 和 `chunk_indices`，每条
sequence 独立处理。Prefill 的 Q/K normalization 由公共 l2norm helper 或上层
完成；recurrent decode 的同一公式在 [kda_decode.md](./kda_decode.md) 中说明。

## 4. 子 kernel、配置与 fallback

### 4.1 Token-parallel raw

`chunk_kda_fwd_kernel_intra_token_parallel` 是通用 diagonal 路径；
`chunk_kda_fwd_kernel_intra_subchunk_raw_optimized` 在 `K=128,BT=64,BC=16`
时以 head/subchunk 粒度复用 q/k/g tile。主配置为 `BK=128,w1,s1`，并保留
`w1,s2`、`w1,s3`。

### 4.2 Polynomial diagonal solve

`chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized` 对 `BC=16` 使用
FP32：

```text
(I + A)^-1 = (I-A)(I+A^2)(I+A^4)(I+A^8)
```

主配置为 `BK=128,w1,s1,early=true`。严格下三角、`BC=16` 或 FP32 条件不满足
时回退 upstream safe solve；BF16 MMAC 输入不采用。

`K=128,BT=64,BC=16` 时该 kernel **不再单独 launch**：其计算已被折进 inter
solve（见 4.3 的 `FUSE_DIAG`）。它仍是 token-parallel 路径与 A/B 对照的实现。

q/k 的 in-kernel l2norm write-back（内部 `normalize_qk=True` 选项）只存在于
该 kernel。`FUSE_DIAG` 下它不再 launch，wrapper 会在 inter solve 前用公共
l2norm helper 对调用方的 q/k buffer **原地**归一化，保证 recompute_w_u 与
state/output 阶段读到的行与 sub-chunk write-back 路径一致。

### 4.3 Inter solve

`chunk_kda_fwd_kernel_inter_solve_fused` 计算 off-diagonal `Aqk/Akk`，消费
diagonal inverse；已预解时跳过重复的 diagonal forward substitution。

#### `FUSE_DIAG`：把 diagonal 折进来

`K=128,BT=64,BC=16` 时置 `FUSE_DIAG=True`：kernel 自己用
`_diag_block_products` + `_invert_strict_lower` 生成四个已求逆的 diagonal
块，不再 launch polynomial kernel，`Akkd` 也不再 materialize。

动机是**带宽**而非算力。两个 kernel 读的是同一批 q/k/g 行，而 BW1100 上
polynomial 实测 838 GB/s、inter 实测 782 GB/s——都已贴在实测 HBM 峰值
(~837 GB/s) 上，说明第二趟读几乎是纯浪费。融合省掉一整趟 q/k/g 读
(~1073 MB) 加 `Akkd` 往返 (~134 MB)。

两条关键约束：

* diagonal 块的 gate 参考行必须取 sub-chunk 的**中点**，不能跟 off-diagonal
  共用首行。off-diagonal 形成的指数全部 ≤ 0，而 diagonal 同时需要
  `exp2(+gm)` 和 `exp2(-gm)`；取中点把每个方向要覆盖的跨度减半。实测 KDA
  gate 在 16 token 上的 log2 跨度可达 ~105，FP32 上限是 2^128，所以这一步是
  正确性所必需，不是微优化。
* `_invert_strict_lower` 全程 FP32。发布出去的 inverse 是 BF16，但在 A2/A4/A8
  链中间收窄会累积误差。

输出对两条路径 **bit-identical**（`Aqk`/`Akk` maxdiff 恰为 0，fixed/varlen/
`T<BC` 均已验证）。

`BK` 随 `FUSE_DIAG` 反转。未融合时 kernel 是读带宽受限，宽 tile 靠减少扫过
q/k/g 的趟数取胜；融合后多出八个 `[BC,BC]` 累加器、一个 `q0` tile 和 A2/A4/A8
链，kernel 变成**寄存器受限**——`T=16384,H=64` 下 `BK=32` 是 2.46 ms、
`BK=64` 是 2.90 ms（546 vs 463 GB/s），窄 tile 尽管要读四趟反而更快。因此
fixed/varlen 未融合时仍取 `BK=64`（fixed `s2`、varlen `s1`），融合时取
`BK=32,w1,s2`。

BW1100 实测（intra 阶段 = polynomial + inter 之和，两臂交错采样以抵消频率
漂移）：

| shape | split | fused | 加速 | e2e 加速 |
|---|---:|---:|---:|---:|
| `T=16384,H=64` | 2.850 ms | 2.175 ms | 1.310x | 1.083x |
| `T=16384,H=32` | 1.339 ms | 1.105 ms | 1.212x | 1.062x |
| `T=16384,H=16` | 0.671 ms | 0.601 ms | 1.116x | 1.023x |
| `T=4096,H=64` | 0.757 ms | 0.609 ms | 1.243x | 1.054x |
| `T=4096,H=32` | 0.404 ms | 0.361 ms | 1.118x | 1.038x |
| `T=512,H=16` | 0.120 ms | 0.121 ms | 0.991x | 1.082x |

`T=512` 的 intra 本身持平，e2e 仍然变快，因为少了一次 kernel launch——短序列
上主导的是 enqueue 而不是 device time。

融合后实测 546 GB/s，仍低于 837 的峰值：省下的流量只兑现了约三成
(预测 1.44 ms，实测 0.68 ms)。剩下的差距来自并行度——inversion 现在集中在
1/4 的 program 里（每 program 24 个 16x16 dot，原来是 6 个），延迟更难藏。
这是下一步的着力点，不是流量问题。

fixed 之前用的是 `BK=32`。`K=128` 时那是四趟扫过 q/k/g 的行，而 `BK=64` 只需
两趟；该 kernel 是读带宽受限而非寄存器受限，所以宽 tile 是纯赚。BW1100 实测
输出 **bit-identical**（`Aqk`/`Akk` maxdiff 恰为 0），耗时：

| shape | `BK=32` | `BK=64,s2` | 加速 |
|---|---|---|---|
| `T=16384,H=64` | 2.243 ms | 1.811 ms | 1.24x |
| `T=4096,H=64` | 0.540 ms | 0.430 ms | 1.26x |
| `T=16384,H=16` | 0.445 ms | 0.358 ms | 1.24x |
| `T=4096,H=16` | 0.128 ms | 0.111 ms | 1.16x |

## 5. 性能与证据

BW1000_H、`H=24,T=8192,K=V=128`：

| 逻辑路径 | Layout | Baseline XCU | 最终 XCU | 加速 | 有效 HBM GB/s（model） |
|---|---|---:|---:|---:|---:|
| 完整 intra（token→polynomial + inter） | fixed | `1.744800 ms` | `0.612960 ms` | `2.846x` | `739.37` |
| 完整 intra（token→polynomial + inter） | varlen | `3.144800 ms` | `0.608800 ms` | `5.165x` | `780.94` |
| polynomial component | fixed/varlen | `1.314400/2.723840 ms` | `0.260480/0.261120 ms` | `5.046x/10.430x` | `4440.44/9223.84` |
| inter component | fixed/varlen | `0.430400/0.420960 ms` | `0.352480/0.347680 ms` | `1.221x/1.211x` | `813.09/887.41` |

组件不与完整 intra 路径重复计入；所有路径无 scratch/spill。

2026-08-27 XCU 复核将 token-parallel/polynomial 与 inter 两个 dispatch 的
`execution` 相加；HBM 为相同 traffic model 的有效带宽换算值，本机 XCU
memory PMC 返回 `0 metrics`。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_chunk.py
```

源码：`boltops/fla/kda/triton/chunk_intra.py`、
`boltops/fla/kda/triton/chunk_intra_token_parallel.py`。
