# KDA Prefill Orchestration 阶段说明

本文描述 KDA Prefill 的完整阶段串联。各阶段的具体接口、kernel 配置、fallback
和性能数据分别写在阶段文档中；本文只说明端到端调用边界和阶段间 tensor 契约。

## 1. 阶段数据流

```text
raw_g/raw_beta/A_log/g_bias
        |
        v
[1] Gate / chunk-local cumsum
        | gk, beta
        v
[2] Intra-chunk coefficients
        | Aqk, A
        v
[3] WY reconstruction
        | w, u, kg
        v
[4] State propagation
        | h, v_new
        v
[5] Chunk output
        |
        v
output / optional final_state
```

阶段文档：

1. [Gate 阶段](./kda_prefill_gate.md)
2. [Intra 阶段](./kda_prefill_intra.md)
3. [WY 重建阶段](./kda_prefill_wy.md)
4. [State 阶段](./kda_prefill_state.md)
5. [Output 阶段](./kda_prefill_output.md)
6. [Decode/Recurrent 阶段](./kda_decode.md)

## 2. Wrapper 使用方式

输入已经完成 gate/beta 处理时：

```python
o, final_state = chunk_kda_fwd(
    q, k, v, g, beta,
    scale=K**-0.5,
    initial_state=initial_state,
    output_final_state=False,
    cu_seqlens=cu_seqlens,
)
```

仍持有 raw gate/beta 时：

```python
o, final_state = chunk_kda_with_fused_gate_fwd(
    q, k, v, raw_g, raw_beta, A_log, g_bias,
    scale=K**-0.5,
    initial_state=initial_state,
    output_final_state=False,
    lower_bound=lower_bound,
    cu_seqlens=cu_seqlens,
)
```

`chunk_kda` 和 `chunk_kda_with_fused_gate` 是带默认参数的 facade；阶段 wrapper
仍可单独调用，用于中间 tensor 调试、复用或独立 benchmark。

四个 prefill 入口（`chunk_kda_fwd`/`chunk_kda_with_fused_gate_fwd`/
`chunk_kda`/`chunk_kda_with_fused_gate`）都接受 state pool 参数
`initial_state_indices`、`inplace_final_state`、`null_state_index`（语义见
[State 阶段](./kda_prefill_state.md)）。`chunk_kda_with_fused_gate` 的
fixed-schedule prefill（`prefill_optimized_fwd`）是单 sequence 路径，传入
`initial_state_indices` 或 `inplace_final_state=True` 时自动回退标准 pipeline。

## 3. fixed/varlen

- fixed：`cu_seqlens=None`，每个 batch row 长度均为 `T`，阶段可用规则 chunk id
  和 stride 寻址；
- varlen：`B=1`，提供 `cu_seqlens=[N+1]`，必要时生成
  `chunk_indices:[NT,2]`；cumsum、intra、state、output 都在 sequence 边界内
  计算，partial chunk 不跨 sequence；
- 两种 layout 使用同一数学语义，差异只在索引和边界处理。

## 4. 融合与 fallback

State/output fusion 是跨第 4、5 阶段的优化：在 `BT=64`、BF16 主路径、长序列、
同头数且 `output_final_state=False` 时，将两个阶段合并为一次 dispatch；请求
final state 时回退 split pipeline，以保持 final-state 归约语义。State standalone
wrapper、output wrapper 及上游 `Hg != H` 兼容路径仍保留。

其它阶段按自己的文档选择 fast path；任何 shape、dtype、stride、head mapping 或
边界条件不满足时，回退 upstream-compatible kernel，不改变公共 output/state 契约。

## 5. 端到端覆盖与性能

完整矩阵覆盖 `H={96,24,6,3,32}`、`T={2048,8192,38912}`、
`K,V={64,128,256}` 和 fixed/varlen，共 `270/270` 通过；每个点均为
`status=ok, finite=1`。本轮还修复并验证了 `chunk.py` 的 fusion wrapper 导入
和 `K=256` state 的 LDS-safe fallback。矩阵 acceptance 使用
[`run_kda_workload_matrix.py`](../../../test/fla/run_kda_workload_matrix.py)，
分片 CSV 为 H=96 与 H=24/6/3/32；其 wall time 含首次编译，不用于 kernel
性能结论。单阶段 XCU、HBM、VGPR 和加速比记录在各阶段文档中，不在此处重复
计算或简单相加。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_chunk.py test/fla/test_kda_state.py
```

源码：`boltops/fla/kda/triton/chunk.py`、`chunk_intra.py`、`chunk_state.py`。
