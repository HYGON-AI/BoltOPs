# `FLA/KDA Triton Kernel 使用说明`

Kimi-K3 Kernelized Delta Attention（KDA）的完整 forward Prefill/Chunk 与
Decode/Recurrent Triton Kernel 集合。当前包已包含上游 vLLM AMD KDA 的四个
Prefill orchestration 入口，以及底层 kernel/wrapper。

## 1. 算子用途与数学语义

KDA 使用 recurrent state 表示线性注意力。以数学布局
`S_t ∈ R^[H,K,V]` 表示 state，单 token recurrent 更新可写为：

```text
q_hat_t = l2norm(q_t) * scale
k_hat_t = l2norm(k_t)

S_decay = exp(g_t)[..., None] * S_(t-1)
delta_v = v_t - k_hat_t @ S_decay
S_t     = S_decay + outer(beta_t * k_hat_t, delta_v)
o_t     = q_hat_t @ S_t
```

其中：

- `g_t` 是按 key 维度作用的负衰减 gate
- `beta_t = sigmoid(raw_beta_t)`
- state 在 Kernel 内实际按 `[V, K]` 存储，因此缓存 shape 为
  `[cache_slots, H, V, K]`
- `lower_bound` 非空时，gate 使用有界 sigmoid；否则使用 softplus 路径

Prefill 不逐 token 启动 recurrent Kernel，而是按 `BT=64` 分块，依次计算：

```text
raw gate / raw beta
    -> gate 激活 + beta sigmoid + chunk-local cumsum
    -> chunk 内 Aqk / Akk 与三角求解
    -> WY 张量 w / u 重建
    -> chunk state 项与 chunk 内局部项合并
```

下面的数据流图给出各算子的输入、输出和先后关系。虚线节点是完整 Prefill
编排中包含本仓库新增的 state propagation 实现。

```mermaid
flowchart LR
    subgraph P[Prefill / Chunk]
        P0[raw_g / raw_beta / A_log / bias]
        P1[fused_kda_gate_chunk_cumsum<br/>gate、beta 与 chunk-local cumsum]
        P2[chunk_kda_fwd_intra<br/>生成 Aqk 与三角求解后的 A]
        P3[recompute_w_u_fwd<br/>生成 w、u、kg]
        PX[[chunk state propagation<br/>生成 chunk state h 与 v_new]]
        P4[chunk_gla_fwd_o_gk<br/>state 项 + tril(Aqk) @ v_new]
        PO[Prefill output / final state]

        P0 --> P1
        QK[q / k] --> P2
        P1 -->|gk、beta| P2
        KV[k / v] --> P3
        P1 -->|gk、beta| P3
        P2 -->|A| P3
        P3 -->|w、u、kg| PX
        P1 -->|gk| PX
        P2 -->|Aqk| P4
        Q[q] --> P4
        PX -->|h、v_new| P4
        P4 --> PO
        PX -->|final state| PO
    end

    subgraph D[Decode / Recurrent]
        D0[q / k / v 或 packed QKV]
        D1[raw_g / raw_beta / A_log / dt_bias]
        D2[_kda_gate_beta_fwd_kernel<br/>可选的 gate、beta 预处理]
        D3[fused_recurrent_kda_fwd_kernel<br/>dense / multi-token recurrence]
        D4[fused_recurrent_kda_packed_decode_3d_kernel<br/>packed one-token recurrence]
        DS[(FP32 recurrent state)]
        DO[Decode output / updated state]

        D1 -->|非融合路径| D2
        D2 -->|g、beta| D3
        D0 --> D3
        D1 -->|融合 gate、beta| D3
        D0 --> D4
        D1 --> D4
        DS --> D3
        DS --> D4
        D3 --> DO
        D4 --> DO
    end

    classDef external stroke-dasharray: 6 4,fill:#f7f7f7,color:#555;
    class PX external;
```

当前移植提供上述直接 Triton 计算组件，包括本地 `chunk_state.py` 中的 state
propagation；上游依赖的公共 cumsum/l2norm helper 也已在本仓库补齐。

## 2. 接口说明

公开接口从 `bolt_ops.fla.kda` 导入：

```python
from bolt_ops.fla.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
    recompute_w_u_fwd,
)
```

### 2.1 Gate 激活与 chunk-local cumsum

```python
def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

```python
g_cumsum, beta_out = fused_kda_gate_chunk_cumsum(
    raw_g,
    raw_beta,
    A_log,
    g_bias=dt_bias,
    cu_seqlens=cu_seqlens,
)
```

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `raw_g` | `[B,T,H,D]` | BF16/FP16 | 未激活的 gate 投影 |
| `raw_beta` | `[B,T,H]` | BF16/FP16 | 未激活的 beta 投影 |
| `A_log` | `[H]` | FP32 | 每个 head 的 gate 参数 |
| `g_bias` | `[H,D]` 或展平等价布局 | FP32 | 可选 gate bias |
| `cu_seqlens` | `[N+1]` | int32/int64 | packed sequence 边界 |
| `chunk_indices` | `[NT,2]` | 同 `cu_seqlens` | 可选 `(sequence, local_chunk)` 表 |
| 返回 `g_cumsum` | `[B,T,H,D]` | 默认 FP32 | chunk 内累计 gate，已乘 `1/ln(2)` |
| 返回 `beta_out` | `[B,T,H]` | FP32 | `sigmoid(raw_beta)` |

只需要逐 token gate、不需要 cumsum 时可调用：

```python
gate = fused_kda_gate(raw_g_2d, A_log, head_k_dim, g_bias=dt_bias)
```

### 2.2 Prefill intra-chunk 矩阵

```python
def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
):
    ...
```

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `q` / `k` | `[B,T,H,K]` | BF16/FP16 | query / key |
| `gk` | `[B,T,HV,K]` | FP32 | chunk 内累计且按 `1/ln(2)` 缩放的 gate |
| `beta` | `[B,T,HV]` | FP32 | 更新系数 |
| `scale` | scalar | float | 通常为 `K**-0.5` |
| `cu_seqlens` | `[N+1]` | int32/int64 | packed sequence 边界 |
| `safe_gate` | scalar | bool | 选择 sub-chunk safe-gate 路径 |
| 返回 `Aqk` | `[B,T,HV,BT]` | 同 `q` | chunk 内 QK 系数矩阵 |
| 返回 `Akk` | `[B,T,HV,BT]` | 同 `q` | 三角求解后的 KDA 系数矩阵 |

`Aqk/Akk` 只有每个 chunk 的有效下三角区域有定义；上三角或尾部 padding 不应读取。

底层 token-parallel 路径允许调用方复用输出 buffer：

```python
def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

该接口原地写入 `Aqk` 和 diagonal-block buffer `Akk`，并返回同一对 tensor。

### 2.3 WY 重建与 chunk output

```python
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
    ...
```

实际返回值为：

```python
w, u, qg, kg = recompute_w_u_fwd(...)
```

当前 wrapper 不分配 `qg`，因此第三项固定为 `None`；传入 `gk` 时第四项返回
`kg`。主要 shape：

| 参数/返回值 | Shape | 常见 dtype |
|---|---|---|
| `k` / `w` / `kg` | `[B,T,H,K]` | BF16/FP16 |
| `v` / `u` | `[B,T,H,V]` | BF16/FP16 |
| `beta` | `[B,T,H]` | FP32/BF16 |
| `A` | `[B,T,H,BT]` | BF16/FP16 |
| `gk` | `[B,T,H,K]` | FP32 |

最终 chunk output 接口：

```python
output = chunk_gla_fwd_o_gk(
    q, v, gk, Aqk, chunk_states, output,
    scale=K**-0.5,
    cu_seqlens=cu_seqlens,
)
```

它计算 chunk 初始 state 贡献与 `tril(Aqk) @ v` 的和，并原地写入传入的
`output` tensor。

### 2.4 Packed Decode

```python
def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `mixed_qkv` | `[tokens, 2*H*K + H*V]` | BF16 | packed Q/K/V；Kimi-K3 常见 `K=V` |
| `raw_g` | `[1,tokens,H,K]` | BF16 | 未激活 gate |
| `raw_beta` | `[1,tokens,H]` | BF16 | 未激活 beta |
| `A_log` | `[H]` | FP32 | gate 参数 |
| `dt_bias` | `[H,K]` 或连续展平布局 | FP32 | gate bias |
| `initial_state` | `[cache_slots,H,V,K]` | FP32 | recurrent cache，原地更新 |
| `state_indices` | `[tokens]` | int32/int64 | 每个 token 对应的 cache slot；`<=0` 输出零 |
| 返回 `output` | `[1,tokens,H,V]` | 同 `mixed_qkv` | decode 输出 |
| 返回 `state` | 同 `initial_state` | FP32 | 与输入 state 为同一 tensor |

多 token/speculative decode 使用 `fused_recurrent_kda`；已完成 gate/beta 预处理时可
直接调用 `fused_recurrent_kda_fwd`。两者参数签名保持上游源码不变。

### 2.5 前置条件

- 所有计算 tensor 位于同一 CUDA/HIP device
- 最内维连续；recurrent state 的内层 stride 必须为 `(V*K, K, 1)`
- packed decode 的 `mixed_qkv` 必须为二维且最后一维连续
- varlen Prefill 当前要求 `B=1`，序列由 `cu_seqlens` 分隔，chunk 不跨序列
- 默认 `BT=64`，intra sub-chunk 大小固定为 `BC=16`
- 累计 gate 供 `exp2` Kernel 使用，必须处于 log2 空间，即自然对数累计值乘
  `1/ln(2)`
- 本移植为 forward-only，不包含 backward

### 2.6 KDA kernel family

当前实现新增并导出的 KDA kernel/wrapper 如下：

| 类别 | Kernel / wrapper | 用途 |
|---|---|---|
| Prefill orchestration | `chunk_kda`, `chunk_kda_fwd` | materialized gate 的完整 prefill 编排 |
| Fused-gate orchestration | `chunk_kda_with_fused_gate`, `chunk_kda_with_fused_gate_fwd` | raw gate/beta 到 output 的完整路径 |
| Q/K normalization | `l2norm_fwd_kernel`, `l2norm_fwd` | q/k L2 归一化 |
| Gate cumsum | `kda_gate_chunk_cumsum_vector_kernel`, `fused_kda_gate_chunk_cumsum` | gate、beta 和 chunk-local cumsum |
| Intra polynomial | `chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized` | FP32 diagonal polynomial solve |
| Intra/inter solve | `chunk_kda_fwd_kernel_inter_solve_fused` | Aqk/Akk inter solve |
| Token-parallel fallback | `chunk_kda_fwd_kernel_intra_token_parallel`, `chunk_kda_fwd_kernel_intra_subchunk_raw_optimized` | 通用 intra 路径 |
| WY reconstruction | `recompute_w_u_fwd_kernel_beta_factored_head_first` | 重建 w/u/kg |
| State propagation | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 更新 chunk recurrent state |
| Chunk output | `chunk_gla_fwd_kernel_o_local_first` | 合并 local 项和 state 项 |
| Dense recurrent decode | `fused_recurrent_kda_fwd_kernel` | dense/spec decode |
| Packed decode | `fused_recurrent_kda_packed_decode_3d_kernel` | packed bounded-gate decode |

其中新增的 `chunk_state.py`、`common/triton/cumsum.py` 和
`common/triton/l2norm.py` 分别提供 state propagation、chunk-local cumsum 和
q/k normalization；完整 dispatch 耗时及采集命令见
[kda_performance.md](./kda_performance.md)。

## 3. 典型 Shape

Kimi-K3/vLLM 测试中使用的代表配置：

| 场景 | `B/T` | `H` | `K/V` | `BT/BC` | dtype |
|---|---:|---:|---:|---:|---|
| Chunk Prefill | `1 / 15–8192` | `24`、`32` 或 `64` | `128 / 128` | `64 / 16` | FP16/BF16 |
| Packed Decode | `tokens=1–32` | `8–96` | `128 / 128` | N/A | BF16 + state FP32 |
| Spec Decode | `N=3, query_len=3` | `12` 或 `96` | `128 / 128` | N/A | BF16 + state FP32 |

本仓库 benchmark 使用：

Kimi-K3 总 KDA head 数为 `96`。按照 SGLang 的并行方式，TP32、CP8 且使能
DPA 时，attention tensor parallel degree 为 `attn_tp=TP/CP=32/8=4`，所以
本地 `H=96/4=24`；普通 TP16/TP32 分别得到 `H=96/16=6` 和
`H=96/32=3`。完整 head（attention 未切分）路径保留 `H=96`：

| 并行场景 | 本地 head 推导 | benchmark `H` |
|---|---:|---:|
| 完整 head / attention 未切分 | `96` | `96` |
| TP32 + CP8 + DPA | `96 / (32 / 8)` | `24` |
| TP16 | `96 / 16` | `6` |
| TP32 | `96 / 32` | `3` |

所有 Prefill/共用 Kernel 均覆盖
`B=1,T=8192,H∈{96,24,6,3},K=V=128,BT=64,BC=16`。Gate Kernel 使用
对应的 `raw_g/raw_beta` 视图，其余 Prefill Kernel 使用已处理的逐通道 `g`
和 `beta`。

Decode benchmark 保留原有 `tokens=6,H=24,K=V=128` 和
`tokens=1,H=3,K=V=128`，再为同一 `tokens=6` 网络 shape 增加
`H=96/6/3`。对应 packed QKV 最后一维 `3*H*128` 分别为
`36864/9216/2304/1152`。Decode cache slot `0` 是无效 sentinel，benchmark
分配 `[tokens+1,H,128,128]` FP32 state，并使用
`state_indices=[1,...,tokens]`，保证每个 token 都实际执行 recurrence。原有常见
Decode 调度尺寸 `tokens/H=4/3`、`8/12`、`16/24` 继续保留。

所有支持 `cu_seqlens` 的 Prefill benchmark 同时保留 fixed-length 路径，并增加
`cu_seqlens=[0,3000,8192]` 的 packed 多序列 varlen 路径（包含两个 partial
chunk），用于分别选择 `IS_VARLEN=False/True` 下的配置；测试输入中的逐通道
`gk` 同时在每个序列和 chunk 边界重新累计。

| Case | 覆盖的 Kernel / 路径 |
|---|---|
| `gate` | `kda_gate_fwd_kernel` |
| `gate-chunk-cumsum` | `kda_gate_chunk_cumsum_vector_kernel` |
| `intra-token-parallel` | `chunk_kda_fwd_kernel_intra_subchunk_raw_optimized`；其他 shape 回退上游 token kernel |
| `prefill-intra-token` | 模型 shape 使用 polynomial 预解 diagonal + `chunk_kda_fwd_kernel_inter_solve_fused`；其他 shape 回退 token-parallel |
| `prefill-intra-safe` | polynomial 预解 diagonal + `chunk_kda_fwd_kernel_inter_solve_fused` |
| `recompute-w-u` | beta-factored head-first sibling；其他 dtype/shape 回退 `recompute_w_u_fwd_kernel` |
| `chunk-output` | local-first sibling；其他 shape 回退 `chunk_gla_fwd_kernel_o` |
| `decode-gate-beta` | `_kda_gate_beta_fwd_kernel` |
| `fused-recurrent` | `fused_recurrent_kda_fwd_kernel`（gate/beta 融合路径） |
| `packed-decode` | bounded-gate direct-3D sibling；其他语义回退上游 kernel |

Decode-only recurrent Kernel 不复用 Prefill Shape，分别使用上述实际 Decode Shape。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/fla/test_kda_*.py
```

各测试文件职责如下：

| 测试文件 | 测试内容 |
|---|---|
| `test_kda_gate.py` | q/k l2norm、KDA gate、beta sigmoid、fixed/varlen chunk-local cumsum，以及 bounded/softplus gate 路径 |
| `test_kda_chunk.py` | WY `w/u/kg` 重建、完整 fused-gate Prefill 链路，以及 chunk output |
| `test_kda_state.py` | chunk state 更新、`h`、`v_new` 和 final state 与 Torch reference 的一致性 |
| `test_kda_decode.py` | dense recurrent、融合 gate、materialized gate 和 packed decode 的 output/state 一致性 |
| `_kda_reference.py` | 提供 gate、recurrent、WY 和 state 的 PyTorch reference，不执行 Triton kernel |

测试重点包括：Prefill 各阶段的 tensor 依赖、bounded/softplus 两种 gate 语义、
fixed/varlen cumsum 边界，以及 packed decode 与 dense recurrent 的结果一致性。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `bolt_ops/fla/kda/__init__.py` |
| Triton 导出 | `bolt_ops/fla/kda/triton/__init__.py` |
| Chunk Kernel | `bolt_ops/fla/kda/triton/chunk.py` |
| Intra Kernel | `bolt_ops/fla/kda/triton/chunk_intra.py` |
| Token-parallel Kernel | `bolt_ops/fla/kda/triton/chunk_intra_token_parallel.py` |
| Recurrent Kernel | `bolt_ops/fla/kda/triton/fused_recurrent.py` |
| Chunk state Kernel | `bolt_ops/fla/kda/triton/chunk_state.py` |
| Common cumsum helper | `bolt_ops/fla/common/triton/cumsum.py` |
| Common l2norm helper | `bolt_ops/fla/common/triton/l2norm.py` |
| 去 vLLM 兼容 helper | `bolt_ops/fla/common/triton/math.py`、`utils.py` |
| 正确性测试 | `test/fla/test_kda_*.py`、`test/fla/_kda_reference.py` |
| 性能测试 | `test/fla/benchmark_kda_baseline.py`、`test/fla/xcu_kda_stage.py` |

范围：KDA forward 目录直接定义的 Triton Kernel、state propagation 和完整
`chunk_kda` orchestration；不含 backward。

## 附录 B. 溯源

- vLLM PR：[`#50000`](https://github.com/vllm-project/vllm/pull/50000)
- 固定来源 commit：[`5a3eba03481d879451d89b238f9160fa1c1faddf`](https://github.com/vllm-project/vllm/commit/5a3eba03481d879451d89b238f9160fa1c1faddf)
- 来源目录：[`vllm/models/kimi_k3/amd/ops/third_party/kda`](https://github.com/vllm-project/vllm/tree/5a3eba03481d879451d89b238f9160fa1c1faddf/vllm/models/kimi_k3/amd/ops/third_party/kda)
- 正确性参考：[`tests/models/kimi_k3/test_kda.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/tests/models/kimi_k3/test_kda.py)

## 附录 C. 实现与同步备注

- Prefill intra 的模型 shape 使用 polynomial sibling 生成预解 diagonal，再由
  `chunk_kda_fwd_kernel_inter_solve_fused` 完成跨 sub-chunk 项；其他 shape 的
  `safe_gate=False` 回退 token-parallel raw diagonal 与原 inter solve。
- `safe_gate=True` 的 fallback 仍使用原 `chunk_kda_fwd_kernel_intra_sub_chunk`。
- Decode state 采用 `[V,K]` 内存布局，数学参考通常采用 `[K,V]`；对照时需转置。
- 上游 Kernel 定义和函数体保持不改；优化以名称带 `optimized`、`local_first`
  或 `3d` 的 sibling 加入，并只在 wrapper 中做静态配置分派，便于后续同步。
- 后续同步只在 import、`math.py`、`utils.py` 处理去 vLLM 依赖；不要在移植文件内
  做无关格式化、重排或本地优化。

## 附录 D. Kernel 与 wrapper 对照

| Kernel | Wrapper |
|---|---|
| `recompute_w_u_fwd_kernel_beta_factored_head_first` / upstream fallback | `recompute_w_u_fwd` |
| `chunk_gla_fwd_kernel_o_local_first` / `chunk_gla_fwd_kernel_o` fallback | `chunk_gla_fwd_o_gk` |
| `kda_gate_chunk_cumsum_vector_kernel` | `fused_kda_gate_chunk_cumsum` |
| `kda_gate_fwd_kernel` | `fused_kda_gate` |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `chunk_kda_fwd_intra` |
| `chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized` / upstream fallback | `chunk_kda_fwd_intra` |
| `chunk_kda_fwd_kernel_intra_subchunk_raw_optimized` / upstream fallback | `chunk_kda_fwd_intra_token_parallel` |
| `_kda_gate_beta_fwd_kernel` | `_fused_kda_gate_beta` / `fused_recurrent_kda` |
| `fused_recurrent_kda_fwd_kernel` | `fused_recurrent_kda_fwd` |
| `fused_recurrent_kda_packed_decode_3d_kernel` / upstream fallback | `fused_recurrent_kda_packed_decode` |

## 附录 E. BW1100 离线调优记录

### E.1 离线调优与静态配置

下表配置在 BW1100 上离线实测后收敛得到。本轮算法改写以实际 Prefill
`B=1,T=8192,H=HV=24,K=V=128,BT=64,BC=16` 为主证据；benchmark 另覆盖
并行推导得到的 `H=96/6/3`。新增 intra
sibling 另外通过 fixed `B=2,T=130` 和 varlen `cu_seqlens=[0,33,178]`
的多序列、尾块 smoke。Decode selector 覆盖 `(tokens,H)=(1,3),(4,3),`
`(6,3),(6,6),(6,24),(6,96),(8,12),(16,24)`。

源码不使用 `@triton.autotune`。每个 Kernel 以模块级 `*_CONFIGS` 保留 2～3 个
离线筛选结果，由对应的 `get_*_config` 根据 shape、fixed/varlen 路径或融合特征
直接返回唯一配置，wrapper 随后只 launch 一次。这与 MoE activation 的静态配置
选择方式一致，不产生首次调用在线试跑、缓存依赖或 recurrent state 重复写入。

| Kernel | 保留的 Config（`w`=warps，`s`=stages） |
|---|---|
| `recompute_w_u_fwd_kernel_beta_factored_head_first` | `(BK64,BV128,w4,s2)`, `(BK64,BV128,w4,s1)`, `(BK64,BV64,w4,s2)` |
| `chunk_gla_fwd_kernel_o_local_first` | `(BK128,BV128,w4,s2)`, `(BK128,BV128,w8,s2)`, `(BK64,BV64,w8,s2)` |
| `kda_gate_chunk_cumsum_vector_kernel` | `(BS32,w2)`, `(BS32,w4)`, `(BS64,w2)` |
| `kda_gate_fwd_kernel` | `(BT32,w16,s2)`, `(BT64,w4,s2)`, `(BT128,w4,s2)` |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `(BK64,w1,s2)`, `(BK64,w1,s1)`, `(BK32,w1,s1)` |
| `chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized` | `(BK128,w1,s1,early=false)`, `(BK128,w1,s1,early=true)`, `(BK128,w1,s2,early=true)` |
| `chunk_kda_fwd_kernel_intra_subchunk_raw_optimized` | `(BK128,w1,s1)`, `(BK128,w1,s2)`, `(BK128,w1,s3)` |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `(BV64,w4,s2)` baseline；|
| `_kda_gate_beta_fwd_kernel` | `(BT2,w2,s3)`, `(BT2,w8,s2)`, `(BT8,w4,s3)` |
| `fused_recurrent_kda_fwd_kernel` | `(BV8,w1,s2)`, `(BV16,w1,s1)`, `(BV8,w1,s1)` |
| `fused_recurrent_kda_packed_decode_3d_kernel` | `(BV8,w1,s2)`, `(BV8,w1,s1)`, `(BV32,w2,s2)` |

实际 Prefill Shape 静态选择为：recompute `(BK64,BV128,w4,s2)`、chunk
output `(BK128,BV128,w4,s2)`、gate cumsum `(BS64,w2)`、gate
`(BT64,w4,s2)`、inter solve `(BK64,w1,s2)`、safe polynomial
`(BK128,w1,s1,early=false)`、token intra `(BK128,w1,s1)`。有界 gate、
`K=V=128`、FP32 state 的 Packed Decode 使用 direct-3D sibling；`tokens=1`
选择 `(BV32,w2,s2)`，`tokens=2～6` 选择 `(BV8,w1,s1)`，`tokens=8` 选择
`(BV8,w1,s2)`，`tokens>=16` 选择 `(BV32,w2,s2)`。

### E.2 性能结果

Prefill 使用 XCU 的 kernel dispatch 设备时间；`HBM` 为
`L2_Fabric-Rd BW + L2_Fabric-Wr BW`，不是按 tensor 总字节估算的 logical BW。
Baseline 是本轮开始前的上游实现，最终列为当前静态 dispatch。未获真实设备收益
的算子保持原实现。Decode 的总体加速使用五组实际/常见 shape 的 event 均值，
并给出一个 XCU 代表点，因此不能把表中的 Decode 代表点当作五组 shape 的统一
绝对时延。

#### E.2.1 全部算子性能总表

| # | 逻辑算子 / 最终 Kernel | 性能证据 | 实测加速 | 最终 HBM GB/s | 计入保守平均 |
|---:|---|---|---:|---:|---:|
| 1 | gate / `kda_gate_fwd_kernel` | XCU `0.128160→0.128160 ms` | 1.000x | 1220.96 | 1.000x |
| 2 | gate chunk cumsum / `kda_gate_chunk_cumsum_vector_kernel` | XCU `0.232320→0.232320 ms` | 1.000x | 772.73 | 1.000x |
| 3 | recompute W/U / beta-factored sibling | XCU `0.369120→0.299480 ms` | 1.233x | 1251.93 | 1.232536x |
| 4 | chunk output / local-first sibling | XCU `0.442560→0.326800 ms` | 1.354x | 1129.40 | 1.354224x |
| 5 | token intra raw / subchunk sibling | XCU `0.781560→0.199880 ms` | 3.910x | 438.86 | 3.910146x |
| 6 | 16×16 diagonal inverse / FP32 polynomial sibling | XCU `0.541160→0.247500 ms` | 2.186x | 930.10 | 2.186506x |
| 7 | inter solve / `chunk_kda_fwd_kernel_inter_solve_fused` | XCU `0.3979→0.2137 ms` | 1.862x（工作前移） | 1078.50 | 1.000x |
| 8 | decode gate/beta / `_kda_gate_beta_fwd_kernel` | 未得到稳定 XCU 收益，保持原融合实现 | 1.000x | — | 1.000x |
| 9 | dense recurrent / `fused_recurrent_kda_fwd_kernel` | 五 shape event 均值 1.053x；代表 XCU `20.32→16.48 us` | 1.053x | 764.00（代表点） | 1.053000x |
| 10 | packed decode / direct-3D sibling | 五 shape 历史均值 1.1262x；T1/H3 selector 复测 1.0368x | 1.1262x（保守） | 56.73（历史代表点） | 1.126200x |

inter solve 的 `1.862x` 来自把 diagonal inversion 前移到 polynomial sibling 后
跳过串行求解；它不是独立消失的工作，因此总平均按 `1.000x` 计，避免与第 6 项
重复计算。T1/H3 selector 的新复测没有混入原五 shape 历史均值，避免混用不同
采样批次；因此按此口径，十个逻辑算子的保守算术平均为 `1.486261x`，即提升
`48.63%`。若按“每个 kernel dispatch 均独立计数”的口径把 inter solve 的
`1.862x` 也计入，十项算术平均为 `1.572461x`，即提升 `57.25%`；前者更接近
完整 pipeline，后者对应“所有 kernel 平均性能”的验收口径。

#### E.2.2 每个算子的优化方法

| 逻辑算子 | 最终采用的优化方法 | 安全性、回退与未采用方案 |
|---|---|---|
| gate | 按 `T/H` 离线选择 `BT32/64/128`、warps 和 stages；运行时只选择一个静态配置 | 配置变化没有形成稳定 XCU 收益，保留上游 Kernel 计算逻辑 |
| gate chunk cumsum | 为长序列和 varlen 选择 `BS64/w2`，小 shape 保留 `BS32` 候选 | full-D 合并候选 `0.288821 ms`，慢于原路径 `0.286208 ms`，未合入 |
| recompute W/U | head-first grid；将 `beta` 前移成共享的 `A*beta` tile，供 W/U 两个 dot 复用；模型 shape 选 `BK64/BV128/w4/s2` | 仅 BF16 `k/v/A`、`gk` 存在、`BT=64,K/V<=128` 启用；其余回退上游 Kernel。`BK128/BV128` 因更慢未采用 |
| chunk output | local-first：先计算 causal `A@V`，再累加 state `QG@H`，缩短 `q/g/h` tile 的存活区间；模型 shape 选 `BK128/BV128/w4/s2` | 仅 `BT=64,K/V>=128` 启用 sibling；其他 shape 回退上游 Kernel |
| token intra raw | 将 program 粒度从逐 token 改成逐 head、逐 16-token subchunk；一次加载复用 `q/k/g` tile，并用两个 head-wise dot 同时生成 `Aqk/Akk` | 仅 `K=128,BT=64,BC=16` 启用；保留 causal mask、tail 和 varlen 边界，其他 shape 回退原 token Kernel |
| 16×16 diagonal inverse | 利用 strict-lower 矩阵 `A^16=0`，在寄存器中用 `(I-A)(I+A²)(I+A⁴)(I+A⁸)` 计算 `(I+A)⁻¹`，替代逐行 forward substitution 和中间同步 | MMAC 输入、累加及 `A2/A4/A8/X` 级间值全部为 FP32；BF16 输入候选已撤回；仅 `BC=16` 使用 |
| inter solve | 接收已经求逆的 diagonal block，以 `USE_SAFE_GATE=True` 跳过四组串行 forward-substitution；静态选择 `BK64/w1/s2` | 求逆工作已前移到上一算子，所以只报告组件时延，不把该收益重复计入总体平均 |
| decode gate/beta | 保持 gate 与 beta 的单 launch 融合；按 token/head 静态选择 `BT2/BT8` 和 warps/stages | 单独物化 gate/beta 会增加 allocation 和 launch，全部 shape 都更慢；没有稳定收益时保持原实现 |
| dense recurrent | 根据 token/head 和融合特征选择较小 `BV8/BV16`、`w1` 和 `s1/s2`，降低 live state tile 与 VGPR 压力；不改递推公式 | 3D-grid 候选在同 decorator/heuristics 条件下未复现收益并回退；保留原 flattened 映射 |
| packed decode | 新增 `(V tile,head,token)` direct-3D grid，去掉 flattened program-id 映射；T1/H3 改选已有 `BV32/w2/s2`，24 轮 A/B、B/A 复测中位 `1.0368x` | 仅 bounded gate、`K=V=128`、FP32 state 启用；T6/H24 等轻微回退 shape 保持原 selector；新 selector 收益不混入历史五 shape 总均值 |

recompute 的 fixed/varlen CUDA-event 复测分别为
`0.427875→0.388867 ms`（1.100x）和 `0.438165→0.393283 ms`
（1.114x）。beta 前移到共享 `A` 后，W/U 两个 dot 复用同一 tile；XCU 显示
VGPR `189→171`，无 scratch/spill。该思路来自仓库已有 standalone
`recompute_w_u`，但 KDA 保留逐通道 `gk` 与 `kg/qg` 语义，不能直接调用前者。

safe sub-chunk 的 16×16 逆多项式将 MMAC 输入、累加和级间值全部保持 FP32，
不采用曾测试过的 BF16 输入优化。FP32 polynomial 的 XCU 为 `0.2475 ms`，
HBM `930.1 GB/s`，VGPR `192`，无 scratch/spill；相对上游 safe 路径仍有
`2.186x` 加速。BF16 输入候选虽然达到 `0.2223 ms`，但会在 A2/A4/A8 链中
逐级引入舍入误差，已按安全优先原则撤回。

模型 shape 的 token prefill 将 diagonal inversion 前移到 polynomial sibling，
随后 inter 以预解模式跳过四组串行前代。完整两阶段 fixed/varlen 为
`0.5034/0.5049 ms`。独立观察 inter dispatch 为约
`0.3979→0.2137 ms`、HBM `602.6→1078.5 GB/s`；由于这部分工作被前移，下面
的平均值保守地把 inter 记为 1.000x，避免重复计功。

以下 Decode 性能表仍记录本轮调优时使用的原五组 shape；新增的
`tokens=6,H=96/6/3` 作为并行 shape 覆盖，不回写或重算历史性能均值。原五组
shape 均使用修正后的 cache sentinel。静态配置的历史 CUDA-event 算术平均：
dense recurrent 为 1.053x，packed decode 为 1.1262x。本轮另用 3 组随机输入、
每组 8 轮且交替 A/B、B/A 顺序复测 packed T1/H3：`BV32/w2/s2` 相对
`BV8/w1/s1` 的 24 轮总体中位为 `1.036788x`，输出最大误差为 0，state 最大
误差为 `5.96e-8`。由于它与历史五 shape 数据不是同一采样批次，下面的总平均
不把这项额外收益重复外推。两个历史 XCU 代表点如下。

| Decode 代表点 | Baseline XCU | 最终 XCU | 加速比 | HBM GB/s | VGPR |
|---|---:|---:|---:|---:|---:|
| packed `T1/H3` | 9.12 us | 6.88 us | 1.326x | 43.78→56.73 | 89→76 |
| dense `T8/H12` | 20.32 us | 16.48 us | 1.233x | 611.21→764.00 | 147→70 |

按十个逻辑 Kernel 做保守算术平均时，gate、cumsum、inter、decode gate/beta
均记为 1.000x，不计 inter 的工作前移收益；其余采用上表同语义设备时间，dense
和 packed 使用五组 benchmark shape 的均值：

```text
(1 + 1 + 1.232536 + 1.354224 + 3.910146
 + 2.186506 + 1 + 1 + 1.053000 + 1.126200) / 10
= 1.486261x，即保守平均提升 48.63%
```

已拒绝的候选包括：cumsum full-D（0.288821 ms，慢于 0.286208 ms）、
recompute 沿用 standalone 单块 `BK=BV=128` 的配置（约 0.443 ms，慢于
`BK=64/BV=128` 的约 0.383 ms）、decode BV64/BV128、物化 gate/beta 的额外
launch、BF16 inter block DAG、BF16 split scratch、raw+solve 两次 launch，
full 64×64 polynomial inverse，以及 16×16 polynomial 的 BF16 MMAC 输入。
本轮还拒绝了 cumsum 合并 beta（稳定复测 `0.99881x`）以及 recompute 的
`exp2(gn)/exp2(gk)` 复用；后者在 H24/H96 回退，并会缩小原
`exp2(gn-gk)` 的安全数值范围。dense 3D grid 在相同 heuristics 条件下也未
复现子 agent 的收益，直接替换会回退约 4%～6%，因此未合入。

### E.3 Fast path 与 fallback

- token raw sibling 仅用于 `K=128,BT=64,BC=16`；其他 shape 保留原
  vLLM token Kernel。完整 prefill 在同一 shape 使用 polynomial 预解路径，公共
  token raw wrapper 的输出契约不变。
- polynomial inverse 依赖 16×16 strict-lower 矩阵的 `A^16=0`，因此不向
  其他 `BC` 泛化。
- beta-factored recompute 仅用于 BF16 `k/v/A`、存在逐通道 `gk`、`BT=64`、
  `K,V<=128`；varlen 还要求 `B=1`，其他情况回退原 Kernel。
- chunk output local-first sibling 用于 `K,V>=128,BT=64`，其他 shape
  回退原 Kernel。
- Packed direct-3D 仅用于 `lower_bound is not None,K=V=128` 且 state 为
  FP32；softplus、其他 head dim/state dtype 回退原 Kernel。`tokens=1` 的
  `BV32/w2/s2` 仅为静态配置选择，不缩窄语义 shape。
- 所有 selector 都是静态分派，没有 runtime autotune。
