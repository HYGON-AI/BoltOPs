# `KDA Triton Kernel 使用说明`

Kimi-K3 Kernelized Delta Attention（KDA）的 Prefill/Chunk 与
Decode/Recurrent Triton Kernel 集合。当前包提供 KDA 目录内直接定义的 Kernel
及其 launch wrapper，不包含依赖 vLLM 公共 FLA Kernel 的完整 `chunk_kda` 编排。

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

当前移植提供上述直接 Triton 计算组件；state propagation 等来自 vLLM 公共 FLA
目录的步骤不在本次范围内。

## 2. 接口说明

公开接口从 `bolt_ops.kda` 导入：

```python
from bolt_ops.kda import (
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

## 3. 典型 Shape

Kimi-K3/vLLM 测试中使用的代表配置：

| 场景 | `B/T` | `H` | `K/V` | `BT/BC` | dtype |
|---|---:|---:|---:|---:|---|
| Chunk Prefill | `1 / 15–8192` | `24`、`32` 或 `64` | `128 / 128` | `64 / 16` | FP16/BF16 |
| Packed Decode | `tokens=1–32` | `8–96` | `128 / 128` | N/A | BF16 + state FP32 |
| Spec Decode | `N=3, query_len=3` | `12` 或 `96` | `128 / 128` | N/A | BF16 + state FP32 |

本仓库 benchmark 使用：

所有 Prefill/共用 Kernel 均使用同一组模型 Shape：
`B=1,T=8192,H=24,K=V=128,BT=64,BC=16`。Gate Kernel 使用对应的
`raw_g/raw_beta` 视图，其余 Prefill Kernel 使用已处理的逐通道 `g` 和 `beta`。

Decode benchmark 保留 `tokens=6,H=24,K=V=128`，并增加根据 TP-local 投影尺寸
推导出的 `tokens=1,H=3,K=V=128`。前一组 packed QKV 为 `[6,9216]`，后一组
为 `[1,1152]`。对应的 `raw_g` 分别为 `[1,6,24,128]`、
`[1,1,3,128]`，`raw_beta` 为 `[1,6,24]`、`[1,1,3]`，`dt_bias`
为 `[3072]`、`[384]`，recurrent state 为 `[6,24,128,128]`、
`[1,3,128,128]` FP32。

| Case | 覆盖的 Kernel / 路径 |
|---|---|
| `gate` | `kda_gate_fwd_kernel` |
| `gate-chunk-cumsum` | `kda_gate_chunk_cumsum_vector_kernel` |
| `intra-token-parallel` | `chunk_kda_fwd_kernel_intra_token_parallel` |
| `prefill-intra-token` | token-parallel diagonal block + `chunk_kda_fwd_kernel_inter_solve_fused` |
| `prefill-intra-safe` | `chunk_kda_fwd_kernel_intra_sub_chunk` + `chunk_kda_fwd_kernel_inter_solve_fused` |
| `recompute-w-u` | `recompute_w_u_fwd_kernel` |
| `chunk-output` | `chunk_gla_fwd_kernel_o` |
| `decode-gate-beta` | `_kda_gate_beta_fwd_kernel` |
| `fused-recurrent` | `fused_recurrent_kda_fwd_kernel`（gate/beta 融合路径） |
| `packed-decode` | `fused_recurrent_kda_packed_decode_kernel` |

Decode-only recurrent Kernel 不复用 Prefill Shape，分别使用上述实际 Decode Shape。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/test_kda.py

# 全部 Prefill、共用及 Decode KDA Kernel 性能
pytest -q test/test_kda.py --bench -s
```

正确性测试覆盖：

- gate、beta 与 chunk-local cumsum 对照 PyTorch
- `w/u/kg` 重建与 chunk output 对照 PyTorch
- token-parallel 和 safe-gate 两条 intra 路径交叉校验
- materialized gate/beta、融合 gate/beta 两条 recurrent 路径交叉校验
- packed decode 与 dense recurrent Kernel 交叉校验

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `bolt_ops/kda/__init__.py` |
| Triton 导出 | `bolt_ops/kda/triton/__init__.py` |
| Chunk Kernel | `bolt_ops/kda/triton/kda/chunk.py` |
| Intra Kernel | `bolt_ops/kda/triton/kda/chunk_intra.py` |
| Token-parallel Kernel | `bolt_ops/kda/triton/kda/chunk_intra_token_parallel.py` |
| Recurrent Kernel | `bolt_ops/kda/triton/kda/fused_recurrent.py` |
| 去 vLLM 兼容 helper | `bolt_ops/kda/triton/kda/math.py`、`utils.py` |
| 测试 / 性能 | `test/test_kda.py`（`--bench`） |

范围：KDA 目录直接定义的 Triton Kernel 与直接 launch wrapper；不含依赖 vLLM
公共 FLA Kernel 的完整 `chunk_kda` orchestration，不含 backward。

## 附录 B. 溯源

- vLLM PR：[`#50000`](https://github.com/vllm-project/vllm/pull/50000)
- 固定来源 commit：[`5a3eba03481d879451d89b238f9160fa1c1faddf`](https://github.com/vllm-project/vllm/commit/5a3eba03481d879451d89b238f9160fa1c1faddf)
- 来源目录：[`vllm/models/kimi_k3/amd/ops/third_party/kda`](https://github.com/vllm-project/vllm/tree/5a3eba03481d879451d89b238f9160fa1c1faddf/vllm/models/kimi_k3/amd/ops/third_party/kda)
- 正确性参考：[`tests/models/kimi_k3/test_kda.py`](https://github.com/vllm-project/vllm/blob/5a3eba03481d879451d89b238f9160fa1c1faddf/tests/models/kimi_k3/test_kda.py)

## 附录 C. 实现与同步备注

- Prefill intra 默认采用 token-parallel diagonal block Kernel，再由
  `chunk_kda_fwd_kernel_inter_solve_fused` 完成跨 sub-chunk 项与三角求解。
- `safe_gate=True` 时使用 `chunk_kda_fwd_kernel_intra_sub_chunk` 路径。
- Decode state 采用 `[V,K]` 内存布局，数学参考通常采用 `[K,V]`；对照时需转置。
- 保留函数已按 AST 与固定来源 commit 核对；除修正两个与实际返回值不一致的返回
  注解外，Kernel decorator、参数签名、函数体及 launch 参数无本地修改。
- 后续同步只在 import、`math.py`、`utils.py` 处理去 vLLM 依赖；不要在移植文件内
  做无关格式化、重排或本地优化。

## 附录 D. Kernel 与 wrapper 对照

| Kernel | Wrapper |
|---|---|
| `recompute_w_u_fwd_kernel` | `recompute_w_u_fwd` |
| `chunk_gla_fwd_kernel_o` | `chunk_gla_fwd_o_gk` |
| `kda_gate_chunk_cumsum_vector_kernel` | `fused_kda_gate_chunk_cumsum` |
| `kda_gate_fwd_kernel` | `fused_kda_gate` |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `chunk_kda_fwd_intra` |
| `chunk_kda_fwd_kernel_intra_sub_chunk` | `chunk_kda_fwd_intra` |
| `chunk_kda_fwd_kernel_intra_token_parallel` | `chunk_kda_fwd_intra_token_parallel` |
| `_kda_gate_beta_fwd_kernel` | `_fused_kda_gate_beta` / `fused_recurrent_kda` |
| `fused_recurrent_kda_fwd_kernel` | `fused_recurrent_kda_fwd` |
| `fused_recurrent_kda_packed_decode_kernel` | `fused_recurrent_kda_packed_decode` |
