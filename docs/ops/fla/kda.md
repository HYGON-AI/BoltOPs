# FLA/KDA Triton Kernel 使用说明

本文是 KDA 算子库的对外总览，只描述公共 API、数学语义、输入输出、fixed/varlen
约束和 wrapper 使用方式。每个底层算子的实现、配置、fallback 及性能证据见独立
文档；不要把本页的 API 约束与某个硬件的静态调优配置混在一起。

## 1. 算子用途与数学语义

Kernelized Delta Attention（KDA）用 recurrent state 表示线性注意力。数学布局中，
每个 head 的 state 为 `S_t ∈ R^[K,V]`：

```text
q_hat_t = l2norm(q_t) * scale
k_hat_t = l2norm(k_t)

S_decay = exp(g_t)[..., None] * S_(t-1)
delta_v = v_t - k_hat_t @ S_decay
S_t     = S_decay + outer(beta_t * k_hat_t, delta_v)
o_t     = q_hat_t @ S_t
```

其中：

- `g_t` 是按 key 维度作用的负衰减 gate；KDA 的累计 gate 使用 log2 空间；
- `beta_t = sigmoid(raw_beta_t)`；
- state 的物理存储布局是 `[cache_slots, H, V, K]`；
- `lower_bound` 非空时使用有界 sigmoid gate，否则使用 softplus gate；
- 本仓库只提供 forward，不提供 backward。

Prefill 按 `BT=64` 分块，数据流为：

```text
raw gate/beta
  -> gate activation + beta sigmoid + chunk-local cumsum
  -> intra-chunk Aqk/Akk + triangular solve
  -> WY w/u/kg reconstruction
  -> chunk state propagation
  -> chunk output
```

Decode 使用 recurrent 路径逐 token 或逐 speculative batch 更新同一个 FP32 state。

## 2. 公共接口

所有公开入口均可从 `boltops.fla.kda` 导入：

```python
from boltops.fla.kda import (
    chunk_kda,
    chunk_kda_fwd,
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_with_fused_gate,
    chunk_kda_with_fused_gate_fwd,
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_state_output_fused,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    recompute_w_u_fwd,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)
```

| 接口 | 作用 | 独立文档 |
|---|---|---|
| `fused_kda_gate` / `fused_kda_gate_chunk_cumsum` | Gate 与 chunk-local cumsum 阶段 | [kda_prefill_gate.md](./kda/kda_prefill_gate.md) |
| `chunk_kda_fwd_intra` | Intra-chunk 系数生成阶段 | [kda/kda_prefill_intra.md](./kda/kda_prefill_intra.md) |
| `chunk_kda_fwd_intra_token_parallel` | Intra 阶段 token-parallel 子路径 | [kda/kda_prefill_intra.md](./kda/kda_prefill_intra.md) |
| `recompute_w_u_fwd` | WY `w/u/kg` 重建阶段 | [kda/kda_prefill_wy.md](./kda/kda_prefill_wy.md) |
| `chunk_gated_delta_rule_fwd_h` | standalone State propagation wrapper | [kda/kda_prefill_state.md](./kda/kda_prefill_state.md) |
| `chunk_gla_fwd_o_gk` | Chunk output 阶段 | [kda/kda_prefill_output.md](./kda/kda_prefill_output.md) |
| `chunk_kda_fwd_state_output_fused` | State/output 跨阶段融合 | [kda/kda_prefill_state.md](./kda/kda_prefill_state.md) |
| `fused_recurrent_kda_fwd` / `fused_recurrent_kda` | Dense recurrent decode 阶段 | [kda/kda_decode.md](./kda/kda_decode.md) |
| `fused_recurrent_kda_packed_decode` | Packed decode 阶段 | [kda/kda_decode.md](./kda/kda_decode.md) |
| `chunk_kda*` | 完整 Prefill orchestration | [kda/kda_orchestration.md](./kda/kda_orchestration.md) |

Q/K normalization 是 intra 与 decode 阶段共用的预处理，分别在对应阶段文档说明。
每个算子的 XCU、HBM、VGPR、配置和性能结论直接写在对应的独立算子文档中。

## 3. 输入输出与布局

### 3.1 Prefill 输入

普通 Prefill 的主张量为：

| 张量 | Shape | 常见 dtype | 说明 |
|---|---|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16/FP16 | query/key；KDA 为同头数 |
| `v` | `[B,T,H,V]` | BF16/FP16 | value |
| `g`/`gk` | `[B,T,H,K]` | FP32/BF16 | log2 gate；具体入口见算子文档 |
| `beta` | `[B,T,H]` | FP32/BF16 | 已 sigmoid 的更新系数 |
| `A`/`Aqk`/`Akk` | `[B,T,H,BT]` | BF16/FP32 | chunk 内系数矩阵 |
| `initial_state` | `[slots,H,V,K]` | FP32 | 可选输入 state，通常原地更新 |
| `cu_seqlens` | `[N+1]` | int32/int64 | varlen 序列边界 |
| `chunk_indices` | `[NT,2]` | int32/int64 | 可选 `(sequence, local_chunk)` 映射 |

Prefill output 通常为 `o:[B,T,H,V]`；如果请求 final state，还返回
`final_state`，具体 shape 以 wrapper 文档为准。

### 3.2 Decode 输入输出

`fused_recurrent_kda` 使用分开的 `q/k/v/raw_g/raw_beta`；
`fused_recurrent_kda_fwd` 使用已处理的 `g/beta`，也支持直接把 raw gate/beta
传入并设置 `use_gate_in_kernel=True`；`fused_recurrent_kda_packed_decode` 使用
packed QKV：

| 张量 | Shape | 说明 |
|---|---|---|
| `mixed_qkv` | `[tokens, 2*H*K + H*V]` | packed Q/K/V |
| `raw_g` | `[1,tokens,H,K]` | raw key-wise gate |
| `raw_beta` | `[1,tokens,H]` | raw beta |
| `initial_state` | `[cache_slots,H,V,K]` | FP32 cache |
| `state_indices` | `[tokens]` | 每个 token 的 cache slot；无效 slot 不更新 |
| 返回 `output` | `[1,tokens,H,V]` | 与输入 value dtype 相同 |

返回的 state 与输入 `initial_state` 是同一个原地更新的 tensor；调用方必须保证
并发 token 不写同一个有效 state slot。

## 4. fixed 与 varlen

fixed 和 varlen 是两种索引 ABI，不是两种数学算法。

### fixed

- `cu_seqlens=None`；
- 每个 batch row 长度均为 `T`；
- kernel 可用规则的 `B*T`、chunk id 和固定 stride 直接寻址；
- state/output 的 chunk 映射不需要额外 `chunk_indices`。

### varlen

- 提供 `cu_seqlens=[0, L_0, L_0+L_1, ...]`；
- 每个 sequence 的有效范围为 `[cu_seqlens[i], cu_seqlens[i+1])`；
- chunk 不跨 sequence，partial chunk 必须按本 sequence 的边界处理；
- 需要时提供 `chunk_indices:[NT,2]`，将全局 chunk 映射到
  `(sequence, local_chunk)`；
- 当前 Prefill wrapper 的 packed varlen 入口要求 `B=1`，由 `cu_seqlens` 表示
  多条 packed sequence；具体 recurrent decode 的 state index 约束见对应文档。

示例：

```python
# fixed: 一条连续 batch layout
o = chunk_kda(q, k, v, g, beta, cu_seqlens=None)

# varlen: B=1 的 packed layout，包含三条不同长度 sequence
cu_seqlens = torch.tensor([0, 3000, 7000, 8192], device=q.device)
o = chunk_kda(q, k, v, g, beta, cu_seqlens=cu_seqlens)
```

fixed/varlen 均要求相应 tensor contiguous、位于同一 CUDA/HIP device，且 state
内层 stride 为 `(V*K, K, 1)`。不同算子的额外 dtype、head、chunk 和 final-state
限制，以各自独立文档为准。

## 5. 完整 wrapper 使用方式

输入已经完成 gate/beta 处理时，使用 materialized-gate orchestration：

```python
o, final_state = chunk_kda_fwd(
    q, k, v, g, beta,
    scale=K**-0.5,
    initial_state=initial_state,
    output_final_state=False,
    cu_seqlens=cu_seqlens,
)
```

上游仍持有 raw gate/beta 时，使用 fused-gate orchestration：

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

需要单独调试或复用中间 tensor 时，按数据流依次调用 gate/cumsum、intra、WY、state
和 output wrapper；这些入口都保留 standalone 能力，融合 fast path 只由 wrapper
在满足条件时静态选择，不改变单独调用协议。

## 6. 语义边界

- KDA orchestration 本身只有一个 `H`，不存在 GDN 的 `Hg` 分组 key 语义；
- 为兼容 vLLM 上游，standalone state wrapper 仍保留 `Hg != H` 的协议，详见
  [kda/kda_prefill_state.md](./kda/kda_prefill_state.md)；
- KDA 不接受 `initial_state_indices`，不要将 GDN 的该参数传入 KDA API；
- `output_final_state=True` 不再关闭 state/output fusion：融合 kernel 自己写出
  final state，与 split pipeline 的归约语义一致；fusion 现在只按 shape 选择
  （见 [kda/kda_prefill_state.md](./kda/kda_prefill_state.md)）；
- 所有优化路径均有 shape/dtype/layout fallback，不能将某个 BW1000_H 配置当作
  通用硬件上限。

## 7. 正确性测试

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_*.py
```

测试职责、算子、源码文件和性能证据见各自独立算子文档。

## 8. 为什么 `chunk_size` 固定为 64

`FLA_CHUNK_SIZE = 64`（`boltops/fla/common/triton/utils.py`）。这不是历史包袱，
是两条反向曲线的交点：

* intra 的工作量 ∝ `T x BT`（`Aqk`/`Akk` 是 `[B,T,H,BT]`），**BT 越小越便宜**。
* state 递推的工作量 ∝ `T/BT x K x V`——每个 chunk 都要更新一次 `[V,K]` 的
  h 累加器，代价与 BT 基本无关，所以 **BT 越小越贵**，且是线性变贵。

state 阶段是最大单项开销，于是它主导了结论。BW1100 `T=16384,K=V=128`，用
BT-agnostic 的通用 state 路径（`chunk_gated_delta_rule_fwd_h` + `chunk_gla_fwd_o_gk`）
实测：

| BT | `H=64` | 相对 BT=64 | `H=32` | 相对 BT=64 |
|---|---:|---:|---:|---:|
| 64 | 5.298 ms | 1.00x | 2.650 ms | 1.00x |
| 32 | 6.978 ms | 1.32x | 3.876 ms | 1.46x |
| 16 | 13.241 ms | 2.50x | 8.226 ms | 3.10x |

按 `T=16384,H=64` 的实测分阶段成本折算到 BT=16：intra 最多从 2.47 降到
~1.4 ms（流量只减 15%，因为 q/k/g 的读取量不变，省的只是 `Aqk`/`Akk`），
省 ~1.1 ms；而 state 从 2.90 涨到 ~7.2 ms，多花 ~4.3 ms。**净亏 ~3.2 ms。**
BT=32 同样是净亏（约 +0.6 ms）。

反方向也走不通：BT=128 在本机 LDS 上直接 `OutOfResources`
（需要 81920 B，硬件上限 65536 B）。所以 64 既是最优点也是上限。

`chunk_kda_fwd_intra` 现在对 `chunk_size != 64` 显式报错。此前
`chunk_kda_fwd_kernel_inter_solve_fused` 会在 BT=16 时把 sub-chunk 原点算到
后三个 chunk 上并相互覆写——越界列偏移被 `boundary_check` 裁掉，所以既不报错
也不崩，只是结果错。

## 9. Launch 开销与 `TRITON_FAST_JIT`

Prefill 每次前向下发 8 个 Triton kernel。短序列下真正的瓶颈不是 GPU，而是
host 侧的下发开销：BW1100 上 `T=512, H=16, K=V=128` 的 device 时间约 0.25 ms，
而 `JITFunction.run` 要花 0.72 ms，其中绝大部分是 Triton 每次 launch 的参数
specialization（每次前向调用 45 次 `get_tensor_specialization`）。

Hygon 后端提供了 Cython binder + saved-kernel 磁盘缓存来绕过这一步，由
`TRITON_FAST_JIT=1` 开启。该变量在 `import triton` 时把 `triton.jit` 整体替换
成快速版本，因此本仓库的 kernel 只要写普通的 `@triton.jit` 就会自动受益，不需要
任何额外开关；注意它必须在 `import triton` **之前**设置。

实测（BW1100，`T=512, H=16, K=V=128`，单次前向 host 下发时间）：

| | enqueue |
|---|---|
| 未设置 | 0.720 ms |
| `TRITON_FAST_JIT=1` | 0.434 ms（-40%） |

首次运行会把 kernel 写入 `$TRITON_CACHE_DIR/saved_kernel/`；缓存未命中时自动
回退到常规 launch 路径，结果不变。

## 附录 A. 源码入口

| 层级 | 文件 |
|---|---|
| Public wrapper | `boltops/fla/kda/__init__.py` |
| Triton export | `boltops/fla/kda/triton/__init__.py` |
| Gate/output/recompute/orchestration | `boltops/fla/kda/triton/chunk.py` |
| Intra solve | `boltops/fla/kda/triton/chunk_intra.py` |
| Token-parallel intra | `boltops/fla/kda/triton/chunk_intra_token_parallel.py` |
| State/fusion | `boltops/fla/kda/triton/chunk_state.py` |
| Recurrent decode | `boltops/fla/kda/triton/fused_recurrent.py` |
| Lean `heuristics` | `boltops/fla/common/triton/heuristics.py` |
| Correctness tests | `test/fla/test_kda_*.py` |

## 附录 B. 溯源

- vLLM Kimi-K3 AMD KDA：`vllm/models/kimi_k3/amd/ops/third_party/kda`；
- 本仓库保留 wrapper 的独立调用能力，并将 vLLM 依赖替换为本地兼容 helper；
- GDN 与 KDA 的同名 WY/state 算子不能混用，GDN 对照说明见
  [recompute_w_u_fwd.md](./recompute_w_u_fwd.md)。
