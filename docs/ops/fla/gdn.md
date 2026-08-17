# FLA/GDN 接口使用说明

Gated DeltaNet（GDN）提供 Decode/Recurrent 与 Chunk 前向接口。

## 1. 支持范围

公开接口统一从 `boltops.fla.gdn` 延迟导入：

```python
from boltops.fla.gdn import (
    chunk_fwd_o,
    chunk_gated_delta_rule_fwd_h,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_recurrent,
)
```

| 接口 | 主要用途 | 返回值 |
|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode` | 单 token packed QKV decode | `output, updated_state` |
| `fused_sigmoid_gating_delta_rule_update` | continuous batching 和线性 speculative decode | `output, final_state` |
| `fused_sigmoid_gating_delta_rule_update_recurrent` | target-verify、tree cache 和中间状态缓存 | `output` |
| `chunk_gated_delta_rule_fwd_h` | Chunk state propagation | `h, v_new, final_state` |
| `chunk_fwd_o` | Chunk state项和局部 attention项合并 | `output` |

当前范围均为 forward-only，不包含 backward。

## 2. 数学语义和布局

以数学布局 `S ∈ R^[HV,K,V]` 表示 recurrent state，单 token GDN更新为：

```text
g_t     = -exp(A_log) * softplus(a_t + dt_bias)
beta_t  = sigmoid(b_t)
S_decay = exp(g_t)[..., None, None] * S_(t-1)
delta_v = v_t - k_t @ S_decay
S_t     = S_decay + outer(k_t, beta_t * delta_v)
o_t     = scale * q_t @ S_t
```

开启 `use_qk_l2norm_in_kernel=True` 时，q、k 在计算过程中按最后一维归一化。
当 `HV > H` 时，多个 value head 共享同一个 q/k head，要求 `HV % H == 0`。

State物理布局分为两类：

- packed decode、普通 sigmoid和 recurrent target-verify接口固定为
  `[cache_slots, HV, V, K]`
- chunk接口通过 `transpose_state_layout` 选择：
  - `True`：`[cache_slots, H, V, K]`
  - `False`：`[cache_slots, H, K, V]`

## 3. Decode/Recurrent 接口

### 3.1 Packed 单 token decode

```python
def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
    kernel_cfg=None,
):
    ...
```

主要 shape：

| 参数 | Shape | 常见 dtype | 说明 |
|---|---|---|---|
| `mixed_qkv` | `[B, 2*H*K + HV*V]` | FP16/BF16 | packed Q/K/V |
| `a` / `b` | `[B,HV]` | FP16/BF16 | gate和 beta logits |
| `A_log` / `dt_bias` | `[HV]` | FP32/输入 dtype | gate参数 |
| `initial_state` | `[slots,HV,V,K]` | FP32 | 原地更新的 cache |
| `out` | `[B,1,HV,V]` | 同输入 | 调用方提供的输出 buffer |
| `ssm_state_indices` | `[B]` | int32/int64 | cache slot；负数表示 padding |

调用示例：

```python
B, H, HV, K, V = 8, 4, 16, 128, 128
dtype, device = torch.bfloat16, "cuda"

mixed_qkv = torch.randn(B, 2 * H * K + HV * V, device=device, dtype=dtype)
a = torch.randn(B, HV, device=device, dtype=dtype)
b = torch.randn(B, HV, device=device, dtype=dtype)
A_log = torch.randn(HV, device=device, dtype=torch.float32)
dt_bias = torch.randn(HV, device=device, dtype=dtype)
state = torch.zeros(B, HV, V, K, device=device, dtype=torch.float32)
indices = torch.arange(B, device=device, dtype=torch.int32)
out = torch.empty(B, 1, HV, V, device=device, dtype=dtype)

output, state = fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    K**-0.5,
    state,
    out,
    indices,
    use_qk_l2norm_in_kernel=True,
)
```

`output is out`，返回的 state与输入 `initial_state` 是同一 tensor。负索引对应的
输出为零，且不会更新 state。

### 3.2 普通 sigmoid update

```python
def fused_sigmoid_gating_delta_rule_update(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    beta=1.0,
    threshold=20.0,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
    is_kda=False,
    kernel_cfg=None,
):
    ...
```

输入 shape：

- `q/k`：`[B,T,H,K]`
- `v`：`[B,T,HV,V]`
- GDN `a`：`[B,T,HV]`；`dt_bias`：`[HV]`
- `b`：`[B,T,HV]`
- `initial_state`：`[slots,HV,V,K]`
- varlen/continuous batching时 `B=1`，`cu_seqlens=[N+1]`

线性 speculative decode中：

- `ssm_state_indices` 为 `[N,max_steps]`
- `num_accepted_tokens` 为 `[N]`
- 每个 sequence从最后一个 accepted token对应的 cache slot读取初始 state
- `inplace_final_state=True` 时，每一步按对应 state index写入 cache
- `inplace_final_state=False` 时不修改输入 cache，返回按扁平 token排列的
  `[T,HV,V,K]` final states

```python
output, final_state = fused_sigmoid_gating_delta_rule_update(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    initial_state=state,
    inplace_final_state=True,
    cu_seqlens=cu_seqlens,
    ssm_state_indices=state_indices,
    num_accepted_tokens=num_accepted_tokens,
    use_qk_l2norm_in_kernel=True,
)
```

`is_kda=True` 时，`a` 改为每 key维 gate，shape为 `[B,T,HV,K]`，
`dt_bias` 含 `HV*K` 个元素。

### 3.3 Target-verify/tree recurrent update

```python
def fused_sigmoid_gating_delta_rule_update_recurrent(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    is_kda=False,
    disable_state_update=False,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
    kernel_cfg=None,
):
    ...
```

该接口与普通 speculative接口保持独立，用于 target-verify协议：

- `initial_state_indices[N]` 选择每条 sequence的正式 state
- `intermediate_states_buffer` shape为
  `[intermediate_slots,cache_steps,HV,V,K]`
- `intermediate_state_indices[N]` 选择每条 sequence的中间状态 cache行
- `retrieve_parent_token[N,steps]` 给出 tree中每个 token的 parent step
- `disable_state_update=True` 时只写 intermediate cache，不回写正式 state
- 返回值只有 output；正式 state和中间状态通过传入 tensor体现副作用

```python
output = fused_sigmoid_gating_delta_rule_update_recurrent(
    A_log,
    a,
    dt_bias,
    1.0,
    20.0,
    q,
    k,
    v,
    b,
    state,
    state_indices,
    cu_seqlens=cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=intermediate_states,
    intermediate_state_indices=intermediate_indices,
    cache_steps=verify_steps,
    retrieve_parent_token=parent_steps,
)
```

## 4. Chunk 接口

Chunk路径假定调用方已经生成 WY表示中的 `w/u` 以及 chunk-local累计 gate。
两个公开接口可以单独调用，也可以串联：

```text
k / w / u / gate / initial state
    -> chunk_gated_delta_rule_fwd_h
    -> h / v_new / final state

q / k / v_new / h / gate
    -> chunk_fwd_o
    -> output
```

### 4.1 Chunk state propagation

```python
def chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    initial_state_indices=None,
    output_final_state=True,
    inplace_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=True,
    kernel_cfg=None,
):
    ...
```

主要 shape：

| 参数/返回值 | Shape |
|---|---|
| `k` | `[B,T,Hg,K]` |
| `w` | `[B,T,H,K]` |
| `u` / `v_new` | `[B,T,H,V]` |
| `g` | `[B,T,H]`，scalar累计 gate |
| `gk` | `[B,T,H,K]`，per-key累计 gate |
| `initial_state_indices` | `[N]` |
| fixed-length `h` | `[B,ceil(T/BT),H,...state_layout]` |
| varlen `h` | `[1,total_chunks,H,...state_layout]` |

Final state行为：

| 参数组合 | 行为 |
|---|---|
| `output_final_state=False, inplace_final_state=False` | 第三项返回 `None`，不修改输入 state |
| `output_final_state=True, inplace_final_state=False` | 返回 `[N,H,...]` 紧凑 final state，不修改输入 cache |
| `output_final_state=True, inplace_final_state=True` | 按 indices原地更新输入 cache，并返回同一 tensor |

原地模式要求同一次 launch中的有效 `initial_state_indices` 唯一；重复有效索引会让
多个 program并发写同一 cache row，结果不确定。负数或越界索引按零 initial state
计算；原地模式跳过写回，非原地模式仍生成对应的紧凑 final state。

### 4.2 Chunk output

```python
def chunk_fwd_o(
    q,
    k,
    v,
    h,
    g=None,
    g_gamma=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=False,
    kernel_cfg=None,
):
    ...
```

`transpose_state_layout` 必须与生成 `h` 时使用的布局一致。`g_gamma[H]` 可额外
施加按 chunk内 token位置递增的 gate。

完整调用示例：

```python
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=g_cumsum,
    initial_state=state,
    initial_state_indices=state_indices,
    output_final_state=True,
    inplace_final_state=False,
    chunk_size=64,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    transpose_state_layout=True,
)

output = chunk_fwd_o(
    q,
    k,
    v_new,
    h,
    g=g_cumsum,
    scale=k.shape[-1] ** -0.5,
    cu_seqlens=cu_seqlens,
    chunk_size=64,
    chunk_indices=chunk_indices,
    transpose_state_layout=True,
)
```

使用 `use_exp2=False` 时，`g/gk/g_gamma` 位于自然指数空间；使用
`use_exp2=True` 时必须提供 log2空间的累计 gate。varlen模式要求 `B=1`，每条
sequence在自己的 chunk边界重新累计 gate。

## 5. 使用约束

- 所有计算 tensor必须位于同一 CUDA/HIP device
- q/k/v/w/u及 gate应采用标准连续布局；packed QKV允许 token stride大于行宽
- state dtype由上层 cache配置决定：vLLM的 GDN `mamba_ssm_cache_dtype="auto"`
  默认跟随模型 dtype，BF16模型使用 BF16 state；SGLang的 target-verify temporal
  state默认使用 FP32
- chunk默认 `chunk_size=64`，当前接口要求 `K <= 256`
- `chunk_indices` 可由调用方预先生成并复用；其每一行是
  `(sequence_index, local_chunk_index)`

## 6. 正确性测试与性能基准

测试位于 `test/fla`，正确性和性能 case保存在相同文件中。默认运行会跳过标记为
`bench` 的性能 case；传入 `--bench` 后同时执行性能测试。

在 BoltOPs 仓库根目录执行：

```bash
# 全部 GDN 正确性测试
PYTHONPATH=. TILELANG_CACHE_DIR=./cache pytest -q test/fla

# 正确性 + 性能；输出 latency、GB/s 或 TFLOPS
PYTHONPATH=. TILELANG_CACHE_DIR=./cache pytest -q -s test/fla --bench

# 自定义性能 warmup/repeat
PYTHONPATH=. TILELANG_CACHE_DIR=./cache pytest -q -s test/fla \
  --bench --bench-warmup=25 --bench-repeat=100
```

按接口运行：

```bash
# Packed decode
pytest -q test/fla/test_gdn_packed_decode.py

# 普通 sigmoid和 recurrent/target-verify/tree cache
pytest -q test/fla/test_gdn_sigmoid_gating.py

# Chunk h/o
pytest -q test/fla/test_gdn_chunk.py
```

性能输出规则：

- packed decode、普通 sigmoid、target-verify主要受 state/cache流量限制，报告
  latency和逻辑有效带宽 `GB/s`
- chunk h、chunk o及二者端到端为矩阵计算密集型，报告 latency和 `TFLOPS`
- 性能测试只测 Bolt当前实现，不运行 reference、不计算 speedup，也不设置固定性能阈值

## 7. 性能数据

下表为当前 Triton实现的实测数据。带宽型接口使用 `GB/s`，计算密集型接口使用
`TFLOPS`。默认计时参数：

```text
gfx936: HIP_VISIBLE_DEVICES=5
gfx938: HIP_VISIBLE_DEVICES=7
warmup=10
repeat=50
```

### `chunk_gated_delta_rule_fwd_h`

| Shape | gfx936 | gfx938 |
|---|---:|---:|
| `B=1, T=13320, H=16, Hg=8, K=128, V=128, BT=64, FP16, fixed` | 0.604 ms / 23.12 TFLOPS | 1.579 ms / 8.85 TFLOPS |
| `B=1, lengths=[4097,4911,4312], H=16, Hg=8, K=128, V=128, BT=64, FP16, varlen` | 0.511 ms / 27.32 TFLOPS | 1.068 ms / 13.08 TFLOPS |

### `chunk_fwd_o`

| Shape | gfx936 | gfx938 |
|---|---:|---:|
| `B=1, T=13320, H=16, Hg=8, K=128, V=128, BT=64, FP16, fixed` | 0.607 ms / 23.00 TFLOPS | 0.363 ms / 38.45 TFLOPS |
| `B=1, lengths=[4097,4911,4312], H=16, Hg=8, K=128, V=128, BT=64, FP16, varlen` | 0.626 ms / 22.30 TFLOPS | 0.384 ms / 36.32 TFLOPS |

### `fused_recurrent_gated_delta_rule_packed_decode`

| Shape | gfx936 | gfx938 |
|---|---:|---:|
| `B=1, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 13.07 us / 161.25 GB/s | 12.38 us / 170.30 GB/s |
| `B=8, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 22.06 us / 764.41 GB/s | 22.17 us / 760.63 GB/s |
| `B=32, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 63.23 us / 1066.59 GB/s | 57.36 us / 1175.75 GB/s |
| `B=64, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 121.70 us / 1108.24 GB/s | 96.28 us / 1400.84 GB/s |

### `fused_sigmoid_gating_delta_rule_update`

| Shape | gfx936 | gfx938 |
|---|---:|---:|
| `N=4, T=1, H=4, HV=16, K=128, V=128, BF16, state=BF16` | 18.56 us / 228.18 GB/s | 18.88 us / 224.37 GB/s |
| `N=4, T=4, H=4, HV=16, K=128, V=128, BF16, state=BF16` | 32.53 us / 327.46 GB/s | 40.92 us / 260.25 GB/s |

### `fused_sigmoid_gating_delta_rule_update_recurrent`

| Shape | gfx936 | gfx938 |
|---|---:|---:|
| `N=1, T=4, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 23.30 us / 226.75 GB/s | 27.41 us / 192.79 GB/s |
| `N=8, T=4, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 51.39 us / 822.53 GB/s | 56.29 us / 750.94 GB/s |
| `N=32, T=4, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 162.28 us / 1041.98 GB/s | 147.41 us / 1147.11 GB/s |
| `N=48, T=4, H=4, HV=16, K=128, V=128, BF16, state=FP32` | 240.78 us / 1053.41 GB/s | 204.85 us / 1238.14 GB/s |
