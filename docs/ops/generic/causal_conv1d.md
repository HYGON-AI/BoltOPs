# `causal_conv1d` 使用说明（统一入口 + SGLang / vLLM 双 port）

Mamba / SSM 的短卷积核：对每个特征通道做 `width` 抽头的因果 depthwise conv1d，
可选 SiLU 激活，并在前向过程中滑动更新每序列的卷积历史 state。本算子提供
**统一入口** `causal_conv1d_fn` / `causal_conv1d_update`（`backend="auto"`
自动路由），底层是 SGLang 与 vLLM 两套 standalone Triton port：

- SGLang port：`triton/causal_conv1d/sglang.py`（extend / prefill 的 varlen
  连续批处理 + decode，含 spec-decode / Eagle tree）；
- vLLM port：`triton/causal_conv1d/vllm.py`（V1 连续批处理紧凑调度 + 前缀缓存
  APC + varlen decode）。

集成策略：**kernel 层保留两套可 diff 的移植，接口层只保留一个统一入口，共享
基础放在公共模块；两套 port 各自持续跟进上游，不做合并决策**。

## 1. 算子用途与数学语义

设核宽 `width`（SGLang port 支持 2～5，vLLM fwd 支持 2～5、update 支持 2～6，
K3 用 4），`state_len = width - 1`。对每条序列、每个特征通道 `c`，把长度为
`state_len` 的历史 token 拼到当前 token 之前：

```text
x̃ = cat([history, x_seq])            # history 长度 state_len

y[t] = act( bias[c] + sum_{i=0}^{width-1} x̃[t - (state_len - i)] * w[c, i] )
```

等价于 `F.conv1d(x̃, weight.unsqueeze(1), bias, groups=dim)` 取最后 `seqlen` 个
token；`w[c, :]` 是该通道独立的 `width` 抽头卷积核（depthwise、逐通道）。

- 历史来源：`conv_states[cache_indices[b]]` 前 `state_len` 个 token。`fwd` 中仅当
  `has_initial_state[b]=True` 时作为初始历史，否则补零；`update` 总是使用当前 state。
- 每次前向结束后 state 滑动更新：`conv_states[...][:, :state_len] =
  x̃[:, -state_len:]`（保留最近 `state_len` 个 token，shift-left 并写入新 token），
  **inplace** 修改。
- `activation`：`"silu"` / `"swish"` / `True` 应用 SiLU；`None` 不激活。
- 统一入口的 `fn` / `update` 都返回新建的 out 张量（不原地改写 `x`，`update`
  也可通过 `out` 预分配输出），shape / dtype 与 `x` 一致。
- 算子为 forward-only，不包含 backward。

### 1.1 fwd 与 update 的差异

| 维度 | `causal_conv1d_fn` | `causal_conv1d_update` |
| --- | --- | --- |
| 用途 | extend / prefill，一次性处理整个 varlen batch | decode，每次前向一步或多步（draft 验证） |
| `x` 布局 | `[dim, cu_seqlen]`，channel-last | `[batch, dim]` 或 `[batch, dim, seqlen]`；vLLM 另支持 `[num_tokens, dim]` varlen |
| 序列切分 | `query_start_loc`（SGLang 另需 `seq_lens_cpu`） | 每行一条序列；vLLM varlen 用 `query_start_loc` |
| state 更新 | 每个 program 的 chunk 0 写回；非首 chunk 从前一 chunk 尾部取历史 | 滑动窗口 shift 后写回 |
| 附加能力 | vLLM：APC 前缀缓存、V1 紧凑调度、PDL | SGLang：spec-decode（`num_accept_tokens` 等）、Eagle tree mask；vLLM：varlen decode、APC 写回 |

### 1.2 SGLang port 与 vLLM port 的差异

数学语义相同，差异全部在**调度方式与框架特性**上：

| 维度 | SGLang port | vLLM port |
| --- | --- | --- |
| fwd 调度 | 3D grid `(batch, chunk, feat)`；`seq_lens_cpu` 决定 chunk 数，`cache_indices` 为 `pad_slot_id` 的 program 提前返回 | 紧凑 1D 调度：一个 program = 一个 `(seq, chunk)`，由 `batch_ptr` + `token_chunk_offset_ptr` 给出；padded program 用 `PAD_SLOT_ID` 填充并跳过；无 `metadata` 时 host 端由 `query_start_loc` 构造（上限 1024 个 program） |
| 前缀缓存（APC） | 无 | 有：`block_idx_first_scheduled_token` / `block_idx_last_scheduled_token` / `initial_state_idx` / `num_computed_tokens`，state 按 `block_size_to_align` 对齐写回 cache block |
| null block | 无（只有 `pad_slot_id`） | 有：`null_block_id`（vLLM 约定 block 0 为 null block） |
| spec-decode | `num_accept_tokens` 滑动窗口 + `intermediate_conv_window` 逐 step 保存 | `num_accepted_tokens` 滑动窗口（不保存 intermediate） |
| Eagle tree | `retrieve_next_token` / `retrieve_next_sibling` / `retrieve_parent_token` | 无 |
| varlen decode | 无（每行一条序列） | 有：`query_start_loc` + `max_query_len`，`x` 可为 `[num_tokens, dim]` |
| PDL / GDC | `USE_GDC` constexpr（本平台 no-op） | `launch_pdl` / GDC（本平台 no-op） |
| `KERNEL_WIDTH` | fwd / update 均 2～5 | fwd 2～5；update 2～6 |
| out 语义 | 总是新建 out | 默认新建（`out` 可预分配） |
| 依赖输入 | `cache_indices` / `has_initial_state` / `conv_state_indices` 可选（identity / 全 False） | kernel 无条件加载这些张量，需显式传入或由统一入口自动补齐 |

## 2. 集成结构与统一入口

### 2.1 目录结构与职责

| 文件 | 职责 | 可 diff 上游 |
| --- | --- | --- |
| `triton/causal_conv1d/__init__.py` | 子包公共导出（统一入口） | — |
| `triton/causal_conv1d/common.py` | 常量（`PAD_SLOT_ID` / `NULL_BLOCK_ID`）、PDL stub、backend 路由、torch 参考实现 | — |
| `triton/causal_conv1d/api.py` | 统一入口 `causal_conv1d_fn` / `causal_conv1d_update`，`backend="auto"` 路由 | — |
| `triton/causal_conv1d/sglang.py` | SGLang port（kernel + wrapper 逐字保留，仅常量改从 common 导入） | SGLang `causal_conv1d_triton.py` |
| `triton/causal_conv1d/vllm.py` | vLLM port（同上） | vLLM `mamba/ops/causal_conv1d.py` |
| `boltops/generic/__init__.py` | 公开 API：统一入口 | — |

两个 kernel 文件只改动了一处：常量和 `is_arch_support_pdl()` 从
`causal_conv1d/common.py` 导入，其余（Triton kernel 与 wrapper 逻辑）保持与上游
逐字一致，方便继续用 diff 跟进上游更新。

### 2.2 统一入口与 backend 自动路由

```python
from boltops.generic import causal_conv1d_fn, causal_conv1d_update
```

两个入口都带 `backend` 参数：`"auto"`（默认）/ `"sglang"` / `"vllm"`。
`auto` 按实际传入的参数路由：

| 场景（fwd） | 路由 |
| --- | --- |
| 传了 `seq_lens_cpu`（SGLang 的 per-sequence 长度列表） | `sglang` |
| 传了 `metadata` / `block_idx_first_scheduled_token` / `block_idx_last_scheduled_token` / `initial_state_idx` / `num_computed_tokens` / `block_size_to_align` | `vllm` |
| 同时出现两类信号 | 抛 `ValueError`，要求显式指定 `backend` |
| 都没有 | `vllm`（默认基座） |

| 场景（update） | 路由 |
| --- | --- |
| 传了 `cache_seqlens` / `num_accept_tokens` / `intermediate_*` / `retrieve_*`（spec-decode / Eagle tree） | `sglang` |
| 传了 `num_accepted_tokens` / `query_start_loc` / `max_query_len` / `block_idx_last_scheduled_token` / `initial_state_idx` / `null_block_id` / `out` | `vllm` |
| 同时出现两类信号 | 抛 `ValueError` |
| 都没有 | `vllm`（默认基座） |

`num_accept_tokens`（SGLang 拼写）与 `num_accepted_tokens`（vLLM 拼写）语义相同，
统一入口两种拼写都接受。

### 2.3 行为契约（统一入口与 raw port 的差异）

两个后端通过统一入口时行为一致，**与 vLLM raw port 的差异**如下：

- `update` 默认不原地改写 `x`（`out=None` 时分配新输出）；需要预分配输出时传
  `out`。
- 跳过标记统一为 `pad_slot_id`（默认 `-1`），映射到 vLLM 侧的
  `null_block_id`。
- vLLM kernel 无条件加载 `cache_indices` / `has_initial_state`（fwd）和
  `conv_state_indices`（update），统一入口在未传时自动构造 identity 映射 /
  全 False mask，语义与 SGLang 的 identity-cache 模式一致。
- `fwd` 与 `update` 都要求真实传入 `conv_states` / `conv_state`（kernel 会
  **inplace 更新** state cache），这是两个 port 共同的前置条件。

### 2.4 参数速览

```python
out = causal_conv1d_fn(
    x, weight, bias=None, conv_states=None, query_start_loc=None,
    seq_lens_cpu=None, cache_indices=None, has_initial_state=None,
    activation="silu", pad_slot_id=-1,
    null_block_id=None, block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None, initial_state_idx=None,
    num_computed_tokens=None, block_size_to_align=0, metadata=None,
    backend="auto", validate_data=False,
)

out = causal_conv1d_update(
    x, conv_state, weight, bias=None, activation=None,
    conv_state_indices=None, pad_slot_id=-1,
    cache_seqlens=None, num_accept_tokens=None,
    intermediate_conv_window=None, intermediate_state_indices=None,
    retrieve_next_token=None, retrieve_next_sibling=None, retrieve_parent_token=None,
    num_accepted_tokens=None, query_start_loc=None, max_query_len=-1,
    null_block_id=None, block_idx_last_scheduled_token=None,
    initial_state_idx=None, out=None,
    backend="auto", validate_data=False,
)
```

## 3. 参数详解

### 3.1 `causal_conv1d_fn`

| 参数 | Shape | 常见 dtype | 归属 | 含义 |
| --- | --- | --- | --- | --- |
| `x` | `[dim, cu_seqlen]` | BF16/FP16/FP32 | 通用 | varlen 拼接的输入；要求 feature 维（`dim`）连续，即 channel-last |
| `weight` | `[dim, width]` | 同 x 或 FP32 | 通用 | 逐通道卷积权重；要求 `width` 维连续 |
| `bias` | `[dim]` 或 None | 同 x | 通用 | 可选偏置 |
| `conv_states` | `[num_cache_lines, dim, state_len]` | 同 x | 通用 | state cache，`state_len >= width-1`；**inplace 更新**，可为非连续 view |
| `query_start_loc` | `[batch+1]` | int32/int64 | 通用 | 各序列累计 token 数，0 开头；末值为 `cu_seqlen` |
| `seq_lens_cpu` | `[batch]` list | int | sglang | 每序列 token 数（CPU 侧 list，用于计算 grid）；sglang 路由必需 |
| `cache_indices` | `[batch]` | int32/int64 | 通用 | 每序列对应的 cache 行：`conv_states[cache_indices[b]]`；为 `pad_slot_id` 的条目跳过；vLLM 路径未传时自动 identity |
| `has_initial_state` | `[batch]` | bool | 通用 | 为 True 时以 cache 现有内容作为初始历史；需要 `conv_states`；vLLM 路径未传时自动全 False |
| `activation` | scalar | str/bool/None | 通用 | `"silu"`/`"swish"`/`True` 启用 SiLU，`None` 关闭 |
| `pad_slot_id` | scalar | int | 通用 | 跳过标记，默认 `-1` |
| `null_block_id` | scalar | int | vllm | vLLM null block 标记（上游默认 `0`）；统一入口默认 None → 用 `pad_slot_id` |
| `block_idx_first_scheduled_token` | `[batch]` | int32 | vllm | APC：`cache_indices` 中本次要填充的第一个 cache block 位置 |
| `block_idx_last_scheduled_token` | `[batch]` | int32 | vllm | APC：`cache_indices` 中本次要填充的最后一个 cache block 位置 |
| `initial_state_idx` | `[batch]` | int32 | vllm | APC：`cache_indices` 中初始 state 所在 block 位置 |
| `num_computed_tokens` | `[batch]` | int32 | vllm | APC：每条序列已完成（可复用）的 token 数 |
| `block_size_to_align` | scalar | int | vllm | APC：state 对齐的 mamba block size，须为 `BLOCK_M=8` 的倍数；0 表示不启用对齐 |
| `metadata` | vLLM V1 对象 | — | vllm | V1 scheduler metadata（`nums_dict` / `batch_ptr` / `token_chunk_offset_ptr`）；None 时由 `query_start_loc` 在 host 端构造 |
| `backend` | scalar | str | 通用 | `"auto"` / `"sglang"` / `"vllm"` |
| `validate_data` | scalar | bool | 通用 | 入口做 shape / stride 断言 |
| 返回 `out` | `[dim, cu_seqlen]` | 同 x | 通用 | 卷积结果，新建张量 |

### 3.2 `causal_conv1d_update`

| 参数 | Shape | 常见 dtype | 归属 | 含义 |
| --- | --- | --- | --- | --- |
| `x` | `[batch, dim]` 或 `[batch, dim, seqlen]`；vLLM 另支持 `[num_tokens, dim]` | BF16/FP16/FP32 | 通用 | decode 输入；2D 自动 `unsqueeze(-1)`；要求 `stride(1)==1` |
| `conv_state` | `[num_cache_lines, dim, state_len]` | 同 x | 通用 | state cache，`state_len >= width-1`；**inplace 更新** |
| `weight` | `[dim, width]` | 同 x 或 FP32 | 通用 | 卷积权重，`width` 维连续 |
| `bias` | `[dim]` 或 None | 同 x | 通用 | 可选偏置 |
| `activation` | scalar | str/bool/None | 通用 | `"silu"`/`"swish"`/`True` 启用 SiLU |
| `conv_state_indices` | `[batch]` | int32 | 通用 | continuous batching：从更大的 cache 中选择行；`pad_slot_id` 跳过；vLLM 路径未传时自动 identity |
| `pad_slot_id` | scalar | int | 通用 | 跳过标记，默认 `-1` |
| `cache_seqlens` | `[batch]` | int32 | sglang | 循环缓冲模式；当前未实现（`validate_data` 要求为 None） |
| `num_accept_tokens` | `[batch]` | int32 | sglang | spec-decode：本次已接受 token 数；非 None 时启用 spec-decode 语义 |
| `intermediate_conv_window` | `[num_cache_lines, steps, dim, width-1]` | 同 x | sglang | spec-decode：逐 step 保存卷积窗口 |
| `intermediate_state_indices` | `[batch]` | int32 | sglang | spec-decode：中间 window 的 cache 行选择 |
| `retrieve_next_token` / `retrieve_next_sibling` | `[batch, seqlen]` | int32 | sglang | Eagle tree 结构：每个 token 的 child / sibling 引用 |
| `retrieve_parent_token` | `[batch, seqlen]` | int32 | sglang | Eagle tree 输出：kernel 计算出的 parent 映射 |
| `num_accepted_tokens` | `[batch]` | int32 | vllm | spec-decode：已接受 token 数（与 `num_accept_tokens` 同义） |
| `query_start_loc` | `[batch+1]` | int32 | vllm | varlen decode：各序列起点；非 None 时 `x` 为 `[num_tokens, dim]` 连续拼接 |
| `max_query_len` | scalar | int | vllm | varlen decode：batch 内最大 query 长度（决定 kernel 的 `seqlen`） |
| `null_block_id` | scalar | int | vllm | null block 标记；统一入口默认 None → 用 `pad_slot_id` |
| `block_idx_last_scheduled_token` | `[batch]` | int32 | vllm | APC：更新后的 state 写回 `cache_indices` 的最后一个 block 位置 |
| `initial_state_idx` | `[batch]` | int32 | vllm | APC：初始 state 所在 block 位置 |
| `out` | 同 `x` | 同 x | vllm | 可选预分配输出；默认新建（不原地改写 `x`） |
| `backend` | scalar | str | 通用 | `"auto"` / `"sglang"` / `"vllm"` |
| `validate_data` | scalar | bool | 通用 | 入口做 shape / stride 断言 |
| 返回 `out` | 同 `x` | 同 x | 通用 | 卷积结果；统一入口为新建张量 |

### 3.3 前置条件

**公共（两个 port 一致）**

- 所有 tensor 位于同一 CUDA/HIP device
- `conv_states` / `conv_state` 必须真实传入（统一入口强制；kernel **inplace** 更新 state cache）
- `weight.stride(1)==1`，shape 为 `(dim, width)`；`bias` 长度等于 `dim`
- `conv_states.shape[1] == dim`、`state_len >= width-1`
- `cache_indices` / `has_initial_state` / `conv_state_indices` 长度等于 batch（传了的话）
- 外层 stride 灵活：`conv_states` 可为 `transpose` 得到的非连续 view

**fwd**

- `x.dim()==2`，channel-last（`x.stride(0)==1`）；`validate_data` 打开时强制校验
- `query_start_loc[-1] == cu_seqlen`，`len(query_start_loc)-1 == batch`
- `width ∈ [2, 5]`（两个 port 的 fwd kernel 均到 5）
- vLLM 路径：`block_size_to_align` 为 `BLOCK_M=8` 的倍数；APC 四元组
  （`block_idx_first/last_scheduled_token`、`initial_state_idx`、
  `num_computed_tokens`）通常一起提供，由 `block_idx_last_scheduled_token`
  是否非 None 决定 `IS_APC_ENABLED`

**update**

- `x.stride(1)==1`（feature 维连续）
- `conv_state.stride(-2)==1`（dim 维连续，sglang `validate_data` 断言）、
  `dim == weight.size(0) == conv_state.size(1)`、`num_cache_lines >= batch`
- 未传 `conv_state_indices` 时 `conv_state.size(0) >= batch`，行号即序列号
- 常规 decode（KDA 主路径）只需 `x / conv_state / weight / activation`，其余高级参数传 None
- vLLM 路径：`width` 最高 6；varlen decode 时 `conv_state_indices` 必填
  （统一入口未传时自动 identity）
- sglang `validate_data=True` 时：`cache_seqlens` 必须为 None；使用
  `conv_state_indices` 时要求同时提供 `intermediate_state_indices`
  （断言偏保守，正常路径不需要）

## 4. 典型 Shape（Kimi-K3 / KDA 运行时）

Kimi-K3 / KDA 运行时（TP=8，每 rank 12 头 × 128；全模型 96×128=12288）实测值，
两个 port 均可直接运行（vLLM 路径无需 `seq_lens_cpu`）：

| 场景 | `x` | `weight` | `conv_states` | index / 其他 |
| --- | --- | --- | --- | --- |
| fwd（每 rank，q/k/v 各一次） | `[1536, T_total]` BF16 | `[1536, 4]` FP32 | `[33, 1536, 3]` BF16，非连续 view | `query_start_loc [batch+1]` int32；`cache_indices [batch]` int32；`has_initial_state [batch]` bool；`activation="silu"` |
| decode（每 rank） | `[batch, 4608]` BF16（q+k+v 拼接 3×1536） | `[4608, 4]` FP32 | `[33, 4608, 3]` BF16，非连续 view | `conv_state_indices [batch]` int32；`activation="silu"` |

- `T_total` 为 extend batch 全部 token 数。性能口径三档（8 序列均匀 varlen，
  与测试 `k3-tp8-bf16x-fp32w-t*` 对齐）：
  - **小** = `T=1024`（8×128）
  - **中** = `T=4096`（8×512）
  - **大** = `T=16384`（8×2048）
  decode batch 受 `--max-running-requests` 约束，当前上限 32（b1/b8/b32）。
- `conv_states` 底层池布局为 `[slots, state_len, dim]`，调用前 `transpose(-1,-2)`
  得到 `[slots, dim, state_len]`；实测 strides `(13824, 1, 4608)`，dim 连续、
  state_len 步长 4608，kernel 直接接受该非连续 stride。
- 混合精度：`x` / `conv_states` 为 BF16，`weight` 为 FP32（K3 checkpoint 原始精度），
  kernel 内以 FP32 累加。
- K3 的 `qkv_conv1d(bias=False)`，两个入口的 `bias` 运行时恒为 None
  （`fn` / `update`）。
- vLLM V1 场景中，fwd 的 `metadata` 由 scheduler 提供紧凑 launch 数组；无
  scheduler 时（如单测、离线脚本）统一入口会由 `query_start_loc` 在 host 端
  构造 `batch_ptr` / `token_chunk_offset_ptr`。

## 5. 正确性测试与性能基准

```bash
# 唯一测试套件：路由规则（CPU）+ 双后端交叉正确性 + 各 port 边界用例
# （measured Kimi-K3 shape、非连续布局、vLLM varlen decode、large state
#  cache、hand-computed known values），全部按 backend 参数化
pytest -q test/test_generic_causal_conv1d.py

# 性能对比：同一批真实 shape 上同时测两个后端（仅对比，不做合并决策）
pytest -q test/test_generic_causal_conv1d.py --bench -s
```

PyTorch reference 等价于（逐序列）：

```python
state = conv_states[line][:, :state_len]          # has_initial_state / update 路径
x_new = torch.cat([state, x_seq], dim=-1)
out_seq = F.conv1d(x_new, weight.unsqueeze(1), bias, groups=dim)
out = silu_if(out_seq, activation)                # 仅取最后 seqlen 个 token
conv_states[line][:, :state_len] = x_new[:, -state_len:]
```

参考实现位于 `triton/causal_conv1d/common.py`，两个 port 的交叉正确性共用同一
oracle，feature 漂移会直接暴露为测试失败。

## 6. 溯源

- vLLM：`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`，固定来源 commit
  `aeeb36b1`（2026-07-30，main）；含 PDL / APC / V1 紧凑调度；CPU fallback
  （`vllm ... cpu.causal_conv1d`）未移植；`PAD_SLOT_ID` / `NULL_BLOCK_ID` 源自
  `vllm.v1.attention.backends.utils`。
- SGLang：`python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`，固定来源 commit
  [`246b3c3e`](https://github.com/sgl-project/sglang/commit/246b3c3eafde7c18748d3bdc6ecd4a65c51d7b7b)；
  ROCm/Triton fallback（SGLang 快速路径已迁移至 `sgl_kernel`），不含 NPU 实现。
- 上游 lineage：两个文件头部均注明改编自
  [vLLM `causal_conv1d.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py)，
  最终源自 Dao-AILab [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d)
  （Tri Dao）。

## 附录 A. 仓库文件

| 用途 | 文件 |
| --- | --- |
| 公开接口 | `boltops/generic/__init__.py`（统一入口） |
| 子包公共导出 | `boltops/generic/triton/causal_conv1d/__init__.py` |
| 共享基础 / 参考实现 / 路由 / 统一入口| `boltops/generic/triton/causal_conv1d/core.py` |
| 统一入口 | `boltops/generic/triton/causal_conv1d/api.py` |
| SGLang/vLLM Triton 实现 | `boltops/generic/triton/causal_conv1d/kernels.py` |
| 统一测试 / 性能对比 | `test/test_generic_causal_conv1d.py`（`--bench`；双后端参数化 + 各 port 边界用例） |

## 附录 B. 实现备注

- 两个 port 均固定 `BLOCK_N=256`、`num_stages=2`，无 autotune；fwd 固定
  `BLOCK_M=8`。
- SGLang fwd grid：`(batch, ceil(max(seq_lens_cpu)/BLOCK_M), ceil(dim/BLOCK_N))`；
  vLLM fwd grid：`(total_chunks, ceil(dim/BLOCK_N))` 紧凑调度，`batch_ptr` /
  `token_chunk_offset_ptr` 给出每个 program 的 `(seq, chunk)`，无 `metadata` 时
  host 端构造（`MAX_NUM_PROGRAMS=1024`，多余 slot 填 `PAD_SLOT_ID` 跳过）。
- update grid 两个 port 相同：`(batch, ceil(dim/BLOCK_N))`。
- 每个 Triton program 沿 feature 处理 `BLOCK_N` 个通道；`KERNEL_WIDTH=width`、
  `NP2_STATELEN=next_power_of_2(state_len)` 为编译期常量（vLLM update 支持
  `width=2..6`，其余 kernel 到 5）。
- fwd 中只有 `chunk_offset==0` 的 program 负责读初始态并 shift-left 写回 state；
  非首 chunk 从 `x` 前一 chunk 末尾取历史（SGLang 用 `.ca` cache modifier）。
- 累加使用 FP32（`acc` 以 FP32 预载 bias / 零并累加），支持 bf16 x × fp32 weight
  的混合精度；SiLU 为 `acc / (1 + exp(-acc))`。
- update 的 state 更新是滑动窗口：`state_len >= seqlen` 时先读 shift 后的旧 state，
  再以 `tl.where` 混合新 token 写回；spec-decode 时按 `num_accept_tokens - 1`
  偏移，有效 `state_len = width - 1 + (seqlen - 1)`。
- Eagle tree 模式（`retrieve_next_token` 非 None）：历史沿 parent 链取，kernel
  同时计算并写出 `retrieve_parent_token`。
- vLLM fwd 的 APC 路径：按 `block_size_to_align` 计算对齐后的首尾 full block，
  从 `initial_state_idx` 读初始 state、把新 state 写回 `block_idx_last_scheduled_token`
  指向的 cache block（`num_computed_tokens` 决定复用部分）。
- vLLM update 的 varlen 路径：`IS_VARLEN` 下用 `query_start_loc` 修正实际
  `seqlen` 与 `state_len`，空序列提前返回。
- vLLM 入口内部把 `x` 转到 `conv_states` 的 dtype 计算，返回前转回原 dtype。
- `is_arch_support_pdl()` 恒为 False，两个 port 的 GDC / PDL launch 路径均不启用
  （CUDA-only，当前平台为 HIP）。
- vLLM port 的 fwd 已迁移 sglang 的 vec8（dwordx4）+ fast_div + auto-chunks
  优化（非 APC 路径默认启用；APC 与 `width≠4` 回退上游 kernel，上游 kernel
  保持逐字不变）；update 侧仅对齐 `silu_fast` 参数（默认关闭）。

## 附录 C. Benchmark Shape

正确性之外的性能 case（`pytest --bench`；fwd/update 均为 BF16 x + FP32 weight、
width=4、`activation="silu"`）。统一性能对比（`test_generic_causal_conv1d.py
--bench`）在相同 shape 上同时测两个 port 并输出 `vllm/sglang` 比值（仅作
对比，不做合并决策）。fwd 主口径为三档（**小 T=1024 / 中 T=4096 /
大 T=16384**，8 序列均匀 varlen），其余为历史/补充 case：

| Case | dim | `lengths`（fwd）或 batch / seqlen（update） |
| --- | ---: | --- |
| fwd `small-t1024` | 1536 | `(128,) × 8` → T=1024 |
| fwd `medium-t4096` | 1536 | `(512,) × 8` → T=4096 |
| fwd `large-t16384` | 1536 | `(2048,) × 8` → T=16384 |
| fwd `measured-single` | 1536 | `(46,)` |
| fwd `varlen-b8` | 1536 | `(46, 91, 137, 182, 57, 203, 118, 76)` |
| fwd `varlen-max-batch` | 1536 | 32 条序列，40–250 token/条（确定性生成） |
| fwd `long-prefill` | 1536 | `(4096,)` |
| fwd `full-model` | 12288 | `(4156,)` |
| update `decode-b1-measured` | 4608 | batch=1, seqlen=1 |
| update `decode-b8` | 4608 | batch=8, seqlen=1 |
| update `decode-b32-max` | 4608 | batch=32, seqlen=1 |
| update `decode-multi-token` | 4608 | batch=8, seqlen=5 |

性能口径为 useful FLOPs（`2*dim*tokens*width`）。fwd 主口径覆盖
T=1024/4096/16384 三档（附录 D 统一按此记录），补充 case 覆盖实测单 seq
（46 token）、多序列 varlen 与 batch 上限、全模型 dim；update 覆盖实测
`batch=1`、常规 batch 8/32（`--max-running-requests` 上限）与多 token decode。

## 附录 D. 性能基准与调优记录

调优历程按 **基线 → fast_div → vec8 → chunks** 四个阶段记录，fwd 统一使用三档尺寸
（**小 T=1024 / 中 T=4096 / 大 T=16384**，8 序列均匀 varlen，
`dim=1536`、width=4、BF16 x + FP32 weight、SiLU）；decode 用 b1/b8/b32。

### D.1 平台与测量口径

目标平台为海光 HCU3（gfx936 / BoWenA0），80 CU @ 1.5 GHz，HBM 1.8 TB/s。
硬件峰值按 roofline 口径：FP32 vector 15.36 TFLOPS（80 CU × 64 lanes ×
2 FLOP × 1.5 GHz）、BF16 packed 30.72 TFLOPS；本算子无 `v_mfma`，MFMA 峰值
不适用。本机 GPU 时长波动大（同一二进制实测波动可达 ±30%），所有 A/B 均多次
取最小值，xcu/Xprofiler 指标取自 .perf dispatch 记录。

算子本身是**低算术强度**算子：以 fwd medium shape 为例
（`dim=1536`、T=4096、width=4、BF16 x + FP32 weight），useful FLOPs 仅
50.3 MFLOP，最小 DRAM 流量约 25.5 MB（按此口径算术强度约 2 FLOP/B），
远低于 HCU3 的 ridge 点（约 8.5 FLOP/B）。因此性能目标不是逼近算力峰值，
而是控制每 token 的访存请求路径与启动开销。

### D.2 调优历程总览（基线 → fast_div → vec8 → chunks）

fwd dispatch 级（torch.profiler kernel GPU 时间，best-of 同窗口，
channel-last x，dim=1536，width=4，SiLU）：

| 阶段 | 关键改动 | T=1024 | T=4096 | T=16384 | 相对基线 |
|---|---|---:|---:|---:|---|
| **基线** | 原始 SGLang 标量 kernel（2B `buffer_load_ushort`，精确 `/`） | 17.81 µs | 55.32 µs | 201.68 µs | 1.0x |
| **fast_div** | SiLU 除法链 → `tl.fdiv_fast`（`v_rcp_f32 + v_mul_f32`） | — | 51.8 µs（历史口径，基线 56.5） | — | 0.91x（T=4096） |
| **vec8** | 8-token tile `dwordx4` 向量化 + fast_div 默认（CHUNKS=1） | 15.16 µs | 43.85 µs | 154.5 µs | **1.17x / 1.26x / 1.31x** |
| **chunks** | 多-chunk MLP 分组（`CHUNKS` 自动 1/2/4，仅 channel-last） | 15.16 µs | 39.46 µs | 128.79 µs | **1.17x / 1.40x / 1.57x**（vs vec8：0% / -10% / -17%） |


### D.3 基线阶段：原始 SGLang 标量 kernel

**fwd dispatch 级**（`vectorize=False` 回退上游 kernel，普通 `/` 除法，
torch.profiler 口径）：

| 尺寸 | 场景 / shape | dispatch | 瓶颈 |
|---|---:|---:|---|
| 小 | T=1024（8×128） | 17.81 µs | launch/请求通路 |
| 中 | T=4096（8×512） | 55.32 µs | L2 请求通路（标量 load 请求数最多） |
| 大 | T=16384（8×2048） | 201.68 µs | L2 请求通路 + VALU |

**Roofline 分析**（历史基线 dispatch，真实 8 序列 910 token、`dim=1536`）：

| 指标 | 实测 | 含义 |
|---|---:|---|
| GPU 时长 | 75.68 µs | 单个 dispatch 的 GPU 窗口 |
| VALU FLOPs | 650.85 GFLOP/s（4.24%） | 算力利用率极低 |
| DRAM fabric BW | 125.3 GB/s（6.84%） | 远未触及 1.8 TB/s 峰值 |
| L2-Util | 98.08% | **L2 请求通路饱和，主瓶颈** |
| TA-Busy / TD-Busy | 71.4% / 73.3% | VMEM 地址/数据通路忙碌 |
| VL1_L2-BW | 2586.6 GB/s | L1→L2 请求带宽 |
| L1 / L2 hit | 24.8% / 97.0% | x 复用少，但大部分命中 L2 |
| active waves / CU | 19.75 / 40（49.4%） | 占用率一半 |
| wr/rd | 2.18x | 写放大明显（out + state 写回） |

结论：kernel 处于 **request-path-bound**（L2 98% 饱和）而非 compute-bound
或 DRAM-bound。据此给出的优化排序：

1. 降低 L1→L2 请求数（宽加载 / 合并率），这是请求通路瓶颈的直接来源；
2. 提升 occupancy（当前 ~50%，减 VGPR / 软件流水）；
3. 增大 `BLOCK_M` 摊薄每 program 固定开销；降低写放大
   （streaming store / no-allocate）；
4. 算力侧优化排在最后：SiLU 的除法链（见 D.4）。

**decode 基线**（update kernel，`BLOCK_N=256/num_stages=2`、普通 `/`，
xcu dispatch 级）：

| 尺寸 | 场景 / shape | dispatch | 瓶颈 |
|---|---:|---:|---|
| 小 | batch=1，`[1, 4608, 1]` | 4.96 µs | grid 18 program，不足 1 wave，latency 主导 |
| 中 | batch=8，`[8, 4608, 1]` | 6.40 µs | wave 欠占（5.84/40），latency/occupancy 主导 |
| 大 | batch=32，`[32, 4608, 1]` | 11.68 µs | kernel 随 batch 线性上升；wave 欠占（17.98/40） |

decode kernel 仅 5–12 µs；b1 只有 18 个 program（不足 1 wave），kernel 内
为 **latency/occupancy-bound**，优化收益有限（见 D.4 / D.6）。

### D.4 fast_div 阶段：SiLU 快速除法（`tl.fdiv_fast`）

fwd / update 的 SiLU 为 `acc / (1 + tl.exp(-acc))`。反汇编显示默认路径在
HCU3 上生成完整精度的 Newton-Raphson 除法链（约 13 条指令：
`v_div_scale ×2 + v_rcp + v_fma ×3 + v_div_fmas + v_div_fixup`），每输出元素
多出约 10 条指令；fwd 处于 SIMD issue-bound（VALU 仅占 ~58% issue 槽）。

| 项 | 结果 |
|---|---|
| 实现 | `tl.fdiv_fast` 语义随本机 Triton fork（`51c0a1d2b [HCU] add fdiv_fast port lower to llvm.amdgcn.fdiv.fast`）落地，f32 下降为 `v_rcp_f32 + v_mul_f32` |
| ISA | 快速路径含 `v_rcp_f32`、无 `v_div_fixup_f32`；精确路径保持完整 13 条链（向量指令 57 → 38） |
| 精度 | 快速 vs 精确最大相对差 3.5e-6；bf16 容差内无风险（SiLU 分母恒在 `[1, 2]`，不落入 denormal 边界） |
| 性能（标量 kernel，T=4096 历史口径） | best-of 51.8 µs vs 同窗口基线 56.5 µs，**约 -9%** |
| 正确性 | 交叉正确性全绿（`test_unified_fwd_matches_torch` 等，bf16 容差） |

**合入状态（2026-08-12）**：fast-div 语义已随 Triton fork 落地为
`tl.fdiv_fast`，vec8 sibling kernel 的 SiLU 行已改为
`tl.fdiv_fast(acc, 1 + tl.exp(-acc))`。接入方式为 wrapper 内
`hasattr(tl, "fdiv_fast")` 自动探测：fork 构建自动启用快速路径，stock Triton
回退普通 `/`（上游 kernel 与 vLLM port 保持逐字不变）。特殊值契约：denormal
输入/结果会被 `v_rcp_f32` 冲刷为 0/inf，通用用户需用 `ieee_rounding=True` /
`tl.math.div_rn`。

### D.5 vec8 阶段：load/store dwordx4 向量化

**背景**：生产输入 x 是 channel-last（feature 维连续，`x.stride(0)==1`，见
§3.1 / §4）。基线 fwd kernel 的主循环逐 token 加载 1D `[BLOCK_N=256]`
feature tile，在 num_warps=4（256 lanes）下每线程只有 1 个元素，生成 2B
标量 `buffer_load_ushort`；out 的 store 同理。反汇编确认没有 16B 访问。

**方案**：新增 sibling kernel `_causal_conv1d_fwd_kernel_vec8`，主循环一次
处理 8 个 token，x/out 用 `[BLOCK_N, BLOCK_T=8]` tile（feature × token），
feature 维连续（stride = 1），每线程 8 个连续元素 = 16B。Triton 布局推导沿
连续维每线程 8 元素，生成 `buffer_load_dwordx4` / `buffer_store_dwordx4`。
卷积的 4-tap 滑动窗口用 4 次错位 load（`x[tok-3..tok]`）并行计算，chunk 0
的初始 state 用 `tl.where` 合并，state 写回逻辑与上游一致；SiLU 默认走
fast_div（D.4）。

**接入方式**：`causal_conv1d_fn` 在 `width==4` 时默认走 vec8 路径，可传
`vectorize=False` 回退上游 kernel；上游 kernel 原样保留（diff 可追溯）。

**验证结果**（dispatch 级，channel-last x，dim=1536，width=4，SiLU；
修正尺寸为 torch.profiler 口径，历史口径为 xcu dispatch 口径）：

| 尺寸 | 基线（标量 kernel） | vec8（CHUNKS=1，含 fast_div） | 实测加速 |
|---|---:|---:|---|
| T=1024 | 17.81 µs | 15.16 µs | 1.17x |
| T=4096 | 55.32 µs | 43.85 µs | 1.26x |
| T=16384 | 201.68 µs | 154.5 µs | 1.31x |

历史口径：varlen-b8（910 token）27.84→22.72 µs（1.23x）；long-prefill
（4096）57.92→46.24 µs（1.25x）；ISA 为 `dwordx4`×7 load + `dwordx4` store。
多-chunk MLP 分组与 vLLM 迁移见 D.6。

正确性：`test/test_generic_causal_conv1d.py` 全绿（36 passed，含双后端交叉
正确性与各边界用例）。kernel 层收益需按 dispatch 级口径评估。

**边界与未覆盖**：

- 仅 `width==4`（Kimi-K3 实际核宽）；其他 width 自动回退上游 kernel；
- row-major x（token 连续、feature stride=cu）下 vec8 结果正确但 feature 维
  连续跨度不足，无法获得 dwordx4（Triton 对 `BLOCK_T=8` 的小连续维拒绝
  向量化），生产 channel-last 布局才有收益；
- state 的读/写（`[BLOCK_N]` 1D）仍为标量，非主循环热点，未纳入本次向量化；
- update kernel 未做同样改造：decode kernel 仅 5–12 µs 且为
  latency/occupancy-bound（D.3），kernel 内向量化收益预期有限。

### D.6 chunks 阶段：多-chunk MLP 分组

每个 program 用 `tl.static_range` 处理 `CHUNKS` 个连续 8-token chunk，权重
只加载一次，每线程 outstanding loads 从 4 提升到 4×CHUNKS（T=4096 时 8、
T=16384 时 16）。实测（kernel GPU 时间，channel-last）：

| CHUNKS | T=1024 | T=4096 | T=16384 |
|---|---:|---:|---:|
| 1 | 15.16 µs | 43.85 µs | 154.5 µs |
| 2 | 16.20 µs（+7%） | 39.71 µs（-9%） | 132.8 µs（-14%） |
| 4 | 17.82 µs（+18%） | 39.34 µs（-10%） | 128.8 µs（-17%） |

规律：program 充足时 MLP 提升 > occupancy 损失（T=4096/16384 各 -10%/-17%）；
短 prefill program 稀少时分组反而饿死 occupancy（T=1024 回退）。wrapper
按 `n_chunks = ceil(max_seq_len/8)` 自动选：`>=64 → 4`、`>=32 → 2`、否则 1，
且**仅在 channel-last 布局生效**（row-major x 下 load 是标量，分组只会放大
寄存器压力，实测 full-model 端到端 0.615→1.107 ms 回退）；可用 `chunks=`
kwarg 显式覆盖。

**vLLM port 迁移**（同 kernel GPU 口径，channel-last）：vLLM 版 vec8 sibling
保留紧凑调度（`batch_ptr` / `token_chunk_offset_ptr`），实测
`vllm_vec8 vs vllm_upstream`：T=4096 58.0→50.6 µs（-13%）、T=16384
206.4→147.1 µs（-29%）、T=1024 20.8→21.9 µs（噪声）；相对 sglang vec8
仍慢 ~1.1–1.5x（紧凑调度的每-entry 间接寻址与分支开销）；**精度与 sglang
输出/state 逐位一致（max diff = 0）**。端到端 vLLM 仍被 wrapper 的 host
构造开销主导（每次调用重建 `batch_ptr`，~0.4–0.5 ms），kernel 级收益需按
dispatch 口径评估。

### D.7 全尺寸 xcu 复核与上限结论（2026-08-12）

本节在 D.3–D.6 之后，用 xcu 对三档 prefill（**T=1024 / 4096 /
16384，8 序列均匀 varlen**，与测试 `k3-tp8-bf16x-fp32w-t*` 对齐）与三档
decode 复核当前状态，并给出是否达到硬件理论上限的结论。测量口径：GPU 2
（`HIP_VISIBLE_DEVICES=2`），xcu 4.4.4，
`-S speed_of_light,memory_workload,occupancy --kernels _causal_conv1d_fwd_kernel`
（substring 匹配 vec8 kernel；`launch_statistics` section 会触发 xcu
core dump，不可用），每个 bench 脚本 2 次 warmup + 5 次 timed dispatch，
取最小时长 dispatch；**当前默认含 auto-chunks**（T=4096/16384 自动
CHUNKS=4，见 D.6）。**布局必须为 channel-last**（`x.stride(0)==1`）：若按
`randn(dim, T)` 生成 row-major x，vec8 的 8-token tile 跨 `T` 的 stride，
dwordx4 全部退化为标量 load，fwd T=16384 从 158 µs 恶化到 2.45 ms（L2-Util
100%、TA/TD 97% 饱和）。

#### D.7.1 当前状态（dispatch 级，vec8 + fast_div + auto-chunks）

| 尺寸 | 场景 / shape | dispatch | VALU | VL1→L2 BW | L2-Util | TA/TD-Busy | achieved waves/CU | 瓶颈 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 小 | fwd T=1024（8×128，CHUNKS=1） | 16.32 µs | 48.0% | 682 GB/s | 96.6% | 55.0/69.1% | 23.4 | L2 请求通路 + occupancy |
| 中 | fwd T=4096（8×512，CHUNKS=4） | 39.84 µs | 71.1% | 817 GB/s | 98.7% | 43.3/71.4% | 22.4 | VALU + L2 请求通路 |
| 大 | fwd T=16384（8×2048，CHUNKS=4） | 128.16 µs | 88.0% | 1029 GB/s | 98.6% | 57.3/87.4% | 30.6 | VALU / TD / L2 三向饱和 |
| decode b1 | 4608, batch=1 | 4.96 µs | 1.1% | 30 GB/s | 44% | 4.9/10.5% | 0.8 | latency 主导（18 program，不足 1 wave） |
| decode b8 | 4608, batch=8 | 5.60 µs | 7.5% | 211 GB/s | 68.9% | 33.1/57.6% | 6.0 | launch + 请求通路 |
| decode b32 | 4608, batch=32 | 11.84 µs | 14.3% | 399 GB/s | 79.3% | 56.6/68.1% | 18.4 | 请求通路随 batch 上升 |

decode 三档与 D.3 基线同量级：kernel 仅 5–12 µs，b1 只有 18 个 program（不足 1 wave），kernel 内为 **latency/occupancy-bound**。

> 与 CHUNKS=1 相比（T=4096 43.84 µs、T=16384 153.9 µs），CHUNKS=4 同时把
> **VL1→L2 请求数降 ~33%**（chunk 间 halo token 被同一 program 的 L1 吸收：
> T=4096 7.1M→4.7M、T=16384 28.3M→18.9M）并摊薄 TA 地址处理（T=16384
> TA-Busy 90%→57%）。VALU 相对上升（T=16384 80%→88%），瓶颈进一步移交
> 算力侧。

#### D.7.2 结论：是否达到硬件理论上限（只看 kernel GPU 耗时）

**结论：未达到硬件理论上限。** fwd 大/中（T=16384/4096）已顶到该访问模式的
L1→L2 请求通路上限（L2-Util 98.6%/98.7%），与 DRAM roofline 差 2.3x/2.8x，
差距来自每输出 4 次错位 load 的请求数；fwd 小（T=1024）接近同一上限（L2
96.6%，occupancy 次要）；仅 decode（L2 ≤79%）是 occupancy/latency 未满。

| 场景 | DRAM roofline | kernel GPU | 差距 | kernel 内卡点 |
|---|---:|---:|---:|---|
| fwd T=1024 | 3.6 µs | 16.3 µs | 4.5x | L2 96.6% 接近饱和 + occupancy（次要） |
| fwd T=4096 | 14 µs | 39.8 µs | 2.8x | L2 98.7% 请求通路饱和 + VALU 71% |
| fwd T=16384 | 56 µs | 128.2 µs | 2.3x | L2 98.6% + VALU 88% + TD 87% 三向饱和 |
| decode b1/b8/b32 | 0.03/0.3/1.3 µs | 4.96/5.60/11.8 µs | 9–165x | occupancy/latency 未满（waves 0.8–18.4/40） |

依据：请求数来自每输出 4 次 `dwordx4` load（CHUNKS=4 已把 L2 请求降 33%）；
T=4096 用 CHUNKS=1（occupancy 更高）反而更慢（43.85 vs 39.46 µs），证明 fwd
中瓶颈是请求数而非 occupancy。逼近 roofline 需请求数降 ~4x（load-once + 寄存器
移位），Triton 表达不了（`tl.gather` 走 LDS 更慢）；decode 可挖降 VGPR / 增
program 并行度。kernel 侧其余候选（`BLOCK_T=16`、stream store、eviction、
`BLOCK_N/num_warps` 等）均无收益或回退，增量已收敛。
