# KDA `target_verify` 接口

BoltOPs 按算法家族组织代码：

| 包 | 边界 |
|---|---|
| `boltops.fla.gdn` | GDN 家族 API；包含 vLLM GDN decode ABI |
| `boltops.fla.kda` | KDA 家族 API；包含 KDA prefill/decode 算子 |
| `boltops.fla.common` | GDN/KDA 共享 Triton kernel；包含 SGLang decode/verify recurrent kernel |

`target_verify` 不是另一个算子名，而是 SGLang adapter 对共享
decode/verify kernel 的一种调用模式。该 kernel 同时通过 `is_kda` 支持 GDN
和 KDA，因此推荐从 `common` 导入：

```python
from boltops.fla.common.triton.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
```

同名函数 `boltops.fla.gdn.fused_sigmoid_gating_delta_rule_update` 是 vLLM
GDN decode ABI，实现位于
`boltops/fla/gdn/triton/fused_sigmoid_gating.py`。它与本页的 SGLang 共享
decode/verify ABI 不是同一个函数，不能互换。

## 调用方式

```python
output = fused_sigmoid_gating_delta_rule_update(
    A_log=A_log,
    a=a,
    dt_bias=dt_bias,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    q=q,
    k=k,
    v=v,
    b=b,
    initial_state_source=state,
    initial_state_indices=state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=cu_seqlens,
    is_kda=True,
    lower_bound=-5.0,
    disable_state_update=True,
    intermediate_states_buffer=intermediate_states,
    intermediate_state_indices=intermediate_indices,
    cache_steps=draft_token_num,
    retrieve_parent_token=None,
    cache_ring=False,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
)
```

调用后只返回 output；不回写 `state`，而是把每个 draft step 的 post-state 写入
`intermediate_states`。

## 主要参数

| 参数 | Shape/类型 | 说明 |
|---|---|---|
| `q`, `k` | `[1, total_tokens, H, K]` | KDA verify 使用 varlen 打平 batch |
| `v` | `[1, total_tokens, HV, V]` | attention output 的 value 布局 |
| `a` | `[1, total_tokens, HV, K]` | `is_kda=True` 时按 key 维 gate |
| `b` | `[1, total_tokens, HV]` | beta logits |
| `state` | `[slots, HV, V, K]` | committed SSM state，这里只读 |
| `state_indices` | `[N]` | 每条 request 的 state slot |
| `intermediate_states` | `[intermediate_slots, draft_token_num, HV, V, K]` | speculative state cache |
| `intermediate_indices` | `[N]` | 每条 request 的 intermediate state 行 |
| `cu_seqlens` | `[N + 1]` | request 边界 |
| `lower_bound` | float 或 `None` | Kimi K3 设为 `-5.0`，启用 safe gate |
| `retrieve_parent_token` | `[N, draft_token_num]` 或 `None` | tree verify 的 parent step |
| `replayssm_rawv` | `[slots, HV, L, V]` | ReplaySSM raw value ring；`cache_ring=True` 时必传 |
| `replayssm_rawk` | `[slots, H, L, K]` | ReplaySSM pre-norm key ring；`cache_ring=True` 时必传 |
| `replayssm_g` | `[slots, HV, L, K]` | KDA gate ring；`cache_ring=True` 时必传 |
| `replayssm_beta` | `[slots, HV, L]` | ReplaySSM beta ring；`cache_ring=True` 时必传 |

`cache_ring=True` 时，四个 `replayssm_*` tensor 必须一起传入，用于
ReplaySSM ring-write；`retrieve_parent_token` 只在 tree verify 时传入。

## 源码溯源

### vLLM/GDN 专属 decode

- wrapper：`boltops/fla/gdn/triton/fused_sigmoid_gating.py:251`
- configs：`boltops/fla/gdn/triton/configs/fused_sigmoid_gating_delta_rule_update/`

### SGLang GDN/KDA 共享 decode/verify

- wrapper 入口与参数校验：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:434`
- Triton kernel：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:109`
- safe gate 分支：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:289`
- tree parent state 读取：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:259`
- ReplaySSM ring write：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:313`
- intermediate state 写回：
  `boltops/fla/common/triton/fused_sigmoid_gating_recurrent.py:397`
- 第 9 节验收用例：
  `test/fla/test_kda_target_verify_acceptance.py:178`
- ReplaySSM ring/commit 用例：
  `test/fla/test_kda_target_verify_acceptance.py:270`
- 旧 FLA/torch development 用例：
  `test/fla/test_gdn_sigmoid_gating.py:283`

`gdn` facade 保持 GDN/vLLM decode ABI；GDN/KDA 共享 SGLang decode/verify
ABI 保留在 `common`。按算法家族和共享实现分文件，不引入额外 framework
compatibility 包。

## 正确性验收状态

已添加 SGLang Triton baseline 对比验收。baseline 来自 `/workspace/sglang`
commit `94602c9c2b`（tag `v0.5.20`）。对比误差阈值为 `atol=3e-2,
rtol=2e-2`。

```bash
pytest test/fla/test_kda_target_verify_acceptance.py -q
```

结果：**34 passed**。

在把 vLLM/GDN 专属 decode kernel 移入 `gdn` 后，已复跑共享 verify 验收和
GDN sigmoid gating 套件：

```bash
pytest test/fla/test_gdn_sigmoid_gating.py test/fla/test_kda_target_verify_acceptance.py -q
```

结果：**45 passed, 6 skipped**；skip 项是默认关闭的 benchmark。

已按需求第 9 节覆盖：

| 需求项 | 覆盖 |
|---|---|
| SGLang Triton baseline 对比 | output、intermediate state、committed state |
| `draft_token_num = 1/2/4/8` | 全部覆盖 |
| 单/多 request | `N=1` 和 `N=4` |
| 等长/非等长、ragged/ghost | 等长、非等长长度和 ghost padding |
| chain verify | 全部场景 |
| tree verify | 多个 parent step，BF16/FP16 |
| `lower_bound=None`/非空 | softplus 和 `-5.0` safe gate |
| 非零/非连续 cache index | state `[3,0,5,2]`，intermediate `[0,1,3,2]` |
| `disable_state_update=True` | 两次 baseline/Bolt committed state 位相等 |
| intermediate 写入位置 | 请求行映射 + draft step |
| ReplaySSM ring | rawv/rawk/g/beta baseline 对比 |
| ring commit | ring replay/fold 与 eager intermediate state 对比 |
| FP16/BF16 | 两类输入，FP32 accumulation |
| 多次 verify + accept/reject commit | 两轮 verify，accepted prefix 提交后再对比 |

结论：**需求第 9 节算子级 correctness 验收已通过**。端到端 speculative
decode、TTFT/TPOT 和完整 tuning 仍属于第 8/11 节交付，不在本项 correctness
验收范围内。

## 性能证据

一次内部探索性实测，还不是需求要求的完整 tuning 报告或端到端性能报告。

- GPU：Hygon BW200 / `gfx936`
- Shape：Kimi K3 TP=8 local，`H=HV=12`，`K=V=128`
- Request：`N=48`，varlen 等长切分
- 输入：BF16；state/intermediate cache：FP32
- 模式：KDA、`lower_bound=-5.0`、chain intermediate cache、不回写 committed state
- 计时：`triton.testing.do_bench(warmup=10, rep=50)`

| `draft_token_num` | latency | logical traffic | bandwidth |
|---:|---:|---:|---:|
| 1 | 72.24 us | 76.237 MiB | 1055.29 GB/s |
| 2 | 110.80 us | 114.725 MiB | 1035.45 GB/s |
| 4 | 187.28 us | 191.702 MiB | 1023.60 GB/s |
| 8 | 344.31 us | 345.655 MiB | 1003.90 GB/s |

`logical traffic` 统计 q/k/v/a/b/output 读写、一次 committed state 读和每个
draft step 的 intermediate state 写，不含框架侧 commit/replay。correctness 已
与 SGLang Triton baseline 对齐；当前性能证据只覆盖 TP=8 和 chain verify，
还缺与 SGLang baseline 的性能对比、TP=1/2/4/8 完整矩阵、
register/spill/occupancy，以及 TTFT/TPOT 端到端数据。
