# UnifiedAttention 使用说明

UnifiedAttention 对 packed varlen query 和 paged KV cache（KV block/page）计算 attention。它支持
prefill/decode、MHA/GQA/MQA、2D 与 3D split-KV、滑动窗口、attention sinks、
ALiBi、多模态 prefix，以及多种 KV cache 量化模式。

本包提供独立的底层算子，不包含 SGLang/vLLM 的 layer、KV cache manager、slot
mapping、CUDA/HIP Graph 或 runtime backend 注册。

## 1. 算子用途与数学语义

对 batch 中第 `s` 条请求，设 query 长度为 `Q_s`、有效 KV 长度为 `L_s`、cache
KV block 大小为 `P`。逻辑位置 `j` 对应的物理 KV 地址为：

```text
block = block_table[s, j // P]
slot = j % P
K_s[j] = k[block, slot]
V_s[j] = v[block, slot]

S_s = (Q_s @ K_s^T) * softmax_scale
O_s = softmax(mask(S_s)) @ V_s
```

query 使用 packed 布局，`cu_seqlens_q` 给出各请求在 `q/out` 中的边界；
`seqused_k` 给出每条请求的实际 KV 长度。causal 模式下，第 `i` 个 query 的绝对
位置为 `L_s - Q_s + i`，只能看到不晚于该位置的 K/V。GQA/MQA 要求
`Hq % Hkv == 0`。

attention sink 只加入 softmax 分母，不贡献 value。普通路径直接由 2D kernel 写入
`out`；小 batch、单 token decode 可使用 3D split-KV kernel 生成分段 partial，
再由 reduce kernel 合并。

## 2. 接口说明

```python
from boltops.fa.unified_attention import (
    KVQuantMode,
    reshape_and_cache_int4,
    unified_attention,
)
```

| 接口 | 作用 |
|---|---|
| `unified_attention` | 从 paged KV cache 执行 varlen attention，结果原地写入 `out` |
| `reshape_and_cache_int4` | 对 K/V 执行 RHT、INT4 量化和打包，并写入 paged cache |
| `KVQuantMode` (Quant type) | 指定 KV cache 的量化与 scale 解释方式 |

主接口保持 vLLM 兼容的 35 参数 ABI：

```python
def unified_attention(
    q, k, v, out,
    cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale,
    seq_threshold_3D=None, num_par_softmax_segments=None,
    softmax_segm_output=None, softmax_segm_max=None,
    softmax_segm_expsum=None, alibi_slopes=None, output_scale=None,
    qq_bias=None, sinks=None, mm_prefix_range=None,
    rswa_prefix_lens=None, rswa_window=None, use_alibi_sqrt=False,
    kv_quant_mode=None, k_scale_cache=None, v_scale_cache=None,
    chunk_lookback=-1, use_td=False,
    mm_prefix_clamp_sliding_window=False,
) -> None:
    ...
```

建议使用关键字参数调用扩展参数，避免混淆 `mm_prefix_range` 之后的 vLLM/AITER
兼容参数位置。

示例：单请求、单页、BF16 causal decode。

```python
import torch
from boltops.fa.unified_attention import unified_attention

device = "cuda"
page_size, kv_len = 16, 13
num_query_heads, num_kv_heads, head_size = 4, 2, 32

q = torch.randn(
    1, num_query_heads, head_size, device=device, dtype=torch.bfloat16
)
k_cache = torch.randn(
    1, page_size, num_kv_heads, head_size,
    device=device, dtype=torch.bfloat16,
)
v_cache = torch.randn_like(k_cache)
out = torch.empty_like(q)

unified_attention(
    q=q,
    k=k_cache,
    v=v_cache,
    out=out,
    cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
    max_seqlen_q=1,
    seqused_k=torch.tensor([kv_len], device=device, dtype=torch.int64),
    max_seqlen_k=kv_len,
    softmax_scale=head_size**-0.5,
    causal=True,
    window_size=(-1, -1),
    block_table=torch.tensor([[0]], device=device, dtype=torch.int32),
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
)
```

### 2.1 主要参数

记 `B` 为请求数、`Tq` 为 packed query token 总数、`NB` 为物理 KV block 数、
`BS` 为 KV block 大小。

| 参数 | Shape / 类型 | 含义 |
|---|---|---|
| `q/out` | `[Tq,Hq,D]` | packed query 与调用方分配的输出 |
| `k/v` | `[NB,BS,Hkv,D]` | paged KV cache；量化模式下 dtype/layout 依模式变化 |
| `cu_seqlens_q` | `[B+1]`，整数 | query 的累积边界 |
| `seqused_k` | `[B]`，整数 | 每条请求的实际 KV 长度 |
| `max_seqlen_q/k` | int | 当前 batch 的 query/KV 长度上界 |
| `block_table` | `[B,max_blocks]`，整数 | 逻辑 KV block 到物理 KV block 的映射 |
| `causal` | bool 或 `[B]` bool tensor | 全 batch 或逐请求 causal 控制 |
| `window_size` | `(left,right)` | `(-1,-1)` 关闭滑窗；左窗 `W` 传 `(W-1,0)` |
| `softcap` | float | 大于零时对 attention logit 执行 soft cap |
| `alibi_slopes` | `[Hq]`，optional | ALiBi slope；`use_alibi_sqrt` 选择 sqrt 变体 |
| `qq_bias` | `[Q,Q]`，optional | query-query 区域的额外 bias |
| `sinks` | `[Hq]`，optional | attention sink logits |
| `mm_prefix_range` | `[B,R,2]`，optional | 每条请求的多模态双向 prefix 范围 |
| `k_descale/v_descale` | scalar/tensor，optional | per-tensor cache 反量化 scale |
| `output_scale` | scalar tensor，optional | FP8 output 的量化 scale |
| `rswa_prefix_lens/rswa_window` | `[B]` / int，optional | R-SWA：prefix 全局可见，其余 token 使用固定窗口 |
| `chunk_lookback` | int | 非负时启用 chunk-local attention；依赖有效滑窗 |
| `use_td` | bool | 启用 tensor descriptor 读写路径，默认关闭 |

### 2.2 2D/3D 路径

3D 路径只在下列条件同时满足时使用：

- `num_seqs <= seq_threshold_3D`；
- `num_par_softmax_segments` 和三个 scratch tensor 均已提供；
- decode（`max_seqlen_q == 1`），或短 prefill（`1 < max_seqlen_q <= 512`，
  且 scratch 首维不少于 query token 总数，且该 shape 在 3D JSON 中**精确命中**
  已有配置）。

短 prefill 走 3D split-KV 是显式 opt-in：运行时 probe 只做精确 shape 匹配
（不做最近 shape 回退），没有配置的 prefill shape 一律留在 2D——decode 的默认
3D 配置对小 prefill 明显偏慢，不能作为回退。decode 路径的最近 shape 回退
行为不变。

scratch 布局如下，其中 `Dp` 为 head layout 宽度（`head_size_layout`：
可拆成两个 2 幂之和的非 2 幂 head（如 `192 = 128 + 64`）取 `D` 本身，
2 幂与不可拆分的 head 取 `next_power_of_2(D)`）：

| 参数 | Shape / dtype |
|---|---|
| `softmax_segm_output` | `[rows,Hq,segments,Dp]`, FP32 |
| `softmax_segm_max` | `[rows,Hq,segments]`, FP32 |
| `softmax_segm_expsum` | `[rows,Hq,segments]`, FP32 |

`rows` 按全局 query token 下标寻址：decode 批取 `seq_threshold_3D`
（≥ batch 即可），短 prefill 批必须 ≥ query token 总数。

其他情况走 2D 路径。3D 配置除 attention kernel 的参数外，还单独保存 reduce kernel
的 `reduce_num_warps/reduce_num_stages`。

### 2.3 KV cache 量化

| 模式 | cache/scale 约定 |
|---|---|
| `NONE` | 普通 BF16/FP16 cache |
| `FP8_PER_TENSOR` | FP8 cache，使用 `k_descale/v_descale` |
| `INT8_PER_TOKEN_HEAD` | INT8 cache，并提供 `[NB,BS,Hkv]` 的 K/V scale cache |
| `FP8_PER_TOKEN_HEAD` | FP8 cache，并提供 `[NB,BS,Hkv]` 的 K/V scale cache |
| `INT4_PER_TOKEN_HEAD` | `uint8` packed cache，必须先用 `reshape_and_cache_int4` 写入 |

`KVQuantMode.NVFP4` 当前仅保留枚举值；传入该模式不会启用 NVFP4 量化实现，
正式使用前必须等待对应 kernel 和 scale 语义落地。当前它只是可枚举值，不代表
对应量化路径已经实现或经过验证。

### 2.4 前置条件与限制

- Q、KV cache、输出及 metadata 必须位于同一 CUDA/HIP device。
- `Hq` 必须能被 `Hkv` 整除；`block_table` 必须覆盖每条请求所需的全部 KV block。
- `sinks` 长度必须等于 `Hq`；`mm_prefix_range` 必须为 3-D。
- per-token-head 量化必须同时提供 `k_scale_cache` 和 `v_scale_cache`。
- INT4 路径仅支持统一 causal 模式；head size 必须为 2 的幂并可被 packing
  factor 整除。
- `chunk_lookback >= 0` 时必须提供有效 sliding window，且计算出的 chunk size
  必须大于零。
- `use_td=True` 要求 cache block size 可被 tile size 整除；Q/O descriptor 路径还
  要求 head 维连续。
- 算子只读取已经写好的 paged KV cache；除 `reshape_and_cache_int4` 外，不负责
  cache 分配、普通 KV 写入或 slot mapping。
- 本包只提供 forward 算子，返回 `None`，结果写入调用方提供的 `out`。

## 3. 正确性测试、性能基准与调优

在仓库根目录运行：

```bash
# 正确性与配置加载测试；默认跳过 benchmark
PYTHONPATH=. python3 -m pytest -q test/fa/test_unified_attention.py

# 运行 6 个固定 BF16 benchmark case
PYTHONPATH=. python3 -m pytest -q -s \
  test/fa/test_unified_attention.py --bench

# 查看调优 case
PYTHONPATH=. python3 -m \
  boltops.tools.unified_attention_triton_tune.tune_unified_attention \
  --list-cases

# 调优当前设备的全部 enabled case
PYTHONPATH=. python3 -m \
  boltops.tools.unified_attention_triton_tune.tune_unified_attention \
  --warmup-ms 25 \
  --rep-ms 100

# 只调一个 case
PYTHONPATH=. python3 -m \
  boltops.tools.unified_attention_triton_tune.tune_unified_attention \
  --case prefill-mqa-multibatch
```

benchmark 与调优都在 **CUDA graph replay 模式**下计时（`time_ms` fixture
graph 优先、调优器调 `do_bench_cudagraph`），与生产 graph 运行模式一致；
capture 失败自动回退 event 计时并告警，CSV `timing` 列按行记录实际所用
方法。两种方法的差异来自 L2 状态（event 每次迭代 flush L2，graph 不
flush）：2026-10-15 同机实测 decode 在 event 下慢约 1.3–1.4x、
prefill-mqa 约 1.08x、multibatch 约 1.01x（L2 fabric 已打满，几乎一致）。
本文 3.1/3.2 的 A/B 表为 event 计时数据，方法内自洽可比。

kernel 配置是**调优产物**：调优器（`tune_unified_attention.py`）先用独立
PyTorch reference 检查每个候选，再测量默认配置和有效候选，最后把最优写入
当前 Triton capability 对应的 JSON（已有 key 增量合并，不覆盖其他 shape）。
个别配置（如 prefill 3D 的 `nseg=2` 组合）由离线 A/B 实测后手工补进 JSON。
`benchmark_cases.yaml` 固定了 6 个 shape，既是 benchmark 用例也是配置 JSON
的 key 来源。

固定 case（`benchmark_cases.yaml`）是 benchmark、调优与配置 key 的唯一 shape 来源：

| Case | 路径 | Query/KV 长度 | Hq/Hkv | Block | SW / sinks |
|---|---|---|---:|---:|---|
| `prefill-gqa-sink-swa` | 2D | `49/4096` | `16/2` | 16 | `128 / yes` |
| `prefill-mqa` | 3D（有配置；无配置设备回退 2D） | `49/4096` | `16/1` | 32 | `off / no` |
| `prefill-gqa-sink-swa-multibatch` | 2D | `3,49,511 / 4096` | `16/2` | 16 | `128 / yes` |
| `prefill-mqa-multibatch` | 3D（nseg=2，有配置；无配置设备回退 2D） | `3,49,511 / 4096` | `16/1` | 32 | `off / no` |
| `decode-gqa-sink-swa-3d` | 3D | `1/4096` | `16/2` | 16 | `128 / yes` |
| `decode-mqa-3d` | 3D | `1/4096` | `16/1` | 32 | `off / no` |

文件名包含架构、CU 数、block/head、功能开关和 KV dtype；JSON key 为：

```text
(num_seqs, num_query_tokens, max_seqlen_q, max_seqlen_k,
 num_queries_per_kv)
```

运行时优先精确命中；未命中时只在相同 `num_queries_per_kv` 布局中选择最近 shape；
没有兼容配置时回退安全默认值并输出一次 warning。例外：短 prefill 的 3D
probe 只接受精确命中（见 2.2）。不同架构和 CU 数需分别实测并补充对应 JSON，
不能直接共用。

### 3.1 Head 维拆分（192）与实测收益

6 个固定 case 全部是 `head_size=192` 的生产 GQA/MQA 布局。192 不是 2 的幂，
`tl.arange` 只接受 2 的幂宽度，旧实现把头维 pad 到 256——K/V 加载流量、
MMA 工作量、fp32 acc 寄存器全部虚耗 25%。现在 core kernel 按
`128 + 64` 拆分执行（机制见附录 C）：

- QK 侧 `S = scale * (dot(Q1,K1) + dot(Q2,K2))`，PV 侧同理，acc 分半累加；
- K/V 加载、MMA、acc 寄存器、LDS 各省 25%，并解锁更宽 tile 的合法配置
  （如 2D `TILE=64` 双缓冲）；
- 改动与架构正交：两个 dot 的宽度都是 2 的幂，任何 CDNA 设备同形成立，
  gfx946（BW200 系）同样受益，无需代码分叉——在 gfx946 机器上做 A/B 实测后
  补充该设备的 JSON（文件名按架构/CU 数区分）。

gfx938（72 CU）实测（BF16，event 计时、每次迭代 flush L2，A/B 历史方法；
bench/CI perf 现用 graph 计时，见 §3；旧实现 = 拆分前 kernel + 旧配置，
A/B 同窗口）：

| Case | 旧（256 pad）µs | 新（192 拆分）µs | 加速 |
|---|---:|---:|---:|
| `prefill-gqa-sink-swa` | 30.4 | 27.5 | 1.11x |
| `prefill-mqa` | 320.3 | 258.9 | 1.24x |
| `prefill-gqa-sink-swa-multibatch` | 52.3 | 47.5 | 1.10x |
| `prefill-mqa-multibatch` | 692.8 | 552.3 | 1.25x |
| `decode-gqa-sink-swa-3d` | 30.2 | 28.0 | 1.08x |
| `decode-mqa-3d` | 36.0 | 34.2 | 1.05x |

A/B 测量在 ~30µs 级的小 kernel 上存在噪声：候选间 3–5% 的差距可能随时
计窗口翻转，落配置前应在目标计时条件下对 top 候选做 A/B 复核（本次
`decode-gqa-sink-swa-3d` 即由 A/B 保留旧配置而非当轮测量首选）。

### 3.2 短 prefill 的 3D split-KV（gfx938 实测）

`prefill-mqa`（单序列 `q=49/kv=4096`，MQA，head 192）此前走 2D：
`BLOCK_Q=1` 后 grid 只有 50 个 block 摊在 72 个 CU 上，每 block 串行走
完整条 KV，259µs 里约 240µs 是暴露的 L2 往返延迟。xcu roofline 佐证：
MMA 利用率 5%、HBM 利用率 ~1%（整 kernel 只从 HBM 拉 3.75MB，KV 全部
L2 命中）——两个 roof 都远未达到，瓶颈是并行度不足（1 block/CU、
VGPR 限制 occupancy 20%），compiler 侧旋钮（`sanitize_overflow`、
`use_block_pingpong`、`kpack`、exp2 改写等）实测全部无效。

对策是把 decode 的 3D split-KV kernel 泛化到短 prefill（kernel 本身对
q 维通用，无需改动）：KV 切 16 段后 grid 变为 `50×1×16`，延迟被跨段
并行掩盖。配合 `BLOCK_M=32`（`BLOCK_Q=2`）把每个 q token 的 KV 重读
减半，gfx938_cu72 实测（event 计时，A/B 同窗口；bench/CI perf 现用
graph 计时，见 §3）：

| Case | 2D µs | 3D µs | 加速 |
|---|---:|---:|---:|
| `prefill-mqa` | 258.9 | 90.9 | 2.85x |

tuned 配置（`gfx938_cu72` 3D JSON，key `(1, 49, 49, 4096, 16)`）：
`BLOCK_M=32, TILE_SIZE=32, num_warps=2, num_stages=1,
reduce_num_warps=1, reduce_num_stages=1`。`num_segments=8` 时同配置为
83µs；segment 数由调用方传入，配置本身与 segment 数弱耦合。

其余 case 的实测结果：

- `prefill-mqa-multibatch`：与短 prefill 相反，这个 shape 的 2D 是
  **L2 fabric 带宽受限**——284 个 q block 各自把整条 3.1MB KV 从 L2
  重读一遍（634MB，L2-Util 114% 打满，xcu 实测），所以 2D 调 tile
  全平（552µs），而 3D 用 `nseg=16` 也没用：segm partial buffer 是
  per-(token, head, seg) 的 f32，16 段带来 ~112MB 的 partial 写+读，
  把 KV 重读省下的流量又加回去（实测 970µs）。正确组合是
  **`BLOCK_M=128`（8 token/组，KV 重读降 4 倍）+ `nseg=2`（partial
  只有 ~14MB，并行度由 73 组×2 段=146 block 提供）**：552.2 →
  505.9µs（1.09x）。case 定义相应改为 `num_segments: 2`、
  `seq_threshold_3d: 128`（原值 2 把 3 seqs 挡在 3D gate 外），
  tuned 配置写在 `seg=2` 的 3D JSON（key `(3, 563, 511, 4096, 16)`，
  `BLOCK_M=128, TILE_SIZE=16, num_warps=4, num_stages=1`）。
  注：2D 里 `BLOCK_M=64`（KV 重读减半）实测无收益（552µs 不变，
  block 数减半的延迟代价正好抵消 L2 收益）；瓶颈换回
  SIMD/依赖链（91% busy）后继续压缩属于 kernel 重写级别，未做。
- 两个 SWA case（27.5/47.2µs）：window=128 小于 segment 长度，16 段中
  绝大多数空转，实测 48µs 起，比 2D 差，保持 2D。

改动与架构正交：split-KV 是调度层变化，kernel 无架构专属代码，
gfx946 同样可用——在 gfx946 机器上 A/B 实测后补充对应的 3D JSON 即可
（无配置的设备自动回退 2D，无回归风险）。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/fa/unified_attention/__init__.py` |
| Core 2D/3D kernel | `boltops/fa/unified_attention/triton/unified_attention.py` |
| 通用与扩展功能 kernel | `boltops/fa/unified_attention/triton/general_attention.py` |
| INT4 cache 与 attention | `boltops/fa/unified_attention/triton/int4_per_token_head.py` |
| 配置加载 | `boltops/fa/unified_attention/triton/config.py` |
| JSON 配置 | `boltops/fa/unified_attention/triton/configs/triton_<capability>/` |
| 调优工具 | `boltops/tools/unified_attention_triton_tune/tune_unified_attention.py` |
| 基准 case 定义 | `boltops/tools/unified_attention_triton_tune/benchmark_cases.yaml` |
| 测试 | `test/fa/test_unified_attention.py` |

## 附录 B. 溯源

Core 2D/3D kernel、reduce 和基础配置选择参考 AITER 快照
`31fdae52e22caa2871bbe93ad0aa8c61ff2bfc25` 中的
`aiter/aiter/ops/triton/unified_attention.py`。完整 35 参数 ABI、R-SWA、chunk
lookback、KV quant 和 tensor descriptor 语义参考 vLLM 快照
`deeeae75d0e42199aaeb627ad66e38b28337d2bf` 中的 UnifiedAttention 实现。运行时
不依赖 AITER、SGLang 或 vLLM。

## 附录 C. 实现备注

- `causal is True` 且未启用 R-SWA、显式 KV quant scale、chunk lookback、TD 等扩展
  功能时优先使用 core kernel；其他组合使用 general kernel。
- core 路径支持 tuned JSON；general/INT4 路径当前使用各自的运行时启发式参数。
- core kernel 对非 2 幂 head（如 192）按 `head_size_layout` 拆成两个 2 幂之和
  （`192 = 128 + 64`）：所有 head 宽的 load / dot / store 拆成两半分别执行
  （QK 侧 `dot(Q1,K1) + dot(Q2,K2)`，PV 侧同理，acc 分半累加），不再 pad
  到 256——K/V 加载流量、MMA 工作量、fp32 acc 寄存器各省 25%，LDS 占用同步
  下降；3D scratch `softmax_segm_output` 的末维宽度同取 layout 宽度（192）。
  2 幂 head（64/128/256）保持原单 dot 布局不变，不可拆分的 head 回退
  2 幂 pad 的旧布局。
- Gemma3 的 `(head_size, sliding_window)=(128|256,1024)` 使用专门 tile 选择。
- `mm_prefix_clamp_sliding_window=True` 时，多模态 prefix 范围仍受滑窗约束；默认值
  `False` 保留普通 prefix 可见性。
- `--quick` 只验证一组默认候选，适合检查环境和输出，不等价于完整调优。
