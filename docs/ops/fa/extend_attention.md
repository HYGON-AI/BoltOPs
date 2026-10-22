# ExtendAttention 使用说明

ExtendAttention 对本轮新增 query 计算 attention，KV 由历史 prefix cache 和本轮
新增 K/V 组成。支持 packed varlen、MHA/GQA/MQA、滑动窗口、attention sinks，
并提供 unified KV、dense-prefill 及其四维 `[B,H,S,D]` 入口。

相关文档：

- [Decode Attention](decode_attention.md)：单 query、FP8 KV 的四维 decode 入口

## 1. 算子用途与数学语义

对单条请求，设 prefix 长度为 `P`、本轮新增长度为 `Q`：

```text
K = concat(k_buffer[kv_indices], k_extend)
V = concat(v_buffer[kv_indices], v_extend)
S = (q_extend @ K^T) * sm_scale
O = softmax(mask(S)) @ V
```

基础入口的 `k_scale/v_scale` 只作用于 prefix cache；unified 入口对统一 cache
中的全部 K/V 应用 scale；dense-prefill 入口对完整物化的全部 K/V 应用 scale。
`is_causal=True` 时，第 `i` 个新增 query 可见全部 prefix 和本轮第 `0..i` 个
K/V。GQA/MQA 要求 `Hq % Hkv == 0`；attention sink 只参与 softmax 分母，不贡献
value。

HCU 上会按 query 长度自动选择 short-query 或 long-query 优化 kernel；不满足
优化路径条件的功能组合自动使用通用 kernel。

## 2. 接口说明

```python
from boltops.fa.extend_attention import (
    extend_attention_fwd,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    dense_prefill_attention_fwd,
    dense_prefill_attention_fwd_4d,
    dense_prefill_attention_fwd_segmented,
    can_use_dense_prefill_fp8,
)
```

| 接口 | 作用 |
|---|---|
| `extend_attention_fwd` | 分别读取 prefix cache 与新增 K/V，支持 LSE 和分阶段计算 |
| `build_unified_kv_indices` | 合并 prefix/current KV 索引 |
| `extend_attention_fwd_unified` | 从统一 KV cache 执行 attention |
| `dense_prefill_attention_fwd` | 对已完整物化的 K/V 执行 dense varlen prefill |
| `dense_prefill_attention_fwd_4d` | 对 `[B,H,S,D]`、batch 内各请求等长的 Q/K/V 执行 dense prefill，返回新分配的输出 |
| `dense_prefill_attention_fwd_segmented` | 四维分段 Q，各段指定有效 KV 前缀；详见 [分段 API](#24-四维分段-dense-prefill-入口) |
| `can_use_dense_prefill_fp8` | 检查 dense-prefill FP8 路径的设备、dtype、causal 和 logit-cap 条件；其他限制由调用方预先排除 |

基础接口签名如下：

```python
def extend_attention_fwd(
    q_extend, k_extend, v_extend, o_extend,
    k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, is_causal, mask_indptr, max_len_extend,
    k_scale, v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    page_size=1,
    score_mod=None,
    aux_tensors=None,
    extend_seq_lens_cpu=None,
    identity_kv_indices=False,
):
    ...
```

示例：

```python
import torch
from boltops.fa.extend_attention import extend_attention_fwd

q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 2, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
k_cache = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
out = torch.empty_like(q)

extend_attention_fwd(
    q, k, v, out, k_cache, v_cache,
    qo_indptr=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
    kv_indptr=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
    kv_indices=torch.tensor([5, 1, 7], device="cuda", dtype=torch.int32),
    custom_mask=None,
    is_causal=True,
    mask_indptr=None,
    max_len_extend=2,
    k_scale=1.0,
    v_scale=1.0,
)
```

### 2.1 主要参数

记 `B` 为请求数、`T` 为新增 token 总数、`N` 为 cache slot 数。

| 参数 | Shape / 类型 | 含义 |
|---|---|---|
| `q_extend` | `[T,Hq,Dqk]` | 新增 query |
| `k_extend/v_extend` | `[T,Hkv,Dqk/Dv]` | 新增 K/V |
| `o_extend` | `[T,Hq,Dv]` | 调用方分配的输出 |
| `k_buffer/v_buffer` | `[N,Hkv,D]` 或 paged 4-D | prefix KV cache |
| `qo_indptr/kv_indptr` | `[B+1]` | 每条请求的累积边界 |
| `kv_indices` | `[sum(prefix_lens)]` | prefix 的物理 slot 索引 |
| `custom_mask/mask_indptr` | optional | 展平的逐请求 mask 及边界 |
| `sliding_window_size` | int | 大于零时启用滑动窗口 |
| `sinks` | `[Hq]`，optional | attention sink logits |
| `lse_extend` | `[T,Hq]`，optional | FP32 LSE 输出 |
| `page_size` | int | KV cache 每页 token 数，默认 1 |

### 2.2 前置条件与限制

- Q/K/V/O 和 KV cache 的最后一维须连续，张量位于同一 CUDA/HIP device。
- `k_scale/v_scale` 必须传数值；非量化 cache 使用 `1.0`。
- paged KV buffer 的第二维须等于 `page_size`，`kv_indices` 仍表示物理 token。
- `skip_prefix` 与 `skip_extend` 不能同时为真。
- unified 路径要求调用方先写入当前 K/V；index builder 只合并索引，不搬运数据。
- dense-prefill 路径不支持 custom mask、sinks、滑窗或 score modifier。
- `dense_prefill_attention_fwd` 的 kernel 以 int32 计算元素偏移，packed K/V 的
  `总 token 数 × token stride` 须小于 `2^31`。例如 `Hkv=64、Dqk=192` 时，K 的总
  token 数须小于 174763；更长的 batch 请拆分调用或改用四维入口。
- `can_use_dense_prefill_fp8` 不检查 custom mask、sinks、滑窗或 score modifier；调用方
  必须先排除这些组合。该函数只检查 gfx95、环境开关、BF16 Q/K/V、causal 和
  `logit_cap` 条件。
- `identity_kv_indices=True` 是调用方承诺，算子不会核验索引内容。
- 本包只提供 forward 算子，不负责 KV pool、SGLang backend 注册或服务端调度。

### 2.3 四维 dense-prefill 入口

```python
def dense_prefill_attention_fwd_4d(q, k, v, *, sm_scale=None, is_causal=True):
    ...
```

| 参数 | Shape / 类型 | 含义 |
|---|---|---|
| `q` | `[B,H,Sq,192]`，BF16 | 每条请求最后 `Sq` 个 token 的 query |
| `k` | `[B,H,Sk,192]`，BF16 | 每条请求完整、按时间顺序排列的 key |
| `v` | `[B,H,Sk,128]`，BF16 | 与 K 对应的 value |
| `sm_scale` | `None` 或有限正数 | `None` 使用 `1/sqrt(192)` |
| `is_causal` | bool | 是否使用右下对齐的 causal mask |
| 返回值 | `[B,H,Sq,128]`，BF16 | 新分配的连续输出 |

`is_causal=True` 时 causal mask 右下对齐：`q` 是完整 K/V 的最后 `Sq` 个 token，
令 `P = Sk - Sq`，第 `i` 个 query 可见 key `0..P+i`。算子无法从 shape 核验这一
位置关系，由调用方保证。

```python
import math
import torch
from boltops.fa import dense_prefill_attention_fwd_4d

q = torch.randn(4, 64, 6656, 192, device="cuda", dtype=torch.bfloat16)
k = torch.randn(4, 64, 13312, 192, device="cuda", dtype=torch.bfloat16)
v = torch.randn(4, 64, 13312, 128, device="cuda", dtype=torch.bfloat16)

# Kimi-K2.6：1/sqrt(192) 乘 YaRN mscale^2（factor=64，mscale_all_dim=1）
sm_scale = 192**-0.5 * (0.1 * math.log(64.0) + 1.0) ** 2
out = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=sm_scale)  # [4,64,6656,128]
```

前置条件与限制：

- 仅支持 BF16、MHA（`Hq == Hkv`）、`Dqk=192/Dv=128`；batch 内各请求等长、
  无 padding，且 `Sk >= Sq`（`is_causal=False` 时同样要求）。其他组合直接报错。
- Q/K/V 按各自 stride 寻址，只要求最后一维连续；最后一维不连续的输入会先复制。
- 单条请求内任一张量的元素跨度须小于 `2^31`（`H=64`、`Dqk=192` 时 `Sk` 约
  17.4 万），batch 维不计入这一上限。
- `sm_scale=None` 只是通用默认值；模型带 YaRN 等 scale 修正时须显式传入。
- 只做前向，输入需要梯度时报错；不检查 NaN/Inf。

执行方式按 shape 自动选择，两者结果在误差范围内一致：

| 方式 | 条件 | 说明 |
|---|---|---|
| 逐请求 | 默认 | 每条请求启动一次 kernel，传入 `[S,H,D]` 转置视图，输出由 kernel 直接写入 |
| 打包 | `B > 1`，每条请求 `H * ceil(Sq/64) * Sk <= 2^15`，且打包后实际元素跨度小于 `2^31` | 整个 batch 转为 packed 布局后只启动一次 kernel，再复制回四维输出；用于 host 开销占主导的小请求 |

打包阈值按 HCU 实测选取，换设备后需重新评估。

### 2.4 四维分段 dense-prefill 入口

```python
def dense_prefill_attention_fwd_segmented(
    q, k, v, *, q_segment_lens, kv_prefix_lens, sm_scale,
):
    ...
```

对每条请求的两段 Q 分别执行 causal attention，各段读取自己的 KV 前缀，输出与
本地 Q 顺序一致。适用于上游已完成 zigzag 等 context parallel 位置分配和 KV
收集的输入。

| 参数 | Shape / 类型 | 含义 |
|---|---|---|
| `q` | `[B,H,Slocal_max,192]`，BF16 | 每条请求按前段、后段、尾部 padding 排列 |
| `k` | `[B,H,Sk_max,192]`，BF16 | 从全局位置 0 开始按原序排列 |
| `v` | `[B,H,Sk_max,128]`，BF16 | 与 K 的 batch/head/token 一致 |
| `q_segment_lens` | `[B,2]`，host 整数序列或 CPU 整型 tensor | 两段 Q 的长度 `L` |
| `kv_prefix_lens` | `[B,2]`，host 整数序列或 CPU 整型 tensor | 两段 KV 前缀的右端 `E`，包含历史缓存 |
| `sm_scale` | 有限正数，必填 | softmax scale，按模型配置传入 |
| 返回值 | `[B,H,Slocal_max,128]`，BF16 | 新分配的连续输出；有效行保持 Q 顺序，尾部 padding 置零 |

一段 Q 长度为 `L`、KV 前缀右端为 `E` 时，该段对应全局位置 `[E-L, E)`，段内
第 `i` 行可见 key `0..E-L+i`。接口校验长度边界，但无法从数值核验 Q 来自哪些
token，由调用方保证数据与元数据一致。

```python
import torch
from boltops.fa import dense_prefill_attention_fwd_segmented

q = torch.randn(2, 4, 7, 192, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 4, 23, 192, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 4, 23, 128, device="cuda", dtype=torch.bfloat16)

out = dense_prefill_attention_fwd_segmented(
    q, k, v,
    q_segment_lens=[[2, 3], [0, 4]],
    kv_prefix_lens=[[5, 19], [0, 11]],
    sm_scale=192**-0.5,
)
# 请求 0：Q[:2] 对应位置 [3,5)，Q[2:5] 对应 [16,19)，out[0,:,5:] 为零。
# 请求 1：前段为空，Q[:4] 对应位置 [7,11)，out[1,:,4:] 为零。
```

以 CP8 为例：本轮新增 53248 个 token、无历史前缀，等分为 16 块后，rank 1 的
两段 Q 长度均为 3328，`kv_prefix_lens` 为 `[6656, 49920]`，卡上完整 KV 长度
为 53248。

前置条件与限制：

- 每段满足 `0 <= L <= E <= Sk_max`，两段 `L` 之和不超过 `Slocal_max`；两段都
  非空时须有序且不重叠，允许相邻。空段跳过计算，两段都为空时该请求输出全零。
- 元数据须位于 host，全部校验通过后才启动 kernel。
- 仅支持 BF16、MHA、`Dqk=192/Dv=128`，只做前向；不支持 custom mask、滑窗、
  FP8/分页 KV 和 CUDA Graph capture。
- K/V 只读，支持 batch stride 为 0 的共享 KV 视图；stride 与元素跨度要求同 2.3。
- 内部按请求、按段调用 2.3 入口（`is_causal=True`），各段结果直接写回，无需
  LSE 合并。每条请求最多两次 launch，未针对性能优化。
- 不负责 rank 分配、RoPE、KV all-gather 或模型集成；SGLang packed token 格式
  需由调用方转换为四维格式。

## 3. 正确性测试与性能基准

在仓库根目录运行：

```bash
# 正确性测试
PYTHONPATH=. python3 -m pytest -q test/fa/test_extend_attention.py

# 单独运行 128K 偏移边界用例（默认测试已包含，约需 11 GiB 显存）
PYTHONPATH=. python3 -m pytest -q \
  test/fa/test_extend_attention.py::test_dense_prefill_4d_long_context_past_packed_int32_offsets

# 正确性及性能用例（含网络 shape 用例）
PYTHONPATH=. python3 -m pytest -q -s test/fa/test_extend_attention.py --bench

# 查看调优 case
boltops-extend-attention-tune-cli --list-cases

# 调优当前设备；gfx938/gfx92a 生成 BF16/FP8 JSON，其他架构只生成 BF16
boltops-extend-attention-tune-cli \
  --kv-cache-dtype both \
  --warmup-ms 5 \
  --rep-ms 20
```

通用 benchmark 包含 8 个固定 BF16 case，网络 shape 用例见 [3.1](#31-网络-shape-用例)。
调优集合额外覆盖代表性的 long-query、
short-query case、benchmark key，以及 gfx938 CU64/CU72 携带的 FP8 JSON key 并集。
FP8 KV cache 只在 gfx938/gfx92a 上启用；gfx936 仅发布 BF16 配置。
每种设备应分别调优，不能共用配置。

| KV Cache dtype | long-query key | short-query key | 合计 |
|---|---:|---:|---:|
| BF16 | 8 | 8 | 16 |
| FP8 E4M3FN（仅 gfx938/gfx92a） | 8 | 31 | 39 |
| 合计（支持 FP8 的设备） | 16 | 39 | 55 |

支持 FP8 的设备完整运行结束时日志应包含：

```text
SUMMARY tuned=55 duplicate_keys=0 inapplicable_dtype_jobs=39
```

在 gfx936 上使用 `--kv-cache-dtype both` 时只调优 BF16，共 16 个配置；显式指定
`--kv-cache-dtype fp8` 会报错。

### 3.1 网络 shape 用例

[_fa_bench_cases.py](../../../test/fa/_fa_bench_cases.py) 收录 Kimi-K2.6 128K
部署中的 24 组 attention shape，按路径、布局和 CP rank 展开为 49 个用例，覆盖
dense、segmented（CP8）和 Decode。输入构造、FP32 reference 和显存预检
由 [_fa_bench_utils.py](../../../test/fa/_fa_bench_utils.py) 共用。

- **正确性**：`test_original_shape_prefill` 运行 dense 和 CP8 segmented 用例，
  检查全输出有限性，并对抽样 query 在完整 KV 上做 FP32 对照。Decode 用例见
  [Decode Attention](decode_attention.md#4-正确性测试与性能基准)。
- **CP8 用例**：B=32，Q/K/V 净容量 73–164 GiB，超出 CI 机器能力，因此标记为
  `slow`。正确性和性能用例都只在传入 `--runslow` 时运行，CI 默认跳过。
- **显存**：只按当前卡总显存预检。Q/K/V 和输出的最低净容量超过总显存时 skip；
  构造输入、首次调用算子或计算 reference 时发生 OOM，也按资源不足 skip，
  其他错误正常失败。不拆分 batch。
- **性能**：`test_bench_fa_prefill` 覆盖 dense 和 segmented，计时和采集
  使用[公共性能框架](../../optest_perf.md)的 `time_ms`。结果写入
  `extend_attention.csv`，Decode 写入 `fa_decode_4d.csv`，不同路径按
  `case_id` / `path` 区分。

| 环境变量 | 作用 |
|---|---|
| `FA_BENCH_SET=full` | 性能用例运行全部 49 个；默认只运行 13 个代表用例 |
| `FA_BENCH_CASES=<case_id>[,...]` | 只运行指定用例，例如 `DENSE-S00,DECODE-P64` |

```bash
# 只运行网络 shape 的 Prefill 正确性用例
PYTHONPATH=. python3 -m pytest -q -s test/fa/test_extend_attention.py \
  -k test_original_shape_prefill

# 同上，并包含 CP8 segmented 用例
PYTHONPATH=. python3 -m pytest -q -s test/fa/test_extend_attention.py \
  -k test_original_shape_prefill --runslow

# 网络 shape 性能用例，公共执行器输出标准 CSV 和 raw 记录
# （PYTEST_ADDOPTS 中加入 --runslow 可包含 CP8 用例）
PYTHONPATH=. PYTEST_ADDOPTS='-k test_bench_fa_' \
  python3 scripts/run_operator_benchmarks.py --ops extend_attention fa_decode_4d
```

计时参数可通过 `PYTEST_ADDOPTS` 传入 `--bench-warmup` / `--bench-repeat`。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/fa/extend_attention/__init__.py` |
| Kernel 与四维适配层 | `boltops/fa/extend_attention/triton/extend_attention.py` |
| 优化 kernel | `boltops/fa/extend_attention/triton/_optimized_kernels.py` |
| 配置加载 | `boltops/fa/extend_attention/triton/config.py` |
| JSON 配置 | `boltops/fa/extend_attention/triton/configs/triton_<capability>/` |
| 调优工具 | `boltops/tools/extend_attention_triton_tune/` |
| 测试 | `test/fa/test_extend_attention.py` |
| 网络 shape 用例与测试工具 | `test/fa/_fa_bench_cases.py`、`test/fa/_fa_bench_utils.py` |

## 附录 B. 溯源

公共接口由 SGLang `extend_attention.py` 迁入，参考提交为
`ee0f2375e5fb285e4fb021a9831fa88018244679`。运行时不依赖 SGLang。

分段入口的元数据和调度约定参照 SGLang ZigzagCPStrategy，参考提交为
[`d0a75654368a470d8484e3ea45b06e500fee57af`](https://github.com/sgl-project/sglang/blob/d0a75654368a470d8484e3ea45b06e500fee57af/python/sglang/srt/layers/cp/zigzag.py)
（`build_metadata` / `run_attention`）。运行时不依赖 SGLang。

## 附录 C. 实现备注

- `max_len_extend < 32` 使用 short-query kernel，其余使用 long-query kernel。
- 配置按设备、CU 数、dtype 和 shape key 匹配；精确 grid 未命中时选择同 shape
  的最近配置，无兼容 shape 时使用默认配置。
- `BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION=0` 可关闭 compact grid。
- `BOLT_OPS_TRITON_FP8_PREFILL_ATTN=0` 可关闭 FP8 prefill 路径。
