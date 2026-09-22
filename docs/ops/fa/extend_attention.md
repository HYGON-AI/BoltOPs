# ExtendAttention 使用说明

ExtendAttention 对本轮新增 query 计算 attention，KV 由历史 prefix cache 和本轮
新增 K/V 组成。支持 packed varlen、MHA/GQA/MQA、滑动窗口、attention sinks，
并提供 unified KV 和 dense-prefill 入口。

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
    can_use_dense_prefill_fp8,
)
```

| 接口 | 作用 |
|---|---|
| `extend_attention_fwd` | 分别读取 prefix cache 与新增 K/V，支持 LSE 和分阶段计算 |
| `build_unified_kv_indices` | 合并 prefix/current KV 索引 |
| `extend_attention_fwd_unified` | 从统一 KV cache 执行 attention |
| `dense_prefill_attention_fwd` | 对已完整物化的 K/V 执行 dense varlen prefill |
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
- `can_use_dense_prefill_fp8` 不检查 custom mask、sinks、滑窗或 score modifier；调用方
  必须先排除这些组合。该函数只检查 gfx95、环境开关、BF16 Q/K/V、causal 和
  `logit_cap` 条件。
- `identity_kv_indices=True` 是调用方承诺，算子不会核验索引内容。
- 本包只提供 forward 算子，不负责 KV pool、SGLang backend 注册或服务端调度。

## 3. 正确性测试与性能基准

在仓库根目录运行：

```bash
# 正确性测试
PYTHONPATH=. python3 -m pytest -q test/fa/test_extend_attention.py

# 正确性测试和 8 个固定 BF16 benchmark case
PYTHONPATH=. python3 -m pytest -q -s test/fa/test_extend_attention.py --bench

# 查看调优 case
boltops-extend-attention-tune-cli --list-cases

# 调优当前设备；gfx938/gfx92a 生成 BF16/FP8 JSON，其他架构只生成 BF16
boltops-extend-attention-tune-cli \
  --kv-cache-dtype both \
  --warmup-ms 5 \
  --rep-ms 20
```

benchmark 包含 8 个固定 BF16 case。调优集合额外覆盖代表性的 long-query、
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

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/fa/extend_attention/__init__.py` |
| Kernel | `boltops/fa/extend_attention/triton/extend_attention.py` |
| 优化 kernel | `boltops/fa/extend_attention/triton/_optimized_kernels.py` |
| 配置加载 | `boltops/fa/extend_attention/triton/config.py` |
| JSON 配置 | `boltops/fa/extend_attention/triton/configs/triton_<capability>/` |
| 调优工具 | `boltops/tools/extend_attention_triton_tune/` |
| 测试 | `test/fa/test_extend_attention.py` |

## 附录 B. 溯源

公共接口由 SGLang `extend_attention.py` 迁入，参考提交为
`ee0f2375e5fb285e4fb021a9831fa88018244679`。运行时不依赖 SGLang。

## 附录 C. 实现备注

- `max_len_extend < 32` 使用 short-query kernel，其余使用 long-query kernel。
- 配置按设备、CU 数、dtype 和 shape key 匹配；精确 grid 未命中时选择同 shape
  的最近配置，无兼容 shape 时使用默认配置。
- `BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION=0` 可关闭 compact grid。
- `BOLT_OPS_TRITON_FP8_PREFILL_ATTN=0` 可关闭 FP8 prefill 路径。
