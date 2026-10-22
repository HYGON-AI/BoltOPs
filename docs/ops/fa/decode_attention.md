# Decode Attention 使用说明

Decode Attention 对每条请求的最后一个 token 计算单 query MHA，Q 为 BF16，
KV cache 为 FP8 E4M3。KV 可以是逻辑四维 `[B,H,S,D]`，也可以是按页表寻址的
物理分页缓存。输出由算子分配并返回；不提供 backward。

相关文档：

- [ExtendAttention](extend_attention.md)：Prefill 侧的四维与分段入口

## 1. 算子用途与数学语义

对每个 batch 和 head，设有效 cache 长度为 `S`：

```text
K_real = FP32(K) * k_scale
V_real = FP32(V) * v_scale
O = softmax((Q @ K_real^T) * sm_scale) @ V_real      # Q: [1,D]，K/V: [S,D]
```

Q 对应有效 cache 的最后一个 token，可见全部 `S` 个有效 token，包括它自己。
调用方须在调用前写入本 token 的 K/V。无 causal mask、滑窗、attention sink
或 tree mask。

Kernel 在 tile 内把 FP8 KV 转为 FP32 并乘 scale，softmax 统计和输出累加使用
FP32，输出为 BF16。

## 2. 接口说明

```python
import torch
from boltops.fa import decode_attention_fwd_4d

B, H, S = 4, 64, 4096
q = torch.randn(B, H, 1, 128, device="cuda", dtype=torch.bfloat16)

# 逻辑四维 KV：每条请求的 S 个 token 全部有效
k = torch.randn(B, H, S, 128, device="cuda").to(torch.float8_e4m3fn)
v = torch.randn(B, H, S, 128, device="cuda").to(torch.float8_e4m3fn)
out = decode_attention_fwd_4d(q, k, v, sm_scale=128**-0.5, k_scale=1.0, v_scale=1.0)

# 物理分页 KV：[num_pages, page_size, H, D]，按页表和有效长度读取
page_size, pages_per_req = 64, S // 64
cache_shape = (B * pages_per_req, page_size, H, 128)
k_cache = torch.randn(cache_shape, device="cuda").to(torch.float8_e4m3fn)
v_cache = torch.randn(cache_shape, device="cuda").to(torch.float8_e4m3fn)
block_table = torch.arange(B * pages_per_req, device="cuda", dtype=torch.int32).view(B, -1)
seq_lens = torch.full((B,), S, device="cuda", dtype=torch.int32)
out = decode_attention_fwd_4d(
    q, k_cache, v_cache,
    sm_scale=128**-0.5, k_scale=1.0, v_scale=1.0,
    block_table=block_table, seq_lens=seq_lens,
)  # [B,H,1,128]
```

函数签名：

```python
def decode_attention_fwd_4d(
    q, k, v, *, sm_scale, k_scale, v_scale,
    block_table=None, seq_lens=None, num_kv_splits=None,
):
    ...
```

也可以从 `boltops.fa.decode_attention` 导入。

### 2.1 参数含义

| 参数 | Shape / 类型 | 含义 |
|---|---|---|
| `q` | `[B,H,1,128]`，BF16 | 每条请求最后一个 token 的 query |
| `k` / `v`（无 `block_table`） | `[B,H,S,128]`，FP8 E4M3FN | 逻辑 KV，`S` 个 token 全部有效 |
| `k` / `v`（有 `block_table`） | `[num_pages,page_size,H,128]`，FP8 E4M3FN | 物理分页 KV cache |
| `sm_scale` | 有限正数，必填 | softmax scale |
| `k_scale` / `v_scale` | 有限正数，或同设备单元素 FP32 tensor，必填 | 真实 K/V = FP8 存储值 × scale |
| `block_table` | `[B,max_pages]`，int32 / int64，optional | 物理页编号；只校验实际用到的页，未用位置可填 `-1` |
| `seq_lens` | `[B]`，int32 / int64 | 有效长度；有 `block_table` 时必填，须为正且不超过页表容量 |
| `num_kv_splits` | 正整数，optional | split-KV 数；默认按长度和 CU 数自动选择 |
| 返回值 | `[B,H,1,128]`，BF16 | 新分配的连续输出 |

### 2.2 前置条件与限制

- 仅支持单 query、MHA（Q 与 KV head 数相同）、BF16 Q、`torch.float8_e4m3fn`
  KV、head dim 128。不支持 GQA、MLA latent 或多 query。
- 所有张量位于同一 CUDA/HIP device，最后一维 stride 为 1；cache 的 token、head
  和 page 维可以带 padding，但单请求或单页内的元素跨度须小于 `2^31`。
- 只支持标量 scale，不支持 per-head 或 per-block scale。scale、page size 和
  页表由调用方按模型与缓存配置提供，算子不做推断。
- 入口在 host 上校验页表、有效长度和 scale，并分配临时 workspace，因此不能
  被 CUDA Graph capture。
- 只做前向；输入需要梯度时报错。

## 3. 实现说明

- 复用 `boltops.mla.decode_attention_fwd_normal` 的 split-KV kernel，这条路径
  执行的是 MHA。MLA 侧代码不做修改。
- 分页模式直接按 `block_table` 读取物理页，页可以乱序或被多条请求共享。
- 逻辑模式把每条请求的 `[H,S,D]` 转置视图当作一个 strided page，不复制 cache。
  传给 kernel 的虚拟 page size 向上取整到 2 的幂，真实长度仍由 `seq_lens`
  约束，因此同一分桶内的长度复用同一份编译产物，不会随 decode 步数逐步重编译。
- 除输出外只分配 FP32 split workspace、少量索引和 scale tensor。

## 4. 正确性测试与性能基准

在仓库根目录运行：

```bash
# 正确性测试
PYTHONPATH=. python3 -m pytest -q test/fa/test_decode_attention_4d.py

# 只运行网络 shape 的正确性用例（逻辑模式及三种分页布局），不计时
PYTHONPATH=. python3 -m pytest -q -s test/fa/test_decode_attention_4d.py \
  -k test_original_shape_decode

# 性能用例，公共执行器输出 fa_decode_4d.csv 和 raw 记录
PYTHONPATH=. PYTEST_ADDOPTS='-k test_bench_fa_' \
  python3 scripts/run_operator_benchmarks.py --ops fa_decode_4d
```

小尺寸用例覆盖乱序/共享页、ragged 长度、多种 page size、cache padding、
非单位 scale、空 split、编译分桶和非法输入。网络 shape 用例对全部输出元素
做独立 FP32 对照，reference 以传入的 FP8 存储值乘 scale 为准，不度量量化本身的
误差。用例配置、显存预检和用例选择见
[网络 shape 用例](extend_attention.md#31-网络-shape-用例)。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/fa/decode_attention/__init__.py` |
| 适配层 | `boltops/fa/decode_attention/triton/decode_attention.py` |
| 复用的 kernel | `boltops/mla/triton/decode_attention.py` |
| 测试 | `test/fa/test_decode_attention_4d.py` |
