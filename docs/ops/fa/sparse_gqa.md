# Qwen Sparse GQA Triton Kernel 使用说明

该模块提供 Qwen3.8-Flash-Next QSA 在 Hygon DCU decode 路径上的两个 Triton
算子。函数名、调用签名和核心 kernel 直接复用 Qwen QSA
`qsa/sparse_attn.py`，BoltOPs 侧补充参数校验、独立正确性 reference、benchmark
和 HCU 离线调优配置；不提供 AITer 或 NVIDIA 接口适配层。

相关文档：

- [Sparse GQA Triton autotune 使用手册](../../../boltops/tools/sparse_gqa_triton_tune/tune_sparse_gqa_user_guide.md)

## 1. 算子用途与数据流

Sparse GQA decode 分为 KV compact 和 packed attention 两步：

```text
HCU-FA paged FP8/BF16 K/V cache
        │ selected logical positions + req_to_token
        ▼
qwen_sparse_kv_extraction_compact_hcu_fa_triton
        │ packed ragged NHD K/V + cu_k
        ▼
sparse_gqa_packed_decode_triton
        │
        ▼
[total_q,Hq,D] attention output
```

HCU-FA cache 使用非对称布局：K 为 `[pages,Hkv,page_size,D]`，V 为
`[pages,Hkv,D,page_size]`。Compact 根据请求内逻辑位置查找物理 slot，将选中的
K/V 写入连续的 ragged NHD buffer；packed decode 再对每个请求执行 sparse GQA。

## 2. 公共接口

```python
from boltops.fa.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
```

### 2.1 KV compact

```python
qwen_sparse_kv_extraction_compact_hcu_fa_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size,
    k_scale=None,
    v_scale=None,
)
```

Compact 原地写入调用方预分配的 `out_k/out_v`，返回 `None`。

### 2.2 Packed decode

```python
out = sparse_gqa_packed_decode_triton(
    q,
    packed_k,
    packed_v,
    relative_indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    k_scale=None,
    v_scale=None,
)
```

Packed decode 返回与 `q` 相同 shape 和 dtype 的新 tensor。wrapper 固定使用
`grid_m=1`，并且不通过 `.item()` 读取设备数据，适用于 decode graph capture。

## 3. 输入输出与布局

### 3.1 Compact 输入输出

| 参数 | Shape/类型 | 含义 |
| --- | --- | --- |
| `k` | `[pages,Hkv,page_size,D]` | HCU-FA paged K cache |
| `v` | `[pages,Hkv,D,page_size]` | HCU-FA paged V cache |
| `req_to_token` | `[num_requests,max_context]`，int32/int64 | 请求内位置到物理 slot 的映射 |
| `req_indices` | `[B]`，int32/int64 | 当前 batch 对应的请求编号 |
| `indices` | `[B,topk]` | 每个请求选中的逻辑位置，`-1` 为 padding |
| `seq_lens` | `[num_requests]` | 每个请求的有效上下文长度 |
| `cu_k` | `[B+1]` | packed K/V 的请求边界 |
| `out_k/out_v` | `[sum(valid_counts),Hkv,D]` | 调用方预分配的 packed 输出 |
| `k_scale/v_scale` | float，可选 | FP8 cache 到输出 dtype 的反量化 scale |

Compact 先用 `req_indices` 选择请求，再以 `indices` 查询 `req_to_token`，最后把
物理 slot 拆成 `page/offset` 并写入 `cu_k` 描述的连续输出。

### 3.2 Decode 输入输出

| 参数 | Shape/类型 | 含义 |
| --- | --- | --- |
| `q` | `[total_q,Hq,D]` | packed query；decode 路径每个请求一行 |
| `packed_k/packed_v` | `[total_k,Hkv,D]` | compact 后的连续 K/V |
| `relative_indices` | `[B,topk]` 或 `[B,Hkv,topk]` | packed 请求段内的相对索引 |
| `cu_q` | `[B+1]` | query 请求边界 |
| `cu_k` | `[B+1]` | K/V 请求边界 |
| `kv_lens` | `[B]` | 每个请求的有效 K/V 数量 |
| `scale` | float | attention score scale，通常为 `D**-0.5` |
| `k_scale/v_scale` | float，可选 | packed K/V 为 FP8 时的反量化 scale |
| 返回值 | `[total_q,Hq,D]` | sparse GQA 输出 |

## 4. 生产 Shape

所有生产 case 固定 `D=256`、`Hkv=1`、`topk_width=2051`、
`selected_tokens=2048`、`page_size=64`。

| 分发 | Case | B | Context | Hq | Hkv | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP8 | D1 | 1 | 4096 | 3 | 1 | 256 |
| TP8 | D8 | 8 | 32768 | 3 | 1 | 256 |
| TP8 | D32 | 32 | 32768 | 3 | 1 | 256 |
| TP4 | D1 | 1 | 4096 | 6 | 1 | 256 |
| TP4 | D8 | 8 | 32768 | 6 | 1 | 256 |
| TP4 | D32 | 32 | 32768 | 6 | 1 | 256 |

`production` 模拟在完整 context 上生成 512 个 QSA 压缩 block，将每个 block
展开为 4 个 token，并打乱物理 page 分配；`contiguous` 只用于测量理想局部性
上界。每行 2051 宽的 indices 中有 2048 个有效 token，其余位置为 `-1`。

## 5. 使用约束

- 所有输入 tensor 必须位于同一设备。
- metadata 必须使用 int32 或 int64。
- `Hq % Hkv == 0`。
- K/V cache 为 FP8 且输出为 BF16/FP16 时，需要传入正确的
  `k_scale/v_scale`。
- Compact 的 K/V cache 布局不对称，不能把 V 当作
  `[pages,Hkv,page_size,D]` 读取。
- Packed decode 使用 packed 请求段内的相对索引，不能直接传 paged cache 的
  绝对 slot。
- 配置与 Triton capability、gfx arch、每卡 CU 数、shape 和 dtype 绑定，不应
  跨硬件档位直接复制最优配置。

## 6. 正确性测试与性能基准

```bash
pytest -q \
  test/fa/test_sparse_gqa.py \
  test/fa/test_sparse_gqa_config.py \
  test/test_package_smoke.py \
  -x
```

测试覆盖 HCU-FA 非对称布局、请求到物理 slot 映射、ragged `cu_k`、
FP16/BF16/FP8、TP4/TP8 的 D1/D8/D32、乱序与 per-KV-head indices、非连续
输入、端到端 compact→decode、配置精确匹配和旧文件兼容回退。

```bash
# FLA 风格的可选 pytest benchmark
pytest -q -s test/fa/test_sparse_gqa.py --bench
```

每个 case 在计时前都会对比独立 Torch compact/attention reference，然后输出
compact、decode 和端到端延迟。

## 7. Autotune 与配置选择

```bash
# 单 case
boltops-sparse-gqa-tune-cli --tp 8 --case D8 --fp8-cache

# TP4/TP8 全量生产 case；不要添加 --quick
python3 -m boltops.tools.sparse_gqa_triton_tune.tune_sparse_gqa \
  --tp 4 \
  --tp 8 \
  --case D1 \
  --case D8 \
  --case D32 \
  --fp8-cache \
  --access-pattern production \
  --warmup-ms 50 \
  --rep-ms 200
```

Compact 搜索 `BLOCK_TOPK/num_warps/num_stages`，decode 搜索
`BLOCK_N/num_warps/num_stages`。每个候选先通过 correctness gate，再参与计时。

配置目录按 Triton capability 分层，文件名按 kernel、gfx arch 和当前设备 CU 数
区分：

```text
boltops/fa/sparse_gqa/triton/configs/
└── triton_3_6/
    ├── qwen_sparse_kv_compact-gfx936-cu80.json
    ├── qwen_sparse_kv_compact-gfx938-cu64.json
    ├── qwen_sparse_kv_compact-gfx938-cu72.json
    ├── sparse_gqa_packed_decode-gfx936-cu80.json
    ├── sparse_gqa_packed_decode-gfx938-cu64.json
    └── sparse_gqa_packed_decode-gfx938-cu72.json
```

运行时使用 `torch.cuda.get_device_properties(...).multi_processor_count` 获取每卡
CU 数，优先加载当前 capability 下的精确 `arch+CU` 文件，再尝试兼容的低版本
capability 目录，最后回退到旧版无 CU 文件；未命中精确 shape 时使用安全默认
配置。完整 shape 和 dtype 保存在每个 JSON 的 `match` 字段中。

## 附录 A. 仓库文件

| 用途 | 路径 |
| --- | --- |
| 公共接口 | `boltops/fa/sparse_gqa/__init__.py` |
| KV compact | `boltops/fa/sparse_gqa/triton/compact.py` |
| Packed decode | `boltops/fa/sparse_gqa/triton/packed_decode.py` |
| 配置加载 | `boltops/fa/sparse_gqa/triton/config.py` |
| 生产 case/reference | `boltops/tools/sparse_gqa_triton_tune/` |
| 正确性测试 | `test/fa/test_sparse_gqa.py` |
| 配置测试 | `test/fa/test_sparse_gqa_config.py` |

## 附录 B. 溯源

- 源码合同：Qwen QSA `qsa/sparse_attn.py` 中 `_compact_kv_hcu_fa`、
  `_sparse_gqa_chunk_prefill` 及其 wrapper。
- 上层合同：`qwen_sparse_attn_backend.py` 的 HCU compact + packed decode 路径。
- 生产 shape：交接材料 `SHAPES_TP8.md` 与 `SHAPES_TP4.md`。

## 附录 C. 实现备注

- Compact 的默认配置为 `BLOCK_TOPK=16,num_warps=8,num_stages=1`。
- Packed decode 的默认配置为 `BLOCK_N=32,num_warps=8,num_stages=2`。
- `config=None` 时自动加载离线调优配置；内部测试和 autotune 可以显式传入
  config，以保证 baseline 与候选测量不受运行时自动选择影响。
- Compact 对相同 B/topk/Hkv/D/dtype/page_size 的 TP4/TP8 case 复用配置；decode
  因 Hq 不同分别保存精确匹配项。
