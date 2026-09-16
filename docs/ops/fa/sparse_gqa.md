# Qwen Sparse-GQA 使用说明

Qwen Sparse-GQA 提供 KV compact 和 packed attention 两个 forward 算子，用于
QSA decode/verify 路径。Compact 从 HCU-FA paged KV cache 中收集被选中的 token，
Decode 在连续的 ragged K/V 上计算 grouped-query attention。

相关文档：

- [KV Compact](sparse_gqa/sparse_gqa_kv_compact.md)
- [Packed Decode](sparse_gqa/sparse_gqa_decode.md)

## 1. 算子用途与支持范围

公开接口从 `boltops.fa.sparse_gqa` 导入：

```python
from boltops.fa.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
```

| 接口 | 主要用途 | 返回值 |
| --- | --- | --- |
| `qwen_sparse_kv_extraction_compact_hcu_fa_triton` | 从 paged cache 提取选中的 K/V | 原地写入 `out_k/out_v`，返回 `None` |
| `sparse_gqa_packed_decode_triton` | 在 packed ragged K/V 上计算 Sparse-GQA | 与 `q` shape、dtype 相同的 tensor |

当前仅支持 forward，不包含 backward。

## 2. 数据流和数学语义

```text
HCU-FA paged K/V cache
        │ req_to_token + selected indices
        ▼
KV compact
        │ packed K/V + cu_k
        ▼
Packed Sparse GQA
        │ q + relative indices + cu_q/cu_k
        ▼
attention output
```

### 2.1 KV compact

HCU-FA 的 K/V 使用不同的物理布局：

```text
K: [pages, Hkv, page_size, D]
V: [pages, Hkv, D, page_size]
```

对 batch `b` 的第 `j` 列选择项 `p`：

```text
request = req_indices[b]
slot    = req_to_token[request, p]
page    = slot // page_size
offset  = slot % page_size
dst     = cu_k[b] + j

out_k[dst, h, :] = k[page, h, offset, :]
out_v[dst, h, :] = v[page, h, :, offset]
```

负索引、超出 `seq_lens[b]` 的位置和 top-k padding 不参与写出，且不会把后续
有效项向前压紧；对应目标行保持调用前的内容。FP8 cache 反量化到 BF16/FP16
时，K 和 V 分别应用 `k_scale`、`v_scale`，未传入的 scale 按 `1.0` 处理。

### 2.2 Packed Sparse-GQA

令 `G = Hq / Hkv`。Query head `hq` 使用 KV head
`hkv = floor(hq / G)`。对 query `t` 的有效相对索引集合 `I_t`：

```text
score(t,hq,i) = scale * dot(q[t,hq,:], k[cu_k[b]+i,hkv,:])
prob           = softmax(score, dim=i)
out[t,hq,:]    = sum_i prob(i) * v[cu_k[b]+i,hkv,:]
```

二维 `indices:[T,topk]` 表示所有 KV heads 共用一组选择；三维
`indices:[T,Hkv,topk]` 表示每个 KV head 使用独立选择。索引均相对于当前请求
在 packed K/V 中的起点，而不是 paged cache 的绝对 slot。

## 3. 接口说明

### 3.1 KV compact

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

完整参数、布局和 dtype 约束见 [KV Compact](sparse_gqa/sparse_gqa_kv_compact.md)。

### 3.2 Packed decode

```python
out = sparse_gqa_packed_decode_triton(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    k_scale=None,
    v_scale=None,
)
```

完整参数、ragged ABI 和 dtype 约束见
[Packed Decode](sparse_gqa/sparse_gqa_decode.md)。

## 4. 生产 shape

`D1/D8/D32` 中的数字表示 decode batch size，head dim 记为 `D`，两者不要
混淆。

| 分发 | Case | Batch | Context | Hq | Hkv | Head dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP8 | D1 | 1 | 4096 | 3 | 1 | 256 |
| TP8 | D8 | 8 | 32768 | 3 | 1 | 256 |
| TP8 | D32 | 32 | 32768 | 3 | 1 | 256 |
| TP4 | D1 | 1 | 4096 | 6 | 1 | 256 |
| TP4 | D8 | 8 | 32768 | 6 | 1 | 256 |
| TP4 | D32 | 32 | 32768 | 6 | 1 | 256 |

这些 case 使用 `page_size=64`、2048 个有效 token 和宽度为 2051 的 indices；
多出的列为 padding。

## 5. 共同前置条件

- 所有输入、metadata 和输出 buffer 必须位于同一 CUDA/HIP device。
- `q` 必须为 FP16 或 BF16。
- Decode 的 K/V dtype 必须相同，并且为与 `q` 相同的 dtype 或受支持的 FP8。
- `Hq % Hkv == 0`，Q/K/V 的 head dim 必须一致。
- `req_to_token`、`req_indices`、`indices`、`seq_lens`、`cu_q`、
  `cu_k` 和 `kv_lens` 使用 int32 或 int64。
- `cu_q`、`cu_k` 必须是合法的单调前缀和，并与实际 packed tensor 分段一致。
- FP8 K/V 转换为 FP16/BF16 时，`k_scale/v_scale` 缺省为 `1.0`；调用方应按
  实际量化方式传入 scale。
- Compact 输入使用 paged cache 绝对映射；Decode 输入使用请求分段内的相对索引，
  两种索引不能互换。
- Wrapper 只校验 rank、shape、dtype 和 device 等结构条件；前缀和单调性、索引
  上界和输出容量由调用方保证，详细要求见两个子文档。

## 6. 正确性测试、性能基准与调优

### 6.1 Pytest

```bash
PYTHONPATH=. pytest -q test/fa/test_sparse_gqa.py
```

测试覆盖：

- HCU-FA 非对称 K/V 布局和请求到物理 slot 的映射；
- ragged `cu_k`、乱序选择、padding 和跨 tile 访问；
- FP16、BF16、FP8 cache 与独立 K/V scale；
- 共享 indices 和 per-KV-head indices；
- TP4/TP8 的 D1、D8、D32 生产几何；
- Compact 到 Decode 的组合路径和非法输入校验。

可选 benchmark：

```bash
PYTHONPATH=. pytest -q -s test/fa/test_sparse_gqa.py --bench
```

benchmark 在计时前先与独立 Torch reference 比较，然后分别输出 compact、
decode 和端到端时间。

### 6.2 配置调优

安装项目后可使用 CLI 对生产 case 做正确性校验和候选配置搜索：

```bash
boltops-sparse-gqa-tune-cli \
  --tp 8 \
  --case D8 \
  --fp8-cache \
  --samples 3 \
  --report sparse_gqa_tune_report.json
```

`--quick` 仅校验并写出 baseline；完整模式会遍历 compact 和 decode 候选。默认
输出目录为当前 Triton capability 对应的配置目录，可用 `--output-dir` 指向临时
目录，避免直接更新仓库 JSON。

## 7. 语义边界

- Compact 只负责 gather 和可选反量化，不计算 attention。
- Decode 接收已经连续化的 K/V；它不读取 paged cache，也不执行
  `req_to_token` 映射。
- `indices` 的 `-1` 表示 padding。其他非负值必须落在对应请求的 packed K/V
  有效范围内。
- `out_k/out_v` 由调用方分配，容量必须覆盖 `cu_k[-1]` 描述的全部有效行。
- Decode 返回新 tensor；Compact 原地写入输出 buffer。

## 附录 A. 仓库文件

| 用途 | 文件 |
| --- | --- |
| 公开接口 | `boltops/fa/sparse_gqa/__init__.py` |
| KV compact | `boltops/fa/sparse_gqa/triton/kv_compact.py` |
| Packed decode | `boltops/fa/sparse_gqa/triton/packed_decode.py` |
| 配置加载 | `boltops/fa/sparse_gqa/triton/config.py` |
| 调优入口 | `boltops-sparse-gqa-tune-cli` → `boltops.tools.sparse_gqa_triton_tune.tune_sparse_gqa:main` |
| 正确性与性能测试 | `test/fa/test_sparse_gqa.py` |
| Compact 文档 | [`sparse_gqa/sparse_gqa_kv_compact.md`](sparse_gqa/sparse_gqa_kv_compact.md) |
| Decode 文档 | [`sparse_gqa/sparse_gqa_decode.md`](sparse_gqa/sparse_gqa_decode.md) |
