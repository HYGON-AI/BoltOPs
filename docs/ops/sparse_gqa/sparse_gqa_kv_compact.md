# Sparse-GQA KV Compact 使用说明

## 1. 算子用途

`qwen_sparse_kv_extraction_compact_hcu_fa_triton` 根据 QSA 选中的逻辑 token
位置，从 HCU-FA paged K/V cache 收集数据，写入连续的 ragged NHD buffer。

该算子只执行索引映射、gather 和可选 FP8 反量化，不计算 attention。

## 2. 接口说明

```python
from boltops.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
)

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

函数原地写入 `out_k` 和 `out_v`，返回 `None`。

## 3. 输入输出

| 参数 | Shape / 类型 | 含义 |
| --- | --- | --- |
| `k` | `[pages,Hkv,page_size,D]`，FP16/BF16/FP8 | HCU-FA paged K cache |
| `v` | `[pages,Hkv,D,page_size]`，与 `k` 同 dtype | HCU-FA paged V cache |
| `req_to_token` | `[num_requests,max_context]`，int32/int64 | 逻辑位置到物理 slot 的映射 |
| `req_indices` | `[B]`，int32/int64 | 当前 batch row 对应的 request id |
| `indices` | `[B,topk]`，int32/int64 | 每个请求选中的逻辑位置；`-1` 为 padding |
| `seq_lens` | `[B]`，int32/int64 | 当前 batch row 的有效上下文长度 |
| `cu_k` | `[B+1]`，int32/int64 | packed 输出的请求边界 |
| `out_k/out_v` | `[N,Hkv,D]`，FP16/BF16 或与 cache 相同的 FP8 | 调用方预分配的 packed 输出 |
| `batch` | int | 本次处理的请求数 |
| `topk` | int | 每行读取的 indices 宽度 |
| `page_size` | int | cache page 中的 token 数 |
| `k_scale/v_scale` | float，可选，缺省 `1.0` | FP8 K/V 的独立反量化 scale |

`N` 至少覆盖 `cu_k[B]`。K 和 V 的 paged 物理布局不同，但 packed 输出均为
`[N,Hkv,D]`。

## 4. 索引和 padding 语义

对 batch row `b`：

```text
request    = req_indices[b]
pack_start = cu_k[b]
pack_end   = cu_k[b+1]
valid_n    = pack_end - pack_start
```

只处理 `indices[b]` 的前 `valid_n` 列。对其中第 `j` 列：

```text
position = indices[b,j]
slot     = req_to_token[request,position]
page     = slot // page_size
offset   = slot % page_size
dst      = pack_start + j
```

仅当 `0 <= position < seq_lens[b]` 时写入。负索引、超出 sequence 长度的位置及
`valid_n` 之后的列不会写入输出。无效列不会触发稳定压紧：后续有效列仍写入
`pack_start+j`，对应无效列的目标行保持调用前内容。调用方应让
`cu_k[b+1]-cu_k[b]` 与该请求预留的 packed 行数一致。

## 5. Dtype 规则

- `k` 和 `v` 的 dtype、device 必须相同。
- 非 FP8 cache：`out_k/out_v` 必须与 cache dtype 相同。
- FP8 cache：
  - 输出可保持同一种 FP8，按位收集；
  - 输出也可为 FP16/BF16，此时分别使用 `k_scale`、`v_scale` 反量化；未传入
    的 scale 按 `1.0` 处理。
- 非 FP8 cache 或 FP8 输出路径不使用 `k_scale/v_scale`。
- `out_k` 和 `out_v` 必须具有相同的 shape、dtype 和 device。

## 6. 前置条件

- K/V cache 必须是 rank-4，并满足
  `K[P,H,S,D]`、`V[P,H,D,S]`。
- `page_size == k.shape[2] == v.shape[3]`。
- K/V 的 page 数、KV head 数和 head dim 必须匹配。
- `req_to_token` 与 `indices` 是 rank-2；`req_indices`、`seq_lens`、
  `cu_k` 是 rank-1。
- 所有 tensor 必须位于同一设备。
- metadata dtype 必须为 int32 或 int64。
- `0 <= req_indices[b] < num_requests`，`0 <= seq_lens[b] <= max_context`，且
  被访问的逻辑位置不能超过 `req_to_token.shape[1]`。
- `req_to_token` 产生的物理 slot 必须满足
  `0 <= slot < pages * page_size`。
- `cu_k` 必须非负且单调，并满足
  `0 <= cu_k[b+1]-cu_k[b] <= topk`。
- 输出容量必须满足 `cu_k[batch] <= N`。

Wrapper 会校验 rank、shape、dtype、device 以及 metadata 长度，但不会遍历设备
tensor 校验上述索引上界和前缀和值；调用方必须在进入算子前保证这些条件。

## 7. 正确性测试与性能基准

```bash
PYTHONPATH=. pytest -q test/sparse_gqa/ \
  -k 'compact or fp8_compact'
```

性能 case 通过总测试文件的 `--bench` 运行：

```bash
PYTHONPATH=. pytest -q -s test/sparse_gqa/ --bench
```

## 附录 A. 仓库文件

| 用途 | 文件 |
| --- | --- |
| 公开接口 | `boltops/sparse_gqa/__init__.py` |
| Triton 实现 | `boltops/sparse_gqa/triton/kv_compact.py` |
| 配置加载 | `boltops/sparse_gqa/triton/config.py` |
| 测试 | `test/sparse_gqa/` |
