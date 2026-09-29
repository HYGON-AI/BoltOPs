# Sparse-GQA Packed Decode 使用说明

## 1. 算子用途

`sparse_gqa_packed_decode_triton` 在 Compact 产生的连续 ragged K/V 上执行
Sparse Grouped-Query Attention。每条 query 只访问 `indices` 选中的 K/V 行。

## 2. 接口说明

```python
from boltops.sparse_gqa import sparse_gqa_packed_decode_triton

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

返回 `out` 的 shape 和 dtype 与 `q` 相同。

## 3. 输入输出

| 参数 | Shape / 类型 | 含义 |
| --- | --- | --- |
| `q` | `[Tq,Hq,D]`，FP16/BF16 | packed query |
| `k/v` | `[Tk,Hkv,D]`，与 `q` 同 dtype 或 FP8 | 每个请求连续存放的 packed K/V |
| `indices` | `[Tq,topk]` 或 `[Tq,Hkv,topk]`，int32/int64 | 请求分段内的相对 K/V 索引 |
| `cu_q` | `[B+1]`，int32/int64 | query 请求边界 |
| `cu_k` | `[B+1]`，int32/int64 | K/V 请求边界 |
| `kv_lens` | `[B]`，int32/int64 | 每个请求的有效 K/V 长度 |
| `scale` | float | attention score scale，通常为 `D**-0.5` |
| `k_scale/v_scale` | float，可选，缺省 `1.0` | FP8 K/V 的独立反量化 scale |
| 返回值 | `[Tq,Hq,D]` | Sparse-GQA 输出 |

## 4. Ragged 和索引 ABI

请求 `b` 的 query 位于：

```text
q[cu_q[b] : cu_q[b+1]]
```

对应 K/V 位于：

```text
k[cu_k[b] : cu_k[b+1]]
v[cu_k[b] : cu_k[b+1]]
```

`indices` 中的非负值是相对于 `cu_k[b]` 的偏移。例如 `indices[t,j]=7`
表示读取当前请求 packed 段中的第 7 行，不表示 paged cache 的物理 slot 7。

两种 indices 布局：

- `[Tq,topk]`：同一 query 的全部 KV heads 共用选择项；
- `[Tq,Hkv,topk]`：每个 KV head 使用独立选择项。

`-1` 表示 padding。其他非负索引必须小于当前请求的 `kv_lens[b]`。Kernel 不对
每个非负索引执行上界检查，调用方必须保证索引合法。

## 5. GQA 数学语义

要求 `Hq % Hkv == 0`。令 `G = Hq / Hkv`，query head `hq` 映射到
`hkv = floor(hq/G)`。

对有效选择集合 `I_t`：

```text
s_i = scale * dot(q[t,hq,:], k[cu_k[b]+i,hkv,:])
p_i = exp(s_i) / sum_j exp(s_j)
out[t,hq,:] = sum_i p_i * v[cu_k[b]+i,hkv,:]
```

多 query/request 时，令 `q_len = cu_q[b+1]-cu_q[b]`、
`query_relative = t-cu_q[b]`，实现使用：

```text
visible = query_relative + kv_lens[b] - q_len + 1
row_topk = min(topk, visible)
```

只读取当前 `indices` 行的前 `row_topk` 列。调用方必须按因果可见顺序组织选择
项，并保证被读取列中的非负索引小于 `kv_lens[b]`，且每条 query 至少有一个
有效选择项。

## 6. Dtype 和 scale

- `q` 只支持 FP16 或 BF16。
- `k` 与 `v` dtype 必须相同。
- 非 FP8 K/V 必须与 `q` dtype 相同。
- FP8 K/V 使用 `k_scale` 缩放 attention score，使用 `v_scale` 缩放 value
  累加结果；未传入的 scale 按 `1.0` 处理。
- 输出 dtype 与 `q` 相同。

## 7. 前置条件

- Q/K/V 都是 rank-3 `[N,H,D]`，head dim 必须一致。
- K/V shape 必须完全相同。
- `indices` 必须为 rank-2 或 rank-3，并至少为每条 query 提供一行。
- rank-3 indices 的第二维必须等于 `Hkv`。
- `cu_q`、`cu_k` shape 相同；`kv_lens.shape[0] == len(cu_q)-1`。
- `cu_q`、`cu_k` 必须非负且单调，末项分别不能超过 `Tq`、`Tk`；每个
  `kv_lens[b]` 必须落在对应 K/V 请求分段内。
- 所有 tensor 必须位于同一设备。
- `indices`、`cu_q`、`cu_k`、`kv_lens` 必须为 int32 或 int64。

Wrapper 会校验 rank、shape、dtype 和 device，但不会遍历设备 tensor 校验前缀
和、索引上界或每行有效项数量；调用方必须保证这些条件。

## 8. 执行路径

- 调用入口会先把 `k/v` 转为 contiguous；非连续输入会产生额外复制。
- 每个请求恰好一个 query、`q` contiguous 且 K/V 非 FP8 时，使用 split decode
  路径。
- FP8、非连续 `q` 或每请求多个 query 时，使用 chunk-prefill 路径。
- 两条路径均返回 `torch.empty_like(q)` 创建的输出。

## 9. 正确性测试与性能基准

```bash
PYTHONPATH=. pytest -q test/sparse_gqa/ \
  -k 'decode or compact_then_decode'
```

性能 case 通过总测试文件的 `--bench` 运行：

```bash
PYTHONPATH=. pytest -q -s test/sparse_gqa/ --bench
```

## 附录 A. 仓库文件

| 用途 | 文件 |
| --- | --- |
| 公开接口 | `boltops/sparse_gqa/__init__.py` |
| Triton 实现 | `boltops/sparse_gqa/triton/packed_decode.py` |
| 配置加载 | `boltops/sparse_gqa/triton/config.py` |
| 测试 | `test/sparse_gqa/` |
