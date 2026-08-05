# `recompute_w_u_fwd 使用说明`

Gated Delta Net 分块 WY 表示的重计算算子：根据 `k`、`v`、`beta`、`g_cumsum`
和 chunk 矩阵 `A`，重建 `w`、`u`。

## 1. 算子用途与数学语义

前向已得到每个 chunk 的 `A`；继续计算前需重建 WY 张量。对序列第 `c` 个
chunk、value head `h`，有效长度为 `L`，chunk 大小为 `BT`：

```text
key_head(h) = floor(h / (H / Hg))   # 要求 H % Hg == 0

u_c = A_c @ (v_c * beta_c[..., None])

w_c = A_c @ (
    k_c[:, key_head(h), :]
    * beta_c[:, h, None]
    * exp(g_cumsum_c[:, h, None])
)
```

- `u` 与 `v` 同 shape / dtype
- `w` 为 value-head 布局，shape `[B, T, H, K]`，dtype 同 `k`
- gate 使用自然指数 `exp`（不是 `exp2`）

## 2. 接口说明

```python
from bolt_ops.generic import recompute_w_u_fwd

w, u = recompute_w_u_fwd(
    k=k,
    v=v,
    beta=beta,
    g_cumsum=g_cumsum,
    A=A,
    cu_seqlens=None,
)
```

函数签名：

```python
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

### 2.1 参数含义

| 参数              | Shape             | 常见 dtype       | 含义                                                  |
| ----------------- | ----------------- | ---------------- | ----------------------------------------------------- |
| `k`             | `[B, T, Hg, K]` | FP16/BF16        | key；允许`Hg <= H`                                  |
| `v`             | `[B, T, H, V]`  | FP16/BF16        | value                                                 |
| `beta`          | `[B, T, H]`     | BF16/FP32        | 每 token/head 更新系数                                |
| `g_cumsum`      | `[B, T, H]`     | FP32             | chunk 内累计自然对数 gate                             |
| `A`             | `[B, T, H, BT]` | 同`k`          | 每行对应 chunk 内一个 WY 系数行；`BT = A.shape[-1]` |
| `cu_seqlens`    | `[N+1]`         | int32/int64      | packed varlen 边界；定长时传`None`                  |
| `chunk_indices` | `[NT, 2]`       | 同`cu_seqlens` | 可选`(sequence, local_chunk)` 表；未传则入口内生成  |
| 返回`w`         | `[B, T, H, K]`  | 同`k`          | 重建的 gated key WY                                   |
| 返回`u`         | `[B, T, H, V]`  | 同`v`          | 重建的 value WY                                       |

### 2.2 前置条件

- 张量在 GPU 上，且相互兼容的 device
- `k/v/beta/g_cumsum/A`（及传入的 index）为 **contiguous** 标准布局；不支持任意 stride
- `H % Hg == 0`
- `A.shape[-1]` 即为 chunk size `BT`
- 定长：`cu_seqlens=None`，每个 batch 长度为 `T`，chunk 从 `0, BT, 2*BT, ...` 切开
- varlen：`B=1`，`cu_seqlens` 给出各序列起止；chunk 不得跨序列边界。若已缓存
  `chunk_indices` 可传入，否则由入口按 `cu_seqlens` 与 `BT` 生成

## 3. 典型 Shape

典型静态配置：

| 字段                           | 值                  |
| ------------------------------ | ------------------- |
| `B / H / Hg`                 | `1 / 16 / 8`      |
| `K / V / BT`                 | `128 / 128 / 64`  |
| `k / v / w / u / A`          | BF16                |
| `beta / g_cumsum`            | FP32                |
| `cu_seqlens / chunk_indices` | int32               |
| `k`                          | `[1, T, 8, 128]`  |
| `v`                          | `[1, T, 16, 128]` |
| `beta / g_cumsum`            | `[1, T, 16]`      |
| `A`                          | `[1, T, 16, 64]`  |
| `w / u`                      | `[1, T, 16, 128]` |

`T` 随 packed 序列变化（约 60–13674）；测试与默认 benchmark 覆盖的代表点见附录。

## 4. 正确性测试与性能基准

```bash
# 正确性测试
pytest -q test/test_generic_recompute_w_u.py

# 性能（真实模型尺寸；与正确性 case 集不同）
pytest -q test/test_generic_recompute_w_u.py --bench -s
```

## 附录 A. 仓库文件

| 用途        | 文件                                      |
| ----------- | ----------------------------------------- |
| 公开接口    | `bolt_ops/generic/__init__.py`                    |
| Triton 实现 | `bolt_ops/generic/triton/recompute_w_u.py`         |
| 测试 / 性能 | `test/test_generic_recompute_w_u.py`（`--bench`） |

范围：`recompute_w_u_fwd` / `recompute_w_u_fwd_kernel` 及 chunk-index helper；
不含 backward。

## 附录 B. 溯源

- SGLang commit：[`132ade55cd65c3679a60adec563c8e6ca81238c5`](https://github.com/sgl-project/sglang/commit/132ade55cd65c3679a60adec563c8e6ca81238c5)
- 源文件：[`wy_fast.py`](https://github.com/sgl-project/sglang/blob/132ade55cd65c3679a60adec563c8e6ca81238c5/python/sglang/kernels/ops/attention/fla/wy_fast.py)
- 正确性 case 参考 FLA：[`test_gdn_kernels.py`](https://github.com/fla-org/flash-linear-attention/blob/fe8fce9fc6984f22905f54cfa885dce1502baf26/tests/ops/test_gdn_kernels.py) @ [`fe8fce9`](https://github.com/fla-org/flash-linear-attention/commit/fe8fce9fc6984f22905f54cfa885dce1502baf26)

## 附录 C. 实现备注

- Kernel grid 为 `(chunk, batch × value_head)`；按固定 `BK/BV` 扫 `K/V`；尾 chunk
  用 `boundary_check` 处理。
- BW1000_H 上按 `K/V` 在三档固定 launch 配置中选择，无运行时 autotune。

## 附录 D. 需求 shape 抓取明细

packed sequence 数分布（`N = cu_seqlens.shape[0] - 1`）：

| sequence 数 | 记录数 |   `T` 范围 | `NT` 范围 |
| ----------: | -----: | -----------: | ----------: |
|           1 |    226 |     60–4832 |       1–76 |
|           2 |     12 |   8323–9009 |    131–142 |
|           3 |     16 | 12188–13674 |    192–216 |

测试 / benchmark 代表点：

| Case                   | `T` | sequence 数 | `NT` | 说明       |
| ---------------------- | ----: | ----------: | -----: | ---------- |
| `observed-short`     |    60 |           1 |      1 | 最短       |
| `observed-common`    |  4156 |           1 |     65 | 常见单序列 |
| `observed-hot`       |  4543 |           1 |     71 | 高频       |
| `observed-common-n2` |  8652 |           2 |    136 | 双序列     |
| `observed-hot-n3`    | 12617 |           3 |    199 | 三序列高频 |
| `observed-max`       | 13674 |           3 |    216 | 最大       |

日志未记录各序列真实长度；`N>1` 的 case 由 helper 在 `T`、`N`、`NT` 约束下
合成长度拆分，shape/grid 与抓取一致，单序列切分不是日志还原值。
