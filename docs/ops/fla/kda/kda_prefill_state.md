# KDA Prefill State 阶段使用说明

## 1. 阶段职责

State 阶段根据 WY 的 `w/u/kg` 更新每个 chunk 的 recurrent state，并生成
`h`、`v_new` 和可选 final state。物理 state 布局为 `[slots,H,V,K]`。

## 2. 接口与 I/O

standalone wrapper 可从 `boltops.fla.kda` 导入：

```python
from boltops.fla.kda import chunk_gated_delta_rule_fwd_h

h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=kg, w=w, u=u, gk=gk,
    initial_state=initial_state,
    initial_state_indices=initial_state_indices,
    output_final_state=False,
    inplace_final_state=False,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_size=64,
    use_exp2=True,
    null_state_index=-1,
)
```

| 参数/返回值 | Shape | 说明 |
|---|---|---|
| `k/w/kg` | `[B,T,H,K]` | key-side intermediates |
| `u/v_new` | `[B,T,H,V]` | value/update intermediates |
| `gk` | `[B,T,H,K]` | `chunk_local_cumsum` 后的 FP32 log2 gate |
| `initial_state` | `[slots,H,V,K]` | FP32，可选；state pool |
| `initial_state_indices` | `[N]` | int32/int64，可选；每 sequence 的 pool 行 |
| `inplace_final_state` | bool | 默认 `False`；`True` 时 final state 写回索引到的 pool 行，第三个返回值即 pool 本身 |
| `null_state_index` | int | 默认 `-1`（SGLang 约定：仅负数无效，slot 0 为保留 dummy 行）；`0` = vLLM NULL_BLOCK_ID（`<=0` 无效，slot 从 1 开始） |
| `h` | `[B,NT,H,V,K]` | per-chunk state |
| `v_new` | `[B,T,H,V]` | decayed value |

## 3. fixed/varlen 与兼容边界

fixed 使用规则 chunk；varlen 使用 `cu_seqlens`、chunk offset 和
`chunk_indices`，chunk 不跨 sequence。KDA orchestration 只有一个 H；standalone
wrapper 为兼容上游 vLLM 保留 `Hg != H` grouped-key 协议。

### 3.1 State pool（`initial_state_indices`）

与 GDN `chunk_gated_delta_rule_fwd_h` 相同的池索引语义（见
[gdn.md 的 State index ABI 一节](../gdn.md)）：

- `initial_state_indices` 把每个 sequence 映射到 pool 的一行；无效行由
  `null_state_index` 判定（默认 `-1`，SGLang 约定；SGLang KDA prefill 即此
  用法），无效行的初始 state 按零处理且不写回；
- `inplace_final_state=True` 时 final state 直接写回索引行（返回 pool 本身），
  与 SGLang `chunk_kda_fwd` 的 `inplace_update=True` 默认行为对齐；调用方必须
  保证并发 sequence 不共享同一有效行；
- pool 行内必须 row-contiguous（stride 后三维为 `[V*K,K,1]`），行间距
  `stride(0)` 可以是 envelope-strided（大于 `H*V*K`，如 page-major 池）；
- 同 kernel 的 state/output 融合 fast path
  （`chunk_kda_fwd_state_output_fused`）支持同样的池参数，orchestration
  （`chunk_kda`/`chunk_kda_with_fused_gate`）会把三个参数透传到两条路径。

模型的 raw gate 可以是 BF16；进入 state/output kernel 前应经过
`chunk_local_cumsum` 并物化为 FP32，这与上游调用约定一致。

## 4. Kernel、配置与 fallback

- Kernel：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`；
- fixed K/V=128 主路径为 `BV64,w4,s2`，varlen 为 `BV32,w2,s2`；K/V=256 或
  LDS 受限 shape 使用较小 tile；
- state/output fusion 替换该阶段与 output 阶段的两个 dispatch；融合 kernel
  自己写出 final state，`output_final_state=True` 不再回退 split pipeline；
- standalone state、`Hg != H` 和 final-state 语义保持不变。

### 4.1 融合 kernel 的 V-block 选择

融合 grid 是 `(V/BV, N*H)`。拆分 V 是有代价的：每个 V-block 都要重读整个 K 侧
（`q`/`kg`/`w`/`g`/`Aqk`），所以拆分只为把更多 workgroup 放上设备。`BV=V` 的宽
tile 只读一遍 K 侧，但它持有完整的 `[V,K]` FP32 累加器（K/V=128 时 64 KiB），
一个 CU 上放不下第二个 workgroup；一旦 grid 超过一个 CU wave，它就失去了延迟
隐藏，窄 tile 反而更快。因此 `get_fused_state_output_config` 按 `N*H` 相对一个
wave 的位置选 tile。BW1100（72 CU，`K=V=128`，`T=16384`）实测最优：

| `N*H` | 16 | 24 | 32 | 40 | 48 | 64 | 72 | 96 |
|---|---|---|---|---|---|---|---|---|
| `BV` | 32 | 64 | 64 | 128 | 128 | 128 | 128 | 64 |

`_NUM_CU` 只影响该选择，取值过时只损失性能，不影响正确性。

### 4.2 fusion 不再有 `N*H` 下限

旧的 `N*H >= 40` 门槛属于固定 `BV=64` tile：那时 grid 是 `2*N*H`，小 `N*H`
下填不满设备。现在 tile 随 `N*H` 变化，grid 始终贴着一个 wave，fusion 在所有
head 数上都赢。BW1100 在 `T=16384`（对 fused 路径最不利：serial chunk 最多、
可让给 split 的 chunk 级并行最少）实测 split/fused：

| `N*H` | 2 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|
| 加速比 | 1.25x | 1.37x | 1.48x | 1.66x | 1.65x |

## 5. 性能与证据

BW1000_H、`H=24,T=8192,K=V=128` standalone state：

| Layout | XCU dispatch | 有效 HBM GB/s（model） |
|---|---:|---:|
| fixed | `0.958720 ms` | `318.28` |
| varlen | `0.776480 ms` | `540.98` |

2026-08-27 XCU 复核（`K=V=128`）使用 dispatch `execution`；HBM 为相同
traffic model 的有效带宽换算值。本机 XCU memory PMC 返回 `0 metrics`，
因此不宣称为硬件 counter 实测。

原始 vLLM 没有对应 standalone dispatch，因此不计算 baseline 加速比。

## 6. 测试与源码

```bash
PYTHONPATH=. pytest -q test/fla/test_kda_state.py test/fla/test_kda_chunk.py
```

源码：`boltops/fla/kda/triton/chunk_state.py`。
