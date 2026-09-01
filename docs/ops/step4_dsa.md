# Step-4 DSA 系列算子说明

`boltops.step4_dsa` 公开 Step-4 Dynamic Sparse Attention 的 **12 个数值 Triton API**（合同 `step4-dsa-ops-v1`）。Triton kernel 在 `boltops/step4_dsa/triton/`；模型几何在 `geometry.py`；生产 shape / 造数在 `test/step4_dsa/`（不进 wheel）；TileLang 目录为占位。

不包含 vLLM planner / KV 缓存写入，也不包含 `kernel.py` 里的 MoE FP8 / 主注意力 `fused_qknorm_rope`。

溯源见 [`boltops/step4_dsa/triton/SOURCE.md`](../../boltops/step4_dsa/triton/SOURCE.md)。

## 当前支持一览

| 公开 API | 场景 |
| --- | --- |
| `triton_build_rope_cache` | NeoX 半宽 RoPE 表，`rotary_span=32` |
| `triton_indexer_norm_rope` | indexer q RMSNorm / k LayerNorm + 前 32 维 RoPE；z 透传 |
| `triton_csa_region_layout` / `triton_csa_compress_regions` | 每 8 token 压成 FP32 + e4m3 summary |
| `triton_csa_prefill` | layout + compress 门面 |
| `triton_indexer_logits` | Weighted-ReLU 分数，FP32 `[groups * Q, regions]` |
| `triton_region_topk_ids` | 有符号 top-k，逻辑 region id |
| `triton_topk_select_and_pack` | 融合 select + prefill pack（物理 id） |
| `triton_prefill_sparse_meta` | 宽 `topk+1` 的 prefill metadata（当前 region 追加） |
| `triton_decode_sparse_meta` | int64 decode metadata |
| `triton_sparse_attention_prefill` | 稀疏 GQA prefill；tiled 路径 `block_q=128` |
| `triton_sparse_attention_decode` | 稀疏 decode，可选 split-KV |
| `triton_merge_split_states` | split LSE 合并 |

Vendor 常量（可直接作为 handoff `--ops-module boltops.step4_dsa`）：

- `STEP4_DSA_OPS_API_VERSION = "step4-dsa-ops-v1"`
- `STEP4_DSA_CAPABILITIES` 含 `core_dsa_v1`、`logical_prefill_ids_v1`、`tiled_prefill_v1`

```python
from boltops.step4_dsa import (
    triton_indexer_logits,
    triton_sparse_attention_decode,
)
```

## 几何（TP8 单 rank / 生产）

| 参数 | 值 |
| --- | --- |
| proxy_dim / indexer heads（本 rank） | 256 / 4 |
| attention heads / KV groups / head_dim | 8 / 1 / 192 |
| region / page / topk / metadata width | 8 / 16 / 512 / **513** |

全模型 TP1 几何在 `boltops.step4_dsa.geometry.STEP4`（16 indexer heads、64 Q heads、4 groups）。

## 测试与 bench

```bash
pytest -q test/step4_dsa/
pytest -q test/step4_dsa/ --bench -s
python3 scripts/benchmarks/triton/benchmark_step4_dsa.py --output-dir /tmp/dsa-bench
```

正确性对 `test/step4_dsa/ref.py`（handoff `torch_ref`）的 FP32 oracle，禁止 Triton-vs-Triton 作为门禁。`--bench` 跑 `tp8-production` 全表，包含 metadata 宽度 513、`num_splits=16` 的 decode。

独立脚本输出 `triton_step4_dsa.csv`。也可：

```bash
# 在 handoff 仓库内，把本包装成 vendor 模块
python3 -m operator_parity.run_all --suite dsa --ops-module boltops.step4_dsa
```

occupancy JSON 对照（`tuned` = 读 HCU JSON，`default` = `config={}` 内置 tile）：

```bash
python3 scripts/benchmarks/triton/step4_dsa/benchmark_sparse_attn_prefill.py \
  --output-dir scripts/benchmarks/triton/step4_dsa/result
python3 scripts/benchmarks/triton/step4_dsa/benchmark_sparse_attn_prefill_tiled.py \
  --output-dir scripts/benchmarks/triton/step4_dsa/result
python3 scripts/benchmarks/triton/step4_dsa/benchmark_sparse_attn_decode.py \
  --output-dir scripts/benchmarks/triton/step4_dsa/result
```

数字见下文 BW1100 性能记录。

## 已知问题

decode 在 **width=513 + `num_splits=16`** 上，现有 kernel 会让相邻 split 的最后一个 `BLOCK_R` tile 重叠（`ceil(513/16)=33`）。这些 case 已保留在 production bench 中，不要删。修复属于后续独立改动。本轮性能表只覆盖已落地 JSON 的 `splits=1` occupancy。

---

## `triton_sparse_attention_prefill`

稀疏 GQA prefill。未传 `logical_regions` 时每 query 一个 program；传入
logical ids / positions / request ids 后走 tiled 路径（`block_q=128`）。

```python
out, lse = triton_sparse_attention_prefill(
    query, key_cache, value_cache, packed_regions, region_counts,
    num_kv_groups=1, region_size=8, softmax_scale=None,
    block_regions=16, logical_regions=None, q_positions=None,
    request_indices=None, block_q=128, config=None,
)
```

- `query`：`[Q, H, D]`；`key_cache` / `value_cache`：`[tokens, G, D]`
- `packed_regions`：`[G * Q, packed_width]` int32；可选 tiled 还要
  `logical_regions`、`q_positions`、`request_indices`
- `config=None` 读 HCU JSON；`config={}` 跳过查找，用内置 tile

### BW1100 TP8 性能

固定环境为 BW1100（gfx938）、BF16、TP8 单 rank 几何（`H=8, G=1, topk=512,
packed_width=513, D=192, region_size=8`），`cache_regions=4096`。Triton
数字使用 `triton.testing.do_bench`（warmup 10 ms、rep 30 ms、中位数）。
本轮只替换 launch JSON，没有改 kernel。加速比定义为
`default latency / tuned latency`，大于 1 表示 JSON 更快。CSV 来自
`scripts/benchmarks/triton/step4_dsa/result/`。GPU 未锁频，数据用于同轮
相对比较，不作为稳定性能门槛。

#### per-query launch JSON（2026-09-03）

JSON 不按 `Q` 分文件；tuner 与 bench 都只记生产 occupancy `Q=128`。

| 环境项 | 值 |
| --- | --- |
| GPU | BW1100（gfx938） |
| Shape | `Q=128, H=8, G=1, topk=512, cache_regions=4096, D=192` |
| 输入 / 输出 dtype | BF16 / BF16 |
| 计时 | `triton.testing.do_bench`，warmup 10 ms、rep 30 ms、中位数 |
| default | `[BLOCK_R,BLOCK_H,BLOCK_D]=[8,16,256]`，未指定 warps/stages/waves |
| tuned | `[8,16,256, warps=4, stages=1, waves=2]` |

tile 尺寸与 default 相同，收益来自 `num_warps` / `num_stages` /
`waves_per_eu`。

| 实现 | 延迟 (ms) | 相对 default |
|---|---:|---:|
| default | 1.603678 | 1.00x |
| tuned | 0.677920 | 2.37x |

- HCU JSON：`triton/config/sparse_attn_prefill/`
  - `prefill-k{width}-hpg{hpg}-d{dim}-r{region}_{bf16|fp16}_{device}.json`
  - 当前：`prefill-k513-hpg8-d192-r8_bf16_BW1100.json`
- 无 JSON 或 `config={}` 时使用内置 `_per_query_tile_defaults`

实现：`triton/prefill.py`。溯源：Step-4 DSA handoff `sparse_attn.py`。

#### tiled launch JSON（2026-09-03）

生产 tile `Q=128`、`block_q=128`。实测 tile union=528，配置按
`next_power_of_2` 落到 `u=1024`。`latency_ms` 只含 tiled Triton kernel
（与 xprof `.perf` 对齐）；`_build_prefill_tiles` 在 `do_bench` 外单独记为
`tile_build_ms`（和 decode 把 metadata 放在计时区外同一做法）。
公开 API `triton_sparse_attention_prefill` 每次调用仍会重建 tiles。

| 实现 | launch `[BLOCK_Q,BLOCK_R,BLOCK_D,warps,stages,waves]` | kernel (ms) | 相对 default |
|---|---|---:|---:|
| default | `[128,4,256,4,2,—]` | 1.139039 | 1.00x |
| tuned | `[128,8,256,8,1,2]` | 0.709520 | 1.61x |

`_build_prefill_tiles` 约 1.79 ms，与 launch JSON 无关。加上它之后
e2e 约为 2.50 / 2.93 ms，接近此前把构造算进 `do_bench` 的 2.67 / 3.07 ms。

同一 `Q=128` 上，tiled kernel tuned（0.71 ms）与 per-query tuned（0.68 ms）
接近；生产入口仍要付 tile 构造，e2e 仍慢于 per-query。本轮只记 config
前后，不做路径切换。

- HCU JSON：`triton/config/sparse_attn_prefill/`
  - `prefill-tiled-k{width}-u{union}-hpg{hpg}-d{dim}-r{region}-q{bq}_{bf16|fp16}_{device}.json`
  - 当前：`prefill-tiled-k513-u1024-hpg8-d192-r8-q128_bf16_BW1100.json`
- 无 JSON 或 `config={}` 时使用内置 tile（`BLOCK_R` 由 union 收入，
  `num_warps=4`、`num_stages=2`）

实现：`triton/prefill.py`（tiled kernel）。溯源：同上。

---

## `triton_sparse_attention_decode`

稀疏 GQA decode。`num_splits==1` 直接返回最终 `out` / `lse`；`>1` 时返回
带 leading split 轴的 partial，交给 `triton_merge_split_states`。

```python
out, lse = triton_sparse_attention_decode(
    query, key_cache, value_cache, packed_regions, region_counts,
    kv_seqlens, num_kv_groups=1, region_size=8, softmax_scale=None,
    num_splits=1, block_regions=16, config=None,
)
```

- `query`：`[B, H, D]`；`kv_seqlens`：`[B]` int32
- `packed_regions`：`[G * B, packed_width]` **int64**
- `config=None` 按 occupancy 读 JSON（`batch=B`，`context=kv_seqlens[0]`）；
  `config={}` 或缺文件走内置 tile

### BW1100 TP8 性能

环境与 prefill 相同。metadata（`triton_decode_sparse_meta`）在计时区外。
当前只测已落地 JSON 的 `(B=4, ctx=16384, splits=1)`。

#### occupancy JSON（2026-09-03）

| 环境项 | 值 |
| --- | --- |
| GPU | BW1100（gfx938） |
| Shape | `B=4, context=16384, splits=1, H=8, G=1, topk=512, D=192` |
| 输入 / 输出 dtype | BF16 / BF16 |
| 计时 | `triton.testing.do_bench`，warmup 10 ms、rep 30 ms、中位数 |
| default | `[BLOCK_R,BLOCK_H,BLOCK_D]=[8,16,256]`，未指定 warps/stages/waves |
| tuned | `[2,16,256, warps=2, stages=2, waves=1]` |

| 实现 | 延迟 (ms) | 相对 default |
|---|---:|---:|
| default | 0.834720 | 1.00x |
| tuned | 0.656000 | 1.27x |

- HCU JSON：`triton/config/sparse_attn_decode/`
  - 专用：`decode-k{width}-s{splits}-b{B}-ctx{ctx}-hpg{hpg}-d{dim}-r{region}_{bf16|fp16}_{device}.json`
  - 当前：`decode-k513-s1-b4-ctx16384-hpg8-d192-r8_bf16_BW1100.json`
  - 缺文件时静默使用内置 `_decode_tile_defaults`

实现：`triton/decode.py`。溯源：Step-4 DSA handoff `sparse_attn.py`。

