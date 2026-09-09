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
| `triton_sparse_attention_prefill` | 稀疏 GQA prefill；tiled 路径 `block_q=128`；per-query 在 `Q*G` 填不满 CU 时自动 KV-split |
| `triton_sparse_attention_decode` | 稀疏 decode；默认 `num_splits=1` 自动按 CU 做 KV-split |
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
python3 scripts/benchmarks/triton/step4_dsa/benchmark_step4_dsa.py --output-dir /tmp/dsa-bench
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

### BW1100 短上下文 decode 有效槽位分区（2026-09-09）

Decode launch 配置支持 `BALANCE_LIVE`（默认 false）。开启时，自动 split
按每个 metadata row 的设备端 `region_counts` 分配窗口：
`slots_per_split = ceil(min(count, packed_width) / num_splits)`。
无需 host scalar 读取或 metadata 拷贝；grid 和 merge 输出形状保持不变。
显式 split、返回 partial states、单 split 均忽略此选项，保持原窗口合同。

仅以下三个 BW1100 JSON 开启，其他配置保持原路径。动态窗口在长上下文
没有稳定收益，不应全局启用。`config={}` 仍使用原默认策略。

| B | context | HEAD 基线 us | 优化 us | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 23.62 | 17.74 | 1.33x |
| 4 | 512 | 44.50 | 18.25 | 2.44x |
| 4 | 2048 | 46.62 | 34.48 | 1.35x |

环境：BW1100/gfx938/72 CU，Triton Hygon release/3.6.x `4e013043f`，
`AMDGCN_USE_BUFFER_OPS=1`。基线为 BoltOps `734417d`；同进程、相同输入和
原 launch tile，三轮正/逆/正顺序 `do_bench_cudagraph(rep=30)` 中位数，
计时包含 decode + merge。显式传 config，metadata 预先生成；不包含 Python
配置查找同步，不能视为完整 Python API 的端到端收益。GPU 未锁频，其他
size 有明显测量波动，未将其计为收益。18 个已有 decode 配置均对 FP32
Torch oracle 检查；额外测试覆盖空行、混合有效长度和显式 partial 窗口。

XCU 最初的 SIGABRT 已定位为 profiler 不支持 `launch_statistics`，
移除该 section 后计数器和 SQTT 均能采集。XCU 4.4.4 分析器仍报告
`v_fmac_f32` 指令识别警告，部分流水统计为 N/A，不据此量化 stall 比例。

### Decode 归一化共享倒数（2026-09-09）

最终输出由 `accumulator / running_sum` 改为每 head 计算一次 FP32
`tl.div_rn(1.0, safe_sum)`，再广播乘以 accumulator。空行使用分母 1
并输出零，LSE 仍为负无穷。不改变 QK/PV 的 MMAC 路径；倒数与乘法有
额外一次 FP32 舍入，因此正确性继续按 FP32 Torch oracle 的原容差验证。

以下基线包含上一节 BALANCE_LIVE 优化，计时方式和环境与上节相同：

| B | context | 上一版 us | 共享倒数 us | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 17.72 | 14.70 | 1.21x |
| 1 | 16384 | 23.53 | 20.48 | 1.15x |
| 4 | 512 | 18.30 | 15.18 | 1.21x |
| 4 | 2048 | 35.10 | 32.09 | 1.09x |
| 4 | 16384 | 49.10 | 45.89 | 1.07x |

18 个已有 decode 配置通过 FP32 oracle。`ctx=31929` 首轮结果波动，
追加八轮交替测量：B=4 为 47.94 -> 44.68 us；B=8 仍有显著时间漂移，
不据此宣称稳定收益。大 batch 的收益较小，不将小 batch 比例外推。

SQTT 已对 B=4/context=512 的前后版本采集（`sqtt status: enabled`）。
静态汇编中 `v_div*` 从 256 条降到 4 条，`v_rcp*` 从 65 条降到 2 条，
`v_fmac*` 从 133 条降到 7 条，`v_mmac*` 保持 96 条；这是静态指令数量，
不是动态执行次数或流水周期。SQTT 分析仍受上述分析器限制，收益以不带
profiler 的独立 Graph replay 计时为准。

### Decode 非 2 次幂 head_dim 分块 + occupancy 重调（2026-09-09）

`head_dim=192` 不再 pad 到 `BLOCK_D=256`，而是切成两个 pow2 块
`128+64`（`_sparse_attn_decode_dsplit_kernel`）。QK 为
`dot(q1,k1)+dot(q2,k2)`，PV 同样分块，去掉 padding 的 LDS / MMAC /
accumulator rescale（192 上约 25%）以及 `dims < head_dim` mask。
online softmax 改到 log2 域：`log2(e)` 折进 scale，`exp2` 替换 `exp`，
写出的 LSE 再乘 `ln2` 转回自然对数，merge 合同不变。occupancy JSON
不再写 `BLOCK_D`（块大小由 `head_dim` 推导），并按 dsplit kernel 重
tune `BALANCE_LIVE` / `BLOCK_R` / warps / stages / waves。

以下基线为 BoltOps `8bbee9b`（含 BALANCE_LIVE 与共享倒数，仍走
padded `BLOCK_D=256`）。计时方式和环境与上节相同：BW1100/gfx938/72 CU，
`HIP_VISIBLE_DEVICES=2`，`AMDGCN_USE_BUFFER_OPS=1`，显式传 config，
metadata 预先生成，三轮正/逆/正顺序 `do_bench_cudagraph(rep=30)`
中位数，计时包含 decode + merge。两次独立进程、相同输入种子，不是
同进程切 kernel。GPU 未锁频。

覆盖全部 18 个已落地 decode occupancy JSON（`splits=1` 快路径）。
小 batch / 短上下文上两段 `tl.dot` 的额外开销盖过 padding 节省，
出现回退，不把大 batch 比例外推到这些点。

| B | context | HEAD 基线 us | 优化 us | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 14.69 | 20.21 | 0.73x |
| 1 | 16384 | 20.32 | 21.65 | 0.94x |
| 2 | 16384 | 27.18 | 27.03 | 1.01x |
| 4 | 512 | 15.10 | 15.81 | 0.95x |
| 4 | 2048 | 30.63 | 34.31 | 0.89x |
| 4 | 4096 | 44.13 | 39.34 | 1.12x |
| 4 | 8192 | 44.17 | 37.38 | 1.18x |
| 4 | 16384 | 44.40 | 31.90 | 1.39x |
| 4 | 31929 | 44.69 | 44.30 | 1.01x |
| 4 | 32768 | 44.40 | 44.26 | 1.00x |
| 4 | 65536 | 44.52 | 40.53 | 1.10x |
| 8 | 16384 | 72.48 | 51.48 | 1.41x |
| 8 | 31929 | 75.07 | 53.86 | 1.39x |
| 16 | 16384 | 152.20 | 117.91 | 1.29x |
| 32 | 16384 | 240.74 | 184.17 | 1.31x |
| 64 | 16384 | 336.46 | 271.23 | 1.24x |
| 128 | 16384 | 650.33 | 516.65 | 1.26x |
| 128 | 65536 | 660.13 | 518.28 | 1.27x |

`B=4 ctx=16384` 的 occupancy 在 2026-09-10 重 tune 为 `num_warps=4` /
`num_stages=2`（原 `8/3`），同一套 cudagraph 计时从 39.02 us 降到
31.90 us（两独立进程 31.90 / 31.96）。稳定收益集中在 `B>=8`，以及
`B=4` 且 `context>=4096`（`ctx=31929/32768` 接近持平）。`B=1 ctx=512`
与 `B=4 ctx=2048` 明显变慢，保留 dsplit occupancy JSON 是因为其它
shape 的收益来自同一 kernel，不是因为这两点更快。

decode 在 **width=513 + `num_splits=16`** 上，旧 kernel 会让相邻 split 的最后一个 `BLOCK_R` tile 重叠（`ceil(513/16)=33`）。2026-09-04 已用 `slots < window_end` 修掉；这些 case 仍留在 production bench 里，不要删。occupancy 性能表只覆盖已落地 JSON 的 `splits=1` 入口（公开 API 会自动再切 KV）。

---

## `triton_sparse_attention_prefill`

稀疏 GQA prefill。未传 `logical_regions` 时每 query 一个 program；传入
logical ids / positions / request ids 后走 tiled 路径（`block_q=128`）。
per-query 路径在 ``Q * G < CUs`` 时按 ``ceil(CUs / (Q*G))`` 切 packed slots，
再 ``merge_split_states``；生产 occupancy ``Q=128, G=1`` 已经盖住 72 CU，
``splits=1`` 仍走同一个 ``_sparse_attn_prefill_kernel``（grid 第三维为 1）。

```python
out, lse = triton_sparse_attention_prefill(
    query, key_cache, value_cache, packed_regions, region_counts,
    num_kv_groups=1, region_size=8, softmax_scale=None,
    block_regions=16, logical_regions=None, q_positions=None,
    request_indices=None, block_q=128, config=None,
)
```

- `query`：`[Q, H, D]`；`key_cache` / `value_cache`：`[tokens, G, D]`
- Q / K / V 以及写出的 `out` 最后一维必须连续（`stride(-1) == 1`），
  否则 `ValueError`。token / head stride 可以不连续
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

#### per-query launch JSON（2026-09-08）

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
| tuned | 0.681120 | 2.35x |

- HCU JSON：`triton/config/sparse_attn_prefill/`
  - `prefill-k{width}-hpg{hpg}-d{dim}-r{region}_{bf16|fp16}_{device}.json`
  - 当前：`prefill-k513-hpg8-d192-r8_bf16_BW1100.json`
- 无 JSON 或 `config={}` 时使用内置 `_per_query_tile_defaults`

实现：`triton/prefill.py`。溯源：Step-4 DSA handoff `sparse_attn.py`。

#### KV-split occupancy（2026-09-04）

公开入口 per-query 在 ``Q * G < CUs`` 时自动切 packed slots，再
``merge_split_states``；``slots < window_end`` 避免 ``BLOCK_R`` 尾 tile 重叠。
生产 occupancy ``Q=128, G=1`` 已有 128 workgroup（>72 CU），chooser 保持
``splits=1``，与上一小节 tuned JSON 同行，仍走同一个
``_sparse_attn_prefill_kernel``。小 ``Q`` 才吃 occupancy（与 decode 的
``B=4`` 同类）。tile 仍复用上一小节的 ``prefill-k513-..._BW1100.json``，
``num_warps`` / ``num_stages`` / ``waves_per_eu`` 都会传给 launch。

| 环境项 | 值 |
| --- | --- |
| GPU | BW1100（gfx938），72 CU |
| Shape | `H=8, G=1, packed_width=513, D=192, region_size=8, cache_regions=4096` |
| 输入 / 输出 dtype | BF16 / BF16 |
| 计时 | `triton.testing.do_bench`，warmup 10 ms、rep 30 ms、中位数 |
| tile | `[BLOCK_R,BLOCK_H,BLOCK_D]=[8,16,256]`，`warps=4`，`stages=1` |

| Q | splits | splits=1 (ms) | auto-split (ms) | 相对 splits=1 |
|---|---:|---:|---:|---:|
| 128 | 1 | 0.682560 | 0.682640 | 1.00x |
| 4 | 18 | 0.572480 | 0.056320 | 10.2x |

``Q=4`` 的 grid 从 4 个 workgroup 升到 72；``Q=128`` 不额外加 wave（extra-wave
split 在证据形状上不更快）。scalar kernel 已合并为带 split 轴的
``_sparse_attn_prefill_kernel``。

#### tiled launch JSON（2026-09-08）

生产 tile `Q=128`、`block_q=128`。实测 tile union=528，配置按
`next_power_of_2` 落到 `u=1024`。`latency_ms` 只含 tiled Triton kernel
（与 xprof `.perf` 对齐）；`_build_prefill_tiles` 在 `do_bench` 外单独记为
`tile_build_ms`（和 decode 把 metadata 放在计时区外同一做法）。
公开 API `triton_sparse_attention_prefill` 每次调用仍会重建 tiles。

| 实现 | launch `[BLOCK_Q,BLOCK_R,BLOCK_D,warps,stages,waves]` | kernel (ms) | 相对 default |
|---|---|---:|---:|
| default | `[128,4,256,4,2,—]` | 1.139039 | 1.00x |
| tuned | `[128,8,256,8,1,1]` | 0.706560 | 1.61x |

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

稀疏 GQA decode。`num_splits==1` 按设备 CU 数自动 KV-split（
`ceil(CUs / (B·G))`，再按 packed 宽度封顶），merge 后返回最终 `out` /
`lse`；`>1` 时按调用方份数切，返回带 leading split 轴的 partial，交给
`triton_merge_split_states`。split 窗口用 `slots < window_end` 做分区，
避免 `ceil(packed_width / splits) % BLOCK_R != 0` 时相邻 split 重叠计数。
K 按 `[BLOCK_D, tokens]` 加载，QK 走 `tl.dot(Q, K)`，不再 `tl.trans(K)`；
V 仍是 `[tokens, BLOCK_D]` 做 `P @ V`。

```python
out, lse = triton_sparse_attention_decode(
    query, key_cache, value_cache, packed_regions, region_counts,
    kv_seqlens, num_kv_groups=1, region_size=8, softmax_scale=None,
    num_splits=1, block_regions=16, config=None,
)
```

- `query`：`[B, H, D]`；`kv_seqlens`：`[B]` int32
- Q / K / V 以及写出的 `out` 最后一维必须连续（`stride(-1) == 1`），
  否则 `ValueError`。token / head stride 可以不连续
- `packed_regions`：`[G * B, packed_width]` **int64**
- `config=None` 按 occupancy 读 JSON（`batch=B`，`context=kv_seqlens[0]`）；
  自动切分后若没有 `s{N}` 文件，回退 `s1` JSON，再缺则走内置 tile。
  `config={}` 跳过查找，用内置 tile。

### BW1100 TP8 性能

环境与 prefill 相同。metadata（`triton_decode_sparse_meta`）在计时区外。
记录表仍只引用默认生产点 `(B=4, ctx=16384, splits=1)`。加速比定义为
`default latency / tuned latency`，大于 1 表示 JSON 更快。CSV 来自
`scripts/benchmarks/triton/step4_dsa/result/`。GPU 未锁频，数据用于同轮
相对比较，不作为稳定性能门槛。

#### occupancy JSON（2026-09-03）

当时 kernel 未做 KV-split，`grid=(B, G, 1)` 只有 4 个 workgroup。本表只
对照 launch JSON，没有改 kernel。

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
  - 当时：`decode-k513-s1-b4-ctx16384-hpg8-d192-r8_bf16_BW1100.json`
    （`[2,16,256, warps=2, stages=2, waves=1]`）
  - 缺文件时静默使用内置 `_decode_tile_defaults`

#### KV-split occupancy（2026-09-04）

公开入口 `num_splits=1` 自动切 packed slots，再 `merge_split_states`。
72 CU 上 `ceil(72 / (B·G)) = 18` splits，grid `(4, 1, 18)` = 72 workgroup。
`latency_ms` 含 decode + merge（生产 e2e）；tile 仍用上一小节的 `s1` JSON，
没有按 18-way 再 tune 一份。

| 环境项 | 值 |
| --- | --- |
| GPU | BW1100（gfx938），72 CU |
| Shape | `B=4, context=16384, splits=1（launch 18）, H=8, G=1, topk=512, D=192` |
| 输入 / 输出 dtype | BF16 / BF16 |
| 计时 | `triton.testing.do_bench`，warmup 10 ms、rep 30 ms、中位数 |
| default | `[BLOCK_R,BLOCK_H,BLOCK_D]=[8,16,256]`，未指定 warps/stages/waves |
| tuned | `[2,16,256, warps=2, stages=2, waves=1]`（复用 s1 JSON） |

| 实现 | 延迟 (ms) | 相对 default | 相对 09-03 同行 |
|---|---:|---:|---:|
| default | 0.074080 | 1.00x | 11.3x |
| tuned | 0.058080 | 1.28x | 11.3x |

相对 09-03 的倍数是 `当时同行延迟 / 本行延迟`（default `0.834720`，tuned
`0.656000`）。收益来自 occupancy（4 → 72 workgroup），不是更大的
`BLOCK_R` / warps / stages。

- HCU JSON：当时为 `decode-k513-s1-b4-ctx16384-hpg8-d192-r8_bf16_BW1100.json`
  （`warps=2, stages=2`）；无对应 `s18` 文件时回退该 `s1` tile

#### QK layout + occupancy JSON（2026-09-08）

`_sparse_attn_decode_kernel` 把 K 改成 `[BLOCK_D, tokens]` 加载，QK 直接
`tl.dot(Q, K)`，去掉 `tl.trans(K)`；V 仍是 `[tokens, BLOCK_D]` 做 `P @ V`。
公开入口和内部 `sparse_attention_decode` 对 vLLM 一致：`num_splits=1`
自动 KV-split + merge（快路径），`>1` 返回 partial。tuner 的 `--splits`
按调用方命名 JSON：`--splits 1` 测的就是这条快路径（B=4 时 launch 18-way），
文件仍是 `s1`。本表记该默认 shape 的 **s1 JSON / tuner**。

| 环境项 | 值 |
| --- | --- |
| GPU | BW1100（gfx938），72 CU |
| Shape | `B=4, context=16384, splits=1, H=8, G=1, topk=512, D=192` |
| 输入 / 输出 dtype | BF16 / BF16 |
| 计时 | `triton.testing.do_bench`，warmup 10 ms、rep 30 ms、中位数 |
| tuned | `[2,16,256, warps=4, stages=3, waves=2]` |

| 实现 | 延迟 (ms) |
|---|---:|
| tuned | 0.053120 |

- HCU JSON：`decode-k513-s1-b4-ctx16384-hpg8-d192-r8_bf16_BW1100.json`
- 无对应 `s18` 文件时回退该 `s1` tile；`config={}` 走内置 `_decode_tile_defaults`

实现：`triton/decode.py`（`sparse_attention_decode`）。溯源：Step-4 DSA
handoff `sparse_attn.py`。
