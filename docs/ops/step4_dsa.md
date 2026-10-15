# Step-4 DSA 系列算子说明

`boltops.step4_dsa` 公开 Step-4 Dynamic Sparse Attention 的 **12 个核心数值 Triton API**（合同 `step4-dsa-ops-v1`），另提供 `triton_csa_prefill` 便利封装，共 13 个可调用入口。Triton kernel 在 `boltops/step4_dsa/triton/`；模型几何在 `geometry.py`；测试和性能测试输入构造在 `test/step4_dsa/`（不进 wheel）。当前仅实现 Triton 后端。

不包含 vLLM planner / KV 缓存写入，也不包含 `kernel.py` 里的 MoE FP8 / 主注意力 `fused_qknorm_rope`。sparse prefill/decode 可以**读取**框架侧写入的 FP8 KV cache（`float8_e4m3fn` + per-tensor `k_scale`/`v_scale`）；量化写入仍由框架完成。

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
| `triton_sparse_attention_prefill` | 稀疏 GQA prefill；tiled 路径 `block_q=128`；per-query 在 `Q*G` 填不满 CU 时自动 KV-split；可读 FP8 KV cache |
| `triton_sparse_attention_decode` | 稀疏 decode；默认 `num_splits=1` 自动按 CU 做 KV-split；可读 FP8 KV cache |
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
pytest -q test/step4_dsa/test_step4_dsa_decode.py test/step4_dsa/test_step4_dsa_prefill.py --bench -s
python3 scripts/run_operator_benchmarks.py --help
```

正确性对 `test/step4_dsa/ref.py` 的 FP32 Torch oracle，禁止 Triton-vs-Triton 作为正确性门禁。显式 split decode 在合并 partial 后验证。

当前性能测试由 `test_step4_dsa_decode.py` / `test_step4_dsa_prefill.py` 内的 case 列表定义，共 20 个配置覆盖点（decode 9 个、prefill 11 个），包含 BF16 和 FP8。decode bench 使用 `num_splits=1` 的自动切分入口；宽度 513 的显式 split 边界由正确性测试覆盖。性能测试输入由 `test_step4_dsa_decode.py` / `test_step4_dsa_prefill.py` 各自构造，TP8 单分片几何参数由测试公共辅助文件 `_common.py` 提供；不再保留独立的历史 benchmark case 清单。

统一性能入口会生成 `step4_dsa_decode.csv` / `step4_dsa_prefill.csv`，对应 raw 文件为 `step4_dsa_decode.raw.jsonl` / `step4_dsa_prefill.raw.jsonl`（由测试文件名派生），调用参数见 [`docs/optest_perf.md`](../optest_perf.md)。历史 JSONL/CSV 和 `optest_results_step4/` 为测量证据，不参与算子运行；此次整理保留数据，仅将本地结果目录加入忽略规则。

在 handoff 仓库内仍可把本包作为 vendor 模块：

```bash
# 在 handoff 仓库内，把本包装成 vendor 模块
python3 -m operator_parity.run_all --suite dsa --ops-module boltops.step4_dsa
```

## 当前配置与代码组织（2026-10-14）

- `common.py` 统一 Python 侧设备别名、JSON 读取与匹配、launch 参数过滤、自动 KV-split 策略；prefill/decode 保留各自 shape 和内核参数合同。
- 配置按设备集中存储：`config/sparse_attn_decode/sparse_attn_decode_BW1100.json`、`config/sparse_attn_prefill/sparse_attn_prefill_BW1100.json`、`config/sparse_attn_prefill/sparse_attn_prefill_tiled_BW1100.json`。每个文件通过 `configs[].match` 匹配 shape；更具体的 exact/range 约束优先。
- `config=None` 查设备 JSON；`config={}` 使用内置默认值。decode 不读取 `kv_seqlens[0]` 做配置匹配，也不按 context 或自动 split 数查找旧文件。
- `head_dim=192` 当前拆成 `64+64+64`；BF16/FP8 和 decode pipeline 变体有不同访存/编译行为，保留专用内核，不以文本相似为由合并。
- BW1100 的性能验证使用 `AMDGCN_USE_BUFFER_OPS=1`（与下方历史测量设置一致）。可以在上述 pytest 命令前显式加该环境变量；其他平台不应直接套用此 HIP 编译设置。
- 已移除未实际使用的离线调优脚本及专属造数、测试；已有三个设备 JSON 配置、运行时读取与匹配逻辑、算子正确性和性能测试继续保留。`test_helpers.py` 仅测试运行时配置辅助逻辑。

## BW1100 三类稀疏注意力 baseline 与优化版对比（2026-10-14）

本轮覆盖 `boltops/step4_dsa/triton/config/` 中三个 BW1100 JSON 的全部 **20 个 shape**：decode 9 个、普通 prefill 9 个、tiled prefill 2 个。Baseline 为优化前算子及其默认配置，优化版为当前算子及其匹配的调优 JSON；收益同时包含 kernel 实现、自动 KV-split 与 launch 配置的变化，**不是仅替换 JSON 的收益**。两版时间均在本轮测量，未使用 JSON 中的历史 `latency_ms`。

### 测试环境、输入与计时口径

- 最终测量全部使用物理 **GPU 1：BW1100 / gfx938，72 个计算单元**；PyTorch 2.10.0、Triton 3.6.0、HIP 6.3.26093。
- 共同几何：`H=8, G=1, D=192, region_size=8, topk=512, packed_width=513`；query 均为 BF16，表中的 KV 类型表示 KV cache 数据类型，FP8 为 `float8_e4m3fn`，`k_scale=v_scale=1.0`。
- 输入复用 `test/step4_dsa/test_step4_dsa_decode.py` / `test/step4_dsa/test_step4_dsa_prefill.py` 的生成函数，`seed=20260831`；每个 shape 的两版共享同一组输入。decode 的 context 为 16,384 tokens；两类 prefill 的 KV cache 为 32,768 tokens。
- 使用与 pytest benchmark 相同的 `boltops.utility.bench.time_ms`。每版 **5 轮**，交替测量顺序，每轮取中位数，再取 5 轮中位数；`repeat=50ms`。decode / 普通 prefill 使用 graph replay 及其内部预热；tiled 使用 event、`warmup=10ms`，包含 tile 元数据构建与主机同步。
- 优化版计入所有内部 split / merge kernel；decode 的 selector 和输入 metadata 生成在计时外。这里是算子测量，不代表完整模型端到端加速；tiled 的 event 时间也不能与历史纯 kernel 调优时间直接比较。
- 运行前连续检查 GPU 空闲状态，每个 shape 和每轮测量前再次检查利用率与其他显存占用，稳定性复测还检查其他进程的计算单元占用；检测到活动任务时等待或拒绝结果。波动较大的项已复测。GPU 未预约独占，采样间极短的外部干扰不能完全排除。

时间单位为 **μs**，保留三位小数；加速比 = `baseline / optimized`，耗时降低 = `(1 - optimized / baseline) × 100%`，均按未舍入数据计算。

### Baseline 配置口径

- BF16 / FP8 baseline 均使用优化前的通用 kernel；FP8 不走 `head_dim=192` 专用分支。
- Baseline 不读取调优 JSON，使用默认 `block_regions=16` 和共享内存自适应收缩，`BLOCK_H=16, BLOCK_D=256`；decode 使用 `num_splits=1`，不增加自动 KV-split。普通 prefill / decode 沿用 Triton 默认编译参数；tiled 使用 `block_q=128, num_warps=4, num_stages=2`。
- **19 个 shape 严格使用 baseline 默认配置**；BF16 tiled 默认配置无法启动，以下明确区分失败与 `block_regions=4` 补测。其他成功的默认配置实际 `BLOCK_R=8`，由资源适配决定，不属于最优配置选择。

### Decode：9 个 shape

`B` 为请求 batch，context 固定为 16,384 tokens；所有行的 baseline 均为严格默认配置，计时包含优化版的 split / merge。

| KV 类型 | B | Baseline（μs） | 优化后（μs） | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 841.389 | 11.514 | 73.08× | 98.6% |
| BF16 | 4 | 853.941 | 19.857 | 43.00× | 97.7% |
| BF16 | 8 | 855.333 | 28.713 | 29.79× | 96.6% |
| BF16 | 32 | 857.404 | 73.010 | 11.74× | 91.5% |
| BF16 | 64 | 859.472 | 140.353 | 6.12× | 83.7% |
| BF16 | 128 | 1716.057 | 260.210 | 6.59× | 84.8% |
| FP8 | 1 | 1241.214 | 14.757 | 84.11× | 98.8% |
| FP8 | 4 | 1195.323 | 15.847 | 75.43× | 98.7% |
| FP8 | 8 | 1237.552 | 27.236 | 45.44× | 97.8% |

本轮 decode 加速比范围为 **6.12–84.11×**。

### 普通 Prefill：9 个 shape

`Q` 为 query token 数，KV cache 固定为 32,768 tokens；使用 per-query 路径，所有行的 baseline 均为严格默认配置。

| KV 类型 | Q | Baseline（μs） | 优化后（μs） | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 823.546 | 12.218 | 67.40× | 98.5% |
| BF16 | 4 | 822.922 | 18.933 | 43.47× | 97.7% |
| BF16 | 8 | 823.185 | 19.590 | 42.02× | 97.6% |
| BF16 | 32 | 822.862 | 45.336 | 18.15× | 94.5% |
| BF16 | 64 | 822.922 | 75.626 | 10.88× | 90.8% |
| BF16 | 128 | 1643.514 | 128.579 | 12.78× | 92.2% |
| BF16 | 512 | 6583.084 | 473.834 | 13.89× | 92.8% |
| BF16 | 4096 | 47092.068 | 3277.893 | 14.37× | 93.0% |
| FP8 | 128 | 1939.311 | 165.523 | 11.72× | 91.5% |

本轮普通 prefill 加速比范围为 **10.88–67.40×**。

### Tiled Prefill：2 个 shape

`Q=128, block_q=128`，KV cache 为 32,768 tokens；event 计时包含 tile 元数据构建及主机同步。

| KV 类型 | Q | Baseline（μs） | 优化后（μs） | 加速比 | 耗时降低 | Baseline 配置 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 128 | 3068.639（补测） | 575.520 | 5.33×（补测） | 81.2% | 默认超限；仅改 `block_regions=4` |
| FP8 | 128 | 4188.799 | 595.200 | 7.04× | 85.8% | 严格默认配置 |

**BF16 tiled 的严格默认配置无有效时间**：默认 `block_regions=16` 经资源适配收缩至 `BLOCK_R=8` 后，仍需要 **66,560 B** 共享内存，超过硬件上限 **65,536 B**，报 `OutOfResources`。表中的 BF16 baseline 沿用现有 `test_step4_dsa_prefill.py` 的 `block_regions=4` 补测，仍使用优化前的通用 kernel，其余默认参数不变；对应加速比和耗时降低仅适用于这次补测，不能作为严格默认配置的收益。FP8 tiled 没有采用该补测参数。

### 一致性检查

20 个可运行对比的输出和 LSE（log-sum-exp）一致性检查全部通过。该检查用于本次两版结果对照，**不替代前述 FP32 Torch oracle 正确性门禁**。

## 历史优化与性能记录

以下日期化记录保留用于溯源；其中 `prefill-k...` / `decode-k...` 文件名、旧 benchmark 脚本、`128+64` 分块及当时的配置选择策略均为历史状态，不代表当前接口。当前配置和运行入口以“当前配置与代码组织”一节为准。历史性能数字没有在此次代码整理中重新测定。

## 接口对齐、健壮性与配置规范化（2026-09-12）

- **接口与健壮性增强**：修复 step4_dsa 系列算子在空行输入、极值边界等情况下的潜在异常，全面对齐 step4_dsa 标准调用接口，强化张量维度连续性（如 `_require_contiguous_last_dim`）及合法性校验。
- **配置接口规范化**：规范化 launch config 的 JSON 格式与字段解析体系；支持统一的设备级集中式配置匹配与跨平台编译能力自适应过滤。

## 已知问题

### BW1100 短上下文 decode 有效槽位分区（2026-09-09）

Decode launch 配置支持 `BALANCE_LIVE`（默认 false）。开启时，自动 split
按每个 metadata row 的设备端 `region_counts` 分配窗口：
`slots_per_split = ceil(min(count, packed_width) / num_splits)`。
无需 host scalar 读取或 metadata 拷贝；grid 和 merge 输出形状保持不变。
显式 split、返回 partial states、单 split 均忽略此选项，保持原窗口合同。

`AUTO_KV_SPLITS` 选择整次 launch 共用的 split 数；`BALANCE_LIVE` 则按每行
运行时 `region_counts` 重新划分窗口，两者作用维度不同，前者不能替代后者。
历史上曾在三个短上下文 BW1100 点开启并得到下表收益；后续重新调优的当前
BW1100 配置均选择 false，因此 JSON 省略该默认值，但保留内核支持，
供高度 ragged 的自定义输入通过显式 config 继续评估。动态窗口在长上下文没有稳定收益，不应
全局启用。`config={}` 仍使用原默认策略。

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
    k_scale=1.0, v_scale=1.0,
)
```

- `query`：`[Q, H, D]`，BF16；`key_cache` / `value_cache`：`[tokens, G, D]`，BF16 或 `float8_e4m3fn`
- `k_scale` / `v_scale`：FP8 cache 的 per-tensor 反量化标量，默认 `1.0`。真实值是 `cache * scale`。`k_scale` 在 host 折进 `softmax_scale`；`v_scale != 1` 时才乘在 FP32 输出上（`APPLY_V_SCALE` 编译期特化）
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
`scripts/perf/triton/step4_dsa/result/`。GPU 未锁频，数据用于同轮
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

#### 编译器调度与寄存器压力调优（2026-09-12）

在 per-query autotune 探索空间中引入 HIP 编译器底层调度选项：`empty_arrive_after_mmac`、`max_memory_cluster_dwords`、`misched_regpressure`。在生产 shape `Q=128, H=8, G=1, topk=512, cache_regions=4096, D=192` 上：

| 实现 | launch 调度参数 | 延迟 (ms) | 相对上一版 | 相对 default |
|---|---|---:|---:|---:|
| tuned (09-08) | `[8,16,256, warps=4, stages=1, waves=2]` | 0.681120 | 1.00x | 2.35x |
| tuned (09-12) | 扩展调度选项 (`misched_regpressure` 等) | 0.615680 | **1.1063x** | 2.60x |

加速比：`0.681120 / 0.615680 = 1.1063x`。

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
    k_scale=1.0, v_scale=1.0,
)
```

- `query`：`[B, H, D]`，BF16；`kv_seqlens`：`[B]` int32
- `key_cache` / `value_cache`：BF16 或 `float8_e4m3fn`；`k_scale` / `v_scale` 与 prefill 相同
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
`scripts/perf/triton/step4_dsa/result/`。GPU 未锁频，数据用于同轮
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

#### 综合性能优化与调度/配置升级（2026-09-12）

针对 decode 算子综合实施以下优化组合并在生产点完成加速比测定：
1. **K Cache 访存布局对齐**：调整 K 加载为 `[BLOCK_D, tokens]` 布局，使 QK 矩阵乘直接执行 `tl.dot(Q, K)`，消除片上 `tl.trans(K)` 的寄存器与指令开销。
2. **非 2 次幂 head_dim 双块切分（dsplit）**：针对 `head_dim=192` 拆为 `128 + 64` 双块并行计算并累加，彻底消除 pad 到 256 带来的 25% 访存及算力浪费。
3. **Log2 域 Online Softmax 加速**：Softmax 缩放因子折入 $\log_2(e)$ 并采用硬件高吞吐 `exp2` 替代原生 `exp`，保存 LSE 时还原自然对数，精度严格在 step4_dsa 允差范围内通过验收。
4. **归一化单次倒数复用**：逐元素除法优化为每个 head 单次计算倒数 `div_rn` 并广播乘法复用，大幅减少高延迟除法指令。
5. **动态槽位负载均衡（BALANCE_LIVE）**：在 GPU 端基于各 request 实际有效 token 动态均衡 `slots_per_split`，消除切片空转。
6. **硬件 Buffer 访存指令**：开启 `AMDGCN_USE_BUFFER_OPS=1` 走硬件原生 Buffer 操作，显著降低 VGPR 压力。

##### 生产基准加速比测定（B=4, context=16384）

| 实现 | 优化项 | 延迟 (us) | 加速比 |
|---|---|---:|---:|
| HEAD 基线 (8bbee9b) | padded `BLOCK_D=256` | 44.40 | 1.00x |
| 最终优化版本 | dsplit + exp2 + 倒数复用 + BALANCE_LIVE + BufferOps | 24.0815 | **1.8437x** |

加速比：`44.40us / 24.0815us = 1.8437x`（耗时降幅约 45.8%）。

实现：`triton/decode.py`（`sparse_attention_decode`）。溯源：Step-4 DSA
handoff `sparse_attn.py`。
