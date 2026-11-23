# TileLang MQA 的 QSA / 非 QSA 测试对齐

非 QSA 测试沿用 QSA 的 pytest 与 optest 入口；不修改 QSA 实现、
CI 触发条件或公共收集规则。QSA prefill 测试此前已有的 Q=256/1024
性能测点扩展一并纳入，保持两侧默认性能网格一致。额外的加权、FP8
存储和流水线测试保留。测试文件必须随实现提交，远端 CI 才能收集到。

## 正确性

| QSA 覆盖 | 非 QSA 对应覆盖 | CI 分层 |
| --- | --- | --- |
| BF16 prefill 公共入口，Q=41/513、K=96、H=4、不同有效范围 | `test_weighted_prefill_public_matches_torch` | Q=41 默认；513 为 slow |
| Common prefill 非整块 Q=33、K=83 | `test_common_weighted_prefill_handles_partial_rows_without_host_padding` | slow |
| 多 Key 长度、WASP 尾块、phase 间距、head/block_q 组合 | `test_tilelang_mqa_weighted.py` 中对应的边界测试 | slow |
| WASP persistent 非完整轮次，Q=9152/9280 | `test_wasp_weighted_persistent_partial_rounds` | slow |
| FP8 prefill Q=41、K=96、H=4 | `test_weighted_fp8_public_matches_torch`（另外覆盖 H8/32/64） | 默认 |
| FP8 prefill Q=4608、K=8192、H=32/64 | `test_weighted_fp8_heuristic_shape_matches_torch` | slow |
| BF16 decode page=16/H4 默认，以及不同 page/head | `test_weighted_decode_public_heads_and_pages`（BF16/FP8） | page16/H4 默认；其余 slow |
| FP8 packed decode H4 默认，H16/32/64 扩展 | 同上；另有独立存储、scale、权重测试 | H4 默认；扩展 slow |
| decode group128，以及 H16/32/64 × S2/4/8/16/32 | `test_weighted_decode_group128_ordinary_matches_torch` / `test_weighted_decode_group128_persistent_matches_torch` | slow；额外覆盖 H4/H8 和 FP8 |
| decode suffix split 策略的 B/K 组合 | 原 QSA 策略测试，加上 `test_weighted_decode_public_schedule_buckets` 的公共入口验证 | 策略默认；大尺寸入口 slow |

QSA 和非 QSA 复用的 mask kernel 仍由原 QSA mask 测试验证；非 QSA
公共入口另外检查无效区间为 `-inf`。数值参考使用各自的 PyTorch FP32
公式；非 QSA 包含带正负号的 head 权重，不要求两侧输出相等。

CI 默认执行 `python -m pytest test -s --tb=short -m "not slow"`。
`bench` 用例在该阶段跳过；slow 网格不会自动执行。MQA 全量正确性命令：

```bash
python -m pytest test/mqa_logits --runslow -m "not bench" -q
```

WASP 与非 QSA decode 需要 gfx938；跳过用例不代表已验证 GPU 正确性。

## 性能

| 项目 | 非 QSA 对齐方式 |
| --- | --- |
| BF16 prefill | 与当前 QSA 一样：Q=256/1024/4096/8192/16384，K=4096，每个 CTA 访问交替的 2048-Key 区间，H=4、D=128，`apply_scale=False`；仅计时 compute kernel |
| BF16 decode | 与 QSA bench 的配置范围一样：B=16/32/64/128，Context=8192/32768/65536，H=4/32/64，page16、D128、连续页表、完整 Context、运行时 `sqrt(D)` scale；仅计时 compute kernel |
| 额外测点 | prefill Q=64/128 保留为 slow bench；已有 decode B16/Context8192 的 BF16/FP8 head 网格保留 |

QSA decode 的 `test_bench_qsa_mqa_decode_cooperative_load` 当前因已知
VMFault 无条件跳过。非 QSA 的对应网格可独立执行，但不能据此宣称 CI
已完成两侧的性能比较。QSA 当前没有 FP8 prefill bench，因此本次不
额外新增该性能网格。

两侧使用各自生产默认分块；相同 workload 不要求内部 tiling 一致。
这里的 prefill scale 设置遵循当前 QSA bench，与包含运行时 scale 的
独立性能报告口径不同。

性能用例使用 `@pytest.mark.bench` 与 `time_ms`，由 optest 自动发现，
无需注册。非 QSA 性能入口：

```bash
python scripts/run_operator_benchmarks.py \
  --ops tilelang_mqa_weighted tilelang_mqa_decode_weighted
```

结果默认生成到 `optest_results/`，框架说明见
[optest 性能基准](../optest_perf.md)。该入口记录性能数据，不自动完成
QSA 对照百分比、spill 或 bank conflict 分析。

## 本机验证（2026-11-23）

- gfx938 / GPU 6：TileLang 默认用例、配置测试与包导出检查，117 passed。
- 新增及相关正确性检查 31 项，加上完整的 36 组 BF16 decode bench，
  合计 67 passed。每组新 decode bench 在计时前对比 PyTorch FP32
  参考输出；本轮 bench 使用 1/3 ms 的 warmup/repeat 预算验证可执行性。
- 通过 `run_operator_benchmarks.py` 各运行一个 prefill/decode 代表用例，
  成功生成两份 CSV，并检查 backend、有效延迟和错误字段。
- 对应的主要参数网格、slow/bench 标记已在实际 pytest 收集结果中
  核对；未全量重跑所有 slow 用例。

本机使用 Torch 2.11.0 / DTK2604 与 TileLang 0.1.15 开发源码。
CI 配置安装 TileLang 0.1.12，并包含多个 Python/Torch 组合；上述本机
结果不替代这些 CI 环境的实际流水线验证。
