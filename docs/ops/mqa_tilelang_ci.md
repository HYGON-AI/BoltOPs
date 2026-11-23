# TileLang MQA 的 QSA / 非 QSA 测试对齐

非 QSA 测试沿用 QSA 的 pytest 与 optest 入口；不修改 QSA 实现、
CI 触发条件或公共收集规则。QSA prefill 测试此前已有的 Q=256/1024
性能测点扩展一并纳入，保持两侧默认性能网格一致。额外的加权、FP8
存储和流水线测试保留。测试文件必须随实现提交，远端 CI 才能收集到。

## 正确性

### 最新主线的 CI 分工

GitLab 迁移基线 `main` 的 `f97d4ea` 中，`.gitlab-ci.yml` 仅在
目标为 `main` 的 MR 中构建 DTK 27.04 / Python 3.12 wheel；合入后
另行发布。当前没有正确性或性能任务，GitLab 构建通过不代表 GPU
正确性和性能测试通过。

GitHub 本次迁移基线为 `release/0.1.0` 的 `3fa0991`。其 workflow
在 PR 中构建 DTK 26.04 / Python 3.10 / Torch 2.11.0 和 DTK 26.10 /
Python 3.12 / Torch 2.13.0，并执行正确性与性能任务。个人 Fork 的 PR
须由有相应仓库权限的人员添加 `ok-to-test` 标签，通过 `authorize-ci`
检查后才能使用内部 Runner；创建 PR 不代表测试已获准执行。

两端触发条件以各自目标分支中的实际配置为准。GitHub 基线保留自己的
`.gitlab-ci.yml` 和 `.github/` 文件，本次仅迁移非 QSA 实现与测试，
不将 GitLab 的构建工作流覆盖到 GitHub。

下表中的默认 / slow 是 pytest 分层。GitLab 仍保留这些测试文件，
以便本地或启用测试的 CI 使用，不能据此声称当前 GitLab 自动运行它们。

| QSA 覆盖 | 非 QSA 对应覆盖 | pytest 分层 |
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

主线 GitHub 的正确性脚本执行
`python -m pytest test -s --tb=short -m "not slow"`。
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

新主线还将 `test_qsa_mqa_prefill_fp8_matches_torch` 标记为已知失败并
跳过；该标记原样保留。非 QSA FP8 测试独立运行，不继承这一跳过标记。

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

## 本机验证（迁移前，2026-11-23）

- gfx938 / GPU 6：TileLang 默认用例、配置测试与包导出检查，117 passed。
- 新增及相关正确性检查 31 项，加上完整的 36 组 BF16 decode bench，
  合计 67 passed。每组新 decode bench 在计时前对比 PyTorch FP32
  参考输出；本轮 bench 使用 1/3 ms 的 warmup/repeat 预算验证可执行性。
- 通过 `run_operator_benchmarks.py` 各运行一个 prefill/decode 代表用例，
  成功生成两份 CSV，并检查 backend、有效延迟和错误字段。
- 对应的主要参数网格、slow/bench 标记已在实际 pytest 收集结果中
  核对；未全量重跑所有 slow 用例。

本机使用 Torch 2.11.0 / DTK2604 与 TileLang 0.1.15 开发源码。
以上记录对应旧基线的验证，不代表迁移后的代码已验证；新主线 CI 分工
见上文，也不能以本机结果代替 DTK 27.04 的实际构建结果。

## 本机验证（GitLab 迁移后，2026-11-23）

基于 GitLab `main` 的 `f97d4ea`，迁移四笔非 QSA 提交后完成本地回归。
测试期间保持 kernel 源码不变，启用 `--runslow` 并排除 `bench`：

```bash
python -m pytest test/mqa_logits/test_tilelang*.py test/test_package_smoke.py \
  --runslow -m 'not bench' -q --tb=short
```

- 共 460 项：457 passed、1 skipped、2 failed；中断后按剩余用例续跑，
  以唯一 pytest nodeid 汇总覆盖。
- 非 QSA BF16 prefill 95 项、FP8 prefill 22 项、BF16/FP8 decode 173 项
  全部通过。数值用例对照 PyTorch FP32 参考结果。
- 两项失败均在不含非 QSA 改动的纯 `f97d4ea` 上复现：QSA FP8 prefill
  H32 无可用 layout；QSA BF16 persistent decode H32/S32 的 ThreadSync
  无法处理有 3 个参与线程的分支。上游原有 FP8 prefill skip 保留。
- QSA BF16/FP8 prefill、decode 四个核心文件与 `f97d4ea` 逐字节一致。
  非 QSA BF16 decode 对齐上游的 `TL_Z3_RLIMIT=50000` 编译配置。
- 本地原始 wheel 构建成功，43 个 MQA Python 文件与验证源码一致；
  安装后 package smoke 测试 8 项通过。

验证环境为 Python 3.10 / Torch 2.11.0 / DTK2604 / TileLang 0.1.15
开发源码，使用 gfx938 GPU 6。本次记录为正确性与本地打包验证，不包含
迁移后的性能复测，也不替代 GitLab DTK 27.04 / Python 3.12 构建。

## GitHub 迁移核对（2026-11-23）

从 `release/0.1.0` 的 `3fa0991` 创建独立功能分支，迁移非 QSA 的
实现、测试和编译适配。64 个 MQA 源码和测试文件与上述已完成全量回归
的 GitLab 分支逐字节一致；QSA 四个核心 kernel 与 GitHub 基线一致。
GitHub 的 CI、版本号及包版本检查保留上游内容，未复制 GitLab 的
工作流改动。

在同一本机环境中运行默认 CI 分层的 MQA 与 package smoke 子集：

```bash
python -m pytest test/mqa_logits/test_tilelang*.py test/test_package_smoke.py \
  -m 'not slow and not bench' -q --tb=short
```

结果为 116 passed、1 skipped、402 deselected。该次未重新运行 slow
网格；其源码和 MQA 实现已与 GitLab 全量验证版本核对一致。本次子集
验证不代表已执行 GitHub 全仓库 CI、DTK 26.10 环境或性能测试。
