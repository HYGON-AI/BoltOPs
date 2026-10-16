# GEMM 支持与优化记录

本文件维护 GEMM 的需求、验证进度和调优结论。需求来源为 `/xukang/GEMM支持清单.html`（生成日期 2026-10-12），共 511 个形状、2504 条算子实例、9 类算子与精度组合；另有 ResNet50 INT8 隐式 GEMM 卷积 24 类及末端 FC。

完整 511 行需求记录在 [GEMM 支持需求清单](REQUIREMENTS.md)，按算子及精度分为 9 张 Markdown 表格，保留形状、batch/experts、阶段、频次、模型与 scale 信息，并提供状态和验证/优化记录列。该清单是待验证需求；INT8 卷积和 FC 仍以 HTML 为准。

## 文档和配置的位置

- 本文件：支持范围、待办、验证进度、新实验摘要和结论。
- [RESULTS.md](RESULTS.md)：已有 FP16 WASP/WDRA 性能结果、历史实验与回归记录；原内容从 `boltops/gemm/triton/configs/gemm_fp16/RESULTS.md` 迁移。
- [VGPR.md](VGPR.md)：已有 WDRA VGPR 分配策略和约束；原内容从同目录迁移。
- `boltops/gemm/triton/configs/`：保留可执行调优配置，性能说明不再新增在这里。
- [GEMM API 文档](../ops/gemm.md)：公开算子接口和语义。

历史 FP16 调优结果不代表 BF16、FP8 或其他精度已验证。编译成功、数值正确、性能达标分别记录；PMD 仿真结果与真实设备结果分别记录。

## 基线环境（2026-10-14）

| 项目 | 基线 |
| --- | --- |
| 容器 | `kxu_pmd_8569` |
| 独立目录 | `/xukang/code/pmd-triton-sessions/gemm-opt-20261014` |
| Triton | `release/3.6.x`，`2003073a691387e54f5c6f5369bf5faf0652cee4`，已确认远端最新 |
| boltOps | `shb_bu_wasp`，内核基线 `4174c756c701185574eda72e473def361289fb6f`；文档在此分支维护，通过 MR 合入 main |
| 当前分支 | boltOps 为 `shb_bu_wasp`，Triton 为 `release/3.6.x`；不另设工作分支 |
| LLVM | `188a510cd6a165dd18510235daabc4bebc3c27d3`，复用匹配的既有 LLVM 构建 |
| 构建和缓存 | 独立 `venv/`、`build/`、`cache/`、`logs/` |

文档和后续 GEMM 优化在 `shb_bu_wasp` 分支维护，再通过 MR 提交到 `main`。本次 MR 只包含 GEMM 内核、调优配置、测试及文档；FA/MLA WDRA 实验保留在独立分支 `shb_bu_wasp_fa_mla`，另行提交。

进入容器后：

```bash
source /xukang/code/pmd-triton-sessions/gemm-opt-20261014/env.sh
# 重编译并安装到独立虚拟环境
"$GEMM_OPT_ROOT/build.sh"
# 无需运行设备的 gfx946 GEMM 编译检查
python "$GEMM_OPT_ROOT/smoke_gemm.py"
```

GitLab 连接通过旧 `kxu_pmd` 容器已有的 SSH 代理配置完成，拉取写入共享的独立目录。当前 `kxu_pmd_8569` 未配置该代理，直接使用 `gitlab` 别名仍可能失败。

## 支持需求与验证状态

下列入口是代码核查和扩展的起点，不代表已经覆盖需求的 dtype、scale、布局或全部形状。布局必须记录 stride，不能仅由 M/N/K 推断。

| 算子 | 输入 → 输出 | 形状数 / 实例数 | 候选入口 | 当前状态 |
| --- | --- | ---: | --- | --- |
| linear | FP8E4M3 × FP8E4M3 → BF16 | 207 / 1206 | `gemm_a8w8_blockscale`、`w8a8_triton_block_scaled_mm` | 待逐项验证 scale 语义与形状 |
| linear | BF16 × BF16 → BF16 | 133 / 326 | `gemm_a16w16`、`matmul_persistent`，内部 WASP/WDRA GEMM | 待验证 |
| linear | BF16 × BF16 → FP32 | 27 / 172 | 核查上述入口的输出 dtype 与 epilogue | 待核查支持与验证 |
| linear | FP32 × FP32 → FP32 | 35 / 218 | `matmul_persistent` | 待确认 IEEE/TF32 精度口径并验证 |
| bmm | BF16 × BF16 → BF16 | 29 / 194 | `batched_gemm_bf16` | 待验证 batch、布局与形状 |
| linear | BF16 × INT4 → BF16 | 4 / 4 | `awq_gemm_triton` | 待确认 packed 格式、group size、zero point |
| moe_grouped_gemm | FP8E4M3 × FP8E4M3 → BF16 | 36 / 70 | MoE grouped GEMM 路径 | 待核查分组调度、scale 和接口 |
| moe_grouped_gemm | FP8E4M3 × FP4E2M1 → BF16 | 34 / 304 | MoE 与 FP4 路径 | 待核查混合精度支持；不能用 A16WFP4 或 A4W4 直接等同 |
| moe_grouped_gemm | BF16 × INT4 → BF16 | 6 / 10 | MoE 与 W4A16 路径 | 待核查 |

补充需求：INT8 卷积 24 类需保持 stride、padding、分组、NCHWc32 和 bias/add/ReLU/requant 语义；末端 FC 为 `(batch, 1000, 2048)`，batch 覆盖 1～512。它们不能仅通过普通 dense GEMM 验证宣称支持。

## 优先级与验收

1. 在新 Triton 基线上重放既有 FP16 WASP/WDRA TT/TN 胜出配置，先验证正确性和是否出现性能回归；不得直接沿用历史 TOPS 为新基线结果。
2. 核查 BF16 → BF16/FP32 与 BF16 bmm，建立逐形状参考结果，再调优大 M prefill、小 M decode、长 K、窄 N、尾块和不同 B stride。
3. 优先覆盖 207 个 dense FP8 形状及 36 个 grouped FP8 形状；同时验证 FP32/E8M0 scale、K 方向 1×128 和 per-tensor 路径。
4. 扩展 grouped FP8×MXFP4、dense/grouped W4A16，先确认量化格式与参考反量化语义，再优化。
5. 单列 INT8 FC 与隐式 GEMM 卷积，建立融合算子的正确性与性能用例。

顺序是初始工程安排，具体 shape 优先级结合来源文件 P0、频次、实际耗时与复用范围调整，频次不能直接等同耗时占比。

关键语义：

- 输出 dtype 指写回显存的 dtype，与累加 dtype 分开记录。BF16 输出不能替代需求中的 FP32 输出。
- FP8 需要带 scale；主流为沿 K 每 128 个元素共享 scale，激活支持 FP32 与 E8M0，权重 scale 来自离线量化。HTML 中部分 scale 信息是推断值，接入模型时需确认实际布局。
- MXFP4 权重 scale 为 1×32，不能套用 FP8 的 1×128 索引。
- 关注清单中的 `K=192` FP8 尾块，以及 grouped 的不规则 M、空组和专家偏移。
- INT4 必须记录 group size、packed 布局、signed/unsigned 与 zero point；INT8 记录 INT32 累加、requant 舍入和饱和语义。
- 每个晋升配置需给出编译器版本、硬件/PMD 版本、launch、正确性误差及阈值、baseline、重复测量和原始产物路径。

## 实验记录

### 2026-10-14：工作目录和编译基线

- 已建立独立源码仓库，已有源码中的未提交修改未带入。boltOps 使用 shb_bu_wasp，不另设工作分支；新增文档与记录迁移通过该分支的 MR 提交到 main。
- Triton 3.6.0 编译并安装成功，HCU 后端可加载。
- 独立 `smoke_gemm.py` 将 FP16 `32×32×64` GEMM 编译为 gfx946 HSACO（5880 bytes）。这是编译验证，未运行内核，未验证数值或性能。
- 构建补齐了已有工作区中未跟踪的 NVIDIA 依赖头文件；运行环境补入 `/xukang/downloads/pmd.tar.gz` 中的 `libgem5_opt.so` 以满足 PyTorch 导入。该库不构成 PMD 8569 仿真验证，正式测量前须对齐 PMD 版本与启动配置。
- 原始日志：工作目录 `logs/build-triton-retry.log`、`logs/install-triton.log`、`logs/smoke-gemm.log`。
- 已将历史 RESULTS/VGPR 记录迁到本目录；没有新增性能结论或标记任何需求形状已通过。

### 2026-10-14：WDRA48 小用例性能复测

详见 [完整复测记录](PERF_20261014.md)：512³ 普通版两次正确性通过，平均 19.885 TOPS；256×256×4096 MLS 两次通过，平均 7.405 TOPS，比历史约低 3.5%；512³ MLS 历史 tile 在最新 Triton 的 MLS 编码选择阶段编译失败。记录包含旧 API 转换和当前 compiler 固定 ping-pong 的可比性限制。

### 2026-10-15：正式迁移 WASP/WDRA launch options

GEMM 普通调用、autotune 配置和调优工具现直接使用新版 Triton 参数，无需临时适配脚本：

| 项目 | 新参数 / 行为 |
| --- | --- |
| WASP 请求 | 内核循环 `warp_specialize=WARP_SPECIALIZE` |
| WDRA48 | `wasp_partition_warps=(4,4,4)`，`wasp_wdra=True` |
| WDRA88 | `wasp_partition_warps=(4,4,4,4)`，两个 Producer 的 quota 分别填写 |
| VGPR quota | `wasp_partition_regs` 按 Load-first 顺序填写；由 Triton 校验与归一化 |
| consumer ping-pong | 编译器控制，不再作为 launch option；CLI 仅兼容 `--consumer-pingpong true`，禁用或扫描会报错 |

23 份历史调优 JSON 的 launch 字段同步迁移，性能数值保留为历史测量；旧 ping-pong 取值保留在 `historical_enable_consumer_pingpong`，不表示新编译器能关闭该行为。Python 包装接口的 `wasp`/`wdra` 与 load/mma warp 数、CLI 的三个 VGPR 参数仍作为便捷输入，内部转换成新版 tuple。CSV 兼容旧统计列，并同时记录新版 tuple；晋升配置只保存新版 launch 字段。

直接执行 `python -m boltops.tools.tune_gemm_wasp` 的 PMD 验证（无 monkey-patch）：

| 用例 | cycles | TOPS | 正确性 |
| --- | ---: | ---: | --- |
| WDRA48 TT，256×256×4096，MLS/matrix-store，tile 256×256×64，quota 16/244/244 | 159300 | 7.407005 | PASS |
| nowasp TT，64³，tile 64³ | 16242 | 0.070945 | PASS |

Triton 为 `69b5d8075`，其余环境沿用 [复测记录](PERF_20261014.md)。原始产物在工作区 `runs/native-options-256-mls-verified/` 和 `runs/native-options-nowasp64/`，各含 run.log、result.csv 与 m5out。

同时修正调优工具跨工作区读取统计的问题：只扫描和清理当前目录的 m5out，累计周期只减去同一 XCD 输出目录的基准。初次原生调用曾因减去其他目录的旧周期而高估 TOPS，该次 `runs/native-options-256-mls/` 的 CSV 作废，原始 stats 为 159360 cycles；以上结果为修复后重跑。

验证：13 项 CPU 回归测试通过（收集阶段禁用设备探测），覆盖四种 mode 的当前后端参数校验、历史配置迁移、WDRA48/88 晋升配置与跨目录周期归属。无设备的 CPU 测试不替代上表的 PMD 正确性验证。

### 2026-10-15：修复 nowasp 的 K 尾块越界读取

`test_internal_gemm_fp16_nowasp` 默认 tile 为 128³，但用例含 K=64/96。普通加载内核原先对每一轮 K 都执行不带 mask 的 tl.load；即使输入有限，也可能读取相邻内存中的 NaN，最后在 assert_fp16_gemm_close 中失败。修复为 A/B 的 K 方向加载添加 remaining-K mask，越界元素填零；没有放宽正确性阈值或跳过 nowasp。

回归用例在输入后分配 NaN guard，覆盖 TT/TN 的 `(64,64,64)`、`(48,80,96)`、`(64,64,128)`、`(48,80,192)`。NaN guard 在 CPU 初始化后传入设备，结果有限性检查在 CPU 执行。普通加载的 K 尾块不再 xfail；MLS 尾块的已有 xfail 不受此次修改影响。

当前 PMD 上 8 个用例全部通过。对同一个 TT 64³ 用例替换回修复前的加载内核，会稳定触发 `GEMM returned NaN/Inf with finite inputs`，确认回归用例能检出原问题。产物位于工作区 `runs/nowasp-k-tail-regression-cpu-fill/`（8 passed）及 `runs/nowasp-k-tail-before-fix/`（预期失败）。首次使用 GPU 初始化 NaN guard 的尝试触发当前 PMD 内存读路径崩溃，改为 CPU 初始化后完成验证；未在 nmz 实机上复跑，CI 结果仍需确认。

CI 同时出现的 FlashMLA sparse `no matching finite config bucket; using built-in defaults` 是配置回退 RuntimeWarning；仓库 CI 未设置 warnings-as-errors，对应两个 warning 回归测试通过。该 warning 本身不证明 MLA 正确性测试失败，仍需实际失败 traceback 判断；本次没有改动 MLA 实现或测试。

### 2026-10-15：修复 wheel 缺少 GEMM 调优配置

CI 的 `test_saved_configs_use_current_hcu_options - assert []` 是打包遗漏：源码中存在的 23 份 gfx946 GEMM 调优 JSON 未列入 setuptools package-data，安装 wheel 后配置目录为空。已增加 `gemm/triton/configs/gemm_fp16/*/*.json` 打包规则，并让配置缺失时的测试断言显示实际查找目录；保留配置必须存在的检查。

使用独立干净构建环境构建 wheel，核对压缩包包含全部 23 份配置，再按 CI 流程安装到临时 target、复制测试与 pyproject.toml，并从该目录执行测试。确认实际导入来自 installed/boltops，原失败用例通过（1 passed，32 deselected）。验证时禁用 GPU 探测并将架构查询设为 gfx938；这是安装产物的 CPU 配置检查，不是 nmz 实机内核运行。产物与日志位于工作区 `runs/wheel-package-data/`。

### 后续记录模板

| 项目 | 内容 |
| --- | --- |
| 日期 / 实验目标 | |
| 需求行 / 模型 / 阶段 | 对应 REQUIREMENTS.md 的条目编号和完整语义 |
| 算子 / A、B、输出 / scale | |
| M/N/K / batch 或 experts / stride | |
| Triton / boltOps / LLVM / PMD 或设备版本 | |
| 基线 / 改动 / launch / 配置文件 | |
| 正确性 | 参考、误差阈值、实际误差、通过/失败 |
| 性能 | baseline、候选、重复次数、统计口径、提升 |
| 产物 | 日志、ISA、TTGIR、PMD stats 或设备测量路径 |
| 结论 / 下一步 | 是否晋升配置、遗留问题 |
