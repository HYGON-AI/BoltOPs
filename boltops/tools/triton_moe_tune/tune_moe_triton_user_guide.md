# Triton Fused MoE Autotune 一线使用手册

**文档版本**

- `v1.0`
- 发布日期：`2026-04-20`

**Release 说明**

- 统一 `fused_moe` 训练入口为 `boltops-fused-moe-tune-cli`
- 支持 `TP=1/2/4/8` 循环训练、按 TP 分目录产出日志与配置
- 支持 `hy-smi` 自动选连续空闲卡
- 支持训练模式与 benchmark-only 模式分离
- 支持通过 `--activation` 覆盖 MoE activation，包括 `gelu_tanh`
- 补充了各 `datatype` 的支持情况与对应脚本映射

**适用对象**

- 需要在 `nmz/bmz/zd/yy` 机器上执行 Triton fused_moe autotune 的项目人员
- 需要生成 MoE 配置 JSON 并回归性能/精度的项目与验证人员
- 一线支持与问题定位人员

**非适用对象**

- 不适用于 Triton 编译器底层实现分析
- 不替代模型算法设计文档

---

## 1. 使用范围

适用于以下任务：

1. 执行 fused_moe autotune（compile-only + run + parse）
2. 生成并拷贝 MoE 配置 JSON 到 `boltops/fused_moe/triton/configs/moe`
3. 只跑 benchmark（复用现有配置，不再训练）
4. 快速排障（选卡、日志、参数一致性）

---

## 2. 目录与入口

运行位置：

- `pip install "boltops[tune]"` / wheel 后，可在任意目录执行 `boltops-fused-moe-tune-cli ...`
- editable 开发：在仓库根执行 `pip install -e ".[tune]"`，同样可在任意目录运行

核心文件：

- 统一入口：`boltops-fused-moe-tune-cli`（`boltops.tools.triton_moe_tune.tune_moe_cli:main`）
- 调参代码：`boltops/tools/triton_moe_tune/`
- 公共逻辑：`boltops/tools/triton_moe_tune/common.py`
- patch 逻辑：`boltops/tools/triton_moe_tune/autotune_patches.py`
- 日志解析：`boltops/tools/triton_moe_tune/moe_log_parser.py`

安装形态统一约定（推荐）：

1. 仓库根：`pip install -e ".[tune]"`（或安装带 `tune` extra 的 wheel）
2. 直接执行：`boltops-fused-moe-tune-cli <device_name> <test_type> ...`

> **CLI 站点码 vs 配置 arch**：命令行第一个位置参数仍是机房/站点码
> （`nmz`/`bmz`/`zd`/`yy`），用于日志目录命名；写入/查找的 MoE JSON
> 文件名则使用 GPU ISA `arch=`（不再使用产品名 `device_name=BW*`）。
>
> | `device_name`（CLI） | JSON 中的 `arch=` |
> |---|---|
> | `zd` | `gfx928` |
> | `bmz` | `gfx936` |
> | `nmz` | `gfx938` |
> | `yy` | `gfx92a` |
>
> 例：`boltops-fused-moe-tune-cli nmz bf16` → 生成/期望
> `E=...,N=...,arch=gfx938.json`；日志目录仍为 `logs_nmz_bf16_tp*`。

---

## 3. Datatype 支持矩阵（含脚本映射）

> **重要提醒**：Triton 所支持的所有数据类型的 weight 格式，与 sglang/vllm 原生的 weight 格式完全一致，**不需要** lightop 等模型的 shuffle weight！
>
> **重要提醒**：Triton 所支持的所有数据类型的 weight 格式，与 sglang/vllm 原生的 weight 格式完全一致，**不需要** lightop 等模型的 shuffle weight！
>
> **重要提醒**：Triton 所支持的所有数据类型的 weight 格式，与 sglang/vllm 原生的 weight 格式完全一致，**不需要** lightop 等模型的 shuffle weight！

### 3.1 总览矩阵（脚本映射）

| test_type | 对应脚本 | shared_expert | EP(`--ep-size`) | 关键约束/说明 |
|---|---|---|---|---|
| `bf16` | `tune_moe_bf16.py` | 支持（`MOE_NUM_SHARED_EXPERTS`） | 支持 | 建议 `num_experts % ep_size == 0`；训练/benchmark 均走 `fused_experts_impl_benchmark` 计时路径 |
| `fp8` | `tune_moe_fp8.py` | 支持 | 支持 | block 路径默认 `[128,128]`；推荐 `MOE_HIDDEN_SIZE`、`MOE_FFN_HIDDEN_SIZE` 对 `K` 对齐 |
| `fp8_channel` | `tune_moe_fp8_channel.py` | 支持 | 支持 | channel 路径，不依赖 block 对齐，适合非 128 对齐 hidden |
| `int8` | `tune_moe_int8.py` | 支持 | 支持 | block 路径默认 `[128,128]`；推荐 `MOE_HIDDEN_SIZE`、`MOE_FFN_HIDDEN_SIZE` 对 `K` 对齐 |
| `int8_channel` | `tune_moe_int8_channel.py` | 支持 | 支持 | channel 路径，不依赖 block 对齐，适合非 128 对齐 hidden |
| `int4` | `tune_moe_int4.py` | 不支持 | 支持 | `group_size` 固定 `64`（脚本常量），`has_zp` 固定 `True`（脚本常量）；`num_experts % ep_size == 0` |
| `int4int8` | `tune_moe_int4int8.py` | 不支持 | 支持 | `group_size` 固定 `64`；默认无完整对结果参考框架，需先确认 weight 格式/scale 布局 |
| `int4int8_channel` | `tune_moe_int4int8_channel.py` | 不支持 | 支持 | channel-wise W4A8；默认无完整对结果参考框架，需先确认 weight 格式/scale 布局 |

### 3.2 CLI 参数与 datatype 对齐关系

| 参数 | 支持的 test_type |
|---|---|
| `--num-shared-experts` | `bf16`, `fp8`, `fp8_channel`, `int8`, `int8_channel` |
| `--block-shape` | `fp8`, `int8` |
| 固定 `group_size=64`（CLI 不导出） | `int4`, `int4int8` |
| 固定 `has_zp=True`（CLI 不导出） | `int4` |
| `--ep-size` | 所有 `tune_moe_*` 主脚本（`bf16/fp8/fp8_channel/int8/int8_channel/int4/int4int8/int4int8_channel`） |

### 3.3 关于 EP 支持的统一说明

- `--ep-size` 在上述 8 个 datatype 脚本里都生效。
- 实践上应保证 `num_experts % ep_size == 0`（多数脚本有显式 assert）。
- 若使用 shared experts（`--num-shared-experts > 0`），仅限 `bf16/fp8/fp8_channel/int8/int8_channel`。

补充：

- `tune_moe_sum.py` 是辅助测试脚本，不是 `boltops-fused-moe-tune-cli` 的 `test_type` 目标。
- `boltops/tools/triton_moe_tune/autotune_patches.py` 不是独立算子实现，而是对 `fused_moe.py` 的外挂 patch 层，主要用于 autotune 辅助与测试路径接管。
- 对 `int4int8/int4int8_channel`，建议先用小 batch 做 weight 格式与量化参数一致性校验，再进入大规模 autotune。
- `int4/int4int8` 的 `group_size` 当前固定为 `64`，不通过 CLI 导出。
- `int4` 的 `has_zp` 当前固定为 `True`，不通过 CLI 导出。

---

## 4. 快速开始

### 4.1 最简训练命令

```bash
boltops-fused-moe-tune-cli nmz int8_channel
```

### 4.2 先看日志判断是否需要训练（推荐）

在框架运行日志中，先检查是否命中已存在的 MoE config：

```text
[boltops] Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.10/dist-packages/boltops/fused_moe/triton/configs/moe/E=512,N=168,arch=gfx936.json
[2026-04-20 17:44:46 TP7] Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.10/dist-packages/boltops/fused_moe/triton/configs/moe/E=512,N=168,arch=gfx936.json
[boltops] Using configuration from /usr/local/lib/python3.10/dist-packages/boltops/fused_moe/triton/configs/moe/E=512,N=336,arch=gfx936,is_bottom=True.json for MoE layer.
```

判断规则：

- 出现 `Using default MoE config ... Config file not found ...`：说明该 shape/config 缺失，建议执行 `boltops-fused-moe-tune-cli` 训练并生成对应 json。
- 出现 `Using configuration from ... for MoE layer.`：说明该配置已命中，可继续使用，通常不需要重复训练同一组参数。

### 4.3 默认执行步骤

该命令默认执行：

1. Step1 compile-only
2. Step2 autotune run
3. Step3 parse 生成 JSON 并拷贝到配置目录
4. Step4 benchmark/correctness

### 4.4 benchmark-only 命令

```bash
boltops-fused-moe-tune-cli nmz int8_channel --benchmark
```

该模式只跑 benchmark，不执行训练步骤。

---

## 5. 标准流程（推荐）

### 阶段A：训练并生成配置（默认包含 benchmark）

```bash
boltops-fused-moe-tune-cli nmz int8_channel \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8
```

说明：默认训练模式会执行 Step1/2/3/4，其中 Step4 即 benchmark/correctness。

### 阶段B（可选）：benchmark-only 复核

```bash
boltops-fused-moe-tune-cli nmz int8_channel \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8 \
  --benchmark
```

说明：

- 阶段A 已执行过 benchmark/correctness；阶段B主要用于框架人员二次确认结果。
- 阶段B请保持与阶段A相同的核心模型参数。
- benchmark-only 会在终端实时打印 pytest 输出。

### 阶段C（建议）：确认是否提交配置到 boltops 仓库

训练后生成的配置会拷贝到本机安装目录（如 `/usr/local/lib/python3.10/dist-packages/boltops/fused_moe/triton/configs/moe`）。
建议再做一步判断：这些配置是否需要回传到 boltops 仓库长期保存。

目标仓库路径：`boltops/fused_moe/triton/configs/moe`

建议提交的场景：

- 日志出现 `Config file not found`，且本次 autotune 生成了对应新 shape 的 json。
- 同一 shape 下，新配置 benchmark 明显优于仓库已有配置。
- 该配置会被后续版本/项目人员复用（例如主干模型常用 TP/EP 组合）。

可暂不提交的场景：

- 仅临时调试参数，复用价值低。
- benchmark 波动较大，尚未完成复核。

建议在 MR 描述中附上：

- 对应训练参数（device/test_type/TP/EP/num_experts/hidden_size/top_k/ffn_hidden_size）。
- 关键 benchmark 对比（新旧配置或默认配置对比）。
- 命中的配置文件名（top/bottom）。

---

## 6. 参数说明（按重要性分组）

### 6.1 核心 MoE 参数（常用）

- `--tp-list`：TP 列表，建议 `1,2,4,8`
- `--num-experts`：总 experts 数
- `--hidden-size-default`：基准 hidden size，实际每个 TP 为 `hidden_size_default / TP`
- `--ep-size`：EP 大小（默认 `1`）。支持 `bf16/fp8/fp8_channel/int8/int8_channel/int4/int4int8/int4int8_channel`；建议满足 `num_experts % ep_size == 0`
- `--top-k`：覆盖脚本默认 top-k
- `--ffn-hidden-size`：覆盖脚本默认 FFN hidden
- `--batch-sizes`：覆盖批次集合
- `--activation`：覆盖脚本默认 activation。当前支持 `silu`、`gelu`、`gelu_tanh`、`swigluoai`、`swiglustep`、`silu_no_mul`、`gelu_no_mul`、`gelu_tanh_no_mul`、`relu2`、`relu2_no_mul`
- `--is-gated`：覆盖 activation 的 gated/no_mul 推断。通常无需手工设置；默认由 `--activation` 自动推断

若不指定 `--batch-sizes`，默认使用：

`1,2,4,8,16,24,32,64,128,256,512,1024,2048,4096,8192,16384,32768`

建议：非必要不要修改默认 `--batch-sizes`，以保证 autotune 覆盖完整性与结果可比性。

补充：

- `gelu_tanh` 表示 `GELU(..., approximate="tanh")`，用于对齐 vLLM 的 `GELU_TANH`。
- `gelu_pytorch_tanh` 和 `gelu_pytorch_tanh_no_mul` 作为兼容别名也可传入；脚本内部会归一化到 `gelu_tanh` / `gelu_tanh_no_mul`。
- gated activation（如 `silu`、`gelu`、`gelu_tanh`）要求 `w1` 输出维度为 `2 * inter_dim`；`*_no_mul` 与 `relu2` 走 non-gated 路径，`w1` 输出维度保持 `inter_dim`。

### 6.2 运行与调度参数（高级）

- `--skip-compile-only`：跳过 Step1
- `--num-groups-compile`：compile-only 分组数
- `--device-start-id`：手工指定起始设备
- `--device-count`：手工指定连续设备数量
- `--disable-auto-select-devices-by-hy-smi`：禁用自动选卡
- `--hy-smi-max-vram-pct`：自动选卡阈值（VRAM/HCU 共用）
- `--break-on-error [true|false]`：遇错是否立即退出
- `--dry-run`：只打印计划，不执行
- `--benchmark`：仅 benchmark

### 6.3 运维参数

- `--kill`：杀掉 MoE 相关训练/pytest 进程后退出

示例：

```bash
boltops-fused-moe-tune-cli nmz int8 --kill
```

补充：

- 正常训练模式（不带 `--benchmark`）会在启动时自动检测并清理已在运行的 MoE 进程，避免并发冲突。

---

## 7. 自动选卡规则（hy-smi）

默认开启自动选卡，策略如下：

1. 优先选“连续设备段”
2. 长度优先（先满足期望卡数）
3. 在满足阈值的候选中，优先平均占用更低的段

阈值规则：

- 单卡 `VRAM` 或 `HCU` 任一指标 `>= 阈值` 即视为不可用
- 只有 `VRAM` 与 `HCU` 都 `< 阈值` 才参与候选

---

## 8. 输出与日志

每个 TP 独立日志目录：

- `logs_<device_name>_<test_type>_tp<tp>`（此处 `device_name` 仍是站点码，如 `nmz`）

配置 JSON 文件名使用 `arch=`（由站点码映射，见 §2），例如：

- `E=256,N=384,arch=gfx938.json`
- `E=256,N=384,arch=gfx938,is_bottom=True.json`

启动时会打印 `Effective Config`，包括：

- 生效后的设备与 test_type
- TP 列表、batch_sizes、device range
- 自动选卡结果
- 预期 JSON 文件名（top/bottom）

---

## 9. 常见问题

### Q1：`int8/fp8` 报 hidden size 相关错误

`int8/fp8` 的 block 路径默认 `block_shape=[128,128]`，因此要求 `MOE_HIDDEN_SIZE % 128 == 0`。
若不满足（如 `320`），请使用 `int8_channel/fp8_channel`。

### Q2：benchmark-only 为什么不训练

这是设计行为。`--benchmark` 模式只执行 benchmark，用于复用已有配置快速对比性能。

### Q3：日志目录清理报错或被占用

通常是已有同名训练仍在运行。
先执行 `--kill` 清理进程，再重启训练。

### Q4：如何判断训练参数与运行时配置是否一致

> **重点检查**
> `log.training` / `log.training_tp*` 中打印的 `expected_config_jsons`（例如 `E=...,N=...,arch=...json` 和 `...,is_bottom=True.json`）就是本次将生成并供框架运行时使用的配置文件名。
>
> **如果这组名字与运行时实际加载的配置名不一致，通常说明训练参数（如 `test_type/TP/N/block_shape`）设置不匹配，应先对齐参数再训练。**

---

## 10. 常用命令模板

最简训练：

```bash
boltops-fused-moe-tune-cli <device_name> <test_type>
```

多 TP 训练：

```bash
boltops-fused-moe-tune-cli <device_name> <test_type> --tp-list 1,2,4,8
```

手工指定设备段：

```bash
boltops-fused-moe-tune-cli <device_name> <test_type> \
  --device-start-id 0 \
  --device-count 4
```

benchmark-only：

```bash
boltops-fused-moe-tune-cli <device_name> <test_type> --benchmark
```

标准训练命令示例：

`int8`（block，默认 `block_shape=128,128`）

```bash
boltops-fused-moe-tune-cli <device_name> int8 \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8
```

`int8` + `gelu_tanh`

```bash
boltops-fused-moe-tune-cli <device_name> int8 \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8 \
  --activation gelu_tanh
```

`fp8_channel`（channel）

```bash
boltops-fused-moe-tune-cli <device_name> fp8_channel \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8
```

`bf16`

```bash
boltops-fused-moe-tune-cli <device_name> bf16 \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8
```

`w4a16`（对应 `test_type=int4`，`group_size` 固定为 `64`，`has_zp` 固定为 `True`）

```bash
boltops-fused-moe-tune-cli <device_name> int4 \
  --tp-list 1,2,4,8 \
  --num-experts 256 \
  --hidden-size-default 2048 \
  --ffn-hidden-size 7168 \
  --top-k 8
```
