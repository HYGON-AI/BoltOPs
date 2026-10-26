# `hc_combine_norm` 使用说明

qwen4_exp（qwen3.8-flash）HyperConnection 边界算子：门控残差 combine 与分组
Gemma RMSNorm 的融合 Triton kernel。每个 decoder 层的 HC 边界
（attn→mlp、mlp→下一层）各调用一次，单次 launch 同时产出 combine 后的新残差流
`out` 与归一化后的 `y`（下一个 down/gate 投影的输入）。

## 1. 算子用途与数学语义

对每个 token row、每个 hc-stream `s`：

```text
gate_s = 2 * sigmoid(injection_logits[s] / HC)
out_s  = bf16(residual_s + block_output * gate_s)   # 先 round 再归一化
y_s    = rmsnorm(out_s) * (1 + w_s)                 # Gemma 式 (1+w) 仿射
```

- `out` 的 bf16 rounding 点对齐 unfused combine → RMSNorm 边界：先 round 到
  bf16 再参与归一化，归一化内累加为 FP32
- 权重 `w` 两种模式：shared（`numel == HCDIM`，跨 stream 广播）与
  per-branch（`numel == DIM`，与 residual 同样的 HC 展平布局）
- 输入不被修改；返回 `(out, y)`，shape/dtype 与 `residual` 一致
- 算子为 forward-only，不包含 backward，不含 `injection_logits=None` 的
  final mixer 路径（与 AMD 生产版一致；NV 版有该路径）

上游调用链（qwen4_exp vLLM）：

```text
GatedResidual.combine_and_mix()               vllm/models/qwen4_exp/amd/hyperconnection.py:164
  └─ torch.ops.vllm.qwen4_exp_hc_combine_norm  amd/ops/hc.py:463
      └─ _hc_combine_norm_kernel[(N, HC)]      amd/ops/hc.py:267
```

## 2. 接口说明

```python
from boltops.generic import hc_combine_norm

out, y = hc_combine_norm(
    residual, block_output, injection_logits, norm_weight, eps, hc_count
)
```

函数签名：

```python
def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

### 2.1 参数含义

| 参数 | Shape | 常见 dtype | 含义 |
|---|---|---|---|
| `residual` | `[N, DIM]`，`DIM = hc_count * HCDIM` | BF16 | 当前全部 hc-stream 的残差 |
| `block_output` | `[N, HCDIM]` | BF16 | 子层输出（attn 或 mlp） |
| `injection_logits` | `[N, HC]` | BF16 | 门控 logits，逐 stream |
| `norm_weight` | `[HCDIM]`（shared）或 `[DIM]`（per-branch） | BF16 | Gemma RMSNorm 权重 |
| `eps` | scalar | float | RMSNorm epsilon |
| `hc_count` | scalar | int | hc-stream 数（qwen3.8-flash 为 4） |
| 返回 `out` | `[N, DIM]` | 同 `residual` | combine 后的新残差流 |
| 返回 `y` | `[N, DIM]` | 同 `residual` | 归一化输出 |

### 2.2 前置条件

- 所有 tensor 位于同一 CUDA/HIP device
- `DIM % hc_count == 0`；`block_output` / `injection_logits` 最后一维连续
- `norm_weight` contiguous 且 `numel ∈ {HCDIM, DIM}`
- 行 kernel（row_split）额外要求 `residual` 行主连续（`stride(0) == DIM`）；不满足
  时入口自动回退 baseline kernel，无需调用方处理

## 3. 典型 Shape

qwen3.8-flash：`hidden=2560`、`hc_count=4` → `HCDIM=2560`、`DIM=10240`，bf16，
eps 1e-6，per-branch 权重：

| 场景 | N（token/rank） | 单次流量 |
|---|---|---|
| decode draft / ngram-MTP verify / 小 decode | 1–512 | 66 KB–33 MB |
| chunked prefill（16-4096 生产 chunk） | 8192 | 0.545 GB |
| full prefill | 16384 | 1.09 GB |

bytes/token = 读 `residual`(20480B) + 读 `block`(5120B) + 写 `out`/`y`
(各 20480B) = 66560 B。prof 占比（bw151 w4a8-dp4ep4 16-4096 dense trace，
4 卡合计）：2.99s / 99.5s ≈ 3%，全 trace 第 9 名、glue 类 kernel 第 1 名，
20235 次调用。

## 4. 正确性测试与性能基准

```bash
# 正确性（对 eager fp32 reference，含 dispatch 两侧、shared/per-branch 权重、
# 可拆与不可拆的 HCDIM、非连续回退）
pytest -q test/test_generic_hc_combine_norm.py

# 性能（baseline vs ship 入口，qwen3.8-flash 12 档 token 数）
pytest -q test/test_generic_hc_combine_norm.py --bench -s
```

PyTorch reference 等价于：

```python
gate = 2.0 * torch.sigmoid(inj.float() / HC)
of0 = res.float().view(-1, HC, HCDIM) + block.float().view(-1, 1, HCDIM) * gate.unsqueeze(-1)
out = of0.to(torch.bfloat16)                       # round 点对齐 kernel
of = out.float()
y = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
y = (y + y * w.float().view(1, HC, HCDIM)).to(torch.bfloat16)
```

正确性判据：max-abs-diff ≤ `0.05 + 2% × |输出最大幅度|`（覆盖几个 bf16 ulp
的归约顺序噪声；真正的语义错误是 O(1) 级 diff）。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/generic/__init__.py` |
| Triton 实现 | `boltops/generic/triton/hc_combine_norm.py` |
| 测试 / 性能 | `test/test_generic_hc_combine_norm.py`（`--bench`） |

范围：`hc_combine_norm`（ship 入口）与同文件内上游 kernel
`_hc_combine_norm_kernel` + 本地 sibling `_hc_combine_norm_row_split_kernel`；
不含 NV CUDA 实现、不含 `injection_logits=None` final mixer 路径、不含 backward。

## 附录 B. 溯源

- 上游：vLLM 内部树 `vllm/vllm/models/qwen4_exp/amd/ops/hc.py`
  （Triton kernel L267、wrapper L334、custom-op 注册 L432/L463），
  wrapper `_hc_combine_norm` 为其原样拷贝（仅去掉 custom-op 注册层，
  `launch_pdl` 恒 False——gfx936 无 PDL，生产值亦为 False）
- 本地结构优化：`_hc_combine_norm_row_split`（一行一 program、去 padding lane
  拆段）为 BoltOPs sibling kernel，与上游 kernel 同文件并存、上游逐字不改（同 `attn_res` 的
  lazy-prefix 模式），便于 diff 跟进上游；调优中间版 v3（单段 padded tile）
  未合入仓库，历史数据见附录 E；E.3 的 v4 合入名为 `_hc_combine_norm_row_split`
- 交付包（正确性/消融原始脚本与完整调优记录）：
  `qwen3_8-next-flash/hc_combine_norm_kernel/`

## 附录 C. 实现备注（分派策略）

入口按 shape 二级分派，回退输出与 baseline **逐位一致**：

| 条件（全部满足才走行 kernel） | 实现 |
|---|---|
| `N ≥ 256` 且 `HCDIM ≤ 4096` 且 `residual` 连续 且 `HCDIM = 2^k + 2^m` | row_split（`_hc_combine_norm_row_split_kernel`，无 mask lane） |
| 其余（小 batch、非连续、不可拆 HCDIM、超大 HCDIM） | baseline（`_hc_combine_norm_kernel`，上游逐字） |

- `N ≥ 256`：一行一 CTA 的网格对 80 CU 量化敏感，96–192 区间 row_split 反而输
  baseline 4–10%（只有 1.2–2.4 个 wave，尾巴占比最差），故小 batch 留在
  4N 个 CTA 的 baseline 上
- `HCDIM ≤ 4096`：一个 program 持有整行 FP32 tile，再大寄存器溢出
- row_split 的两段 tile 精确覆盖整行、无 mask load：2560 = 2048+512、4096 = 2048+2048、
  1280 = 1024+256 ✓；3000 ✗ → 回退 baseline
- 两者数值语义一致（`out` 逐位相同；`y` 仅归约顺序不同，bf16 噪声级差异）；
  wrapper 均为私有（`_hc_combine_norm` / `_hc_combine_norm_row_split`），公开入口只有
  `hc_combine_norm`
- 行 kernel 的 `w` 偏移必须保持 stream 相对（不可加行号，误加会越界 →
  GPU VM fault）；`num_warps=4` 即 gfx936 最优（launch 参数扫描已证伪，瓶颈
  在结构）
- VGPR（HCDIM=2560、per-branch、nw=4）：单段 padded tile 101 → row_split 两段
  tile 65（-36%），spill 均 0；row_split 的 LDS 归约 scratch 16 KB/CTA，不影响占用

## 附录 D. Benchmark Shape

`test_bench_hc_combine_norm`（`pytest --bench -s`）：qwen3.8-flash 生产
dims（DIM=10240、HC=4、bf16、eps 1e-6、per-branch 权重），N ∈
{1, 4, 16, 64, 148, 256, 512, 1024, 2048, 4096, 8192, 16384}，覆盖 decode
draft、ngram/MTP verify、小 decode、chunked prefill（8192/rank）与 full
prefill。每档同时测 baseline 与 ship 入口并输出加速比与 GB/s。

## 附录 E. BW151 调优记录

E.2 的 v3 是调优中间步骤（用于定位结构瓶颈），仓库只合入 E.3 的 v4。

### E.1 优化动机与瓶颈定位

同数据量（N=8192，0.545 GB/call，已按 dp4/tp4 归一）对比：

| 设备 | 单次耗时 | 实测带宽 | 峰值带宽 | 利用率 |
|---|---|---|---|---|
| NVIDIA H20-3e（HBM3） | 152.3 µs | 3.58 TB/s | 4.0 TB/s | ~90% |
| BW151（gfx936, HBM2e） | 544.8 µs | 1.00 TB/s | ~1.6 TB/s | ~63% |

3.6× 总差距 = 硬件带宽 2.5× × 软件利用率 1.4×，本算子可追的是后者。定位过程：

1. 每元素仅 2–3 次乘加，纯 memory-bound，唯一指标 TB/s
2. 极简 copy kernel 实测设备天花板 **1.35 TB/s**（torch d2d copy 同数），
   原版仅 71%
3. `num_warps ∈ {1,2,4,8} × BLOCK ∈ {256,512,1024,2048}` 全扫描 12 种组合
   钉在 0.92–0.96 TB/s → 瓶颈在 kernel 结构而非 launch 参数：
   每个 program 只负责一个 (row, stream) 段（~5 KB 读），在飞字节数太少；
   RMSNorm 归约形成依赖链憋住访存（旁证：无归约的 r2+w1 copy 变体能跑
   1.32 TB/s，排除读写比例因素）

### E.2 v3：一行一 program

- grid `(N, HC)` → `(N,)`：一次发出 4 个 stream 共 80 KB 访存请求，
  在飞字节数 ×4，填满访存延迟
- 4 个 stream 的 sum_sq 合并为一次 `[HC]` 向量归约（原 4 个 program 各自
  做跨 warp 归约，warpSize=64 下代价更高）
- gate 选取从「加载全部 4 个 logit 再 `tl.where` 挑 1」简化为按下标直取

N=8192：574 → 441 µs（1.27×），0.95 → 1.24 TB/s = 天花板的 92%。

### E.3 v4：拆段去 padding lane

v3 的行 tile 是 `[HC, next_pow2(HCDIM)]` = `[4, 4096]`，其中 37.5% lane 被
mask（不产生 HBM 流量，但占寄存器与发射槽）。v4 把行拆成 `2048+512` 两段
2 的幂 tile，全 lane 有效：VGPR 101 → 65，N=8192 再快 4–6%。

### E.4 实测（BW151 GPU0，bf16，DIM=10240，`time_ms` 中位数）

| N | baseline | ship（分派后） | 加速 | ship 带宽 |
|---:|---:|---:|---:|---:|
| 1 | 3.3 µs | 3.3 µs | 1.00× | 20 GB/s |
| 16 | 3.3 µs | 3.3 µs | 1.00× | 318 GB/s |
| 148 | 11.9 µs | 11.8 µs | 1.00× | 832 GB/s |
| 256 | 19.9 µs | 19.1 µs | 1.04× | 891 GB/s |
| 512 | 40.1 µs | 32.7 µs | 1.23× | 1044 GB/s |
| 1024 | 75.7 µs | 60.3 µs | 1.25× | 1130 GB/s |
| 2048 | 147.0 µs | 114.5 µs | 1.28× | 1191 GB/s |
| 4096 | 290.6 µs | 224.9 µs | 1.29× | 1212 GB/s |
| 8192 | 570.1 µs | 440.4 µs | **1.29×** | **1238 GB/s** |
| 16384 | 1136.3 µs | 877.1 µs | **1.30×** | 1243 GB/s |

- N=8192 的 1.24 TB/s = 实测 copy 天花板 1.35 TB/s 的 **92%**
- N < 256 自动回退 baseline，decode 段零回退风险；N=256 处 1.04–1.12×
  （机器状态波动 ±5%，交付包加密重测口径为 1.12×）
- 96–192 区间的网格量化损失实测（v4/base）：N=96 0.92×、N=112 0.96×、
  N=128/160/192 ~0.99× —— 这就是 `_MIN_ROW_TOKENS = 256` 的依据

### E.5 上限结论

生产 chunk（N=8192）已达实测设备带宽天花板的 92%，剩余差距主要是
HBM2e 带宽本身（对 H20 的 2.5× 硬件差）；launch 参数、`num_warps`、
L2 policy 类旋钮均已扫描无收益，增量收敛。
