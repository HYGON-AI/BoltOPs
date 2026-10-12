# Triangle Attention 使用说明

Triangle Attention 对三角形表示执行前向 attention。当前提供 BF16 Triton 实现，
输入采用 `[B, H, N, D]` 布局，输出由算子分配并返回；不提供 backward。

## 1. 算子用途与数学语义

对每个 batch 和 head，计算：

```text
S = Q @ K^T + mask_bias + triangle_bias
O = softmax(S, dim=-1) @ V
```

`Q` 须由调用方预先乘以 `1/sqrt(D)`，算子内不再缩放。`mask_bias` 和
`triangle_bias` 都是加性 bias，顺序与 PyTorch 参考表达式一致。

Kernel 使用在线 softmax、FP32 归一化和点积累加，并将概率转为 BF16 后执行
`P @ V`。概率转换采用 round-half-up 整数运算，不使用 RTZ 截断。

## 2. 接口说明

```python
import torch
from boltops.fa import triangle_attention_fwd

q, k, v = [
    torch.randn(2, 2, 65, 32, device="cuda", dtype=torch.bfloat16)
    for _ in range(3)
]
q /= 32 ** 0.5
mask_bias = torch.zeros(2, 1, 1, 65, device="cuda", dtype=torch.float32)
triangle_bias = torch.zeros(1, 2, 65, 65, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    out = triangle_attention_fwd(q, k, v, [mask_bias, triangle_bias])
```

公开接口签名为 `triangle_attention_fwd(q, k, v, biases)`，其中 `biases` 按顺序
包含两个张量。返回新分配的输出张量，不接受外部 `out` 参数。

### 2.1 主要参数

| 参数 | Shape / dtype | 含义 |
|---|---|---|
| `q`, `k`, `v` | `[B,H,N,D]`，BF16 | Q/K/V；`q` 已乘 `1/sqrt(D)` |
| `mask_bias` | `[B,1,1,N]` 或 `[B,N]`，BF16 / FP32 / FP64 | 未屏蔽 key 为 `0`，屏蔽 key 为有限值 `-1e9` |
| `triangle_bias` | `[1,H,N,N]` 或 `[B,H,N,N]`，BF16 / FP16 / FP32 / FP64 | 共享或逐 batch 的三角形 bias |
| 返回值 | `[B,H,N,D]`，BF16 | 新分配的 attention 输出 |

Q/K/V 和 bias 按各自 stride 寻址；测试覆盖非连续 Q/K/V、非连续 mask 和
共享/逐 batch bias。
两种 bias 可以使用不同的浮点 dtype；预处理保留其 dtype，kernel 加载后转换为
FP32 累加。接收 FP64 bias 不代表使用 FP64 累加。FP16 无法表示有限的 `-1e9`，
采用该屏蔽约定时应将 mask 保存为 BF16、FP32 或 FP64。

### 2.2 前置条件与限制

- 调用方须保证形状、dtype、设备符合上表。公开入口不会完整校验这些元信息。
- Q/K/V 和 `triangle_bias` 须为有限值，点积不能溢出。
- `mask_bias` 统一使用 `0` 表示未屏蔽 key，使用其 dtype 中的有限值 `-1e9`
  表示屏蔽 key；不支持使用负无穷作为输入 mask。每个 query 至少须有一个
  未屏蔽 key；整行使用 `-1e9` 屏蔽可能得到有限但无意义的输出。
- 本包仅提供 forward 算子，不提供模型级精度保证。
- 使用 `torch.no_grad()` 或 `torch.inference_mode()` 调用。FP32 融合累加与
  BF16 PyTorch 中间舍入存在差异，不保证任意有限输入都满足下述测试容差。

## 3. 正确性测试、性能基准与调优

在 BoltOPs 仓库根目录、可用的 Triton/HIP 环境中运行。工具需要 PyYAML，
已包含在 `tune` 和 `dev` extras 中。

```bash
# 算子正确性、配置和调优工具测试
PYTHONPATH=. python3 -m pytest -q test/fa/test_triangle_attention.py

# 打包导入冒烟测试
PYTHONPATH=. python3 -m pytest -q test/test_package_smoke.py

# 查看固定性能 case
PYTHONPATH=. python3 -m \
  boltops.tools.triangle_attention_triton_tune.tune_triangle_attention \
  --list-cases

# 只测量已保存配置，不搜索候选
PYTHONPATH=. python3 -m \
  boltops.tools.triangle_attention_triton_tune.tune_triangle_attention \
  --benchmark-only --report /tmp/triangle_attention_benchmark.json

# 只调优一个 case；完整搜索可能耗时较长
PYTHONPATH=. python3 -m \
  boltops.tools.triangle_attention_triton_tune.tune_triangle_attention \
  --case production-h2 --report /tmp/triangle_attention_tune.json
```

`benchmark_cases.yaml` 是正确性测试和调优 shape 的来源。性能测量只选择
`performance: true` 的三个 case；`non-power-two-head-dim` 只参与正确性测试。

| Case | Q/K/V shape | mask 布局 | 用途 |
|---|---|---|---|
| `production-h2` | `[693,2,693,32]` | 连续 | 正确性、性能与调优 |
| `production-h2-mask-strided` | `[693,2,693,32]` | 非连续 | 正确性、性能与调优 |
| `production-h8` | `[693,8,693,32]` | 连续 | 正确性、性能与调优 |
| `non-power-two-head-dim` | `[2,2,65,48]` | 连续 | 正确性 |

测试覆盖三个随机种子、非 2 的幂序列长度和 head dimension、非连续张量、
共享/逐 batch bias、输出分配及 BF16 bias 预处理。与 BF16 PyTorch 表达式比较时
使用 `atol=0.03`、`rtol=0.02`，并检查非有限输出。

### 3.1 设备配置

运行时从 `triton/config.py` 按设备和配置 key 查找 JSON。key 包含
`(batch, heads, seq_len, head_dim, mask_layout)`；mask 的 key stride 为 1 时
记为 `contiguous`，否则记为 `strided`。因此两个 H2 mask 布局分别保存和调优。
未命中时使用通用默认配置，其他设备或 shape 的性能未在此验证。

gfx936 的三个生产 case 均保存 `BLOCK_M=64`、`BLOCK_N=32`、
`num_warps=1`、`num_stages=1`；运行时额外设置 `kpack=2`。后端函数的
`config=` 参数供调优使用，公开入口使用保存的配置。

完整搜索每个生产 case 的 320 个候选。此前 gfx936 搜索中，每个 case 有
280 个候选完成计时，40 个编译失败或超出设备资源。该次搜索使用
`--skip-correctness`，所以未逐一验证候选输出；选中的配置另行通过正确性验证。
使用该选项重新调优后，应运行上面的正确性测试。

### 3.2 已有验证与性能结果

Triangle Attention 测试覆盖算子输出、混合 bias dtype、配置回退和调优工具。
此前独立复测使用最终源码
和配置，对三个生产 case、种子 0/17/42 得到 **9/9 通过**；共
553,246,848 个输出元素，按上述容差统计为零失败。结果来自合成输入，
不等同于完整模型精度。

在空闲 BW200 gfx936 上，已有以下 GPU event 中位数（毫秒）：

| Case | Triangle Attention | 配对 PyTorch 表达式 |
|---|---:|---:|
| `production-h2` | 1.680 | 20.222 |
| `production-h2-mask-strided` | 1.708 | 27.084 |
| `production-h8` | 6.539 | 89.836 |

基准调用完整 forward 接口，包含输出分配和 bias 预处理；GPU event 数字主要反映
设备耗时，不代表完整 Python 墙钟时间。输入生成、Q 缩放和编译不计入测量。
这些数据不能推断完整模型延迟。

## 附录 A. 仓库文件

| 用途 | 文件 |
|---|---|
| 公开接口 | `boltops/fa/triangle_attention/__init__.py` |
| Triton kernel | `boltops/fa/triangle_attention/triton/triangle_attention.py` |
| 配置加载 | `boltops/fa/triangle_attention/triton/config.py` |
| JSON 配置 | `boltops/fa/triangle_attention/triton/configs/triton_<capability>/` |
| 调优工具与固定 case | `boltops/tools/triangle_attention_triton_tune/` |
| 测试 | `test/fa/test_triangle_attention.py` |

## 附录 B. 实现备注

- triangle bias 预处理保持输入 dtype，并用负无穷填充超出原始 key 长度的列。
- Q/K/V 始终按 head dimension 掩码加载，包括 `D` 为 2 的幂时；没有单独的
  `EVEN_D` kernel 路径。
