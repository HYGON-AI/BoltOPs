# 非 QSA paged decode：BF16 与 packed FP8

入口：`boltops.mqa_logits.tilelang_mqa_decode`。gfx938 上的独立加权实现，
对齐 QSA decode 的 head/page 范围、普通/persistent 调度和 packed FP8 数据格式。
QSA kernel、配置和现有 prefill 路径不作修改；其它架构暂不支持非 QSA decode。

```python
from boltops.mqa_logits import tilelang_mqa_decode

# q: [B,H,128]，BF16 或 torch.float8_e4m3fn
# k_cache: BF16 [pages,page_size,1,128]
#   或 packed FP8 [pages,page_size,1,132]，uint8 / E4M3FN 字节存储
# weights: [B,H] FP32，可以为负
# page_table: [B,max_pages] int32/int64
# context_lens: [B] int32/int64
logits = tilelang_mqa_decode(
    q, k_cache, weights, page_table, context_lens, max_model_len,
    score_scale=3.25,  # 运行时除数；省略时为 sqrt(D)，1.0 表示不缩放
    # D_out=destination,  # 可选：同设备连续 FP32 [B,max_model_len]
)
```

H 支持 4/8/16/32/64，page_size 支持 8/16/32/64，D 固定为 128。
对有效逻辑 Key 位置 t：

```text
page_id = page_table[b, t // page_size]
key = cache_key[page_id, t % page_size, 0, :]
score = sum_h(relu(dot(q[b,h,:], key)) * weights[b,h])
logits[b,t] = score / Scale                            # BF16
logits[b,t] = score * key_scale[page_id,offset] / Scale   # FP8
```

`score_scale` 是 kernel 的 FP32 运行时参数（除数），默认 `sqrt(D)`，与 QSA 一致。
显式传 `1.0` 可恢复无缩放输出。attention scale 不由调用方折入 weights；
输入权重保持不变，除法在算子内部的 head 规约后完成。Query/head 的量化 scale 仍由 weights 表达。
权重在 ReLU 之后相乘；FP8 每 Key scale 在 head 规约之后相乘，不随 head 重复。
输出为 FP32，`t >= context_lens[b]` 为 `-inf`。
输入权重为 FP32；当前不接受 BF8/E5M2、FNUZ 或未打包的 FP8 cache。

Packed cache 的张量形状为 `[pages,page_size,1,D+4]`，但每页的真实字节内容是
**整页 `page_size*D` 个 Key 字节，后接 `page_size` 个 FP32 scale**，不是逐 token
交错存储。可调用已有只读辅助函数
`common.mqa.pack_fp8_paged_k_cache(keys, scales)` 构造，格式与 QSA 相同。

有效页号须在缓存池范围内；长度须满足
`0 <= context_lens[b] <= min(max_model_len, max_pages * page_size)`。
这些值约束由调用方保证，入口不将 GPU 张量同步回主机校验。
无效/未使用的页表项可以设为 -1；内核只访问有效 context。
入口支持非连续输入、int64 索引转 int32、空 batch/输出和 D_out 校验。

## 计算与调度

| 项目 | 非 QSA 实现 |
| --- | --- |
| 线程与 GEMM | 256 threads / CTA，FullRow，BF16 或 E4M3FN MMAC，FP32 累加 |
| Query | H4/H8 补到 H16；其余保持实际 H，Query 在 Key 循环外加载 |
| 权重 | 按 Query/head 加载一次，跨 Key 轮复用，填充 head 的权重为零 |
| group_size | 支持64/128，公共入口沿用 QSA 配置中的128 |
| 普通分区 | 按页表容量划分 CTA，每 CTA 最多 groups_per_cta 轮 |
| persistent 分区 | 按实际 `ceil(context_len/group_size)` 将 group 连续分给 splits 个 CTA |
| Key | 页表 gather，完整 group 可异步搬运，尾 group 有效加载并补零 |
| FP8 scale | 同页表 gather；默认直接载入 fragment，另支持 scale_async LDS 分支 |
| 输出 | compute 只写有效 Key；公共入口复用已有 suffix mask helper |

page8（group64/128）及 group64/page16、32 特化在当前 HCU 编译器下无法可靠映射异步 gather，
即使请求 `use_async=True` 也明确使用同步 gather，并禁用自动 async 注入。
这一回退同时用于 BF16/FP8，并关闭该特化的软件流水线和 scale async；
保留页表、分区和数值语义。公共配置的 page16/32/64、group128 不受该回退影响；
公共 page8 也明确走同步 gather。group64/page64 及公共 group128/page16/32/64
的异步路径分别验证，不把编译成功视为数值正确。

Persistent S16/S32 在当前编译器下也采用同步 gather、stage0 和显式 CTA 屏障。
自动 ThreadSync 对部分该类分区报参与线程数错误；仅关闭自动同步再补屏障时，
软件流水线会将屏障放到异步等待之前，实测出现少量错误结果。
因此这些分支同时禁用异步搬运和软件流水线，保证读 LDS 前所有写入完成，
并在各 lane 读完后才允许覆盖 LDS。`scale_async=True` 在此分支也回退为直接
加载 scale fragment。分区数、group 大小、页表与数学语义不变，
但搬运策略与 QSA 不完全相同；这部分后续可以单独优化性能。

配置位于独立的 `configs/mqa_decode/weighted.py`，初始规则与 gfx938 QSA 表一致，
分别选择 BF16/FP8 配置。只复用不可变配置记录、小 batch 规则和公共 suffix mask，
不复用 QSA compute kernel；不宣称 QSA 的配置已经是非 QSA 的性能最优配置。
内部 `_weighted_decode_kernel` 通过编译期参数生成 BF16/FP8、普通/persistent 四种分支，
不需要在 GPU 运行时判断数据类型或切换调度。

例如 B16/H4/page64/容量8192：BF16 使用 group128/GPC8 普通模式，共128 CTA；
FP8 使用 persistent S16，共256 CTA。短 context 跳过无效 group，但 CTA 本身仍会启动。

## 测试覆盖对照

| QSA 已有项目 | 非 QSA 对应覆盖 |
| --- | --- |
| BF16 公共入口 page8/16/32/64、H4/8/16/32/64 | 两个数据类型均测试完整 page×head 矩阵 |
| Packed FP8 公共入口 H4/16/32/64 | 对应范围及额外 H8；signed 权重、零 scale、两种字节存储 |
| 普通 group128/GPC4，H16、尾块 | 所有 head、BF16/FP8，额外整块、多轮、CTA 边界、奇数页数 |
| Persistent group128，H16/32/64 × S2/4/8/16/32 | 扩展至全部5种 head 和两个数据类型；短任务与真正多轮长任务 |
| Suffix mask 切分、reverse、边界 | 公用同一个 helper；回归原有 mask 用例，公共入口覆盖长短后缀 |
| BF16/FP8 配置表、小 batch、宽 head、长 context | 独立配置与 QSA 全表对照；真实公共入口覆盖 B1/8/16/32/64、8K至128K |
| Compute benchmark | 独立 BF16/FP8、H4/8/16/32/64 compute-only benchmark |
| — | 额外检查无效列 sentinel、-1 无效页、负权重、非连续输入、空 context 和接口拒绝 |

参考实现独立按页表构造逻辑 Key，以实际 BF16/FP8 输入转 FP32 做 Torch einsum、
ReLU、signed weighting 和规约；FP8 按字节合同解析 cache 并在规约后乘 Key scale。
对全部输出比较，BF16 `atol=rtol=2e-3`，FP8 `atol=rtol=2e-2`。
全一权重另外与相同运行时 `score_scale` 的 QSA 输出交叉比较。
新增用例在 BF16/FP8、全部 head、普通/S16 调度下复用同一个编译好的 kernel，
连续传入 `1.0`、`2.5`、`sqrt(128)`，检查 Torch 数值、无效列和输入权重保持不变。

```bash
HIP_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
  python -m pytest \
    test/mqa_logits/test_tilelang_mqa_decode_weighted.py \
    test/mqa_logits/test_tilelang_mqa_decode_weighted_config.py -q --runslow
```

不加 `--runslow` 仅运行默认 smoke/接口用例；性能用例另需 `--bench`。

## 与 Triton 参考的关系

BF16 的 FP32 点积、ReLU、signed head 加权和规约定义与 Triton 一致。
FP8 的 Triton Torch 参考先将 Key 乘逐 token scale 后做点积；这里先规约再乘 Key scale。
对逐 Key 共享、非负的量化 scale，两者数学等价，浮点舍入可能不同。
当前实现不检查 Key scale 的正负；负值按上述规约后缩放定义执行，不据此声明与 Triton 等价。
测试使用独立 Torch FP32 参考，没有直接调用 Triton 的参考函数。

## 提交前验证

原有 QSA kernel、配置、后缀 mask 以及 prefill 实现与父提交逐字节一致；
已有公共函数保持不变，dispatch 只增加 weighted decode 条目。
新实现仅复用不可变配置记录、小 batch 配置辅助函数和原有后缀 mask helper。

2026-11-20 提交前验证（GPU 6，gfx938）：

- 包导出 smoke：8 passed。
- 全量尝试完成57项，均通过；因特化编译耗时主动中断，未宣称全量完成。
- 补充回归：56 passed、1 failed；包含全部 head 的 BF16/FP8 普通及 S16 运行时 Scale、
  FP8 scale_async 的普通/S4/S16/S32 分支、独立配置与原 QSA 入口。
- 唯一失败是未修改的原 QSA `test_qsa_mqa_decode_group128_persistent_matches_torch[16-16]`，
  当前编译器报 ThreadSync 的254个参与线程不是64的整数倍，与此前记录一致。
- 独立进程仅加载原 QSA 测试、未导入任何 weighted decode 模块时，同样复现该 ThreadSync 错误。
- 原 QSA decode 入口、辅助函数及配置也与添加非 QSA 前的 `52f132a` 核对，保持不变；
  decode 开始前保存的28个原文件哈希也全部一致。

完整尝试和补充回归的日志分别为
`/tmp/mqa_decode_precommit_gpu_20261120.log`、
`/tmp/mqa_decode_precommit_targeted_20261120.log`。
