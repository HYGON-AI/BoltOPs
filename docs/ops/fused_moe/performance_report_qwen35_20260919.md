# Fused MoE：E256/N512/K2048，BoltOPs vs Aiter g122f2b

复测日期：2026-09-20。性能数据基于 BoltOPs `da34513` 加本次配置；下表记录该版本的实测结果，不代表后续版本已经重测。

本次提交仅包含 BMZ 完整配置，覆盖 16 个 M（1～16384，不含 32768）。NMZ 训练和正式 Aiter 对比尚未全部完成，配置暂不提交。

配置仅精确匹配 K=2048、topk=8；其他 K/topk 的复用收益尚未验证，本次不提供模糊匹配范围。

调优使用 `MOE_ACTIVATION_DTYPE=bf16`；该环境变量支持 `fp16` / `bf16`，不设置时仍使用原来的 FP16 默认值，调优和 CLI benchmark 使用同一 dtype。

## BMZ：BoltOPs vs Aiter

### 测试环境与口径

- Shape：INT8 channel / E256 / N512 / K2048 / topk=8，BF16 输入输出，SiLU gated，EP1，无 shared expert。
- Aiter：`0.1.6+dtk2604.torch2100.2609181720.g122f2b`，路径 `/root/aiter_016_g122f2b/aiter/__init__.py`。
- Torch：`2.10.0`；Triton：`3.6.0`；GPU：`_CudaDeviceProperties(name='BW200, UBB BW1000', major=9, minor=3, gcnArchName='gfx936:sramecc+:xnack-', total_memory=65520MB, multi_processor_count=80, uuid=30303135-3961-6466-6666-383831313231, pci_bus_id=159, pci_device_id=0, pci_domain_id=0, L2_cache_size=8MB)`。
- 输入遵循 Aiter channel-wise CI：非零随机浮点权重逐输出通道量化、fused_topk 真实 routing；所有 backend 共用输入。
- CUDA Graph 计时，5 次采样取中位数，单位 µs；JIT、数据准备、routing 和权重 shuffle 不计时。
- 正提升表示 BoltOPs 更快：`(Aiter / BoltOPs - 1) × 100%`。
- M≤128 额外与浮点 torch_moe 参考比较，记录 atol=0.5、rtol=0.01 的超差比例；性能与正确性状态分开记录。

### 性能结果

| M | BoltOPs µs | ASM µs | vs ASM | ASM Shuffle µs | vs ASM Shuffle | HIPC/MOE_C 预 Shuffle µs | vs HIPC/MOE_C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.519 | 49.303 | +3.75% | 54.976 | +15.69% | 50.725 | +6.75% |
| 2 | 68.951 | 72.904 | +5.73% | 75.195 | +9.06% | 72.499 | +5.15% |
| 4 | 106.105 | 116.844 | +10.12% | 108.582 | +2.33% | 103.952 | -2.03% |
| 8 | 178.488 | 194.659 | +9.06% | 177.462 | -0.57% | 177.239 | -0.70% |
| 16 | 284.944 | 316.529 | +11.08% | 277.993 | -2.44% | 286.169 | +0.43% |
| 24 | 402.648 | 434.194 | +7.83% | 390.406 | -3.04% | 402.963 | +0.08% |
| 32 | 465.963 | 535.261 | +14.87% | 451.554 | -3.09% | 502.614 | +7.87% |
| 64 | 575.362 | 644.883 | +12.08% | 582.759 | +1.29% | 636.154 | +10.57% |
| 128 | 665.240 | 796.341 | +19.71% | 686.836 | +3.25% | 729.436 | +9.65% |
| 256 | 732.884 | 854.259 | +16.56% | 737.486 | +0.63% | 811.294 | +10.70% |
| 512 | 801.438 | 927.158 | +15.69% | 828.979 | +3.44% | 851.564 | +6.25% |
| 1024 | 932.284 | 986.688 | +5.84% | 947.064 | +1.59% | 1107.554 | +18.80% |
| 2048 | 1124.766 | 1160.154 | +3.15% | 1138.506 | +1.22% | 1260.367 | +12.06% |
| 4096 | 1611.625 | 1565.718 | -2.85% | 1550.690 | -3.78% | 1661.033 | +3.07% |
| 8192 | 2440.706 | 2414.348 | -1.08% | 2381.782 | -2.41% | 2623.252 | +7.48% |
| 16384 | 4260.339 | 4251.665 | -0.20% | 4249.505 | -0.25% | 4817.054 | +13.07% |

### Backend 状态与数值检查

| M | Backend | 状态 / 超差比例 |
|---:|---|---|
| 1 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 1 | asm | ok; finite=True; bad_ratio=0.0000% |
| 1 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 1 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 2 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 2 | asm | ok; finite=True; bad_ratio=0.0000% |
| 2 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 2 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 4 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 4 | asm | ok; finite=True; bad_ratio=0.0000% |
| 4 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 4 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 8 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 8 | asm | ok; finite=True; bad_ratio=0.0000% |
| 8 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 8 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 16 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 16 | asm | ok; finite=True; bad_ratio=0.0000% |
| 16 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 16 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 24 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 24 | asm | ok; finite=True; bad_ratio=0.0000% |
| 24 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 24 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 32 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 32 | asm | ok; finite=True; bad_ratio=0.0000% |
| 32 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 32 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 64 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 64 | asm | ok; finite=True; bad_ratio=0.0000% |
| 64 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 64 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 128 | boltops | ok; finite=True; bad_ratio=0.0000% |
| 128 | asm | ok; finite=True; bad_ratio=0.0000% |
| 128 | asm_shuffle | ok; finite=True; bad_ratio=0.0000% |
| 128 | moe_c | ok; finite=True; bad_ratio=0.0000% |
| 256 | boltops | ok |
| 256 | asm | ok |
| 256 | asm_shuffle | ok |
| 256 | moe_c | ok |
| 512 | boltops | ok |
| 512 | asm | ok |
| 512 | asm_shuffle | ok |
| 512 | moe_c | ok |
| 1024 | boltops | ok |
| 1024 | asm | ok |
| 1024 | asm_shuffle | ok |
| 1024 | moe_c | ok |
| 2048 | boltops | ok |
| 2048 | asm | ok |
| 2048 | asm_shuffle | ok |
| 2048 | moe_c | ok |
| 4096 | boltops | ok |
| 4096 | asm | ok |
| 4096 | asm_shuffle | ok |
| 4096 | moe_c | ok |
| 8192 | boltops | ok |
| 8192 | asm | ok |
| 8192 | asm_shuffle | ok |
| 8192 | moe_c | ok |
| 16384 | boltops | ok |
| 16384 | asm | ok |
| 16384 | asm_shuffle | ok |
| 16384 | moe_c | ok |

## NMZ

测试尚未完成。
