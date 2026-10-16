# GEMM 支持需求清单

来源：`/xukang/GEMM支持清单.html`（生成日期 2026-10-12）。按算子类型及输入、输出精度分为 9 张表，完整保留 511 行需求和 2504 条算子实例。

原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

本清单用于逐项记录支持和优化进度。所有条目初始标记为「待验证」，不代表已有内核不支持，也不代表历史 FP16 调优结果已经覆盖这些需求。实验详情与记录规范见 [OPTIMIZATION.md](OPTIMIZATION.md)。

## 记录口径

- M/N/K 沿用 HTML 中的激活行数、输出通道数和规约维，不据此推断矩阵 stride。
- batch / experts 沿用源数据；bmm 表示 batch，moe_grouped_gemm 表示 experts，linear 的该字段也按原值保留。
- 输出精度指写回显存的精度，与累加精度分开验证。
- scale 信息原样保留；含推断值的条目接入模型时需核实。
- 状态可记为「待验证」「编译通过」「正确性通过」「已调优」或「受阻」，最后一列链接实验记录，注明版本、误差、性能和产物。
- ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 511 行中，仍以原 HTML 及优化记录中的说明为准。

## 分类索引

| 表 | 算子 | A × B → 输出 | 需求行数 | 实例数 |
| --- | --- | --- | ---: | ---: |
| [表 1](#table-1) | bmm | BF16 × BF16 → BF16 | 29 | 194 |
| [表 2](#table-2) | linear | FP8E4M3 × FP8E4M3 → BF16 | 207 | 1206 |
| [表 3](#table-3) | linear | BF16 × BF16 → BF16 | 133 | 326 |
| [表 4](#table-4) | linear | FP32 × FP32 → FP32 | 35 | 218 |
| [表 5](#table-5) | linear | BF16 × BF16 → FP32 | 27 | 172 |
| [表 6](#table-6) | linear | BF16 × INT4 → BF16 | 4 | 4 |
| [表 7](#table-7) | moe_grouped_gemm | FP8E4M3 × FP8E4M3 → BF16 | 36 | 70 |
| [表 8](#table-8) | moe_grouped_gemm | FP8E4M3 × FP4E2M1 → BF16 | 34 | 304 |
| [表 9](#table-9) | moe_grouped_gemm | BF16 × INT4 → BF16 | 6 | 10 |

<a id="table-1"></a>

## 表 1：bmm · BF16 × BF16 → BF16

29 行需求，累计 194 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G001 | 6 × 256 × 512 | 64 | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | vb_bmm | 待验证 | — |
| G002 | 6 × 512 × 192 | 64 | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G003 | 12 × 256 × 512 | 64 | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | vb_bmm | 待验证 | — |
| G004 | 12 × 512 × 192 | 64 | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G005 | 16 × 128 × 512 | 64 | decode | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | vb_bmm | 待验证 | — |
| G006 | 16 × 512 × 128 | 64 | decode | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G007 | 24 × 256 × 512 | 64 | decode | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | vb_bmm | 待验证 | — |
| G008 | 24 × 512 × 192 | 64 | decode | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G009 | 48 × 256 × 512 | 64 | decode | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | vb_bmm | 待验证 | — |
| G010 | 48 × 512 × 192 | 64 | decode | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G011 | 80 × 1024 × 4096 | 16 | decode | 6 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G012 | 96 × 1024 × 4096 | 16 | decode | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G013 | 160 × 1024 × 4096 | 8 | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wo_a | 待验证 | — |
| G014 | 192 × 1024 × 4096 | 8 | decode | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wo_a | 待验证 | — |
| G015 | 240 × 1024 × 4096 | 8 | decode | 9 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wo_a | 待验证 | — |
| G016 | 288 × 1024 × 4096 | 8 | decode | 12 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wo_a | 待验证 | — |
| G017 | 2560 × 256 × 512 | 64 | prefill | 4 | — | — | GLM-5.1/FP8[64k/1k/s2000] | vb_bmm | 待验证 | — |
| G018 | 2560 × 512 × 192 | 64 | prefill | 4 | — | — | GLM-5.1/FP8[64k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G019 | 2560 × 1024 × 4096 | 16 | prefill | 12 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G020 | 2560 × 1024 × 4096 | 8 | prefill | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G021 | 3584 × 1024 × 4096 | 8 | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | wo_a | 待验证 | — |
| G022 | 4608 × 256 × 512 | 64 | prefill | 4 | — | — | GLM-5.1/FP8[128k/1k/s2000] | vb_bmm | 待验证 | — |
| G023 | 4608 × 512 × 192 | 64 | prefill | 4 | — | — | GLM-5.1/FP8[128k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G024 | 4608 × 1024 × 4096 | 16 | prefill | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G025 | 5120 × 1024 × 4096 | 8 | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | wo_a | 待验证 | — |
| G026 | 6656 × 256 × 512 | 64 | prefill | 8 | — | — | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | vb_bmm | 待验证 | — |
| G027 | 6656 × 512 × 192 | 64 | prefill | 8 | — | — | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | q_nope_bmm | 待验证 | — |
| G028 | 6656 × 1024 × 4096 | 8 | prefill | 36 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |
| G029 | 6656 × 1024 × 4096 | 16 | prefill | 16 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_a | 待验证 | — |

<a id="table-2"></a>

## 表 2：linear · FP8E4M3 × FP8E4M3 → BF16

207 行需求，累计 1206 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G030 | 4 × 3072 × 4096 |  | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G031 | 4 × 4096 × 1536 |  | decode | 1 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G032 | 4 × 4096 × 8192 |  | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G033 | 4 × 10240 × 4096 |  | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G034 | 6 × 128 × 6144 |  | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wk | 待验证 | — |
| G035 | 6 × 2624 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G036 | 6 × 4096 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G037 | 6 × 4096 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G038 | 6 × 6144 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G039 | 6 × 6144 × 16384 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G040 | 6 × 16384 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G041 | 12 × 128 × 6144 |  | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wk | 待验证 | — |
| G042 | 12 × 2624 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G043 | 12 × 4096 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G044 | 12 × 4096 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G045 | 12 × 6144 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G046 | 12 × 6144 × 16384 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | o_proj | 待验证 | — |
| G047 | 12 × 16384 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | q_b_proj | 待验证 | — |
| G048 | 16 × 3072 × 4096 |  | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G049 | 16 × 4096 × 1536 |  | decode | 1 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G050 | 16 × 4096 × 8192 |  | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G051 | 16 × 4096 × 13312 |  | decode | 1 | FP32@1×13312 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G052 | 16 × 10240 × 4096 |  | decode | 2 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G053 | 16 × 26624 × 4096 |  | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G054 | 24 × 128 × 6144 |  | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wk | 待验证 | — |
| G055 | 24 × 2624 × 6144 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G056 | 24 × 4096 × 2048 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G057 | 24 × 4096 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G058 | 24 × 6144 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G059 | 24 × 6144 × 12288 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G060 | 24 × 6144 × 16384 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G061 | 24 × 16384 × 2048 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G062 | 24 × 24576 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G063 | 48 × 128 × 6144 |  | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wk | 待验证 | — |
| G064 | 48 × 2048 × 4096 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G065 | 48 × 2624 × 6144 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G066 | 48 × 4096 × 1024 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G067 | 48 × 4096 × 2048 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G068 | 48 × 4096 × 4096 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G069 | 48 × 4096 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G070 | 48 × 6144 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G071 | 48 × 6144 × 12288 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G072 | 48 × 6144 × 16384 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | o_proj | 待验证 | — |
| G073 | 48 × 8704 × 4096 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G074 | 48 × 16384 × 2048 |  | decode | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | q_b_proj | 待验证 | — |
| G075 | 48 × 24576 × 6144 |  | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G076 | 64 × 1024 × 2048 |  | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G077 | 64 × 2048 × 512 |  | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_down_proj | 待验证 | — |
| G078 | 64 × 2048 × 4096 |  | decode | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G079 | 64 × 9216 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | qkv_proj | 待验证 | — |
| G080 | 64 × 12288 × 2048 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G081 | 80 × 512 × 7168 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wkv | 待验证 | — |
| G082 | 80 × 1536 × 7168 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_a | 待验证 | — |
| G083 | 80 × 6144 × 7168 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G084 | 80 × 7168 × 3072 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G085 | 80 × 7168 × 16384 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G086 | 80 × 65536 × 1536 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G087 | 96 × 512 × 7168 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wkv | 待验证 | — |
| G088 | 96 × 1536 × 7168 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_a | 待验证 | — |
| G089 | 96 × 6144 × 7168 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G090 | 96 × 7168 × 3072 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G091 | 96 × 7168 × 16384 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G092 | 96 × 8192 × 1536 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G093 | 96 × 65536 × 1536 |  | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G094 | 160 × 512 × 4096 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wkv | 待验证 | — |
| G095 | 160 × 1024 × 4096 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wq_a | 待验证 | — |
| G096 | 160 × 4096 × 2048 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | shared_down_proj | 待验证 | — |
| G097 | 160 × 4096 × 4096 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G098 | 160 × 4096 × 8192 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wo_b | 待验证 | — |
| G099 | 160 × 32768 × 1024 |  | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wq_b | 待验证 | — |
| G100 | 192 × 512 × 4096 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wkv | 待验证 | — |
| G101 | 192 × 1024 × 4096 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wq_a | 待验证 | — |
| G102 | 192 × 2048 × 4096 |  | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G103 | 192 × 4096 × 1024 |  | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G104 | 192 × 4096 × 2048 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | shared_down_proj | 待验证 | — |
| G105 | 192 × 4096 × 4096 |  | decode | 6 | FP8E8M0@1×128(4) / FP32@1×128(2) | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | o_proj, out_proj, shared_gate_up_proj | 待验证 | — |
| G106 | 192 × 4096 × 8192 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wo_b | 待验证 | — |
| G107 | 192 × 8192 × 1024 |  | decode | 2 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | indexer_wq_b | 待验证 | — |
| G108 | 192 × 8704 × 4096 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G109 | 192 × 10240 × 4096 |  | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_qkvz | 待验证 | — |
| G110 | 192 × 32768 × 1024 |  | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | wq_b | 待验证 | — |
| G111 | 240 × 512 × 4096 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wkv | 待验证 | — |
| G112 | 240 × 1024 × 4096 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_a | 待验证 | — |
| G113 | 240 × 4096 × 2048 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G114 | 240 × 4096 × 4096 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G115 | 240 × 4096 × 8192 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wo_b | 待验证 | — |
| G116 | 240 × 32768 × 1024 |  | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_b | 待验证 | — |
| G117 | 288 × 512 × 4096 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wkv | 待验证 | — |
| G118 | 288 × 1024 × 4096 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_a | 待验证 | — |
| G119 | 288 × 4096 × 2048 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G120 | 288 × 4096 × 4096 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G121 | 288 × 4096 × 8192 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wo_b | 待验证 | — |
| G122 | 288 × 8192 × 1024 |  | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G123 | 288 × 32768 × 1024 |  | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_b | 待验证 | — |
| G124 | 512 × 384 × 4096 |  | prefill | 2 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G125 | 512 × 3328 × 4096 |  | prefill | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G126 | 512 × 4096 × 192 |  | prefill | 2 | FP32@1×192 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G127 | 512 × 4096 × 1664 |  | prefill | 1 | FP32@1×1664 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G128 | 512 × 4096 × 8192 |  | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G129 | 512 × 10240 × 4096 |  | prefill | 4 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G130 | 2560 × 128 × 6144 |  | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wk | 待验证 | — |
| G131 | 2560 × 512 × 6144 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G132 | 2560 × 512 × 7168 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wkv | 待验证 | — |
| G133 | 2560 × 768 × 7168 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G134 | 2560 × 1024 × 2048 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G135 | 2560 × 1536 × 7168 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_a | 待验证 | — |
| G136 | 2560 × 2048 × 512 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_down_proj | 待验证 | — |
| G137 | 2560 × 2048 × 4096 |  | prefill | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G138 | 2560 × 2624 × 6144 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G139 | 2560 × 3072 × 6144 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G140 | 2560 × 4096 × 2048 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G141 | 2560 × 6144 × 256 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G142 | 2560 × 6144 × 1536 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G143 | 2560 × 6144 × 16384 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | o_proj | 待验证 | — |
| G144 | 2560 × 7168 × 384 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G145 | 2560 × 7168 × 8192 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G146 | 2560 × 7168 × 16384 |  | prefill | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G147 | 2560 × 8192 × 1536 |  | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G148 | 2560 × 9216 × 2048 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | qkv_proj | 待验证 | — |
| G149 | 2560 × 12288 × 2048 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G150 | 2560 × 16384 × 2048 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | q_b_proj | 待验证 | — |
| G151 | 2560 × 32768 × 1536 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G152 | 2560 × 65536 × 1536 |  | prefill | 12 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G153 | 3584 × 256 × 4096 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G154 | 3584 × 512 × 4096 |  | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | shared_gate_up_proj, wkv | 待验证 | — |
| G155 | 3584 × 1024 × 4096 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | wq_a | 待验证 | — |
| G156 | 3584 × 4096 × 128 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G157 | 3584 × 4096 × 256 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G158 | 3584 × 4096 × 8192 |  | prefill | 19 | FP8E8M0@1×128(16) / FP8E8M0/FP32@1×128 (推断)(2) / FP32@1×128(1) | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | o_proj, out_proj, wo_b | 待验证 | — |
| G159 | 3584 × 8192 × 1024 |  | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G160 | 3584 × 17408 × 4096 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G161 | 3584 × 20480 × 4096 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_qkvz | 待验证 | — |
| G162 | 3584 × 32768 × 1024 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | wq_b | 待验证 | — |
| G163 | 4096 × 3328 × 4096 |  | prefill | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G164 | 4096 × 4096 × 1664 |  | prefill | 1 | FP32@1×1664 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G165 | 4608 × 128 × 6144 |  | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wk | 待验证 | — |
| G166 | 4608 × 512 × 6144 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G167 | 4608 × 512 × 7168 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wkv | 待验证 | — |
| G168 | 4608 × 768 × 7168 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G169 | 4608 × 1536 × 7168 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_a | 待验证 | — |
| G170 | 4608 × 2624 × 6144 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G171 | 4608 × 3072 × 6144 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G172 | 4608 × 4096 × 2048 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G173 | 4608 × 6144 × 256 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G174 | 4608 × 6144 × 1536 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G175 | 4608 × 6144 × 16384 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | o_proj | 待验证 | — |
| G176 | 4608 × 7168 × 384 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G177 | 4608 × 7168 × 16384 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G178 | 4608 × 8192 × 1536 |  | prefill | 2 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G179 | 4608 × 16384 × 2048 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G180 | 4608 × 65536 × 1536 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G181 | 5120 × 512 × 4096 |  | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | shared_gate_up_proj, wkv | 待验证 | — |
| G182 | 5120 × 1024 × 4096 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | wq_a | 待验证 | — |
| G183 | 5120 × 4096 × 256 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G184 | 5120 × 4096 × 8192 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | wo_b | 待验证 | — |
| G185 | 5120 × 8192 × 1024 |  | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G186 | 5120 × 32768 × 1024 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | wq_b | 待验证 | — |
| G187 | 6656 × 128 × 6144 |  | prefill | 8 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | indexer_wk | 待验证 | — |
| G188 | 6656 × 256 × 4096 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G189 | 6656 × 384 × 4096 |  | prefill | 2 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G190 | 6656 × 512 × 4096 |  | prefill | 64 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_gate_up_proj, wkv | 待验证 | — |
| G191 | 6656 × 512 × 6144 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G192 | 6656 × 512 × 7168 |  | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wkv | 待验证 | — |
| G193 | 6656 × 768 × 7168 |  | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G194 | 6656 × 1024 × 2048 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G195 | 6656 × 1024 × 4096 |  | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_a | 待验证 | — |
| G196 | 6656 × 1536 × 7168 |  | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_a | 待验证 | — |
| G197 | 6656 × 2048 × 512 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | shared_down_proj | 待验证 | — |
| G198 | 6656 × 2048 × 4096 |  | prefill | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G199 | 6656 × 2624 × 6144 |  | prefill | 8 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G200 | 6656 × 3072 × 6144 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G201 | 6656 × 3328 × 4096 |  | prefill | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G202 | 6656 × 4096 × 128 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G203 | 6656 × 4096 × 192 |  | prefill | 2 | FP32@1×192 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G204 | 6656 × 4096 × 256 |  | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G205 | 6656 × 4096 × 1664 |  | prefill | 1 | FP32@1×1664 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G206 | 6656 × 4096 × 2048 |  | prefill | 8 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G207 | 6656 × 4096 × 8192 |  | prefill | 39 | FP8E8M0@1×128(32) / FP8E8M0/FP32@1×128 (推断)(6) / FP32@1×128(1) | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; Hunyuan-V3/FP8[128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | o_proj, out_proj, wo_b | 待验证 | — |
| G208 | 6656 × 6144 × 256 |  | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | shared_down_proj | 待验证 | — |
| G209 | 6656 × 6144 × 1536 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G210 | 6656 × 6144 × 16384 |  | prefill | 8 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | o_proj | 待验证 | — |
| G211 | 6656 × 7168 × 384 |  | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G212 | 6656 × 7168 × 8192 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G213 | 6656 × 7168 × 16384 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wo_b | 待验证 | — |
| G214 | 6656 × 8192 × 1024 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G215 | 6656 × 8192 × 1536 |  | prefill | 10 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_wq_b | 待验证 | — |
| G216 | 6656 × 9216 × 2048 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | qkv_proj | 待验证 | — |
| G217 | 6656 × 10240 × 4096 |  | prefill | 4 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G218 | 6656 × 12288 × 2048 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G219 | 6656 × 16384 × 2048 |  | prefill | 8 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | q_b_proj | 待验证 | — |
| G220 | 6656 × 17408 × 4096 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | qkv_proj | 待验证 | — |
| G221 | 6656 × 20480 × 4096 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_qkvz | 待验证 | — |
| G222 | 6656 × 32768 × 1024 |  | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | wq_b | 待验证 | — |
| G223 | 6656 × 32768 × 1536 |  | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G224 | 6656 × 65536 × 1536 |  | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | wq_b | 待验证 | — |
| G225 | 20480 × 3072 × 6144 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G226 | 20480 × 6144 × 1536 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G227 | 28672 × 2560 × 4096 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_qkvz | 待验证 | — |
| G228 | 28672 × 4096 × 1024 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | out_proj | 待验证 | — |
| G229 | 36864 × 3072 × 6144 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G230 | 36864 × 6144 × 1536 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G231 | 53248 × 2560 × 4096 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_qkvz | 待验证 | — |
| G232 | 53248 × 3072 × 6144 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G233 | 53248 × 3328 × 4096 |  | prefill | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G234 | 53248 × 4096 × 1024 |  | prefill | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | out_proj | 待验证 | — |
| G235 | 53248 × 4096 × 1664 |  | prefill | 1 | FP32@1×1664 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G236 | 53248 × 6144 × 1536 |  | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | mlp_down_proj | 待验证 | — |

<a id="table-3"></a>

## 表 3：linear · BF16 × BF16 → BF16

133 行需求，累计 326 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G237 | 1 × 154880 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G238 | 2 × 64640 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G239 | 2 × 129280 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G240 | 3 × 120832 × 4096 |  | prefill | 2 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G241 | 4 × 192 × 4096 |  | decode | 1 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G242 | 4 × 4096 × 8192 |  | decode | 1 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | nextn_eh_proj | 待验证 | — |
| G243 | 4 × 7168 × 21504 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | eagle3_fc | 待验证 | — |
| G244 | 4 × 120832 × 4096 |  | decode | 1 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G245 | 4 × 154880 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G246 | 4 × 163840 × 7168 |  | decode,prefill | 3 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | lm_head_decode, lm_head_prefill | 待验证 | — |
| G247 | 6 × 32 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | indexers_proj | 待验证 | — |
| G248 | 6 × 256 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G249 | 6 × 6144 × 12288 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | nextn_eh_proj | 待验证 | — |
| G250 | 6 × 154880 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G251 | 8 × 16160 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G252 | 8 × 19360 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G253 | 8 × 129280 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G254 | 8 × 248320 × 4096 |  | prefill | 2 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G255 | 12 × 32 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | indexers_proj | 待验证 | — |
| G256 | 12 × 64 × 2048 |  | decode | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_ba | 待验证 | — |
| G257 | 12 × 256 × 2048 |  | decode | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | fused_moe_gate | 待验证 | — |
| G258 | 12 × 256 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G259 | 12 × 1024 × 2048 |  | decode | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G260 | 12 × 2048 × 512 |  | decode | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_down_proj | 待验证 | — |
| G261 | 12 × 2048 × 4096 |  | decode | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G262 | 12 × 6144 × 12288 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | nextn_eh_proj | 待验证 | — |
| G263 | 12 × 9216 × 2048 |  | decode | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | qkv_proj | 待验证 | — |
| G264 | 12 × 12288 × 2048 |  | decode | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G265 | 12 × 154880 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | lm_head_decode | 待验证 | — |
| G266 | 12 × 248320 × 2048 |  | decode,prefill | 3 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | lm_head_decode, lm_head_prefill | 待验证 | — |
| G267 | 16 × 192 × 4096 |  | decode | 1 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G268 | 16 × 384 × 7168 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G269 | 16 × 2112 × 7168 |  | decode | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G270 | 16 × 4096 × 7168 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G271 | 16 × 7168 × 2048 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G272 | 16 × 7168 × 8192 |  | decode | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | o_proj | 待验证 | — |
| G273 | 16 × 7168 × 18432 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G274 | 16 × 12288 × 1536 |  | decode | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G275 | 16 × 16160 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G276 | 16 × 36864 × 7168 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G277 | 16 × 120832 × 4096 |  | decode | 1 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G278 | 16 × 163840 × 7168 |  | decode | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G279 | 24 × 32 × 6144 |  | decode | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | indexers_proj | 待验证 | — |
| G280 | 24 × 256 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G281 | 24 × 15104 × 4096 |  | prefill | 2 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G282 | 24 × 154880 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G283 | 32 × 19360 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G284 | 32 × 20480 × 7168 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G285 | 48 × 32 × 6144 |  | decode | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | indexers_proj | 待验证 | — |
| G286 | 48 × 256 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G287 | 48 × 512 × 4096 |  | decode | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G288 | 48 × 4096 × 8192 |  | decode | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | mtp_fc | 待验证 | — |
| G289 | 48 × 124160 × 4096 |  | decode | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G290 | 48 × 154880 × 6144 |  | decode | 1 | — | — | GLM-5.1/FP8[64k/1k/s2000] | lm_head_decode | 待验证 | — |
| G291 | 64 × 64 × 2048 |  | decode | 1 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | in_proj_ba | 待验证 | — |
| G292 | 64 × 256 × 2048 |  | decode | 2 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | fused_moe_gate | 待验证 | — |
| G293 | 64 × 16160 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G294 | 64 × 31040 × 4096 |  | prefill | 2 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | lm_head_prefill | 待验证 | — |
| G295 | 64 × 248320 × 2048 |  | decode,prefill | 3 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | lm_head_decode, lm_head_prefill | 待验证 | — |
| G296 | 80 × 129280 × 7168 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G297 | 96 × 64 × 7168 |  | decode | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G298 | 96 × 512 × 7168 |  | decode | 6 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | dspark_ctx_wkv_0, dspark_ctx_wkv_1, dspark_ctx_wkv_2 | 待验证 | — |
| G299 | 96 × 7168 × 21504 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | dspark_main_proj | 待验证 | — |
| G300 | 96 × 129280 × 7168 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G301 | 160 × 129280 × 4096 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | lm_head_decode | 待验证 | — |
| G302 | 192 × 64 × 4096 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_ba, indexer_weights_proj | 待验证 | — |
| G303 | 192 × 512 × 4096 |  | decode | 5 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | dspark_ctx_wkv_0, dspark_ctx_wkv_1, dspark_ctx_wkv_2, fused_moe_gate | 待验证 | — |
| G304 | 192 × 4096 × 12288 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | dspark_main_proj | 待验证 | — |
| G305 | 192 × 124160 × 4096 |  | decode | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | lm_head_decode | 待验证 | — |
| G306 | 192 × 129280 × 4096 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | lm_head_decode | 待验证 | — |
| G307 | 240 × 129280 × 4096 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | lm_head_decode | 待验证 | — |
| G308 | 288 × 64 × 4096 |  | decode | 6 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G309 | 288 × 512 × 4096 |  | decode | 9 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | dspark_ctx_wkv_0, dspark_ctx_wkv_1, dspark_ctx_wkv_2 | 待验证 | — |
| G310 | 288 × 4096 × 12288 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | dspark_main_proj | 待验证 | — |
| G311 | 288 × 129280 × 4096 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | lm_head_decode | 待验证 | — |
| G312 | 512 × 192 × 4096 |  | prefill | 2 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G313 | 2048 × 64 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_ba | 待验证 | — |
| G314 | 2048 × 256 × 2048 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | fused_moe_gate | 待验证 | — |
| G315 | 2048 × 1024 × 2048 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G316 | 2048 × 2048 × 512 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_down_proj | 待验证 | — |
| G317 | 2048 × 2048 × 4096 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G318 | 2048 × 9216 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | qkv_proj | 待验证 | — |
| G319 | 2048 × 12288 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G320 | 2560 × 32 × 6144 |  | prefill | 4 | — | — | GLM-5.1/FP8[64k/1k/s2000] | indexers_proj | 待验证 | — |
| G321 | 2560 × 64 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | in_proj_ba | 待验证 | — |
| G322 | 2560 × 64 × 7168 |  | prefill | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G323 | 2560 × 256 × 2048 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | fused_moe_gate | 待验证 | — |
| G324 | 2560 × 256 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G325 | 3584 × 64 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G326 | 3584 × 128 × 4096 |  | prefill | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_ba | 待验证 | — |
| G327 | 3584 × 512 × 4096 |  | prefill | 4 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G328 | 4608 × 32 × 6144 |  | prefill | 4 | — | — | GLM-5.1/FP8[128k/1k/s2000] | indexers_proj | 待验证 | — |
| G329 | 4608 × 64 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G330 | 4608 × 256 × 6144 |  | prefill | 2 | — | — | GLM-5.1/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G331 | 5120 × 64 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G332 | 5120 × 384 × 7168 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G333 | 5120 × 512 × 7168 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G334 | 5120 × 2112 × 7168 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G335 | 5120 × 4608 × 7168 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G336 | 5120 × 7168 × 256 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G337 | 5120 × 7168 × 2304 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G338 | 5120 × 7168 × 8192 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | o_proj | 待验证 | — |
| G339 | 5120 × 12288 × 1536 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G340 | 5120 × 16384 × 512 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | kv_b_proj | 待验证 | — |
| G341 | 6656 × 32 × 6144 |  | prefill | 8 | — | — | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | indexers_proj | 待验证 | — |
| G342 | 6656 × 64 × 2048 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150]; Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_ba | 待验证 | — |
| G343 | 6656 × 64 × 4096 |  | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G344 | 6656 × 64 × 7168 |  | prefill | 10 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_weights_proj | 待验证 | — |
| G345 | 6656 × 128 × 4096 |  | prefill | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_ba | 待验证 | — |
| G346 | 6656 × 192 × 4096 |  | prefill | 2 | — | — | Hunyuan-V3/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G347 | 6656 × 256 × 2048 |  | prefill | 4 | — | — | Qwen3.5-35B-A3B/FP8[4k/1k/s150]; Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | fused_moe_gate | 待验证 | — |
| G348 | 6656 × 256 × 6144 |  | prefill | 4 | — | — | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G349 | 6656 × 384 × 7168 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G350 | 6656 × 512 × 4096 |  | prefill | 4 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G351 | 6656 × 512 × 7168 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_gate_up_proj | 待验证 | — |
| G352 | 6656 × 1024 × 2048 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_gate_up_proj | 待验证 | — |
| G353 | 6656 × 2048 × 512 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | shared_down_proj | 待验证 | — |
| G354 | 6656 × 2048 × 4096 |  | prefill | 2 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | o_proj, out_proj | 待验证 | — |
| G355 | 6656 × 2112 × 7168 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | fused_qkv_a_proj | 待验证 | — |
| G356 | 6656 × 4608 × 7168 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G357 | 6656 × 7168 × 256 |  | prefill | 2 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | shared_down_proj | 待验证 | — |
| G358 | 6656 × 7168 × 2304 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G359 | 6656 × 7168 × 8192 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | o_proj | 待验证 | — |
| G360 | 6656 × 9216 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | qkv_proj | 待验证 | — |
| G361 | 6656 × 12288 × 1536 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | q_b_proj | 待验证 | — |
| G362 | 6656 × 12288 × 2048 |  | prefill | 1 | — | — | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | in_proj_qkvz | 待验证 | — |
| G363 | 6656 × 16384 × 512 |  | prefill | 4 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | kv_b_proj | 待验证 | — |
| G364 | 28672 × 16 × 4096 |  | prefill | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_ba | 待验证 | — |
| G365 | 40960 × 4608 × 7168 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G366 | 40960 × 7168 × 2304 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_down_proj | 待验证 | — |
| G367 | 53248 × 16 × 4096 |  | prefill | 1 | — | — | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | in_proj_ba | 待验证 | — |
| G368 | 53248 × 4608 × 7168 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_gate_up_proj | 待验证 | — |
| G369 | 53248 × 7168 × 2304 |  | prefill | 1 | — | — | Kimi-2.6/W4A16[128k/1k/s2000] | mlp_down_proj | 待验证 | — |

<a id="table-4"></a>

## 表 4：linear · FP32 × FP32 → FP32

35 行需求，累计 218 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G370 | 16 × 129280 × 512 |  | decode | 10 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | dspark_markov_w2_0, dspark_markov_w2_1, dspark_markov_w2_2, dspark_markov_w2_3, dspark_markov_w2_4 | 待验证 | — |
| G371 | 32 × 129280 × 256 |  | decode | 5 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | dspark_markov_w2_0, dspark_markov_w2_1, dspark_markov_w2_2, dspark_markov_w2_3, dspark_markov_w2_4 | 待验证 | — |
| G372 | 48 × 129280 × 256 |  | decode | 15 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | dspark_markov_w2_0, dspark_markov_w2_1, dspark_markov_w2_2, dspark_markov_w2_3, dspark_markov_w2_4 | 待验证 | — |
| G373 | 80 × 1 × 7680 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | dspark_confidence | 待验证 | — |
| G374 | 80 × 4 × 28672 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G375 | 80 × 384 × 7168 |  | decode | 6 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G376 | 96 × 4 × 28672 |  | decode | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G377 | 96 × 384 × 7168 |  | decode | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G378 | 160 × 1 × 4352 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | dspark_confidence | 待验证 | — |
| G379 | 160 × 4 × 16384 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | hc_head_linear | 待验证 | — |
| G380 | 160 × 256 × 4096 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | fused_moe_gate | 待验证 | — |
| G381 | 192 × 4 × 16384 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | hc_head_linear | 待验证 | — |
| G382 | 192 × 256 × 4096 |  | decode | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | fused_moe_gate | 待验证 | — |
| G383 | 240 × 1 × 4352 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | dspark_confidence | 待验证 | — |
| G384 | 240 × 4 × 16384 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G385 | 240 × 256 × 4096 |  | decode | 9 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G386 | 288 × 4 × 16384 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G387 | 288 × 256 × 4096 |  | decode | 12 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G388 | 2560 × 4 × 28672 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G389 | 2560 × 384 × 7168 |  | prefill | 16 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G390 | 3584 × 4 × 16384 |  | prefill | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G391 | 3584 × 256 × 4096 |  | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G392 | 4608 × 384 × 7168 |  | prefill | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G393 | 5120 × 4 × 16384 |  | prefill | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G394 | 5120 × 256 × 4096 |  | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G395 | 6656 × 4 × 16384 |  | prefill | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G396 | 6656 × 4 × 28672 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G397 | 6656 × 256 × 4096 |  | prefill | 32 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G398 | 6656 × 384 × 7168 |  | prefill | 20 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | fused_moe_gate | 待验证 | — |
| G399 | 20480 × 4 × 28672 |  | prefill | 1 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G400 | 28672 × 4 × 16384 |  | prefill | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G401 | 36864 × 4 × 28672 |  | prefill | 1 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |
| G402 | 40960 × 4 × 16384 |  | prefill | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G403 | 53248 × 4 × 16384 |  | prefill | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | hc_head_linear | 待验证 | — |
| G404 | 53248 × 4 × 28672 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | hc_head_linear | 待验证 | — |

<a id="table-5"></a>

## 表 5：linear · BF16 × BF16 → FP32

27 行需求，累计 172 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G405 | 96 × 512 × 7168 |  | decode | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G406 | 96 × 1024 × 7168 |  | decode | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G407 | 96 × 2048 × 7168 |  | decode | 4 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G408 | 192 × 512 × 4096 |  | decode | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | indexer_compressor_wkv_gate | 待验证 | — |
| G409 | 192 × 1024 × 4096 |  | decode | 1 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | compressor_wkv_gate | 待验证 | — |
| G410 | 192 × 2048 × 4096 |  | decode | 2 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | compressor_wkv_gate | 待验证 | — |
| G411 | 288 × 512 × 4096 |  | decode | 6 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G412 | 288 × 1024 × 4096 |  | decode | 3 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G413 | 288 × 2048 × 4096 |  | decode | 6 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G414 | 2560 × 512 × 7168 |  | prefill | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G415 | 2560 × 1024 × 7168 |  | prefill | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G416 | 2560 × 2048 × 7168 |  | prefill | 8 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G417 | 3584 × 512 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G418 | 3584 × 1024 × 4096 |  | prefill | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G419 | 3584 × 2048 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G420 | 4608 × 512 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G421 | 4608 × 1024 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G422 | 4608 × 2048 × 7168 |  | prefill | 2 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G423 | 5120 × 512 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G424 | 5120 × 1024 × 4096 |  | prefill | 4 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G425 | 5120 × 2048 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G426 | 6656 × 512 × 4096 |  | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G427 | 6656 × 512 × 7168 |  | prefill | 10 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | indexer_compressor_wkv_gate | 待验证 | — |
| G428 | 6656 × 1024 × 4096 |  | prefill | 8 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G429 | 6656 × 1024 × 7168 |  | prefill | 10 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G430 | 6656 × 2048 × 4096 |  | prefill | 16 | — | — | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | compressor_wkv_gate | 待验证 | — |
| G431 | 6656 × 2048 × 7168 |  | prefill | 10 | — | — | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | compressor_wkv_gate | 待验证 | — |

<a id="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求，累计 4 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G432 | 4 × 7168 × 8192 |  | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | eagle3_o_proj | 待验证 | — |
| G433 | 4 × 7168 × 12288 |  | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | eagle3_down_proj | 待验证 | — |
| G434 | 4 × 24576 × 7168 |  | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | eagle3_gate_up_proj | 待验证 | — |
| G435 | 4 × 24576 × 14336 |  | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | eagle3_qkv_proj | 待验证 | — |

<a id="table-7"></a>

## 表 7：moe_grouped_gemm · FP8E4M3 × FP8E4M3 → BF16

36 行需求，累计 70 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G436 | 3 × 1024 × 2048 | 221 | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_gate_up | 待验证 | — |
| G437 | 3 × 2048 × 512 | 221 | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_down | 待验证 | — |
| G438 | 3 × 3072 × 4096 | 11 | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G439 | 3 × 4096 × 1536 | 11 | decode | 1 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G440 | 4 × 2048 × 4096 | 63 | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G441 | 4 × 4096 × 1024 | 63 | decode | 1 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G442 | 4 × 4096 × 6144 | 15 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G443 | 4 × 6144 × 2048 | 15 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G444 | 6 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G445 | 6 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_down | 待验证 | — |
| G446 | 11 × 3072 × 4096 | 12 | decode | 1 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G447 | 11 × 4096 × 1536 | 12 | decode | 1 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G448 | 12 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G449 | 12 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G450 | 15 × 2048 × 4096 | 64 | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G451 | 15 × 4096 × 1024 | 64 | decode | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G452 | 24 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G453 | 24 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_down | 待验证 | — |
| G454 | 80 × 1024 × 2048 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_gate_up | 待验证 | — |
| G455 | 80 × 2048 × 512 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_down | 待验证 | — |
| G456 | 171 × 3072 × 4096 | 24 | prefill | 2 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G457 | 171 × 4096 × 1536 | 24 | prefill | 2 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G458 | 208 × 1024 × 2048 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_gate_up | 待验证 | — |
| G459 | 208 × 2048 × 512 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | Qwen3.5-35B-A3B/FP8[4k/1k/s150] | moe_down | 待验证 | — |
| G460 | 560 × 2048 × 4096 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G461 | 560 × 4096 × 1024 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G462 | 640 × 4096 × 6144 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G463 | 640 × 6144 × 2048 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[64k/1k/s2000] | moe_down | 待验证 | — |
| G464 | 1040 × 2048 × 4096 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G465 | 1040 × 4096 × 1024 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | Qwen3.5-397B-A17B/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G466 | 1152 × 4096 × 6144 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G467 | 1152 × 6144 × 2048 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G468 | 1664 × 4096 × 6144 | 32 | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G469 | 1664 × 6144 × 2048 | 32 | prefill | 4 | FP32@1×128 | 离线 1×128 | GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000] | moe_down | 待验证 | — |
| G470 | 2219 × 3072 × 4096 | 24 | prefill | 2 | FP32@1×4096 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G471 | 2219 × 4096 × 1536 | 24 | prefill | 2 | FP32@1×1536 | 离线 1×128 | Hunyuan-V3/FP8[128k/1k/s2000] | moe_down | 待验证 | — |

<a id="table-8"></a>

## 表 8：moe_grouped_gemm · FP8E4M3 × FP4E2M1 → BF16

34 行需求，累计 304 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G472 | 2 × 1024 × 2048 | 80 | decode | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_gate_up | 待验证 | — |
| G473 | 2 × 2048 × 512 | 80 | decode | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_down | 待验证 | — |
| G474 | 20 × 6144 × 7168 | 24 | decode | 6 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G475 | 20 × 7168 × 3072 | 24 | decode | 6 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G476 | 24 × 6144 × 7168 | 24 | decode | 8 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G477 | 24 × 7168 × 3072 | 24 | decode | 8 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G478 | 30 × 4096 × 2048 | 32 | decode | 3 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | moe_down | 待验证 | — |
| G479 | 30 × 4096 × 4096 | 32 | decode | 3 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | moe_gate_up | 待验证 | — |
| G480 | 36 × 4096 × 2048 | 32 | decode | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | moe_down | 待验证 | — |
| G481 | 36 × 4096 × 4096 | 32 | decode | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150] | moe_gate_up | 待验证 | — |
| G482 | 45 × 4096 × 2048 | 32 | decode | 9 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_down | 待验证 | — |
| G483 | 45 × 4096 × 4096 | 32 | decode | 9 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G484 | 54 × 4096 × 2048 | 32 | decode | 12 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_down | 待验证 | — |
| G485 | 54 × 4096 × 4096 | 32 | decode | 12 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G486 | 64 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_gate_up | 待验证 | — |
| G487 | 64 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_down | 待验证 | — |
| G488 | 160 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G489 | 160 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G490 | 208 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_gate_up | 待验证 | — |
| G491 | 208 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | Qwen3.5-35B-A3B/MXFP4[128k/1k/s150] | moe_down | 待验证 | — |
| G492 | 320 × 6144 × 7168 | 48 | prefill | 12 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G493 | 320 × 7168 × 3072 | 48 | prefill | 12 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G494 | 416 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G495 | 416 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G496 | 576 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G497 | 576 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G498 | 672 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | moe_down | 待验证 | — |
| G499 | 672 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G500 | 832 × 6144 × 7168 | 48 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G501 | 832 × 7168 × 3072 | 48 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Pro/FP8[128k/1k/s2000] | moe_down | 待验证 | — |
| G502 | 960 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | moe_down | 待验证 | — |
| G503 | 960 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000] | moe_gate_up | 待验证 | — |
| G504 | 1248 × 4096 × 2048 | 32 | prefill | 32 | FP32@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_down | 待验证 | — |
| G505 | 1248 × 4096 × 4096 | 32 | prefill | 32 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000] | moe_gate_up | 待验证 | — |

<a id="table-9"></a>

## 表 9：moe_grouped_gemm · BF16 × INT4 → BF16

6 行需求，累计 10 条实例。编号在本清单中全局唯一。

| 编号 | M × N × K | batch / experts | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | op_name | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G506 | 6 × 4096 × 7168 | 24 | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G507 | 6 × 7168 × 2048 | 24 | decode | 1 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_down | 待验证 | — |
| G508 | 854 × 4096 × 7168 | 48 | prefill | 2 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G509 | 854 × 7168 × 2048 | 48 | prefill | 2 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_down | 待验证 | — |
| G510 | 1110 × 4096 × 7168 | 48 | prefill | 2 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_gate_up | 待验证 | — |
| G511 | 1110 × 7168 × 2048 | 48 | prefill | 2 | — | 离线 per-group | Kimi-2.6/W4A16[128k/1k/s2000] | moe_down | 待验证 | — |
