# Gemm 系列算子说明

`boltops.gemm` 公开 Triton GEMM 族（布局对齐 vLLM / aiter / SGLang 调用侧）。
实现位于 `boltops/gemm/triton/`；维护人 Triton @xukang。

## 当前支持一览

| 公开 API | 精度 / 量化 | 布局要点 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| `matmul_persistent` | FP16 / BF16 / FP32 | `C = A @ B`（NN） | vLLM | LDS≤64KiB 自动降 tile |
| `gemm_a16w16` | FP16 / BF16 | `Y = X @ W^T`，`W[N,K]` | aiter | 可选 epilogue activation |
| `w8a8_triton_block_scaled_mm` | FP8 / INT8 block-scale | `B[N,K]`，`block_size=[bn,bk]` | vLLM | |
| `triton_scaled_mm` | INT8 / FP8 + scale | `C = (A @ B) * sa * sb`（NN） | vLLM | per-tensor / per-channel scale |
| `awq_gemm_triton` | W4A16（AWQ packed） | int4 `qweight` | vLLM | HCU JSON：`K100_AI` / `BW200` |
| `gemm_a8w8_blockscale` | A8W8 block-scale | `Y = X @ W^T` | aiter | |
| `gemm_afp4wfp4` | MXFP4 | `Y = X @ W^T` | aiter | 需 `is_fp4_avail()` |
| `batched_gemm_a16wfp4` | A16WFP4 batched | `Y[i] = X[i] @ W[i]^T` | aiter | 需 FP4 |
| `batched_gemm_bf16` | BF16 / FP16 batched | `Y[i] = X[i] @ W[i]^T`，`W[B,N,K]` | aiter | HCU JSON：`K100_AI` / `BW200` |
| `w8a8_block_int8_matmul` | INT8 block-scale | `B[N,K]` | SGLang | HCU JSON：`BW200` / `gfx938` |

未公开：`gemm_fp16`（内部 HCU WASP/WDRA 实验）。

```python
from boltops.gemm import (
    matmul_persistent,
    gemm_a16w16,
    w8a8_triton_block_scaled_mm,
    triton_scaled_mm,
    awq_gemm_triton,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    batched_gemm_a16wfp4,
    batched_gemm_bf16,
    w8a8_block_int8_matmul,
)
# 或 from boltops.gemm import ...
```

测试统一目录：

```bash
pytest -q test/gemm/
pytest -q test/gemm/ --bench -s   # 含 bench 的用例
```

溯源总表见 `boltops/gemm/triton/SOURCE.md`。

---

## `matmul_persistent`

vLLM Triton persistent GEMM：`C = A @ B [+ bias]`（NN）。

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `a` | `[M, K]` | FP16/BF16/FP32 | 左矩阵 |
| `b` | `[K, N]` | 同 `a` | 右矩阵 |
| `bias` | `[N]` | 同 `a` | 可选 1D |

Launch 按 `utility.arch_info.get_lds_capacity_bytes()`：LDS ≤ 64KiB 用小 tile，
否则用 vLLM 默认（FP16 `128/256/64`、`num_stages=3`）。

实现：`triton/matmul_persistent.py`。溯源：vLLM `batch_invariant.py`。

---

## `gemm_a16w16`

aiter Triton A16W16：`Y = X @ W^T [+ bias]`（`W` 为 `[N, K]`）。

```python
y = gemm_a16w16(x, weight, bias=None, dtype=None, config=None, activation=None)
```

可选 epilogue activation（`gelu` / `silu` / `relu` 等）。暂无 HCU JSON；
默认见 `builtin_gemm_configs.py`。

实现：`triton/gemm_a16w16.py` + `_op.py`。溯源：ROCm/aiter `gemm_a16w16`。

---

## `w8a8_triton_block_scaled_mm`

vLLM Triton W8A8 / FP8 **block-scale**：`B` 布局 `[N, K]`，`block_size=[bn, bk]`。

```python
C = w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128], output_dtype=torch.float16)
```

实现：`triton/w8a8_triton_block_scaled_mm.py`。溯源：vLLM `fp8_utils.py`。

---

## `triton_scaled_mm`

vLLM compressed-tensors scaled matmul：`C = (A @ B) * scale_a * scale_b [+ bias]`。

- `a`：`[M, K]`；`b`：`[K, N]`（NN）
- `scale_a`：scalar 或 `[M, 1]`；`scale_b`：scalar 或 `[N, 1]`

实现：`triton/triton_scaled_mm.py`。溯源：vLLM `triton_scaled_mm.py`。

---

## `awq_gemm_triton`

vLLM Triton AWQ（int4 packed weights）。

```python
out = awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters, config=None)
```

- `qweight`：`[K, N//8]` int32；`split_k_iters`：2 的幂，≤ 32
- HCU JSON：`triton/configs/awq_gemm_triton/`（`K100_AI` / `BW200`）

实现：`triton/awq_gemm_triton.py`。溯源：vLLM `awq_triton.py`。

---

## `gemm_a8w8_blockscale`

aiter Triton A8W8 block-scale：`Y = X @ W^T`。

```python
y = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)
```

`GROUP_K`（由 scale 推导）须等于 launch `BLOCK_SIZE_K`。默认见
`builtin_gemm_configs.py`。

实现：`triton/gemm_a8w8_blockscale.py` + `_op.py`。溯源：aiter `gemm_a8w8_blockscale`。

---

## `gemm_afp4wfp4`

aiter Triton MXFP4：`Y = X @ W^T`。需 FP4 硬件（`utility.arch_info.is_fp4_avail()`）。

```python
y = gemm_afp4wfp4(x, w, x_scales, w_scales, dtype=torch.bfloat16)
```

- `x` / `w`：`[M, K//2]` / `[N, K//2]`；scales：`[*, K//32]` E8M0

实现：`triton/gemm_afp4wfp4.py` + `_op.py`。溯源：aiter `gemm_afp4wfp4`。

---

## `batched_gemm_a16wfp4`

aiter Triton batched A16WFP4：`Y[i] = X[i] @ W[i]^T`。需 FP4 硬件。

```python
y = batched_gemm_a16wfp4(x, w, w_scales, dtype=torch.bfloat16)
```

- `x`：`[B, M, K]` BF16/FP16；`w`：`[B, N, K//2]` FP4

实现：`triton/batched_gemm_a16wfp4.py` + `_op.py`。溯源：aiter batched path。

---

## `batched_gemm_bf16`

aiter Triton batched BF16/FP16 GEMM：`Y[i] = X[i] @ W[i]^T`。

```python
y = batched_gemm_bf16(x, w, bias=None, dtype=torch.bfloat16, y=None, config=None)
```

- `x`：`[B, M, K]`；`w`：`[B, N, K]`（内部转置）；可选 `bias`：`[B, 1, N]`
- `dtype`：`bfloat16` / `float16`；`splitK` 必须为 `None`
- HCU JSON（可选）：`triton/configs/batched_gemm_bf16/`  
  - batch 专用：`BATCHED_BF16-B={B}-N={N}-K={K}_{bf16|fp16}_{device}.json`
  - shape 通用：`BATCHED_BF16-N={N}-K={K}_{bf16|fp16}_{device}.json`
  - 设备默认：`BATCHED_BF16-DEFAULT_{bf16|fp16}_{device}.json`
  - 按上述顺序回退，表内选择最接近的 `M`
- 无 JSON 时静默使用内置 large/small tile（`M+N >= 4096` → large）

### BW200 FP16 性能记录

#### BlockPingpong / optimized epilogue 四组合 autotune（2026-08-26）

更新至 Triton `release/3.6.x` 提交 `80d21d979`（llvm-triton
`ddb5c4b4b`）后，对 `BlockPingpong` 开/关和 optimized epilogue 开/关四种组合
重新 autotune。候选初筛使用 warmup 10 ms、repeat 30 ms；四套 winner 再统一使用
warmup 20 ms、repeat 100 ms 复测，表中 latency 为复测中位数。

| 环境项 | 值 |
| --- | --- |
| GPU | BW200, UBB BW1000（gfx936），64 GiB |
| CU 数量 | 80 CU |
| SCLK | DPM 300–1500 MHz；调优后采样活动档为 1350 MHz，未锁频 |
| PyTorch / HIP | PyTorch 2.10.0 / HIP 6.3.26113 |
| Triton | 3.6.0 (`80d21d979`)；llvm-triton `ddb5c4b4b` |
| 输入 / 输出 dtype | FP16 / FP16 |

| Shape `[B,M,N,K]` | 最优配置 `[BM,BN,BK,GROUP,warps,stages,waves,PP,epi]` | latency | TFLOPS |
| --- | --- | ---: | ---: |
| `[56,234,4928,1232]` | `[256,128,32,4,8,2,2,true,false]` | 1.480240 ms | 107.49 |
| `[56,234,1232,2464]` | `[256,256,16,4,8,3,2,true,true]` | 0.726720 ms | 109.48 |

此外，`B=16` 的三组 AITER size（`N/K=1024/4096`、`8192/1024`、
`1280/8192`）已对全部 15 个 M 点重新调优。47 条专用配置的四组合胜出分布为：
PP关/epi关 21 条、PP关/epi开 9 条、PP开/epi关 15 条、PP开/epi开 2 条。
TFLOPS 按 `2*B*M*N*K / latency` 计算。

#### xprof 历史记录（2026-08-24）

2026-08-24 使用 GPU 2、`xprof --enable-sqtt` 采集。每个 shape 单独运行，
kernel 过滤器为 `batched_gemm`，计数器覆盖 `speed_of_light`、memory、occupancy、
instruction、compute、scheduler、wave、cache、UMC 和 UTCL2 等全部 11 个 section。

| Shape `[B,M,N,K]` | 专用 launch 配置 `[BM,BN,BK,GROUP,warps,stages,waves]` | 原始 dispatch | PMC replay 中位数 | replay TFLOPS |
| --- | --- | ---: | ---: | ---: |
| `[56,234,4928,1232]` | `[256,128,32,4,4,2,2]` | 2.001 ms | 1.735 ms | 91.73 |
| `[56,234,1232,2464]` | `[256,256,64,4,8,1,1]` | 0.914 ms | 0.703 ms | 113.14 |

TFLOPS 按 `2*B*M*N*K / latency` 计算。以上数据来自 xprof PMC replay，适合分析
单 kernel 的硬件利用率；由于计数器 replay/SQTT 会扰动执行时间，不作为无 profiler
端到端 benchmark 的性能门槛。

实现：`triton/batched_gemm_bf16.py`。溯源：aiter `batched_gemm_bf16`。

---

## `w8a8_block_int8_matmul`

SGLang Triton INT8 block-scale GEMM。

```python
C = w8a8_block_int8_matmul(A, B, As, Bs, [128, 128], output_dtype=torch.float16)
```

HCU JSON：`triton/configs/w8a8_block_int8_matmul/`（`BW200` / `gfx938`，来自 hygon aiter
`block_w8a8`）。

实现：`triton/w8a8_block_int8_matmul.py`。溯源：SGLang `int8_kernel.py`。
