# MLA 系列算子说明

`boltops.mla` 公开 Triton **dense MLA / GQA decode** 注意力（布局对齐 vLLM
`TRITON_MLA`）。实现位于 `boltops/mla/triton/`；维护人 Triton @xukang。

同时提供与 vLLM HCU 调用方式兼容的 Sparse FlashMLA 前向接口。

## 当前支持一览

| 公开 API | 场景 | 来源 | 备注 |
| --- | --- | --- | --- |
| `decode_attention_fwd` | MHA / GQA / MLA paged decode | vLLM | MLA 时传 `is_mla=True` |
| `decode_attention_fwd_grouped` | GQA/MQA/MLA 路径 | vLLM | 底层 grouped stage1 |
| `decode_attention_fwd_normal` | MHA 路径 | vLLM | 底层 normal stage1 |
| `compute_num_kv_splits` | split-KV 数量启发式 | vLLM `triton_mla` | 与 workspace 预留一致 |
| `flash_mla_sparse_fwd` | Sparse MLA prefill/decode | FlashMLA | 按 top-k token 索引 gather |

```python
from boltops.mla import (
    decode_attention_fwd,
    compute_num_kv_splits,
    flash_mla_sparse_fwd,
)
```

测试：

```bash
pytest -q test/mla/
pytest -q test/mla/ --bench -s
python scripts/benchmarks/triton/benchmark_flash_mla_sparse.py -h
```

精度测试位于 `test/mla/test_flash_mla_sparse.py`；独立性能脚本会运行典型
`topk=2048/4096` shape，并输出包含延迟和有效 TFLOPS 的 CSV。

### Sparse FlashMLA benchmark 与 autotune

测试环境：Triton 3.6.0、HIP 6.3、BW200 UBB BW1000（gfx936）、BF16。
下表为 `100 ms warmup + 500 ms repeat` 的 median；baseline 使用迁入时的
默认配置，tuned 使用随包发布的 size-specific config。

| S_q | S_kv | H_q/H_kv | top-k | baseline (ms) | tuned (ms) | speedup | tuned TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16384 | 16/1 | 2048 | 0.68464 | 0.47680 | 1.436x | 0.15 |
| 4 | 16384 | 16/1 | 2048 | 0.68432 | 0.47664 | 1.436x | 0.60 |
| 32 | 16384 | 16/1 | 2048 | 0.68560 | 0.48208 | 1.422x | 4.73 |
| 4 | 32768 | 16/1 | 4096 | 1.34880 | 0.93360 | 1.445x | 0.61 |

每个 size 的完整调优包含 108 个结构候选、48 个编译/launch 候选和 16 个
调度优化组合，调节
`BLOCK_N`、`BLOCK_H`、`BLOCK_DMODEL`、`num_warps`、`num_stages`、
`waves_per_eu`、`kpack`、max-ILP、MachineScheduler register-pressure、
optimized epilogue 和 BlockPingpong。运行方式：

```bash
python boltops/tools/tune_flash_mla_sparse.py \
  --output /tmp/flash_mla_sparse.json
```

当前四个典型 size 的调度优化最优项一致为 max-ILP off、
register-pressure-aware scheduling on、optimized epilogue off、BlockPingpong off。

配置按 Triton major/minor、gfx arch、完整设备名定位文件，再按 dtype 和完整
shape 精确匹配；未命中配置时自动使用安全默认值。当前配置文件是
`boltops/mla/triton/configs/triton_3_6/flash_mla_sparse_gfx936_BW200_UBB_BW1000.json`。

溯源见 `boltops/mla/triton/SOURCE.md`。

---

## `decode_attention_fwd`

vLLM Triton paged decode attention（split-KV stage1 + online-softmax merge stage2）。

对 MLA absorb-decode（vLLM `TritonMLAImpl.forward_mqa`）：

- `q`：`[B, H_q, Lk]`，DeepSeek-V3 典型 `Lk=576`（512 latent + 64 RoPE）
- `k_buffer`：`[num_pages, PAGE_SIZE, 1, Lk]`（拼接的 `kv_c | k_pe`）
- `v_buffer`：`[num_pages, PAGE_SIZE, 1, Lv]`，通常为 `k_buffer[..., :kv_lora_rank]`
- `is_mla=True`：按 latent / RoPE 切开 QK，并用 `tl.trans(k)` 作为 V（不再二次 load）

```python
num_kv_splits = compute_num_kv_splits(max_seq_len)
attn_logits = torch.empty(
    B, H_q, num_kv_splits, Lv + 1, dtype=torch.float32, device=q.device
)
o = torch.empty(B, H_q, Lv, dtype=q.dtype, device=q.device)
lse = torch.empty(B, H_q, dtype=q.dtype, device=q.device)

decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    block_table,       # [B, max_pages] int32
    seq_lens,          # [B] int32
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=PAGE_SIZE,
    is_mla=True,
)
```

### 参数

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `q` | `[B, H_q, Lk]` | FP16/BF16 | query（MLA 为 absorb 后） |
| `k_buffer` | `[..., PAGE_SIZE, H_kv, Lk]` | FP16/BF16/FP8 | paged K |
| `v_buffer` | `[..., PAGE_SIZE, H_kv, Lv]` | 同 K | paged V；MLA 常为 latent 视图 |
| `o` | `[B, H_q, Lv]` | 同 `q` | 预分配输出 |
| `lse` | `[B, H_q]` | 同 `q` | 预分配 log-sum-exp |
| `req_to_token` | `[B, max_pages]` | int32/int64 | block table（page 索引） |
| `b_seq_len` | `[B]` | int32/int64 | 每条序列长度 |
| `attn_logits` | `[B, H_q, splits, Lv+1]` | float32 | split 中间结果 + LSE |
| `num_kv_splits` | scalar | int | 须等于 `attn_logits.shape[2]` |
| `sm_scale` | scalar | float | softmax scale |
| `page_size` | scalar | int | 每页 token 数 |
| `logit_cap` | scalar | float | `0` 关闭 soft-cap |
| `k_scale` / `v_scale` | scalar tensor | float32 | FP8 dequant；默认 `1.0` |
| `is_mla` | bool | | MLA latent+RoPE 路径 |

算子为 **in-place 写 `o` / `lse`**，无返回值。Forward-only。

HIP（DCU / ROCm）上自动启用 `waves_per_eu` / `kpack` 等 AMD launch 参数；
MLA 的 `Lk=576` CUDA 专用 `512+64` tile 在 HIP 上走 `next_power_of_2(Lv)` 分支
（与上游 vLLM 一致）。

实现：`triton/decode_attention.py`。

---

## `compute_num_kv_splits`

```python
splits = compute_num_kv_splits(max_seq_len)           # 使用当前设备 SM 数
splits = compute_num_kv_splits(max_seq_len, sm_count=64)
```

公式（与 vLLM 相同）：

```text
ideal = next_power_of_2(max(1, max_seq_len // 512))
splits = min(ideal, sm_count * 2)
```

实现：`triton/num_kv_splits.py`。
