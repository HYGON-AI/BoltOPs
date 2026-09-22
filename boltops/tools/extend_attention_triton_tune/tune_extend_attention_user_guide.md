# ExtendAttention Triton 调优指南

调优工具与运行时共用配置 key 和文件选择逻辑，结果写入当前 Triton 能力目录，
例如 `boltops/fa/extend_attention/triton/configs/triton_3_6/`。
FP8 KV cache 调优只支持 gfx938/gfx92a；gfx936 使用 `both` 时只调优 BF16，
显式指定 `fp8` 会报错。

## 环境

```bash
export TRITON_HIP_CLANG_PATH=/root/dtk_llvm_ffe19a/aillvm/bin/clang
export AMDGPU_USE_BUFFER_OPS=1
export PYTHONPATH=.
```

## 常用命令

列出所有 case：

```bash
boltops-extend-attention-tune-cli --list-cases
```

先使用默认配置做单 case 冒烟：

```bash
boltops-extend-attention-tune-cli \
  --case short-kv1 --kv-cache-dtype bf16 --quick \
  --output-dir /tmp/extend_attention_quick
```

调优一个 kernel family：

```bash
boltops-extend-attention-tune-cli \
  --kind short_query --kv-cache-dtype both
```

完整调优：

```bash
boltops/tools/extend_attention_triton_tune/tune_example.sh
```

## 配置组织

配置按以下维度分文件：

- `short_query` 或 `long_query`；
- GPU 架构和 CU 数；
- Q dtype；
- KV cache dtype。

每个 case 完成后会原子更新对应 JSON。BF16 与 FP8 使用不同搜索空间和配置文件，
不会相互覆盖。提交前应运行 ExtendAttention correctness、配置命中和 package smoke
测试，并确认 JSON 只包含目标设备生成的最终配置。调优日志、临时目录、CSV 和
`__pycache__` 不属于发布内容，不应提交。
