# Sparse-GQA Triton Autotune 使用说明

## 作用

该工具为 Qwen Sparse-GQA 的两个 HCU Triton kernel 搜索配置：

- `qwen_sparse_kv_extraction_compact_hcu_fa_triton`
- `sparse_gqa_packed_decode_triton`

调优、pytest 性能测试和独立 benchmark 共用同一组生产模拟 case、输入生成逻辑和 PyTorch reference。

## 安装与入口

在 BoltOPs 仓库根目录执行：

```bash
pip install -e ".[tune]"
```

安装后使用统一入口：

```bash
boltops-sparse-gqa-tune-cli --help
```

也可以从源码直接运行：

```bash
python -m boltops.tools.sparse_gqa_triton_tune.tune_sparse_gqa --help
```

## 生产模拟 case

默认覆盖 TP4、TP8 的 D1、D8、D32 六个 case：

| Case | Batch | Context | TP8 Hq | TP4 Hq | Hkv | Head dim |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 4096 | 3 | 6 | 1 | 256 |
| D8 | 8 | 32768 | 3 | 6 | 1 | 256 |
| D32 | 32 | 32768 | 3 | 6 | 1 | 256 |

其余固定参数为 `token_topk=2048`、`topk width=2051`、`page_size=64`。默认使用 FP8 paged cache、BF16 packed K/V，以及随机分散 QSA block 和物理 page 的生产模拟访问模式。

## 常用命令

调优全部六个 case：

```bash
boltops-sparse-gqa-tune-cli
```

只调 TP8/D8：

```bash
boltops-sparse-gqa-tune-cli --tp 8 --case D8
```

选择多个 TP 或 case 时重复传参：

```bash
boltops-sparse-gqa-tune-cli \
  --tp 4 --tp 8 \
  --case D1 --case D8
```

仅用 baseline 配置完成快速正确性和写入流程验证：

```bash
boltops-sparse-gqa-tune-cli \
  --tp 8 --case D1 \
  --quick --warmup-ms 5 --rep-ms 15 \
  --output-dir /tmp/sparse_gqa_configs
```

`--warmup-ms` 和 `--rep-ms` 的单位都是毫秒，不是迭代次数。

## 正确性保证

每个候选配置在计时前都会与共享 PyTorch oracle 比较：

- compact 对 K/V 结果进行精确比较；
- decode 使用 FP32 score、softmax 和累加 reference；
- 未通过正确性检查或无法编译的候选不会参与最优配置选择；
- 所有候选均失败时，调优任务直接报错，不生成错误配置。

## 配置输出

默认写入：

```text
boltops/fa/sparse_gqa/triton/configs/triton_<capability>/
```

当前 Triton 3.6 对应 `triton_3_6/`，运行时找不到配置时会继续尝试兼容的
`triton_3_2/`。

compact 和 decode 分别为每个 gfx arch 和每卡 CU 数生成一个文件：

```text
qwen_sparse_kv_compact-gfx938-cu64.json
sparse_gqa_packed_decode-gfx938-cu64.json
```

CU 数通过当前设备的 `multi_processor_count` 自动获取。同为 `gfx938`、但
CU 数不同的设备会写入不同文件，不会覆盖彼此的同 shape 配置。运行时优先加载
精确的 `arch+CU` 文件；找不到时会回退到旧版 `*-gfx938.json` 文件。

完整 shape 和 dtype 保存在 JSON 的 `match` 字段中；设备产品名、完整
`gcnArchName`、测试延迟和 seed 不会进入 JSON 的 shape 匹配项。D1、D8、D32
作为不同的精确匹配项聚合在同一文件中，例如：

```json
{
  "configs": [
    {
      "match": {
        "batch": 8,
        "topk": 2051,
        "hq": 3,
        "hkv": 1,
        "head_dim": 256,
        "dtype": "bfloat16"
      },
      "config": {
        "BLOCK_N": 32,
        "num_stages": 3,
        "num_warps": 4
      }
    }
  ]
}
```

详细延迟、设备、Triton 版本、seed 和访问模式由独立 benchmark CSV 保存。可通过
`--output-dir` 将试运行配置写到其他目录，确认后再纳入正式 capability 目录。
