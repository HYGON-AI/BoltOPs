# optest 性能基准（perf）

本文档是性能基准的唯一文档入口，只覆盖 optest `per` 入口的性能测试框架。
正确性测试在 `test/`（各算子文档有说明），调参脚本在 `scripts/tune/`，
均不属于本框架。

设计原则：**测试文件不侵入，脚本只收集和格式化**。

- 算子 owner 在 `test/` 下正常写 bench 单测（`@pytest.mark.bench`，
  自己构造输入、**计时统一用 `time_ms` fixture**（§3.1）；建议加
  一行 print 输出 case 摘要——**print 不参与收集**，CSV 只来自
  计时调用与 parametrize），**不需要 import / 调用任何 perf
  API**，不需要注册。
- optest `per` 入口（`scripts/run_operator_benchmarks.py`）自动
  发现 `test/` 下的 bench 文件（§8）驱动 pytest 跑这些 bench；运行期间 `$OPTEST_BENCH_RAW_DIR` 被设置，`test/conftest.py`
  的收集层会**观测** bench 测试里的每次计时调用（统一入口
  `time_ms` fixture；存量直调 `do_bench(_cudagraph)` 的测例同点
  观测，§3.1），每次调用产出一行 raw 结果；入口按 op 归组、
  格式化为统一 CSV。
- 测数值就是测试自己测的数值（测试用什么计时方式，CSV 里就是什么
  口径），脚本不改口径、不重复测量。

## 1. 目录结构

```text
boltops/utility/bench.py         # 计时 / 结果行契约（emit_row / read_rows）、
                                 # op 名自动派生 / bench 文件发现（入口与
                                 # conftest 共享，测试可不依赖）
boltops/utility/backends.py      # backend 子包发现（discover_backends，
                                 # 双 backend 测试共用，§6）
test/conftest.py                 # --bench 选项/标记 + perf 收集层
                                 # （$OPTEST_BENCH_RAW_DIR 生效时观测计时调用）
scripts/
└── run_operator_benchmarks.py   # optest per 入口（路径被 optest source marker
                                 #   固定，勿移动）：自动发现 bench、跑、收集 raw
                                 #   行、按 op 格式化 CSV
test/                            # 各算子的 bench 单测（perf 测点就在这里）
```

新增一个 perf 测点 = 写一个正常 bench 单测（入口自动发现，§8）。
测试文件里没有任何 perf 专属代码（backend / op 名声明除外，§8）。

## 2. 运行

正式入口是 optest CLI（本地与 CI 同一套）：

```bash
# 只跑性能（CI performance:optest job 同条命令，CIsmi 会锁定 HIP_VISIBLE_DEVICES）
optest --op_name boltops --test_dir . --test_mode per --gpus "$HIP_VISIBLE_DEVICES"

# 正确性 + 性能（本地全量跑；正确性套件存在存量失败，
# 目前未作为 CI 门禁，修复后可将 CI 升级为 --test_mode all）
optest --op_name boltops --test_dir . --test_mode all
```

optest 的 `per` 模式内部即调用 `scripts/run_operator_benchmarks.py`
（并设置 `$OPTEST_OUTPUT_DIR`）。入口把 CSV 写到源码树内的
`<repo>/optest_results/`；optest 在运行前后对源码树内的 `*.csv` 做
快照 diff，把本次新增/变更的 CSV 复制到
`/op_test_output/boltops/<branch>_<commit>_<ts>/`（即报告的
`csv=N`），其余结果汇入同目录的 `.optest-results.json`，CI 用
`ci_script/optest_to_junit.py` 转成 junit。

本地快速跑可直接调用下层入口（支持选子集与指定输出目录）：

```bash
python scripts/run_operator_benchmarks.py              # 全部 op

# 只跑部分 op
python scripts/run_operator_benchmarks.py --ops gdn_chunk kda_decode

# 指定 CSV 输出目录（默认 <repo>/optest_results/；
# 在 optest 下运行时必须位于源码树内，否则 optest 收集不到）
python scripts/run_operator_benchmarks.py -o /tmp/bench_out
```

不带 `--ops` 跑自动发现的全部 op（§8）。

## 3. 收集机制（参数捕获 → raw → CSV）

整条链路：**入口起 pytest → conftest 观测计时调用并捕获参数 → 测试
通过后写 raw → 入口格式化 CSV**。每一步由谁执行：

```
scripts/run_operator_benchmarks.py（入口）
  │ 设置 $OPTEST_BENCH_RAW_DIR=<raw 目录>、
  │     $OPTEST_BACKEND=<本 pass 的 backend>
  ▼
pytest <自动发现的 bench 文件> --bench -m bench
  │ 每个 bench 标记测试，test/conftest.py 的 pytest_runtest_call
  │ hookwrapper 做四件事：
  │ ① 装观测：把 triton.testing / tilelang.profiler 的计时入口
  │    （do_bench / do_bench_cudagraph）模块属性临时替换为包装
  │    函数——time_ms fixture 内部走的就是这两个入口，统一用
  │    time_ms 的测试与存量直调测试都汇聚到同一观测点；tilelang
  │    的 from-import 形式另临时重绑测试模块的 globals；
  │    $OPTEST_BENCH_RAW_DIR 未设时整段跳过（本地裸跑 = 纯 print，
  │    见 §3.1）
  │ ② 测试照常执行：构造输入 → 断言 → 计时（代码、返回值不变）
  │ ③ 包装函数每返回一次 → 内存记一条记录（参数捕获，§3.2）
  │ ④ 测试结束 → flush 逐条追加 raw 行（§3.3）：通过 = 正常行；
  │    失败 = 同形状 error 行（latency 留空 / 补错误摘要）；
  │    skip = 不产行；测试结束恢复全部观测状态
  ▼
入口（pytest 退出后）：读 raw → 按 op 归并 → 格式化（§3.4）
  → <repo>/optest_results/<op>.csv → optest diff 复制到结果目录（§4）
```

### 3.1 计时入口：统一用 `time_ms` fixture

**新增 bench 测试一律在测试签名请求 `time_ms` fixture**（`def
test_bench_xxx(..., time_ms, bench_warmup, bench_repeat)`），调用
`ms = time_ms(run, bench_warmup, bench_repeat)`。fixture 内部走观测
层的 `do_bench_cudagraph` / `do_bench` 入口：**cudagraph 优先**，捕获
失败打警告回退 event 计时，观测层恰好记一行；非捕获错误（kernel 真
坏了）照抛，不会静默用回退路径计时。`warmup` / `repeat` 是毫秒预算
（与 do_bench 同约定）。可选关键字（见 §5）：`return_mode=` 选统计量（默认 `median`，可选
`min` / `max` / `mean`，两工具链同名同义）、`flops=` 输出 `tflops`
列、`dtype=` 输出 `dtype` 列、`use_graph=` 是否 CUDA Graph 计时
（默认 `true`；`false` = 纯 event 计时，不做捕获尝试，由 owner 按
被测调用自行选择）。fixture 按当前 backend 自动路由到对应工具
链 profiler（`triton.testing` / `tilelang.profiler`，两者同为毫秒
预算的 `do_bench(_cudagraph)` 契约），路由规则与行 `backend` 字段
一致：case 内 parametrize 的 `(backend, 实现)` 值 > 函数名 backend
名（`test_bench_triton_op` 这类，§6）> `$OPTEST_BACKEND` > 模块级
`PERF_BACKEND(S)` 显式声明 > `importorskip` 自动检测（§8）>
`triton`。测试代码无需 import 任何计时函数，实际用了哪种方式如实
记进 CSV `timing` 列。

直调计时函数是**例外，新测试不要再直调**（仅适用于现有
特殊场景：双 port 两侧显式 event 对比、工具链特有参数如
`quantiles` / `grad_to_none` 等，现状见 §8.1 `计时入口` 列）。此时观测是对**模块属性**的临时替换，
计时函数要写成 `triton.testing.do_bench(...)`、
`tilelang.profiler.do_bench_cudagraph(...)` 这类模块限定调用。
`from triton.testing import do_bench` 这种提前绑定的引用不会被拦截；
tilelang 侧对 `from tilelang.profiler import ...` 有 conftest 兼容
（测试运行期间临时重绑测试模块 globals，结束恢复，测试文件无需修
改），triton 侧无此兼容。

### 3.2 参数捕获：case 列从哪来

**测试不向任何 API 传参数。** 计时包装函数被触发时，conftest 直接读
pytest 自己的账本：

- pytest 为每个 parametrize case 实例化测试时，把该 case 的参数值存在
  `item.callspec.params`（参数名 → 值的 dict；`bench_warmup` 这类
  fixture 不在其中）；
- conftest 把这个 dict 交给 `extract_case_params`
  （boltops/utility/bench.py）展平：

| parametrize 值类型 | 处理 |
| --- | --- |
| 标量（str / int / float / bool） | 原样成列：`参数名 → 值` |
| dict | 其标量条目按键名展开 |
| dataclass | `dataclasses.asdict` 后取标量字段，字段名成列名 |
| tensor / tuple / list / None / 其他 | 跳过（无法可靠写成 CSV） |

- 结果 dict 的**插入顺序**决定 CSV 列顺序（同一 op 内按行首现序）；
  保留列冲突：case 参数名与框架列（`operator` / `backend` /
  `dtype` / `arch` / `version` / `timing` / `stat` / `kernels` /
  `latency_ms` / `tflops`）重名时自动改名为 `case_<原名>`（一次性警告），
  测试不会因列名冲突而失败。
- 所以一个值进 CSV 列的唯一条件：**它是 parametrize 的值（或 dict /
  dataclass parametrize 值里的标量字段）**。函数体里硬编码的 shape
  对观测层不可见——这就是"shape 尽量 parametrize 化"的原因。
- 同一测试内多次计时调用自动加 `call` 序号（1, 2, ...），第 n 行对应
  测试里第 n 次计时；只计时一次则不输出该列。
- 两个框架列按优先级推导：
  - `operator`：模块级 `PERF_OP` 常量 > 文件名去 `test_` /
    `generic_` 前缀（泛化名或重名自动加目录前缀，§8；多个文件写
    同一个 `PERF_OP` 即合并为一个 op）；
  - `backend`：`$OPTEST_BACKEND`（入口 pass
    维度，§6）> 模块级 `PERF_BACKEND(S)` > `importorskip` 检测（§8）
    > 默认 `triton`。

### 3.3 raw 行：格式与落盘时机

一行 raw = 一次计时调用（一次包装函数触发），JSON 一行。真实示例：

```json
{"operator": "gdn_chunk", "backend": "tilelang", "params": {"transpose_state": true, "use_varlen": false}, "dtype": null, "latency_ms": 0.39059069752693176, "flops": null, "timing": "cudagraph"}
{"operator": "gdn_chunk", "backend": "triton", "params": {"transpose_state": true, "use_varlen": false}, "dtype": null, "latency_ms": 0.8296505500530374, "flops": null, "timing": "cudagraph", "kernels": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"}
```

同一测试内多次计时会带 `call` 列（sparse_gqa，`call`=1/2 对应
compact / packed 两个计时点）：

```json
{"operator": "sparse_gqa", "backend": "triton", "params": {"name": "D1", "tp": 8, "batch": 1, "context_len": 4096, "call": 1}, "dtype": null, "latency_ms": 0.008141053253970753, "flops": null, "timing": "cudagraph"}
```

失败行的 raw 示例（`latency_ms` 为 `null`，带 `error`）：

```json
{"operator": "gdn_chunk", "backend": "triton", "params": {"transpose_state": true, "use_varlen": false}, "dtype": null, "latency_ms": null, "flops": null, "timing": "cudagraph", "error": "RuntimeError: kernel launch failed"}
```

（示例为节选：`params` 只保留前几个键。）

- 文件命名：`<raw_dir>/<测试文件自动派生 op>.raw.jsonl`（文件名去
  `test_` / `generic_` 前缀，泛化名 / 重名自动加目录前缀，§8）。
  **始终按测试文件命名**：`PERF_OP` 只改行内 `operator` 字段，不改
  文件名——入口按"发现的文件 → 文件名"找行（§8），这样多文件合并
  op 时行不会丢。
- 落盘时机：记录先在内存缓冲（每测试一份），**测试 call 阶段结束**
  时整批追加：
  - 通过 → 正常行；
  - 失败（断言 / 异常）→ 不丢数据，改为 error 行：已计时的行保留
    `latency_ms` 并在行尾记测试级错误摘要；计时调用本身抛错的行
    `latency_ms=null`、`timing` 记尝试的方式；全程没计时就抛错的
    测试合成一条 triage 行（params 同规则捕获）——该 op 仍有 CSV，
    失败在表里可见。`error` 形如 `<ExcType>: <首行>`（≤200 字符）；
  - skip → 不产行。
  pytest 记 failed 继续其余 case，入口按 pytest 退出码归因、整体
  退出码 1。
- 字段：`latency_ms` = 测试所调用计时函数的返回值（`time_ms` 默认
  median，`return_mode=` 可选 min / max / mean，与测试 print 的数值
  一致；直调计时函数可用 return_mode 自选）；`timing`
  记录计时方式（event / cudagraph），**进 CSV 的 `timing` 列**——
  它影响数值解读：cudagraph 回放不含 host 派发 / 分配开销，event
  计时含，两者不可直接混比（`time_ms(use_graph=False)` 可显式钉
  event 路径，§5）；`stat` 记录统计量（仅非默认 `median` 的行携带，
  §5），同为"数值怎么解读"列；`dtype` / `flops` 仅
  `time_ms(dtype=...)` / `time_ms(flops=...)` 可选参数（§5）可能
  携带；`kernels` = 该次计时窗口内捕获到的 kernel 名（见下）。

- `kernels` 捕获机制：采集层在**进程级一次性**给两个工具链的
  launch 汇聚点装 hook（triton 的 `JITFunction.run` / tilelang 的
  `JITKernel.__call__`），每次 kernel 发射记一个名字（triton =
  `@triton.jit` 函数名，tilelang = `global_symbol`）。计时调用时截取
  该调用窗口内的发射记录——**按首发顺序去重、`;` 连接**写入该行
  （cudagraph 计时下 kernel 在捕获阶段发射一次、回放不走 Python，
  照常捕获；失败行同样携带，便于 triage）。单 kernel 的 op 就是一
  个名字；多 kernel 的 op 如
  `_fwd_grouped_kernel_stage1;_fwd_kernel_stage2`（列表末位不是
  严格意义的"最后发射"，循环发射场景下去重保留的是首发现场）。
  非 triton / tilelang 发射（torch 原生算子等）不产生名字；整行无
  名时 `kernels` 为空（CSV 该列不出现）。测试文件零改动。

### 3.4 CSV 格式化：raw 行怎么变成 `<op>.csv`

pytest 跑完后入口（`scripts/run_operator_benchmarks.py`）四步：

1. `read_rows`：解析 `<raw_dir>/*.raw.jsonl` 全部行 → `{op: [行]}`
   （key 即 raw 文件名，= 测试文件自动派生 op，§8）；
2. `collect_op_rows`：把同一 op（自动发现，§8）的所有测试文件的行
   合并成一组（一个 op 可跨多文件；同一 op 跑多个 backend 时各 pass 的
   行也在这里合到一起，§6）；
3. `format_op_csv` 定列写行：
   - case 参数列 = 全部行 `params` 键的**并集**，按 op 内首现序
     （含 `call`）；某文件缺某参数时，该文件的行留空该列；
   - 固定列序：`operator`、`backend`、case 参数、`version`（该行
     自身工具链的 dist 版本，与 `pip show` 一致，自带该工具链的
     build commit，如 `3.6.0+git46766868.staging`；未知 backend
     留空）、`arch`、`timing`（仅当有行携带时出现）、`latency_ms`
     （ms，6 位小数；失败行留空）；
     `stat`（统计量，仅当有行非默认 `median` 时出现，默认行填
     `median`）、`kernels`（计时窗口内的 kernel 名，§3.3，仅当有
     行携带时出现）、`dtype` / `tflops` 列仅当至少一行携带该值时
     出现（§5 `time_ms` 可选参数）；
     末尾 `error` 列仅当至少一行携带错误时出现（失败行见 §3.3，
     通过行的该列为空）；
   - 每行 raw 对应 CSV 一行；`tflops`（若有）由 `flops ÷ 时延` 算出
     （失败行无时延，留空）；
4. 写 `<repo>/optest_results/<op>.csv`（optest 如何把它收进结果目录
   见 §4）。

## 4. 输出

- 每个 op 一个 CSV `<op>.csv`：先写到 `<repo>/optest_results/`
  （或 `-o` 指定目录），optest 对源码树内 `*.csv` 做运行前后 diff，
  把本次新增 / 变更的 CSV 复制到结果目录；最终布局
  `<result_dir>/<op>.csv`。列规则见 §3.4。
- raw 行保留在 `$OPTEST_OUTPUT_DIR/raw/<测试文件自动派生 op>.raw.jsonl`
  供审计（standalone 运行时落在 CSV 目录的 `raw/` 下）。
- 多 backend 的 op 所有 backend 的行合并进同一个 `<op>.csv`，
  `backend` 列区分（§6）。
- `tflops`（若有）= 估算 useful-FLOPS ÷ 时延，仅作跨 shape / 跨 run
  对比参考，不是 roofline 指标。FLOPS 公式在各自 bench 的 print 里，
  与对应测试 / 调参脚本保持同步。

## 5. 行的可选增强（`time_ms` 关键字参数，非必须）

`time_ms` 支持三个可选关键字（默认不带，只影响当次调用对应的那一行，
其余行与列布局不受影响）：

- `return_mode=`：统计量，默认 `median`，可选 `min` / `max` / `mean`
  （两工具链同名同义；triton 的 `all` 返回非标量，不暴露）。非默认值
  记入 raw 行 `stat` 字段，CSV 出现条件 `stat` 列（仅当有行非默认时，
  未指定的行填 `median`）——统计量不同的行不可直接混比，与 `timing`
  列同理。
- `flops=`：该行的 useful-FLOPS，CSV 出现条件 `tflops` 列
  （`flops ÷ 时延`，§4）。
- `dtype=`：该行计算精度，CSV 出现条件 `dtype` 列。
- `use_graph=`：是否 CUDA Graph 计时，默认 `true`（graph 优先、捕获
  失败自动回退 event，§3.1）；`false` = 纯 event 计时、不做捕获
  尝试，该行的 `timing` 列记 `event`。适用于被测调用含 host 侧工作
  （必然不可捕获）或要与其他 event 行直接对比的场景——由算子 owner
  自行决定，CSV 如实记录实际方式。

**不提供**逐行 case 列覆盖（旧 `bench_time` 的 `params=`，fixture 已
移除）：需要"每行不同 case 列"的循环计时点，优先把维度 parametrize
化（§3.2）或按 §6 每 backend 一个函数表达；需要工具链特有计时参数
（`quantiles` / `grad_to_none` 等）时走 §3.1 的直调例外。

## 6. 多 backend（一个 op，多个 backend）

约定：**一个 op 对应一份 bench 单测**，测试只面向 op 的统一接口
（不同 backend 对上层接口必须一致）；"支持哪些 backend" 从**测试文件
自动检测**（模块级 `pytest.importorskip("<toolchain>")`，§8；
`PERF_BACKEND(S)` 可显式覆盖）。

**默认首选：文件内 parametrize 双 backend**（`gdn_chunk` 模式）。
接口一致意味着 backend 就是天然的 case 维度：同一组 shape 在一个
pytest pass 内按 backend 顺序跑完，A/B 对比同窗口、同输入构造，行的
`backend` / `version` 列跟随 case，测试体零重复：

```python
# test/fla/test_gdn_chunk.py（节选；完整文件另有正确性测试与 API 对齐断言）
from boltops.utility.backends import discover_backends

CHUNK_H_BACKENDS = discover_backends(gdn, "chunk_gated_delta_rule_fwd_h")
# → {"triton": <fn>, "tilelang": <fn>}：遍历 boltops/<op>/ 的 backend 子包，
#   缺依赖的 backend 不出现

@pytest.fixture(
    params=[
        pytest.param((name, operator), id=name)
        for name, operator in CHUNK_H_BACKENDS.items()
    ],
)
def chunk_h_backend(request: pytest.FixtureRequest) -> tuple[str, Callable]:
    return request.param


@pytest.mark.bench
@pytest.mark.parametrize("use_varlen", [False, True], ids=["fixed", "varlen"])
def test_bench_chunk_h_tflops(
    chunk_h_backend: tuple[str, Callable],
    use_varlen: bool,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
):
    backend, operator = chunk_h_backend
    ...  # 各 backend 共享同一输入构造
    ms = time_ms(lambda: operator(**inputs), bench_warmup, bench_repeat)
```

机制：

- 行的 `backend` / `version` 列跟随 case（取元组的 backend 名，优先级
  高于 `$OPTEST_BACKEND`），`time_ms` fixture 也按 case 的 backend 选
  计时工具链（§3.1）；
- 入口仍按文件检测的 backend 排 pass：该文件只声明
  `importorskip("triton")`（检测为 triton），只跑一次 pass，不重复；
  文件内 backend 是 case 维度，不参与 `--backends` 过滤；
- backend 发现统一用 `boltops.utility.backends.discover_backends(op
  包, 算子名)`（遍历 `boltops/<op>/<backend>/` 子包，缺依赖视为不可
  用），各测试文件不要各自复制发现逻辑。

**备选 A：每 backend 一个测试函数**（两个 backend 的接口无法逐参数
对齐、或 shape / 输入不同，无法共享同一 parametrize 时用）：文件头
两个 `importorskip`（双 backend 文件，§8），每个 bench 函数名带自己
的 backend 名（`test_bench_triton_op` / `test_bench_tilelang_op`），
函数体各写各的。入口照跑两个 pass；每个 pass 自动 skip 函数名属于
另一 backend 的测试，每个测试只在自己的 pass 里跑一次。行的
`backend` / `version`、`time_ms` 的计时工具链都按函数名路由（看**裸
函数名**，parametrize id 不算；函数名含两个或零个 backend 名时此规则
不生效）。代价是函数体重复。

**备选 B：统一入口分发**（`$OPTEST_BACKEND`，当前无测试文件使用）：
测试调用 op 的统一入口，`boltops/` 内分发层读 `$OPTEST_BACKEND` 选
实现。仅当 op 的分发层已在统一入口时使用；否则优先上面的
parametrize 模式。

backend 声明与覆盖：

- **默认自动检测**：文件模块级 `pytest.importorskip("<toolchain>")`
  即声明（tilelang 实现的文件本就保留该写法，缺依赖时自动 skip；文件
  同时 skip 两个工具链 = 多 backend 文件）。
- 显式覆盖（检测不符或要强制参与某 pass 时才需要）：模块级
  `PERF_BACKEND = "tilelang"`（单）或 `PERF_BACKENDS =
  ["triton", "tilelang"]`（多，同一份测试参与多个 pass）。

- 运行：入口按文件声明的 backend 分组、每个 backend 一个 pytest
  pass，pass 内设 `$OPTEST_BACKEND=<name>`；测试代码不感知具体
  backend，本地 `pytest --bench` 下变量未设，op 走自己的默认实现。
- 输出：同一 op 的各 backend 行合并进一个 `<op>.csv`，`backend` 列
  区分；`version` 列填该行 backend 对应工具链的 dist 版本
  （版本串自带该工具链 build commit）。
- 局部运行：`python scripts/run_operator_benchmarks.py --ops <op>
  --backends triton tilelang`。
- 新增 backend = op 侧加实现 + 相关文件 `importorskip` 覆盖该
  工具链（或 `PERF_BACKENDS` 加一项），测试与框架零改动；parametrize
  模式下新 backend 子包导出同名算子后 `discover_backends` 自动带上，
  连测试都不必改。

## 7. 新增测点（示例）

### 7.1 已有 op 加新 case（最常见）

已有 op 只需在其 bench 文件里加测点（自动发现，无需注册）。例：给
`gdn_packed_decode` 的 B 网格加一个 `128`：

```python
# test/fla/test_gdn_packed_decode.py（现状）
@pytest.mark.bench
@pytest.mark.parametrize("B", [1, 8, 32, 64], ids=lambda x: f"B{x}")
def test_bench_packed_decode_bandwidth(B, bench_warmup, bench_repeat):
    ...
```

把列表改成 `[1, 8, 32, 64, 128]`（或另加一个 `@pytest.mark.bench`
函数）即可。验证：

```bash
python scripts/run_operator_benchmarks.py --ops gdn_packed_decode
# optest_results/gdn_packed_decode.csv 应多出 B=128 一行
```

### 7.2 全新 op

**第 1 步：写普通 bench 单测**（不 import perf 框架、不出现 backend
名；case 用 parametrize 表达，dataclass / dict / 标量都会自动进 CSV
列）：

```python
# test/fla/test_my_kernel.py
from dataclasses import dataclass

import pytest
import torch
import triton

from boltops.fla.my_kernel import my_kernel_fwd

GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HCU device is required"
)


@dataclass(frozen=True)
class _Shape:
    B: int
    T: int
    H: int
    K: int
    V: int


_SHAPES = [
    _Shape(B=16, T=1024, H=16, K=128, V=128),
    _Shape(B=64, T=2048, H=16, K=128, V=128),
]


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"B{s.B}T{s.T}")
def test_bench_my_kernel(shape: _Shape, time_ms, bench_warmup: int, bench_repeat: int):
    q = torch.randn(
        shape.B, shape.T, shape.H, shape.K,
        device="cuda", dtype=torch.bfloat16,
    )

    def run():
        return my_kernel_fwd(q)

    out = run()
    assert torch.isfinite(out).all()

    ms = time_ms(run, bench_warmup, bench_repeat)
    print(f"[bench] my_kernel B={shape.B} T={shape.T}: {ms:.4f} ms")
```

**第 2 步：（可选）文件级声明**——backend 由文件的
`importorskip("<toolchain>")` 自动检测（§8），什么都不用做；仅当
检测不符（要强制覆盖）或需要 op 改名时，在文件头部加模块级常量：

```python
# test/fla/test_my_kernel.py
PERF_BACKENDS = ["triton", "tilelang"]  # 强制声明 backend（默认自动
                                           # 检测；多 backend 文件参与
                                           # 多个 pass，§6）
PERF_OP = "my_kernel"                     # 人工钉 op 名（仅 test/ 根下
                                           # 无法自动派生时用，§8）
```

**第 3 步：跑并验证**：

```bash
python scripts/run_operator_benchmarks.py --ops my_kernel
head optest_results/my_kernel.csv
```

预期 CSV（dataclass 标量字段自动成为 case 列）：

```
operator,backend,B,T,H,K,V,version,arch,timing,latency_ms
my_kernel,triton,16,1024,16,128,128,3.6.0+git46766868.staging,gfx938,cudagraph,0.123456
my_kernel,triton,64,2048,16,128,128,3.6.0+git46766868.staging,gfx938,cudagraph,0.234567
```

全量 `optest --op_name boltops --test_dir . --test_mode per` 的 csv
计数应 +1，结果目录出现 `my_kernel.csv`。

### 7.3 易错点清单

- 计时统一用 `time_ms` fixture（**新测试不要直调 `do_bench`**，
  §3.1）：cudagraph 优先、捕获失败自动回退 event、默认 median
  （`return_mode=` 可换，非默认记 `stat` 列，§5）、毫秒
  预算；被测调用含 host 侧工作（每次调用
  `.item()` / `.to("cpu")` 等）时不可捕获，fixture 自动回退 event，
  该行 `timing` 列会显示 event（也可显式 `use_graph=False` 跳过捕获
  尝试，§5）。
- 直调计时函数（模块限定名）仅限现有特殊场景（双 port 显式
  event 对比、工具链特有参数）：tilelang 的 `from ... import` 形式有
  conftest 兼容，triton 侧没有（§3.1）。
- 同一测试内两个计时点要互相比较（如双 port 比率）时必须同方法：
  若一侧不可捕获而另一侧可，`time_ms` 会两侧方法不同，此时应两侧
  显式 `do_bench`（event），见 causal_conv1d fwd。
- case 参数必须经 parametrize（标量原样、dict / dataclass 取标量
  字段）才进 CSV 列；shape 不要硬编码在函数体里。
- op 名 = 文件名去 `test_` / `generic_` 前缀，泛化名（`bench` /
  `benchmark` / `perf` / `test`）或与其他文件重名时自动加所在目录
  前缀（`test/step4_dsa/test_bench.py` → `step4_dsa_bench`）：
  决定 `<op>.csv` 文件名、raw 行 `operator` 列、raw 文件名三者一致
  （§8）。`PERF_OP` 常量可钉名 / 把多个文件合并为一个 op。
- 测试不 import perf 框架、不调用收集 API（`time_ms` 可选关键字
  除外，§5）；失败 case 产 error 行（§3.3），入口以退出码 1 上报。
- 共享机首跑 / 高峰时段数值可能抖动，对比用同窗口 A/B（§9）。

## 8. 自动发现（无注册表、无注册）

入口启动时扫描 `test/` **自动发现** op bench：没有清单要维护，
新增 perf 测点 = 写一个正常 bench 单测，完事。

- **哪些文件是 bench 文件**：`test/` 下任意 `test_*.py` 含有
  `@pytest.mark.bench` 行（静态行扫描，不 import 文件）。
- **op 名**：文件名去 `test_` / `generic_` 前缀（与行 `operator`
  字段同规则），**自动派生、无需声明**：
  - 派生名是泛化词（`bench` / `benchmark` / `perf` / `test`，如
    `test/step4_dsa/test_bench.py`）或与其他文件重名（如
    `test/ihc/test_post.py` 与 `test/mhc/test_post.py`）时，自动加
    所在目录前缀：`step4_dsa_bench`、`ihc_post` / `mhc_post`。
  - 直接放在 `test/` 根下的文件没有目录可加前缀，无法自动区分时
    才用模块级 `PERF_OP = "..."` 钉名（人工兜底，当前无文件需要）。
- **backend**：从文件模块级 `pytest.importorskip("<toolchain>")`
  自动检测（同时 skip 两个工具链 = 多 backend 文件）；模块级
  `PERF_BACKEND = "tilelang"`（单）或 `PERF_BACKENDS =
  ["triton", "tilelang"]`（多，文件参与多个 pass）可覆盖检测结果。
  每个文件按其 backend 各跑一个 pytest pass（§6）；行 `backend`
  字段解析与 `time_ms` fixture 路由同规则（case 内 parametrize 的
  `(backend, 实现)` 值 > 函数名 backend 名 > `$OPTEST_BACKEND` >
  显式声明 > 检测 > `triton`，§3.1 / §6）。
- **排除**：`PERF_SKIP = "<reason>"` 把文件移出入口（本地
  `pytest --bench` 照常跑）。
- **无 CSV 的 op**：计时不被观测层拦截的 bench 测试（如手动
  `torch.cuda.Event` 采样）照常运行但不产行；全部测试 skip 也不产行
  → 该 op 无 CSV，入口打印提示。注意：失败的测试**会**产 error 行，
  因此"测试失败"不再导致 op 缺 CSV。

常量经 AST 静态解析（只认字面量），不 import 测试模块。

### 8.1 当前 op 与 bench 测例（盘点快照）

2026-10-14 盘点：30 个文件、38 个 `@pytest.mark.bench` 测试函数、
30 个 op（8 个文件为 tilelang backend，1 个文件为文件内双 backend，
见 §6）。37 个可被观测层自动收集，唯一例外是 `moe_sorting_hipc`
的手动 `torch.cuda.Event` 采样（备注列说明）。

| op | 测试文件 | bench 测试函数 | 计时入口 | backend | 备注 |
| --- | --- | --- | --- | --- | --- |
| `causal_conv1d` | `test/test_generic_causal_conv1d.py` | `test_bench_causal_conv1d_fn_compare` | 模块限定 `triton.testing.do_bench`（两侧 event） | triton | fwd 双 port 对比（call=1/2 = sglang/vllm port）；fwd 两侧显式 event（vllm 端口 host 同步不可捕获，比率须同方法）；`x` 按生产 channel-last 布局构建 |
| `causal_conv1d` | `test/test_generic_causal_conv1d.py` | `test_bench_causal_conv1d_update_compare` | `time_ms` | triton | update 双 port 对比（call=1/2） |
| `gdn_chunk` | `test/fla/test_gdn_chunk.py` | `test_bench_chunk_h_tflops` | `time_ms`（AITER 参考计时不收集） | triton + tilelang（文件内，§6） | chunk_h：backend 由 `chunk_h_backend` parametrize（tilelang 仅 VK state 布局）；tilelang VK 案例另有 AITER kernel 数值 + 性能参考（print） |
| `gdn_chunk` | `test/fla/test_gdn_chunk.py` | `test_bench_chunk_o_tflops` | `time_ms` | triton | chunk_o：tilelang 侧暂无 chunk_o 实现（`CHUNK_O_BACKENDS` 仅 triton）；fixed T=13312 + varlen，BF16 |
| `gdn_packed_decode` | `test/fla/test_gdn_packed_decode.py` | `test_bench_packed_decode_bandwidth` | `time_ms` | triton | packed decode B=1/8/32/64（BF16/FP32 state） |
| `gdn_recompute_w_u` | `test/fla/test_gdn_recompute_w_u.py` | `test_bench_recompute_w_u` | `time_ms` | triton | `recompute_w_u_fwd` w/u 生成，8 个实测/推导 shape |
| `gdn_sigmoid_gating` | `test/fla/test_gdn_sigmoid_gating.py` | `test_bench_sigmoid_update_bandwidth` | `time_ms` | triton | vLLM decode operator（N=4 × T=1/4） |
| `gdn_sigmoid_gating` | `test/fla/test_gdn_sigmoid_gating.py` | `test_bench_recurrent_speculative_verify_bandwidth` | `time_ms` | triton | speculative verify（N=1/8/32/48） |
| `kda_decode` | `test/fla/test_kda_decode.py` | `test_bench_kda_decode` | `time_ms` | triton | 三个 decode 入口（dense gate fused / dense_pregated SGLang 路径 / packed decode），N=1/6/32/64；与 `docs/ops/fla/kda/kda_decode.md` 证据表同步 |
| `kda_prefill` | `test/fla/test_kda_prefill.py` | `test_bench_kda_prefill` | `time_ms` | triton | `chunk_kda_with_fused_gate_fwd` 端到端 mainline（出厂 config）；shape 与 `scripts/tune/triton/tune_kda_prefill.py` 的 `SHAPE_PRESETS` 同步 |
| `pre_big_fuse` | `test/mhc/test_pre_big_fuse.py` | `test_bench_mhc_pre_big_fuse` | `time_ms` | tilelang | `mhc_pre_big_fuse` CUDA graph replay（对齐 AITER graph 回放方式），n1=1..16384 × H=4096/7168，MHC=4；backend 由 `importorskip("tilelang")` 自动检测 |
| `batched_gemm_bf16` | `test/gemm/test_batched_gemm_bf16.py` | `test_bench_batched_gemm_bf16` | `time_ms` | triton | 出厂 tuned config × NT / TT 两 layout × M grid（FP16，`BATCH_LAST_GRID`） |
| `flash_mla_sparse` | `test/mla/test_flash_mla_sparse.py` | `test_bench_flash_mla_sparse` | `time_ms`（独立 300ms 预算） | triton | baseline / tuned 两种 config × 6 个代表 shape（含 2 个真实模型大 shape）；`_BENCH_REPEAT_MS=300` |
| `sparse_gqa` | `test/sparse_gqa/test_sparse_gqa.py` | `test_bench_simulated_production_decode` | `time_ms` ×3 | triton | compact（HCU-FA paged KV → packed selected K/V，FP8 cache）+ packed decode + e2e（`call`=1/2/3）× 6 个生产 shape |
| `extend_attention` | `test/fa/test_extend_attention.py` | `test_extend_attention_benchmark` | 模块限定 `triton.testing.do_bench_cudagraph`（默认 rep=20ms / mean） | triton | 可收集；可迁 `time_ms` 统一为 median / 毫秒预算 |
| `unified_attention` | `test/fa/test_unified_attention.py` | `test_unified_attention_benchmark` | `time_ms` | triton | |
| `moe_activation` | `test/fused_moe/test_moe_activation.py` | `test_moe_activation_benchmark` | `time_ms` | triton | 无 case 列：单一硬编码 shape，未 parametrize |
| `moe_sorting_hipc` | `test/fused_moe/test_moe_sorting_hipc.py` | `test_bench_moe_sorting` | 手动 `torch.cuda.Event` 逐次采样（median/p10/p90） | triton | **唯一不收集**（无 CSV）：p10/p90 分布统计超出当前 CSV（单 `latency_ms` 列）的表达；要进 CSV 需把计时改为 `time_ms`（p10/p90 保留在 print） |
| `gemm_a16w16` | `test/gemm/test_gemm_a16w16.py` | `test_bench_gemm_a16w16` | `time_ms` | triton | |
| `matmul_persistent` | `test/gemm/test_matmul_persistent.py` | `test_bench_matmul_persistent` | `time_ms` | triton | |
| `head` | `test/ihc/test_head.py` | `test_bench_ihc_head` | `time_ms` | tilelang | op 名自动派生（`test/ihc/test_head.py` → `head`） |
| `ihc_post` | `test/ihc/test_post.py` | `test_bench_ihc_post` | `time_ms` | tilelang | op 名自动派生（`test_post` 重名，加目录前缀） |
| `pre` | `test/ihc/test_pre.py` | `test_bench_ihc_pre` | `time_ms` | tilelang | op 名自动派生（`test/ihc/test_pre.py` → `pre`） |
| `fused_post_pre` | `test/mhc/test_fused_post_pre.py` | `test_bench_mhc_fused_tilelang` | `time_ms` | tilelang | |
| `mhc_post` | `test/mhc/test_post.py` | `test_bench_mhc_post_fwd` | `time_ms` | tilelang | op 名自动派生（`test_post` 重名，加目录前缀） |
| `split_sinkhorn` | `test/mhc/test_split_sinkhorn.py` | `test_bench_hc_split_sinkhorn` | `time_ms` | tilelang | |
| `decode_attention` | `test/mla/test_decode_attention.py` | `test_bench_mla_decode_attention` | `time_ms` | triton | |
| `fp8_mqa_logits` | `test/mqa_logits/test_triton_fp8_mqa_logits.py` | `test_bench_triton_fp8_mqa_logits_size_grid` | `time_ms` + `flops=` | triton | `PERF_OP`；CI 默认收集；`s_q,s_k,num_heads,head_dim` |
| `fp8_mqa_logits` | `test/mqa_logits/test_triton_fp8_mqa_logits.py` | `test_bench_triton_fp8_mqa_logits_vs_hipc_size_grid` | `time_ms` ×2 | triton | `-k vs_hipc` 才跑；默认 CI skip；`call`=1/2 |
| `fp8_mqa_logits` | `test/mqa_logits/test_triton_fp8_mqa_logits.py` | `test_bench_triton_fp8_mqa_logits_vs_hipc_prod_slices` | `time_ms` ×2 | triton | `-k vs_hipc` 才跑；默认 CI skip；`call`=1/2 |
| `half_mqa_logits` | `test/mqa_logits/test_triton_mqa_logits_half.py` | `test_bench_triton_mqa_logits_half` | `time_ms` | triton | `PERF_OP`；CI 默认收集；`s_q,s_k,num_heads` |
| `half_mqa_logits` | `test/mqa_logits/test_triton_mqa_logits_half.py` | `test_bench_triton_mqa_logits_half_vs_hipc` | `time_ms` ×2 | triton | `-k vs_hipc` 才跑；默认 CI skip；比值不足再测一次 |
| `fp8_paged_mqa_logits` | `test/mqa_logits/test_triton_fp8_paged_mqa_logits.py` | `test_bench_triton_fp8_paged_mqa_logits_gpu` | `time_ms` | triton | `PERF_OP`；CI 默认收集；标量 `batch,next_n,heads,avg_kv`；只测 splitkv_fuse |
| `fp8_paged_mqa_logits` | `test/mqa_logits/test_triton_fp8_paged_mqa_logits.py` | `test_bench_triton_fp8_paged_mqa_logits_vs_hipc` | `time_ms` ×2 | triton | `-k vs_hipc` 才跑；默认 CI skip；`call`=1/2 |
| `tilelang_mqa` | `test/mqa_logits/test_tilelang_mqa.py` | `test_bench_qsa_mqa_prefill_kernel` | `time_ms` | tilelang | |
| `tilelang_mqa` | `test/mqa_logits/test_tilelang_mqa.py` | `test_bench_qsa_mqa_decode_cooperative_load` | `time_ms`（循环内多计时点） | tilelang | 循环变量（query_heads 等）不进 CSV，仅 `call` 自动编号；要成 case 列需改 parametrize；本测例在本机可复现 GPU VM fault（存量 kernel 问题，§9），修复前会带崩 tilelang pass |
| `step4_dsa_prefill` | `test/step4_dsa/test_step4_dsa_prefill.py` | `test_bench_sparse_prefill` | `time_ms`（tiled 用 `use_graph=False`） | triton | `PERF_OP` 钉名；shape 与 `boltops/step4_dsa/triton/config/sparse_attn_prefill*` 逐条对应（scalar q=1..4096 bf16 + q=128 fp8；tiled q=128 bf16/fp8）；tiled 建 tile 有 host 同步不可 graph 捕获，timing=event |
| `step4_dsa_decode` | `test/step4_dsa/test_step4_dsa_decode.py` | `test_bench_sparse_decode` | `time_ms` | triton | `PERF_OP` 钉名；shape 与 `config/sparse_attn_decode` 逐条对应（bf16 b=1..128、fp8 b=1/4/8，splits=1）；context=16384 取自 TP8 production case；metadata 在计时外生成 |
| `attn_res` | `test/test_generic_attn_res.py` | `test_bench_attn_res` | `time_ms` | triton | |
| `mamba_state_scatter` | `test/test_generic_mamba_state_scatter.py` | `test_bench_mamba_state_scatter` | `time_ms` | triton | |

本表为人工快照：新增 / 改造 bench 测例时同步更新（§9）。

**预留的第二 backend 槽位**（`boltops/<op>/tilelang/` 空包已存在、
尚无实现）：`gemm`、`mla`、`fused_moe`、`step4_dsa`。槽位落地时
**不需要注册任何东西**：子包开始导出同名算子后，`discover_backends`
与 perf 自动发现（§6 / §8）自然把它带上——按 §6 任一模式写一个真实
bench 测试文件即可，op 名 / CSV / `backend` / `version` 列自动就位。
不提前写占位 bench 文件：自动发现下它在落地前不产生任何行为，只会
过期。

## 9. 维护约定

- 入口脚本路径被 optest source marker 硬固定，勿移动；op 自动发现（§8），
  新增测点 = bench 单测，零额外改动。
- 测试文件保持普通单测形态：计时统一用 `time_ms` fixture（§3.1）、
  shape 用 parametrize / dataclass 表达、建议 print 输出 case 摘要
  （可选：不参与收集，但日志可读性与崩溃定位靠它）；
  不 import perf 框架、不调用收集 API（`time_ms` 可选关键字
  除外，§5）。
- perf 框架不 import `test/`（仓库 `test/` 无顶层 `__init__.py`，会被
  标准库 `test` 包遮蔽，import 不到）。perf 靠 pytest 驱动 + 文件契约
  打通：入口以子进程跑 `pytest --bench`（测量结果靠
  `$OPTEST_BENCH_RAW_DIR` 下的 raw 行），不 import 任何 test 模块。
- 机器为共享环境，同一 case 跨 run 可能有明显抖动。下结论时用同一窗口
  A/B 或多次运行，不要用单次 run 的数值差判断回归。
- 已知问题（2026-10-09）：`test/mqa_logits/test_tilelang_mqa.py` 的
  `qsa_mqa_decode_persistent_kernel` 在本机可复现 GPU VM fault
  （dmesg XID 81），裸 `pytest --bench`（无任何 perf 框架参与）同样
  崩溃，属存量 kernel / 环境问题而非本框架：修复前全量跑时 tilelang
  pass 会在此中断（入口 rc≠0），但其前的 op 不受影响——raw 行按测试
  逐条落盘，各 op CSV 照常产出。请 qsa_mqa owner 跟进。
- CI 的 perf 环境需要 `pytest`（入口以 `python -m pytest` 子进程驱动）。
