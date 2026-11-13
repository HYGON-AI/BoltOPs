# BoltOPs GitHub Actions 迁移与编写说明

当前迁移分支已合入 GitLab main 的 e4c2c58（2026-11-03 同步），保留 GitLab 原始提交历史及 GitHub CI 改造。GitLab CI 暂时保留，正式切换需按下面的顺序操作，避免重复发布。参考 lightop/master 的 pr-test.yml、release-pypi.yml 和 .github/scripts/ 结构，使用 ARC runner pools 与预装 DTK 的 RockyLinux 作业容器。

## 1. 先理解整体流程

PR 到 main 和手动运行时，`build2604` 和 `build2610` 各自的两组环境都在 `hygon-build` 构建 wheel。DTK26.04 / Python3.10 / Torch2.11.0，以及 DTK26.10 / Python3.12 下的 Torch2.11.0 和 Torch2.13.0 在 `hygon-bw1100-4card` 运行正确性测试，通过后在 `hygon-bw1100-1card` 下载 wheel 并运行性能测试。`ci-result` 在 `hygon-cpu` 汇总两个 DTK 的构建和测试结果。

push 到 main 时构建并修复四组 wheel，各 DTK 的构建成功后，由 `publish-2604` 和 `publish-2610` 在 `hygon-build` 分别发布。正确性和性能测试在 PR 或手动运行时执行。

## 2. 文件怎么写、怎么看

| 文件 | 职责 |
|---|---|
| .github/workflows/ci.yml | 事件、矩阵、依赖 needs、制品传递、发布条件 |
| .github/workflows/_build-test-wheel.yml | workflow_call 复用构建流程，供两套 DTK 的版本矩阵调用 |
| .github/workflows/_publish.yml | workflow_call 接收 DTK 镜像版本、发布标识、包名及发布密码，统一下载、校验、发布 wheel |
| .github/scripts/checkout-source.sh | 校验实际 checkout 的事件 SHA |
| .github/scripts/common.sh | 每一步加载 DTK；设置索引、输出目录和唯一运行标识 |
| .github/scripts/setup-env.sh | 安装 AICC、精确选择 Torch/Triton/TileLang/NumPy 并验证版本 |
| .github/scripts/_build_hcu_wheel.sh | 统一检查环境并打包；CIUpload REPAIR 由复用工作流执行 |
| .github/scripts/test-correctness.sh | 把 wheel 安装到临时目录，在该目录运行 test，避免误测源码 |
| .github/scripts/test-performance.sh | 下载并安装对应版本构建的 wheel，运行 optest，收集 JSON、日志、JUnit |
| .github/scripts/publish.sh | 收集 wheel 到发布目录，使用 CIUpload NIGHTLY 上传 |

先写触发条件 on，再写矩阵和 container，然后给每个动作写一个 shell 脚本，最后用 needs 串起来。不要把 Actions 的 `${{ ... }}` 写进独立 shell 文件；先通过 env 传入，脚本只使用 `${VAR}`。

PR checkout 采用 github.sha（GitHub 生成的 PR 合并提交），所有矩阵与性能任务使用同一个 SHA，包含目标分支集成结果。push 则验证实际提交。无需像 GitLab 那样在每个 job 临时 merge main。

DTK26.04 构建 Python3.10 / NumPy1.24.3 与 Python3.12 / NumPy1.26.4，两组都搭配 Torch2.11.0。DTK26.10 构建 Python3.12 / NumPy1.26.4，搭配 Torch2.11.0 和 Torch2.13.0。环境准备阶段校验 HCU Torch 版本以及 Triton、TileLang 的 DTK 和 Torch 标签。

统一入口为`.github/scripts/_build_hcu_wheel.sh`，旧 `build_wheel.sh` 转发到新入口；GitLab 与 GitHub CI 都调用同一脚本。环境需预先准备，入口不隐式安装工具。检查通过后输出 wheel，测试和发布使用同一构建路径。

lightop 的 manylinux_repair.sh 面向含原生 .so 的 wheel；BoltOPs 当前是纯 Python wheel，因此保留 CIUpload 的版本标签修复，不增加 auditwheel repair。

## 3. 管理员先配置这些项

GitHub BoltOPs → Settings → Secrets and variables → Actions → Variables：

| Variable | 值/含义 |
|---|---|
| PYPI_URL | 通用 devpi 服务根地址（包含 http:// 与端口，无 /nightly 后缀）；构建及性能任务默认 http://12.12.12.201:9929，与 lightop 一样支持 fork PR 的变量缺失场景；发布任务仍读取组织/仓库 Variable |
| TRITON_PYPI_URL | Triton 镜像根地址；可与 PYPI_URL 相同，未设置时回退到 PYPI_URL |
| AICC_INSTALLER_URL | 默认使用提供的 http://42.228.13.241:18000/ai_cc/Nightly/hcu_llvm_installer.sh；可配置覆盖 |
| RESOURCE_SERVER_URL | 可选，准备脚本单独使用时用于拼接 AICC 地址的回退值 |
| DTK_IMAGE_REGISTRY | 可选；默认 10.16.1.152:5000/hcu/base/dtk |

Secret：PYPI_PASSWORD，使用现有 nightly 发布密码（不要把密码写入文件或 PR）。直接使用授权给 BoltOPs 的组织级 Secret，也可以配置仓库级 Secret。

发布需要 Variable `PYPI_URL` 和发布 Secret `PYPI_PASSWORD`。两者都可以使用已授权给 BoltOPs 的组织级配置，无需在仓库重复设置；其他构建 Variables 也可以使用组织级或仓库级配置。GitHub 自动提供工作流使用的 `GITHUB_TOKEN`，这套 CI 不需要额外配置 GitHub PAT 或 `GITHUB_MIRROR_TOKEN`。fork PR 运行可能拿不到 `vars`；构建与性能任务为非敏感包源地址提供默认值，不能仅依赖组织变量授权。个人 fork 还需要能使用对应 runner。发布 Secret 不提供默认值，也不传给 PR 构建与性能任务。

发布任务与 lightop 一样，不绑定 GitHub Environment，无需创建 nightly 环境；工作流中的条件限制仅在 main 分支 push 后发布。

Settings → Actions → Runners：确保 ARC 的 `hygon-cpu`、`hygon-build` 、`hygon-bw1100-1card` 和 `hygon-bw1100-4card` pools 都授权给 BoltOPs。其他仓库可使用不代表 BoltOPs 自动有权使用。构建、正确性、性能和发布继续使用同一个基础镜像：

- `${DTK_IMAGE_REGISTRY}:26.04-rockylinux8.6-py3.10` 或 `:26.04-rockylinux8.6-py3.12`
- `${DTK_IMAGE_REGISTRY}:26.10-rockylinux8.6-py3.12`
- 默认 registry 为 `10.16.1.152:5000/hcu/base/dtk`

ARC/Kubernetes 负责拉镜像、创建和销毁 Pod，给正确性测试 Pod 分配四张 HCU、给性能测试 Pod 分配一张 HCU；工作流不传 Docker 的 `--device`、特权、共享内存或手工挂载选项。平台需要给构建和测试 Pod 提供所需的 `/opt/hyhal`，给测试 Pod 提供设备与足够的 `/dev/shm`。原 `--shm-size=80G` 不再由 workflow 设置；接入文档中的四卡池提供 96G、单卡池提供 32G 共享内存，实际容量需在平台 Pod 模板上确认。私有镜像认证由平台现有配置负责。

AICC、Torch、Triton、TileLang、NumPy 和测试依赖继续在各自的临时作业容器中安装，版本和校验逻辑不变，不重新打环境镜像。编译并发沿用平台注入的 `MAX_JOBS`，未提供时回退到 4，不再覆盖为 32。

正确性工作流校验可见 HCU 数量必须为 4，性能工作流校验必须为 1，不运行 `CIsmi need`，不重写平台提供的 `ROCR_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES`。性能 job 在 env 中设置 `BOLTOPS_TEST_GPU: '0'`，optest 使用 Pod 内逻辑卡号 0，保留现有单卡基准语义；正确性测试可见逻辑卡 0～3，原有跨卡用例可以执行。没有额外的设备选择脚本。

各版本组合的构建、正确性测试和性能测试可独立调度。性能任务各用一张 HCU，按单卡 runner 池的容量并行运行。同一节点上的任务仍可能共享 CPU、内存和 I/O。工作流仍会取消同一 PR 的旧运行。

同仓库 PR 和 fork PR 均使用 pull_request 事件进入构建和 GPU 测试，不使用 pull_request_target。是否等待维护者批准取决于 GitHub 组织/仓库的 fork 工作流审批设置；个人 fork 还需要相应 pools 的使用权限。PR 不进入发布任务，发布 Secret 只用于 main push 的发布任务。

## 4. 把文件提交到 GitHub、试跑

当前 GitHub 登录账号 guochenyang99 对 BoltOPs 只有 pull 权限。迁移修改位于 `ci/github-actions-migration` 分支；origin 指向 HYGON-AI/BoltOPs，fork 指向 guochenyang99/BoltOPs。

当前迁移分支已整合 GitLab main@e4c2c58，但这不代表 GitHub main 已更新。维护者合入 PR 前应确认 GitLab 是否又有新提交；如有，应再次合并并验证，避免遗漏近期算子/测试修复。

可将交付 zip 中的 .github/ 与 docs/ci/ 合入该分支，或者使用交付 patch：

```bash
git switch -c ci/github-actions-migration main
git apply --check boltops-github-actions.patch
git apply boltops-github-actions.patch
git add -A -- .gitattributes .github .agents docs/ci scripts/ci ci_script build_wheel.sh .gitlab-ci.yml
git commit -m 'ci: add GitHub Actions build test and nightly workflows'
git push -u fork ci/github-actions-migration
```

从原仓库分支或个人 fork 创建 PR 到 main；如页面要求，等待维护者批准工作流运行。检查四组构建、三组正确性及性能作业、ci-result 和已有 Quality Gate。PR 不会发布，也不接触 PYPI_PASSWORD。下载 artifact 核对 wheel、JUnit 与性能报告。三组性能作业均需在 BW1100 上实际验证。

由于需要真实内网 runner、编译器和 GPU，本地只能验证 YAML/Actions 表达式、Bash 语法、矩阵、设备选择分支和脚本迁移逻辑；未实际运行 GPU 构建、测试或上传。

## 5. 正式切换

PR 验证通过后，安排同一次切换窗口：

1. 停止接受 GitLab 独立 main 更新，选定 GitHub 为唯一主入口。
2. 维护者停用 GitLab 的发布与 mirror:github（或停用整个 GitLab CI），并等待/取消仍在运行的发布任务。不要同时让两边对同一提交发布。
3. 合入 GitHub 的迁移 PR；main push 只构建并发布，不重复 PR 的测试。
4. 确认发布成功。将 BoltOPs CI 的 ci-result 设为必需检查，保留已有 Checks / All required checks。必须先实际运行一次，让 GitHub 能识别检查名称。
5. 后续开发走 GitHub PR，GitLab 如需保留副本，应另行设计反向镜像，不能继续两边独立合入。

迁移代码不会自动停用 GitLab 项目设置。现有 .gitlab-ci.yml 仅将 wheel 编译命令替换为统一脚本，原镜像脚本不变。

## 6. 本地验证

```bash
actionlint -config-file .github/actionlint.yaml .github/workflows/ci.yml .github/workflows/_build-test-wheel.yml .github/workflows/_publish.yml
for f in .github/scripts/*.sh; do bash -n "$f"; done
git diff --check
```

原性能测试可能因共享 CPU、内存、I/O 等资源而慢或失败；迁移保留现有测试标准，不将失败改成忽略。optest 自身也可能修改已约束的 NumPy，原始日志中曾出现；若 GitHub 实测复现，应单独修复 optest 的环境安装逻辑。

性能阶段虽然下载并安装构建产物，但 `scripts/run_operator_benchmarks.py` 会将仓库根目录插入 `sys.path`，pytest 配置也包含 `pythonpath = ["."]`，因此当前不能保证实际测量的是安装的 wheel，而非同一提交的仓库源码。若要求性能阶段严格验证 wheel，需要同时调整 benchmark 的导入方式和 optest 的环境准备行为，并在 runner 上确认实际导入路径。正确性测试通过复制测试到临时目录来隔离源码。

本地重复打包前会清理 setuptools 的 `build/` 目录，避免已删除源码被残留文件带入 wheel。报告转换对缺失、空 JSON 或无测试记录的结果返回失败，并生成失败 JUnit，防止未执行测试却显示成功。

Shell 入口由 `.gitattributes` 固定为 LF 换行，避免 Windows 提交的 CRLF 影响 Linux 执行。临时目录标识包含 job 名称，区分构建与性能任务；性能制品显式上传隐藏文件，保留 `.optest-results.json`。构建 artifact 只传 wheel；正确性测试 JUnit 使用独立的 correctness-reports-* artifact，避免对已上传的 wheel artifact 重复写入。AICC 目前按大版本选择最新包，Torch/Triton 按包源解析后验证版本与 DTK/Torch 标签；包源可访问性、最新 AICC 是否适配、optest 是否改动依赖仍需在 BW1100 试跑确认。

## 7. 本地统一编译入口与仓库技能

```bash
bash .github/scripts/_build_hcu_wheel.sh --help
bash .github/scripts/_build_hcu_wheel.sh --check-only
bash .github/scripts/_build_hcu_wheel.sh --out-dir dist
```

要求 Linux、Python >=3.9、DTK、HIP 版 Torch、AICC 和 Python build。脚本默认从 /opt/dtk/env.sh 加载环境；可用 DTK_ROOT、PYTHON、BOLT_OPS_AICC 指定路径。缺少工具会在打包前报错，不会自动安装或偷偷退回 CPU Torch。检查可导入 Torch，但不要求能访问 GPU；GPU 验证交由测试负责。

仓库技能位于 `.agents/skills/build-hcu-wheel/SKILL.md`，记录环境前提、参数、编译方法、错误处理与 CI 对应关系。技能是给智能助手阅读的说明，不是 CI 执行程序。

### 下载当前性能测试 CSV 和原始结果

打开 GitHub Actions 对应运行的 Summary，在 Artifacts 中下载
`performance-reports-dtk<DTK>-py3.12-torch<版本>`。不同 Torch 版本分别归档，保留 7 天。

- `optest_results/`：本次性能入口写出的各算子 CSV。
- `optest_artifacts/BW1100/dtk<DTK>/py3.12/torch<版本>/results/`：本 job 的
  optest 输出目录，保留实际生成的 CSV、`.optest-results.json`、`raw/` 等文件。
- 同级 `logs/`：benchmark 日志；`collection.log` 记录收集路径。
- `scripts/*_boltops_output.xml`：JUnit 性能报告。

工作流在测试之后使用独立的 `always()` 步骤收集文件，再上传 artifacts。
即使测试失败或 JSON 未生成，也会保留已有输出；不改变测试的失败状态。
收集只匹配当前 run、attempt、job、DTK、Python 和 Torch 的唯一目录前缀，
多个结果目录分别保留。若环境安装失败而未开始测试，可能没有 CSV；
任务超时、runner 异常等情况不保证收集和上传完成。

目录规则见 [CI 目录与文件职责](README.md)。GitLab/GitHub 共用的报告转换工具统一放在 `scripts/ci/optest_to_junit.py`。
