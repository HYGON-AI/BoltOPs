# BoltOPs GitHub Actions 迁移与编写说明

当前迁移分支已合入 GitLab main 的 e4c2c58（2026-11-03 同步），保留 GitLab 原始提交历史及 GitHub CI 改造。GitLab CI 暂时保留，正式切换需按下面的顺序操作，避免重复发布。参考 lightop/master 的 pr-test.yml、release-pypi.yml 和 .github/scripts/ 结构，使用 ARC runner pools 与预装 DTK 的 RockyLinux 作业容器。

## 1. 先理解整体流程

PR 到 release/0.1.0 时，`build2604` 和 `build2610` 各自在 `hygon-build` 构建一个 wheel。DTK26.04 / Python3.10 / Torch2.11.0 与 DTK26.10 / Python3.12 / Torch2.13.0 两组在 `hygon-bw1100-4card` 运行正确性测试，通过后在 `hygon-bw1100-1card` 下载对应 wheel 并运行性能测试。`ci-result` 在 `hygon-cpu` 汇总两个 DTK 的构建和已启用的测试结果。

push 到 release/0.1.0 时按 DTK 各构建一个 wheel，并重新执行两组正确性和性能测试。每组 DTK 的构建、正确性及性能测试全部成功后，由对应的 `publish-2604` 或 `publish-2610` 在 `hygon-build` 发布；一组失败或取消只阻止该组发布，不影响另一组。手动运行按输入构建一组 wheel，并上传到本次 Actions 运行的附件。

## 2. 文件怎么写、怎么看

| 文件 | 职责 |
|---|---|
| .github/workflows/ci.yml | 事件、矩阵、依赖 needs、制品传递、发布条件 |
| .github/workflows/_build-test-wheel.yml | workflow_call 复用构建流程，供两套 DTK 的版本矩阵调用 |
| .github/workflows/_publish.yml | 接收 DTK 镜像版本、构建产物标识及发布密码，从镜像读取 DTK 版本后发布 wheel |
| .github/scripts/checkout-source.sh | 校验事件 SHA，并在 PR 任务中合并目标分支提交 |
| .github/scripts/common.sh | 每一步加载 DTK；设置索引、输出目录和唯一运行标识 |
| .github/scripts/setup-env.sh | 安装 AICC、精确选择 Torch/Triton/TileLang/NumPy 并验证版本 |
| .github/scripts/_build_hcu_wheel.sh | 本地与 GitLab 使用的环境检查和打包入口 |
| .github/scripts/test-correctness.sh | 把 wheel 安装到临时目录，在该目录运行 test，避免误测源码 |
| .github/scripts/test-performance.sh | 下载并安装对应版本构建的 wheel，运行 optest，收集 JSON、日志、JUnit |
| .github/scripts/publish.sh | 收集 wheel 到发布目录，使用 CIUpload NIGHTLY 上传 |

先写触发条件 on，再写矩阵和 container，然后给每个动作写一个 shell 脚本，最后用 needs 串起来。不要把 Actions 的 `${{ ... }}` 写进独立 shell 文件；先通过 env 传入，脚本只使用 `${VAR}`。

PR 使用 `pull_request_target` 触发，工作流定义来自目标分支。工作流按事件记录的完整 SHA 检出同仓库或 fork PR。构建、正确性和性能任务均校验触发事件记录的 PR head SHA，再合并该事件记录的目标分支 base SHA，保证使用相同的代码内容。push 使用事件 SHA。手动运行优先使用填写的完整 commit SHA，留空时使用事件 SHA。

DTK26.04 测试使用 Python3.10 / NumPy1.25.2 / Torch2.11.0。DTK26.10 测试使用 Python3.12 / NumPy1.26.4 / Torch2.13.0。测试环境准备阶段校验 HCU Torch 版本以及 Triton、TileLang 的 DTK 和 Torch 标签。

GitHub CI 在独立 Python 环境运行 `python -m build --wheel`，打包阶段只安装 Python 构建依赖。测试和发布使用同一份 wheel。`build_wheel.sh` 仍转发到本地和 GitLab 的环境检查入口。

GitHub 在打包前生成 wheel 的 DTK、构建时间和提交版本标识。产物的兼容性标签为 `py3-none-any`。

## 3. 管理员先配置这些项

GitHub BoltOPs → Settings → Secrets and variables → Actions → Variables：

| Variable | 值/含义 |
|---|---|
| PYPI_URL | 必填。devpi 服务根地址，包含协议和端口。构建和测试用它拼接 `nightly/dtk2604/+simple/` 或 `nightly/dtk2610/+simple/` |
| AICC_INSTALLER_URL | AICC 安装脚本的完整 URL，与 `RESOURCE_SERVER_URL` 至少配置一个 |
| RESOURCE_SERVER_URL | 资源服务器根地址，未设置 `AICC_INSTALLER_URL` 时用于拼接安装脚本 URL |
| HARBOR_SITE | 必填。Harbor 主机和端口，工作流拼接 `/hcu/base/dtk` 和镜像 tag |

Secret：PYPI_PASSWORD，使用现有 nightly 发布密码（不要把密码写入文件或 PR）。直接使用授权给 BoltOPs 的组织级 Secret，也可以配置仓库级 Secret。

构建、测试和发布使用 BoltOPs 可访问的组织级或仓库级 Actions Variables。发布使用 `PYPI_PASSWORD` Secret。

发布任务与 lightop 一样，不绑定 GitHub Environment，无需创建 nightly 环境；工作流中的条件限制仅在 release/0.1.0 分支 push 后发布。

Settings → Actions → Runners：确保 ARC 的 `hygon-cpu`、`hygon-build` 、`hygon-bw1100-1card` 和 `hygon-bw1100-4card` pools 都授权给 BoltOPs。其他仓库可使用不代表 BoltOPs 自动有权使用。构建、正确性、性能和发布继续使用同一个基础镜像：

- `${HARBOR_SITE}/hcu/base/dtk:26.04-rockylinux8.6-py3.10` 或 `:26.04-rockylinux8.6-py3.12`
- `${HARBOR_SITE}/hcu/base/dtk:26.10-rockylinux8.6-py3.12`

ARC/Kubernetes 负责拉镜像、创建和销毁 Pod，给正确性测试 Pod 分配四张 HCU、给性能测试 Pod 分配一张 HCU；工作流不传 Docker 的 `--device`、特权、共享内存或手工挂载选项。平台需要给构建和测试 Pod 提供所需的 `/opt/hyhal`，给测试 Pod 提供设备与足够的 `/dev/shm`。原 `--shm-size=80G` 不再由 workflow 设置；接入文档中的四卡池提供 96G、单卡池提供 32G 共享内存，实际容量需在平台 Pod 模板上确认。私有镜像认证由平台现有配置负责。

AICC、Torch、Triton、TileLang、NumPy 和其他测试依赖在各测试容器中安装，版本和校验逻辑不变，不重新打环境镜像。编译并发沿用平台注入的 `MAX_JOBS`，未提供时回退到 4，不再覆盖为 32。

正确性工作流校验可见 HCU 数量必须为 4，性能工作流校验必须为 1，不运行 `CIsmi need`，不重写平台提供的 `ROCR_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES`。性能 job 在 env 中设置 `BOLTOPS_TEST_GPU: '0'`，optest 使用 Pod 内逻辑卡号 0，保留现有单卡基准语义；正确性测试可见逻辑卡 0～3，原有跨卡用例可以执行。没有额外的设备选择脚本。

各版本组合的构建、正确性测试和性能测试可独立调度。性能任务各用一张 HCU，按单卡 runner 池的容量并行运行。同一节点上的任务仍可能共享 CPU、内存和 I/O。工作流仍会取消同一 PR 的旧运行。

同仓库 PR 自动运行。外部 fork PR 需要有写权限的维护者添加 `ok-to-test` 标签，之后同一 PR 的新提交自动运行。移除标签会撤销批准，新 PR 需重新批准。release/0.1.0 push 触发发布任务。

## 4. 把文件提交到 GitHub、试跑

迁移修改位于 `ci/github-actions-migration` 分支；origin 指向 HYGON-AI/BoltOPs，fork 指向 guochenyang99/BoltOPs。

当前迁移分支已整合 GitLab main@e4c2c58，但这不代表 GitHub main 已更新。维护者合入 PR 前应确认 GitLab 是否又有新提交；如有，应再次合并并验证，避免遗漏近期算子/测试修复。

可将交付 zip 中的 .github/ 与 docs/ci/ 合入该分支，或者使用交付 patch：

```bash
git switch -c ci/github-actions-migration release/0.1.0
git apply --check boltops-github-actions.patch
git apply boltops-github-actions.patch
git add -A -- .gitattributes .github .agents docs/ci scripts/ci ci_script build_wheel.sh .gitlab-ci.yml
git commit -m 'ci: add GitHub Actions build test and nightly workflows'
git push -u fork ci/github-actions-migration
```

创建 PR 到 release/0.1.0，检查两组构建、正确性和性能作业，以及 ci-result 和已有 Quality Gate。下载 artifact 核对 wheel、JUnit 与性能报告。两组测试均需在 BW1100 上实际验证。

由于需要真实内网 runner、编译器和 GPU，本地只能验证 YAML/Actions 表达式、Bash 语法、矩阵、设备选择分支和脚本迁移逻辑；未实际运行 GPU 构建、测试或上传。

## 5. 正式切换

PR 验证通过后，安排同一次切换窗口：

1. 停止接受 GitLab 独立 main 更新，选定 GitHub 为唯一主入口。
2. 维护者停用 GitLab 的发布与 mirror:github（或停用整个 GitLab CI），并等待/取消仍在运行的发布任务。不要同时让两边对同一提交发布。
3. 合入 GitHub 的迁移 PR；release/0.1.0 push 只构建并发布，不重复 PR 的测试。
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

发布直接依赖本次 push 的各组构建及测试结果，不查询合入前的 PR 历史。即使 PR 测试未通过就合入，也必须在合入后对应 DTK 的构建、正确性及性能测试全部通过才发布该组包。手动运行仍只构建，不测试、不发布。
