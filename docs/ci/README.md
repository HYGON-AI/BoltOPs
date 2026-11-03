# CI 目录与文件职责

| 目录或文件 | 放置规则 |
|---|---|
| `.github/workflows/` | 只放 GitHub Actions 工作流 YAML，不放 shell 脚本；可复用工作流采用 `_` 前缀 |
| `.github/scripts/` | GitHub CI 专用的环境准备、构建包装、测试及发布脚本 |
| `scripts/ci/` | GitLab、GitHub 等平台共用的 CI 辅助工具 |
| `scripts/` 的其他目录 | 调优、benchmark 等项目通用工具 |
| `.github/scripts/_build_hcu_wheel.sh` | 本地开发与各 CI 共用的环境检查和 wheel 构建入口 |
| 根目录 `build_wheel.sh` | 旧命令兼容入口，转发到统一构建脚本 |
| `.agents/skills/build-hcu-wheel/SKILL.md` | 智能助手使用的构建操作说明 |
| `docs/ci/` | CI 架构、配置和迁移说明，不放执行脚本或构建产物 |
| `test/` | 正确性和性能测试用例 |

## 调用关系

```text
.github/workflows/ci.yml
  → .github/workflows/_build-test-wheel.yml
    → .github/scripts/setup-env.sh
    → .github/scripts/_build_hcu_wheel.sh
    → .github/scripts/test-wheel.sh（按事件决定）
  → .github/workflows/_performance.yml（PR/手动运行，Torch2.11.0/2.13.0）
    → .github/scripts/setup-env.sh
    → .github/scripts/performance.sh
      → scripts/ci/optest_to_junit.py
  → .github/workflows/_publish.yml（main push）
    → .github/scripts/publish.sh
```

`.github/scripts/common.sh` 由需要 DTK 的 CI 脚本加载，负责公共初始化。`quality-gate.yml` 是独立的 PR 质量检查工作流。GitLab 保留在根目录的 `.gitlab-ci.yml`，使用同一个构建入口和报告工具。

`_performance.yml` 分别接收 DTK26.10、Python3.12、Torch2.11.0/2.13.0，下载各自构建产物后运行性能测试，不额外构建 wheel。报告制品名称包含 DTK、Python 和 Torch 版本，避免两组上传冲突。实际硬件兼容性需要在 BW1100 上验证。

## 命名与维护

- 新增 GitHub 工作流和 CI 脚本使用小写短横线命名，如 `setup-env.sh`。
- `_build-test-wheel.yml`、`_performance.yml` 和 `_publish.yml` 表示供其他工作流调用的配置；`_` 是约定，不是 GitHub 功能开关。
- 统一构建脚本保留既定名称 `.github/scripts/_build_hcu_wheel.sh`。
- CI 产物不放在脚本目录；wheel 使用 `output/`，性能日志使用 `optest_artifacts/`，按工作流上传为 artifact。
- 移动文件后，同步更新 GitHub/GitLab 引用和文档，不复制第二套实现。

## 版本配置

测试版本在 `ci.yml` 的 `build2610.strategy.matrix.torch_version` 中维护，当前仅包含 `2.11.0` 和 `2.13.0`。两组均使用 DTK26.10、Python3.12、NumPy1.26.4；版本变更通过 PR 评审。

PR 和手动触发时，两组构建均执行正确性测试，之后两组均运行性能测试；main push 时仅构建，两组都成功后由一个 `publish-2610` 任务发布全部 DTK2610 wheel。`ci-result` 要求两组构建成功；PR/手动触发还要求两组性能均成功，push 则要求性能跳过。构建和性能矩阵通过 YAML 锚点 `&torch_versions` 与别名 `*torch_versions` 复用同一个版本列表。

升级版本时，检查对应的 DTK/Python 容器及 HCU Torch/Triton/TileLang 包是否可用。不同 DTK 若需要不同版本，应拆开配置，不能只为减少重复而强行共用。发布任务的 Python3.12 容器只用于上传 wheel，不决定 wheel 的目标 Python 版本。

发布任务调用 `_publish.yml`；在 `ci.yml` 中通过 `with.dtk_ver` 和 `with.dtk_pkg_name` 配置目标版本及真实 DTK 包名。发布密码通过 `secrets.PYPI_PASSWORD` 显式传给复用工作流，不放在 `with` 中。DTK 包名必须与发布服务实际使用的名称一致，不能仅通过替换版本号生成。

学习、配置与切换步骤见 [GitHub Actions 迁移说明](github-actions-migration.md)。
