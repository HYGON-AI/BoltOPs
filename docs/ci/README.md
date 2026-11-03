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
  → .github/scripts/performance.sh（PR/手动运行）
    → scripts/ci/optest_to_junit.py
  → .github/scripts/publish.sh（main push）
```

`.github/scripts/common.sh` 由需要 DTK 的 CI 脚本加载，负责公共初始化。`quality-gate.yml` 是独立的 PR 质量检查工作流。GitLab 保留在根目录的 `.gitlab-ci.yml`，使用同一个构建入口和报告工具。

## 命名与维护

- 新增 GitHub 工作流和 CI 脚本使用小写短横线命名，如 `setup-env.sh`。
- `_build-test-wheel.yml` 表示供其他工作流调用的配置；`_` 是约定，不是 GitHub 功能开关。
- 统一构建脚本保留既定名称 `.github/scripts/_build_hcu_wheel.sh`。
- CI 产物不放在脚本目录；wheel 使用 `output/`，性能日志使用 `optest_artifacts/`，按工作流上传为 artifact。
- 移动文件后，同步更新 GitHub/GitLab 引用和文档，不复制第二套实现。

学习、配置与切换步骤见 [GitHub Actions 迁移说明](github-actions-migration.md)。
