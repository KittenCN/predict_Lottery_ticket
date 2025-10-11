# Changelog

All notable changes to this project should be documented in this file.


## [Unreleased]
- ❌ **KL8（快乐8）相关内容迁移**：所有KL8相关源码、数据、分析与文档已迁移至独立项目 [KL8-Lottery-Analyzer](https://github.com/KittenCN/kl8-lottery-analyzer)。本仓库不再包含KL8相关功能。
- 🐛 **修复红球预测重复问题**：修正了双色球等玩法预测结果中红球可能重复的bug，现已确保每注红球号码唯一（见 `src/pipeline.py` 2024-06 修正）。
- 📚 **文档同步**：更新README、ASSUMPTIONS、决策记录等文档，反映KL8迁移和预测逻辑修正。
- Add `src.bootstrap.py` compatibility shim to map deprecated TF ragged symbol and avoid import-time crashes.
- Create `requirements.lock.txt` (portable pip lock) and provide guidance for `environment.yml` (conda) to reproduce binary dependencies.
- Uninstall standalone `keras` in local environment and prefer `tf.keras`.
- Update `scripts/train.py` to import `src.bootstrap` early.
- Update `README.md`, add `ASSUMPTIONS.md` and `docs/` guidance for environment reproducibility and verification.
# 更新日志 (Changelog)

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [3.0.0] - 2024-12-20

### 重大变更 (Breaking Changes)
- 🧠 **全面升级 TensorFlow**：迁移至 TensorFlow 2.15.1 + Keras 2.15，弃用所有 `tf.compat.v1` API 并移除对 tensorflow-addons 的依赖
- 💾 **模型格式调整**：训练产物改为 `.keras`（SavedModel v3），旧版 `.ckpt` 文件不再兼容
- 🧱 **训练/预测流程重写**：新增 `data_fetcher`、`preprocessing`、`pipeline` 等模块，原有 Session 流程下线

### 新增功能 (Added)
- ✨ **数据抓取客户端**：提供带重试、白名单校验的 `LotteryHttpClient`
- 🧪 **单元测试**：新增配置、预处理、模型与训练管线的覆盖，并引入 `pytest --cov`
- 🛠️ **运维文档**：补充 `docs/decision_record.md`、`docs/ops.md` 以及最新 API 说明

### 改进优化 (Changed)
- 📦 **依赖锁定**：更新 `requirements.txt`（TensorFlow 2.15.1、pandas 2.2、ruff/mypy 等）
- 🧾 **Makefile 流程**：`lint` 使用 ruff + mypy，`test` 默认输出覆盖率，`train/predict` 命令参数同步新版脚本
- 📚 **README/示例**：同步最新命令、模型架构说明与注意事项

- 🧹 **旧版测试/脚本**：删除 `tf.compat` 相关代码与失效测试（多线程进度条样例等），替换为新版覆盖
- 🗑️ **CRF 依赖**：移除对 `tensorflow-addons` 与 CRF 层的使用，统一改为纯 LSTM + Softmax 模型

---

## [2.0.0] - 2024-12-19

### 重大变更 (Breaking Changes)
- 🏗️ **重构项目结构**: 按照 AGENTS.md 规范重新组织代码结构
- 📁 **目录结构变更**: 
  - 核心代码移至 `src/` 目录
  - 脚本文件移至 `scripts/` 目录  
  - 测试文件移至 `tests/` 目录
  - 配置文件移至 `config/` 目录
- 🔧 **导入路径更新**: 所有导入语句已更新以适配新结构

### 新增功能 (Added)
- ✨ **项目模板化**: 添加标准化的项目结构和开发工具
- 🛠️ **Makefile支持**: 提供一键式任务命令
- 🐳 **Docker支持**: 添加 Dockerfile 和 docker-compose.yml
- 🔄 **CI/CD流水线**: 添加 GitHub Actions 自动化流水线
- 📝 **完整文档**: 添加 README、ASSUMPTIONS、CHANGELOG 等文档
- 🧪 **单元测试**: 添加基础测试框架和测试用例
- 📊 **示例程序**: 添加快速开始和数据分析示例
- ⚙️ **配置模板**: 提供 .env.example 和 config.yaml 模板

### 改进优化 (Changed)
- 📦 **依赖管理**: 更新 requirements.txt，添加版本约束
- 🔧 **配置管理**: 改进配置文件结构和管理方式
- 📚 **代码组织**: 按功能模块重新组织代码
- 🎯 **错误处理**: 改进异常处理和日志记录

### 修复问题 (Fixed)
- 🐛 **导入错误**: 修复模块导入路径问题
- 🔗 **依赖缺失**: 添加缺失的依赖包

### 文档更新 (Documentation)
- 📖 **README重写**: 完全重写项目说明文档
- 📋 **架构文档**: 添加项目架构和技术选择说明  
- 🤝 **贡献指南**: 添加代码贡献和开发规范
- ⚖️ **许可协议**: 明确项目许可和免责声明

---

## [1.x.x] - 历史版本 (Legacy)

### [1.3.1] - 2023-10-31
#### 新增
- 增加 kl8 plus 系列文件，支持多线程数据处理
- 修改 kl8_running，提升数据处理速度

#### 注意
- 多线程处理对CPU要求较高，请根据硬件配置谨慎使用

### [1.3.0] - 2023-09-03  
#### 新增
- 增加两个 kl8_ 开头的文件用于数据预处理和获奖计算
- 新增数据分析和预测验证功能

### [1.2.0] - 2023-03-27
#### 新增
- 增加对七星彩（qxc）的支持
- 增加对福彩3D（sd）的支持
- 完善数据获取和模型训练流程

### [1.1.0] - 2023-03-22
#### 新增  
- 增加执行参数开关
- red_epochs、blue_epochs、batch_size 参数支持 -1 值读取配置文件
- 修改参数默认值为 -1

#### 改进
- 优化参数配置管理
- 改进命令行参数处理

### [1.0.0] - 2023-01-01
#### 初始版本
- 🎯 支持双色球、大乐透、排列三、快乐8预测
- 🧠 基于 LSTM + CRF 的深度学习模型  
- 📊 自动数据获取和处理
- 🔧 CPU/GPU 自动切换支持
- 📈 基础数据分析功能

---

## 版本说明

### 版本号格式
采用语义化版本号 `MAJOR.MINOR.PATCH`：
- **MAJOR**: 重大变更，可能不向后兼容
- **MINOR**: 新功能添加，向后兼容  
- **PATCH**: 问题修复，向后兼容

### 变更类型
- 🏗️ **重大变更**: 不向后兼容的变更
- ✨ **新增功能**: 新功能或特性
- 🔧 **改进优化**: 现有功能的改进
- 🐛 **修复问题**: 错误修复
- 📚 **文档更新**: 文档相关变更
- 🔒 **安全修复**: 安全相关修复
- ⚠️ **废弃功能**: 计划废弃的功能
- ❌ **移除功能**: 已移除的功能

### 发布计划
- 主要版本：根据功能发展情况决定
- 次要版本：新功能稳定后发布
- 补丁版本：重要bug修复后及时发布

### 兼容性政策
- 主版本升级可能包含不兼容变更
- 次版本升级保持向后兼容
- 补丁版本仅包含bug修复

---

**说明**: 1.x.x 版本的详细记录可能不完整，从 2.0.0 开始将严格遵循此变更日志格式。
