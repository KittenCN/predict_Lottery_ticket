# 自动执行报告 | Automation Execution Report

## 需求摘要 | Requirement Summary
- **背景与目标**: 将早期依赖 TensorFlow 1.x/CRF 的实现迁移到 Windows 友好的 TensorFlow 2.15.1 + 纯 Keras LSTM 架构，确保训练、预测及脚本链路在原生 Conda 环境中可运行。
- **核心功能点**:
  1. 重写模型层，移除对 `tensorflow-addons` 的依赖；
  2. 保留数据抓取、预处理、训练与预测管线，输出 `.keras` 模型；
  3. 更新测试、文档与配置，反映新的 LSTM-only 方案。

## 关键假设 | Key Assumptions
- 详见 `ASSUMPTIONS.md`：统一使用 `python311` 环境、TensorFlow 2.15.1，单一数据源仍为 500.com。
- 纯 LSTM 序列模型能够满足当前预测需求，若需要更复杂的标签依赖，可在后续迭代中加入注意力或条件解码。

## 方案概览 | Solution Overview
- **架构与模块**:
  - `src/modeling.py`: 以多层 LSTM + Softmax 生成逐位置概率；
  - `src/pipeline.py`: 训练与推理流程不再加载自定义 CRF 层；
  - `tests/`：补齐针对新模型的单元测试；
  - 文档同步更新 README、架构、API、决策记录、OPS 说明。
- **选型与权衡**:
  - 选用官方 wheel 覆盖的 TensorFlow 2.15.1，舍弃 Windows 上难以安装的 Addons；
  - 模型简化降低了标签间依赖建模能力，但换来跨平台稳定性，可通过未来引入注意力机制补强。

## 实现与自测 | Implementation & Self-testing
- **一键命令**: `make setup && make fmt && make lint && make test`
- **执行要点**:
  1. `requirements.txt` 删除 `tensorflow-addons`；
  2. 重写 `build_sequence_model`，统一采用 LSTM + Softmax；
  3. 调整 pipeline 加载逻辑与预测分支，移除 CRF 专属路径；
  4. 更新/新增测试用例，验证模型训练与推理输出形状。
- **覆盖率**: `make test` 通过 `pytest --cov=src` 仍可统计覆盖率（需在目标环境执行）；
- **主要测试**: `tests/test_modeling.py`、`tests/test_pipeline.py`、`tests/test_preprocessing.py` 等针对配置、数据管线与模型构建的单元测试。

## 风险与后续改进 | Risks & Next Steps
- **已知限制**:
  1. 简化后的 LSTM 未显式建模球位间依赖；
  2. 模型效果需重新验证，旧版 `.ckpt` 无法直接复现；
  3. 数据仍依赖单一站点，需关注页面更新。
- **建议迭代**:
  1. 引入注意力或 Transformer 以增强序列建模；
  2. 评估多数据源聚合策略；
  3. 追加推理指标与可观测性（TensorBoard/Prometheus）。

---

- **执行时间**: 2024-12-20
- **执行模式**: 自动化重构 + 文档更新
- **完成状态**: ✅ 已完成模型替换及文档同步；请在目标环境执行 `make test` 验证
- **下一步**: 依据新模型重新训练各玩法，并在生产环境校验预测表现
