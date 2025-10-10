# 彩票AI预测系统 Makefile
# 提供一键任务命令

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
PYTHON := python
PIP := pip
VENV := python311
SRC_DIR := src
SCRIPTS_DIR := scripts
TESTS_DIR := tests
EXAMPLES_DIR := examples

# 帮助信息
help: ## 显示帮助信息
	@echo "彩票AI预测系统 - 可用命令："
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# 环境设置
setup: ## 安装依赖并初始化环境
	@echo "正在设置环境..."
	@if not exist conda.exe (echo "请先安装 Anaconda 或 Miniconda") else (echo "Conda 已安装")
	@echo "激活 python311 环境..."
	@conda activate $(VENV) && $(PIP) install -r requirements.txt
	@echo "创建必要的目录..."
	@if not exist data mkdir data
	@if not exist model mkdir model
	@if not exist predict mkdir predict
	@if not exist logs mkdir logs
	@echo "环境设置完成!"

# 代码格式化
fmt: ## 代码格式化
	@echo "正在格式化代码..."
	@conda activate $(VENV) && $(PYTHON) -m black $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)
	@echo "代码格式化完成!"

# 静态代码检查
lint: ## 静态检查
	@echo "正在进行静态代码检查..."
	@conda activate $(VENV) && ruff check $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)
	@conda activate $(VENV) && mypy $(SRC_DIR) $(SCRIPTS_DIR)
	@echo "静态检查完成!"

# 运行测试
test: ## 运行测试
	@echo "正在运行测试..."
	@conda activate $(VENV) && $(PYTHON) -m pytest $(TESTS_DIR) -v --tb=short --cov=src --cov-report=term-missing
	@echo "测试完成!"

# 运行示例
run: ## 启动应用或示例
	@echo "运行快速开始示例..."
	@conda activate $(VENV) && $(PYTHON) $(EXAMPLES_DIR)/quick_start.py

# 构建产物
build: ## 构建产物
	@echo "正在构建项目..."
	@if not exist dist mkdir dist
	@echo "项目构建完成!"

# 本地CI流程
ci: lint test build ## 本地模拟 CI：lint + test + build
	@echo "CI 流程完成!"

# 数据获取
get-data: ## 获取彩票数据
	@echo "获取双色球数据..."
	@conda activate $(VENV) && $(PYTHON) $(SCRIPTS_DIR)/get_data.py --name ssq

# 训练模型
train: ## 训练模型
	@echo "训练双色球模型..."
	@conda activate $(VENV) && $(PYTHON) $(SCRIPTS_DIR)/train.py --name ssq --window-size 5 --red-epochs 60

# 预测
predict: ## 运行预测
	@echo "运行双色球预测..."
	@conda activate $(VENV) && $(PYTHON) $(SCRIPTS_DIR)/predict.py --name ssq --window-size 5 --save

# 清理
clean: ## 清理生成的文件
	@echo "清理中..."
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist *.pyc del /q *.pyc
	@if exist dist rmdir /s /q dist
	@echo "清理完成!"

# 深度清理
clean-all: clean ## 深度清理（包括模型和数据）
	@echo "深度清理中..."
	@if exist model rmdir /s /q model
	@if exist predict rmdir /s /q predict
	@echo "深度清理完成!"

# 安装开发依赖
dev-setup: setup ## 安装开发依赖
	@conda activate $(VENV) && $(PIP) install black ruff mypy pytest pytest-cov

.PHONY: help setup fmt lint test run build ci get-data train predict clean clean-all dev-setup
