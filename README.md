# 彩票AI预测系统 v2.0

> **重要声明**：彩票理论上属于完全随机事件，任何一种单一算法都不可能精确预测彩票结果！本项目仅供学习和研究使用，请合理投资，切勿沉迷！

基于深度学习技术的彩票号码预测系统，支持多种彩票类型的数据分析和号码预测。


> 
> [KL8-Lottery-Analyzer (GitHub)](https://github.com/KittenCN/kl8-lottery-analyzer)
>
> 本仓库不再包含KL8数据和分析相关的核心算法、数据处理和分析功能，后续KL8相关更新请关注新项目。

## ✨ 功能特点

- 📊 支持多种彩票：双色球、大乐透、排列三、七星彩、福彩3D
- 🧠 基于多层 LSTM 的序列建模方案
- 🔄 自动数据获取和处理
- 📈 多维度数据分析
- 🐳 Docker 容器化支持
- 🛠️ 完整的开发工具链

> **重要说明：KL8（快乐8）数据和分析功能已于2025年10月独立为新项目维护，原有KL8数据和分析相关源码已全部迁移。请前往新项目仓库获取和使用：
> 
> [KL8-Lottery-Analyzer (GitHub)](https://github.com/KittenCN/kl8-lottery-analyzer)
>
> 本仓库不再包含KL8数据和分析相关的核心算法、数据处理和分析功能，后续KL8相关更新请关注新项目。

> **2024-06 预测逻辑修正说明**：双色球等玩法的红球预测结果已修正为“每注红球号码唯一”，彻底避免重复红球。详见 `src/pipeline.py`。


```bash
# 1. 克隆项目
git clone https://github.com/KittenCN/predict_Lottery_ticket.git
cd predict_Lottery_ticket

# 2. 创建 conda 环境
conda create -n python311 python=3.11
conda activate python311

# 3. 安装依赖（推荐：使用 conda + pip 锁定方案）
#  - `environment.yml` 用于通过 conda 安装二进制依赖（numpy、tensorflow-intel、pytorch 等）
#  - `requirements.lock.txt` 包含可移植的 pip 包精确版本
# 示例（在仓库根目录运行）:
conda env create -f environment.yml
conda activate predict_lottery
python -m pip install -r requirements.lock.txt

# 兼容性 shim
# 项目在启动时会自动加载 `src.bootstrap`，该模块在导入第三方库前
# 为 TensorFlow/Keras 提供必要的兼容映射（例如 RaggedTensorValue），以
# 减少因环境差异导致的导入/弃用问题。

## 📁 项目结构

```

### 2. 数据获取

```bash
# 获取双色球数据
make get-data
# 或手动执行
python scripts/get_data.py --name ssq
```

### 3. 模型训练

```bash
# 训练双色球模型（使用窗口大小 5，红球 60 轮）
make train
# 或手动执行
python scripts/train.py --name ssq --window-size 5 --red-epochs 60
```

### 4. 预测

```bash
```
# 运行预测并保存结果
make predict
# 或手动执行
python scripts/predict.py --name ssq --window-size 5 --save
```
> __有些朋友发消息问我最近（2023.12.03）发生的快8选7中50000倍的可能性，这么说，这个事其实也跟其他朋友问我为啥最近开始研究统计学的应用是同一个原因，因为我早几个月也发现了，纯粹的某些特定的统计学算法，就可以使得快8选7的平均返奖率维持在60%左右，如果再运用热力图，分布律等特殊的策略，还能使得返奖率在一定范围内维持更高。我当时使用这个方法，也获得了一定的收益。当然这个方法是高投入型的，需要长期稳定的高投入，所以不是我想要的算法，也就没有在这里推荐，而是打算作为神经网络的数据预处理算法来用__
---
> __至于这次事情的单注50000倍玩法，不管你信不信，我是不信的。__

## 📁 项目结构

```
predict_Lottery_ticket/
├─ src/                         # 核心源码
│  ├─ __init__.py
│  ├─ common.py                 # 高层接口封装
│  ├─ config.py                 # 配置/超参管理
│  ├─ data_fetcher.py           # 历史数据抓取
│  ├─ preprocessing.py          # 数据预处理
│  ├─ modeling.py               # TensorFlow 模型定义
│  ├─ pipeline.py               # 训练与预测流程
│  ├─ analysis.py               # 数据分析
│  └─ analysis/                 # 分析工具
├─ scripts/                     # 执行脚本
│  ├─ get_data.py              # 数据获取
│  ├─ train.py                 # 模型训练
│  └─ predict.py               # 预测脚本
├─ tests/                       # 单元测试
├─ examples/                    # 使用示例
├─ config/                      # 配置模板
├─ docs/                        # 项目文档
├─ Makefile                     # 一键任务
├─ Dockerfile                   # 容器化
└─ docker-compose.yml           # 服务编排
```


## 🎯 支持的彩票类型

| 彩票类型 | 代码 | 说明 |
|---------|------|------|
| 双色球 | ssq | 6红球+1蓝球 |
| 大乐透 | dlt | 5红球+2蓝球 |
| 排列三 | pls | 3位数字 |
| 七星彩 | qxc | 7位数字 |
| 福彩3D | sd | 3位数字 |

> **KL8（快乐8）玩法已迁移至独立项目 [KL8-Lottery-Analyzer](https://github.com/KittenCN/kl8-lottery-analyzer)**

## 🔧 技术架构

- **深度学习框架**: TensorFlow 2.15.1 + Keras 2.15
- **模型架构**: 多层 LSTM + Softmax 分类
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib
- **日志**: Loguru
- **容器化**: Docker

## 🛠️ Makefile 命令

```bash
make setup      # 安装依赖和初始化环境
make fmt        # 代码格式化
make lint       # ruff + mypy 静态检查
make test       # pytest --cov 运行测试并生成覆盖率
make run        # 运行示例
make build      # 构建项目
make ci         # 完整CI流程
make get-data   # 获取数据
make train      # 训练模型
make predict    # 运行预测
make clean      # 清理文件
```

## 📊 模型参数

主要参数配置在 `src/config.py` 的 `LOTTERY_CONFIGS` 中：

- `windows_size`: 时间窗口大小（默认3）
- `batch_size`: 批处理大小（默认32）
- `red_epochs` / `blue_epochs`: 训练轮数
- `learning_rate`: 学习率（默认 5e-4~8e-4）
- `SequenceModelSpec`: 控制嵌入维度、隐藏层深度与 dropout


## ⚠️ 注意事项

1. **训练要求**: 必须先用 `get_data.py` 下载数据，再进行模型训练
2. **依赖版本**: 默认依赖 TensorFlow 2.15.1（Windows/macOS/Linux 官方 wheel 可用）
3. **模型格式**: 模型保存为 `.keras`（SavedModel），旧版 `.ckpt` 不再兼容
4. **目录结构**: 确保 `data/`, `model/`, `predict/` 目录存在
5. **红球预测唯一性**: 预测结果已确保每注红球号码唯一，避免重复（2024-06 修正）。

## 🐳 Docker 使用

```bash
# 构建镜像
docker build -t lottery-predict .

# 运行容器
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/predict:/app/predict \
  lottery-predict

# 使用 docker-compose
docker-compose up -d
```

## 📝 更新日志

请查看 [CHANGELOG.md](CHANGELOG.md) 了解详细更新记录。

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

项目初始灵感来自 [zepen](https://github.com/zepen/predict_Lottery_ticket) 的作品，在此基础上进行了重构和增强。

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。彩票投资有风险，参与需谨慎。

