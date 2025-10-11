# API 接口文档（TensorFlow 2.15 版）

## 概述

新版系统围绕以下核心模块组织：

- `src.config`：集中管理路径、彩票玩法配置、超参数；
- `src.data_fetcher`：负责网络访问、历史数据下载与本地加载；
- `src.preprocessing`：提供窗口化数据构建与训练集拆分；
- `src.modeling`：基于 Keras 的多层 LSTM 模型构建；
- `src.pipeline`：训练、保存、载入模型以及生成预测；
- `src.common`：面向脚本层的高层封装。

文档中的示例均假设已激活 `python311` Conda 环境，并位于项目根目录。

---

## 1. `src.common` 模块


### `get_data_run(name: str, start_issue: int | None = None, end_issue: int | None = None) -> None`
下载指定彩票的历史开奖数据，并保存到 `data/<name>/data.csv`。

- `name`：彩票代码，如 `ssq`、`dlt` 等（KL8已迁移至独立项目）；
- `start_issue` / `end_issue`：可选的期号范围，缺省则下载全量。

```python
from src.common import get_data_run
get_data_run("ssq", start_issue=2023001, end_issue=2023350)
```

### `get_current_number(name: str) -> str`
返回指定彩票最新一期的期号。

```python
from src.common import get_current_number
print(get_current_number("ssq"))
```

### `train_pipeline(name: str, window_size: int | None = None, batch_size: int | None = None, red_epochs: int | None = None, blue_epochs: int | None = None) -> TrainingSummary`
封装后的训练入口，内部调用 `pipeline.train_lottery_models`。返回 `TrainingSummary` 数据类，包含训练轮次、样本量和元数据路径。

```python
from src.common import train_pipeline
summary = train_pipeline("ssq", window_size=5, red_epochs=20, blue_epochs=5)
print(summary.window_size, summary.trained_on_issues)
```


### `predict_latest(name: str, window_size: int | None = None) -> dict[str, list[int]]`
使用最新训练好的模型预测下一期号码，返回 `{"red": [...], "blue": [...]}` 字典。

> 2024-06起，红球预测结果已确保每注号码唯一，彻底避免重复红球。

```python
from src.common import predict_latest
result = predict_latest("ssq", window_size=5)
print(result["red"])
```

---

## 2. `src.config` 模块

### 数据类
- `SequenceModelSpec`：描述序列模型（窗口长度内的单个号码序列）的结构，包括：
  - `sequence_len`、`num_classes`、`embedding_dim`、`hidden_units`、`dropout`。
- `LotteryModelConfig`：定义玩法整体配置，包含红/蓝球 `SequenceModelSpec`、默认窗口、批大小、训练轮数与学习率。

### 常量
- `PATHS`：运行目录（`data`、`model`、`predict`、`logs`），可用于构造自定义路径；
- `LOTTERY_CONFIGS`：玩法配置字典；
- `name_path`：保留旧接口兼容的简化映射；
- `DATA_FILE_NAME` / `MODEL_METADATA_FILE`：数据/元数据文件名。

### 函数
- `ensure_runtime_directories() -> None`：创建 PATHS 中定义的目录；
- `get_lottery_config(code: str) -> LotteryModelConfig`：获取玩法配置。

```python
from src.config import get_lottery_config
cfg = get_lottery_config("ssq")
print(cfg.red.sequence_len, cfg.red.num_classes)
```

---

## 3. `src.data_fetcher` 模块

### `LotteryHttpClient`
带重试、超时、白名单校验的 requests 封装，通常无需直接实例化，高层函数会自动创建。

### `download_history(code: str, start: int | None = None, end: int | None = None, use_sequence_order: bool = False, client: LotteryHttpClient | None = None) -> DownloadResult`
下载历史数据并写入 CSV，同时生成 `download_meta.json`。

### `get_current_issue(code: str, client: LotteryHttpClient | None = None) -> str`
返回最新期号（供 `common.get_current_number` 调用）。

### `load_history(code: str) -> pandas.DataFrame`
读取 `data/<code>/data.csv`到 DataFrame。若文件不存在则抛出异常。

---

## 4. `src.preprocessing` 模块

### `ComponentDataset`
不可变数据类，包含 `features`、`labels`、`needs_offset`。`needs_offset` 表示原始数据是否以 1 为起点（例如双色球红球需要在预测时 +1）。

### `prepare_training_arrays(df: pandas.DataFrame, config: LotteryModelConfig, window_size: int) -> dict[str, ComponentDataset]`
将 DataFrame 转换为滑动窗口序列。返回 `{"red": ComponentDataset, "blue": ...}`。

### `train_validation_split(x, y, validation_ratio=0.1)`
拆分训练/验证集，保证最少保留一个样本在验证集中。

```python
from src.preprocessing import prepare_training_arrays
datasets = prepare_training_arrays(df, cfg, window_size=5)
red_ds = datasets["red"]
print(red_ds.features.shape, red_ds.needs_offset)
```

---

## 5. `src.modeling` 模块

### `build_sequence_model(spec: SequenceModelSpec, window_size: int, learning_rate: float, name: str) -> tf.keras.Model`
构建并编译单个 LSTM 序列模型，输出逐位置的分类概率。

### `build_models_for_lottery(config: LotteryModelConfig, window_size: int) -> dict[str, tf.keras.Model]`
根据玩法配置同时构建红/蓝球模型，默认返回 `{"red": model, "blue": model}`（若玩法无蓝球则缺失）。

---

## 6. `src.pipeline` 模块

### `train_lottery_models(code: str, window_size: int | None = None, batch_size: int | None = None, red_epochs: int | None = None, blue_epochs: int | None = None, validation_ratio: float = 0.15) -> TrainingSummary`
训练并保存模型，核心流程：
1. 调用 `load_history` 读取数据；
2. `prepare_training_arrays` 构建窗口；
3. 使用 `build_models_for_lottery` 建模；
4. 训练并保存 `.keras` 模型与 `metadata.json`。

### `load_trained_models(code: str, window_size: int | None = None) -> dict[str, tf.keras.Model]`
从 `model/<code>/window_<n>/` 中载入模型，返回模型字典。

### `predict_next_draw(code: str, window_size: int | None = None) -> dict[str, numpy.ndarray]`
利用训练好的模型预测下一期号码。内部自动读取最新窗口，并根据 `needs_offset` 转换回真实号码。

### `TrainingSummary`
训练摘要数据类，字段包括：
- `code` / `name`：玩法；
- `window_size`：训练窗口；
- `trained_on_issues`：最早与最新的期号；
- `components`：红/蓝球样本及指标摘要；
- `timestamp`：UTC 时间戳。

---

## 7. 示例：端到端调用

```python
from src.common import get_data_run, train_pipeline, predict_latest

# 1. 下载数据
get_data_run("ssq", start_issue=2023001)

# 2. 训练模型
summary = train_pipeline("ssq", window_size=5, red_epochs=30, blue_epochs=10)
print(f"模型存储窗口: {summary.window_size}")

# 3. 预测
prediction = predict_latest("ssq", window_size=5)
print("红球预测:", prediction["red"])
```

---

更多细节与架构权衡请参考 `docs/architecture.md` 与 `docs/decision_record.md`。如需扩展新玩法，可在 `src/config.py` 中添加新的 `LotteryModelConfig` 并重用上述流程。***
