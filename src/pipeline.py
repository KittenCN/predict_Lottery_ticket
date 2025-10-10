# -*- coding: utf-8 -*-
"""
训练与预测流程封装。

暴露的核心函数：
- train_lottery_models：基于历史数据训练模型并写入本地；
- load_trained_models：从磁盘加载已训练模型；
- predict_next_draw：使用最新窗口数据给出预测结果。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger

from .config import (
    DATA_FILE_NAME,
    MODEL_METADATA_FILE,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
    get_lottery_config,
)
from .data_fetcher import load_history
from .modeling import build_models_for_lottery
from .preprocessing import ComponentDataset, prepare_training_arrays, train_validation_split


@dataclass
class ComponentTrainingSummary:
    train_samples: int
    val_samples: int
    best_val_loss: Optional[float]
    best_val_metric: Optional[float]
    epochs_trained: int


@dataclass
class TrainingSummary:
    code: str
    name: str
    window_size: int
    trained_on_issues: Tuple[str, str]
    components: Dict[str, ComponentTrainingSummary]
    timestamp: str


def _ensure_enough_samples(dataset: ComponentDataset, window_size: int, name: str) -> None:
    if dataset.features.shape[0] == 0:
        raise ValueError(
            f"{name} 可用数据不足，窗口大小 {window_size} 生成的样本数为 0，请增加历史期数或减小窗口。"
        )


def _build_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        buffer = min(len(features), max(batch_size * 4, 256))
        ds = ds.shuffle(buffer)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _denormalize(pred: np.ndarray, spec_classes: int) -> np.ndarray:
    if pred.min() < 0:
        raise ValueError("预测结果包含负数，可能是模型输出异常")
    return pred + 1


def _get_latest_window(arr: np.ndarray, window_size: int) -> np.ndarray:
    if arr.shape[0] < window_size:
        raise ValueError(f"历史数据不足，无法获取 {window_size} 条窗口序列")
    return arr[-window_size:]


def train_lottery_models(
    code: str,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    red_epochs: Optional[int] = None,
    blue_epochs: Optional[int] = None,
    validation_ratio: float = 0.15,
) -> TrainingSummary:
    """训练指定彩票模型，并返回训练摘要。"""

    ensure_runtime_directories()
    cfg: LotteryModelConfig = get_lottery_config(code)
    df = load_history(cfg.code)
    window = window_size or cfg.default_window
    arrays = prepare_training_arrays(df, cfg, window)
    summary_components: Dict[str, ComponentTrainingSummary] = {}

    models = build_models_for_lottery(cfg, window)
    save_dir = PATHS["model"] / cfg.code / f"window_{window}"
    save_dir.mkdir(parents=True, exist_ok=True)

    first_issue = str(df["期数"].min())
    last_issue = str(df["期数"].max())

    for component, model in models.items():
        dataset = arrays[component]
        _ensure_enough_samples(dataset, window, f"{cfg.name}-{component}")
        (x_train, y_train), (x_val, y_val) = train_validation_split(
            dataset.features, dataset.labels, validation_ratio=validation_ratio
        )
        effective_batch = max(1, min(batch_size or cfg.default_batch_size, x_train.shape[0]))
        train_ds = _build_tf_dataset(x_train, y_train, effective_batch, shuffle=True)
        val_ds = None
        if x_val.shape[0] > 0:
            val_ds = _build_tf_dataset(x_val, y_val, effective_batch, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if val_ds is None:
            callbacks = []

        epochs = red_epochs if component == "red" else blue_epochs
        if epochs is None:
            epochs = cfg.default_red_epochs if component == "red" else cfg.default_blue_epochs
        epochs = max(1, epochs)

        logger.info(
            "训练模型 {}-{}: 样本={}，验证集={}，窗口={}，批大小={}，轮数={}",
            cfg.code,
            component,
            dataset.features.shape[0],
            x_val.shape[0],
            window,
            effective_batch,
            epochs,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
        )

        model_path = save_dir / f"{component}.keras"
        model.save(model_path, overwrite=True)
        logger.success("模型已保存至 {}", model_path)

        best_loss = min(history.history.get("val_loss", history.history.get("loss", [None])))
        metric_key = None
        for candidate in ("val_accuracy", "val_sparse_categorical_accuracy", "accuracy"):
            if candidate in history.history:
                metric_key = candidate
                break
        best_metric = None
        if metric_key is not None:
            best_metric = max(history.history[metric_key])
        summary_components[component] = ComponentTrainingSummary(
            train_samples=int(x_train.shape[0]),
            val_samples=int(x_val.shape[0]),
            best_val_loss=float(best_loss) if best_loss is not None else None,
            best_val_metric=float(best_metric) if best_metric is not None else None,
            epochs_trained=len(history.history.get("loss", [])),
        )

    metadata = TrainingSummary(
        code=cfg.code,
        name=cfg.name,
        window_size=window,
        trained_on_issues=(first_issue, last_issue),
        components=summary_components,
        timestamp=datetime.utcnow().isoformat(),
    )
    metadata_path = save_dir / MODEL_METADATA_FILE
    metadata_path.write_text(
        json.dumps(
            {
                **asdict(metadata),
                "components": {key: asdict(value) for key, value in summary_components.items()},
                "data_file": str(PATHS["data"] / cfg.code / DATA_FILE_NAME),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.success("训练摘要已写入 {}", metadata_path)
    return metadata


def load_trained_models(code: str, window_size: Optional[int] = None) -> Dict[str, tf.keras.Model]:
    """从磁盘加载训练好的模型。"""

    cfg = get_lottery_config(code)
    window = window_size or cfg.default_window
    directory = PATHS["model"] / cfg.code / f"window_{window}"
    if not directory.exists():
        raise FileNotFoundError(f"未找到已训练的模型目录: {directory}")

    models: Dict[str, tf.keras.Model] = {}

    for component in ("red", "blue"):
        model_path = directory / f"{component}.keras"
        if model_path.exists():
            models[component] = tf.keras.models.load_model(
                model_path,
                compile=True,
                safe_mode=False,
            )
            logger.info("载入模型 {}", model_path)
    if not models:
        raise FileNotFoundError(f"{directory} 下未找到 red/blue 模型文件")
    return models


def predict_next_draw(
    code: str,
    window_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """使用最新模型预测下一期开奖号码。"""

    cfg = get_lottery_config(code)
    window = window_size or cfg.default_window
    df = load_history(cfg.code)
    arrays = prepare_training_arrays(df, cfg, window)
    models = load_trained_models(cfg.code, window)

    predictions: Dict[str, np.ndarray] = {}

    # 最新窗口特征取 prepare_training_arrays 中的原始数组最后 window 条
    red_dataset = arrays["red"]
    latest_features = _get_latest_window(red_dataset.features, 1).reshape(1, window, cfg.red.sequence_len)
    red_model = models["red"]
    red_pred = red_model.predict(latest_features, verbose=0)
    predictions["red"] = np.argmax(red_pred, axis=-1).squeeze(axis=0).astype(int)

    if cfg.blue and "blue" in models:
        blue_dataset = arrays["blue"]
        latest_blue = _get_latest_window(blue_dataset.features, 1).reshape(1, window, cfg.blue.sequence_len)
        blue_pred_raw = models["blue"].predict(latest_blue, verbose=0)
        predictions["blue"] = np.argmax(blue_pred_raw, axis=-1).squeeze(axis=0).astype(int)

    # 将预测结果转换回原始编号（0-based -> 1-based）
    if red_dataset.needs_offset:
        predictions["red"] = _denormalize(predictions["red"], cfg.red.num_classes)
    if "blue" in predictions:
        blue_dataset = arrays["blue"]
        if blue_dataset.needs_offset:
            predictions["blue"] = _denormalize(predictions["blue"], cfg.blue.num_classes)
    return predictions


__all__ = ["train_lottery_models", "load_trained_models", "predict_next_draw", "TrainingSummary"]
