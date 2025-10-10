# -*- coding: utf-8 -*-
"""
模型构建模块。

提供基于 TensorFlow 2.15.1 的多层 LSTM 序列模型，并针对红球/蓝球输出
逐位置的类别概率。
"""

from __future__ import annotations

from typing import Dict

import importlib
import warnings

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise ImportError(
        "TensorFlow import failed. Ensure TensorFlow (e.g. tensorflow or tensorflow-intel) is installed."
    ) from exc

from loguru import logger


# Ensure tf.keras is available; prefer bundled tf.keras over standalone `keras` package.
if not hasattr(tf, "keras"):
    # Some environments may have an incomplete TensorFlow install where `tf.keras` is missing.
    # Try to import standalone `keras` as a best-effort fallback, but do not fail hard here;
    # instead emit a clear warning so downstream code that expects tf.keras will see a
    # friendlier message.
    try:
        import keras  # type: ignore

        warnings.warn(
            "Standalone `keras` was found but `tf.keras` is missing. Using standalone keras may cause incompatibilities.",
            UserWarning,
        )
    except Exception:
        raise ImportError(
            "Keras cannot be imported. Check that it is installed or that your TensorFlow installation is complete."
        )

from src.config import LotteryModelConfig, SequenceModelSpec


def _time_distributed_lstm(
    inputs: tf.Tensor,
    units: int,
    name: str,
) -> tf.Tensor:
    """对每个球位独立应用 LSTM，提取窗口维度特征。"""

    layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.LSTM(units, return_sequences=False, name=f"{name}_inner"),
        name=name,
    )
    return layer(inputs)


def build_sequence_model(
    spec: SequenceModelSpec,
    window_size: int,
    learning_rate: float,
    name: str,
) -> tf.keras.Model:
    """根据给定规格构建序列模型。"""

    inputs = tf.keras.layers.Input(
        shape=(window_size, spec.sequence_len),
        dtype=tf.int32,
        name=f"{name}_input",
    )
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=spec.num_classes,
        output_dim=spec.embedding_dim,
        embeddings_initializer="he_normal",
        name=f"{name}_embedding",
    )
    embedded = embedding_layer(inputs)  # (batch, window, seq_len, embed_dim)
    # 将球位与时间维度交换，便于对每个球做 LSTM
    per_ball_sequence = tf.transpose(embedded, perm=(0, 2, 1, 3), name=f"{name}_permute")
    per_ball_encoded = _time_distributed_lstm(
        per_ball_sequence,
        units=int(spec.hidden_units[0]),
        name=f"{name}_per_ball_lstm",
    )

    x = per_ball_encoded
    for layer_idx, units in enumerate(spec.hidden_units[1:], start=1):
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=spec.dropout,
            recurrent_dropout=0.0,
            name=f"{name}_global_lstm_{layer_idx}",
        )(x)

    if len(spec.hidden_units) == 1:
        x = tf.keras.layers.LSTM(
            spec.hidden_units[0],
            return_sequences=True,
            dropout=spec.dropout,
            name=f"{name}_global_lstm",
        )(x)

    x = tf.keras.layers.Dropout(spec.dropout, name=f"{name}_dropout")(x)
    logits = tf.keras.layers.Dense(
        spec.num_classes,
        name=f"{name}_logits",
    )(x)
    output = tf.keras.layers.Activation("softmax", name=f"{name}_softmax")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=output, name=f"{name}_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    logger.debug("构建模型 {}：窗口={}，序列长={}，类别数={}", name, window_size, spec.sequence_len, spec.num_classes)
    return model


def build_models_for_lottery(
    config: LotteryModelConfig,
    window_size: int,
) -> Dict[str, tf.keras.Model]:
    """构建指定彩票的红/蓝球模型。"""

    models: Dict[str, tf.keras.Model] = {
        "red": build_sequence_model(config.red, window_size, config.learning_rate, f"{config.code}_red"),
    }
    if config.blue:
        models["blue"] = build_sequence_model(
            config.blue,
            window_size,
            config.learning_rate,
            f"{config.code}_blue",
        )
    return models


__all__ = ["build_models_for_lottery", "build_sequence_model"]
