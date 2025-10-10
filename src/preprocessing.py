from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import LotteryModelConfig, SequenceModelSpec


@dataclass(frozen=True)
class ComponentDataset:
    """封装单个号码序列的特征、标签与偏移信息。"""

    features: np.ndarray
    labels: np.ndarray
    needs_offset: bool


def _select_number_columns(df: pd.DataFrame, prefix: str, count: int) -> pd.DataFrame:
    columns = [f"{prefix}_{idx + 1}" for idx in range(count)]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺失列: {missing}")
    return df[columns]


def _needs_offset(values: np.ndarray, spec: SequenceModelSpec) -> bool:
    if values.size == 0:
        return False
    arr = values.astype(int)
    return arr.min() >= 1 and arr.max() <= spec.num_classes


def _to_zero_based(values: np.ndarray, shift: bool) -> np.ndarray:
    arr = values.astype(np.int32)
    if shift:
        arr = arr - 1
    return arr


def _build_windows(array: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for idx in range(window_size, len(array)):
        features.append(array[idx - window_size : idx])
        labels.append(array[idx])
    return np.asarray(features, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def prepare_training_arrays(
    df: pd.DataFrame,
    config: LotteryModelConfig,
    window_size: int,
) -> Dict[str, ComponentDataset]:
    """基于历史数据构建训练所需的 NumPy 数组。"""

    sorted_df = df.sort_values("期数").reset_index(drop=True)
    result: Dict[str, ComponentDataset] = {}

    red_df = _select_number_columns(sorted_df, "红球", config.red.sequence_len)
    red_shift = _needs_offset(red_df.values, config.red)
    red_array = _to_zero_based(red_df.values, red_shift)
    red_x, red_y = _build_windows(red_array, window_size)
    result["red"] = ComponentDataset(red_x, red_y, red_shift)

    if config.blue:
        blue_df = _select_number_columns(sorted_df, "蓝球", config.blue.sequence_len)
        blue_shift = _needs_offset(blue_df.values, config.blue)
        blue_array = _to_zero_based(blue_df.values, blue_shift)
        blue_x, blue_y = _build_windows(blue_array, window_size)
        result["blue"] = ComponentDataset(blue_x, blue_y, blue_shift)

    return result


def train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_ratio: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """将窗口数据按比例划分训练与验证集合。"""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须介于 (0, 1) 之间")
    total = x.shape[0]
    split_index = max(1, int(total * (1 - validation_ratio)))
    if split_index >= total:
        split_index = total - 1
    if split_index <= 0:
        split_index = total // 2 or 1
    return (x[:split_index], y[:split_index]), (x[split_index:], y[split_index:])


__all__ = ["prepare_training_arrays", "train_validation_split", "ComponentDataset"]
