# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from src.config import LotteryModelConfig, SequenceModelSpec
from src.preprocessing import ComponentDataset, prepare_training_arrays, train_validation_split


@pytest.fixture
def tiny_config() -> LotteryModelConfig:
    return LotteryModelConfig(
        code="demo",
        name="示例彩票",
        red=SequenceModelSpec(
            sequence_len=3,
            num_classes=5,
            embedding_dim=4,
            hidden_units=(8,),
        ),
        blue=SequenceModelSpec(
            sequence_len=1,
            num_classes=3,
            embedding_dim=2,
            hidden_units=(4,),
        ),
        default_window=2,
        default_batch_size=4,
    )


def test_prepare_training_arrays_handles_offset(tiny_config):
    df = pd.DataFrame(
        {
            "期数": ["1", "2", "3", "4"],
            "红球_1": [1, 2, 3, 4],
            "红球_2": [2, 3, 4, 5],
            "红球_3": [3, 4, 5, 1],
            "蓝球_1": [0, 1, 2, 0],
        }
    )
    datasets = prepare_training_arrays(df, tiny_config, window_size=2)
    assert set(datasets.keys()) == {"red", "blue"}
    red_ds: ComponentDataset = datasets["red"]
    blue_ds: ComponentDataset = datasets["blue"]
    assert red_ds.features.shape == (2, 2, 3)
    assert red_ds.needs_offset is True
    assert blue_ds.needs_offset is False
    assert (red_ds.features.min(), red_ds.features.max()) == (0, 4)


def test_train_validation_split_minimum_samples():
    x = [[1], [2], [3], [4]]
    y = [[10], [20], [30], [40]]
    (x_train, y_train), (x_val, y_val) = train_validation_split(
        x,
        y,
        validation_ratio=0.25,
    )
    assert len(x_train) == 3
    assert len(x_val) == 1
