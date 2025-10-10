# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LOTTERY_CONFIGS, LotteryModelConfig, SequenceModelSpec
from src.pipeline import load_trained_models, predict_next_draw, train_lottery_models


def build_tiny_config() -> LotteryModelConfig:
    return LotteryModelConfig(
        code="ssq",
        name="双色球",
        red=SequenceModelSpec(
            sequence_len=6,
            num_classes=33,
            embedding_dim=8,
            hidden_units=(16,),
            dropout=0.1,
        ),
        blue=SequenceModelSpec(
            sequence_len=1,
            num_classes=16,
            embedding_dim=4,
            hidden_units=(8,),
            dropout=0.1,
        ),
        default_window=3,
        default_batch_size=8,
        default_red_epochs=3,
        default_blue_epochs=2,
        learning_rate=1e-3,
    )


def create_fake_dataset(path: Path, rows: int = 12) -> None:
    records = []
    for idx in range(rows):
        issue = f"2024{idx:03d}"
        base = np.arange(1, 7) + idx % 5
        red_numbers = (base % 33) + 1
        blue_number = (idx % 16) + 1
        record = {"期数": issue}
        for i, value in enumerate(red_numbers, start=1):
            record[f"红球_{i}"] = int(value)
        record["蓝球_1"] = int(blue_number)
        records.append(record)
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def test_train_and_predict_pipeline(monkeypatch, tmp_path):
    # 替换 ssq 配置为轻量级版本，保证测试速度
    tiny_cfg = build_tiny_config()
    monkeypatch.setitem(LOTTERY_CONFIGS, "ssq", tiny_cfg)
    data_path = tmp_path / "data" / "ssq" / "data.csv"
    create_fake_dataset(data_path, rows=12)

    summary = train_lottery_models(
        code="ssq",
        window_size=3,
        batch_size=4,
        red_epochs=2,
        blue_epochs=1,
        validation_ratio=0.2,
    )

    assert summary.code == "ssq"
    model_dir = tmp_path / "model" / "ssq" / "window_3"
    assert (model_dir / "red.keras").exists()
    assert (model_dir / "metadata.json").exists()

    models = load_trained_models("ssq", window_size=3)
    assert set(models.keys()) == {"red", "blue"}

    predictions = predict_next_draw("ssq", window_size=3)
    assert "red" in predictions
    assert len(predictions["red"]) == tiny_cfg.red.sequence_len
    assert all(1 <= value <= tiny_cfg.red.num_classes for value in predictions["red"])

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["code"] == "ssq"
