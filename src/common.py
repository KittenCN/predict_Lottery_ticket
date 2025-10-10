# -*- coding: utf-8 -*-
"""
公共接口封装。

为脚本层提供以下能力：
1. 下载历史数据：`get_data_run`
2. 查询最新期号：`get_current_number`
3. 训练模型：`train_pipeline`
4. 预测下一期开奖：`predict_latest`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from loguru import logger

from .config import LOTTERY_CONFIGS, ensure_runtime_directories, get_lottery_config
from .data_fetcher import download_history, get_current_issue, load_history

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import TrainingSummary


_PIPELINE_MODULE = None


def _load_pipeline():
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is None:
        from . import pipeline as _pipeline  # noqa: WPS433

        _PIPELINE_MODULE = _pipeline
    return _PIPELINE_MODULE


def get_data_run(
    name: str,
    cq: int = 0,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
) -> None:
    """下载指定彩票的历史数据。"""

    ensure_runtime_directories()
    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"不支持的彩票类型: {name}")
    use_sequence = bool(cq) and code == "kl8"
    download_history(code, start=start_issue, end=end_issue, use_sequence_order=use_sequence)


def get_current_number(name: str) -> str:
    """返回指定彩票的当前期号。"""

    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"不支持的彩票类型: {name}")
    return get_current_issue(code)


def train_pipeline(
    name: str,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    red_epochs: Optional[int] = None,
    blue_epochs: Optional[int] = None,
) -> "TrainingSummary":
    """高层训练接口，封装 pipeline.train_lottery_models。"""

    code = name.lower().strip()
    logger.info("开始训练【{}】模型...", LOTTERY_CONFIGS[code].name)
    pipeline_module = _load_pipeline()
    summary = pipeline_module.train_lottery_models(
        code=code,
        window_size=window_size,
        batch_size=batch_size,
        red_epochs=red_epochs,
        blue_epochs=blue_epochs,
    )
    logger.success("训练完成: {}", summary)
    return summary


def predict_latest(name: str, window_size: Optional[int] = None) -> Dict[str, list]:
    """使用最新模型预测下一期号码。"""

    code = name.lower().strip()
    cfg = get_lottery_config(code)
    pipeline_module = _load_pipeline()
    predictions = pipeline_module.predict_next_draw(code=code, window_size=window_size)
    readable = {key: list(map(int, value.tolist())) for key, value in predictions.items()}
    logger.info("【{}】预测结果: {}", cfg.name, readable)
    return readable


__all__ = [
    "get_data_run",
    "get_current_number",
    "train_pipeline",
    "predict_latest",
    "download_history",
    "load_history",
]
