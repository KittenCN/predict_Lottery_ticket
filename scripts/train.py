# -*- coding: utf-8 -*-
"""
模型训练脚本（TensorFlow 2.15+ 版本）。

示例：
    python scripts/train.py --name ssq --window-size 5 --red-epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure the project root is on sys.path so `src` imports work and the
# bootstrap shim can be imported early.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import as early as possible; src.bootstrap is best-effort
try:
    import src.bootstrap  # noqa: F401
except Exception:
    # If bootstrap fails, continue; the bootstrap shim is non-critical
    pass

from src.common import get_data_run, train_pipeline  # noqa: E402
from src.config import LOTTERY_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练指定彩票的 LSTM 模型")
    parser.add_argument(
        "--name",
        default="ssq",
        help="彩票类型代码，如 ssq / dlt / kl8，默认 ssq",
    )
    parser.add_argument("--window-size", type=int, default=None, help="时间窗口大小，默认使用玩法配置")
    parser.add_argument("--batch-size", type=int, default=None, help="训练批大小，默认使用玩法配置")
    parser.add_argument("--red-epochs", type=int, default=None, help="红球模型训练轮数")
    parser.add_argument("--blue-epochs", type=int, default=None, help="蓝球模型训练轮数")
    parser.add_argument("--download-data", action="store_true", help="训练前自动下载最新数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"不支持的彩票类型：{args.name}，有效选项：{', '.join(LOTTERY_CONFIGS.keys())}")

    if args.download_data:
        logger.info("开始下载数据...")
        get_data_run(code)

    summary = train_pipeline(
        name=code,
        window_size=args.window_size,
        batch_size=args.batch_size,
        red_epochs=args.red_epochs,
        blue_epochs=args.blue_epochs,
    )
    logger.success("训练完成，详情见 model/{}/window_{}/{}", summary.code, summary.window_size, "metadata.json")


if __name__ == "__main__":
    main()

