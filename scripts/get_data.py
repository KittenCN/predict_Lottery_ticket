# -*- coding:utf-8 -*-
"""
历史数据下载脚本。

示例：
    python scripts/get_data.py --name ssq --start 2024001 --end 2024350
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_data_run  # noqa: E402
from src.config import LOTTERY_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 500.com 下载历史开奖数据")
    parser.add_argument(
        "--name",
        default="ssq",
        help="彩票类型代码，如 ssq / dlt / kl8，默认 ssq",
    )
    parser.add_argument("--start", type=int, default=None, help="起始期号（包含），默认从最早可用期开始")
    parser.add_argument("--end", type=int, default=None, help="结束期号（包含），默认至最新期")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="快乐8 是否使用出球顺序数据（仅 kl8 有效）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"不支持的彩票类型：{args.name}，有效选项：{', '.join(LOTTERY_CONFIGS.keys())}")
    get_data_run(code, cq=int(args.sequence), start_issue=args.start, end_issue=args.end)
    logger.success("数据下载完成：{}", code)


if __name__ == "__main__":
    main()

