# -*- coding: utf-8 -*-
"""
彩票预测系统快速开始示例

本示例展示如何使用彩票预测系统进行数据获取、模型训练和预测

Author: KittenCN
"""
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.common import get_current_number, get_data_run
from src.config import LOTTERY_CONFIGS


def quick_start_example():
    """快速开始示例"""
    print("=== 彩票AI预测系统快速开始示例 ===")
    
    # 1. 获取数据
    print("\n1. 获取双色球数据...")
    try:
        get_data_run('ssq')
        print("数据获取成功！")
    except Exception as e:
        print(f"数据获取失败: {e}")
        return
    
    # 2. 查看当前期号
    print("\n2. 查看当前期号...")
    try:
        current_num = get_current_number('ssq')
        print(f"双色球最新期号: {current_num}")
    except Exception as e:
        print(f"获取期号失败: {e}")
    
    # 3. 显示支持的彩票类型
    print("\n3. 支持的彩票类型:")
    for code, cfg in LOTTERY_CONFIGS.items():
        print(f"  - {code}: {cfg.name}")
    
    print("\n=== 示例完成 ===")
    print("\n下一步:")
    print("1. 使用 'python scripts/train.py --name ssq --window-size 5 --red-epochs 60' 训练模型")
    print("2. 使用 'python scripts/predict.py --name ssq --window-size 5 --save' 进行预测")


if __name__ == "__main__":
    quick_start_example()
