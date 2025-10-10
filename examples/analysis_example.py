# -*- coding: utf-8 -*-
"""
数据分析示例

展示如何使用数据分析功能分析彩票数据

Author: KittenCN
"""
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis import BasicAnalysis


def analysis_example():
    """数据分析示例"""
    print("=== 彩票数据分析示例 ===")
    
    # 示例数据
    sample_data = [
        [1, 5, 12, 23, 34, 45, 67, 78, 80, 77, 66, 55, 44, 33, 22, 11, 2, 3, 4, 6],
        [2, 8, 15, 29, 38, 47, 59, 68, 79, 71, 62, 53, 42, 31, 28, 17, 9, 7, 13, 19],
        [3, 11, 18, 27, 36, 49, 58, 69, 72, 64, 56, 48, 39, 21, 14, 26, 35, 41, 52, 63]
    ]
    
    print("示例数据分析结果:")
    try:
        datacnt, dataori = BasicAnalysis(sample_data)
        print(f"\n分析完成，共分析 {len(sample_data)} 组数据")
        print("出现频率统计已显示在上方")
    except Exception as e:
        print(f"分析失败: {e}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    analysis_example()