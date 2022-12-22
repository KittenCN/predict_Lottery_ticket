# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import argparse
from common import get_data_run

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="选择爬取数据")
parser.add_argument('--cq', default=0, type=int, help="是否使用出球顺序，0：不使用（即按从小到大排序），1：使用")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        get_data_run(name=args.name, cq=args.cq)
