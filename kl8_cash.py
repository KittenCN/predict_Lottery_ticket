# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
import argparse

from tqdm import tqdm
from config import *
from itertools import combinations
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=1, type=int, help="download data")
parser.add_argument('--cash_file_name', default="-1", type=str, help='cash_file_name')
parser.add_argument('--current_nums', default=-1, type=int, help='current nums')
parser.add_argument('--path', default="", type=str, help='path')
parser.add_argument('--simple_mode', default=0, type=int, help='simple mode')
#--------------------------------------------------------------------------------------------------#
parser.add_argument('--limit_line', default=0, type=int, help='useless')
parser.add_argument('--total_create', default=50, type=int, help='useless')
parser.add_argument('--cal_nums', default=10, type=int, help='useless')
parser.add_argument('--multiple', default=1, type=int, help='useless')
parser.add_argument('--multiple_ratio', default="1,0", type=str, help='useless')
parser.add_argument('--repeat', default=1, type=int, help='useless')
parser.add_argument('--calculate_rate', default=0, type=int, help='useless')
parser.add_argument('--calculate_rate_list', default="5", type=str, help='useless')
args = parser.parse_args()

file_path = "./results/" 
endstring = ["csv"]
name = args.name
if args.download == 1:
    from common import get_data_run
    get_data_run(name=name, cq=0)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][1:]
# if args.current_nums >= 0:
#     index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - (args.current_nums + 1)
#     if index >= 0:
#         ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
cash_select_list = []
for i in range(0, 11):
    _t = [element for element in range(i, -1, -1)]
    cash_select_list.append(_t)
cash_price_list = [[5000000, 8000, 800, 80, 5, 3, 0, 0, 0, 0, 2], \
                    [300000, 2000, 200, 20, 5, 3, 0, 0, 0, 2], \
                    [50000, 800, 88, 10, 3, 0, 0, 0, 2], \
                    [10000, 288, 28, 4, 0, 0, 0, 2], \
                    [3000, 30, 10, 3, 0, 0, 0], \
                    [1000, 21, 3, 0, 0, 0], \
                    [100, 5, 3, 0, 0], \
                    [53, 3, 0, 0], \
                    [19, 0, 0], \
                    [4.6, 0]]

def check_lottery(cash_file_name, args, all_cash=0, all_lucky=0, path_mode=0):
    global ori_numpy
    if args.current_nums >= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[-1][0] and args.current_nums <= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]:
        index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - args.current_nums
        if path_mode == 0:
            logger.info("当前期数为{}。".format(args.current_nums))
        if index >= 0:
            ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
    else:
        if path_mode == 0:
            logger.info("当前期数为{}，计算期数为{}。".format(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0], ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]))
    if path_mode == 0:
        logger.info("中奖号码为:{}".format(ori_numpy))
    cash_data = pd.read_csv(cash_file_name)
    cash_numpy = cash_data.to_numpy()
    cash_select = cash_select_list[cash_numpy.shape[1]]
    cash_price = cash_price_list[10 - (cash_numpy.shape[1])]
    cash_list = [0] * len(cash_select)

    x = 0
    if args.simple_mode == 1:
        sub_bar = tqdm(total=len(cash_numpy), leave=False)
    for item in cash_numpy:
        if args.simple_mode == 1:
            sub_bar.update(1)
        x += 1
        for index in  range(len(cash_select)):
            ori_split = list(combinations(ori_numpy, cash_select[index]))
            cash_split = list(combinations(item, cash_select[index]))
            cash_set = set(ori_split) & set(cash_split)
            if cash_select[index] != 0:
                cash_list[index] += len(cash_set)
                if cash_price[index] != 0 and len(cash_set) != 0:
                    if args.simple_mode == 0:
                        logger.info("第{}注, 号码{}中奖。".format(x, cash_set))
                    break
            elif cash_select[index] == 0 and len(cash_set) == 0:
                cash_list[index] += 1
                if args.simple_mode == 0:
                    logger.info("第{}注, 号码{}中奖。".format(x, cash_set))
                break
    if args.simple_mode == 1:        
        sub_bar.close()
    total_cash = 0
    for i in range(len(cash_select)):
        if args.simple_mode == 0:
            logger.info("中{}个球，共{}注，奖金为{}元。".format(cash_select[i], cash_list[i], cash_list[i] * cash_price[i]))
        total_cash += cash_list[i] * cash_price[i]
    if args.simple_mode == 0:
        logger.info("本期共投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(len(cash_numpy) * 2, total_cash, total_cash / (len(cash_numpy) * 2) * 100))
    all_cash += len(cash_numpy) * 2
    all_lucky += total_cash
    return all_cash, all_lucky

if __name__ == "__main__":
    if args.path == "" or args.cash_file_name != "-1":
        file_path = "./results/" 
        if args.cash_file_name != "-1":
            cash_file_name = file_path + args.cash_file_name + ".csv"
        else:
            ## 寻找目录下最新的文件
            import os
            file_list = [_ for _ in os.listdir(file_path) if _.split('.')[1] in endstring]
            file_list.sort(key=lambda fn: os.path.getmtime(file_path + fn))
            cash_file_name = file_path + file_list[-1]   
            filename_split = file_list[-1].split('_')
            if len(filename_split) == 4:
                if int(filename_split[-1].split('.')[0]) > 0:
                    args.current_nums = int(filename_split[-1].split('.')[0])
        check_lottery(cash_file_name=cash_file_name, args=args, path_mode=0)
    else:
        file_path = "./results_" + args.path + "/"
        all_cash, all_lucky = 0, 0
        import os
        file_list = [_ for _ in os.listdir(file_path) if _.split('.')[1] in endstring]
        file_list.sort(key=lambda fn: os.path.getmtime(file_path + fn))
        if args.simple_mode == 1:
            pbar = tqdm(total=len(file_list))
        for filename in file_list:
            if args.simple_mode == 1:
                pbar.update(1)
            cash_file_name = file_path + filename
            filename_split = filename.split('_')
            if len(filename_split) == 4:
                if int(filename_split[-1].split('.')[0]) > 0:
                    args.current_nums = int(filename_split[-1].split('.')[0])
            all_cash, all_lucky = check_lottery(cash_file_name=cash_file_name, args=args, all_cash=all_cash, all_lucky=all_lucky, path_mode=1)
        if args.simple_mode == 1:
            pbar.close()
        logger.info("总投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(all_cash, all_lucky, all_lucky / all_cash * 100))
    