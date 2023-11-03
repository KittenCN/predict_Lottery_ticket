# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
import argparse
import subprocess
import threading
from tqdm import tqdm
from config import *
from itertools import combinations
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=0, type=int, help="download data")
parser.add_argument('--cash_file_name', default="-1", type=str, help='cash_file_name')
parser.add_argument('--current_nums', default=-1, type=int, help='current nums')
parser.add_argument('--path', default="", type=str, help='path')
parser.add_argument('--simple_mode', default=1, type=int, help='simple mode')
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
nums_index = 0
content = []
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

def sub_check_lottery(item, cash_select, cash_price, cash_list):
    for index in range(len(cash_select)):
        ori_split = list(combinations(ori_numpy, cash_select[index]))
        cash_split = list(combinations(item, cash_select[index]))
        cash_set = set(ori_split) & set(cash_split)
        if cash_select[index] != 0:
            cash_list[index] += len(cash_set)
            if cash_price[index] != 0 and len(cash_set) != 0:
                return cash_list
        elif cash_select[index] == 0 and len(cash_set) == 0:
            cash_list[index] += 1
            return cash_list

def check_lottery(cash_file_name, args, path_mode=1):
    global ori_numpy, nums_index, all_cash, all_lucky, content
    nums_index += 1
    if args.current_nums >= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[-1][0] and args.current_nums <= ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]:
        index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - args.current_nums
        if index >= 0:
            ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
    cash_data = pd.read_csv(cash_file_name)
    cash_numpy = cash_data.to_numpy()
    cash_select = cash_select_list[cash_numpy.shape[1]]
    cash_price = cash_price_list[10 - (cash_numpy.shape[1])]
    cash_list = [0] * len(cash_select)

    # for j in tqdm(range(len(cash_numpy)), desc='subCashThread {}'.format(args.path), leave=False):
    # for item in cash_numpy:
        # item = cash_numpy[j]
        # for index in range(len(cash_select)):
        #     ori_split = list(combinations(ori_numpy, cash_select[index]))
        #     cash_split = list(combinations(item, cash_select[index]))
        #     cash_set = set(ori_split) & set(cash_split)
        #     if cash_select[index] != 0:
        #         cash_list[index] += len(cash_set)
        #         if cash_price[index] != 0 and len(cash_set) != 0:
        #             break
        #     elif cash_select[index] == 0 and len(cash_set) == 0:
        #         cash_list[index] += 1
        #         break
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(sub_check_lottery, item, cash_select, cash_price, cash_list): item for item in cash_numpy}
        for future in as_completed(future_to_url):
            data = future.result()
            if data != None:
                cash_list = data
    total_cash = 0
    for i in range(len(cash_select)):
        total_cash += cash_list[i] * cash_price[i]
    if args.simple_mode == 0 or (args.simple_mode == 2 and total_cash / (len(cash_numpy) * 2) * 100 >= 100):
        # logger.info("{}, 第{}期，本期共投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, nums_index, len(cash_numpy) * 2, total_cash, total_cash / (len(cash_numpy) * 2) * 100))
        content.append("{}, 第{}期，本期共投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, nums_index, len(cash_numpy) * 2, total_cash, total_cash / (len(cash_numpy) * 2) * 100))
    all_cash += len(cash_numpy) * 2
    all_lucky += total_cash
    return all_cash, all_lucky

## 判断文件是否存在
def check_file(_file_name):
    if os.path.exists(_file_name):
        return True
    else:
        return False

## 多线程调用写入文件
def write_file(_content,_file_name="./kl8_runnint_results.txt"):
    t = threading.Thread(target=write_file_core, args=(_content, _file_name))
    t.start()

## 写入文件
def write_file_core(_content,_file_name="./kl8_runnint_results.txt"):
    if check_file(_file_name):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(_file_name, write_mode) as f:
        for item in _content:
            f.write(item + "\n")

if __name__ == "__main__":
    nums_index = 0
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
        threads = []
        # for j in tqdm(range(len(file_list)), desc='CashThread {}'.format(args.path), leave=False):
        for j in range(len(file_list)):
            filename = file_list[j]
            cash_file_name = file_path + filename
            filename_split = filename.split('_') 
            if len(filename_split) == 4:
                if int(filename_split[-1].split('.')[0]) > 0:
                    args.current_nums = int(filename_split[-1].split('.')[0])
            t = threading.Thread(target=check_lottery, args=(cash_file_name, args, 1))
            threads.append(t)
        for t in threads:
            t.start()
        # for t in threads:
        for t_index in tqdm(range(len(threads)), desc='CashThread {}'.format(args.path), leave=False):
            t = threads[t_index]
            t.join()
        # logger.info("{}, 总投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, all_cash, all_lucky, all_lucky / all_cash * 100))
        content.append("{}, 总投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(args.path, all_cash, all_lucky, all_lucky / all_cash * 100))
    write_file(content)