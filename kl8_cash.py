import pandas as pd
import argparse
from config import *
from common import get_data_run
from itertools import combinations


parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=1, type=int, help="download data")
parser.add_argument('--cash_file_name', default="result_20230823162832", type=str, help='cash_file_name')
parser.add_argument('--index', default=-1, type=int, help='index')
args = parser.parse_args()

name = args.name
if args.download == 1:
    get_data_run(name=name, cq=0)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][1:]
if args.index >= 0:
    index = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0] - args.index
    if index >= 0:
        ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[index][1:]
cash_select = [10, 9, 8, 7, 6 ,5 ,4, 3, 2, 1, 0]
cash_price = [5000000, 8000, 800, 80, 5, 3, 0, 0, 0, 0, 2]
cash_list = [0] * len(cash_select)

cash_file_name = args.cash_file_name + ".csv"
cash_data = pd.read_csv(cash_file_name)
cash_numpy = cash_data.to_numpy()

x = 0
for item in cash_numpy:
    x += 1
    for index in  range(len(cash_select)):
        ori_split = list(combinations(ori_numpy, cash_select[index]))
        cash_split = list(combinations(item, cash_select[index]))
        cash_set = set(ori_split) & set(cash_split)
        if cash_select[index] != 0:
            cash_list[index] += len(cash_set)
            if cash_price[index] != 0 and len(cash_set) != 0:
                print("第{}注, 号码{}中奖。".format(x, cash_set))
                break
        elif cash_select[index] == 0 and len(cash_set) == 0:
            cash_list[index] += 1
            print("第{}注, 号码{}中奖。".format(x, cash_set))
            break

total_cash = 0
for i in range(len(cash_select)):
    print("中{}个球，共{}注，奖金为{}元。".format(cash_select[i], cash_list[i], cash_list[i] * cash_price[i]))
    total_cash += cash_list[i] * cash_price[i]
print("本期共投入{}元，总奖金为{}元，返奖率{:.2f}%。".format(len(cash_numpy) * 2, total_cash, total_cash / (len(cash_numpy) * 2) * 100))
