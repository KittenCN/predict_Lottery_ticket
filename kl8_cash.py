import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
from config import *
from common import get_data_run, datetime
from itertools import combinations

name = "kl8"
get_data_run(name=name, cq=0)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][1:]
cash_select = [10, 9, 8, 7, 6 ,5 ,0]
cash_price = [5000000, 8000, 800, 80, 5, 3, 2]
cash_list = [0] * len(cash_select)

cash_file_name = "result_20230823120804.csv"
cash_data = pd.read_csv(cash_file_name)
cash_numpy = cash_data.to_numpy()

for index in  range(len(cash_select)):
    for item in cash_numpy:
        ori_split = list(combinations(ori_numpy, cash_select[index]))
        cash_split = list(combinations(item, cash_select[index]))

        cash_set = set(ori_split) & set(cash_split)
        cash_list[index] += len(cash_set)

for i in range(len(cash_select)):
    print("中{}个球，共{}注，奖金为{}元。".format(cash_select[i], cash_list[i], cash_list[i] * cash_price[i]))
