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
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[1]
cash_select = [10, 9, 8, 7, 6 ,5 ,0]


