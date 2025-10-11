# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse
import datetime
import os
# import time
import threading
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
from ..config import *
from itertools import combinations
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="kl8", type=str, help="lottery name")
parser.add_argument('--download', default=1, type=int, help="download data")
parser.add_argument('--limit_line', default=50, type=int, help='limit line')
parser.add_argument('--total_create', default=50, type=int, help='total create')
parser.add_argument('--err_nums', default=1000, type=int, help='err nums')
parser.add_argument('--cal_nums', default=10, type=int, help='cal nums')
parser.add_argument('--analysis_history', default=1, type=int, help='analysis history')
parser.add_argument('--current_nums', default=-1, type=int, help='current nums')
parser.add_argument('--check_in_main', default=0, type=int, help='check in main')
parser.add_argument('--calculate_rate', default=0, type=int, help='calculate rate')
parser.add_argument('--calculate_rate_list', default="5", type=str, help='calculate rate list')
parser.add_argument('--multiple', default=1, type=int, help='multiple')
parser.add_argument('--multiple_ratio', default="1,0", type=str, help='multiple_ratio')
parser.add_argument('--repeat', default=1, type=int, help='repeat') 
parser.add_argument('--path', default="", type=str, help='path')
parser.add_argument('--simple_mode', default=0, type=int, help='simple mode') 
parser.add_argument('--random_mode', default=0, type=int, help='random mode')
parser.add_argument('--advanced_mode', default=0, type=int, help='advanced algorithm mode: 0=original, 1=genetic+bayesian, 2=full_advanced')
args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
name = args.name
if args.cal_nums < 0:
    args.cal_nums = abs(args.cal_nums) + 1
if args.download == 1:
    from ..common import get_data_run
    get_data_run(name=name, cq=0)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()

if args.current_nums > 0 and args.current_nums >= ori_numpy[-1][0] and args.current_nums <= ori_numpy[0][0]:
    index_diff = ori_numpy[0][0] - args.current_nums + 1
    ori_numpy = ori_numpy[index_diff:]

if args.random_mode == 0:
    if args.path == "":
            file_path = "./results/" 
    else:
        file_path = "./results_" + args.path + "/"
elif args.random_mode == 1:
    if args.path == "":
        file_path = "./random/"
    else:
        file_path = "./random_" + args.path + "/"

# limit_line = len(ori_numpy)
limit_line = args.limit_line
ori_avg_rate = [0.05, 0.05, 0.05, 0.05, 0.01, 0.05]
ori_shiftings_list = [ori_avg_rate] * 10
rate_file = "./kl8_rate.csv"
if os.path.exists(rate_file):
    rate_data = pd.read_csv(rate_file)
    ori_shiftings_list = rate_data.to_numpy()
ori_shiftings = ori_shiftings_list[args.cal_nums - 1]
if len(ori_shiftings) != len(ori_avg_rate):
    ori_shiftings = ori_avg_rate
shifting = ori_shiftings.copy()
total_create = args.total_create * args.multiple
err_nums = args.err_nums
shiftings = []
err = -1
group_size = 50
prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
analysis_history = [3, 5, 7, 9]
err_num_rate = 5
shifting_rate = 0.1

## 计算当前与上期不重复元素间间隔为1的概率
def cal_not_repeat_rate(limit=limit_line, result_list=None, j_shiftint=1):
    total_march = 0
    march_num = 0
    if result_list is None:
        result_list = ori_numpy
        j_shiftint = 1
    for i in range(limit):
        ele_diff = list(set(result_list[i][1:]) & set(ori_numpy[i + j_shiftint][1:]))
        for item in result_list[i][1:]:
            total_march += 1
            if item not in ele_diff and (item + 1 in ori_numpy[i + j_shiftint][1:] or item - 1 in ori_numpy[i + j_shiftint][1:]):
                march_num += 1
    march_rate = march_num / total_march
    # logger.info("{:2f}%".format(march_rate * 100))
    return march_rate

## 计算往期重复的概率（优化版）
def cal_repeat_rate(limit=limit_line, result_list=None, j_shiftint=1):
    import numpy as np
    from collections import Counter
    
    march_cal = [0] * (args.cal_nums + 1)
    march_rate = [0.0] * (args.cal_nums + 1)
    total_march = 0
    
    if result_list is None:
        result_list = ori_numpy
        j_shiftint = 1
    
    # 优化：预处理所有集合，减少重复计算
    processed_sets = []
    for i in range(min(limit, len(result_list))):
        if i + j_shiftint < len(ori_numpy):
            set1 = set(result_list[i][1:])
            set2 = set(ori_numpy[i + j_shiftint][1:])
            processed_sets.append((set1, set2))
    
    # 批量计算交集
    for set1, set2 in processed_sets:
        total_march += 1
        march_num = len(set1 & set2)
        
        # 修正：根据实际选择数量调整
        if len(result_list[0]) > (args.cal_nums + 1):
            march_num = int(round(march_num * args.cal_nums / 20, 0))
        
        if march_num <= args.cal_nums:
            march_cal[march_num] += 1
    
    # 计算概率分布
    if total_march > 0:
        for i in range(args.cal_nums + 1):
            march_rate[i] = march_cal[i] / total_march
    
    return march_rate

## 遗传算法优化号码选择（新增）
def genetic_algorithm_optimization(population_size=100, generations=50, mutation_rate=0.1):
    """使用遗传算法优化号码组合"""
    import numpy as np
    import random
    
    def create_individual():
        """创建个体（号码组合）"""
        return sorted(random.sample(range(1, 81), args.cal_nums))
    
    def fitness(individual):
        """适应度函数：基于多维约束的综合评分"""
        score = 0.0
        
        # 检查重复率约束
        result_list = [[0] + individual]
        try:
            current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
            repeat_score = sum(abs(his_repeat_rate[i] - current_repeat_rate[i]) 
                             for i in range(1, len(current_repeat_rate)))
            score -= repeat_score
        except:
            score -= 10
        
        # 检查冷热号比例
        try:
            hot_count = sum(1 for num in individual if num in hot_list)
            cold_count = sum(1 for num in individual if num in cold_list)
            hot_ratio = hot_count / len(individual)
            cold_ratio = cold_count / len(individual)
            score -= abs(hot_ratio - his_hot_balls) + abs(cold_ratio - his_cold_balls)
        except:
            score -= 5
        
        # 检查奇偶比例
        odd_count = sum(1 for num in individual if num % 2 == 1)
        even_count = len(individual) - odd_count
        odd_ratio = odd_count / len(individual)
        even_ratio = even_count / len(individual)
        score -= abs(odd_ratio - his_odd) + abs(even_ratio - his_even)
        
        # 检查组分布
        group_counts = [0] * 8
        for num in individual:
            group_idx = (num - 1) // 10
            group_counts[group_idx] += 1
        
        group_ratios = [count / len(individual) for count in group_counts]
        score -= sum(abs(group_ratios[i] - his_group_rate[i]) for i in range(8))
        
        return score
    
    def crossover(parent1, parent2):
        """交叉操作"""
        # 保持号码唯一性的交叉
        child = []
        all_nums = set(parent1 + parent2)
        
        # 随机选择一半来自parent1
        child.extend(random.sample(parent1, args.cal_nums // 2))
        
        # 从parent2中选择剩余号码
        remaining = [num for num in parent2 if num not in child]
        needed = args.cal_nums - len(child)
        
        if len(remaining) >= needed:
            child.extend(random.sample(remaining, needed))
        else:
            child.extend(remaining)
            # 从全局范围补充
            available = [num for num in range(1, 81) if num not in child]
            child.extend(random.sample(available, needed - len(remaining)))
        
        return sorted(child)
    
    def mutate(individual):
        """变异操作"""
        if random.random() < mutation_rate:
            # 随机替换一个号码
            idx = random.randint(0, len(individual) - 1)
            available = [num for num in range(1, 81) if num not in individual]
            if available:
                individual[idx] = random.choice(available)
                individual.sort()
        return individual
    
    # 初始化种群
    population = [create_individual() for _ in range(population_size)]
    
    # 进化过程
    for generation in range(generations):
        # 计算适应度
        fitness_scores = [(individual, fitness(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择优秀个体
        elite_size = population_size // 4
        new_population = [individual for individual, _ in fitness_scores[:elite_size]]
        
        # 生成新个体
        while len(new_population) < population_size:
            # 锦标赛选择
            tournament_size = 3
            tournament = random.sample(fitness_scores[:population_size//2], tournament_size)
            parent1 = max(tournament, key=lambda x: x[1])[0]
            
            tournament = random.sample(fitness_scores[:population_size//2], tournament_size)
            parent2 = max(tournament, key=lambda x: x[1])[0]
            
            # 交叉和变异
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # 返回最优解
    final_fitness = [(individual, fitness(individual)) for individual in population]
    final_fitness.sort(key=lambda x: x[1], reverse=True)
    
    return final_fitness[:10]  # 返回前10个最优解

## 计算前10的冷热号
def cal_hot_cold(begin=0, end=limit_line):
    balls = [0] * 81
    total_balls = 0
    for i in range(begin, end):
        if i >= len(ori_numpy):
            break
        for j in range(1, 21):
            total_balls += 1
            balls[ori_numpy[i][j]] += 1
    balls = [(i, round(balls[i] / total_balls, 5)) for i in range(1, 81)]
    balls.sort(key=lambda x: x[1], reverse=True)
    # logger.info(balls)
    balls = [item[0] for item in balls]
    return balls[:10], balls[-10:]

## 计算指定号码组在每期出现的概率
def cal_ball_rate(limit=limit_line, result_list=None, i_shiftint=1):
    hot_rate_times = 0
    cold_rate_times = 0
    times = 0
    if result_list is None:
        result_list = ori_numpy
        i_shiftint = 1
    length = len(result_list[0])
    
    for i in range(limit):
        hot_balls, cold_balls = cal_hot_cold(i + i_shiftint, i + limit_line)
        for j in range(1, length):
            times += 1
            if result_list[i][j] in hot_balls:
                hot_rate_times += 1
            if result_list[i][j] in cold_balls:
                cold_rate_times += 1
    hot_ball_rate = hot_rate_times / times
    cold_ball_rate = cold_rate_times / times
    # logger.info("{:.2f}%".format(hot_ball_rate * 100))
    # logger.info("{:.2f}%".format(cold_ball_rate * 100))
    return hot_ball_rate, cold_ball_rate

## 计算奇偶比:
def cal_ball_parity(limit=limit_line, result_list=None):
    odd = 0
    even = 0
    if result_list is None:
        result_list = ori_numpy
    length = len(result_list[0])
    for i in range(limit):
        for j in range(1, length):
            if result_list[i][j] % 2 == 0:
                even += 1
            else:
                odd += 1
    # logger.info("{:.2f}%".format(odd / (odd + even) * 100))
    # logger.info("{:.2f}%".format(even / (odd + even) * 100))
    return odd / (odd + even), even / (odd + even)

## 将80个号码分为8组，计算每组的出现概率
def cal_ball_group(limit=limit_line, result_list=None):
    group = [0] * 8
    if result_list is None:
        result_list = ori_numpy
    length = len(result_list[0])
    for i in range(limit):
        for j in range(1, length):
            group_index = (result_list[i][j] - 1) // 10
            group[group_index] += 1
    group_rate = [item / sum(group) for item in group]
    # logger.info(group_rate)
    return group_rate

## 找出连续号码的组合
def find_consecutive_number(numbers):
    consecutive_group = []
    group = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] == 1:
            group.append(numbers[i])
        else:
            if len(group) > 1:
                consecutive_group.append(tuple(group))
            group = [numbers[i]]
    if len(group) > 1:
        consecutive_group.append(tuple(group))
    return consecutive_group

## 分析连续号码组合
def analysis_consecutive_number(limit=limit_line, result_list=None):
    consecutive_group = defaultdict(int)
    total_draws = 0
    if result_list is None:
        result_list = ori_numpy
    length = len(result_list[0])
    consecutive_rate_list = [0] * length
    consecutive_rate = [0.0] * length
    for i in range(limit):
        numbers = result_list[i][1:length]
        numbers.sort()
        consecutive_group_list = find_consecutive_number(numbers)
        for item in consecutive_group_list:
            total_draws += 1
            consecutive_group[item] += 1
    sorted_consecutive_group = sorted(consecutive_group.items(), key=lambda x: x[1], reverse=True)
    for item, count in sorted_consecutive_group:
        consecutive_rate_list[len(item)] += count
    for i in range(length):
        if total_draws > 0:
            consecutive_rate[i] = consecutive_rate_list[i] / total_draws
    # logger.info(consecutive_rate)
    return consecutive_rate

## 分析质数比
def analysis_prime_number(limit=limit_line, result_list=None):
    # prime_group = defaultdict(int)
    total_draws = 0
    if result_list is None:
        result_list = ori_numpy
    for i in range(limit):
        prime_num = 0
        total_draws += 1
        for item in result_list[i]:
            if item in prime_list:
                prime_num += 1
    prime_rate = prime_num / total_draws
    logger.info(prime_rate)
    return prime_rate

## 分析和值概率
def sum_analysis(limit=limit_line, result_list=None):
    sum_group = defaultdict(int)
    sum_rate_group = defaultdict(float)
    total_numbers = 0
    if result_list is None:
        result_list = ori_numpy
    length = len(result_list[0])
    if args.simple_mode == 0:    
        bar = tqdm(total=limit)
    for i in range(limit):
        if args.simple_mode == 0:
            bar.update(1)
        result_list_split = combinations(result_list[i][1:length], args.cal_nums)
        for item in result_list_split:
            current_sum = sum(item)
            group_index = (current_sum - 1) // group_size
            group_key = f"{group_index * group_size + 1}-{(group_index + 1) * group_size}"
            sum_group[group_key] += 1
            total_numbers += 1
    if args.simple_mode == 0:
        bar.close()
    sum_rate_group = {key: count / total_numbers for key, count in sum_group.items()}
    # logger.info(sum_rate_group)
    return sum_rate_group

## 使用贝叶斯定理分析（修正版）
def bayesian_analysis():
    import numpy as np
    from scipy import stats
    
    number_counts = defaultdict(int)
    total_draws = 0
    total_numbers = 0

    # 先验概率：每个号码被抽中的概率（考虑每期选20个数）
    prior_prob = 20/80  # 修正：每期选20个号码

    for row in ori_numpy[:limit_line]:
        total_draws += 1
        for num in row[1:21]:
            total_numbers += 1
            number_counts[int(num)] += 1

    # 计算后验概率（修正贝叶斯公式）
    posterior_probs = {}
    evidence = sum(number_counts.values()) / len(number_counts)  # 修正边际概率
    
    for num in range(1, 81):
        # 修正似然概率计算
        likelihood = number_counts[num] / total_draws if total_draws > 0 else 0
        # 使用Beta-Binomial共轭先验
        alpha = 1 + number_counts[num]  # 后验参数
        beta = 1 + total_draws - number_counts[num]
        posterior_prob = alpha / (alpha + beta)  # Beta分布的期望
        posterior_probs[num] = posterior_prob

    # 按后验概率排序
    sorted_probs = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs

## 马尔可夫链分析（新增）
def markov_chain_analysis(order=1):
    """分析号码间的马尔可夫转移概率"""
    import numpy as np
    
    if order < 1 or order > 3:
        order = 1
        
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(order, len(ori_numpy[:limit_line])):
        # 当前状态：前order期的号码组合特征
        current_state = []
        for j in range(order):
            prev_numbers = set(ori_numpy[i-j-1][1:21])
            current_state.append(tuple(sorted(prev_numbers)))
        
        state_key = tuple(current_state)
        
        # 下一状态：当前期的号码
        next_numbers = set(ori_numpy[i][1:21])
        
        # 统计转移
        for num in next_numbers:
            transition_counts[state_key][num] += 1
    
    # 计算转移概率
    transition_probs = {}
    for state, next_states in transition_counts.items():
        total = sum(next_states.values())
        transition_probs[state] = {num: count/total for num, count in next_states.items()}
    
    return transition_probs

## 信息熵分析（新增）
def entropy_analysis():
    """计算号码选择的信息熵和互信息"""
    import numpy as np
    from scipy.stats import entropy
    
    # 计算每个号码的概率分布
    number_probs = np.zeros(81)
    total_count = 0
    
    for row in ori_numpy[:limit_line]:
        for num in row[1:21]:
            number_probs[num] += 1
            total_count += 1
    
    number_probs = number_probs[1:] / total_count  # 归一化，去掉索引0
    
    # 计算信息熵
    info_entropy = entropy(number_probs, base=2)
    
    # 计算条件熵和互信息
    mutual_info = {}
    for i in range(1, 81):
        for j in range(i+1, 81):
            # 计算号码i和j的联合概率
            joint_count = 0
            count_i = 0
            count_j = 0
            
            for row in ori_numpy[:limit_line]:
                numbers = set(row[1:21])
                if i in numbers and j in numbers:
                    joint_count += 1
                if i in numbers:
                    count_i += 1
                if j in numbers:
                    count_j += 1
            
            if joint_count > 0 and count_i > 0 and count_j > 0:
                p_ij = joint_count / limit_line
                p_i = count_i / limit_line
                p_j = count_j / limit_line
                mi = p_ij * np.log2(p_ij / (p_i * p_j))
                mutual_info[(i, j)] = mi
    
    return info_entropy, mutual_info

## 使用K均值聚类算法
def kmeans_clustering(ori_numpy, n_clusters=3):
    # 使用K均值聚类算法
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(ori_numpy)
    
    # 获取聚类标签和中心点
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers

## 绘制聚类图
def plot_clusters(ori_numpy, labels, centers):
    plt.scatter(ori_numpy[:, 0], ori_numpy[:, 1], c=labels, cmap='rainbow')
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='black')
    plt.show()

## 验证各概率是否正常
def check_rate(result_list):
    ## 验证总数
    if len(result_list[0][1:]) != args.cal_nums:
        # logger.info("总数异常！",len(result_list[0][1:]),args.cal_nums)
        return -1, False
    
    ## 验证重复
    # for i in range(1,args.cal_nums + 1):
    #     for j in range(i + 1, args.cal_nums + 1):
    #         if result_list[0][i] == result_list[0][j]:
    #             # logger.info("重复异常！", result_list[0][i], result_list[0][j])
    #             return -1, False
    if len(result_list[0]) != len(set(result_list[0])):
        return -1, False
    
    for item in results:
        if result_list[0] == item:
            # logger.info("重复异常！", result_list[0], item)
            return -1, False

    ## 验证重复率
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
    for i in range(1, args.cal_nums + 1):
        if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
            # logger.info("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
            return 0, False
        
    his_index = 0
    for i in range(args.cal_nums, 0, -1):
        if his_repeat_rate[i] > 0 and his_repeat_rate[i] >= 0.1:
            his_index = i + 1
            break
    if current_repeat_rate[his_index] - his_repeat_rate[his_index] > shifting[0]:
        # logger.info("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
        return 0, False    
    
    ## 验证冷热号
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list, i_shiftint=0)
    if abs(his_hot_balls - current_hot_balls) > shifting[1] or abs(his_cold_balls - current_cold_balls) > shifting[1]:
        # logger.info("冷热号异常！", abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls), shifting)
        return 1, False
    
    ## 验证奇偶比
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
        # logger.info("奇偶比异常！", abs(his_odd - current_odd), abs(his_even - current_even), shifting)
        return 2, False
    
    ## 验证号码组
    current_group_rate = cal_ball_group(limit=1, result_list=result_list)
    # for i in range(8):
        # if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
        #     # logger.info("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
        #     return 3, False
        # if his_group_rate[i] == 0 and current_group_rate[i] > 0.1 or his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01 :
        #     # logger.info("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
        #     return -1, False
    for i in range(8):
        if args.cal_nums >= 8:
            if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                # logger.info("号码组异常！", i, abs(his_group_rate[i] - current_group_rate[i]), shifting)
                return 3, False
        else:
            if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                # logger.info("号码组异常！", i, abs(his_group_rate[i] - current_group_rate[i]), shifting)
                return 3, False
    
    ## 验证连续号码
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    correct_flag = False
    for i in range(2, args.cal_nums + 1):
        if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
            # logger.info("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
            return 4, False
        if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
            correct_flag = True
    if correct_flag == False:
        return 4, False
    # for i in range(2, args.cal_nums + 1):
    #     if abs(his_consecutive_rate[i] - current_consecutive_rate[i]) > shifting[4]:
    #         # logger.info("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return 4, False
    #     if his_consecutive_rate[i] == 0 and current_consecutive_rate[i] > 0.1 or his_consecutive_rate[i] > 0.1 and current_consecutive_rate[i] < 0.01 :
    #         # logger.info("号码组异常！", abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return -1, False
    
    ## 验证和值
    current_sum = sum(result_list[0][1:])
    group_index = (current_sum - 1) // group_size
    group_key = f"{group_index * group_size + 1}-{(group_index + 1) * group_size}"
    current_sum_rate = his_sum_rate.get(group_key, 0)
    if current_sum_rate < 0.1:
        # logger.info("和值异常！", current_sum_rate, shifting)
        return -1, False
    
    ## 验证非重复元素等差概率:
    current_march_rate = cal_not_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
    if abs(current_march_rate - his_not_repeat_rate) > shifting[5]:
        # logger.info("非重复元素等差概率异常！", abs(current_march_rate - his_not_repeat_rate), shifting)
        return 5, False    
    
    return 99, True

## 判断文件夹是否存在，不存在就创建
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

## 多线程调用写入文件
def write_file(lst,file_name="result"):
    t = threading.Thread(target=write_file_core, args=(lst,file_name))
    t.start()

## 写入文件
def write_file_core(lst,_file_name="result"):
    random_number = random.randint(0, 999999)
    current_time_in = str(int(current_time) + random_number)
    file_name = file_path + "{}_{}_{}_{}.csv".format(_file_name, current_time_in,args.cal_nums,str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums)
    while os.path.exists(file_name):
        random_number = random.randint(0, 999999)
        current_time_in = str(int(current_time) + random_number)
        file_name = file_path + "{}_{}_{}_{}.csv".format(_file_name, current_time_in,args.cal_nums,str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums) 
    with open(file_name, "w") as f:
        for i in range(args.cal_nums - 1):
            f.write("b" + str(i + 1) + ",")
        f.write("b" + str(args.cal_nums) + "\n")
        cnt = 0
        item_index = 0
        for item in lst:
            if args.multiple > 1:
                item_index += 1
                div_nums = args.multiple_ratio.split(",")
                if item_index % int(div_nums[0]) == int(div_nums[1]):
                    cnt += 1
                    for index in range(len(item)-1):
                        f.write("{},".format(item[index]))
                    f.write("{}\n".format(item[-1]))
                    if cnt >= args.total_create:
                        break
            else:
                for index in range(len(item)-1):
                    f.write("{},".format(item[index]))
                f.write("{}\n".format(item[-1]))
## 判断数组中有几个奇数几个偶数
def check_odd_even(lst):
    odd = 0
    even = 0
    for item in lst:
        if item % 2 == 0:
            even += 1
        else:
            odd += 1
    return odd, even

## 计算list中大于0的元素的平均值
def cal_average(lst):
    total = 0
    count = 0
    for item in lst:
        if item > 0:
            total += item
            count += 1
    if count == 0:
        return 0
    return total / count

## 统计显著性检验（新增）
def statistical_significance_test(observed, expected, alpha=0.05):
    """使用卡方检验验证概率差异的统计显著性"""
    from scipy.stats import chisquare, ks_2samp
    import numpy as np
    
    if len(observed) != len(expected):
        return False, 1.0
    
    # 卡方检验
    try:
        chi2_stat, p_value = chisquare(observed, expected)
        is_significant = p_value < alpha
        return is_significant, p_value
    except:
        return False, 1.0

## 改进的动态调整机制（新增）
class AdaptiveThresholdManager:
    def __init__(self, initial_thresholds, learning_rate=0.01, momentum=0.9):
        self.thresholds = initial_thresholds.copy()
        self.initial_thresholds = initial_thresholds.copy()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [0.0] * len(initial_thresholds)
        self.success_history = [[] for _ in range(len(initial_thresholds))]
        self.failure_counts = [0] * len(initial_thresholds)
        self.total_attempts = [0] * len(initial_thresholds)
        
    def update_threshold(self, constraint_idx, success, error_magnitude=None):
        """基于成功/失败和误差大小动态更新阈值"""
        if constraint_idx >= len(self.thresholds):
            return
            
        self.total_attempts[constraint_idx] += 1
        
        if success:
            self.success_history[constraint_idx].append(1)
            # 成功时略微收紧阈值
            gradient = -self.learning_rate * 0.1
        else:
            self.success_history[constraint_idx].append(0)
            self.failure_counts[constraint_idx] += 1
            # 失败时基于误差大小调整
            if error_magnitude:
                gradient = self.learning_rate * min(error_magnitude, 0.5)
            else:
                gradient = self.learning_rate * 0.2
        
        # 动量更新
        self.velocity[constraint_idx] = (
            self.momentum * self.velocity[constraint_idx] + gradient
        )
        self.thresholds[constraint_idx] += self.velocity[constraint_idx]
        
        # 限制阈值范围
        self.thresholds[constraint_idx] = max(
            self.initial_thresholds[constraint_idx] * 0.1,
            min(self.thresholds[constraint_idx], 
                self.initial_thresholds[constraint_idx] * 10)
        )
        
        # 保持历史记录窗口
        if len(self.success_history[constraint_idx]) > 100:
            self.success_history[constraint_idx] = self.success_history[constraint_idx][-100:]
    
    def get_success_rate(self, constraint_idx):
        """获取约束的成功率"""
        if not self.success_history[constraint_idx]:
            return 0.0
        return sum(self.success_history[constraint_idx]) / len(self.success_history[constraint_idx])
    
    def should_relax_constraint(self, constraint_idx, threshold=0.1):
        """判断是否应该放松约束"""
        if self.total_attempts[constraint_idx] < 50:
            return False
        return self.get_success_rate(constraint_idx) < threshold

## 分析当前期与历史概率数据的乖离性
def analysis_rate(rate_mode=0):
    global limit_line
    rate_diff = [] 
    result_list = [ori_numpy[0]]
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=1)
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list, i_shiftint=1)
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    current_group_rate = cal_ball_group(limit=1, result_list=result_list)
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    current_march_rate = cal_not_repeat_rate(limit=1, result_list=result_list, j_shiftint=1)
    if args.simple_mode == 0:
        pbar = tqdm(total=len(analysis_history))
    for item in analysis_history:
        if item == -1:
            item = len(ori_numpy) - 1
        ori_numpy_except_last = ori_numpy[1:item+1]
        limit_line = item
        his_repeat_rate = cal_repeat_rate(limit=item, result_list=ori_numpy_except_last, j_shiftint=2)
        his_hot_balls, his_cold_balls = cal_ball_rate(limit=item, result_list=ori_numpy_except_last, i_shiftint=2)
        his_odd, his_even = cal_ball_parity(limit=item, result_list=ori_numpy_except_last)
        his_group_rate = cal_ball_group(limit=item, result_list=ori_numpy_except_last)
        his_consecutive_rate = analysis_consecutive_number(limit=item, result_list=ori_numpy_except_last)
        hit_march_rate = cal_not_repeat_rate(limit=item, result_list=ori_numpy_except_last, j_shiftint=2)
        rate_diff.append([item, 
            cal_average([abs(his_repeat_rate[i] - current_repeat_rate[i]) for i in range(1, args.cal_nums + 1)]), 
            cal_average([abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls)]),
            cal_average([abs(his_odd - current_odd), abs(his_even - current_even)]),
            cal_average([abs(his_group_rate[i] - current_group_rate[i]) for i in range(8)]),
            cal_average([abs(his_consecutive_rate[i] - current_consecutive_rate[i]) for i in range(2, args.cal_nums + 1)]),
            cal_average([abs(hit_march_rate - current_march_rate)])])
        if args.simple_mode == 0:
            pbar.update(1)
    if args.simple_mode == 0:
        pbar.close()
    avg_rate = [0.0] * len(rate_diff[0])
    max_rate = [0.0] * len(rate_diff[0])
    avg_rate[0] = "avg"
    max_rate[0] = "max"
    for i in range(len(rate_diff)):
        for j in range(len(rate_diff[i])):
            if args.simple_mode == 0:
                print(round(rate_diff[i][j], 5), end=" ")
            if j > 0:
                # avg_rate[j] += rate_diff[i][j] * ((len(rate_diff) - i) / 10)
                avg_rate[j] += rate_diff[i][j]
                # if rate_diff[i][j] > max_rate[j]:
                #     max_rate[j] = rate_diff[i][j]
                max_rate[j] = max(max_rate[j], rate_diff[i][j])
                # max_rate[j] = max(max_rate[j], shifting[j - 1])
        if args.simple_mode == 0:
            print()
    for i in range(len(avg_rate)):
        if i > 0:
            avg_rate[i] = round(avg_rate[i] / len(analysis_history), 5)
            if args.simple_mode == 0:
                print(avg_rate[i], end=" ")
        else:
            if args.simple_mode == 0:
                print(avg_rate[i], end=" ")
    if args.simple_mode == 0:
        print()
    for i in range(len(max_rate)):
        if i > 0:
            max_rate[i] = round(max_rate[i], 5)
            if args.simple_mode == 0:
                print(max_rate[i], end=" ")
        else:
            if args.simple_mode == 0:
                print(max_rate[i], end=" ")
    if args.simple_mode == 0:            
        print()
    # avg_rate = rate_diff[0]
    result_rate = len(avg_rate[1:]) * [0.0]
    for i in range(len(avg_rate[1:])):
        result_rate[i] = max(avg_rate[i + 1], ori_shiftings[i])

    if rate_mode == 1:
        result_rate = len(avg_rate[1:]) * [0.0]
        for i in range(len(avg_rate[1:])):
            result_rate[i] = max(avg_rate[i + 1], ori_shiftings[i])
        return result_rate
    if rate_mode == 2:
        result_rate = len(max_rate[1:]) * [0.0]
        for i in range(len(max_rate[1:])):
            result_rate[i] = max(max_rate[i + 1], ori_shiftings[i])
        return result_rate
    elif rate_mode == 0:
        return avg_rate[1:]

## 判断list长度是否超过限制
def check_list_length(lst):
    if len(lst) > args.cal_nums + 1:
        return True
    return False

def init_func(rate_mode=1):
    global shifting, cal_shiftings, limit_line, his_repeat_rate, hot_list, cold_list, hot_rate, cold_rate, his_hot_balls, his_cold_balls, his_odd, his_even, his_group_rate, his_consecutive_rate, his_sum_rate, his_not_repeat_rate, threshold_manager
    if args.analysis_history == 1:
        cal_shiftings = analysis_rate(rate_mode=rate_mode).copy()
    else:
        analysis_rate(rate_mode=rate_mode)
    limit_line = args.limit_line
    his_repeat_rate = cal_repeat_rate()
    hot_list, cold_list = cal_hot_cold()
    hot_rate, cold_rate = cal_ball_rate()
    his_hot_balls, his_cold_balls = cal_ball_rate(limit_line)
    his_odd, his_even = cal_ball_parity(limit_line)
    his_group_rate = cal_ball_group()
    his_consecutive_rate = analysis_consecutive_number()
    his_sum_rate = sum_analysis()
    his_not_repeat_rate = cal_not_repeat_rate()
    
    # 初始化自适应阈值管理器
    threshold_manager = AdaptiveThresholdManager(cal_shiftings)

def generate_random_numbers(num_rows, num_nums_per_row):
    results = []
    for _ in range(num_rows):
        row = sorted(random.sample(range(1, 81), num_nums_per_row))
        results.append(row)
    return results

## 深度学习特征提取（新增）
def deep_feature_extraction():
    """使用深度学习方法提取高级特征"""
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        
        # 构建特征矩阵
        features = []
        targets = []
        
        for i in range(10, len(ori_numpy[:limit_line])):
            # 特征：前10期的统计信息
            feature_vector = []
            
            # 历史重复率特征
            hist_data = ori_numpy[i-10:i]
            repeat_features = []
            for j in range(1, len(hist_data)):
                overlap = len(set(hist_data[j][1:]) & set(hist_data[j-1][1:]))
                repeat_features.append(overlap)
            feature_vector.extend(repeat_features)
            
            # 号码频率特征
            freq_counts = np.zeros(81)
            for row in hist_data:
                for num in row[1:]:
                    freq_counts[num] += 1
            feature_vector.extend(freq_counts[1:])  # 去掉索引0
            
            # 奇偶比例特征
            odd_counts = []
            for row in hist_data:
                odd_count = sum(1 for num in row[1:] if num % 2 == 1)
                odd_counts.append(odd_count)
            feature_vector.extend(odd_counts)
            
            # 分组分布特征
            group_features = []
            for row in hist_data:
                group_dist = [0] * 8
                for num in row[1:]:
                    group_idx = (num - 1) // 10
                    group_dist[group_idx] += 1
                group_features.extend(group_dist)
            feature_vector.extend(group_features)
            
            features.append(feature_vector)
            
            # 目标：当前期的号码特征（简化为热号数量）
            current_numbers = set(ori_numpy[i][1:])
            if i > 0:
                prev_hot, prev_cold = cal_hot_cold(max(0, i-50), i)
                hot_count = sum(1 for num in current_numbers if num in prev_hot)
                targets.append(hot_count)
        
        if len(features) < 10:
            return None, None
            
        features = np.array(features)
        targets = np.array(targets)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA降维
        pca = PCA(n_components=min(20, features_scaled.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        
        # 训练神经网络
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=500,
            random_state=42,
            alpha=0.01
        )
        
        # 分割训练集和测试集
        split_idx = int(0.8 * len(features_pca))
        X_train, X_test = features_pca[:split_idx], features_pca[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        mlp.fit(X_train, y_train)
        
        # 预测
        predictions = mlp.predict(X_test)
        
        return mlp, scaler, pca, predictions
        
    except ImportError:
        logger.warning("深度学习依赖库未安装，跳过深度特征提取")
        return None, None, None, None
    except Exception as e:
        logger.error(f"深度特征提取出错: {e}")
        return None, None, None, None

## 高级号码生成策略（新增）
def advanced_number_generation(use_genetic=True, use_ml=True):
    """结合多种高级算法的号码生成策略"""
    candidate_solutions = []
    
    # 1. 遗传算法生成
    if use_genetic:
        try:
            genetic_solutions = genetic_algorithm_optimization(
                population_size=50, 
                generations=30, 
                mutation_rate=0.15
            )
            candidate_solutions.extend([sol[0] for sol in genetic_solutions[:5]])
        except Exception as e:
            logger.warning(f"遗传算法失败: {e}")
    
    # 2. 贝叶斯优化生成
    try:
        bayesian_probs = bayesian_analysis()
        top_numbers = [num for num, _ in bayesian_probs[:args.cal_nums*2]]
        
        # 基于贝叶斯概率的智能选择
        bayesian_solution = []
        remaining_numbers = top_numbers.copy()
        
        while len(bayesian_solution) < args.cal_nums and remaining_numbers:
            # 考虑约束的贪心选择
            best_num = None
            best_score = float('-inf')
            
            for num in remaining_numbers:
                test_solution = sorted(bayesian_solution + [num])
                
                # 简单评分
                score = 0
                
                # 奇偶平衡
                odd_count = sum(1 for n in test_solution if n % 2 == 1)
                target_odd = int(his_odd * len(test_solution))
                score -= abs(odd_count - target_odd)
                
                # 分组平衡
                group_counts = [0] * 8
                for n in test_solution:
                    group_counts[(n-1)//10] += 1
                for i, count in enumerate(group_counts):
                    expected = his_group_rate[i] * len(test_solution)
                    score -= abs(count - expected)
                
                if score > best_score:
                    best_score = score
                    best_num = num
            
            if best_num:
                bayesian_solution.append(best_num)
                remaining_numbers.remove(best_num)
            else:
                break
        
        # 补齐不足的号码
        while len(bayesian_solution) < args.cal_nums:
            available = [n for n in range(1, 81) if n not in bayesian_solution]
            if available:
                bayesian_solution.append(random.choice(available))
            else:
                break
                
        candidate_solutions.append(sorted(bayesian_solution))
        
    except Exception as e:
        logger.warning(f"贝叶斯生成失败: {e}")
    
    # 3. 马尔可夫链生成
    try:
        import numpy as np
        markov_transitions = markov_chain_analysis(order=2)
        if markov_transitions:
            # 基于转移概率生成号码
            markov_solution = []
            
            # 获取最近的状态
            recent_states = []
            for i in range(min(2, len(ori_numpy))):
                recent_numbers = set(ori_numpy[i][1:21])
                recent_states.append(tuple(sorted(recent_numbers)))
            
            state_key = tuple(recent_states)
            
            if state_key in markov_transitions:
                transition_probs = markov_transitions[state_key]
                # 按概率加权选择
                numbers = list(transition_probs.keys())
                weights = list(transition_probs.values())
                
                if numbers and weights:
                    # 使用概率分布选择号码
                    selected = np.random.choice(
                        numbers, 
                        size=min(args.cal_nums, len(numbers)), 
                        replace=False, 
                        p=np.array(weights)/sum(weights)
                    )
                    markov_solution = sorted(selected.tolist())
            
            if len(markov_solution) == args.cal_nums:
                candidate_solutions.append(markov_solution)
                
    except Exception as e:
        logger.warning(f"马尔可夫链生成失败: {e}")
    
    # 4. 如果没有足够的候选解，使用改进的随机生成
    while len(candidate_solutions) < 3:
        solution = []
        
        # 按约束概率生成
        hot_count = max(1, int(his_hot_balls * args.cal_nums))
        cold_count = max(1, int(his_cold_balls * args.cal_nums))
        
        # 选择热号
        if hot_list:
            solution.extend(random.sample(hot_list, min(hot_count, len(hot_list))))
        
        # 选择冷号
        if cold_list:
            available_cold = [n for n in cold_list if n not in solution]
            solution.extend(random.sample(available_cold, min(cold_count, len(available_cold))))
        
        # 补充其他号码
        remaining = [n for n in range(1, 81) if n not in solution and n not in hot_list and n not in cold_list]
        needed = args.cal_nums - len(solution)
        if needed > 0 and remaining:
            solution.extend(random.sample(remaining, min(needed, len(remaining))))
        
        if len(solution) == args.cal_nums:
            candidate_solutions.append(sorted(solution))
    
    # 5. 评估并选择最佳解
    best_solution = None
    best_score = float('-inf')
    
    for solution in candidate_solutions:
        try:
            result_list = [[0] + solution]
            _, is_valid = check_rate(result_list)
            
            if is_valid:
                # 计算综合评分
                score = 0
                
                # 重复率评分
                current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list, j_shiftint=0)
                repeat_score = -sum(abs(his_repeat_rate[i] - current_repeat_rate[i]) 
                                  for i in range(1, min(len(his_repeat_rate), len(current_repeat_rate))))
                score += repeat_score
                
                # 其他约束评分 (简化)
                odd_count = sum(1 for n in solution if n % 2 == 1)
                score -= abs(odd_count/len(solution) - his_odd) * 10
                
                if score > best_score:
                    best_score = score
                    best_solution = solution
                    
        except Exception as e:
            continue
    
    return best_solution if best_solution else candidate_solutions[0] if candidate_solutions else None

if __name__ == "__main__":
    # cal_hot_cold()
    # cal_repeat_rate()
    # cal_ball_rate()
    # cal_ball_parity()
    # cal_ball_group()
    # analysis_consecutive_number()
    # bayesian_analysis()
    # analysis_prime_number()
    # sum_analysis()
    # cal_not_repeat_rate()

    # n_clusters = args.cal_nums
    # labels, centers = kmeans_clustering(ori_numpy[:limit_line], n_clusters)
    # plot_clusters(ori_numpy[:limit_line], labels, centers)

    check_dir(file_path)
    last_time = ""
    if args.random_mode == 1:
        for _i in range(args.repeat):
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            while current_time == last_time:
                current_time = str(int(current_time) + 1)
            last_time = current_time
            write_file(generate_random_numbers(args.total_create, args.cal_nums), "random")
        exit(0)
    if args.calculate_rate == 1:
        cal_rate_list = args.calculate_rate_list.split(",")
        if int(cal_rate_list[0]) > 0:
            for rate_item in cal_rate_list:
                rate_data = pd.read_csv(rate_file)
                ori_shiftings_list = rate_data.to_numpy()
                args.cal_nums = int(rate_item)
                if args.current_nums == -1:
                    args.current_nums = int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0]) + 1
                if args.current_nums > 0 and args.current_nums >= ori_numpy[-1][0] and args.current_nums <= ori_numpy[0][0]:
                    index_diff = ori_numpy[0][0] - args.current_nums + 1
                    ori_numpy = ori_numpy[index_diff:]
                init_func(rate_mode=0)
                shifting = cal_shiftings.copy()
                pbar = tqdm(total=total_create)
                err_results = []
                results = []
                start_time = datetime.datetime.now()
                for i in range(1, total_create + 1):
                    current_result = [0]
                    err = [0] * len(cal_shiftings)
                    # shifting = [item * 0.9 for item in cal_shiftings]
                    # shifting = [item * 0.9 for item in shifting]
                    # for i in range(len(shifting)):
                    #     shifting[i] = max(shifting[i], ori_shiftings[i])
                    err_code_max = -1
                    while True:
                        pbar.set_description("{current_nums} {err} {shifting}".format(current_nums=[str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums], err=err, shifting=[round(num, 3) for num in shifting]))
                        err_code, check_result = check_rate([current_result])
                        if check_result:
                            break
                        # err_results.append(current_result)
                        current_result = [0]
                        if err_code > -1:
                            if err_code < err_code_max:
                                continue
                            err[err_code] += 1
                            if err[err_code] > err_nums // err_num_rate:
                                err_code_max = err_code
                            if err[err_code] > err_nums:
                                shifting[err_code] += 0.01 if shifting[err_code] * shifting_rate > 0.01 else shifting[err_code] * shifting_rate

                                err[err_code] = 0
                                for j in range(err_code + 1, len(err)):
                                    shifting[j] = cal_shiftings[j]
                                    err[j] = 0
                        ## 按比例插入冷热号
                        hot_selection = random.randint(int(round((hot_rate - 0) * args.cal_nums,0)), int(round((hot_rate + 0) * args.cal_nums,0)))
                        cold_selection = random.randint(int(round((cold_rate - 0) * args.cal_nums,0)), int(round((cold_rate + 0) * args.cal_nums,0)))
                        hot_selection = 1 if hot_selection < 1 else hot_selection
                        cold_selection = 1 if cold_selection < 1 else cold_selection
                        current_result.extend(random.sample(hot_list, hot_selection))
                        current_result.extend(random.sample(cold_list, cold_selection))
                        
                        repeat_flag = True
                        temp_result = current_result.copy()
                        repeat_start_time = datetime.datetime.now()
                        last_result_length = 0
                        while repeat_flag:
                            repeat_flag = False
                            current_result = temp_result.copy()
                            ## 随机插入其他数字
                            useful_list_odd = []
                            useful_list_even = []
                            for item in range(1, 81):
                                if item not in current_result \
                                    and item not in hot_list \
                                    and item not in cold_list:
                                    # and item not in prime_list:
                                    if item % 2 == 1:
                                        useful_list_odd.append(item)
                                    else:
                                        useful_list_even.append(item)
                            current_odd, current_even = check_odd_even(current_result[1:])
                            odd_need = random.randint(int(round((his_odd - shifting[2]) * args.cal_nums,0)), int(round((his_odd + shifting[2]) * args.cal_nums,0)))
                            if current_odd > odd_need:
                                odd_need = current_odd
                            even_need = args.cal_nums - odd_need
                            current_result.extend(random.sample(useful_list_odd, odd_need - current_odd))
                            if check_list_length(current_result):
                                repeat_flag = True
                                continue
                            current_result.extend(random.sample(useful_list_even, args.cal_nums + 1 - len(current_result)))
                            current_result.sort()
                            # if current_result in err_results or current_result[1:] in results:
                            #     if (datetime.datetime.now() - repeat_start_time).seconds > 5:
                            #         break
                            #     repeat_flag = True
                            #     continue
                            if args.check_in_main == 1:
                                ## 验证重复率
                                current_repeat_rate = cal_repeat_rate(limit=1, result_list=[current_result], j_shiftint=0)
                                for i in range(1, args.cal_nums + 1):
                                    if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                        break
                                ## 验证奇偶比
                                if repeat_flag == False:
                                    current_odd, current_even = cal_ball_parity(limit=1, result_list=[current_result])
                                    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                ## 验证号码组
                                if repeat_flag == False:
                                    current_group_rate = cal_ball_group(limit=1, result_list=[current_result])
                                #     for i in range(8):
                                #         if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
                                #             repeat_flag = True
                                #             err_results.append(current_result)
                                #             break
                                    for i in range(8):
                                        if args.cal_nums >= 8:
                                            if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                                                repeat_flag = True
                                                err_results.append(current_result)
                                                break
                                        else:
                                            if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                                                repeat_flag = True
                                                err_results.append(current_result)
                                                break
                                ## 验证连续号码
                                if repeat_flag == False:
                                    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=[current_result])
                                    correct_flag = False
                                    for i in range(2, args.cal_nums + 1):
                                        if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
                                            repeat_flag = True
                                            err_results.append(current_result)
                                            break
                                        if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
                                            correct_flag = True
                                    if correct_flag == False:
                                        repeat_flag = True
                                        err_results.append(current_result)
                                        break
                            if (datetime.datetime.now() - start_time).seconds > 60 and len(results) > last_result_length:
                                last_result_length = len(results)
                                start_time = datetime.datetime.now()
                                sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
                                sorted_results, sorted_shiftings = zip(*sorted_results)
                                sorted_results = list(sorted_results)
                                write_file(sorted_results, "result")
                    results.append(current_result[1:])
                    shiftings.append(shifting)
                    shifting = [round(num, 3) for num in shifting]
                    if args.simple_mode == 0:
                        tqdm.write("{current_result} {shifting}".format(current_result=[num for num in current_result[1:]], shifting=[round(num, 3) for num in shifting]))
                    pbar.update(1)
                pbar.close()
                avg_rate = [round(sum(col) / len(col), 3) for col in zip(*shiftings)]     
                ori_shiftings_list[int(rate_item) - 1] = avg_rate  
                # for avg_rate_index in range (len(avg_rate)):
                #     ori_shiftings_list[int(rate_item) - 1][avg_rate_index] = avg_rate[avg_rate_index]
                with open(rate_file, "w") as f:
                    for i in range(len(ori_avg_rate) - 1):
                        f.write("s" + str(i + 1) + ",")
                    f.write("s" + str(len(ori_avg_rate)) + "\n")
                    for item in ori_shiftings_list:
                        for index in range(len(item)-1):
                            f.write("{},".format(item[index]))
                        f.write("{}\n".format(item[-1]))
    else: 
        init_func(rate_mode=2)      
        shifting = cal_shiftings.copy()
        pbar = tqdm(total=total_create * int(args.repeat))
        for _i in range(args.repeat):
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            while current_time == last_time:
                current_time = str(int(current_time) + 1)
                # time.sleep(0.1)
                # current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            last_time = current_time
            err_results = []
            results = []
            start_time = datetime.datetime.now()
            for i in range(1, total_create + 1):
                current_result = [0]
                
                # 使用高级算法模式
                if args.advanced_mode > 0:
                    advanced_solution = advanced_number_generation(
                        use_genetic=(args.advanced_mode >= 1),
                        use_ml=(args.advanced_mode >= 2)
                    )
                    
                    if advanced_solution:
                        current_result = [0] + advanced_solution
                        # 验证高级解是否符合约束
                        err_code, check_result = check_rate([current_result])
                        if check_result:
                            results.append(current_result[1:])
                            shiftings.append(shifting)
                            if args.simple_mode == 0:
                                tqdm.write("Advanced: {current_result}".format(current_result=[num for num in current_result[1:]]))
                            pbar.update(1)
                            continue
                
                # 原始算法作为后备
                err = [0] * len(cal_shiftings)
                err_code_max = -1
                attempt_count = 0
                max_attempts = 1000 if args.advanced_mode == 0 else 500  # 高级模式下减少尝试次数
                while True:
                    attempt_count += 1
                    if attempt_count > max_attempts:
                        # 强制生成一个基本解
                        current_result = [0] + sorted(random.sample(range(1, 81), args.cal_nums))
                        break
                        
                    pbar.set_description("{mode} {current_nums} {err} {shifting}".format(
                        mode=f"ADV{args.advanced_mode}" if args.advanced_mode > 0 else "STD",
                        current_nums=[str(int(ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[0][0])+1) if args.current_nums == -1 else args.current_nums], 
                        err=err, 
                        shifting=[round(num, 3) for num in shifting]
                    ))
                    err_code, check_result = check_rate([current_result])
                    if check_result:
                        break
                    # err_results.append(current_result)
                    current_result = [0]
                    if err_code > -1:
                        if err_code < err_code_max:
                            continue
                        err[err_code] += 1
                        if err[err_code] > err_nums // err_num_rate:
                            err_code_max = err_code
                        if err[err_code] > err_nums:
                            shifting[err_code] += 0.01 if shifting[err_code] * shifting_rate > 0.01 else shifting[err_code] * shifting_rate

                            err[err_code] = 0
                            for j in range(err_code + 1, len(err)):
                                shifting[j] = cal_shiftings[j]
                                err[j] = 0
                    ## 按比例插入冷热号
                    hot_selection = random.randint(int(round((hot_rate - 0) * args.cal_nums,0)), int(round((hot_rate + 0) * args.cal_nums,0)))
                    cold_selection = random.randint(int(round((cold_rate - 0) * args.cal_nums,0)), int(round((cold_rate + 0) * args.cal_nums,0)))
                    hot_selection = 1 if hot_selection < 1 else hot_selection
                    cold_selection = 1 if cold_selection < 1 else cold_selection
                    current_result.extend(random.sample(hot_list, hot_selection))
                    current_result.extend(random.sample(cold_list, cold_selection))
                    
                    repeat_flag = True
                    temp_result = current_result.copy()
                    repeat_start_time = datetime.datetime.now()
                    last_result_length = 0
                    while repeat_flag:
                        repeat_flag = False
                        current_result = temp_result.copy()
                        ## 随机插入其他数字
                        useful_list_odd = []
                        useful_list_even = []
                        for item in range(1, 81):
                            if item not in current_result \
                                and item not in hot_list \
                                and item not in cold_list:
                                # and item not in prime_list:
                                if item % 2 == 1:
                                    useful_list_odd.append(item)
                                else:
                                    useful_list_even.append(item)
                        current_odd, current_even = check_odd_even(current_result[1:])
                        odd_need = random.randint(int(round((his_odd - shifting[2]) * args.cal_nums,0)), int(round((his_odd + shifting[2]) * args.cal_nums,0)))
                        if current_odd > odd_need:
                            odd_need = current_odd
                        even_need = args.cal_nums - odd_need
                        current_result.extend(random.sample(useful_list_odd, odd_need - current_odd))
                        if check_list_length(current_result):
                            repeat_flag = True
                            continue
                        current_result.extend(random.sample(useful_list_even, args.cal_nums + 1 - len(current_result)))
                        current_result.sort()
                        # if current_result in err_results or current_result[1:] in results:
                        #     if (datetime.datetime.now() - repeat_start_time).seconds > 5:
                        #         break
                        #     repeat_flag = True
                        #     continue
                        if args.check_in_main == 1:
                            ## 验证重复率
                            current_repeat_rate = cal_repeat_rate(limit=1, result_list=[current_result], j_shiftint=0)
                            for i in range(1, args.cal_nums + 1):
                                if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
                                    repeat_flag = True
                                    err_results.append(current_result)
                                    break
                            ## 验证奇偶比
                            if repeat_flag == False:
                                current_odd, current_even = cal_ball_parity(limit=1, result_list=[current_result])
                                if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
                                    repeat_flag = True
                                    err_results.append(current_result)
                            ## 验证号码组
                            if repeat_flag == False:
                                current_group_rate = cal_ball_group(limit=1, result_list=[current_result])
                            #     for i in range(8):
                            #         if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
                            #             repeat_flag = True
                            #             err_results.append(current_result)
                            #             break
                                for i in range(8):
                                    if args.cal_nums >= 8:
                                        if (his_group_rate[i] > 0.1 and current_group_rate[i] < 0.01) or (his_group_rate[i] <= 0.01 and current_group_rate[i] > 0.1):
                                            repeat_flag = True
                                            err_results.append(current_result)
                                            break
                                    else:
                                        if (current_group_rate[i] > 0 and his_group_rate[i] < 0.01):
                                            repeat_flag = True
                                            err_results.append(current_result)
                                            break
                            ## 验证连续号码
                            if repeat_flag == False:
                                current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=[current_result])
                                correct_flag = False
                                for i in range(2, args.cal_nums + 1):
                                    if (current_consecutive_rate[i] >= 0.1 and his_consecutive_rate[i] <= 0.01):
                                        repeat_flag = True
                                        err_results.append(current_result)
                                        break
                                    if (his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0 ):
                                        correct_flag = True
                                if correct_flag == False:
                                    repeat_flag = True
                                    err_results.append(current_result)
                                    break
                        if (datetime.datetime.now() - start_time).seconds > 60 and len(results) > last_result_length:
                            last_result_length = len(results)
                            start_time = datetime.datetime.now()
                            sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
                            sorted_results, sorted_shiftings = zip(*sorted_results)
                            sorted_results = list(sorted_results)
                            write_file(sorted_results, "result")
                results.append(current_result[1:])
                shiftings.append(shifting)
                shifting = [round(num, 3) for num in shifting]
                if args.simple_mode == 0:
                    tqdm.write("{current_result} {shifting}".format(current_result=[num for num in current_result[1:]], shifting=[round(num, 3) for num in shifting]))
                pbar.update(1)
            sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
            sorted_results, sorted_shiftings = zip(*sorted_results)
            sorted_results = list(sorted_results)
            write_file(sorted_results, "result")
            # for i in range(total_create):
            #     logger.info(sorted_results[i])
            # sorted_shiftings = list(sorted_shiftings)
            # for i in range(total_create):
            #     sorted_shiftings[i] = [round(num, 3) for num in sorted_shiftings[i]]
            # for i in range(total_create):
            #     logger.info(sorted_shiftings[i])
            # write_file(sorted_shiftings, "shifting")
        pbar.close()