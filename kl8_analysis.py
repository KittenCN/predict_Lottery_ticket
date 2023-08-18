import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from config import *

name = "kl8"
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()
# limit_line = len(ori_numpy)
limit_line = 30

## 计算往期重复的概率
def cal_repeat_rate():
    march_cal = [0] * 21
    march_rate = [0.0] * 21
    total_march = 0
    for i in range(limit_line):
        for j in range(i + 1, limit_line):
            march_num = 0
            total_march += 1
            for k in range(1, 21):
                if ori_numpy[i][k] == ori_numpy[j][k]:
                    march_num += 1
            march_cal[march_num] += 1
    for i in range(21):
        march_rate[i] = march_cal[i] / total_march

    # print(march_cal[:20])
    # print(["{:.2f}%".format(item*100) for item in  march_rate[:20]])
    return march_rate

## 计算前10的冷热号
def cal_hot_cold(begin, end):
    balls = [0] * 81
    for i in range(begin, end):
        if i >= len(ori_numpy):
            break
        for j in range(1, 21):
            balls[ori_numpy[i][j]] += 1
    balls = [(i, balls[i]) for i in range(1, 81)]
    balls.sort(key=lambda x: x[1], reverse=True)
    balls = [item[0] for item in balls]
    # print(balls[:10])
    # print(balls[-10:])
    return balls[:10], balls[-10:]

## 计算指定号码组在每期出现的概率
def cal_ball_rate(limit=limit_line):
    hot_rate_times = 0
    cold_rate_times = 0
    times = 0
    for i in range(limit):
        hot_balls, cold_balls = cal_hot_cold(i + 1, i + limit)
        for j in range(1, 21):
            times += 1
            if ori_numpy[i][j] in hot_balls:
                hot_rate_times += 1
            if ori_numpy[i][j] in cold_balls:
                cold_rate_times += 1
    hot_ball_rate = hot_rate_times / times
    cold_ball_rate = cold_rate_times / times
    # print("{:.2f}%".format(hot_ball_rate * 100))
    # print("{:.2f}%".format(cold_ball_rate * 100))
    return hot_ball_rate, cold_ball_rate

## 计算奇偶比:
def cal_ball_parity():
    odd = 0
    even = 0
    for i in range(limit_line):
        for j in range(1, 21):
            if ori_numpy[i][j] % 2 == 0:
                even += 1
            else:
                odd += 1
    # print("{:.2f}%".format(odd / (odd + even) * 100))
    # print("{:.2f}%".format(even / (odd + even) * 100))
    return odd / (odd + even), even / (odd + even)

## 将80个号码分为8组，计算每组的出现概率
def cal_ball_group():
    group = [0] * 8
    for i in range(limit_line):
        for j in range(1, 21):
            if ori_numpy[i][j] <= 10:
                group[0] += 1
            elif ori_numpy[i][j] <= 20:
                group[1] += 1
            elif ori_numpy[i][j] <= 30:
                group[2] += 1
            elif ori_numpy[i][j] <= 40:
                group[3] += 1
            elif ori_numpy[i][j] <= 50:
                group[4] += 1
            elif ori_numpy[i][j] <= 60:
                group[5] += 1
            elif ori_numpy[i][j] <= 70:
                group[6] += 1
            else:
                group[7] += 1
    group_rate = [item / sum(group) * 100 for item in group]
    # print(group_rate)
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
def analysis_consecutive_number():
    consecutive_group = defaultdict(int)
    total_draws = 0
    for i in range(limit_line):
        total_draws += 1
        numbers = ori_numpy[i][1:21]
        numbers.sort()
        consecutive_group_list = find_consecutive_number(numbers)
        for item in consecutive_group_list:
            consecutive_group[item] += 1
    sorted_consecutive_group = sorted(consecutive_group.items(), key=lambda x: x[1], reverse=True)
    for item, count in sorted_consecutive_group:
        print(item, count, "{:.2f}%".format(count / total_draws * 100))

## 使用贝叶斯定理分析
def bayesian_analysis():
    number_counts = defaultdict(int)
    total_draws = 0

    # 先验概率：每个号码被抽中的概率是1/80
    prior_prob = 1/80

    for row in ori_numpy[:limit_line]:
        for num in row[1:21]:
            total_draws += 1
            number_counts[int(num)] += 1

    # 计算后验概率
    posterior_probs = {}
    for num in range(1, 81):
        likelihood = number_counts[num] / total_draws
        marginal_prob = total_draws / 80
        posterior_prob = (likelihood * prior_prob) / marginal_prob
        posterior_probs[num] = posterior_prob

    # 按后验概率排序
    sorted_probs = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_probs[:10])
    return sorted_probs

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

if __name__ == "__main__":
    # cal_repeat_rate()
    # cal_ball_rate()
    # cal_ball_parity()
    # cal_ball_group()
    # analysis_consecutive_number()
    # bayesian_analysis()

    n_clusters = 10
    labels, centers = kmeans_clustering(ori_numpy[:limit_line], n_clusters)
    plot_clusters(ori_numpy[:limit_line], labels, centers)