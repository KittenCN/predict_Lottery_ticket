import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from collections import defaultdict
from config import *

name = "kl8"
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[1:]
# limit_line = len(ori_numpy)
limit_line = 30
shifting = [0.05, 0.05, 0.05, 0.05]
total_create = 10
results = []
err = -1

## 计算往期重复的概率
def cal_repeat_rate(limit=limit_line, result_list=None):
    march_cal = [0] * 21
    march_rate = [0.0] * 21
    total_march = 0
    if result_list is None:
        result_list = ori_numpy
        j_shiftint = 1
    else:
        limit = 1
        j_shiftint = 0
    for i in range(limit):
        for j in range(i + j_shiftint, limit_line):
            march_num = 0
            total_march += 1
            if result_list is None:
                for k in range(1, 21):
                    if result_list[i][k] == ori_numpy[j][k]:
                        march_num += 1
            else:
                for x in range(1,11):
                    for y in range(1,21):
                        if result_list[i][x] == ori_numpy[j][y]:
                            march_num += 1
                        elif result_list[i][x] < ori_numpy[j][y]:
                            break
            march_cal[march_num] += 1
    for i in range(21):
        march_rate[i] = march_cal[i] / total_march

    # print(march_cal[:20])
    # print(["{:.2f}%".format(item*100) for item in  march_rate[:20]])
    return march_rate

## 计算前10的冷热号
def cal_hot_cold(begin=0, end=limit_line):
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
def cal_ball_rate(limit=limit_line, result_list=None):
    hot_rate_times = 0
    cold_rate_times = 0
    times = 0
    if result_list is None:
        result_list = ori_numpy
        i_shiftint = 1
        length = 21
    else:
        limit = 1
        i_shiftint = 0
        length = 11
    
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
    # print("{:.2f}%".format(hot_ball_rate * 100))
    # print("{:.2f}%".format(cold_ball_rate * 100))
    return hot_ball_rate, cold_ball_rate

## 计算奇偶比:
def cal_ball_parity(limit=limit_line, result_list=None):
    odd = 0
    even = 0
    if result_list is None:
        result_list = ori_numpy
        length = 21
    else:
        limit = 1
        length = 11
    for i in range(limit):
        for j in range(1, length):
            if result_list[i][j] % 2 == 0:
                even += 1
            else:
                odd += 1
    # print("{:.2f}%".format(odd / (odd + even) * 100))
    # print("{:.2f}%".format(even / (odd + even) * 100))
    return odd / (odd + even), even / (odd + even)

## 将80个号码分为8组，计算每组的出现概率
def cal_ball_group(result_list=None):
    group = [0] * 8
    if result_list is None:
        result_list = ori_numpy
        limit = limit_line
        length = 21
    else:
        limit = 1
        length = 11
    for i in range(limit):
        for j in range(1, length):
            if result_list[i][j] <= 10:
                group[0] += 1
            elif result_list[i][j] <= 20:
                group[1] += 1
            elif result_list[i][j] <= 30:
                group[2] += 1
            elif result_list[i][j] <= 40:
                group[3] += 1
            elif result_list[i][j] <= 50:
                group[4] += 1
            elif result_list[i][j] <= 60:
                group[5] += 1
            elif result_list[i][j] <= 70:
                group[6] += 1
            else:
                group[7] += 1
    group_rate = [item / sum(group) for item in group]
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
def analysis_consecutive_number(limit=limit_line, result_list=None):
    consecutive_group = defaultdict(int)
    total_draws = 0
    consecutive_rate_list = [0] * 11
    consecutive_rate = [0.0] * 11
    if result_list is None:
        result_list = ori_numpy
        length = 21
    else:
        limit = 1
        length = 11
    for i in range(limit):
        total_draws += 1
        numbers = result_list[i][1:length]
        numbers.sort()
        consecutive_group_list = find_consecutive_number(numbers)
        for item in consecutive_group_list:
            consecutive_group[item] += 1
    sorted_consecutive_group = sorted(consecutive_group.items(), key=lambda x: x[1], reverse=True)
    for item, count in sorted_consecutive_group:
        consecutive_rate_list[len(item)] += count
    for i in range(11):
        consecutive_rate[i] = consecutive_rate_list[i] / total_draws
    # print(consecutive_rate)
    return consecutive_rate

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

## 验证各概率是否正常
def check_rate(result_list):
    ## 验证总数
    if len(result_list[0][1:]) != 10:
        # print("总数异常！",len(result_list[0][1:]),10)
        return -1, False
    
    ## 验证重复
    for i in range(1,11):
        for j in range(i + 1, 11):
            if result_list[0][i] == result_list[0][j]:
                # print("重复异常！", result_list[0][i], result_list[0][j])
                return -1, False
    
    for item in results:
        if result_list[0] == item:
            # print("重复异常！", result_list[0], item)
            return -1, False

    ## 验证重复率
    his_repeat_rate = cal_repeat_rate()
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list)
    for i in range(21):
        if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
            # print("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
            return 0, False
    
    ## 验证冷热号
    his_hot_balls, his_cold_balls = cal_ball_rate(limit_line)
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list)
    if abs(his_hot_balls - current_hot_balls) > shifting[1] or abs(his_cold_balls - current_cold_balls) > shifting[1]:
        # print("冷热号异常！", abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls), shifting)
        return 1, False
    
    ## 验证奇偶比
    his_odd, his_even = cal_ball_parity(limit_line)
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
        # print("奇偶比异常！", abs(his_odd - current_odd), abs(his_even - current_even), shifting)
        return 2, False
    
    ## 验证号码组
    his_group_rate = cal_ball_group()
    current_group_rate = cal_ball_group(result_list=result_list)
    for i in range(8):
        if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
            # print("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
            return 3, False
    
    ## 验证连续号码
    # his_consecutive_rate = analysis_consecutive_number()
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    # for i in range(11):
    #     if abs(his_consecutive_rate[i] - current_consecutive_rate[i]) > shifting:
    #        # print("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return False
    if current_consecutive_rate[2] < 1:
        # print("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
        return -1, False
    
    return 99, True


if __name__ == "__main__":
    # cal_repeat_rate()
    # cal_ball_rate()
    # cal_ball_parity()
    # cal_ball_group()
    # analysis_consecutive_number()
    # bayesian_analysis()

    # n_clusters = 10
    # labels, centers = kmeans_clustering(ori_numpy[:limit_line], n_clusters)
    # plot_clusters(ori_numpy[:limit_line], labels, centers)

    hot_list, cold_list = cal_hot_cold()
    hot_rate, cold_rate = cal_ball_rate()

    for i in range(total_create):
        current_result = [0]
        err = [0, 0, 0, 0]
        shifting = [0.05, 0.05, 0.05, 0.05]
        while True:
            err_code, check_result = check_rate([current_result])
            if check_result:
                break
            current_result = [0]
            if err_code > -1:
                err[err_code] += 1
                if err[err_code] > 10:
                    shifting[err_code] += 0.01
                    err[err_code] = 0
            ## 按比例插入冷热号
            hot_selection = random.randint(int((hot_rate - shifting[1]) * 10), int((hot_rate + shifting[1]) * 10))
            cold_selection = random.randint(int((cold_rate - shifting[1]) * 10), int((cold_rate + shifting[1]) * 10))
            last_num = 0
            for i in range(hot_selection):
                current_num = 0
                while current_num == last_num:
                    current_num = hot_list[random.randint(0, 9)]
                last_num = current_num
                current_result.append(hot_list[random.randint(0, 9)])
            last_num = 0
            for i in range(cold_selection):
                current_num = 0
                while current_num == last_num:
                    current_num = cold_list[random.randint(0, 9)]
                last_num = current_num
                current_result.append(cold_list[random.randint(0, 9)])
            
            ## 随机插入其他数字
            for i in range(10 - len(current_result) + 1):
                current_num = 0
                while (current_num in current_result or current_num <= 0):
                    current_num = random.randint(1, 80)
                current_result.append(current_num)
            current_result.sort()
        results.append(current_result)
        shifting = [round(num, 2) for num in shifting]
        print(current_result[1:], shifting)
    print(results)