import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
from config import *
from common import get_data_run

name = "kl8"
get_data_run(name=name, cq=0)
ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
ori_numpy = ori_data.drop(ori_data.columns[0], axis=1).to_numpy()[1:]

# limit_line = len(ori_numpy)
limit_line = 30
shifting = [0.01] * 5
total_create = 50
err_nums = 1000
results = []
shiftings = []
err = -1
prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]

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

    # print(march_cal)
    # print(["{:.2f}%".format(item*100) for item in  march_rate])
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

## 分析质数比
def analysis_prime_number(limit=limit_line, result_list=None):
    prime_group = defaultdict(int)
    total_draws = 0
    if result_list is None:
        result_list = ori_numpy
        length = 21
    else:
        limit = 1
        length = 11
    for i in range(limit):
        prime_num = 0
        numbers = result_list[i][1:length]
        numbers.sort()
        for item in result_list[i]:
            total_draws += 1
            if item in prime_list:
                prime_num += 1
    prime_rate = prime_num / total_draws
    print(prime_rate)
    return prime_rate

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
    current_repeat_rate = cal_repeat_rate(limit=1, result_list=result_list)
    for i in range(21):
        if abs(his_repeat_rate[i] - current_repeat_rate[i]) > shifting[0]:
            # print("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
            return 0, False
        
    his_index = 0
    for i in range(20, 0, -1):
        if his_repeat_rate[i] > 0 and his_repeat_rate[i] >= 0.1:
            his_index = i + 1
            break
    if current_repeat_rate[his_index] > his_repeat_rate[his_index]:
        # print("重复率异常！",abs(his_repeat_rate[i] - current_repeat_rate[i]), shifting)
        return -1, False    
    
    ## 验证冷热号
    current_hot_balls, current_cold_balls = cal_ball_rate(limit=1, result_list=result_list)
    if abs(his_hot_balls - current_hot_balls) > shifting[1] or abs(his_cold_balls - current_cold_balls) > shifting[1]:
        # print("冷热号异常！", abs(his_hot_balls - current_hot_balls), abs(his_cold_balls - current_cold_balls), shifting)
        return 1, False
    
    ## 验证奇偶比
    current_odd, current_even = cal_ball_parity(limit=1, result_list=result_list)
    if abs(his_odd - current_odd) > shifting[2] or abs(his_even - current_even) > shifting[2]:
        # print("奇偶比异常！", abs(his_odd - current_odd), abs(his_even - current_even), shifting)
        return 2, False
    
    ## 验证号码组
    current_group_rate = cal_ball_group(result_list=result_list)
    for i in range(8):
        if abs(his_group_rate[i] - current_group_rate[i]) > shifting[3]:
            # print("号码组异常！", abs(his_group_rate[i] - current_group_rate[i]), shifting)
            return 3, False
    
    ## 验证连续号码
    current_consecutive_rate = analysis_consecutive_number(limit=1, result_list=result_list)
    # for i in range(11):
    #     if abs(his_consecutive_rate[i] - current_consecutive_rate[i]) > shifting:
    #        # print("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
    #         return False
    w = 0
    b = 0
    for i in range(2, 11):
        if his_consecutive_rate[i] > 0 and current_consecutive_rate[i] > 0:
            # print("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
            w += 1
        elif his_consecutive_rate[i] <= 0 and current_consecutive_rate[i] > 0:
            # print("连续号码异常！", i, abs(his_consecutive_rate[i] - current_consecutive_rate[i]), shifting)
            b += 1
            break
    if w <= 0 or b > 0:
        return 4, False
    
    return 99, True


if __name__ == "__main__":
    # cal_repeat_rate()
    # cal_ball_rate()
    # cal_ball_parity()
    # cal_ball_group()
    # analysis_consecutive_number()
    # bayesian_analysis()
    # analysis_prime_number()

    # n_clusters = 10
    # labels, centers = kmeans_clustering(ori_numpy[:limit_line], n_clusters)
    # plot_clusters(ori_numpy[:limit_line], labels, centers)

    his_repeat_rate = cal_repeat_rate()
    hot_list, cold_list = cal_hot_cold()
    hot_rate, cold_rate = cal_ball_rate()
    his_hot_balls, his_cold_balls = cal_ball_rate(limit_line)
    his_odd, his_even = cal_ball_parity(limit_line)
    his_group_rate = cal_ball_group()
    his_consecutive_rate = analysis_consecutive_number()

    pbar = tqdm(total=total_create)
    for i in range(1, total_create + 1):
        current_result = [0]
        err = [0] * 5
        shifting = [item // 2 for item in shifting]
        err_code_max = -1
        while True:
            pbar.set_description("{err} {shifting}".format(err=err, shifting=[round(num, 3) for num in shifting]))
            err_code, check_result = check_rate([current_result])
            if check_result:
                break
            current_result = [0]
            if err_code > -1:
                if err_code < err_code_max:
                    continue
                err[err_code] += 1
                if err[err_code] > err_nums // 10:
                    err_code_max = err_code
                if err[err_code] > err_nums:
                    shifting[err_code] += 0.01
                    err[err_code] = 0
                    for j in range(err_code + 1, len(err)):
                        shifting[j] = 0.01
                        err[j] = 0
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
                while (current_num in current_result or current_num <= 0 or current_num in prime_list):
                    current_num = random.randint(1, 80)
                current_result.append(current_num)
            current_result.sort()
        results.append(current_result[1:])
        shiftings.append(shifting)
        shifting = [round(num, 3) for num in shifting]
        # print(current_result[1:], shifting)
        pbar.update(1)
    pbar.close()
    sorted_results = sorted(zip(results, shiftings), key=lambda x: x[1])
    sorted_results, sorted_shiftings = zip(*sorted_results)
    sorted_results = list(sorted_results)
    sorted_shiftings = list(sorted_shiftings)
    for i in range(total_create):
        sorted_shiftings[i] = [round(num, 3) for num in sorted_shiftings[i]]
    for i in range(total_create):
        print(sorted_shiftings[i])
    for i in range(total_create):
        print(sorted_results[i])