# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""

import pandas as pd
from config import *

datacnt = [0] * 81
dataori = [i for i in range(81)]
ori_data = []

def BasicAnalysis(oridata):
    # Basic analysis of the data
    # ori_data: original data
    # Return: None
    # Author: KittenCN
    global datacnt, dataori
    datacnt = [0] * 81
    dataori = [i for i in range(81)]
    for row in oridata:
        for item in row:
            datacnt[int(item)] += 1
    datacnt, dataori = sortcnt(datacnt, dataori, 81)
    lastcnt = -1
    for i in range(81):
        if dataori[i] == 0:
            continue
        if lastcnt != datacnt[i]:
            print()
            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
            lastcnt = datacnt[i]
        elif lastcnt == datacnt[i]:
            print(dataori[i], end = " ")
    return datacnt, dataori
    
def sortcnt(datacnt, dataori, rangenum=81):
    for i in range(rangenum):
        for j in range(i + 1, rangenum):
            if datacnt[i] < datacnt[j]:
                datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                dataori[i], dataori[j] = dataori[j], dataori[i]
            elif datacnt[i] == datacnt[j]:
                if dataori[i] < dataori[j]:
                    datacnt[i], datacnt[j] = datacnt[j], datacnt[i]
                    dataori[i], dataori[j] = dataori[j], dataori[i]
    return datacnt, dataori

def getdata():
    strdata = input("输入要统计的出现次数，“，”分隔, -1结束: ").split(',')
    if strdata[0] == "-1":
        return None, None
    data = [int(i) for i in strdata]
    oridata = []
    for i in range(81):
        if dataori[i] == 0:
            continue
        if datacnt[i] in data:
            oridata.append(dataori[i])
    booldata = [False] * len(oridata)
    return oridata, booldata

def dfs(oridata, booldata, getnums, dep, ans, cur):
    if dep == getnums:
        ans.append(cur.copy())
        return
    for i in oridata:
        if booldata[oridata.index(i)] or i <= cur[dep - 1]:
            continue
        booldata[oridata.index(i)] = True
        cur[dep] = i
        dfs(oridata,booldata, getnums, dep + 1, ans, cur)
        booldata[oridata.index(i)] = False
    return ans

def shrink(oridata, booldata):
    getnums = int(input("输入要缩水至几个数? (-1表示结束) "))
    while getnums != -1:
        ans = dfs(oridata,booldata, getnums, 0, [], [0] * getnums)
        print("一共有 {} 条结果，可缩水至 {} 个数.".format(len(ans), getnums))
        strSumMinMax = input("输入和值最小和最大值，用‘，’分隔").split(',')
        SumMinMax = [int(i) for i in strSumMinMax]
        SumMin = SumMinMax[0]
        SumMax = SumMinMax[1]
        for i in range(len(ans)):
            if sum(ans[i]) < SumMin or sum(ans[i]) > SumMax:
                continue
            print(ans[i])
        getnums = int(input("输入要缩水至几个数? (-1表示结束) "))

def sumanalyusis(limit=-1):
    oridata = pd.read_csv("{}{}".format(name_path["kl8"]["path"], data_file_name))
    data = oridata.iloc[:, 2:].values
    sumori = [i for i in range(1401)]
    sumcnt = [0] * 1401
    linenum = 0
    for row in data:
        if limit != -1 and linenum >= limit:
            break
        linenum += 1
        sum = 0
        for item in row:
            sum += item
        sumcnt[sum] += 1
    sumcnt, sumori = sortcnt(sumcnt, sumori, 1401)
    lastcnt = -1
    for i in range(1401):
        if sumori[i] == 0 or sumcnt[i] == 0:
            continue
        if lastcnt != sumcnt[i]:
            print()
            print("{}: {}".format(sumcnt[i], sumori[i]), end = " ")
            lastcnt = sumcnt[i]
        elif lastcnt == sumcnt[i]:
            print(sumori[i], end = " ")
    print()
    sumtop = int(input("输入要统计前几位和值:"))
    lastsum = -1
    sumans = []
    sumanscnt = 0
    for i in range(1401):
        if sumcnt[i] == 0:
            continue
        if sumcnt[i] != lastsum:
            if sumanscnt == sumtop:
                break;
            else:
                lastsum = sumcnt[i]
                sumanscnt += 1
        sumans.append(sumori[i])
    print(sumans)

if __name__ == "__main__":
    while True:
        print()
        print(" 1. 读取预测数据并分析\r\n 2. 缩水\r\n 3. 和值分析\r\n 0. 退出\r\n")
        choice = int(input("input your choice:"))
        if choice == 1:
            _datainrow = []
            n = int(input("输入数据组数，-1为从文件输入:"))
            if n != -1:
                for i in range(n):
                    tmpdata = input("输入第 #{} 组数据: ".format(i + 1)).strip().split(' ')
                    for item in tmpdata:
                        _datainrow.append(int(item))
                    _datainrow.append(int(item) for item in tmpdata)
                    ori_data.append(tmpdata)
            else:
                filename = input("输入文件名: ")
                fileadd = "{}{}{}{}".format(predict_path, "kl8/", filename, ".csv")
                ori_data = pd.read_csv(fileadd).values
                limit = int(input("共有{}组数据，输入要分析前多少组：".format(len(ori_data))))
                ori_data = ori_data[:limit]
                for row in ori_data:
                    for item in row:
                        _datainrow.append(item)           
            datacnt, dataori = BasicAnalysis(ori_data)
            print()
            currentnums = input("输入当前获奖数据，-1为结束： ").split(' ')
            if currentnums[0] != "-1":
                curnums = [int(i) for i in currentnums]
                curcnt = 0
                tmp_cnt = [0] * len(ori_data)
                for item in curnums:
                    for i, row in enumerate(ori_data):
                        if item in row:
                            curcnt += 1
                            tmp_cnt[i] += 1
                            break
                totalnums = len(list(set(_datainrow)))
                for i in range(len(tmp_cnt)):
                    print("第{}组数据中，当前获奖数据出现的次数为{}次，概率为：{:.2f}%".format(i + 1, tmp_cnt[i], tmp_cnt[i] / totalnums * 100))
                print("命中数 / 总预测数: {} / {}".format(curcnt, totalnums))
                lastcnt = -1
                for i in range(81):
                    if dataori[i] == 0:
                        continue
                    elif dataori[i] in curnums:
                        if lastcnt != datacnt[i]:
                            print()
                            print("{}: {}".format(datacnt[i], dataori[i]), end = " ")
                            lastcnt = datacnt[i]
                        elif lastcnt == datacnt[i]:
                            print(dataori[i], end = " ")
                print()
            oridata, booldata = getdata()
            print(oridata)

        elif choice == 2:
            oridata, booldata = getdata()
            shrink(oridata, booldata)
        elif choice == 3:
            limit = int(input("输入要分析的数据组数，-1为全部:"))
            sumanalyusis(limit)
        if choice == 0:
            break
